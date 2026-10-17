"""TwinRead on-device runtime: numpy + onnxruntime (+ Pillow/cv2 for crops). No torch. Research prototype.

Bundle format (ruling R24), all files in one directory:
  model.onnx        input "x" float32 [N,3,224,224] in [0,1]; outputs hA, hB (L2-normalised), logits [N,C], review_logit [N]
  prompt_bank.npz   bankA [C,dA], bankB [C,dB]            (kept for audit; not needed for inference)
  probes.npz        {A,B}_coef [k,d], _intercept [k], _mean [d], _scale [d], _classes [k]
                    probs = softmax(((h - mean) / scale) @ coef.T + intercept); binary probes are pre-expanded to k=2
  config.json       {pair, classes, thresholds: {js, review}, licence_note, distributable}
  SHA256SUMS        sha256sum format over every other file
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIZE = 224
OD_THRESHOLD = 0.15        # StainForge background rule: a pixel is tissue if its optical-density sum exceeds this
NO_TISSUE_FRACTION = 0.10  # no_tissue if fewer than 10 % of pixels are tissue ...
UNIFORM_STD = 6.0          # ... or the tile is near-uniform (std over all pixels < 6 grey levels; catches all-black)


def verify_bundle_dir(bundle_dir) -> bool:
    """True iff SHA256SUMS exists, every listed file matches, and no unlisted file is present. Stdlib only."""
    try:
        d = Path(bundle_dir)
        sums = d / "SHA256SUMS"
        if not sums.is_file():
            return False
        listed = {}
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            h, _, name = line.partition("  ")
            name = name.strip().lstrip("*")
            if not h or not name or "/" in name or name.startswith("."):
                return False
            listed[name] = h.strip().lower()
        if not listed:
            return False
        present = {p.name for p in d.iterdir() if p.is_file() and p.name != "SHA256SUMS"}
        if present != set(listed):
            return False
        for name, h in listed.items():
            sha = hashlib.sha256()
            with open(d / name, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    sha.update(chunk)
            if sha.hexdigest() != h:
                return False
        return True
    except Exception:
        return False


@dataclass
class TileResult:
    label: str
    prob: float
    probsA: list
    probsB: list
    probsF: list
    review: bool
    review_score: float
    reasons: list = field(default_factory=list)
    no_tissue: bool = False


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence, base 2 (0..1); same formula as twinread.review.js_divergence."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    m = (p + q) / 2

    def kl(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * (np.log(a) - np.log(m)), 0.0)
        return t.sum()
    return float(np.clip((kl(p) + kl(q)) / (2 * math.log(2)), 0.0, 1.0))


def tissue_fraction(x_uint8: np.ndarray) -> float:
    """Fraction of pixels whose optical-density sum, OD = -log((I+1)/256) summed over RGB, exceeds 0.15."""
    x = np.asarray(x_uint8, np.float32)
    od = -np.log((x + 1.0) / 256.0).sum(-1)
    return float((od > OD_THRESHOLD).mean())


def _no_tissue(x_uint8: np.ndarray) -> bool:
    return tissue_fraction(x_uint8) < NO_TISSUE_FRACTION or float(np.asarray(x_uint8, np.float32).std()) < UNIFORM_STD


def device_review_rule(pA, pB, pF, review_logit, thr: dict, no_tissue: bool = False):
    """Return (review, score, reasons). Fires on: readers' argmax differ ("disagree"), JS(pA,pB) > thr["js"] ("js"),
    sigmoid(review_logit) > thr["review"] ("review_head"), or no_tissue. score = max(JS, sigmoid(review_logit))."""
    pA, pB = np.asarray(pA, np.float64), np.asarray(pB, np.float64)
    reasons = []
    js = js_divergence(pA, pB)
    rs = 1.0 / (1.0 + math.exp(-max(min(float(review_logit), 60.0), -60.0)))
    if int(np.argmax(pA)) != int(np.argmax(pB)):
        reasons.append("disagree")
    if js > thr["js"]:
        reasons.append("js")
    if rs > thr["review"]:
        reasons.append("review_head")
    if no_tissue:
        reasons.append("no_tissue")
    return bool(reasons), float(max(js, rs)), reasons


def center_box(h: int, w: int):
    """(x0, y0, x1, y1) in frame pixels of the region the centre crop keeps (the central square)."""
    s = min(h, w)
    return ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)


def _center_crop(frame: np.ndarray) -> np.ndarray:
    from PIL import Image
    h, w = frame.shape[:2]
    if (h, w) != (SIZE, SIZE):
        s = min(h, w)  # resize the shorter side to 224 (torchvision Resize(int) semantics), bicubic like K1/PIL
        nh, nw = (SIZE, max(SIZE, int(SIZE * w / s))) if h <= w else (max(SIZE, int(SIZE * h / s)), SIZE)
        frame = np.asarray(Image.fromarray(frame).resize((nw, nh), Image.BICUBIC))
        h, w = nh, nw
    t, l = int(round((h - SIZE) / 2.0)), int(round((w - SIZE) / 2.0))
    return np.ascontiguousarray(frame[t:t + SIZE, l:l + SIZE])


def crop_from_frame(frame: np.ndarray, mode="center") -> np.ndarray:
    """RGB uint8 frame [H,W,3] -> [224,224,3]. "center": resize shorter side to 224 then centre crop. "marker": locate the
    rig's page marker and warp the tile region (needs the rig module); falls back to "center" if it is not found."""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be uint8 [H,W,3] RGB")
    if mode == "marker":
        try:
            import sys
            rig = Path(__file__).resolve().parents[1] / "rig"
            if str(rig) not in sys.path:
                sys.path.append(str(rig))
            import match_captures  # rig module (cv2 + numpy)
            r = match_captures.crop_tile(np.ascontiguousarray(frame[..., ::-1]), SIZE)
            if r is not None:
                return np.ascontiguousarray(r[1][..., ::-1])
        except Exception:
            pass
    return _center_crop(frame)


class EdgeModel:
    def __init__(self, bundle_dir, threads: int = 4):
        d = Path(bundle_dir)
        if not verify_bundle_dir(d):
            raise ValueError(f"bundle {d} failed checksum verification (missing/tampered SHA256SUMS); refusing to load")
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.cfg = json.loads((d / "config.json").read_text())
        self.classes = list(self.cfg["classes"])
        self.thr = {"js": float(self.cfg["thresholds"]["js"]), "review": float(self.cfg["thresholds"]["review"])}
        z = np.load(d / "probes.npz", allow_pickle=False)
        self.probes = {r: {k: z[f"{r}_{k}"].astype(np.float32) for k in ("coef", "intercept", "mean", "scale")}
                       for r in ("A", "B")}
        self.bundle_dir = d

    def _probe(self, r: str, h: np.ndarray) -> np.ndarray:
        p = self.probes[r]
        return _softmax(((h - p["mean"]) / p["scale"]) @ p["coef"].T + p["intercept"])

    def infer(self, x_uint8: np.ndarray) -> list:
        x = np.asarray(x_uint8)
        if x.ndim != 4 or x.shape[1:] != (SIZE, SIZE, 3) or x.dtype != np.uint8:
            raise ValueError("x must be uint8 [N,224,224,3]")
        if len(x) == 0:
            return []
        xin = np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        hA, hB, logits, rl = self.sess.run(["hA", "hB", "logits", "review_logit"], {"x": xin})
        pA, pB, pF = self._probe("A", hA), self._probe("B", hB), _softmax(logits.astype(np.float64))
        out = []
        for i in range(len(x)):
            nt = _no_tissue(x[i])
            rev, score, reasons = device_review_rule(pA[i], pB[i], pF[i], float(rl[i]), self.thr, nt)
            k = int(np.argmax(pF[i]))
            out.append(TileResult(label=self.classes[k], prob=float(pF[i][k]), probsA=pA[i].astype(float).tolist(),
                                  probsB=pB[i].astype(float).tolist(), probsF=pF[i].astype(float).tolist(),
                                  review=rev, review_score=score, reasons=reasons, no_tissue=nt))
        return out
