"""ONNX export, static int8 quantisation and the edge bundle (ruling R24). Research prototype, not a diagnostic device."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ("hA", "hB", "logits", "review_logit")
LICENCE = {"OPEN": "OPEN pair (PathGen-CLIP + QuiltNet-B/16 teachers, CC-BY-4.0 / MIT): distributable.",
           "CM": "CONCH + MUSK derived student: model licences (CC-BY-NC-ND) do NOT permit redistribution; "
                 "keep local, not distributable."}


class _Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        return tuple(o[k] for k in OUTPUTS)


def export_onnx(model, path: Path, opset: int = 17) -> Path:
    """Reparameterise (no-op unless MobileOne-style), then export x float [N,3,224,224] in [0,1] -> OUTPUTS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().float().deploy().eval()
    dyn = {"x": {0: "N"}, **{k: {0: "N"} for k in OUTPUTS}}
    with torch.no_grad():
        torch.onnx.export(_Wrap(model).eval(), (torch.rand(2, 3, 224, 224),), str(path), input_names=["x"],
                          output_names=list(OUTPUTS), dynamic_axes=dyn, opset_version=opset, dynamo=False)
    return path


def quantize_int8(fp32_path: Path, calib_x: np.ndarray, out_path: Path, method: str = "minmax") -> Path:
    """ORT static quantisation, QDQ, per-channel weights; ``method`` minmax | entropy | percentile<P>. ``calib_x`` float32 [N,3,224,224] in [0,1]."""
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(np.asarray(calib_x, np.float32)[i:i + 1] for i in range(len(calib_x)))

        def get_next(self):
            b = next(self.it, None)
            return None if b is None else {"x": b}

    method = method.lower()
    extra = {}
    if method.startswith("percentile"):  # "percentile99.99": clips activation outliers; MinMax/Entropy were poor here
        extra = {"CalibPercentile": float(method[len("percentile"):] or 99.999)}
        method = "percentile"
    m = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
         "percentile": CalibrationMethod.Percentile}[method]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(str(fp32_path), str(out_path), _Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=m, extra_options=extra or None)
    return out_path


def run_onnx(path: Path, x: np.ndarray, batch: int = 64, threads: int = 4) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    outs = [[] for _ in OUTPUTS]
    for i in range(0, len(x), batch):
        for l, o in zip(outs, s.run(list(OUTPUTS), {"x": np.ascontiguousarray(x[i:i + batch], np.float32)})):
            l.append(o)
    return {k: np.concatenate(l).astype(np.float32) for k, l in zip(OUTPUTS, outs)}


def _cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def compare_outputs(o32: dict, o8: dict) -> dict:
    cA, cB = _cos(o32["hA"], o8["hA"]), _cos(o32["hB"], o8["hB"])
    r32, r8 = o32["review_logit"] > 0, o8["review_logit"] > 0
    return {"cos_hA_mean": float(cA.mean()), "cos_hA_min": float(cA.min()),
            "cos_hB_mean": float(cB.mean()), "cos_hB_min": float(cB.min()),
            "label_agreement": float((o32["logits"].argmax(1) == o8["logits"].argmax(1)).mean()),
            "review_agreement": float((r32 == r8).mean()),
            "review_logit_note": "review agreement = sign of review_logit (sigmoid > 0.5) fp32 vs int8", "n": int(len(cA))}


def drift_report(fp32_path, int8_path, x: np.ndarray) -> dict:
    return compare_outputs(run_onnx(fp32_path, x), run_onnx(int8_path, x))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def save_npz_deterministic(path: Path, **arrays) -> None:
    """np.savez with fixed zip timestamps, so identical arrays give a byte-identical file (stable SHA256SUMS).
    Loadable with np.load."""
    import zipfile
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for name, a in arrays.items():
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            with z.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(a), allow_pickle=False)


def build_bundle(pair: str, onnx_int8: Path, banks: dict, probes: dict, classes: list, thresholds: dict,
                 out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    need = [f"{r}_{k}" for r in "AB" for k in ("coef", "intercept", "mean", "scale", "classes")]
    miss = [k for k in need if k not in probes]
    if miss:
        raise KeyError(f"probes missing keys {miss}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()
    shutil.copyfile(onnx_int8, out_dir / "model.onnx")
    save_npz_deterministic(out_dir / "prompt_bank.npz", bankA=np.asarray(banks["A"], np.float32), bankB=np.asarray(banks["B"], np.float32))
    save_npz_deterministic(out_dir / "probes.npz", **{k: (np.asarray(probes[k]) if k.endswith("classes")
                                            else np.asarray(probes[k], np.float32)) for k in need})
    pair = pair.upper()
    cfg = {"pair": pair, "classes": list(classes), "thresholds": {"js": float(thresholds["js"]),
                                                                  "review": float(thresholds["review"])},
           "licence_note": LICENCE.get(pair, LICENCE["CM"]), "distributable": pair == "OPEN"}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    names = sorted(p.name for p in out_dir.iterdir())
    (out_dir / "SHA256SUMS").write_text("".join(f"{_sha(out_dir / n)}  {n}\n" for n in names))
    return out_dir


def verify_bundle(bundle_dir: Path) -> bool:
    """Thin wrapper over the device runtime's checker (ruling R7); no duplicated checksum logic."""
    dev = str(ROOT / "device")
    if dev not in sys.path:
        sys.path.insert(0, dev)
    import twinread_runtime
    return twinread_runtime.verify_bundle_dir(bundle_dir)
