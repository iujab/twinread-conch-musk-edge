"""TwinRead-Edge student: one small CNN distilled from two readers, the fused reader and the review flag.

The student sees camera-of-screen-augmented tiles (K2 kernel, or the plain-torch fallback here) and learns, from teacher
targets computed on clean tiles (``results/<pair>/<dataset>/student_targets.npz``, Task 5):

* ``hA``/``hB``: L2-normalised embeddings in reader A's / reader B's text-aligned space (cosine + relational InfoNCE);
* ``logits``: the fused reader's class distribution (KD at temperature T, plus CE on labels);
* ``review_logit``: the teacher review label (s4 above its val threshold at the 10% budget; BCE).

Device contract (Task 11/14): ONNX input ``x`` float [N,3,224,224] in [0,1]; ImageNet normalisation is inside the
module. Device flag = argmax probe_A(hA) != argmax probe_B(hB) OR JS(pA,pB) > js OR sigmoid(review_logit) > review,
with the probes from ``probes_aligned.npz`` and the thresholds from ``thresholds.json`` (chosen here on VAL).

The torch camera augmentation is a simulation of a camera filming a screen, not real capture.
Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit

from twinread import review, stats

BACKBONES = ("mobilenetv4_conv_small", "efficientnet_b0", "mobileone_s0", "regnetx_002", "resnet18", "lcnet_050")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
REL_TEMPERATURE = 0.07
# Ruling R26 (plan amendment, val evidence in PROGRESS.md): rel = 0.0 for the device student. Row-wise InfoNCE at
# tau 0.07 shrinks the component all teacher embeddings share (they sit in a narrow cone), which distorts the device
# probes and the JS clause. InfoNCE stays available via the weights (CLI --loss-weights) and BigBatch.
DEFAULT_WEIGHTS = {"cos": 1.0, "rel": 0.0, "kd": 1.0, "ce": 0.5, "rev": 0.5}
MAX_OOM_RETRIES = 3  # consecutive training OOM exits (code 3) without a completed epoch before giving up
AUGS = ("k2", "torch", "none")
EVAL_SEVERITIES = (0.0, 0.5, 1.0)
EVAL_SEED = 20261017
EVAL_BATCH = 256
AUG_BLOCK = 64  # eval augmentation seed block (tiles)
STAGE_HEADROOM = 3 << 30  # bytes of VRAM left free when staging the train tiles on the GPU
REVIEW_BUDGET = 0.10
EXIT_RERUN = 3
ROOT = Path(__file__).resolve().parents[2]
DEVICE_FLAG_DEFINITION = ("flag = argmax probe_A(hA) != argmax probe_B(hB) OR JS(pA,pB) > thresholds.js OR "
                          "sigmoid(review_logit) > thresholds.review; pA/pB = numpy softmax of probes_aligned.npz "
                          "applied to the student's hA/hB. The device's no_tissue clause (Task 14) is not included.")
SAFETY = "Research prototype, not a diagnostic device. Camera effects are simulated (camera filming a screen)."


def _sev_name(s: float) -> str:
    return "clean" if s <= 0 else f"sev{s:.1f}"


# =============================================================================================================== model
def _trunk_dim(trunk: nn.Module) -> int:
    was = trunk.training
    trunk.eval()
    with torch.no_grad():
        f = trunk(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)).shape[-1]
    trunk.train(was)
    return int(f)


def _proj(f: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(f, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out))


class TwinStudent(nn.Module):
    """timm trunk (global-pooled features) + four heads. Any timm model name works; ``BACKBONES`` are the candidates.

    MobileOne-style multi-branch trunks must be reparameterised (``deploy()``) before ONNX export.
    """

    def __init__(self, backbone: str, dimA: int, dimB: int, n_classes: int, pretrained: bool = True,
                 hidden: int = 768):
        super().__init__()
        import timm
        self.config = {"backbone": backbone, "dimA": int(dimA), "dimB": int(dimB), "n_classes": int(n_classes),
                       "hidden": int(hidden)}
        self.trunk = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        f = _trunk_dim(self.trunk)
        self.feat_dim = f
        self.head_A = _proj(f, hidden, dimA)
        self.head_B = _proj(f, hidden, dimB)
        self.cls = nn.Linear(f, n_classes)
        self.review = nn.Linear(f, 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.trunk((x - self.mean) / self.std)
        return {"hA": F.normalize(self.head_A(f).float(), dim=-1),
                "hB": F.normalize(self.head_B(f).float(), dim=-1),
                "logits": self.cls(f).float(),
                "review_logit": self.review(f).float().squeeze(-1)}

    def deploy(self) -> "TwinStudent":
        """Fuse multi-branch blocks (MobileOne etc.) for export; a no-op for plain trunks."""
        from timm.utils import reparameterize_model
        self.trunk = reparameterize_model(self.trunk)
        return self.eval()


def load_student(path: Path, map_location="cpu") -> TwinStudent:
    ck = torch.load(Path(path), map_location=map_location, weights_only=False)
    m = TwinStudent(**ck["config"], pretrained=False)
    m.load_state_dict(ck["model"])
    return m.eval()


# =============================================================================================================== losses
_BIGBATCH: dict = {}


def _bigbatch_src() -> Path | None:
    env = os.environ.get("TWINREAD_BIGBATCH_SRC")
    if env:
        return Path(env)
    for p in Path(__file__).resolve().parents:
        cand = p / "bigbatch-flash-contrastive-kernel" / "src"
        if (cand / "bigbatch").is_dir():
            return cand
    return None


def bigbatch_clip_loss():
    """BigBatch's fused Triton CLIP/InfoNCE loss (read-only import), or None when unavailable."""
    if "fn" in _BIGBATCH:
        return _BIGBATCH["fn"]
    fn, src = None, _bigbatch_src()
    if src is not None and torch.cuda.is_available():
        os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "build" / "triton_cache"))  # keep caches in $P/build
        if str(src) not in sys.path:
            sys.path.append(str(src))
        old = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # never write into the other project
        try:
            from bigbatch.triton_kernels import clip_loss_triton as fn
        except Exception as e:  # missing triton, API change, ...
            _BIGBATCH["error"] = repr(e)
            fn = None
        finally:
            sys.dont_write_bytecode = old
    _BIGBATCH["fn"] = fn
    return fn


def info_nce(h: torch.Tensor, t: torch.Tensor, temperature: float = REL_TEMPERATURE,
             impl: Literal["auto", "torch", "bigbatch"] = "auto") -> torch.Tensor:
    """Symmetric InfoNCE between rows of h and t (row i of each is the positive pair); mean of both directions.

    Same math as BigBatch's CLIP loss with scale = 1/temperature (``impl='bigbatch'`` uses its fused Triton kernel).
    """
    if impl not in ("auto", "torch", "bigbatch"):
        raise ValueError(impl)
    n = h.shape[0]
    if n == 0:
        return h.sum() * 0.0
    if impl != "torch" and h.is_cuda:
        fn = bigbatch_clip_loss()
        if fn is not None:
            scale = torch.tensor(1.0 / temperature, device=h.device, dtype=torch.float32)
            return fn(h.float().contiguous(), t.float().contiguous(), scale)
    if impl == "bigbatch":
        raise RuntimeError(f"BigBatch fused CLIP loss unavailable ({_BIGBATCH.get('error', 'needs CUDA')})")
    logits = (h.float() @ t.float().T) / temperature
    labels = torch.arange(n, device=h.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def rel_impl_used(impl: str, device) -> str:
    if impl == "torch" or torch.device(device).type != "cuda":
        return "torch"
    return "bigbatch-triton" if bigbatch_clip_loss() is not None else "torch"


def distill_loss(out: dict, tA: torch.Tensor, tB: torch.Tensor, fused_logits: torch.Tensor, y: torch.Tensor,
                 review_y: torch.Tensor, T: float = 2.0, w: dict | None = None,
                 rel_impl: str = "auto") -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of cos, rel, kd, ce and rev (all in fp32). cos and rel are means over the two readers."""
    w = {**DEFAULT_WEIGHTS, **(w or {})}
    hA, hB = out["hA"].float(), out["hB"].float()
    tA, tB = F.normalize(tA.float(), dim=-1), F.normalize(tB.float(), dim=-1)
    cos_A = (1 - (hA * tA).sum(-1)).mean()
    cos_B = (1 - (hB * tB).sum(-1)).mean()
    with torch.set_grad_enabled(torch.is_grad_enabled() and w["rel"] != 0):  # weight 0: logged, no backward
        rel_A = info_nce(hA, tA, REL_TEMPERATURE, rel_impl)
        rel_B = info_nce(hB, tB, REL_TEMPERATURE, rel_impl)
    logits = out["logits"].float()
    kd = T * T * F.kl_div(F.log_softmax(logits / T, -1), F.log_softmax(fused_logits.float() / T, -1),
                          reduction="batchmean", log_target=True)
    ce = F.cross_entropy(logits, y.long())
    rev = F.binary_cross_entropy_with_logits(out["review_logit"].float(), review_y.float())
    parts = {"cos": (cos_A + cos_B) / 2, "rel": (rel_A + rel_B) / 2, "kd": kd, "ce": ce, "rev": rev}
    total = sum(w[k] * parts[k] for k in ("cos", "rel", "kd", "ce", "rev"))
    parts.update(cos_A=cos_A, cos_B=cos_B, rel_A=rel_A, rel_B=rel_B, total=total)
    keys = list(parts)
    vals = torch.stack([parts[k].detach().float() for k in keys]).tolist()  # one device sync
    return total, dict(zip(keys, vals))


# =============================================================================================================== augmentation
# Stain matrix of Ruifrok & Johnston as used by skimage.color.rgb2hed (rows: H, E, D in RGB optical density).
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)
_JPEG_LUMA = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float64)
_JPEG_CHROMA = np.full((8, 8), 99.0)
_JPEG_CHROMA[:4, :4] = [[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]
_k = np.arange(8)
_DCT8 = np.sqrt(np.where(_k[:, None] == 0, 1 / 8, 2 / 8)) * np.cos(np.pi * (2 * _k[None, :] + 1) * _k[:, None] / 16)


def _jpeg_like(img: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """8x8 DCT quantisation in YCbCr with the IJG-scaled standard tables (no chroma subsampling). img [B,3,H,W] 0..1."""
    B, _, H, W = img.shape
    dev = img.device
    ph, pw = (-H) % 8, (-W) % 8
    x = F.pad(img, (0, pw, 0, ph), mode="replicate") if (ph or pw) else img
    Hp, Wp = x.shape[-2:]
    r, g, b = (x * 255).unbind(1)
    ycc = torch.stack([0.299 * r + 0.587 * g + 0.114 * b,
                       -0.168736 * r - 0.331264 * g + 0.5 * b,
                       0.5 * r - 0.418688 * g - 0.081312 * b], 1)  # level-shifted: Y-128 below, Cb/Cr centred
    ycc[:, 0] -= 128
    blk = ycc.view(B, 3, Hp // 8, 8, Wp // 8, 8).permute(0, 1, 2, 4, 3, 5)  # [B,3,nh,nw,8,8]
    D = torch.tensor(_DCT8, dtype=torch.float32, device=dev)
    coef = D @ blk @ D.T
    q = quality.clamp(1, 100).double()
    s = torch.where(q < 50, 5000 / q, 200 - 2 * q).view(B, 1, 1).cpu()
    tabs = []
    for base in (_JPEG_LUMA, _JPEG_CHROMA):
        t = torch.floor((torch.from_numpy(base)[None] * s + 50) / 100).clamp(1, 255)
        tabs.append(t)
    Q = torch.stack([tabs[0], tabs[1], tabs[1]], 1).float().to(dev).view(B, 3, 1, 1, 8, 8)
    coef = torch.round(coef / Q) * Q
    blk = D.T @ coef @ D
    ycc = blk.permute(0, 1, 2, 4, 3, 5).reshape(B, 3, Hp, Wp)
    yy, cb, cr = ycc[:, 0] + 128, ycc[:, 1], ycc[:, 2]
    rgb = torch.stack([yy + 1.402 * cr, yy - 0.344136 * cb - 0.714136 * cr, yy + 1.772 * cb], 1) / 255
    return rgb[..., :H, :W]


def torch_camera_aug(x: torch.Tensor, severity: float, seed: int, as_float: bool = False) -> torch.Tensor:
    """Plain-torch camera-of-screen simulation (fallback for K2). uint8 [B,H,W,3] -> uint8 [B,H,W,3] (same device).

    Op order follows K2: small affine warp -> HED stain jitter -> display gamma + colour profile -> moire ->
    glare/brightness gradient -> defocus (per-image Gaussian) -> white balance -> sensor noise -> JPEG-like 8x8 DCT
    quantisation (70% of images). Every parameter scales with ``severity``; severity 0 is the identity.
    ``as_float`` returns the float [B,H,W,3] result in [0,1] before 8-bit rounding (tests).
    """
    if x.dtype != torch.uint8 or x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected uint8 [B,H,W,3], got {tuple(x.shape)} {x.dtype}")
    B, H, W, _ = x.shape
    s = max(float(severity), 0.0)
    if B == 0 or s == 0.0:
        return x.float() / 255 if as_float else x.clone()
    dev = x.device
    g = torch.Generator().manual_seed(int(seed) % (2 ** 63))  # per-image params on CPU: same on any device
    u = lambda *shape: torch.rand(*shape, generator=g)  # noqa: E731
    gd = torch.Generator(device=dev).manual_seed(int(seed) % (2 ** 63) + 1)
    img = x.permute(0, 3, 1, 2).float().div(255)

    # 1. warp: rotation +-4 deg, scale +-6%, shear +-0.04, shift +-4% (scaled by severity); reflection border
    ang = (u(B) * 2 - 1) * math.radians(4) * s
    sc, sh = 1 + (u(B) * 2 - 1) * 0.06 * s, (u(B) * 2 - 1) * 0.04 * s
    tx, ty = (u(B) * 2 - 1) * 0.04 * s, (u(B) * 2 - 1) * 0.04 * s
    c, si = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([c * sc, -si * sc + sh, tx], -1), torch.stack([si * sc, c * sc, ty], -1)], 1)
    grid = F.affine_grid(theta.to(dev), list(img.shape), align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    # 2. HED stain jitter (optical density space; alpha +-5%, beta +-0.02 OD)
    alpha = (1 + (u(B, 3) * 2 - 1) * 0.05 * s).to(dev)
    beta = ((u(B, 3) * 2 - 1) * 0.02 * s).to(dev)
    od = -torch.log(img.clamp(1e-3, 1))
    hed = torch.einsum("bchw,cd->bdhw", od, torch.tensor(_HED_FROM_RGB, dtype=torch.float32, device=dev))
    hed = hed * alpha.view(B, 3, 1, 1) + beta.view(B, 3, 1, 1)
    od = torch.einsum("bdhw,dc->bchw", hed, torch.tensor(_RGB_FROM_HED, dtype=torch.float32, device=dev))
    img = torch.exp(-od).clamp(0, 1)

    # 3. display gamma + colour profile (3x3 mix near identity, rows sum to 1 so greys stay grey)
    gamma = torch.exp((u(B) * 2 - 1) * 0.25 * s).to(dev).view(B, 1, 1, 1)
    img = img.pow(gamma)
    M = torch.eye(3) + (u(B, 3, 3) * 2 - 1) * 0.04 * s
    M = (M / M.sum(-1, keepdim=True)).to(dev)
    img = torch.einsum("bij,bjhw->bihw", M, img)

    # 4. moire: sinusoidal screen-grid beat pattern (period 2.5-8.5 px)
    yy = torch.arange(H, device=dev, dtype=torch.float32).view(1, H, 1)
    xx = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, W)
    amp, per = (u(B) * 0.035 * s).to(dev), (2.5 + u(B) * 6).to(dev)
    th, ph = (u(B) * math.pi).to(dev), (u(B) * 2 * math.pi).to(dev)
    v = lambda t: t.view(B, 1, 1)  # noqa: E731
    wave = torch.sin(2 * math.pi * (xx * v(torch.cos(th)) + yy * v(torch.sin(th))) / v(per) + v(ph))
    img = img + (v(amp) * wave).unsqueeze(1)

    # 5. glare / brightness gradient: screen-blend ramp toward white, up to 18% at one edge
    gamp, gth = (u(B) * 0.18 * s).to(dev), (u(B) * 2 * math.pi).to(dev)
    rx, ry = xx / max(W - 1, 1) - 0.5, yy / max(H - 1, 1) - 0.5
    ramp = (rx * v(torch.cos(gth)) + ry * v(torch.sin(gth))) / v(torch.cos(gth).abs() + torch.sin(gth).abs()) + 0.5
    glare = (v(gamp) * ramp).unsqueeze(1)
    img = img + glare * (1 - img.clamp(0, 1))

    # 6. defocus: separable Gaussian, sigma 0-1.3 px per image (radius 4)
    r = 4
    sig = (u(B) * 1.3 * s).clamp_min(1e-3).to(dev)
    t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-t[None] ** 2 / (2 * sig[:, None] ** 2))
    k = (k / k.sum(1, keepdim=True)).repeat_interleave(3, 0)  # [B*3, 2r+1]
    z = img.reshape(1, B * 3, H, W)
    z = F.conv2d(F.pad(z, (r, r, 0, 0), mode="replicate"), k.view(B * 3, 1, 1, 2 * r + 1), groups=B * 3)
    z = F.conv2d(F.pad(z, (0, 0, r, r), mode="replicate"), k.view(B * 3, 1, 2 * r + 1, 1), groups=B * 3)
    img = z.view(B, 3, H, W)

    # 7. white balance +-8% per channel
    img = img * (1 + (u(B, 3) * 2 - 1) * 0.08 * s).to(dev).view(B, 3, 1, 1)

    # 8. sensor noise: Gaussian, std up to 0.025
    std = (u(B) * 0.025 * s).to(dev).view(B, 1, 1, 1)
    img = (img + torch.randn(img.shape, generator=gd, device=dev) * std).clamp(0, 1)

    # 9. JPEG-like quantisation on 70% of images, quality 95 -> 35 at severity 1
    apply = (u(B) < 0.7).to(dev)
    if bool(apply.any()):
        q = 95 - u(B) * 60 * min(s, 1.0)
        img = torch.where(apply.view(B, 1, 1, 1), _jpeg_like(img, q), img).clamp(0, 1)

    out = img.permute(0, 2, 3, 1)
    if as_float:
        return out.contiguous()
    return (out * 255).round().clamp(0, 255).to(torch.uint8).contiguous()


_K2_STATE: dict = {}


def _k2_fallback(reason: str, stage: str = "import") -> None:
    st = _K2_STATE
    st["fns"], st["reason"], st["stage"] = None, reason, stage
    if not st.get("warned"):
        st["warned"] = True
        warnings.warn(f"K2 camera augmentation unavailable: {reason}; falling back to the plain-torch "
                      "camera augmentation (aug='torch')", UserWarning, stacklevel=3)


def _k2():
    """(camera_aug, sample_camera_params) from twinread.kernels (Task 7), or None (one-time warning)."""
    st = _K2_STATE
    if "fns" not in st:
        try:
            from twinread.kernels import camera_aug, sample_camera_params
            st["fns"] = (camera_aug, sample_camera_params)
        except Exception as e:
            _k2_fallback(f"twinread.kernels.camera_aug not importable ({e!r})")
    return st["fns"]


def k2_fallback_info() -> dict | None:
    """{"stage": "import"|"runtime", "reason"} if K2 fell back to the torch augmentation in this process, else None."""
    st = _K2_STATE
    if "fns" in st and st["fns"] is None:
        return {"stage": st.get("stage", "import"), "reason": st.get("reason")}
    return None


def _resolved_aug(requested: str, impl_at_start: str, fallback: dict | None) -> str:
    """What actually ran: 'k2', 'torch' (K2 never importable) or 'k2+torch (runtime fallback)'."""
    if requested != "k2" or impl_at_start != "k2":
        return impl_at_start
    return "k2+torch (runtime fallback)" if fallback and fallback.get("stage") == "runtime" else "k2"


def aug_impl_used(aug: str) -> str:
    if aug not in AUGS:
        raise ValueError(aug)
    if aug == "k2":
        return "k2" if _k2() is not None else "torch"
    return aug


def augment(x: torch.Tensor, aug: str, severity: float, seed: int) -> torch.Tensor:
    """Camera-of-screen augmentation of uint8 [B,H,W,3] tiles on their device (k2 -> K2 kernel, else torch)."""
    if aug not in AUGS:
        raise ValueError(f"unknown aug {aug!r}; choose from {AUGS}")
    if aug == "none" or severity <= 0 or x.shape[0] == 0:
        return x
    seed = int(seed) % (2 ** 31)
    if aug == "k2":
        fns = _k2()
        if fns is not None:
            camera_aug, sample = fns
            try:
                p = sample(int(x.shape[0]), float(severity), seed, device=x.device)
                return camera_aug(x, p, seed, impl="cuda" if x.is_cuda else "torch")
            except torch.OutOfMemoryError:
                raise  # transient on a shared GPU: predict halves its batch, training exits 3 and resumes
            except Exception as e:
                _k2_fallback(f"K2 call failed ({e!r})", stage="runtime")
    return torch_camera_aug(x, severity, seed)


def random_dihedral(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Per-tile random flips (+ transpose for square tiles): the 8 dihedral symmetries of a histology tile."""
    B, H, W, _ = x.shape
    if B == 0:
        return x
    m = torch.randint(0, 8, (B,), generator=torch.Generator().manual_seed(int(seed) % (2 ** 63))).to(x.device)
    sel = lambda bit: (m & bit).bool().view(B, 1, 1, 1)  # noqa: E731
    x = torch.where(sel(1), x.flip(2), x)
    x = torch.where(sel(2), x.flip(1), x)
    if H == W:
        x = torch.where(sel(4), x.transpose(1, 2), x)
    return x


def to_input(x: torch.Tensor, size: int = INPUT_SIZE) -> torch.Tensor:
    """uint8 [B,H,W,3] -> float [B,3,size,size] in [0,1]. Non-square: center square crop first (= resize shorter
    side then center crop, the device semantics)."""
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected [B,H,W,3], got {tuple(x.shape)}")
    B, H, W, _ = x.shape
    if B == 0:
        return torch.zeros(0, 3, size, size, device=x.device)
    if H != W:
        m = min(H, W)
        top, left = (H - m) // 2, (W - m) // 2
        x = x[:, top:top + m, left:left + m]
    z = x.permute(0, 3, 1, 2).float().div(255)
    if z.shape[-1] != size:
        z = F.interpolate(z, size=(size, size), mode="bilinear", align_corners=False, antialias=z.shape[-1] > size)
    return z


# =============================================================================================================== device rule
def probe_probs(h: np.ndarray, arrs: dict, r: str) -> np.ndarray:
    """softmax(((h - mean)/scale) @ coef.T + intercept): the device's numpy probe on the student's hA or hB."""
    z = (np.asarray(h, np.float64) - arrs[f"{r}_mean"].astype(np.float64)) / arrs[f"{r}_scale"].astype(np.float64)
    lo = z @ arrs[f"{r}_coef"].astype(np.float64).T + arrs[f"{r}_intercept"].astype(np.float64)
    lo -= lo.max(1, keepdims=True)
    e = np.exp(lo)
    return e / e.sum(1, keepdims=True)


def device_flags(pA, pB, review_logit, thr: dict) -> dict:
    pA, pB = np.asarray(pA, np.float64), np.asarray(pB, np.float64)
    dis = np.argmax(pA, 1) != np.argmax(pB, 1)
    js = review.js_divergence(pA, pB)
    rp = expit(np.asarray(review_logit, np.float64))
    js_fire, rev_fire = js > thr["js"], rp > thr["review"]
    return {"flag": dis | js_fire | rev_fire, "disagree": dis, "js": js, "js_fire": js_fire, "review_prob": rp,
            "review_fire": rev_fire}


def _topk_threshold(score: np.ndarray, mask: np.ndarray, k: int, empty: float = 1.0) -> float:
    """Threshold so that ``score > thr`` selects the top ``k`` of ``score[mask]`` (fewer on boundary ties)."""
    s = np.sort(np.asarray(score, np.float64)[mask])[::-1]
    if len(s) == 0:
        return float(empty)
    if k <= 0:
        return float(s[0])
    if k >= len(s):
        return float(np.nextafter(s[-1], -np.inf))
    return float((s[k - 1] + s[k]) / 2) if s[k - 1] > s[k] else float(s[k])


def choose_thresholds(pA, pB, review_logit, teacher_flag=None, budget: float = REVIEW_BUDGET, grid: int = 21) -> dict:
    """VAL thresholds {"js", "review"} so the device rule flags ~``budget`` of the tiles.

    The disagreement clause is fixed; the remaining budget is split between the review head and JS. Each split
    s in {0, 0.05, .., 1} gives top-k thresholds (review first among non-disagreeing tiles, then JS among tiles not yet
    flagged). Among splits whose flagged count is closest to the budget (within 0.5% of n), the one with the most
    teacher-flag overlap wins (fewest disagreements with the teacher flag at a fixed flagged count); ties -> smaller s.
    If disagreement alone reaches the budget, both thresholds are 1.0 (those clauses never fire).
    """
    pA, pB = np.asarray(pA, np.float64), np.asarray(pB, np.float64)
    n = len(pA)
    dis = np.argmax(pA, 1) != np.argmax(pB, 1) if n else np.zeros(0, bool)
    js = review.js_divergence(pA, pB) if n else np.zeros(0)
    rp = expit(np.asarray(review_logit, np.float64))
    k_total = int(round(budget * n))
    k_left = k_total - int(dis.sum())
    info = {"budget": budget, "n": int(n), "disagree_fraction": float(dis.mean()) if n else None}
    if n == 0 or k_left <= 0:
        return {"js": 1.0, "review": 1.0, "share_review": None, "flag_fraction": float(dis.mean()) if n else None,
                "teacher_overlap": None, "note": "disagreement alone reaches the budget", **info}
    teacher = None if teacher_flag is None else np.asarray(teacher_flag).astype(bool)
    cands = []
    for share in np.linspace(0, 1, grid):
        kr = int(round(share * k_left))
        t_rev = _topk_threshold(rp, ~dis, kr)
        fl = dis | (rp > t_rev)
        t_js = _topk_threshold(js, ~fl, k_total - int(fl.sum()))
        fl = fl | (js > t_js)
        tp = int((fl & teacher).sum()) if teacher is not None else 0
        cands.append({"share": float(share), "js": t_js, "review": t_rev, "n_flag": int(fl.sum()), "tp": tp})
    gap = min(abs(c["n_flag"] - k_total) for c in cands)
    tol = max(gap, int(0.005 * n))
    ok = [c for c in cands if abs(c["n_flag"] - k_total) <= tol]
    best = max(ok, key=lambda c: (c["tp"], -c["share"]))
    return {"js": float(min(max(best["js"], 0.0), 1.0)), "review": float(min(max(best["review"], 0.0), 1.0)),
            "share_review": best["share"], "flag_fraction": best["n_flag"] / n,
            "teacher_overlap": best["tp"] if teacher is not None else None, **info}


# =============================================================================================================== metrics
def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(s, np.float64)))


def _kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if len(a) == 0:
        return None
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = float(pa * pb + (1 - pa) * (1 - pb))
    return None if pe >= 1 else (po - pe) / (1 - pe)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _centred_cos(h, t, mu) -> float:
    a, b = np.asarray(h, np.float64) - mu, np.asarray(t, np.float64) - mu
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float((a * b).sum(1).mean())


def _kl(p, q) -> float:
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
    return float((p * (np.log(p) - np.log(q))).sum(1).mean())


def split_metrics(o: dict, tg: dict, probes: dict | None, classes: np.ndarray, thr: dict | None = None,
                  mu: dict | None = None) -> dict:
    """Metrics of student outputs ``o`` (numpy hA/hB/logits/review_logit) against split targets ``tg``.

    ``mu`` = {"A","B"}: mean of the L2-normalised TRAIN teacher embeddings; gives ``cos_{A,B}_centred`` = cos(h-mu,
    t-mu), which, unlike raw cosine, does not reward predicting the shared cone direction. ``probe_kl_{A,B}`` =
    mean KL(probe(teacher) || probe(student)) with the aligned probes the device uses.
    """
    y = tg["y"]
    pred = classes[np.argmax(o["logits"], 1)]
    tA = tg["tA"] / np.linalg.norm(tg["tA"], axis=1, keepdims=True)
    tB = tg["tB"] / np.linalg.norm(tg["tB"], axis=1, keepdims=True)
    m = {"n": int(len(y)), "bal_acc": stats.balanced_accuracy(y, pred), "accuracy": float((pred == y).mean()),
         "macro_f1": stats.macro_f1(y, pred),
         "agree_with_fused_teacher": float((np.argmax(o["logits"], 1) == np.argmax(tg["fused_logits"], 1)).mean()),
         "cos_A": float((o["hA"] * tA).sum(1).mean()), "cos_B": float((o["hB"] * tB).sum(1).mean()),
         "review_auroc": _auroc(tg["review"], o["review_logit"])}
    if mu is not None:
        m["cos_A_centred"] = _centred_cos(o["hA"], tA, mu["A"])
        m["cos_B_centred"] = _centred_cos(o["hB"], tB, mu["B"])
    if probes is not None:
        pA, pB = probe_probs(o["hA"], probes, "A"), probe_probs(o["hB"], probes, "B")
        m["probe_kl_A"] = _kl(probe_probs(tg["tA"], probes, "A"), pA)
        m["probe_kl_B"] = _kl(probe_probs(tg["tB"], probes, "B"), pB)
        m["probe_A_bal_acc"] = stats.balanced_accuracy(y, classes[pA.argmax(1)])
        m["probe_B_bal_acc"] = stats.balanced_accuracy(y, classes[pB.argmax(1)])
        if thr is not None:
            fl = device_flags(pA, pB, o["review_logit"], thr)
            f, t = fl["flag"], np.asarray(tg["review"]).astype(bool)
            err = pred != y
            m["device_flag"] = {
                "fraction": float(f.mean()), "teacher_fraction": float(t.mean()),
                "agreement": float((f == t).mean()), "kappa": _kappa(f, t),
                "teacher_recall": float(f[t].mean()) if t.any() else None,
                "precision_vs_teacher": float(t[f].mean()) if f.any() else None,
                "clauses": {"disagree": float(fl["disagree"].mean()), "js": float(fl["js_fire"].mean()),
                            "review": float(fl["review_fire"].mean())},
                "student_error_rate": float(err.mean()),
                "student_errors_caught": float(f[err].mean()) if err.any() else None,
                "acc_cleared": float((~err[~f]).mean()) if (~f).any() else None,
                "review_head_auroc_vs_student_error": _auroc(err, o["review_logit"]),
                "js_auroc_vs_teacher_flag": _auroc(t, fl["js"])}
    return m


def eval_augment(xb: torch.Tensor, aug: str, severity: float, start: int) -> torch.Tensor:
    """Fixed-seed evaluation augmentation in blocks of AUG_BLOCK tiles seeded by the global block index, so the
    augmented tiles do not depend on the eval batch size (``start`` = global index of ``xb[0]``, a block multiple)."""
    if severity <= 0 or len(xb) == 0:
        return xb
    if start % AUG_BLOCK:
        raise ValueError(f"start must be a multiple of {AUG_BLOCK}")
    base = EVAL_SEED + 1000 * int(round(severity * 100))
    return torch.cat([augment(xb[j:j + AUG_BLOCK], aug, severity, base + (start + j) // AUG_BLOCK)
                      for j in range(0, len(xb), AUG_BLOCK)])


@torch.no_grad()
def predict(model: nn.Module, x, severity: float, aug_impl: str, device, batch: int = EVAL_BATCH) -> dict:
    """Student outputs on uint8 tiles, optionally augmented at a fixed severity (fixed per-block seeds).

    On CUDA OOM (shared GPU) the batch is halved down to AUG_BLOCK; results are unchanged because seeds are per block.
    """
    if batch % AUG_BLOCK:
        raise ValueError(f"eval batch must be a multiple of {AUG_BLOCK}")
    dev = torch.device(device)
    xt = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    model.eval()
    outs = defaultdict(list)
    amp = dev.type == "cuda"
    i = 0
    while i < len(xt):
        try:
            xb = eval_augment(xt[i:i + batch].to(dev), aug_impl, severity, i)
            z = to_input(xb)
            if amp:
                z = z.contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                o = model(z)
            o = {k: v.float().cpu().numpy() for k, v in o.items()}
        except torch.OutOfMemoryError:
            if batch <= AUG_BLOCK:
                raise
            xb = z = None
            if amp:
                torch.cuda.empty_cache()
            batch = max(AUG_BLOCK, (batch // 2) // AUG_BLOCK * AUG_BLOCK)
            warnings.warn(f"CUDA OOM in predict; eval batch halved to {batch}")
            continue
        for k, v in o.items():
            outs[k].append(v)
        i += batch
    keys = ("hA", "hB", "logits", "review_logit")
    if not outs:
        return {k: np.zeros((0,)) for k in keys}
    return {k: np.concatenate(outs[k]).astype(np.float32) for k in keys}


# =============================================================================================================== training
def resolve_targets_dir(pair: str, dataset: str, results_dir: Path | None = None) -> Path:
    rd = Path(results_dir) if results_dir else ROOT / "results"
    P, p = pair.upper(), pair.lower()
    cands = [rd / P / dataset, rd / p / P / dataset, rd / p / dataset]
    for c in cands:
        if (c / "student_targets.npz").exists():
            return c
    raise FileNotFoundError("student_targets.npz not found in any of " + ", ".join(map(str, cands)) +
                            " (run twinread.analyze.run_pair_analysis first, or pass --targets-dir)")


def _split_targets(T: dict, s: str) -> dict:
    return {"tA": T[f"tA_{s}"].astype(np.float32), "tB": T[f"tB_{s}"].astype(np.float32),
            "fused_logits": T[f"fused_logits_{s}"].astype(np.float32), "review": T[f"review_{s}"].astype(np.int64),
            "y": T[f"y_{s}"].astype(np.int64)}


def _check_aligned(split: str, y_img: np.ndarray, tg: dict) -> None:
    n = len(tg["y"])
    if len(y_img) != n or not np.array_equal(np.asarray(y_img).astype(np.int64), tg["y"]):
        raise ValueError(f"{split}: images from load_split ({len(y_img)} tiles) do not align with the targets "
                         f"({n} rows): labels differ; were the targets built from the same data/splits?")
    for k in ("tA", "tB", "fused_logits", "review"):
        if len(tg[k]) != n:
            raise ValueError(f"{split}: targets {k} has {len(tg[k])} rows, y has {n}; arrays do not align")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _mix(*ints) -> int:
    return int(np.random.SeedSequence([int(i) for i in ints]).generate_state(1)[0])


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=2))
    os.replace(tmp, path)


def _fits_on_gpu(nbytes: int, free: int, headroom: int = STAGE_HEADROOM) -> bool:
    return nbytes + headroom < free


def _stage(x: np.ndarray, dev: torch.device) -> torch.Tensor:
    """uint8 tiles as a torch tensor: GPU-resident when they fit with STAGE_HEADROOM to spare (shared GPU), else CPU."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dev.type == "cuda" and _fits_on_gpu(t.numel(), torch.cuda.mem_get_info(dev)[0]):
        return t.to(dev)
    return t


def _eval_val(model, xv, tv, probes, classes, dev, eval_aug, mu=None) -> dict:
    res = {}
    for sev in EVAL_SEVERITIES:
        o = predict(model, xv, sev, eval_aug, dev)
        res[_sev_name(sev)] = {k: v for k, v in split_metrics(o, tv, probes, classes, mu=mu).items()
                               if k in ("bal_acc", "cos_A", "cos_B", "cos_A_centred", "cos_B_centred",
                                        "probe_kl_A", "probe_kl_B", "review_auroc", "probe_A_bal_acc",
                                        "probe_B_bal_acc")}
    return res


def _aug_throughput(x: torch.Tensor, aug: str, dev: torch.device, batch: int, steps: int = 10, warmup: int = 2,
                    severity: float = 0.5) -> float | None:
    """images/s of augment + dihedral + resize to the model input, on ``dev``."""
    if len(x) == 0:
        return None
    xb = x[:batch].to(dev)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)  # noqa: E731
    for i in range(warmup + steps):
        if i == warmup:
            sync()
            t0 = time.perf_counter()
        z = to_input(random_dihedral(augment(xb, aug, severity, i), i))
    del z
    sync()
    return steps * len(xb) / (time.perf_counter() - t0)


def train_student(pair: str, dataset: str, backbone: str, epochs: int, batch: int,
                  aug: Literal["k2", "torch", "none"], out_dir: Path, severity_max: float = 1.0, seed: int = 0, *,
                  targets_dir: Path | None = None, data_dir: Path | None = None, results_dir: Path | None = None,
                  pretrained: bool = True, budget_s: float | None = None, device: str | None = None,
                  lr: float = 2e-3, lr_backbone: float = 1e-3, weight_decay: float = 0.01, T: float = 2.0,
                  weights: dict | None = None, rel_impl: str = "auto", eval_aug: str = "k2", log=None) -> dict:
    """Distil the teachers into a TwinStudent. Resumable: with ``budget_s`` it stops after the epoch that would
    overrun the budget and returns ``{"done": False}`` (CLI exit 3); rerun the same call to continue.

    Writes to ``out_dir``: last.pt (resume state), best.pt (best val epoch), selection.json (frozen before any test
    label is read), thresholds.json {"js","review"}, metrics.json, outputs.npz. Test metrics are computed only when
    ``preregistration.json`` exists next to the targets.
    """
    t_start = time.time()
    log = log or (lambda *a: print(*a, flush=True))
    torch.set_num_threads(4)
    if aug not in AUGS:
        raise ValueError(f"unknown aug {aug!r}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dev.type == "cuda"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitignore").write_text("*.pt\n")
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    targets_dir = Path(targets_dir) if targets_dir else resolve_targets_dir(pair, dataset, results_dir)
    tpath, ppath = targets_dir / "student_targets.npz", targets_dir / "probes_aligned.npz"
    with np.load(tpath) as z:
        TG = {k: z[k] for k in z.files}
    with np.load(ppath) as z:
        PR = {k: z[k] for k in z.files}
    if not np.array_equal(PR["A_classes"], PR["B_classes"]):
        raise ValueError("probes A and B have different class lists")
    classes = PR["A_classes"].astype(np.int64)
    tgs = {s: _split_targets(TG, s) for s in ("train", "val", "test")}
    dimA, dimB = tgs["train"]["tA"].shape[1], tgs["train"]["tB"].shape[1]
    n_classes = tgs["train"]["fused_logits"].shape[1]
    if len(classes) != n_classes or PR["A_coef"].shape[1] != dimA or PR["B_coef"].shape[1] != dimB:
        raise ValueError("probes_aligned.npz does not match the targets (classes or dims)")

    from twinread.data import load_split
    xs = {}
    for s in ("train", "val"):
        x, yv = load_split(data_dir, dataset, s)
        _check_aligned(s, yv, tgs[s])
        xs[s] = x
    n_train = len(xs["train"])
    bs = min(batch, n_train)
    steps = max(1, n_train // bs)  # drop_last: no size-1 BatchNorm batches
    cfg = {"pair": pair.upper(), "dataset": dataset, "backbone": backbone, "epochs": int(epochs), "batch": int(batch),
           "aug": aug, "severity_max": float(severity_max), "seed": int(seed), "lr": lr, "lr_backbone": lr_backbone,
           "weight_decay": weight_decay, "T": T, "weights": {**DEFAULT_WEIGHTS, **(weights or {})},
           "pretrained": bool(pretrained), "rel_impl": rel_impl, "eval_aug": eval_aug,
           "aug_impl": aug_impl_used(aug), "eval_aug_impl": aug_impl_used(eval_aug),
           "targets_sha256": _sha256(tpath), "probes_sha256": _sha256(ppath), "n_train": n_train,
           "n_val": len(xs["val"]), "dimA": int(dimA), "dimB": int(dimB), "n_classes": int(n_classes),
           "eval_protocol": {"aug_block": AUG_BLOCK, "eval_seed": EVAL_SEED, "severities": list(EVAL_SEVERITIES),
                             "selection": "mean val bal_acc over severities"}}
    chash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    torch.manual_seed(seed)
    model = TwinStudent(backbone, dimA, dimB, n_classes, pretrained=pretrained).to(dev)
    if amp:
        model = model.to(memory_format=torch.channels_last)
    bb = [p for k, p in model.named_parameters() if k.startswith("trunk.")]
    hd = [p for k, p in model.named_parameters() if not k.startswith("trunk.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": lr_backbone}, {"params": hd, "lr": lr}], weight_decay=weight_decay)
    total = steps * epochs
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (i - warm) /
                                                                                          max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    last, best_p = out_dir / "last.pt", out_dir / "best.pt"
    st = {"epoch": 0, "history": [], "best": {"score": -math.inf, "epoch": 0}, "train_imgs": 0, "train_secs": 0.0}
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck.get("config_hash") == chash:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            st = ck["state"]
            log(f"resuming from epoch {st['epoch']}/{epochs}")
        else:
            log("checkpoint in out_dir has a different config; starting fresh (stale outputs removed)")
    if st["epoch"] == 0:
        for f in ("metrics.json", "thresholds.json", "selection.json", "outputs.npz", "best.pt"):
            (out_dir / f).unlink(missing_ok=True)
    resumed_from = st["epoch"]

    eval_impl0, aug_impl0 = cfg["eval_aug_impl"], cfg["aug_impl"]  # resolved at start (in the hash)
    rel_used = rel_impl_used(rel_impl, dev)
    mu = {r: F.normalize(torch.from_numpy(tgs["train"][f"t{r}"]), dim=1).mean(0).numpy().astype(np.float64)
          for r in "AB"}
    oom_file = out_dir / "oom_retries.json"
    x_train = _stage(xs["train"], dev)
    tt = {k: torch.from_numpy(v).to(dev) for k, v in tgs["train"].items()}
    tt["tA"], tt["tB"] = F.normalize(tt["tA"], dim=1), F.normalize(tt["tB"], dim=1)
    sync = (lambda: torch.cuda.synchronize(dev)) if amp else (lambda: None)  # noqa: E731

    for ep in range(st["epoch"], epochs):
        t_ep = time.time()
        model.train()
        rng = np.random.default_rng([int(seed), ep])
        perm = rng.permutation(n_train)
        sevs = rng.uniform(0, severity_max, steps)
        sums = defaultdict(float)
        try:
            for i in range(steps):
                idx = torch.from_numpy(perm[i * bs:(i + 1) * bs])
                xb = x_train[idx.to(x_train.device)].to(dev, non_blocking=True)
                if aug != "none":
                    s_seed = _mix(seed, ep, i)
                    xb = random_dihedral(augment(xb, aug, float(sevs[i]), s_seed), s_seed)
                z = to_input(xb)
                if amp:
                    z = z.contiguous(memory_format=torch.channels_last)
                idd = idx.to(dev)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    out = model(z)
                loss, parts = distill_loss(out, tt["tA"][idd], tt["tB"][idd], tt["fused_logits"][idd], tt["y"][idd],
                                           tt["review"][idd], T=T, w=weights, rel_impl=rel_impl)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                for k, v in parts.items():
                    sums[k] += v
        except torch.OutOfMemoryError:
            xb = z = out = loss = None
            if amp:
                torch.cuda.empty_cache()
            prev = json.loads(oom_file.read_text()) if oom_file.exists() else {}
            n_oom = (prev.get("count", 0) if prev.get("config_hash") == chash else 0) + 1
            if n_oom > MAX_OOM_RETRIES:
                oom_file.unlink(missing_ok=True)
                raise
            oom_file.write_text(json.dumps({"count": n_oom, "epoch": ep + 1, "config_hash": chash}))
            log(f"CUDA OOM in epoch {ep + 1} (shared GPU; try {n_oom}/{MAX_OOM_RETRIES}); "
                f"rerun to resume from epoch {st['epoch']} (exit {EXIT_RERUN})")
            return {"done": False, "stopped": "cuda_oom", "epochs_completed": st["epoch"],
                    "resumed_from_epoch": resumed_from, "out_dir": str(out_dir)}
        sync()
        secs = time.time() - t_ep
        st["train_secs"] += secs
        st["train_imgs"] += steps * bs
        oom_file.unlink(missing_ok=True)
        val = _eval_val(model, xs["val"], tgs["val"], PR, classes, dev, eval_aug, mu)
        score = float(np.mean([val[_sev_name(s)]["bal_acc"] for s in EVAL_SEVERITIES]))
        st["epoch"] = ep + 1
        fb = k2_fallback_info()
        if fb and fb["stage"] == "runtime":
            st["k2_fallback"] = st.get("k2_fallback") or {**fb, "epoch": ep + 1}
        st["history"].append({"epoch": ep + 1, "train_loss": {k: v / steps for k, v in sums.items()},
                              "val": val, "select_score": score, "train_seconds": round(secs, 2),
                              "epoch_seconds": round(time.time() - t_ep, 2),
                              "lr_backbone": opt.param_groups[0]["lr"], "lr_heads": opt.param_groups[1]["lr"]})
        if score > st["best"]["score"]:
            st["best"] = {"score": score, "epoch": ep + 1}
            _atomic_save({"config": model.config, "classes": classes.tolist(), "epoch": ep + 1, "train_config": cfg,
                          "model": model.state_dict()}, best_p)
        _atomic_save({"config_hash": chash, "model": model.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict(), "scaler": scaler.state_dict(), "state": st}, last)
        log(f"epoch {ep + 1}/{epochs} loss {sums['total'] / steps:.4f} val bal_acc "
            + " ".join(f"{k}={v['bal_acc']:.4f}" for k, v in val.items())
            + f" cosA {val['clean']['cos_A']:.3f} cosB {val['clean']['cos_B']:.3f} ({time.time() - t_ep:.1f}s)")
        ep_total = time.time() - t_ep
        if budget_s is not None and ep + 1 < epochs and (time.time() - t_start) + ep_total > budget_s:
            log(f"time budget {budget_s:.0f}s used; rerun to resume (exit {EXIT_RERUN})")
            return {"done": False, "epochs_completed": ep + 1, "resumed_from_epoch": resumed_from,
                    "out_dir": str(out_dir)}

    # ---- final: best checkpoint -> val outputs -> thresholds -> frozen selection -> test (prereg-gated) ----------
    ck = torch.load(best_p, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    outs = {"val": {s: predict(model, xs["val"], s, eval_aug, dev) for s in EVAL_SEVERITIES}}
    vo = outs["val"][0.0]
    thr_info = choose_thresholds(probe_probs(vo["hA"], PR, "A"), probe_probs(vo["hB"], PR, "B"), vo["review_logit"],
                                 tgs["val"]["review"], budget=REVIEW_BUDGET)
    thr = {"js": float(thr_info["js"]), "review": float(thr_info["review"])}
    _atomic_json(thr, out_dir / "thresholds.json")
    metrics = {"pair": pair.upper(), "dataset": dataset, "backbone": backbone, "epochs": epochs, "batch": batch,
               "seed": seed, "severity_max": severity_max, "aug_requested": aug, "eval_aug_requested": eval_aug,
               "eval_severities": list(EVAL_SEVERITIES), "eval_seed": EVAL_SEED,
               "rel_loss_impl": rel_used, "loss_weights": cfg["weights"], "T": T, "lr": lr, "lr_backbone": lr_backbone,
               "weight_decay": weight_decay, "device": str(dev), "amp_fp16": amp, "targets_dir": str(targets_dir),
               "targets_sha256": cfg["targets_sha256"], "config_hash": chash,
               "n": {"train": n_train, "val": len(xs["val"]), "test": int(len(tgs["test"]["y"]))},
               "best_epoch": st["best"]["epoch"],
               "selection_rule": "best epoch = highest mean val balanced accuracy over severities "
                                 f"{list(EVAL_SEVERITIES)} (eval augmentation {eval_aug}, fixed seed)",
               "device_flag_definition": DEVICE_FLAG_DEFINITION, "thresholds": thr_info, "note": SAFETY}
    metrics["val"] = {_sev_name(s): split_metrics(o, tgs["val"], PR, classes, thr, mu)
                      for s, o in outs["val"].items()}
    _atomic_json({"config_hash": chash, "best_epoch": st["best"]["epoch"], "thresholds": thr,
                  "selection_rule": metrics["selection_rule"], "val_clean": metrics["val"]["clean"],
                  "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
                 out_dir / "selection.json")
    if (targets_dir / "preregistration.json").exists():
        xt, yt = load_split(data_dir, dataset, "test")
        _check_aligned("test", yt, tgs["test"])
        outs["test"] = {s: predict(model, xt, s, eval_aug, dev) for s in EVAL_SEVERITIES}
        metrics["test"] = {_sev_name(s): split_metrics(o, tgs["test"], PR, classes, thr, mu)
                           for s, o in outs["test"].items()}
    else:
        metrics["test_skipped"] = f"no preregistration.json in {targets_dir}; test metrics not computed"
    aug_ips = _aug_throughput(x_train, aug, dev, bs) if aug != "none" else None
    # what actually ran, decided AFTER every augmentation call (a K2 runtime failure switches to torch mid-run)
    fb = k2_fallback_info()
    if fb and fb["stage"] == "runtime" and not st.get("k2_fallback"):
        st["k2_fallback"] = {**fb, "epoch": "final evaluation"}
    fb_all = st.get("k2_fallback") or (fb if "k2" in (aug, eval_aug) else None)
    aug_used, eval_used = _resolved_aug(aug, aug_impl0, fb_all), _resolved_aug(eval_aug, eval_impl0, fb_all)
    metrics.update(aug_used=aug_used, eval_aug_used=eval_used, k2_fallback=fb_all)
    metrics["throughput"] = {
        "train_images_per_s": st["train_imgs"] / max(st["train_secs"], 1e-9),
        "train_note": f"training step incl. aug={aug_used}, host->device copy, fwd+bwd, AMP={amp}; batch {bs}",
        "aug_images_per_s": aug_ips, "device_name": torch.cuda.get_device_name(dev) if amp else "cpu"}
    metrics["history"] = st["history"]
    np.savez_compressed(out_dir / "outputs.npz", **{
        f"{sp}_{k}": v for sp in outs for k, v in {
            "pA": probe_probs(outs[sp][0.0]["hA"], PR, "A").astype(np.float16),
            "pB": probe_probs(outs[sp][0.0]["hB"], PR, "B").astype(np.float16),
            "logits": outs[sp][0.0]["logits"].astype(np.float16),
            "review_logit": outs[sp][0.0]["review_logit"].astype(np.float32)}.items()})
    _atomic_json(metrics, out_dir / "metrics.json")
    log(f"done: best epoch {st['best']['epoch']}; val clean bal_acc {metrics['val']['clean']['bal_acc']:.4f}; "
        f"thresholds {thr}; aug_used {aug_used}; eval_aug_used {eval_used}")
    return {**metrics, "done": True, "epochs_completed": st["epoch"], "resumed_from_epoch": resumed_from,
            "out_dir": str(out_dir)}


# =============================================================================================================== throughput
def measure_throughput(x: np.ndarray, backbone: str, dimA: int, dimB: int, n_classes: int,
                       augs=("none", "torch", "k2"), batch: int = 128, steps: int = 20, warmup: int = 5,
                       device: str | None = None, pretrained: bool = False, severity: float = 0.5) -> dict:
    """images/s of (a) the augmentation path alone and (b) a full training step, per aug. CUDA-event timing on a
    shared GPU (median per-step time), so treat as an estimate."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = dev.type == "cuda"
    torch.manual_seed(0)
    model = TwinStudent(backbone, dimA, dimB, n_classes, pretrained=pretrained).to(dev).train()
    if amp:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    n = min(batch, len(x))
    xb_all = torch.from_numpy(np.ascontiguousarray(x[:n])).to(dev)
    g = torch.Generator().manual_seed(0)
    tA = F.normalize(torch.randn(n, dimA, generator=g), dim=1).to(dev)
    tB = F.normalize(torch.randn(n, dimB, generator=g), dim=1).to(dev)
    fl = torch.log_softmax(torch.randn(n, n_classes, generator=g), 1).to(dev)
    y = torch.randint(0, n_classes, (n,), generator=g).to(dev)
    rv = torch.randint(0, 2, (n,), generator=g).to(dev)

    def timed(fn):
        ts = []
        for i in range(warmup + steps):
            if amp:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fn(i)
                b.record()
                torch.cuda.synchronize(dev)
                ts.append(a.elapsed_time(b) / 1000)
            else:
                t0 = time.perf_counter()
                fn(i)
                ts.append(time.perf_counter() - t0)
        return float(np.median(ts[warmup:]))

    res = {}
    for aug in augs:
        impl = aug_impl_used(aug)

        def prep(i, aug=aug):
            xb = xb_all
            if aug != "none":
                xb = random_dihedral(augment(xb, aug, severity, i), i)
            z = to_input(xb)
            return z.contiguous(memory_format=torch.channels_last) if amp else z

        def step(i, prep=prep, rel="torch"):
            z = prep(i)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                out = model(z)
            loss, _ = distill_loss(out, tA, tB, fl, y, rv, rel_impl=rel)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        t_aug, t_step = timed(prep), timed(step)
        res[aug] = {"impl_used": impl, "aug_images_per_s": n / t_aug, "train_step_images_per_s": n / t_step,
                    "aug_ms_per_batch": 1000 * t_aug, "train_step_ms_per_batch": 1000 * t_step,
                    "rel_loss_impl": "torch"}
        if rel_impl_used("auto", dev) != "torch":
            t_bb = timed(lambda i: step(i, rel="bigbatch"))
            res[aug]["train_step_images_per_s_bigbatch_rel"] = n / t_bb
            res[aug]["train_step_ms_per_batch_bigbatch_rel"] = 1000 * t_bb
    del model, opt
    if amp:
        torch.cuda.empty_cache()
    return {"backbone": backbone, "batch": n, "steps": steps, "warmup": warmup, "severity": severity,
            "device": str(dev), "device_name": torch.cuda.get_device_name(dev) if amp else "cpu",
            "torch": torch.__version__, "timing": "CUDA events, median per step" if amp else "perf_counter median",
            "note": "shared GPU; estimate. Augmentation = simulated camera filming a screen.", "results": res}
