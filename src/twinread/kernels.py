"""K1: fused dual-reader preprocessing on the GPU, bit-faithful to the official PIL-based transforms.

Every reader's official transform is torchvision ``Resize(size, bicubic, antialias) -> CenterCrop(crop) -> ToTensor
-> Normalize`` on a PIL image, so the resize is Pillow's ``ImagingResample``. ``make_plan`` mirrors Pillow's
``precompute_coeffs`` + ``normalize_coeffs_8bpc`` (Resample.c) on the host, in float64 with the same operation order:

* separable, horizontal pass first into a uint8 intermediate, then the vertical pass;
* filter support ``2 * max(scale, 1)`` for bicubic (a = -0.5), ``center = (i + 0.5) * scale``, window bounds
  ``int(center -/+ support + 0.5)`` clipped to the image, taps normalized to sum 1;
* fixed point: ``PRECISION_BITS = 32 - 8 - 2 = 22``, taps scaled by ``1 << 22`` and rounded half away from zero,
  int32 accumulation from ``1 << 21``, ``>> 22`` and clip to 0..255.

The center crop is folded into the plan (only the output columns and rows that survive the crop are computed, and
only the input rows they touch), and ToTensor + Normalize become a 256-entry fp16 lookup per channel computed with
torchvision's own float32 ops, so a correct kernel is bit-exact to ``official(img).half()``.

Three implementations share one plan: the CUDA C++ extension (``csrc/``, two launches covering both readers),
Triton (``k1_triton.py``) and plain torch ops (CPU or GPU).

K2 (``camera_aug``, ``sample_camera_params``): a SIMULATED camera-of-screen capture for student training (warp, HED
jitter, display gamma + colour profile, moire, glare, defocus, white balance, sensor noise, JPEG-like 8x8 DCT
quantization). Every op is defined in ``twinread/ref/camera_ops.py`` (float64 numpy reference); the CUDA path is
``csrc/k2_camera_aug.cu`` (2 launches) and the fallback ``k2_torch.py``. K1 and K2 share one extension module; if
the K2 translation unit fails to build, ``load_ext`` rebuilds K1 alone so K1 stays usable.

Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import functools
import logging
import math
import os
import shutil
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from twinread.readers import PreprocessSpec
from twinread.ref import camera_ops as _co

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[2]
CSRC = Path(__file__).resolve().parent / "csrc"
SOURCES_K1 = [CSRC / "bindings.cpp", CSRC / "k1_dual_preprocess.cu"]
SOURCES = SOURCES_K1 + [CSRC / "k2_camera_aug.cu"]
PRECISION_BITS = 32 - 8 - 2  # Pillow Resample.c
_HALF = 1 << (PRECISION_BITS - 1)

_EXT: dict[str, types.ModuleType | None] = {}
EXT_ERRORS: dict[str, str] = {}
_WARNED: set[str] = set()
_FORCE_SCALAR = False  # tests: force the CUDA scalar fallback kernels (very wide frames / crop % 4 != 0 paths)


# ------------------------------------------------------------------------------------------- Pillow coefficients
def _bicubic(x: float) -> float:
    a = -0.5
    if x < 0.0:
        x = -x
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


def _bilinear(x: float) -> float:
    if x < 0.0:
        x = -x
    if x < 1.0:
        return 1.0 - x
    return 0.0


_FILTERS = {"bicubic": (_bicubic, 2.0), "bilinear": (_bilinear, 1.0)}


def _fixed(v: float) -> int:
    return int(-0.5 + v * (1 << PRECISION_BITS)) if v < 0 else int(0.5 + v * (1 << PRECISION_BITS))


@functools.lru_cache(maxsize=256)
def pil_coeffs(in_size: int, out_size: int, resample: str) -> tuple[tuple, tuple, int]:
    """Pillow ``precompute_coeffs`` + ``normalize_coeffs_8bpc`` for box (0, in_size).

    Returns (bounds, fixed-point taps, ksize): bounds[i] = (xmin, xlen), taps[i] has ksize int entries (zeros past
    xlen)."""
    filt, fsupport = _FILTERS[resample]
    scale = filterscale = in_size / out_size
    if filterscale < 1.0:
        filterscale = 1.0
    support = fsupport * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    ss = 1.0 / filterscale
    bounds, taps = [], []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin
        k, ww = [], 0.0
        for x in range(xmax):
            w = filt((x + xmin - center + 0.5) * ss)
            k.append(w)
            ww += w
        if ww != 0.0:
            k = [w / ww for w in k]
        bounds.append((xmin, xmax))
        taps.append(tuple(_fixed(w) for w in k) + (0,) * (ksize - xmax))
    return tuple(bounds), tuple(taps), ksize


def resized_geometry(H: int, W: int, spec: PreprocessSpec) -> tuple[int, int, int, int]:
    """torchvision Resize(int) (shorter side -> spec.size) and CenterCrop offsets: (new_h, new_w, top, left)."""
    short, long = (W, H) if W <= H else (H, W)
    new_short, new_long = spec.size, int(spec.size * long / short)
    new_w, new_h = (new_short, new_long) if W <= H else (new_long, new_short)
    if spec.crop > new_h or spec.crop > new_w:
        raise ValueError(f"crop {spec.crop} larger than the resized image {new_h}x{new_w} (padding unsupported)")
    top = int(round((new_h - spec.crop) / 2.0))
    left = int(round((new_w - spec.crop) / 2.0))
    return new_h, new_w, top, left


@dataclass(frozen=True, eq=False)
class Plan:
    """Device tensors for one (input shape, spec): horizontal taps for the crop's columns, vertical taps for its
    rows (relative to ``row0``), and the fp16 normalization LUT."""
    crop: int
    row0: int  # first input row the crop needs
    rows: int  # number of input rows the crop needs (= intermediate height)
    kh: int
    kv: int
    hx0: int  # input columns [hx0, hx1) the crop's horizontal windows touch
    hx1: int
    hb: torch.Tensor  # int32 [crop, 2] (xmin, xlen)
    hk: torch.Tensor  # int32 [crop, kh]
    vb: torch.Tensor  # int32 [crop, 2] (ymin - row0, ylen)
    vk: torch.Tensor  # int32 [crop, kv]
    lut: torch.Tensor  # fp16 [3, 256]


def norm_lut(spec: PreprocessSpec) -> torch.Tensor:
    """ToTensor + Normalize for every uint8 level, with torchvision's own CPU float32 ops, cast to fp16 [3, 256]."""
    lv = torch.arange(256, dtype=torch.uint8).to(torch.float32).div(255).view(1, 256).repeat(3, 1)
    mean = torch.as_tensor(spec.mean, dtype=torch.float32).view(3, 1)
    std = torch.as_tensor(spec.std, dtype=torch.float32).view(3, 1)
    return lv.sub_(mean).div_(std).half()


@functools.lru_cache(maxsize=128)
def _plan_cached(H: int, W: int, spec: PreprocessSpec, device: str) -> Plan:
    if spec.crop > spec.size:
        raise ValueError(f"spec crop {spec.crop} > size {spec.size} is unsupported")
    new_h, new_w, top, left = resized_geometry(H, W, spec)
    c = spec.crop
    hbo, hko, kh = pil_coeffs(W, new_w, spec.resample)
    vbo, vko, kv = pil_coeffs(H, new_h, spec.resample)
    hb, hk = hbo[left:left + c], hko[left:left + c]
    vb, vk = vbo[top:top + c], vko[top:top + c]
    row0 = min(b[0] for b in vb)
    rows = max(b[0] + b[1] for b in vb) - row0
    vb = tuple((b[0] - row0, b[1]) for b in vb)
    # bounds invariants (no out-of-bounds reads in any implementation)
    assert all(b[0] >= 0 and b[1] >= 1 and b[0] + b[1] <= W for b in hb)
    assert 0 <= row0 and row0 + rows <= H
    assert all(b[0] >= 0 and b[1] >= 1 and b[0] + b[1] <= rows for b in vb)
    i32 = dict(dtype=torch.int32)
    t = lambda v: torch.tensor(v, **i32).to(device)  # noqa: E731
    hx0, hx1 = min(b[0] for b in hb), max(b[0] + b[1] for b in hb)
    return Plan(c, row0, rows, kh, kv, hx0, hx1, t(hb), t(hk), t(vb), t(vk), norm_lut(spec).to(device))


def make_plan(H: int, W: int, spec: PreprocessSpec, device: str | torch.device = "cpu") -> Plan:
    return _plan_cached(int(H), int(W), spec, str(torch.device(device)))


# ------------------------------------------------------------------------------------------- torch reference path
def _resample_torch(x: torch.Tensor, p: Plan) -> torch.Tensor:
    """uint8 [B,H,W,3] -> uint8-valued int32 [B,crop,crop,3], same fixed-point math as Pillow."""
    B = x.shape[0]
    xs = x[:, p.row0:p.row0 + p.rows].to(torch.int32)
    W = xs.shape[2]
    acc = torch.full((B, p.rows, p.crop, 3), _HALF, dtype=torch.int32, device=x.device)
    for j in range(p.kh):
        idx = (p.hb[:, 0] + j).clamp(max=W - 1).long()  # taps past xlen are 0
        acc += xs[:, :, idx, :] * p.hk[:, j].view(1, 1, -1, 1)
    tmp = (acc >> PRECISION_BITS).clamp_(0, 255)
    del acc, xs
    acc = torch.full((B, p.crop, p.crop, 3), _HALF, dtype=torch.int32, device=x.device)
    for j in range(p.kv):
        idx = (p.vb[:, 0] + j).clamp(max=p.rows - 1).long()
        acc += tmp[:, idx, :, :] * p.vk[:, j].view(1, -1, 1, 1)
    return (acc >> PRECISION_BITS).clamp_(0, 255)


def _torch_one(x: torch.Tensor, p: Plan, chunk: int = 32) -> torch.Tensor:
    B = x.shape[0]
    out = torch.empty((B, 3, p.crop, p.crop), dtype=torch.float16, device=x.device)
    ch = torch.arange(3, device=x.device).view(1, 1, 1, 3)
    for s in range(0, B, chunk):
        lv = _resample_torch(x[s:s + chunk], p).long()
        out[s:s + chunk] = p.lut[ch, lv].permute(0, 3, 1, 2)
    return out


def resize_crop_uint8(x: torch.Tensor, spec: PreprocessSpec) -> torch.Tensor:
    """PIL-exact Resize(shorter side) + CenterCrop as uint8 [B,crop,crop,3] (torch path; for tests/debugging)."""
    _check_input(x)
    p = make_plan(x.shape[1], x.shape[2], spec, x.device)
    return _resample_torch(x, p).to(torch.uint8)


# ------------------------------------------------------------------------------------------- CUDA extension
def _arch_tag(arch: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in arch.replace(".", "").replace("+PTX", "ptx")).strip("_")


def default_arch() -> str:
    """``$TORCH_CUDA_ARCH_LIST``, else the current GPU's compute capability, else "12.0" (the local RTX 5060)."""
    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        return env
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}"
    except Exception:  # noqa: BLE001
        pass
    return "12.0"


def _build(name: str, sources: list[Path], bdir: Path, defines: list[str], verbose: bool) -> types.ModuleType:
    from torch.utils.cpp_extension import load
    bdir.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=[str(s) for s in sources], build_directory=str(bdir),
                extra_cflags=["-O3", *defines], extra_cuda_cflags=["-O3", *defines], verbose=verbose)


def load_ext(arch_list: str | None = None, verbose: bool = False) -> types.ModuleType | None:
    """JIT-build (cached) the K1+K2 CUDA extension into ``$P/build/ext_<arch>``; None + logged reason on failure.

    If the combined build fails, K1 alone is built (``-DTWINREAD_NO_K2``, ``build/ext_<arch>_k1``) so a K2 problem
    leaves K1 usable; the K2 error stays in ``EXT_ERRORS[<arch tag>]`` and ``load_k2_ext`` returns None.
    ``arch_list`` defaults to ``default_arch()``. Delta: "8.0" (A100), "8.6" (A40), "9.0" (GH200 on DeltaAI).
    """
    arch = arch_list or default_arch()
    tag = _arch_tag(arch)
    if tag in _EXT:
        return _EXT[tag]
    saved = {k: os.environ.get(k) for k in ("TORCH_CUDA_ARCH_LIST", "CUDA_HOME", "PATH", "MAX_JOBS")}
    mod = None
    try:
        # a loaded CUDA module may put nvcc on PATH without CUDA_HOME (Delta); honour it, as torch's own lookup does
        nvcc = shutil.which("nvcc")
        cuda_home = (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
                     or (os.path.dirname(os.path.dirname(nvcc)) if nvcc else None) or "/usr/local/cuda")
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        os.environ.setdefault("MAX_JOBS", "4")
        try:
            mod = _build(f"twinread_ext_{tag}", SOURCES, PROJECT / "build" / f"ext_{tag}", [], verbose)
        except Exception as e:  # noqa: BLE001 - K1+K2 failed: keep K1 usable
            EXT_ERRORS[tag] = f"{type(e).__name__}: {e}"
            log.warning("twinread K1+K2 extension (arch %s) failed to build: %s; building K1 alone", arch,
                        EXT_ERRORS[tag])
            mod = _build(f"twinread_ext_{tag}_k1", SOURCES_K1, PROJECT / "build" / f"ext_{tag}_k1",
                         ["-DTWINREAD_NO_K2"], verbose)
    except Exception as e:  # noqa: BLE001 - any build/import failure means "use a fallback"
        EXT_ERRORS[f"{tag}_k1"] = f"{type(e).__name__}: {e}"
        log.warning("twinread CUDA extension (arch %s) unavailable: %s", arch, EXT_ERRORS[f"{tag}_k1"])
        mod = None
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _EXT[tag] = mod
    return mod


def load_k2_ext(arch_list: str | None = None, verbose: bool = False) -> types.ModuleType | None:
    """The extension module if it carries K2 (``camera_aug``), else None (reason in ``EXT_ERRORS``)."""
    mod = load_ext(arch_list, verbose)
    return mod if mod is not None and hasattr(mod, "camera_aug") else None


# ------------------------------------------------------------------------------------------- dispatcher
def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _check_input(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or x.dtype != torch.uint8 or x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected uint8 [B,H,W,3], got {getattr(x, 'dtype', type(x))} "
                         f"{tuple(getattr(x, 'shape', ()))}")
    if x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(f"empty frame {tuple(x.shape)}")


def _triton_ok(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    try:
        import triton  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_impl(impl: str, x: torch.Tensor) -> str:
    """cuda -> triton -> torch, warning once per downgrade."""
    if impl not in ("cuda", "triton", "torch"):
        raise ValueError(f"unknown impl {impl!r}")
    if impl == "cuda":
        if x.is_cuda and load_ext() is not None:
            return "cuda"
        why = "input is not on CUDA" if not x.is_cuda else "CUDA extension unavailable"
        impl = "triton" if _triton_ok(x) else "torch"
        _warn_once(f"cuda->{impl}:{why}", f"K1 {why}; falling back to {impl}")
    if impl == "triton" and not _triton_ok(x):
        _warn_once("triton->torch", "K1 Triton path unavailable (no CUDA input or no triton); falling back to torch")
        impl = "torch"
    return impl


def dual_preprocess(x: torch.Tensor, specA: PreprocessSpec, specB: PreprocessSpec,
                    impl: Literal["cuda", "triton", "torch"] = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Both readers' input tensors from raw uint8 frames in one pass.

    x: uint8 [B,H,W,3] (any H, W; CUDA for the cuda/triton paths). Returns fp16 NCHW [B,3,specA.crop,specA.crop],
    [B,3,specB.crop,specB.crop], equal to ``official_transform(PIL.Image.fromarray(frame)).half()``: shorter side
    to spec.size with PIL's resampling, center crop, /255, normalize."""
    _check_input(x)
    B, H, W, _ = x.shape
    pa, pb = make_plan(H, W, specA, x.device), make_plan(H, W, specB, x.device)
    if B == 0:
        e = lambda p: torch.empty((0, 3, p.crop, p.crop), dtype=torch.float16, device=x.device)  # noqa: E731
        return e(pa), e(pb)
    impl = resolve_impl(impl, x)
    x = x.contiguous()
    if impl == "cuda":
        ext = load_ext()
        a, b = ext.dual_preprocess(x, [pa.hb, pa.hk, pa.vb, pa.vk, pa.lut], [pa.row0, pa.rows, pa.hx0, pa.hx1],
                                   [pb.hb, pb.hk, pb.vb, pb.vk, pb.lut], [pb.row0, pb.rows, pb.hx0, pb.hx1],
                                   force_scalar=_FORCE_SCALAR)
        return a, b
    if impl == "triton":
        from twinread.k1_triton import dual_preprocess_triton
        return dual_preprocess_triton(x, pa, pb)
    return _torch_one(x, pa), _torch_one(x, pb)


# =========================================================================================== K2 camera-of-screen
_K2_COL = {}
_c = 0
for _name, _w in _co.FIELDS:
    _K2_COL[_name] = (_c, _c + _w)
    _c += _w
assert _c == _co.NPARAM == 35


@dataclass
class CameraAugParams:
    """Per-image K2 parameters (a SIMULATED camera filming a screen). Each field is a float32 tensor [B] or [B, k]
    on the same device as the images; formulas in ``twinread/ref/camera_ops.py``.

    gamma [B] display gamma (1 = off); wb_gain [B,3] white-balance gains (1 = off); moire_amp [B] (0 = off),
    moire_period [B] px, moire_angle [B] rad; glare_amp [B] in [0,1] (0 = off), glare_angle [B] rad;
    warp [B,8] homography delta from identity in normalized coords (0 = off); blur_sigma [B] px, <= 4 (0 = off);
    noise_std [B] in [0,1] units (0 = off); jpeg_quality [B] IJG quality 10..95 (0 = JPEG off);
    hed_alpha [B,3] (1 = off), hed_beta [B,3] (0 = off); profile [B,9] display colour-profile matrix, row-major
    (identity = off; None means identity). ``profile`` is an addition to the brief's field list: it carries the
    "colour profile" half of the display op, and its default keeps the brief's 13-field construction valid."""
    gamma: torch.Tensor
    wb_gain: torch.Tensor
    moire_amp: torch.Tensor
    moire_period: torch.Tensor
    moire_angle: torch.Tensor
    glare_amp: torch.Tensor
    glare_angle: torch.Tensor
    warp: torch.Tensor
    blur_sigma: torch.Tensor
    noise_std: torch.Tensor
    jpeg_quality: torch.Tensor
    hed_alpha: torch.Tensor
    hed_beta: torch.Tensor
    profile: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.gamma.shape[0])

    def packed(self, B: int, device: str | torch.device) -> torch.Tensor:
        """float32 [B, 35] in the kernel's column order (``camera_ops.FIELDS``); validates every field's shape."""
        cols = []
        for name, w in _co.FIELDS:
            t = getattr(self, name)
            if t is None:
                t = torch.eye(3, dtype=torch.float32).reshape(1, 9).expand(B, 9)
            t = torch.as_tensor(t)
            ok = ((B, w),) + (((B,),) if w == 1 else ())
            if tuple(t.shape) not in ok:
                raise ValueError(f"CameraAugParams.{name}: expected shape {ok[-1]} for a batch of {B}, "
                                 f"got {tuple(t.shape)}")
            cols.append(t.to(device=device, dtype=torch.float32).reshape(B, w))
        return torch.cat(cols, 1).contiguous()

    @classmethod
    def from_arrays(cls, arrs: dict, device: str | torch.device = "cuda") -> "CameraAugParams":
        """From numpy arrays keyed by field ([B] or [B, k]); one host-to-device copy."""
        B = int(np.asarray(arrs["gamma"]).shape[0])
        P = np.concatenate([np.asarray(arrs[n], dtype=np.float32).reshape(B, w) for n, w in _co.FIELDS], 1)
        T = torch.from_numpy(np.ascontiguousarray(P)).to(device)
        return cls(**{n: (T[:, a] if b - a == 1 else T[:, a:b]) for n, (a, b) in _K2_COL.items()})

    @classmethod
    def from_dicts(cls, dicts: list[dict], device: str | torch.device = "cuda") -> "CameraAugParams":
        """From per-image dicts in the ``camera_ops`` format (extra keys such as seed/index are ignored)."""
        arrs = {n: np.asarray([d[n] for d in dicts], dtype=np.float32).reshape(len(dicts), w)
                for n, w in _co.FIELDS}
        return cls.from_arrays(arrs, device)

    @classmethod
    def identity(cls, B: int, device: str | torch.device = "cuda") -> "CameraAugParams":
        return cls.from_arrays(_co.params_from_uniforms(np.zeros((B, _co.N_UNIFORM)), 0.0), device)

    def to_dicts(self) -> list[dict]:
        """Per-image dicts in the ``camera_ops`` format (without seed/index)."""
        P = self.packed(len(self), "cpu").numpy()
        return [{n: (float(row[a]) if b - a == 1 else [float(v) for v in row[a:b]]) for n, (a, b) in _K2_COL.items()}
                for row in P]


def sample_camera_params(B: int, severity: float, seed: int, device: str | torch.device = "cuda") -> CameraAugParams:
    """Per-image params for a batch of B (severity 0 = identity params; ranges in ``camera_ops.params_from_uniforms``).

    Drawn on the host from numpy's PCG64 seeded by ``seed``, so the values do not depend on the device, and image i
    depends only on (seed, i): ``camera_ops.sample_params(severity, seed)`` is image 0."""
    if B < 0:
        raise ValueError(f"B must be >= 0, got {B}")
    arrs = _co.params_from_uniforms(_co.sample_uniforms(int(B), seed), float(severity))
    return CameraAugParams.from_arrays(arrs, device)


@functools.lru_cache(maxsize=1)
def _k2_consts() -> tuple[float, ...]:
    return tuple(float(v) for v in np.concatenate([_co.HED_FROM_RGB.ravel(), _co.RGB_FROM_HED.ravel(),
                                                   _co.dct_matrix().ravel()]))


def resolve_k2_impl(impl: str, x: torch.Tensor) -> str:
    """cuda -> torch, warning once per reason."""
    if impl not in ("cuda", "torch"):
        raise ValueError(f"unknown impl {impl!r}")
    if impl == "cuda":
        if x.is_cuda and load_k2_ext() is not None:
            return "cuda"
        why = "input is not on CUDA" if not x.is_cuda else "K2 CUDA extension unavailable"
        _warn_once(f"k2:cuda->torch:{why}", f"K2 {why}; falling back to torch")
    return "torch"


def _camera_aug_impl(x: torch.Tensor, params: CameraAugParams, seed: int, impl: str = "cuda",
                     return_float: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(uint8 output, float32 pre-JPEG image in [0,1] if ``return_float`` else None)."""
    _check_input(x)
    if not isinstance(params, CameraAugParams):
        raise TypeError(f"params must be CameraAugParams, got {type(params).__name__}")
    if impl not in ("cuda", "torch"):
        raise ValueError(f"unknown impl {impl!r}")
    B = x.shape[0]
    P = params.packed(B, x.device)
    if B == 0:
        return torch.empty_like(x), (torch.empty(x.shape, dtype=torch.float32, device=x.device) if return_float
                                     else None)
    impl = resolve_k2_impl(impl, x)
    seed = int(seed)
    if impl == "cuda":
        ext = load_k2_ext()
        smax = float(torch.nan_to_num(P[:, _K2_COL["blur_sigma"][0]], nan=0.0).max())  # one host sync per call
        s64 = seed & 0xFFFFFFFFFFFFFFFF
        s64 = s64 - (1 << 64) if s64 >= (1 << 63) else s64
        out, inter = ext.camera_aug(x.contiguous(), P, s64, _co.blur_radius(smax), list(_k2_consts()))
        return out, (inter if return_float else None)
    from twinread.k2_torch import camera_aug_torch
    return camera_aug_torch(x, P, seed, return_float)


def camera_aug(x: torch.Tensor, params: CameraAugParams, seed: int,
               impl: Literal["cuda", "torch"] = "cuda") -> torch.Tensor:
    """K2: simulated camera-of-screen capture. x uint8 [B,H,W,3] -> uint8 [B,H,W,3].

    Op order (fixed): warp (bilinear) -> HED jitter -> display gamma + colour profile -> moire -> glare -> defocus
    (separable Gaussian, radius ceil(3 sigma)) -> white balance -> sensor noise (Philox keyed by ``seed`` and the
    image's index in the batch) -> JPEG-like 8x8 DCT quantization (4:4:4, chroma not subsampled). ``impl="cuda"``
    falls back to torch (one warning) when x is not on CUDA or the extension is unavailable."""
    return _camera_aug_impl(x, params, seed, impl)[0]
