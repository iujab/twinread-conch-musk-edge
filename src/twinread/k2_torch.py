"""K2 torch fallback: the same camera-of-screen math as ``csrc/k2_camera_aug.cu`` and ``ref/camera_ops.py``, written
with batched torch ops (float32, CPU or GPU). Used for ``camera_aug(..., impl="torch")`` and whenever the CUDA
extension is unavailable. Per-image identity params select the unmodified input with ``torch.where``, so identity
params are exact here too. The sensor-noise stream is the same Philox4x32-10 stream as the kernel (emulated with
int64 16-bit-limb multiplies), so the two paths differ only by float32 operation order. TF32 is disabled for the
blur convolution and the DCT matmuls. Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from twinread.ref import camera_ops as co

_OFF = {}
_o = 0
for _name, _w in co.FIELDS:
    _OFF[_name] = (_o, _o + _w)
    _o += _w
CHUNK = 64  # images per chunk (bounds the float temporaries on a shared 8 GB card)
_M0, _M1, _W0, _W1, _MASK = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85, 0xFFFFFFFF


def _col(P: torch.Tensor, name: str) -> torch.Tensor:
    a, b = _OFF[name]
    return P[:, a:b] if b - a > 1 else P[:, a]


@contextlib.contextmanager
def _exact_fp32():
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, benchmark=torch.backends.cudnn.benchmark,
                                        deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
            yield
    finally:
        torch.set_float32_matmul_precision(prev)


def _norm_grid(H: int, W: int, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ys = torch.arange(H, device=dev, dtype=torch.float32).view(1, H, 1)
    xs = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, W)
    return xs, ys, (2.0 * xs + 1.0) / W - 1.0, (2.0 * ys + 1.0) / H - 1.0


def _sel(on: torch.Tensor, new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    return torch.where(on.view(-1, 1, 1, 1), new, old)


def _lin3(v: torch.Tensor, m) -> torch.Tensor:
    """Row-vector times 3x3 (python floats): out[..., j] = sum_i v[..., i] m[i][j] (elementwise, no TF32)."""
    return torch.stack([v[..., 0] * m[0][j] + v[..., 1] * m[1][j] + v[..., 2] * m[2][j] for j in range(3)], -1)


# ------------------------------------------------------------------------------------------- ops
def _warp(v, P, g):
    d = _col(P, "warp")
    on = (d != 0).any(1)
    if not bool(on.any()):
        return v
    xs, ys, xn, yn = g
    dd = [d[:, k].view(-1, 1, 1) for k in range(8)]
    u = (1.0 + dd[0]) * xn + dd[1] * yn + dd[2]
    w = torch.clamp_min(dd[6] * xn + dd[7] * yn + 1.0, 1e-6)
    vv = dd[3] * xn + (1.0 + dd[4]) * yn + dd[5]
    grid = torch.stack([u / w, vv / w], -1)
    out = F.grid_sample(v.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return _sel(on, out.permute(0, 2, 3, 1), v)


def _hed(v, P):
    a, b = _col(P, "hed_alpha"), _col(P, "hed_beta")
    on = ((a != 1) | (b != 0)).any(1)
    if not bool(on.any()):
        return v
    od = torch.log(v.clamp_min(1e-6)) / co.LOG_ADJ
    s = _lin3(od, co.HED_FROM_RGB.tolist()).clamp_min(0.0)
    s = s * a.view(-1, 1, 1, 3) + b.view(-1, 1, 1, 3)
    out = torch.exp(_lin3(s, co.RGB_FROM_HED.tolist()) * co.LOG_ADJ).clamp(0.0, 1.0)
    return _sel(on, out, v)


def _display(v, P):
    gam = _col(P, "gamma")
    on = gam != 1
    if bool(on.any()):
        v = _sel(on, v.clamp(0.0, 1.0).pow(gam.clamp(0.05, 20.0).view(-1, 1, 1, 1)), v)
    M = _col(P, "profile").view(-1, 3, 3)
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    on = (M != eye).flatten(1).any(1)
    if bool(on.any()):
        m = [[M[:, c, k].view(-1, 1, 1) for k in range(3)] for c in range(3)]
        out = torch.stack([m[c][0] * v[..., 0] + m[c][1] * v[..., 1] + m[c][2] * v[..., 2] for c in range(3)], -1)
        v = _sel(on, out.clamp(0.0, 1.0), v)
    return v


def _moire(v, P, g):
    amp = _col(P, "moire_amp")
    on = amp != 0
    if not bool(on.any()):
        return v
    xs, ys, _, _ = g
    per = _col(P, "moire_period").clamp_min(2.0).view(-1, 1, 1)
    th = _col(P, "moire_angle").view(-1, 1, 1)
    t = ((xs + 0.5) * torch.cos(th) + (ys + 0.5) * torch.sin(th)) / per
    ph = torch.arange(3, device=v.device, dtype=v.dtype) * (math.pi / 3.0)
    m = torch.cos(2.0 * math.pi * t.unsqueeze(-1) + ph)
    return _sel(on, (v * (1.0 + amp.view(-1, 1, 1, 1) * m)).clamp(0.0, 1.0), v)


def _glare(v, P, g):
    amp = _col(P, "glare_amp")
    on = amp != 0
    if not bool(on.any()):
        return v
    _, _, xn, yn = g
    phi = _col(P, "glare_angle").view(-1, 1, 1)
    d = (xn * torch.cos(phi) + yn * torch.sin(phi)) * (1.0 / math.sqrt(2.0))
    gg = (amp.clamp(0.0, 1.0).view(-1, 1, 1) * ((d + 1.0) * 0.5) ** 2).unsqueeze(-1)
    return _sel(on, (v + gg * (1.0 - v)).clamp(0.0, 1.0), v)


def _defocus(v, P):
    sraw = _col(P, "blur_sigma")
    on = sraw > 0
    if not bool(on.any()):
        return v
    sig = sraw.clamp(co.MIN_SIGMA, co.MAX_SIGMA)
    r_img = torch.where(on, torch.ceil(3.0 * sig), torch.zeros_like(sig))  # float32 product, as the kernel
    R = int(r_img.max().item())
    B, H, W, _ = v.shape
    k = torch.arange(-R, R + 1, device=v.device, dtype=v.dtype).view(1, -1)
    w = torch.exp(-(k * k) / (2.0 * sig.view(-1, 1) ** 2))
    w = torch.where(k.abs() <= r_img.view(-1, 1), w, torch.zeros_like(w))
    w = w / w.sum(1, keepdim=True)
    w = w.repeat_interleave(3, 0)  # [B*3, 2R+1] (image-major, channel-minor = permuted layout below)
    x = v.permute(0, 3, 1, 2).reshape(1, B * 3, H, W)
    x = F.conv2d(F.pad(x, (R, R, 0, 0), mode="replicate"), w.view(B * 3, 1, 1, -1), groups=B * 3)
    x = F.conv2d(F.pad(x, (0, 0, R, R), mode="replicate"), w.view(B * 3, 1, -1, 1), groups=B * 3)
    return _sel(on, x.view(B, 3, H, W).permute(0, 2, 3, 1), v)


def _wb(v, P):
    g = _col(P, "wb_gain")
    on = (g != 1).any(1)
    if not bool(on.any()):
        return v
    return _sel(on, (v * g.view(-1, 1, 1, 3)).clamp(0.0, 1.0), v)


def _mulhilo(m: int, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """32x32 -> 64-bit product of a constant and int64-held uint32 values, without int64 overflow."""
    t = m * (a & 0xFFFF)
    u = m * (a >> 16) + (t >> 16)
    return u >> 16, ((u & 0xFFFF) << 16) | (t & 0xFFFF)


def philox4x32_10_torch(c0, c1, c2, c3, key: tuple[int, int]):
    """Philox4x32-10 on int64 tensors holding uint32 values (broadcasting); same stream as curand."""
    k0, k1 = key
    for _ in range(10):
        hi0, lo0 = _mulhilo(_M0, c0)
        hi1, lo1 = _mulhilo(_M1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
    return c0, c1, c2, c3


def normals_torch(B: int, H: int, W: int, seed: int, offset: int, device) -> torch.Tensor:
    """[B, H, W, 3] standard normals for images offset..offset+B-1 (counter = (y*W+x, image index, 0, 0))."""
    pix = torch.arange(H * W, device=device, dtype=torch.int64).view(1, -1)
    img = torch.arange(offset, offset + B, device=device, dtype=torch.int64).view(-1, 1)
    zero = torch.zeros(1, 1, device=device, dtype=torch.int64)
    bits = philox4x32_10_torch(pix, img, zero, zero, co.seed_key(seed))
    u = [((b >> 9).to(torch.float32) + 0.5) * (2.0 ** -23) for b in bits]
    r1, r2 = torch.sqrt(-2.0 * torch.log(u[0])), torch.sqrt(-2.0 * torch.log(u[2]))
    a1, a3 = 2.0 * math.pi * u[1], 2.0 * math.pi * u[3]
    z = torch.stack([r1 * torch.cos(a1), r1 * torch.sin(a1), r2 * torch.cos(a3)], -1)
    return z.view(B, H, W, 3)


def _noise(v, P, seed, offset, noise_rng="philox"):
    std = _col(P, "noise_std")
    on = std > 0
    if not bool(on.any()):
        return v
    B, H, W, _ = v.shape
    if noise_rng == "philox":
        z = normals_torch(B, H, W, seed, offset, v.device)
    else:  # benchmark-only variant: torch's own generator, NOT the kernel's noise stream
        gen = torch.Generator(device=v.device).manual_seed((int(seed) + offset) & 0x7FFFFFFFFFFFFFFF)
        z = torch.randn(v.shape, generator=gen, device=v.device, dtype=v.dtype)
    out = v + std.view(-1, 1, 1, 1) * torch.sqrt(co.NOISE_READ + (1.0 - co.NOISE_READ) * v) * z
    return _sel(on, out.clamp(0.0, 1.0), v)


def _tables(q: torch.Tensor) -> torch.Tensor:
    """IJG-scaled tables [B, 3, 8, 8] (luma, chroma, chroma) for integer qualities q (int64, 1..100)."""
    scale = torch.where(q < 50, 5000 // q, 200 - 2 * q).view(-1, 1, 1, 1)
    base = torch.as_tensor(np.stack([co.LUMA_Q, co.CHROMA_Q, co.CHROMA_Q]), device=q.device)
    return torch.clamp((base.unsqueeze(0) * scale + 50) // 100, 1, 255).to(torch.float32)


def _jpeg_round(v, P) -> torch.Tensor:
    q = _col(P, "jpeg_quality")
    on = q > 0
    rgb = v * 255.0
    if bool(on.any()):
        B, H, W, _ = v.shape
        qi = torch.floor(q + 0.5).clamp(1, 100).to(torch.int64)
        Q = _tables(qi).view(B, 3, 1, 1, 8, 8)
        ycc = _lin3(rgb, co.RGB2YCC.T.tolist())
        ycc = ycc - torch.tensor([128.0, 0.0, 0.0], device=v.device)
        ph, pw = -H % 8, -W % 8
        x = F.pad(ycc.permute(0, 3, 1, 2), (0, pw, 0, ph), mode="replicate")
        nby, nbx = x.shape[2] // 8, x.shape[3] // 8
        blk = x.view(B, 3, nby, 8, nbx, 8).permute(0, 1, 2, 4, 3, 5).reshape(B, 3, nby, nbx, 64)
        # 2-D DCT as one GEMM with kron(C, C): F[k*8+l] = sum_{m,n} C[k,m] C[l,n] X[m*8+n]; inverse = F @ kron
        K = torch.as_tensor(np.kron(co.dct_matrix(), co.dct_matrix()), dtype=torch.float32, device=v.device)
        f = blk @ K.T
        Qf = Q.reshape(B, 3, 1, 1, 64)
        f = torch.floor(f / Qf + 0.5) * Qf
        rec = (f @ K).view(B, 3, nby, nbx, 8, 8).permute(0, 1, 2, 4, 3, 5).reshape(B, 3, nby * 8, nbx * 8)[:, :, :H, :W]
        rec = rec.permute(0, 2, 3, 1) + torch.tensor([128.0, 0.0, 0.0], device=v.device)
        rgb = _sel(on, _lin3(rec, co.YCC2RGB.T.tolist()), rgb)
    return torch.round(rgb.clamp(0.0, 255.0)).to(torch.uint8)


def _chunk(x: torch.Tensor, P: torch.Tensor, seed: int, offset: int,
           noise_rng: str = "philox") -> tuple[torch.Tensor, torch.Tensor]:
    B, H, W, _ = x.shape
    g = _norm_grid(H, W, x.device)
    v = x.to(torch.float32) / 255.0
    v = _warp(v, P, g)
    v = _hed(v, P)
    v = _display(v, P)
    v = _moire(v, P, g)
    v = _glare(v, P, g)
    v = _defocus(v, P)
    v = _wb(v, P)
    v = _noise(v, P, seed, offset, noise_rng)
    return _jpeg_round(v, P), v


def camera_aug_torch(x: torch.Tensor, P: torch.Tensor, seed: int, return_float: bool = False,
                     noise_rng: str = "philox"):
    """x uint8 [B,H,W,3], P float32 [B,35] (packed params, same device) -> (uint8 [B,H,W,3], float32 pre-JPEG
    [B,H,W,3] in [0,1] or None). ``noise_rng="randn"`` (benchmark only) draws the noise with ``torch.randn``
    instead of the kernel's Philox stream, to time a plain-torch pipeline; its output differs from the kernel's."""
    if noise_rng not in ("philox", "randn"):
        raise ValueError(f"noise_rng must be 'philox' or 'randn', got {noise_rng!r}")
    B = x.shape[0]
    out = torch.empty_like(x)
    inter = torch.empty(x.shape, dtype=torch.float32, device=x.device) if return_float else None
    with _exact_fp32(), torch.no_grad():
        for s in range(0, B, CHUNK):
            o, v = _chunk(x[s:s + CHUNK], P[s:s + CHUNK], seed, s, noise_rng)
            out[s:s + CHUNK] = o
            if return_float:
                inter[s:s + CHUNK] = v
    return out, inter
