"""K2 numpy references: a SIMULATED USB camera filming a tissue tile shown on a screen (camera-of-screen).

One function per op, all with the signature ``op(img, p) -> img``: ``img`` is one float64 HWC RGB image in [0, 1],
``p`` a dict of scalar (or short-list) params for that image, keyed like the fields of
``twinread.kernels.CameraAugParams``. ``ref_camera_aug(img_uint8, p)`` composes them in the fixed K2 order

    warp -> HED jitter -> display gamma + colour profile -> moire -> glare -> defocus -> white balance
         -> sensor noise -> JPEG-like 8x8 DCT quantization

and rounds to uint8. Each op returns its input unchanged (the same array) when its params are the identity, which
is how severity 0 is an exact identity. Params are read as float32 (``_f``), exactly what the CUDA kernel sees, and
the math is float64. This module is pure numpy (no torch), so the screen-capture rig can use it too.

Op definitions (the CUDA kernel and the torch fallback implement the same formulas):

* warp: homography ``Hm = I + delta`` (``warp`` = the 8 entries of ``delta`` in row-major order, ``Hm[2,2] = 1``)
  in normalized pixel-centre coords ``xn = (2x+1)/W - 1``; output pixel (x, y) samples the input at
  ``Hm (xn, yn, 1)`` (projective divide, denominator floored at 1e-6) with bilinear interpolation and
  clamp-to-edge borders (``grid_sample(align_corners=False, padding_mode="border")`` conventions).
* HED jitter: Ruifrok-Johnston colour deconvolution exactly as skimage's ``rgb2hed``/``hed2rgb``; per stain channel
  ``s' = alpha * s + beta``. Ported from StainForge (``stainforge-cuda-stain-kernels/src/stainforge/reference.py``
  and ``csrc/stainforge_kernels.cu`` ``hed_jitter_kernel``), applied to float input instead of a uint8 LUT and
  returned unrounded.
* display: ``v ** gamma`` (gamma clamped to [0.05, 20]), then the 3x3 colour-profile matrix ``profile``
  (row-major, rows sum to 1 in the sampler so greys stay grey), clipped to [0, 1].
* moire: ``v_c * (1 + amp * cos(2 pi t + c pi / 3))``, ``t = ((x+.5) cos(angle) + (y+.5) sin(angle)) / period``
  (period floored at 2 px): the beat between the screen's pixel grid and the sensor grid as one sinusoid, with the
  RGB sub-pixel stripes (1/3 px apart) giving each channel a phase offset, so the bands are partly coloured.
* glare: veiling reflection ``v + a g (1 - v)``, ``a`` clamped to [0, 1], ``g = ((d + 1) / 2) ** 2`` with
  ``d = (xn cos(angle) + yn sin(angle)) / sqrt(2)``: brightest towards ``angle``, contrast drops where it is bright.
* defocus: separable Gaussian, sigma clamped to [1e-3, 4] px, radius ``ceil(3 sigma)`` (float32 product, as in the
  kernel), taps normalized to sum 1, replicated borders, horizontal pass then vertical pass.
* white balance: per-channel gains, clipped to [0, 1].
* sensor noise: ``v + std * sqrt(0.2 + 0.8 v) * z`` (read + shot noise), clipped. ``z`` is standard normal from
  Philox4x32-10 (Random123 / ``curand_Philox4x32_10``) with key = (seed lo32, seed hi32) and counter =
  (y*W + x, image index, 0, 0); uniforms ``((bits >> 9) + 0.5) * 2**-23`` (exact in float32), Box-Muller,
  normals 0, 1, 2 of the 4 go to R, G, B.
* JPEG-like: RGB -> YCbCr (JFIF, full range), level shift, orthonormal 8x8 DCT-II per channel, quantize with the
  standard Annex K luma/chroma tables scaled by the IJG quality formula (quality rounded to an integer in 1..100;
  0 = JPEG off), ``floor(F/Q + 0.5) * Q`` (libjpeg's float-path rounding), inverse DCT, -> RGB, clip to 0..255.
  **Chroma is not subsampled** (4:4:4). Partial blocks at the right/bottom edge are padded by edge replication
  inside the block; only in-image pixels are kept.

Research prototype, not a diagnostic device.
"""
from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------------------------------- params
# (name, width): the packed per-image parameter row consumed by the CUDA kernel, in this order.
FIELDS: tuple[tuple[str, int], ...] = (
    ("gamma", 1), ("profile", 9), ("wb_gain", 3), ("moire_amp", 1), ("moire_period", 1), ("moire_angle", 1),
    ("glare_amp", 1), ("glare_angle", 1), ("warp", 8), ("blur_sigma", 1), ("noise_std", 1), ("jpeg_quality", 1),
    ("hed_alpha", 3), ("hed_beta", 3),
)
NPARAM = sum(w for _, w in FIELDS)  # 35
MAX_SIGMA = 4.0  # defocus sigma clamp (radius <= 12 keeps the kernel's shared-memory tile under 48 KB)
MIN_SIGMA = 1e-3
NOISE_READ = 0.2  # noise variance fraction at black (read noise); the rest scales with the signal (shot noise)
N_UNIFORM = 26  # uniform draws per image in the sampler
LUMA_W = np.array([0.299, 0.587, 0.114])


def identity_params() -> dict:
    """Params under which every op is skipped (severity 0)."""
    return {"gamma": 1.0, "profile": np.eye(3).ravel().tolist(), "wb_gain": [1.0, 1.0, 1.0], "moire_amp": 0.0,
            "moire_period": 8.0, "moire_angle": 0.0, "glare_amp": 0.0, "glare_angle": 0.0, "warp": [0.0] * 8,
            "blur_sigma": 0.0, "noise_std": 0.0, "jpeg_quality": 0.0, "hed_alpha": [1.0, 1.0, 1.0],
            "hed_beta": [0.0, 0.0, 0.0]}


def params_from_uniforms(u: np.ndarray, severity: float) -> dict[str, np.ndarray]:
    """Map uniforms ``u`` [B, N_UNIFORM] in [0, 1) to per-image params (float32 arrays [B] or [B, k]).

    Ranges at severity 1 (all scale linearly with severity; 0 gives the identity exactly):
    gamma exp(+-0.22) = 0.80..1.25; profile saturation 1 +- 0.25 and +-0.04 red/blue <- green cross-talk;
    moire amplitude 0..0.08, period 3..28 px, any angle; glare 0..0.25, any direction; warp rotation +-3 deg,
    scale +-4 %, shift +-0.03, perspective +-0.04 (normalized coords); defocus sigma 0.3..1.5 px; noise std
    0.004..0.020 (1..5 levels at white); JPEG quality 95..10 (uniform, integer); HED alpha 1 +- 0.08,
    beta +- 0.008; white balance colour temperature exp(+-0.10) on R/B (opposite), tint exp(+-0.04) on G."""
    if severity < 0 or not math.isfinite(severity):
        raise ValueError(f"severity must be finite and >= 0, got {severity}")
    u = np.asarray(u, dtype=np.float64)
    B = u.shape[0]
    assert u.shape == (B, N_UNIFORM), u.shape
    s = float(severity)
    if s == 0.0:
        ident = identity_params()
        return {k: np.tile(np.asarray(ident[k], np.float32).reshape(1, w), (B, 1)).reshape((B,) if w == 1 else (B, w))
                for k, w in FIELDS}
    v = 2.0 * u - 1.0
    out: dict[str, np.ndarray] = {}
    out["gamma"] = np.exp(0.22 * s * v[:, 0])
    sat = 1.0 + 0.25 * s * v[:, 1]
    prof = sat[:, None, None] * np.eye(3)[None] + (1.0 - sat)[:, None, None] * LUMA_W[None, None, :]
    e1, e2 = 0.04 * s * v[:, 2], 0.04 * s * v[:, 3]
    prof[:, 0, 1] += e1
    prof[:, 0, 0] -= e1
    prof[:, 2, 1] += e2
    prof[:, 2, 2] -= e2
    out["profile"] = prof.reshape(B, 9)
    out["moire_amp"] = np.minimum(0.08 * s * u[:, 4], 0.5)
    out["moire_period"] = 3.0 * (28.0 / 3.0) ** u[:, 5]
    out["moire_angle"] = math.pi * u[:, 6]
    out["glare_amp"] = np.minimum(0.25 * s * u[:, 7], 1.0)
    out["glare_angle"] = 2.0 * math.pi * u[:, 8]
    th = math.radians(3.0) * s * v[:, 9]
    k = 1.0 + 0.04 * s * v[:, 10]
    out["warp"] = np.stack([k * np.cos(th) - 1.0, -k * np.sin(th), 0.03 * s * v[:, 11],
                            k * np.sin(th), k * np.cos(th) - 1.0, 0.03 * s * v[:, 12],
                            0.04 * s * v[:, 13], 0.04 * s * v[:, 14]], 1)
    out["blur_sigma"] = np.minimum(s * (0.3 + 1.2 * u[:, 15]), MAX_SIGMA)
    out["noise_std"] = s * (0.004 + 0.016 * u[:, 16])
    out["jpeg_quality"] = np.clip(np.round(95.0 - 85.0 * min(s, 1.0) * u[:, 17]), 10, 95)
    out["hed_alpha"] = 1.0 + 0.08 * s * v[:, 18:21]
    out["hed_beta"] = 0.008 * s * v[:, 21:24]
    t, tint = 0.10 * s * v[:, 24], 0.04 * s * v[:, 25]
    out["wb_gain"] = np.exp(np.stack([t, tint, -t], 1))
    return {k: out[k].astype(np.float32) for k, _ in FIELDS}


def sample_uniforms(B: int, seed: int) -> np.ndarray:
    """[B, N_UNIFORM] uniforms; row i depends only on (seed, i), not on B (PCG64 fills row-major)."""
    return np.random.default_rng(seed).random((B, N_UNIFORM))


def sample_params(severity: float, seed: int) -> dict:
    """Scalar params for ONE image, from the same distribution (and the same draws) as image 0 of
    ``twinread.kernels.sample_camera_params(B, severity, seed)``, in the format ``ref_camera_aug`` consumes.
    ``seed``/``index`` (noise stream) are set to (seed, 0)."""
    arr = params_from_uniforms(sample_uniforms(1, seed), severity)
    p: dict = {k: (float(arr[k][0]) if w == 1 else [float(x) for x in arr[k][0]]) for k, w in FIELDS}
    p["seed"], p["index"] = int(seed), 0
    return p


def _f(p: dict, key: str) -> float:
    return float(np.float32(p[key]))


def _v(p: dict, key: str, n: int) -> np.ndarray:
    a = np.asarray(p[key], dtype=np.float32).astype(np.float64).ravel()
    assert a.shape == (n,), (key, a.shape)
    return a


def _grid(H: int, W: int):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    return xs, ys


# ------------------------------------------------------------------------------------------- ops
def warp(img: np.ndarray, p: dict) -> np.ndarray:
    d = _v(p, "warp", 8)
    if not d.any():
        return img
    H, W = img.shape[:2]
    xs, ys = _grid(H, W)
    xn, yn = (2.0 * xs + 1.0) / W - 1.0, (2.0 * ys + 1.0) / H - 1.0
    u = (1.0 + d[0]) * xn + d[1] * yn + d[2]
    v = d[3] * xn + (1.0 + d[4]) * yn + d[5]
    w = np.maximum(d[6] * xn + d[7] * yn + 1.0, 1e-6)
    sx = np.clip(((u / w + 1.0) * W - 1.0) * 0.5, 0.0, W - 1.0)
    sy = np.clip(((v / w + 1.0) * H - 1.0) * 0.5, 0.0, H - 1.0)
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    top = img[y0, x0] * (1.0 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1.0 - fx) + img[y1, x1] * fx
    return top * (1.0 - fy) + bot * fy


# Ruifrok & Johnston stain vectors as in skimage.color (rgb2hed / hed2rgb); same constants as StainForge.
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
LOG_ADJ = math.log(1e-6)


def hed_jitter(img: np.ndarray, p: dict) -> np.ndarray:
    """StainForge reference.hed_jitter on float input (port; attribution in the module docstring)."""
    a, b = _v(p, "hed_alpha", 3), _v(p, "hed_beta", 3)
    if (a == 1.0).all() and (b == 0.0).all():
        return img
    rgb = np.maximum(img, 1e-6)
    s = np.maximum((np.log(rgb) / LOG_ADJ) @ HED_FROM_RGB, 0.0)
    s = s * a + b
    return np.clip(np.exp((s * LOG_ADJ) @ RGB_FROM_HED), 0.0, 1.0)


def display(img: np.ndarray, p: dict) -> np.ndarray:
    g = _f(p, "gamma")
    m = _v(p, "profile", 9).reshape(3, 3)
    out = img
    if g != 1.0:
        out = np.power(np.clip(out, 0.0, 1.0), min(max(g, 0.05), 20.0))
    if not np.array_equal(m, np.eye(3)):
        out = np.clip(out @ m.T, 0.0, 1.0)
    return out


def moire(img: np.ndarray, p: dict) -> np.ndarray:
    a = _f(p, "moire_amp")
    if a == 0.0:
        return img
    per, th = max(_f(p, "moire_period"), 2.0), _f(p, "moire_angle")
    H, W = img.shape[:2]
    xs, ys = _grid(H, W)
    t = ((xs + 0.5) * math.cos(th) + (ys + 0.5) * math.sin(th)) / per
    m = np.cos(2.0 * math.pi * t[..., None] + np.arange(3) * (math.pi / 3.0))
    return np.clip(img * (1.0 + a * m), 0.0, 1.0)


def glare(img: np.ndarray, p: dict) -> np.ndarray:
    a = _f(p, "glare_amp")
    if a == 0.0:
        return img
    a, phi = min(max(a, 0.0), 1.0), _f(p, "glare_angle")
    H, W = img.shape[:2]
    xs, ys = _grid(H, W)
    xn, yn = (2.0 * xs + 1.0) / W - 1.0, (2.0 * ys + 1.0) / H - 1.0
    d = (xn * math.cos(phi) + yn * math.sin(phi)) / math.sqrt(2.0)
    g = ((d + 1.0) * 0.5) ** 2
    return np.clip(img + (a * g)[..., None] * (1.0 - img), 0.0, 1.0)


def blur_radius(sigma: float) -> int:
    """Radius used by the kernel: ceil(3 * sigma) with the float32 product and the clamp to [MIN, MAX]."""
    if not sigma > 0.0:
        return 0
    s = np.float32(min(max(np.float32(sigma), np.float32(MIN_SIGMA)), np.float32(MAX_SIGMA)))
    return int(np.ceil(np.float32(3.0) * s))


def gaussian_taps(sigma: float) -> np.ndarray:
    r = blur_radius(sigma)
    s = float(np.float32(min(max(np.float32(sigma), np.float32(MIN_SIGMA)), np.float32(MAX_SIGMA))))
    k = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-(k * k) / (2.0 * s * s))
    return w / w.sum()


def defocus(img: np.ndarray, p: dict) -> np.ndarray:
    sigma = _f(p, "blur_sigma")
    if not sigma > 0.0:
        return img
    w, r = gaussian_taps(sigma), blur_radius(sigma)
    H, W = img.shape[:2]
    pad = np.pad(img, ((0, 0), (r, r), (0, 0)), mode="edge")
    h = sum(w[j] * pad[:, j:j + W] for j in range(2 * r + 1))
    pad = np.pad(h, ((r, r), (0, 0), (0, 0)), mode="edge")
    return sum(w[j] * pad[j:j + H] for j in range(2 * r + 1))


def white_balance(img: np.ndarray, p: dict) -> np.ndarray:
    g = _v(p, "wb_gain", 3)
    if (g == 1.0).all():
        return img
    return np.clip(img * g, 0.0, 1.0)


# --- Philox4x32-10 (Salmon et al. 2011, Random123; identical to curand_Philox4x32_10)
_M0, _M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_MASK = np.uint64(0xFFFFFFFF)


def philox4x32_10(ctr: np.ndarray, key: tuple[int, int]) -> np.ndarray:
    """ctr uint32 [N, 4], key (k0, k1) -> uint32 [N, 4]."""
    c = np.asarray(ctr, dtype=np.uint32).astype(np.uint64)
    c0, c1, c2, c3 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    k0, k1 = int(key[0]) & 0xFFFFFFFF, int(key[1]) & 0xFFFFFFFF
    for _ in range(10):
        p0, p1 = _M0 * c0, _M1 * c2  # < 2**64: exact in uint64
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ np.uint64(k0), p1 & _MASK,
                          (p0 >> np.uint64(32)) ^ c3 ^ np.uint64(k1), p0 & _MASK)
        k0, k1 = (k0 + _W0) & 0xFFFFFFFF, (k1 + _W1) & 0xFFFFFFFF
    return np.stack([c0, c1, c2, c3], 1).astype(np.uint32)


def seed_key(seed: int) -> tuple[int, int]:
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    return s & 0xFFFFFFFF, s >> 32


def bits_to_uniform(bits: np.ndarray) -> np.ndarray:
    """uint32 -> (0, 1): ((bits >> 9) + 0.5) * 2**-23, exactly representable in float32."""
    return ((np.asarray(bits, np.uint32) >> np.uint32(9)).astype(np.float64) + 0.5) * 2.0 ** -23


def normals(H: int, W: int, seed: int, index: int) -> np.ndarray:
    """Standard normals [H, W, 3] for image ``index`` of a batch, keyed by ``seed``."""
    n = H * W
    ctr = np.zeros((n, 4), dtype=np.uint32)
    ctr[:, 0] = np.arange(n, dtype=np.uint64).astype(np.uint32)
    ctr[:, 1] = np.uint32(index)
    u = bits_to_uniform(philox4x32_10(ctr, seed_key(seed)))
    r1, r2 = np.sqrt(-2.0 * np.log(u[:, 0])), np.sqrt(-2.0 * np.log(u[:, 2]))
    z = np.stack([r1 * np.cos(2.0 * math.pi * u[:, 1]), r1 * np.sin(2.0 * math.pi * u[:, 1]),
                  r2 * np.cos(2.0 * math.pi * u[:, 3])], 1)
    return z.reshape(H, W, 3)


def sensor_noise(img: np.ndarray, p: dict) -> np.ndarray:
    std = _f(p, "noise_std")
    if not std > 0.0:
        return img
    H, W = img.shape[:2]
    z = normals(H, W, int(p.get("seed", 0)), int(p.get("index", 0)))
    return np.clip(img + std * np.sqrt(NOISE_READ + (1.0 - NOISE_READ) * img) * z, 0.0, 1.0)


# --- JPEG-like
LUMA_Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.int64)
CHROMA_Q = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                     [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99]]
                    + [[99] * 8] * 4, dtype=np.int64)  # ITU-T T.81 Annex K, natural (row-major) order
RGB2YCC = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
YCC2RGB = np.array([[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]])


def quant_table(quality: int, chroma: bool) -> np.ndarray:
    """IJG jpeg_quality_scaling + jpeg_add_quant_table(force_baseline=TRUE)."""
    q = int(min(max(int(quality), 1), 100))
    scale = 5000 // q if q < 50 else 200 - 2 * q
    return np.clip(((CHROMA_Q if chroma else LUMA_Q) * scale + 50) // 100, 1, 255)


def jpeg_int_quality(q: float) -> int:
    """Integer quality the kernel uses: floor(q + 0.5) in float32, clamped to 1..100 (call only when q > 0)."""
    return int(min(max(np.floor(np.float32(q) + np.float32(0.5)), 1), 100))


def dct_matrix() -> np.ndarray:
    k, n = np.arange(8)[:, None], np.arange(8)[None, :]
    c = np.cos(math.pi * (2 * n + 1) * k / 16.0) * math.sqrt(2.0 / 8.0)
    c[0] = math.sqrt(1.0 / 8.0)
    return c


_C = dct_matrix()


def dct2(x: np.ndarray) -> np.ndarray:
    """Orthonormal 2-D DCT-II over the last two axes (8x8)."""
    return _C @ x @ _C.T


def idct2(f: np.ndarray) -> np.ndarray:
    return _C.T @ f @ _C


def jpeg(img: np.ndarray, p: dict) -> np.ndarray:
    q = _f(p, "jpeg_quality")
    if not q > 0.0:
        return img
    qi = jpeg_int_quality(q)
    H, W = img.shape[:2]
    ycc = (img * 255.0) @ RGB2YCC.T
    ycc[..., 0] -= 128.0  # level shift (Cb, Cr: +128 offset and -128 shift cancel)
    ph, pw = -H % 8, -W % 8
    ycc = np.pad(ycc, ((0, ph), (0, pw), (0, 0)), mode="edge")
    nby, nbx = ycc.shape[0] // 8, ycc.shape[1] // 8
    blk = ycc.reshape(nby, 8, nbx, 8, 3).transpose(4, 0, 2, 1, 3)  # [3, nby, nbx, 8, 8]
    tables = np.stack([quant_table(qi, False), quant_table(qi, True), quant_table(qi, True)]).astype(np.float64)
    tq = tables[:, None, None]
    f = np.floor(dct2(blk) / tq + 0.5) * tq
    rec = idct2(f).transpose(1, 3, 2, 4, 0).reshape(nby * 8, nbx * 8, 3)[:H, :W]
    rec[..., 0] += 128.0
    return np.clip(rec @ YCC2RGB.T, 0.0, 255.0) / 255.0


OPS = (("warp", warp), ("hed", hed_jitter), ("display", display), ("moire", moire), ("glare", glare),
       ("defocus", defocus), ("white_balance", white_balance), ("noise", sensor_noise), ("jpeg", jpeg))

# A float32 implementation (the kernel) and this float64 reference can round a DCT coefficient that lies within
# float error of a quantization boundary to different integers ("flip"); the decoded block then differs by up to
# Q * |basis| levels. The two helpers below let tests separate such blocks from genuine errors.
DCT_F32_ERR = 2e-3  # bound on the kernel's own float32 error in a coefficient (levels), YCbCr + 2 passes of 8 FMAs


def ref_pre_jpeg(img: np.ndarray, p: dict) -> np.ndarray:
    """Every op except JPEG: uint8 HWC -> float64 HWC in [0, 1] (the JPEG stage's input)."""
    v = np.asarray(img).astype(np.float64) / 255.0
    for _, op in OPS[:-1]:
        v = op(v, p)
    return v


def jpeg_block_margins(v: np.ndarray, p: dict) -> np.ndarray:
    """[ceil(H/8), ceil(W/8)]: per 8x8 block, the smallest distance (in coefficient units, levels) of any channel's
    F/Q + 0.5 to an integer, times Q, i.e. how far the reference's rounding decisions are from flipping. inf when
    JPEG is off. A coefficient perturbed by |dF| < margin rounds the same way."""
    H, W = v.shape[:2]
    nby, nbx = -(-H // 8), -(-W // 8)
    q = _f(p, "jpeg_quality")
    if not q > 0.0:
        return np.full((nby, nbx), np.inf)
    qi = jpeg_int_quality(q)
    ycc = (v * 255.0) @ RGB2YCC.T
    ycc[..., 0] -= 128.0
    ycc = np.pad(ycc, ((0, -H % 8), (0, -W % 8), (0, 0)), mode="edge")
    blk = ycc.reshape(nby, 8, nbx, 8, 3).transpose(4, 0, 2, 1, 3)
    tq = np.stack([quant_table(qi, False), quant_table(qi, True), quant_table(qi, True)]).astype(np.float64)
    tq = tq[:, None, None]
    r = dct2(blk) / tq + 0.5
    return (np.abs(r - np.round(r)) * tq).min(axis=(0, 3, 4))


def _block_max(a: np.ndarray, mode: str) -> np.ndarray:
    H, W = a.shape[:2]
    a = np.pad(a, ((0, -H % 8), (0, -W % 8), (0, 0)), mode=mode)
    return a.reshape(a.shape[0] // 8, 8, a.shape[1] // 8, 8, 3).max(axis=(1, 3, 4))


def explain_differences(out: np.ndarray, ref: np.ndarray, pre_impl: np.ndarray, pre_ref: np.ndarray, p: dict) -> dict:
    """Split |out - ref| (uint8 HWC) by 8x8 block into blocks where the implementation's JPEG rounding decisions
    provably equal the reference's, and "flip-possible" blocks.

    A coefficient changes by |dF| <= 8 * max|dRGB| (levels; the row sums of |kron(C, C)| are <= 8 and every YCbCr
    row has absolute sum <= 1) plus the implementation's own DCT error ``DCT_F32_ERR``. ``pre_impl``/``pre_ref``
    are the JPEG inputs in [0, 1] (the kernel's float intermediate and ``ref_pre_jpeg``). In a block whose
    ``jpeg_block_margins`` exceeds that bound both sides quantize identically, so the outputs may differ only by the
    final rounding: at most 1 level."""
    e = np.abs(out.astype(np.int32) - ref.astype(np.int32))
    eb = _block_max(e, "constant")
    db = _block_max(np.abs(pre_impl.astype(np.float64) - pre_ref) * 255.0, "edge")
    well = jpeg_block_margins(pre_ref, p) >= 8.0 * db + DCT_F32_ERR
    return {"max_err_provably_same_rounding": int(eb[well].max()) if well.any() else 0,
            "max_err_flip_possible": int(eb[~well].max()) if (~well).any() else 0,
            "n_blocks": int(eb.size), "n_flip_possible": int((~well).sum()),
            "max_pre_jpeg_diff_levels": float(db.max())}


def ref_camera_aug_float(img: np.ndarray, p: dict) -> np.ndarray:
    """uint8 HWC -> float64 HWC in [0, 1] before the final rounding."""
    img = np.asarray(img)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3, got {img.dtype} {img.shape}")
    v = img.astype(np.float64) / 255.0
    for _, op in OPS:
        v = op(v, p)
    return v


def ref_camera_aug(img: np.ndarray, p: dict) -> np.ndarray:
    """The full K2 chain on one uint8 HWC image -> uint8 (round half to even, as the kernel's rintf)."""
    return np.rint(np.clip(ref_camera_aug_float(img, p), 0.0, 1.0) * 255.0).astype(np.uint8)
