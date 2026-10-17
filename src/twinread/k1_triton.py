"""K1 Triton baseline: the same two-pass Pillow fixed-point resample + LUT normalization as the CUDA kernel,
from the same host plan (``twinread.kernels.make_plan``). One Triton launch per pass per reader (4 launches; the
CUDA kernel covers both readers per launch). Research prototype, not a diagnostic device."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _hpass(x_ptr, tmp_ptr, hb_ptr, hk_ptr, H, W, row0, R, C,
           KH: tl.constexpr, BLOCK: tl.constexpr, PB: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < R * C
    r = offs // C
    c = offs - r * C
    xmin = tl.load(hb_ptr + 2 * c, mask=m, other=0)
    xlen = tl.load(hb_ptr + 2 * c + 1, mask=m, other=0)
    src = x_ptr + ((b * H + row0 + r) * W + xmin) * 3
    a0 = tl.full([BLOCK], 1 << (PB - 1), tl.int32)
    a1 = a0
    a2 = a0
    for j in tl.static_range(KH):
        mj = m & (j < xlen)
        w = tl.load(hk_ptr + c * KH + j, mask=mj, other=0)
        a0 += tl.load(src + 3 * j, mask=mj, other=0).to(tl.int32) * w
        a1 += tl.load(src + 3 * j + 1, mask=mj, other=0).to(tl.int32) * w
        a2 += tl.load(src + 3 * j + 2, mask=mj, other=0).to(tl.int32) * w
    dst = tmp_ptr + (b * R * C + offs) * 3
    tl.store(dst, tl.minimum(tl.maximum(a0 >> PB, 0), 255).to(tl.uint8), mask=m)
    tl.store(dst + 1, tl.minimum(tl.maximum(a1 >> PB, 0), 255).to(tl.uint8), mask=m)
    tl.store(dst + 2, tl.minimum(tl.maximum(a2 >> PB, 0), 255).to(tl.uint8), mask=m)


@triton.jit
def _vpass(tmp_ptr, out_ptr, vb_ptr, vk_ptr, lut_ptr, R, C,
           KV: tl.constexpr, BLOCK: tl.constexpr, PB: tl.constexpr):
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n = C * C
    m = offs < n
    y = offs // C
    xx = offs - y * C
    ymin = tl.load(vb_ptr + 2 * y, mask=m, other=0)
    ylen = tl.load(vb_ptr + 2 * y + 1, mask=m, other=0)
    src = tmp_ptr + b * R * C * 3 + (ymin * C + xx) * 3
    a0 = tl.full([BLOCK], 1 << (PB - 1), tl.int32)
    a1 = a0
    a2 = a0
    for j in tl.static_range(KV):
        mj = m & (j < ylen)
        w = tl.load(vk_ptr + y * KV + j, mask=mj, other=0)
        q = src + j * C * 3
        a0 += tl.load(q, mask=mj, other=0).to(tl.int32) * w
        a1 += tl.load(q + 1, mask=mj, other=0).to(tl.int32) * w
        a2 += tl.load(q + 2, mask=mj, other=0).to(tl.int32) * w
    v0 = tl.minimum(tl.maximum(a0 >> PB, 0), 255)
    v1 = tl.minimum(tl.maximum(a1 >> PB, 0), 255)
    v2 = tl.minimum(tl.maximum(a2 >> PB, 0), 255)
    ob = out_ptr + b * 3 * n + offs
    tl.store(ob, tl.load(lut_ptr + v0, mask=m, other=0.0), mask=m)
    tl.store(ob + n, tl.load(lut_ptr + 256 + v1, mask=m, other=0.0), mask=m)
    tl.store(ob + 2 * n, tl.load(lut_ptr + 512 + v2, mask=m, other=0.0), mask=m)


def _one(x: torch.Tensor, p) -> torch.Tensor:
    from twinread.kernels import PRECISION_BITS
    B, H, W, _ = x.shape
    C, R = p.crop, p.rows
    tmp = torch.empty((B, R, C, 3), dtype=torch.uint8, device=x.device)
    out = torch.empty((B, 3, C, C), dtype=torch.float16, device=x.device)
    if B == 0:
        return out
    for s in range(0, B, 65535):  # grid dim 1 limit
        xs, ts, os_ = x[s:s + 65535], tmp[s:s + 65535], out[s:s + 65535]
        nb = xs.shape[0]
        _hpass[(triton.cdiv(R * C, BLOCK), nb)](xs, ts, p.hb, p.hk, H, W, p.row0, R, C,
                                                KH=p.kh, BLOCK=BLOCK, PB=PRECISION_BITS)
        _vpass[(triton.cdiv(C * C, BLOCK), nb)](ts, os_, p.vb, p.vk, p.lut, R, C,
                                                KV=p.kv, BLOCK=BLOCK, PB=PRECISION_BITS)
    return out


def dual_preprocess_triton(x: torch.Tensor, pa, pb) -> tuple[torch.Tensor, torch.Tensor]:
    """x: contiguous uint8 [B,H,W,3] on CUDA; pa/pb: ``kernels.Plan`` on x.device."""
    assert x.is_cuda and x.is_contiguous() and x.dtype == torch.uint8
    with torch.cuda.device(x.device):
        return _one(x, pa), _one(x, pb)
