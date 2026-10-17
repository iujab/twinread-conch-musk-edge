// K1: fused dual-reader preprocessing (TwinRead). Research prototype, not a diagnostic device.
//
// Input: raw uint8 frames [B,H,W,3] on the GPU. Output: BOTH readers' normalized fp16 NCHW tensors, bit-faithful to
// each reader's official torchvision-on-PIL transform (Resize shorter side, bicubic a=-0.5 with antialias support
// scaling, CenterCrop, ToTensor, Normalize). The resampling reproduces Pillow's ImagingResample exactly:
//
//   pass 1, horizontal: taps into a uint8 intermediate, int32 fixed point (PRECISION_BITS = 22, accumulator starts at
//           1 << 21, >> 22, clip 0..255), only for the crop's columns and the input rows the crop's vertical windows
//           touch. Default kernel k1_hpass_dual: one block per (input row, image) stages the row once in shared memory
//           and emits BOTH readers' intermediate rows, so the raw frame is read once for the two readers.
//   pass 2, vertical: taps on the intermediate, same fixed point, then ToTensor + Normalize as a 256-entry fp16 LUT
//           per channel (computed on the host with torchvision's float32 ops), written NCHW. Default kernel
//           k1_vpass_vec4 (4 pixels per thread, 32-bit loads, 64-bit stores); blockIdx.z in {0,1} selects the reader.
//
// Two launches in total, each covering both readers. Scalar fallbacks k1_hpass (frames whose staged row would exceed
// 48 KB of shared memory, i.e. wider than ~16k px) and k1_vpass (crop % 4 != 0) select the reader with blockIdx.z.
// Coefficients (per output column/row: window start, length, fixed-point taps) are precomputed on the host
// (twinread/kernels.py:make_plan), which asserts that every window lies inside the frame; the launcher re-checks the
// row window and column span. Exact integer math: build without --use_fast_math.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

namespace {

constexpr int PRECISION_BITS = 32 - 8 - 2;
constexpr int ROUND_HALF = 1 << (PRECISION_BITS - 1);
constexpr int THREADS = 256;
constexpr int MAX_GRID_Y = 65535;

struct Spec {
  const int32_t* hb;  // [crop, 2] (xmin, xlen) for the crop's output columns
  const int32_t* hk;  // [crop, kh] fixed-point taps
  const int32_t* vb;  // [crop, 2] (ymin - row0, ylen) for the crop's output rows
  const int32_t* vk;  // [crop, kv]
  const __half* lut;  // [3, 256] normalized value per uint8 level
  uint8_t* tmp;       // [B, rows, crop, 3] horizontal-pass intermediate
  __half* out;        // [B, 3, crop, crop]
  int kh, kv, row0, rows, crop;
  int hx0, hx1;       // input columns [hx0, hx1) the crop's horizontal windows touch (windows are monotonic)
};

struct Dual {
  Spec s[2];
};

__device__ __forceinline__ int clip8(int v) {
  v >>= PRECISION_BITS;  // arithmetic shift == floor, as Pillow's clip8
  return v < 0 ? 0 : (v > 255 ? 255 : v);
}

__global__ void __launch_bounds__(THREADS)
k1_hpass(const uint8_t* __restrict__ x, int B, int H, int W, const __grid_constant__ Dual d) {
  const Spec& s = d.s[blockIdx.z];
  const int npix = s.rows * s.crop;
  const int64_t frame = (int64_t)H * W * 3;
  for (int b = blockIdx.y; b < B; b += gridDim.y) {
    const uint8_t* __restrict__ xb = x + (int64_t)b * frame;
    uint8_t* __restrict__ tb = s.tmp + (int64_t)b * npix * 3;
    for (int p = blockIdx.x * THREADS + threadIdx.x; p < npix; p += gridDim.x * THREADS) {
      const int r = p / s.crop, c = p - r * s.crop;
      const int xmin = __ldg(s.hb + 2 * c), xlen = __ldg(s.hb + 2 * c + 1);
      const int32_t* __restrict__ k = s.hk + c * s.kh;
      const uint8_t* __restrict__ src = xb + ((int64_t)(s.row0 + r) * W + xmin) * 3;
      int a0 = ROUND_HALF, a1 = ROUND_HALF, a2 = ROUND_HALF;
      for (int j = 0; j < xlen; ++j) {
        const int w = __ldg(k + j);
        a0 += (int)__ldg(src + 3 * j) * w;
        a1 += (int)__ldg(src + 3 * j + 1) * w;
        a2 += (int)__ldg(src + 3 * j + 2) * w;
      }
      uint8_t* dst = tb + (int64_t)p * 3;
      dst[0] = (uint8_t)clip8(a0);
      dst[1] = (uint8_t)clip8(a1);
      dst[2] = (uint8_t)clip8(a2);
    }
  }
}

__global__ void __launch_bounds__(THREADS)
k1_vpass(int B, const __grid_constant__ Dual d) {
  __shared__ __half lut[3 * 256];
  const Spec& s = d.s[blockIdx.z];
  for (int i = threadIdx.x; i < 3 * 256; i += THREADS) lut[i] = s.lut[i];
  __syncthreads();
  const int npix = s.crop * s.crop;
  const int rstride = s.crop * 3;
  for (int b = blockIdx.y; b < B; b += gridDim.y) {
    const uint8_t* __restrict__ tb = s.tmp + (int64_t)b * s.rows * rstride;
    __half* __restrict__ ob = s.out + (int64_t)b * 3 * npix;
    for (int p = blockIdx.x * THREADS + threadIdx.x; p < npix; p += gridDim.x * THREADS) {
      const int y = p / s.crop, xx = p - y * s.crop;
      const int ymin = __ldg(s.vb + 2 * y), ylen = __ldg(s.vb + 2 * y + 1);
      const int32_t* __restrict__ k = s.vk + y * s.kv;
      const uint8_t* __restrict__ src = tb + (int64_t)ymin * rstride + xx * 3;
      int a0 = ROUND_HALF, a1 = ROUND_HALF, a2 = ROUND_HALF;
      for (int j = 0; j < ylen; ++j) {
        const int w = __ldg(k + j);
        const uint8_t* q = src + (int64_t)j * rstride;
        a0 += (int)q[0] * w;
        a1 += (int)q[1] * w;
        a2 += (int)q[2] * w;
      }
      ob[p] = lut[clip8(a0)];
      ob[npix + p] = lut[256 + clip8(a1)];
      ob[2 * npix + p] = lut[512 + clip8(a2)];
    }
  }
}

// ---- fast paths ------------------------------------------------------------------------------------------------
// hpass, one block per (input row, image), shared by BOTH readers: the union of the two readers' column spans of the
// row is staged once in shared memory with aligned 32-bit loads (bytes at the unaligned ends, so every read stays
// inside the frame); threads then compute reader A's and reader B's intermediate rows for that input row (when the
// row lies in that reader's vertical window) and write them back with 32-bit stores. The raw frame is read once.
constexpr int HT = 128;

__device__ __forceinline__ void hrow(const Spec& s, const uint8_t* seg, int x0, int c, uint8_t* orow) {
  const int xoff = (__ldg(s.hb + 2 * c) - x0) * 3, xlen = __ldg(s.hb + 2 * c + 1);
  const int32_t* __restrict__ k = s.hk + c * s.kh;
  const uint8_t* q = seg + xoff;
  int a0 = ROUND_HALF, a1 = ROUND_HALF, a2 = ROUND_HALF;
#pragma unroll 4
  for (int j = 0; j < xlen; ++j) {
    const int w = __ldg(k + j);
    a0 += (int)q[3 * j] * w;
    a1 += (int)q[3 * j + 1] * w;
    a2 += (int)q[3 * j + 2] * w;
  }
  orow[3 * c] = (uint8_t)clip8(a0);
  orow[3 * c + 1] = (uint8_t)clip8(a1);
  orow[3 * c + 2] = (uint8_t)clip8(a2);
}

__device__ __forceinline__ void store_row(const uint8_t* orow, uint8_t* dst, int nbytes) {
  if ((nbytes & 3) == 0) {  // tmp rows are then 4-aligned (allocation base is 256-aligned)
    for (int k = threadIdx.x; k < (nbytes >> 2); k += HT)
      reinterpret_cast<uint32_t*>(dst)[k] = reinterpret_cast<const uint32_t*>(orow)[k];
  } else {
    for (int k = threadIdx.x; k < nbytes; k += HT) dst[k] = orow[k];
  }
}

__global__ void __launch_bounds__(HT)
k1_hpass_dual(const uint8_t* __restrict__ x, int B, int H, int W, int r0, int x0, int x1,
              const __grid_constant__ Dual d) {
  extern __shared__ __align__(16) uint8_t sm[];
  const Spec& A = d.s[0];
  const Spec& Bs = d.s[1];
  const int row = r0 + blockIdx.x;                 // absolute input row
  const int ra = row - A.row0, rb = row - Bs.row0;  // intermediate rows (valid if inside [0, rows))
  const bool doA = ra >= 0 && ra < A.rows, doB = rb >= 0 && rb < Bs.rows;
  if (!doA && !doB) return;  // uniform per block, before any barrier
  const int nbytes = (x1 - x0) * 3;
  uint8_t* oa = sm + ((3 + nbytes + 15) & ~15);
  uint8_t* ob = oa + ((A.crop * 3 + 15) & ~15);
  const int na = doA ? A.crop : 0, nb = doB ? Bs.crop : 0;
  for (int b = blockIdx.y; b < B; b += gridDim.y) {
    const uint8_t* src = x + (((int64_t)b * H + row) * W + x0) * 3;
    const int pad = (int)((uintptr_t)src & 3);
    const uint8_t* al = src - pad;  // 4-aligned; bytes before `src` are never read
    const int nw = (pad + nbytes + 3) >> 2;
    for (int k = threadIdx.x; k < nw; k += HT) {
      const int lo = 4 * k, hi = lo + 4;
      if (lo >= pad && hi <= pad + nbytes) {
        reinterpret_cast<uint32_t*>(sm)[k] = __ldg(reinterpret_cast<const uint32_t*>(al) + k);
      } else {
        for (int i = max(lo, pad); i < min(hi, pad + nbytes); ++i) sm[i] = __ldg(al + i);
      }
    }
    __syncthreads();
    const uint8_t* seg = sm + pad;
    for (int c = threadIdx.x; c < na + nb; c += HT) {
      if (c < na) hrow(A, seg, x0, c, oa);
      else hrow(Bs, seg, x0, c - na, ob);
    }
    __syncthreads();
    if (doA) store_row(oa, A.tmp + ((int64_t)b * A.rows + ra) * A.crop * 3, A.crop * 3);
    if (doB) store_row(ob, Bs.tmp + ((int64_t)b * Bs.rows + rb) * Bs.crop * 3, Bs.crop * 3);
    __syncthreads();  // smem is reused by the next image
  }
}

// vpass, 4 adjacent output pixels per thread (needs crop % 4 == 0): three 32-bit loads per tap cover 4 RGB pixels of
// the intermediate, one tap weight serves all 12 accumulators, and each channel is written as one 64-bit store.
__device__ __forceinline__ void mac4(int* acc, uint32_t w0, uint32_t w1, uint32_t w2, int w) {
  acc[0] += (int)(w0 & 255u) * w;         acc[1] += (int)((w0 >> 8) & 255u) * w;
  acc[2] += (int)((w0 >> 16) & 255u) * w; acc[3] += (int)(w0 >> 24) * w;
  acc[4] += (int)(w1 & 255u) * w;         acc[5] += (int)((w1 >> 8) & 255u) * w;
  acc[6] += (int)((w1 >> 16) & 255u) * w; acc[7] += (int)(w1 >> 24) * w;
  acc[8] += (int)(w2 & 255u) * w;         acc[9] += (int)((w2 >> 8) & 255u) * w;
  acc[10] += (int)((w2 >> 16) & 255u) * w; acc[11] += (int)(w2 >> 24) * w;
}

__global__ void __launch_bounds__(THREADS)
k1_vpass_vec4(int B, const __grid_constant__ Dual d) {
  __shared__ __half lut[3 * 256];
  const Spec& s = d.s[blockIdx.z];
  for (int i = threadIdx.x; i < 3 * 256; i += THREADS) lut[i] = s.lut[i];
  __syncthreads();
  const int qpr = s.crop >> 2;
  const int q = blockIdx.x * THREADS + threadIdx.x;
  if (q >= s.crop * qpr) return;
  const int y = q / qpr, xq = q - y * qpr;
  const int npix = s.crop * s.crop;
  const int rwords = (s.crop * 3) >> 2;
  const int ymin = __ldg(s.vb + 2 * y), ylen = __ldg(s.vb + 2 * y + 1);
  const int32_t* __restrict__ k = s.vk + y * s.kv;
  for (int b = blockIdx.y; b < B; b += gridDim.y) {
    const uint32_t* __restrict__ src =
        reinterpret_cast<const uint32_t*>(s.tmp + (int64_t)b * s.rows * s.crop * 3) + (int64_t)ymin * rwords + xq * 3;
    int acc[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) acc[i] = ROUND_HALF;
    for (int j = 0; j < ylen; ++j) {
      const uint32_t* p = src + (int64_t)j * rwords;
      mac4(acc, __ldg(p), __ldg(p + 1), __ldg(p + 2), __ldg(k + j));
    }
    __half* o = s.out + (int64_t)b * 3 * npix + (int64_t)y * s.crop + 4 * xq;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const __half2 lo = __halves2half2(lut[c * 256 + clip8(acc[c])], lut[c * 256 + clip8(acc[3 + c])]);
      const __half2 hi = __halves2half2(lut[c * 256 + clip8(acc[6 + c])], lut[c * 256 + clip8(acc[9 + c])]);
      uint2 v;
      v.x = *reinterpret_cast<const uint32_t*>(&lo);
      v.y = *reinterpret_cast<const uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o + (int64_t)c * npix) = v;
    }
  }
}

void check_i32(const at::Tensor& t, const at::Tensor& x, int64_t rows, int64_t cols, const char* what) {
  TORCH_CHECK(t.device() == x.device(), what, " must be on the input's device");
  TORCH_CHECK(t.scalar_type() == at::kInt && t.is_contiguous() && t.dim() == 2, what, " must be contiguous int32 2-D");
  TORCH_CHECK(t.size(0) == rows && (cols < 0 || t.size(1) == cols), what, " has the wrong shape");
}

Spec make_spec(const at::Tensor& x, const std::vector<at::Tensor>& t, const std::vector<int64_t>& g,
               at::Tensor& tmp, at::Tensor& out) {
  TORCH_CHECK(t.size() == 5, "spec tensors: [hb, hk, vb, vk, lut]");
  TORCH_CHECK(g.size() == 4, "spec geometry: [row0, rows, hx0, hx1]");
  const int64_t crop = t[0].size(0);
  check_i32(t[0], x, crop, 2, "hb");
  check_i32(t[1], x, crop, -1, "hk");
  check_i32(t[2], x, crop, 2, "vb");
  check_i32(t[3], x, crop, -1, "vk");
  TORCH_CHECK(t[4].device() == x.device() && t[4].scalar_type() == at::kHalf && t[4].is_contiguous() &&
                  t[4].numel() == 3 * 256, "lut must be contiguous fp16 [3,256] on the input's device");
  const int64_t B = x.size(0), H = x.size(1);
  const int64_t row0 = g[0], rows = g[1];
  TORCH_CHECK(row0 >= 0 && rows >= 1 && row0 + rows <= H, "row window outside the frame");
  TORCH_CHECK(g[2] >= 0 && g[2] < g[3] && g[3] <= x.size(2), "column span outside the frame");
  TORCH_CHECK(crop >= 1 && rows * crop * 3 < INT_MAX && crop * crop * 3 < INT_MAX, "crop/rows too large");
  tmp = at::empty({B, rows, crop, 3}, x.options());
  out = at::empty({B, 3, crop, crop}, x.options().dtype(at::kHalf));
  Spec s;
  s.hb = t[0].data_ptr<int32_t>();
  s.hk = t[1].data_ptr<int32_t>();
  s.vb = t[2].data_ptr<int32_t>();
  s.vk = t[3].data_ptr<int32_t>();
  s.lut = reinterpret_cast<const __half*>(t[4].data_ptr<at::Half>());
  s.tmp = tmp.data_ptr<uint8_t>();
  s.out = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
  s.kh = (int)t[1].size(1);
  s.kv = (int)t[3].size(1);
  s.row0 = (int)row0;
  s.rows = (int)rows;
  s.crop = (int)crop;
  s.hx0 = (int)g[2];
  s.hx1 = (int)g[3];
  return s;
}

}  // namespace

std::vector<at::Tensor> k1_dual_preprocess_cuda(const at::Tensor& x, const std::vector<at::Tensor>& ta,
                                                const std::vector<int64_t>& ga, const std::vector<at::Tensor>& tb,
                                                const std::vector<int64_t>& gb, bool force_scalar) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kByte && x.dim() == 4 && x.size(3) == 3, "x must be uint8 [B,H,W,3]");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.size(1) >= 1 && x.size(2) >= 1 && x.size(1) < INT_MAX && x.size(2) < INT_MAX, "bad frame size");
  const c10::cuda::CUDAGuard guard(x.device());
  at::Tensor tmpA, outA, tmpB, outB;
  Dual d;
  d.s[0] = make_spec(x, ta, ga, tmpA, outA);
  d.s[1] = make_spec(x, tb, gb, tmpB, outB);
  const int64_t B = x.size(0);
  if (B == 0) return {outA, outB};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int gy = (int)std::min<int64_t>(B, MAX_GRID_Y);
  // pass 1
  const int r0 = std::min(d.s[0].row0, d.s[1].row0);
  const int r1 = std::max(d.s[0].row0 + d.s[0].rows, d.s[1].row0 + d.s[1].rows);
  const int x0 = std::min(d.s[0].hx0, d.s[1].hx0), x1 = std::max(d.s[0].hx1, d.s[1].hx1);
  const int64_t smem = ((3 + (int64_t)(x1 - x0) * 3 + 15) & ~15) + ((d.s[0].crop * 3 + 15) & ~15) + d.s[1].crop * 3;
  if (!force_scalar && smem <= 48 * 1024) {
    const dim3 g1((unsigned)(r1 - r0), gy, 1);
    k1_hpass_dual<<<g1, HT, (size_t)smem, stream>>>(x.data_ptr<uint8_t>(), (int)B, (int)x.size(1), (int)x.size(2),
                                                    r0, x0, x1, d);
  } else {  // very wide frames: per-pixel kernel reading global memory, blockIdx.z selects the reader
    const int64_t n1 = std::max<int64_t>((int64_t)d.s[0].rows * d.s[0].crop, (int64_t)d.s[1].rows * d.s[1].crop);
    const dim3 g1((unsigned)((n1 + THREADS - 1) / THREADS), gy, 2);
    k1_hpass<<<g1, THREADS, 0, stream>>>(x.data_ptr<uint8_t>(), (int)B, (int)x.size(1), (int)x.size(2), d);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // pass 2
  if (!force_scalar && d.s[0].crop % 4 == 0 && d.s[1].crop % 4 == 0) {
    const int64_t nq = std::max<int64_t>((int64_t)d.s[0].crop * d.s[0].crop, (int64_t)d.s[1].crop * d.s[1].crop) / 4;
    const dim3 g2((unsigned)((nq + THREADS - 1) / THREADS), gy, 2);
    k1_vpass_vec4<<<g2, THREADS, 0, stream>>>((int)B, d);
  } else {
    const int64_t n2 = std::max<int64_t>((int64_t)d.s[0].crop * d.s[0].crop, (int64_t)d.s[1].crop * d.s[1].crop);
    const dim3 g2((unsigned)((n2 + THREADS - 1) / THREADS), gy, 2);
    k1_vpass<<<g2, THREADS, 0, stream>>>((int)B, d);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {outA, outB};
}
