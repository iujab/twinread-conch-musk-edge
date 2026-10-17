// K2: fused camera-of-screen augmentation (TwinRead). SIMULATION of a USB camera filming a tile on a screen.
// Research prototype, not a diagnostic device.
//
// Input: uint8 [B,H,W,3] on the GPU and a packed per-image parameter table float32 [B,35] (column layout:
// twinread/ref/camera_ops.py FIELDS). Output: uint8 [B,H,W,3]. Op order (fixed):
//   warp (bilinear) -> HED jitter -> display gamma + colour profile -> moire -> glare -> defocus (separable
//   Gaussian, radius ceil(3 sigma)) -> white balance -> sensor noise (Philox) -> JPEG-like 8x8 DCT quantization.
// Every op is defined (formula by formula) in twinread/ref/camera_ops.py, the float64 numpy reference these
// kernels are tested against; an op whose params are the identity is skipped, so identity params are exact.
//
// Launch 1, k2_fused: one 256-thread block per 32x16 output tile of one image (gridDim.z strides over the batch).
//   (1) stages warp..glare are evaluated for the tile plus a halo of r = ceil(3 sigma) pixels at clamped
//       coordinates (= replicated borders for the blur) into a shared-memory tile, (2) horizontal Gaussian pass
//       shared -> shared, (3) vertical pass, white balance and sensor noise per output pixel, written as float32
//       [B,H,W,3] in [0,1]. The dynamic shared-memory tile is sized by the batch's largest radius R <= 12
//       (sigma <= 4): (16+2R)(32+2R)*3 + (16+2R)*32*3 floats, 42.5 KB at R = 12.
//   Noise: curand_Philox4x32_10 with key = (seed lo32, seed hi32), counter = (y*W + x, image index, 0, 0);
//   uniforms ((bits >> 9) + 0.5) * 2^-23, Box-Muller; normals 0..2 -> R, G, B. Read + shot noise:
//   v + std * sqrt(0.2 + 0.8 v) * z.
// Launch 2, k2_jpeg: one 192-thread block per 8x8 pixel block of one image, one thread per (pixel, channel)
//   (the "8x8 x channel" block): RGB -> YCbCr (JFIF), level shift, 8x8 DCT-II (orthonormal) as row and column
//   passes against the cosine table in shared memory (precomputed on the host in float64), quantization with the
//   Annex K tables scaled by the IJG quality formula, floor(F/Q + 0.5) * Q, inverse DCT, -> RGB, clamp, round half to
//   even. Chroma is NOT subsampled (4:4:4). Partial edge blocks read clamped coordinates (edge padding inside the
//   block) and write only in-image pixels. Images with jpeg_quality <= 0 are only rounded.
//
// HED jitter: colour-deconvolution math ported from StainForge (stainforge-cuda-stain-kernels/src/stainforge/csrc/
// stainforge_kernels.cu, hed_jitter_kernel: OD = log(max(v, 1e-6)) / log(1e-6), s = max(0, OD . hed_from_rgb),
// s' = alpha s + beta, rgb = exp(log(1e-6) * s' . rgb_from_hed)), here on float input with logf instead of the
// uint8 LUT and without the uint8 rounding. Build without --use_fast_math.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int NP = 35;
constexpr int P_GAMMA = 0, P_PROFILE = 1, P_WB = 10, P_MOIRE_AMP = 13, P_MOIRE_PERIOD = 14, P_MOIRE_ANGLE = 15,
              P_GLARE_AMP = 16, P_GLARE_ANGLE = 17, P_WARP = 18, P_BLUR = 26, P_NOISE = 27, P_JPEG = 28,
              P_HED_A = 29, P_HED_B = 32;
constexpr int TW = 32, TH = 16, THREADS = 256, MAX_R = 12, JT = 192;
constexpr int MAX_GRID_YZ = 65535;
constexpr float MAX_SIGMA = 4.f, MIN_SIGMA = 1e-3f, NOISE_READ = 0.2f;
constexpr float LOG_ADJ = -13.815510557964274f;  // log(1e-6), skimage's HED scaling (as StainForge)
constexpr float INV_2P23 = 1.1920928955078125e-07f;  // 2^-23
constexpr float INV_SQRT2 = 0.70710678118654752f;

struct Consts {
  float hed_from_rgb[9];  // row-major [i][j]: s_j = sum_i od_i * M[i][j]
  float rgb_from_hed[9];
  float dct[64];          // C[k][n] = c(k) cos(pi (2n+1) k / 16), orthonormal
};

__constant__ int LUMA_Q[64] = {16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
                               14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
                               18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
                               49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
__constant__ int CHROMA_Q[64] = {17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
                                 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
                                 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                                 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

__device__ __forceinline__ float clamp01(float v) { return fminf(fmaxf(v, 0.f), 1.f); }

// Per-image parameters, unpacked once per block from the shared parameter row.
struct ImgP {
  bool warp, hed, gamma, profile, moire, glare, wb, noise;
  float d[8], ha[3], hb[3], g, M[9], m_amp, m_per, m_ct, m_st, gl_amp, gl_ct, gl_st, wbg[3], nstd;
};

__device__ __forceinline__ ImgP unpack(const float* sp) {
  ImgP q;
  q.warp = false;
  for (int k = 0; k < 8; ++k) {
    q.d[k] = sp[P_WARP + k];
    q.warp |= q.d[k] != 0.f;
  }
  q.hed = false;
  for (int k = 0; k < 3; ++k) {
    q.ha[k] = sp[P_HED_A + k];
    q.hb[k] = sp[P_HED_B + k];
    q.hed |= (q.ha[k] != 1.f) || (q.hb[k] != 0.f);
  }
  q.gamma = sp[P_GAMMA] != 1.f;
  q.g = fminf(fmaxf(sp[P_GAMMA], 0.05f), 20.f);
  q.profile = false;
  for (int k = 0; k < 9; ++k) {
    q.M[k] = sp[P_PROFILE + k];
    q.profile |= q.M[k] != ((k % 4 == 0) ? 1.f : 0.f);
  }
  q.m_amp = sp[P_MOIRE_AMP];
  q.moire = q.m_amp != 0.f;
  q.m_per = fmaxf(sp[P_MOIRE_PERIOD], 2.f);
  sincosf(sp[P_MOIRE_ANGLE], &q.m_st, &q.m_ct);
  q.glare = sp[P_GLARE_AMP] != 0.f;
  q.gl_amp = fminf(fmaxf(sp[P_GLARE_AMP], 0.f), 1.f);
  sincosf(sp[P_GLARE_ANGLE], &q.gl_st, &q.gl_ct);
  q.wb = false;
  for (int k = 0; k < 3; ++k) {
    q.wbg[k] = sp[P_WB + k];
    q.wb |= q.wbg[k] != 1.f;
  }
  q.nstd = sp[P_NOISE];
  q.noise = q.nstd > 0.f;
  return q;
}

// warp -> HED -> display -> moire -> glare for the pixel at (gx, gy) (in-image coordinates).
__device__ __forceinline__ void stages_pre_blur(const uint8_t* __restrict__ xb, int H, int W, int gx, int gy,
                                                const ImgP& q, const Consts& kc, float v[3]) {
  if (q.warp) {
    const float xn = (2.f * gx + 1.f) / W - 1.f, yn = (2.f * gy + 1.f) / H - 1.f;
    const float u = (1.f + q.d[0]) * xn + q.d[1] * yn + q.d[2];
    const float vv = q.d[3] * xn + (1.f + q.d[4]) * yn + q.d[5];
    const float w = fmaxf(q.d[6] * xn + q.d[7] * yn + 1.f, 1e-6f);
    const float sx = fminf(fmaxf(((u / w + 1.f) * W - 1.f) * 0.5f, 0.f), (float)(W - 1));
    const float sy = fminf(fmaxf(((vv / w + 1.f) * H - 1.f) * 0.5f, 0.f), (float)(H - 1));
    const int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    const int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    const float fx = sx - x0, fy = sy - y0;
    const uint8_t* p00 = xb + ((int64_t)y0 * W + x0) * 3;
    const uint8_t* p01 = xb + ((int64_t)y0 * W + x1) * 3;
    const uint8_t* p10 = xb + ((int64_t)y1 * W + x0) * 3;
    const uint8_t* p11 = xb + ((int64_t)y1 * W + x1) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const float top = __ldg(p00 + c) / 255.f * (1.f - fx) + __ldg(p01 + c) / 255.f * fx;
      const float bot = __ldg(p10 + c) / 255.f * (1.f - fx) + __ldg(p11 + c) / 255.f * fx;
      v[c] = top * (1.f - fy) + bot * fy;
    }
  } else {
    const uint8_t* p = xb + ((int64_t)gy * W + gx) * 3;
#pragma unroll
    for (int c = 0; c < 3; ++c) v[c] = __ldg(p + c) / 255.f;
  }
  if (q.hed) {  // StainForge hed_jitter_kernel math (see header)
    const float o0 = logf(fmaxf(v[0], 1e-6f)) / LOG_ADJ, o1 = logf(fmaxf(v[1], 1e-6f)) / LOG_ADJ,
                o2 = logf(fmaxf(v[2], 1e-6f)) / LOG_ADJ;
    const float* Hm = kc.hed_from_rgb;
    const float* R = kc.rgb_from_hed;
    float s0 = fmaxf(0.f, fmaf(o2, Hm[6], fmaf(o1, Hm[3], o0 * Hm[0])));
    float s1 = fmaxf(0.f, fmaf(o2, Hm[7], fmaf(o1, Hm[4], o0 * Hm[1])));
    float s2 = fmaxf(0.f, fmaf(o2, Hm[8], fmaf(o1, Hm[5], o0 * Hm[2])));
    s0 = fmaf(s0, q.ha[0], q.hb[0]);
    s1 = fmaf(s1, q.ha[1], q.hb[1]);
    s2 = fmaf(s2, q.ha[2], q.hb[2]);
    v[0] = clamp01(expf(LOG_ADJ * fmaf(s2, R[6], fmaf(s1, R[3], s0 * R[0]))));
    v[1] = clamp01(expf(LOG_ADJ * fmaf(s2, R[7], fmaf(s1, R[4], s0 * R[1]))));
    v[2] = clamp01(expf(LOG_ADJ * fmaf(s2, R[8], fmaf(s1, R[5], s0 * R[2]))));
  }
  if (q.gamma) {
#pragma unroll
    for (int c = 0; c < 3; ++c) v[c] = powf(clamp01(v[c]), q.g);
  }
  if (q.profile) {
    const float a = v[0], b = v[1], c = v[2];
    v[0] = clamp01(q.M[0] * a + q.M[1] * b + q.M[2] * c);
    v[1] = clamp01(q.M[3] * a + q.M[4] * b + q.M[5] * c);
    v[2] = clamp01(q.M[6] * a + q.M[7] * b + q.M[8] * c);
  }
  if (q.moire) {
    const float t = ((gx + 0.5f) * q.m_ct + (gy + 0.5f) * q.m_st) / q.m_per;
#pragma unroll
    for (int c = 0; c < 3; ++c) v[c] = clamp01(v[c] * (1.f + q.m_amp * cospif(2.f * t + c / 3.f)));
  }
  if (q.glare) {
    const float xn = (2.f * gx + 1.f) / W - 1.f, yn = (2.f * gy + 1.f) / H - 1.f;
    const float d = (xn * q.gl_ct + yn * q.gl_st) * INV_SQRT2;
    float g = (d + 1.f) * 0.5f;
    g = q.gl_amp * g * g;
#pragma unroll
    for (int c = 0; c < 3; ++c) v[c] = clamp01(v[c] + g * (1.f - v[c]));
  }
}

__global__ void __launch_bounds__(THREADS)
k2_fused(const uint8_t* __restrict__ x, float* __restrict__ inter, const float* __restrict__ params, int B, int H,
         int W, int R, uint32_t key0, uint32_t key1, const __grid_constant__ Consts kc) {
  extern __shared__ float sm[];
  __shared__ float sp[NP];
  __shared__ float wts[2 * MAX_R + 1];
  const int tid = threadIdx.x;
  float* S = sm;                                    // [(TH+2r)][(TW+2r)][3]
  float* Hs = sm + (TH + 2 * R) * (TW + 2 * R) * 3;  // [(TH+2r)][TW][3]
  const int x0 = blockIdx.x * TW, y0 = blockIdx.y * TH;
  const int64_t frame = (int64_t)H * W * 3;
  for (int b = blockIdx.z; b < B; b += gridDim.z) {
    if (tid < NP) sp[tid] = params[(int64_t)b * NP + tid];
    __syncthreads();
    const ImgP q = unpack(sp);
    const float sraw = sp[P_BLUR];
    int r = 0;
    float sig = 0.f;
    if (sraw > 0.f) {
      sig = fminf(fmaxf(sraw, MIN_SIGMA), MAX_SIGMA);
      r = min((int)ceilf(3.f * sig), R);  // the host sizes R from the batch maximum, so min() never binds
    }
    if (r > 0 && tid <= 2 * r) {
      const float k = (float)(tid - r);
      wts[tid] = (tid == r) ? 1.f : expf(-(k * k) / (2.f * sig * sig));
    }
    __syncthreads();
    if (r > 0 && tid == 0) {
      float s = 0.f;
      for (int j = 0; j <= 2 * r; ++j) s += wts[j];
      for (int j = 0; j <= 2 * r; ++j) wts[j] /= s;
    }
    const uint8_t* __restrict__ xb = x + (int64_t)b * frame;
    const int sw = TW + 2 * r, sh = TH + 2 * r;
    // (1) pre-blur stages on the tile + halo; clamped coordinates replicate the image border
    for (int i = tid; i < sw * sh; i += THREADS) {
      const int ey = i / sw, ex = i - ey * sw;
      const int gy = min(max(y0 - r + ey, 0), H - 1), gx = min(max(x0 - r + ex, 0), W - 1);
      float v[3];
      stages_pre_blur(xb, H, W, gx, gy, q, kc, v);
      S[3 * i] = v[0];
      S[3 * i + 1] = v[1];
      S[3 * i + 2] = v[2];
    }
    __syncthreads();
    // (2) horizontal Gaussian pass
    if (r > 0) {
      for (int i = tid; i < sh * TW; i += THREADS) {
        const int ey = i / TW, xx = i - ey * TW;
        const float* s = S + (ey * sw + xx) * 3;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f;
        for (int j = 0; j <= 2 * r; ++j) {
          const float w = wts[j];
          a0 = fmaf(w, s[3 * j], a0);
          a1 = fmaf(w, s[3 * j + 1], a1);
          a2 = fmaf(w, s[3 * j + 2], a2);
        }
        Hs[3 * i] = a0;
        Hs[3 * i + 1] = a1;
        Hs[3 * i + 2] = a2;
      }
      __syncthreads();
    }
    // (3) vertical pass, white balance, sensor noise, store
    float* __restrict__ ob = inter + (int64_t)b * frame;
    for (int i = tid; i < TH * TW; i += THREADS) {
      const int yy = i / TW, xx = i - yy * TW;
      const int gy = y0 + yy, gx = x0 + xx;
      if (gy >= H || gx >= W) continue;
      float v[3];
      if (r > 0) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f;
        for (int j = 0; j <= 2 * r; ++j) {
          const float w = wts[j];
          const float* h = Hs + ((yy + j) * TW + xx) * 3;
          a0 = fmaf(w, h[0], a0);
          a1 = fmaf(w, h[1], a1);
          a2 = fmaf(w, h[2], a2);
        }
        v[0] = a0;
        v[1] = a1;
        v[2] = a2;
      } else {
        const float* s = S + (yy * sw + xx) * 3;
        v[0] = s[0];
        v[1] = s[1];
        v[2] = s[2];
      }
      if (q.wb) {
#pragma unroll
        for (int c = 0; c < 3; ++c) v[c] = clamp01(v[c] * q.wbg[c]);
      }
      if (q.noise) {
        const uint4 bits = curand_Philox4x32_10(make_uint4((uint32_t)((int64_t)gy * W + gx), (uint32_t)b, 0u, 0u),
                                                make_uint2(key0, key1));
        const float u0 = ((float)(bits.x >> 9) + 0.5f) * INV_2P23, u1 = ((float)(bits.y >> 9) + 0.5f) * INV_2P23;
        const float u2 = ((float)(bits.z >> 9) + 0.5f) * INV_2P23, u3 = ((float)(bits.w >> 9) + 0.5f) * INV_2P23;
        const float r1 = sqrtf(-2.f * logf(u0)), r2 = sqrtf(-2.f * logf(u2));
        float s1, c1;
        sincospif(2.f * u1, &s1, &c1);
        const float z[3] = {r1 * c1, r1 * s1, r2 * cospif(2.f * u3)};
#pragma unroll
        for (int c = 0; c < 3; ++c)
          v[c] = clamp01(v[c] + q.nstd * sqrtf(NOISE_READ + (1.f - NOISE_READ) * v[c]) * z[c]);
      }
      float* o = ob + ((int64_t)gy * W + gx) * 3;
      o[0] = v[0];
      o[1] = v[1];
      o[2] = v[2];
    }
    __syncthreads();  // shared tiles and parameters are reused for the next image
  }
}

__global__ void __launch_bounds__(JT)
k2_jpeg(const float* __restrict__ inter, uint8_t* __restrict__ out, const float* __restrict__ params, int B, int H,
        int W, const __grid_constant__ Consts kc) {
  __shared__ float C[64];
  __shared__ float s0[3][64], s1[3][64];
  const int t = threadIdx.x, c = t >> 6, p = t & 63, i = p >> 3, j = p & 7;
  if (t < 64) C[t] = kc.dct[t];
  const int gx = blockIdx.x * 8 + j, gy = blockIdx.y * 8 + i;
  const bool in = gx < W && gy < H;
  const int64_t src = ((int64_t)min(gy, H - 1) * W + min(gx, W - 1)) * 3 + c;  // edge padding inside the block
  const int64_t dst = ((int64_t)gy * W + gx) * 3 + c;
  const int64_t frame = (int64_t)H * W * 3;
  __syncthreads();
  for (int b = blockIdx.z; b < B; b += gridDim.z) {
    const float q = __ldg(params + (int64_t)b * NP + P_JPEG);  // uniform over the block
    const float* __restrict__ ib = inter + (int64_t)b * frame;
    uint8_t* __restrict__ obuf = out + (int64_t)b * frame;
    if (!(q > 0.f)) {  // JPEG off for this image: round only
      if (in) obuf[dst] = (uint8_t)__float2int_rn(fminf(fmaxf(ib[dst] * 255.f, 0.f), 255.f));
      continue;
    }
    const int qi = min(max((int)floorf(q + 0.5f), 1), 100);
    const int scale = qi < 50 ? 5000 / qi : 200 - 2 * qi;
    const float Q = (float)min(max(((c == 0 ? LUMA_Q[p] : CHROMA_Q[p]) * scale + 50) / 100, 1), 255);
    s0[c][p] = 255.f * ib[src];
    __syncthreads();
    const float r = s0[0][p], g = s0[1][p], bl = s0[2][p];
    float y;
    if (c == 0)
      y = 0.299f * r + 0.587f * g + 0.114f * bl - 128.f;
    else if (c == 1)
      y = -0.168736f * r - 0.331264f * g + 0.5f * bl;
    else
      y = 0.5f * r - 0.418688f * g - 0.081312f * bl;
    s1[c][p] = y;
    __syncthreads();
    float a = 0.f;  // rows: T[i][l] = sum_n X[i][n] C[l][n]
#pragma unroll
    for (int n = 0; n < 8; ++n) a = fmaf(s1[c][i * 8 + n], C[j * 8 + n], a);
    s0[c][p] = a;
    __syncthreads();
    a = 0.f;  // columns: F[k][l] = sum_m C[k][m] T[m][l]; quantize
#pragma unroll
    for (int m = 0; m < 8; ++m) a = fmaf(C[i * 8 + m], s0[c][m * 8 + j], a);
    s1[c][p] = floorf(a / Q + 0.5f) * Q;
    __syncthreads();
    a = 0.f;  // inverse rows: U[k][n] = sum_l F[k][l] C[l][n]
#pragma unroll
    for (int l = 0; l < 8; ++l) a = fmaf(s1[c][i * 8 + l], C[l * 8 + j], a);
    s0[c][p] = a;
    __syncthreads();
    a = 0.f;  // inverse columns: X[m][n] = sum_k C[k][m] U[k][n]
#pragma unroll
    for (int k = 0; k < 8; ++k) a = fmaf(C[k * 8 + i], s0[c][k * 8 + j], a);
    s1[c][p] = a;
    __syncthreads();
    const float Y = s1[0][p] + 128.f, Cb = s1[1][p], Cr = s1[2][p];
    const float o = c == 0 ? Y + 1.402f * Cr : (c == 1 ? Y - 0.344136f * Cb - 0.714136f * Cr : Y + 1.772f * Cb);
    if (in) obuf[dst] = (uint8_t)__float2int_rn(fminf(fmaxf(o, 0.f), 255.f));
    __syncthreads();  // s0/s1 are reused for the next image
  }
}

__global__ void k2_philox(const int64_t* __restrict__ ctr, int64_t* __restrict__ out, int64_t n, uint32_t k0,
                          uint32_t k1) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const uint4 r = curand_Philox4x32_10(
        make_uint4((uint32_t)ctr[4 * i], (uint32_t)ctr[4 * i + 1], (uint32_t)ctr[4 * i + 2], (uint32_t)ctr[4 * i + 3]),
        make_uint2(k0, k1));
    out[4 * i] = r.x;
    out[4 * i + 1] = r.y;
    out[4 * i + 2] = r.z;
    out[4 * i + 3] = r.w;
  }
}

}  // namespace

std::vector<at::Tensor> k2_camera_aug_cuda(const at::Tensor& x, const at::Tensor& params, int64_t seed, int64_t rmax,
                                           const std::vector<double>& consts) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kByte && x.dim() == 4 && x.size(3) == 3 && x.is_contiguous(),
              "K2: x must be a contiguous CUDA uint8 [B,H,W,3] tensor");
  const int64_t B = x.size(0), H = x.size(1), W = x.size(2);
  TORCH_CHECK(H >= 1 && W >= 1, "K2: empty frame");
  TORCH_CHECK(params.is_cuda() && params.device() == x.device() && params.scalar_type() == at::kFloat &&
                  params.is_contiguous() && params.dim() == 2 && params.size(0) == B && params.size(1) == NP,
              "K2: params must be a contiguous float32 [B,35] tensor on x's device");
  TORCH_CHECK(rmax >= 0 && rmax <= MAX_R, "K2: blur radius must be in [0, 12]");
  TORCH_CHECK(consts.size() == 9 + 9 + 64, "K2: consts must hold hed_from_rgb(9), rgb_from_hed(9), dct(64)");
  TORCH_CHECK((W + TW - 1) / TW <= INT32_MAX && (H + TH - 1) / TH <= MAX_GRID_YZ && (H + 7) / 8 <= MAX_GRID_YZ &&
                  H * W <= (int64_t)UINT32_MAX,
              "K2: frame too large");
  c10::cuda::CUDAGuard guard(x.device());
  auto inter = at::empty({B, H, W, 3}, x.options().dtype(at::kFloat));
  auto out = at::empty_like(x);
  if (B == 0) return {out, inter};
  Consts kc;
  for (int k = 0; k < 9; ++k) {
    kc.hed_from_rgb[k] = (float)consts[k];
    kc.rgb_from_hed[k] = (float)consts[9 + k];
  }
  for (int k = 0; k < 64; ++k) kc.dct[k] = (float)consts[18 + k];
  const uint64_t s = (uint64_t)seed;
  const uint32_t key0 = (uint32_t)(s & 0xffffffffull), key1 = (uint32_t)(s >> 32);
  const int R = (int)rmax;
  const size_t smem = (size_t)((TH + 2 * R) * (TW + 2 * R) + (TH + 2 * R) * TW) * 3 * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned gz = (unsigned)std::min<int64_t>(B, MAX_GRID_YZ);
  dim3 g1((unsigned)((W + TW - 1) / TW), (unsigned)((H + TH - 1) / TH), gz);
  k2_fused<<<g1, THREADS, smem, stream>>>(x.data_ptr<uint8_t>(), inter.data_ptr<float>(), params.data_ptr<float>(),
                                          (int)B, (int)H, (int)W, R, key0, key1, kc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dim3 g2((unsigned)((W + 7) / 8), (unsigned)((H + 7) / 8), gz);
  k2_jpeg<<<g2, JT, 0, stream>>>(inter.data_ptr<float>(), out.data_ptr<uint8_t>(), params.data_ptr<float>(), (int)B,
                                 (int)H, (int)W, kc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, inter};
}

at::Tensor k2_philox_bits(const at::Tensor& ctr, int64_t k0, int64_t k1) {
  TORCH_CHECK(ctr.is_cuda() && ctr.scalar_type() == at::kLong && ctr.dim() == 2 && ctr.size(1) == 4 &&
                  ctr.is_contiguous(),
              "philox_bits: ctr must be a contiguous CUDA int64 [N,4] tensor");
  c10::cuda::CUDAGuard guard(ctr.device());
  auto out = at::empty_like(ctr);
  const int64_t n = ctr.size(0);
  if (n == 0) return out;
  const int blocks = (int)std::min<int64_t>((n + 255) / 256, 4096);
  k2_philox<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(ctr.data_ptr<int64_t>(), out.data_ptr<int64_t>(),
                                                                   n, (uint32_t)k0, (uint32_t)k1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
