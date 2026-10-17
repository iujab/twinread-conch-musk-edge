// pybind11 bindings for the TwinRead CUDA kernels (K1, K2). Research prototype, not a diagnostic device.
// Built with -DTWINREAD_NO_K2 (K1 sources only) as the loader's fallback when the K2 translation unit fails to
// build, so a K2 problem never takes K1 down (twinread/kernels.py:load_ext).
#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> k1_dual_preprocess_cuda(const at::Tensor& x, const std::vector<at::Tensor>& ta,
                                                const std::vector<int64_t>& ga, const std::vector<at::Tensor>& tb,
                                                const std::vector<int64_t>& gb, bool force_scalar);
#ifndef TWINREAD_NO_K2
std::vector<at::Tensor> k2_camera_aug_cuda(const at::Tensor& x, const at::Tensor& params, int64_t seed, int64_t rmax,
                                           const std::vector<double>& consts);
at::Tensor k2_philox_bits(const at::Tensor& ctr, int64_t k0, int64_t k1);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dual_preprocess", &k1_dual_preprocess_cuda,
        "K1: uint8 [B,H,W,3] -> (fp16 [B,3,cA,cA], fp16 [B,3,cB,cB]); spec = [hb, hk, vb, vk, lut], geom = [row0, rows, hx0, hx1]",
        pybind11::arg("x"), pybind11::arg("spec_a"), pybind11::arg("geom_a"), pybind11::arg("spec_b"),
        pybind11::arg("geom_b"), pybind11::arg("force_scalar") = false);
#ifndef TWINREAD_NO_K2
  m.def("camera_aug", &k2_camera_aug_cuda,
        "K2: uint8 [B,H,W,3], params float32 [B,35] -> (uint8 [B,H,W,3], float32 pre-JPEG [B,H,W,3]); "
        "consts = hed_from_rgb(9) + rgb_from_hed(9) + dct(64)",
        pybind11::arg("x"), pybind11::arg("params"), pybind11::arg("seed"), pybind11::arg("rmax"),
        pybind11::arg("consts"));
  m.def("philox_bits", &k2_philox_bits, "curand_Philox4x32_10 on int64 [N,4] counters (test hook)",
        pybind11::arg("ctr"), pybind11::arg("k0"), pybind11::arg("k1"));
#endif
}
