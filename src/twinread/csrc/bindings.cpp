// pybind11 bindings for the TwinRead CUDA kernels (K1). Research prototype, not a diagnostic device.
#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> k1_dual_preprocess_cuda(const at::Tensor& x, const std::vector<at::Tensor>& ta,
                                                const std::vector<int64_t>& ga, const std::vector<at::Tensor>& tb,
                                                const std::vector<int64_t>& gb, bool force_scalar);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dual_preprocess", &k1_dual_preprocess_cuda,
        "K1: uint8 [B,H,W,3] -> (fp16 [B,3,cA,cA], fp16 [B,3,cB,cB]); spec = [hb, hk, vb, vk, lut], geom = [row0, rows, hx0, hx1]",
        pybind11::arg("x"), pybind11::arg("spec_a"), pybind11::arg("geom_a"), pybind11::arg("spec_b"),
        pybind11::arg("geom_b"), pybind11::arg("force_scalar") = false);
}
