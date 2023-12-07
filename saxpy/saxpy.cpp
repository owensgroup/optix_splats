#include <torch/extension.h>

torch::Tensor saxpy(torch::Tensor x, torch::Tensor y, float a);
torch::Tensor saxpy_optix(torch::Tensor x, torch::Tensor y, float a);

PYBIND11_MODULE(saxpy_ext, m) {
  m.def("saxpy", &saxpy, "saxpy (cuda)");
  m.def("saxpy_optix", &saxpy_optix, "saxpy (optix)");
}