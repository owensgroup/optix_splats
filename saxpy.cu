#include <iostream>

#define CUDA_CHECK(error)                                                      \
  {                                                                            \
    if (error != cudaSuccess)                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '"             \
                << cudaGetErrorString(error) << "'\n";                         \
  }


__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

void launch_saxpy_cuda(int N, float a, float *x, float *y) {
  saxpy<<<(N + 255) / 256, 256>>>(N, a, x, y);
  CUDA_CHECK(cudaGetLastError());
}