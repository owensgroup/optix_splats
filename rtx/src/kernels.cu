#include <optix.h>

struct SaxpyParameters {
  int N;
  float a;
  float *x;
  float *y;
};

extern "C" static __constant__ SaxpyParameters params;

extern "C" __global__ void __raygen__saxpy() {
  const uint3 idx = optixGetLaunchIndex();

  params.y[idx.x] = params.a * params.x[idx.x] + params.y[idx.x];
}