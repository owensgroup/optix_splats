#include <iostream>
#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>



#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_types.h>
#define BUILD_DIR "/home/teja/research/optix_splats/_skbuild/linux-x86_64-3.11/cmake-build"

#define OPTIX_CHECK(error)                                                     \
  {                                                                            \
    if (error != OPTIX_SUCCESS)                                                \
      std::cerr << __FILE__ << ":" << __LINE__ << " Optix Error: '"            \
                << optixGetErrorString(error) << "'\n";                        \
  }

#define CUDA_CHECK(error)                                                      \
  {                                                                            \
    if (error != cudaSuccess)                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '"             \
                << cudaGetErrorString(error) << "'\n";                         \
  }

#define CUDA_CHECK(error)                                                      \
  {                                                                            \
    if (error != cudaSuccess)                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '"             \
                << cudaGetErrorString(error) << "'\n";                         \
  }

void optixLogCallback(unsigned int level, const char *tag, const char *message,
                      void *cbdata) {
  std::cout << "Optix Log[" << level << "][" << tag << "]: '" << message
            << "'\n";
}
void launch_saxpy_cuda(int N, float a, float *x, float *y, float *z);

std::string loadPtx(std::string filename) {
  std::ifstream ptx_in(filename);
  return std::string((std::istreambuf_iterator<char>(ptx_in)),
                     std::istreambuf_iterator<char>());
}

struct SaxpyParameters {
  int N;
  float a;
  float *x;
  float *y;
  float *z;
};

OptixDeviceContext createOptixContext() {
  cudaFree(0); // creates a CUDA context if there isn't already one
  optixInit(); // loads the optix library and populates the function table

  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &optixLogCallback;
  options.logCallbackLevel = 4;

  OptixDeviceContext optix_context = nullptr;
  optixDeviceContextCreate(0, // use current CUDA context
                           &options, &optix_context);

  return optix_context;
}

// load ptx and create module
void loadSaxpyModule(OptixModule &module, OptixDeviceContext optix_context,
                     OptixPipelineCompileOptions &pipeline_compile_options) {
  
  std::string ptx = loadPtx(BUILD_DIR "/ptx/kernels.ptx");
  
  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.maxRegisterCount =
      OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

  pipeline_compile_options.usesMotionBlur = false;
  pipeline_compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_compile_options.numPayloadValues = 0;
  pipeline_compile_options.numAttributeValues = 2; // 2 is the minimum
  pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  OPTIX_CHECK(optixModuleCreate(optix_context, &module_compile_options,
                                       &pipeline_compile_options, ptx.c_str(),
                                       ptx.size(), nullptr, nullptr, &module));
}

// load ptx and create module
void createSaxpyGroups(OptixProgramGroup *program_groups,
                       OptixDeviceContext optix_context, OptixModule module) {
  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc prog_group_desc[3] = {};

  prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  prog_group_desc[0].raygen.module = module;
  prog_group_desc[0].raygen.entryFunctionName = "__raygen__saxpy";

  // we need to create these but the entryFunctionNames are null
  prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  prog_group_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

  OPTIX_CHECK(optixProgramGroupCreate(optix_context, prog_group_desc, 3,
                                      &program_group_options, nullptr, nullptr,
                                      program_groups));
}

void createSaxpyPipeline(
    OptixPipeline &pipeline, OptixDeviceContext optix_context,
    OptixProgramGroup *program_groups,
    OptixPipelineCompileOptions &pipeline_compile_options) {
  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = 1;
  

  OPTIX_CHECK(optixPipelineCreate(optix_context, &pipeline_compile_options,
                                  &pipeline_link_options, program_groups, 3,
                                  nullptr, nullptr, &pipeline));
}

void populateSaxpySBT(OptixShaderBindingTable &sbt,
                      OptixProgramGroup *program_groups) {
  char *device_records;
  CUDA_CHECK(cudaMalloc(&device_records, 3 * OPTIX_SBT_RECORD_HEADER_SIZE));

  char *raygen_record = device_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *miss_record = device_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *hitgroup_record = device_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE;

  char sbt_records[3 * OPTIX_SBT_RECORD_HEADER_SIZE];
  OPTIX_CHECK(optixSbtRecordPackHeader(
      program_groups[0], sbt_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader(
      program_groups[1], sbt_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader(
      program_groups[2], sbt_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE));

  CUDA_CHECK(cudaMemcpy(device_records, sbt_records,
                        3 * OPTIX_SBT_RECORD_HEADER_SIZE,
                        cudaMemcpyHostToDevice));

  sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_record);

  sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record);
  sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  sbt.missRecordCount = 1;

  sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_record);
  sbt.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  sbt.hitgroupRecordCount = 1;
}

void launch_saxpy_optix(int N, float a, float *x, float *y, float *z) {
  OptixDeviceContext optix_context = createOptixContext(); 
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixModule module = nullptr;
  loadSaxpyModule(module, optix_context, pipeline_compile_options);


  OptixProgramGroup program_groups[3] = {};
  createSaxpyGroups(program_groups, optix_context, module);

  OptixPipeline pipeline = nullptr;
  createSaxpyPipeline(pipeline, optix_context, program_groups,
                      pipeline_compile_options);

  OptixShaderBindingTable sbt = {};
  populateSaxpySBT(sbt, program_groups);

  
  SaxpyParameters params{N, a, x, y, z};
  SaxpyParameters *device_params;
  CUDA_CHECK(cudaMalloc(&device_params, sizeof(SaxpyParameters)));
  CUDA_CHECK(cudaMemcpy(device_params, &params, sizeof(SaxpyParameters),
                        cudaMemcpyHostToDevice));
  

  OPTIX_CHECK(optixLaunch(pipeline, 0,
                          reinterpret_cast<CUdeviceptr>(device_params),
                          sizeof(SaxpyParameters), &sbt, N, 1, 1));
  OPTIX_CHECK(optixPipelineDestroy(pipeline));
  for (int i = 0; i < 3; ++i) {
    OPTIX_CHECK(optixProgramGroupDestroy(program_groups[i]));
  }
  OPTIX_CHECK(optixModuleDestroy(module));
  OPTIX_CHECK(optixDeviceContextDestroy(optix_context));

  CUDA_CHECK(cudaFree(device_params));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));

}
__global__ void saxpy_kernel(int n, float a, float *x, float *y, float *z) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    z[i] = a * x[i] + y[i]; 
}

torch::Tensor saxpy(torch::Tensor x, torch::Tensor y, float a) {
  int N = x.numel();
  torch::Tensor z = torch::zeros_like(x);
  launch_saxpy_cuda(N, a, x.data<float>(), y.data<float>(), z.data<float>());
  return z;
}

torch::Tensor saxpy_optix(torch::Tensor x, torch::Tensor y, float a) {
  int N = x.numel();
  torch::Tensor z = torch::zeros_like(x);
  launch_saxpy_optix(N, a, x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>());
  return z;
}

void launch_saxpy_cuda(int N, float a, float *x, float *y, float *z) {
  saxpy_kernel<<<(N + 255) / 256, 256>>>(N, a, x, y, z);
  CUDA_CHECK(cudaGetLastError());
}
