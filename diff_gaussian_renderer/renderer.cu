#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_types.h>

struct Params
{
    int*  image;
    unsigned int  image_width;
    unsigned int  image_height;
    float3   cam_eye;
    float3   cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};
// These structs represent the data blocks of our SBT records
struct RayGenData   { };// No data needed };
struct HitGroupData { };// No data needed };
struct MissData     { float3 bg_color;  };

// SBT record with an appropriately aligned and sized data block
template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT )
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

// std::vector<torch::Tensor> render_gaussians(
//     const torch::Tensor& means,
//     const torch::Tensor& scales,
//     const torch::Tensor& rotations,
//     const torch::Tensor& colors,
//     const torch::Tensor& opacity,
//     const float tan_fovx,
//     const int image_height,
//     const int image_width,
//     const torch::Tensor& cam_pos
// ) {

    
// }
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
void optixLogCallback(unsigned int level, const char *tag, const char *message,
                      void *cbdata) {
  std::cout << "Optix Log[" << level << "][" << tag << "]: '" << message
            << "'\n";
}

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



std::string loadPtx(std::string filename) {
  std::ifstream ptx_in(filename);
  return std::string((std::istreambuf_iterator<char>(ptx_in)),
                     std::istreambuf_iterator<char>());
}

torch::Tensor render_gaussians(int image_height, int image_width) {
    std::vector<float> vertex_buffer = {
        -0.5f, -0.5f, 0.0f,
        -0.5f,  0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.5f,  0.5f, 0.0f
    };

    std::vector<uint32_t> index_buffer = {
        0, 1, 2,
        1, 2, 3
    };
    

    std::cout << "Making image tensor height " << image_height << " width " << image_width << std::endl;
    // create torch tensor with size of image_height x image_width x 3 
    std::cout << "Creating optix context" << std::endl;
    OptixDeviceContext context = createOptixContext();

    std::cout << "Allocating vertex buffer" << std::endl;
    CUdeviceptr vertex_device;
    cudaMalloc((void**)&vertex_device, sizeof(float) * vertex_buffer.size());
    cudaMemcpy((void*)vertex_device, vertex_buffer.data(), sizeof(float) * vertex_buffer.size(), cudaMemcpyHostToDevice);

    CUdeviceptr index_device;
    cudaMalloc((void**)&index_device, sizeof(uint32_t) * index_buffer.size());
    cudaMemcpy((void*)index_device, index_buffer.data(), sizeof(uint32_t) * index_buffer.size(), cudaMemcpyHostToDevice);

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    std::cout << "Creating build input" << std::endl;
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    // Create optix build input for triangles
    OptixBuildInputTriangleArray& triangleArray = buildInput.triangleArray;
    triangleArray.vertexBuffers = &vertex_device;
    triangleArray.numVertices = vertex_buffer.size();
    triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    triangleArray.indexBuffer = index_device;
    triangleArray.numIndexTriplets = index_buffer.size() / 3;
    triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3; 
    triangleArray.indexStrideInBytes = sizeof(int) * 3;
    triangleArray.preTransform = 0;
    triangleArray.numSbtRecords = 1;
    triangleArray.flags = triangle_input_flags;

    cudaStream_t streamDefault  = 0;
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    std::cout << "Computing memory usage" << std::endl;
    OptixAccelBufferSizes bufferSizes = {};
    optixAccelComputeMemoryUsage(context, &accelOptions,
        &buildInput, 1, &bufferSizes);

    std::cout << "Allocating memory" << std::endl;
    CUdeviceptr d_output;
    CUdeviceptr d_temp;

    std::printf("output size: %llu\n", bufferSizes.outputSizeInBytes);
    std::printf("temp size: %llu\n", bufferSizes.tempSizeInBytes);
    cudaMalloc((void**)&d_output, bufferSizes.outputSizeInBytes);
    cudaMalloc((void**)&d_temp, bufferSizes.tempSizeInBytes);

    OptixTraversableHandle outputHandle = 0;
    std::cout << "Building acceleration structure" << std::endl;
    OptixResult results = optixAccelBuild(context, streamDefault,
     &accelOptions, &buildInput, 1, d_temp,
     bufferSizes.tempSizeInBytes, d_output,
     bufferSizes.outputSizeInBytes, &outputHandle, nullptr, 0);

    if (results == OPTIX_SUCCESS) {
        std::cout << "Successfully built acceleration structure" << std::endl;
    } else {
        std::cout << "Failed to build acceleration structure" << std::endl;
    }

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;

    // This option is important to ensure we compile code which is optimal
    // for our scene hierarchy. We use a single GAS – no instancing or
    // multi-level hierarchies
    pipeline_compile_options.traversableGraphFlags =
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    // Our device code uses 3 payload registers (r,g,b output value)
    pipeline_compile_options.numPayloadValues = 3;

    // This is the name of the param struct variable in our device code
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    
    std::string ptx = loadPtx("/home/teja/research/optix_splats/_skbuild/linux-x86_64-3.11/cmake-build/ptx/kernels.ptx");
    OptixModule module = nullptr;
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.numAttributeValues = 2; // 2 is the minimum
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    OPTIX_CHECK(optixModuleCreate(context, &module_compile_options,
                                        &pipeline_compile_options, ptx.c_str(),
                                        ptx.size(), nullptr, nullptr, &module));
    
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;

    OptixProgramGroupOptions program_group_options = {}; 
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    optixProgramGroupCreate(
        context,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        nullptr,
        nullptr,
        &raygen_prog_group );
    
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    optixProgramGroupCreate(
        context,
        &miss_prog_group_desc,
        1, // num program groups
        &program_group_options,
        nullptr,
        nullptr,
        &miss_prog_group );
    
    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    // We could also specify an any-hit and/or intersection program here
    optixProgramGroupCreate(
        context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        nullptr,
        nullptr,
        &hitgroup_prog_group );
    OptixProgramGroup program_groups[] = 
    { 
        raygen_prog_group, 
        miss_prog_group, 
        hitgroup_prog_group
    };
    
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;

    OptixPipeline pipeline = nullptr;
    optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof( program_groups ) / sizeof( program_groups[0] ),
        nullptr,
        nullptr,
        &pipeline );
    
    // Allocate the miss record on the device 
    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof( MissSbtRecord );
    cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size );

    CUdeviceptr raygen_record;
    size_t raygen_record_size = sizeof( RayGenSbtRecord );
    cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size );

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof( HitGroupSbtRecord );
    cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size );

    // Populate host-side copy of the record with header and data
    MissSbtRecord ms_sbt;
    ms_sbt.data.bg_color = { 0.3f, 0.1f, 0.2f };
    optixSbtRecordPackHeader( miss_prog_group, &ms_sbt );

    RayGenSbtRecord rg_sbt;
    optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt );

    HitGroupSbtRecord hg_sbt;
    optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt );

    // Now copy our host record to the device
    cudaMemcpy(
        reinterpret_cast<void*>( miss_record ),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice );
    
    cudaMemcpy(
        reinterpret_cast<void*>( raygen_record ),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice );
    
    cudaMemcpy(
        reinterpret_cast<void*>( hitgroup_record ),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice );

    
    
    // The shader binding table struct we will populate
    OptixShaderBindingTable sbt = {};

    // Finally we specify how many records and how they are packed in memory
    sbt.raygenRecord  = raygen_record;
    sbt.missRecordBase  = miss_record;
    sbt.missRecordStrideInBytes = sizeof( MissSbtRecord ); 
    sbt.missRecordCount  = 1;
    sbt.hitgroupRecordBase  = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    sbt.hitgroupRecordCount  = 1;
    
    Params params;
    params.image_width = image_width;
    params.image_height = image_height;

    CUdeviceptr d_image;
    cudaMalloc( reinterpret_cast<void**>( &d_image ),
        3 * image_width * image_height * sizeof( int ) );

    params.image = (int*)d_image;

    CUdeviceptr d_param;
    cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) );
    cudaMemcpy( reinterpret_cast<void*>( d_param ),
        &params, sizeof( params ),
        cudaMemcpyHostToDevice );
    
    CUstream stream = 0;
    CUDA_CHECK( cudaStreamCreate( &stream ) ); // 0 is the default stream
    OPTIX_CHECK(optixLaunch( pipeline, 
      stream,   // Default CUDA stream
      d_param,
      sizeof( Params ), 
      &sbt,
      image_width,
      image_height,
      1 ));
    
    cudaDeviceSynchronize();
    std::vector<int> im_host(3 * image_width * image_height, 0);
    cudaMemcpy( im_host.data(), (void*)d_image, 3 * image_width * image_height * sizeof( int ), cudaMemcpyDeviceToHost );
    std::printf("im_host[0, 1 , 2]: %d %d %d\n", im_host[0], im_host[1], im_host[2]);

    torch::Tensor image = torch::from_blob((void*)d_image, {image_height, image_width, 3}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    return image;
}

PYBIND11_MODULE(DiffGaussianRenderer, m) {
    m.def("render_gaussians", &render_gaussians, "Render gaussians");
}
