#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_types.h>
#include <optix_host.h>

#include <glad/gl.h>
#include <SDL.h>
#include <SDL_opengl.h>

#include "sutil/Camera.h"
#include "sutil/Trackball.h"
#include "sutil/GLDisplay.h"

#include <cuda_gl_interop.h>

// Create SDL2 + OpenGL context

struct Params
{
    uchar4*  image;
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

class OptixState {
public:

    // Constructors
    OptixState() {
        std::cout << "Creating optix context" << std::endl;
        init_context();

        std::cout << "Opening SDL2 Window" << std::endl;
        init_sdl2();

        std::cout << "Building gas" << std::endl;
        build_gas();

        std::cout << "Building ias" << std::endl;
        build_ias();

        std::cout << "Building module" << std::endl;
        build_module();

        std::cout << "Building pipeline and sbt" << std::endl;
        // TODO: Change image width and height to be passed in
        build_pipeline_and_sbt(2560, 1440);
    }

    // Destructors
    // ~OptixState() {
    //     cudaFree((void*)d_output);
    //     cudaFree((void*)d2_output);
    // }

    // Functions
    // TODO: Refactor to build gaussians around multiple gaussians
    void build_ias() {
        OptixInstance instance = {};
        float transform[12] = {0.7071,-0.7071,0.0,0,0.7071,0.7071,0.0 ,0,0.0,0.0,1.0,0};
        memcpy( instance.transform, transform, sizeof( float )*12 );
        instance.instanceId = 0;
        instance.visibilityMask = 255;
        instance.sbtOffset = 0;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = gas_handle;

        CUdeviceptr d_instance;
        cudaMalloc( (void**) &d_instance, sizeof( OptixInstance ) );
        cudaMemcpy( (void*)d_instance, &instance, 
            sizeof( OptixInstance ), cudaMemcpyHostToDevice );

        // Reset build input and bufferSizes
        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        OptixBuildInputInstanceArray& instanceArray = buildInput.instanceArray;
        instanceArray.instances = d_instance;
        instanceArray.numInstances = 1;
        
        cudaStream_t streamDefault  = 0;
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        std::cout << "Computing memory usage" << std::endl;
        OptixAccelBufferSizes bufferSizes = {};
        optixAccelComputeMemoryUsage(context, &accelOptions,
            &buildInput, 1, &bufferSizes);

        std::cout << "Allocating memory" << std::endl;
        
        CUdeviceptr d2_temp;

        std::printf("output size: %llu\n", bufferSizes.outputSizeInBytes);
        std::printf("temp size: %llu\n", bufferSizes.tempSizeInBytes);
        cudaMalloc((void**)&d2_output, bufferSizes.outputSizeInBytes);
        cudaMalloc((void**)&d2_temp, bufferSizes.tempSizeInBytes);

        OptixTraversableHandle inHandle = 1;

        std::cout << "Building instance acceleration structure" << std::endl;
        // Build GAS Timer
        std::chrono::high_resolution_clock::time_point build_ias_start = std::chrono::high_resolution_clock::now();
        OptixResult results = optixAccelBuild(context, streamDefault,
            &accelOptions, &buildInput, 1, d2_temp,
            bufferSizes.tempSizeInBytes, d2_output,
            bufferSizes.outputSizeInBytes, &inHandle, nullptr, 0);
        std::chrono::high_resolution_clock::time_point build_ias_end = std::chrono::high_resolution_clock::now();
        std::cout << "Build ias time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(build_ias_end - build_ias_start).count() << " ms" << std::endl;
        instanceHandle = inHandle;
        cudaFree((void*)d2_temp);
    }

    // Variables
    OptixDeviceContext context;

    OptixTraversableHandle gas_handle;
    CUdeviceptr d_output;
    OptixTraversableHandle instanceHandle;
    CUdeviceptr d2_output;

    OptixModule module;
    OptixPipelineCompileOptions pipeline_compile_options;
    OptixShaderBindingTable sbt;

    OptixPipeline pipeline;

    CUdeviceptr d_image;

    int image_width;
    int image_height;

    SDL_Window* window;
    GLuint pbo;
    cudaGraphicsResource* pbo_cuda;
    sutil::Camera cam;
    sutil::Trackball trackball;

private:
    void init_context() {
        context = createOptixContext();
    }

    void build_gas() {
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
        
        CUdeviceptr d_temp;

        std::printf("output size: %llu\n", bufferSizes.outputSizeInBytes);
        std::printf("temp size: %llu\n", bufferSizes.tempSizeInBytes);
        cudaMalloc((void**)&d_output, bufferSizes.outputSizeInBytes);
        cudaMalloc((void**)&d_temp, bufferSizes.tempSizeInBytes);

        OptixTraversableHandle outputHandle = 1;
        std::cout << "Building gas acceleration structure" << std::endl;
        // Build GAS Timer
        std::chrono::high_resolution_clock::time_point build_gas_start = std::chrono::high_resolution_clock::now();
        OptixResult results = optixAccelBuild(context, streamDefault,
            &accelOptions, &buildInput, 1, d_temp,
            bufferSizes.tempSizeInBytes, d_output,
            bufferSizes.outputSizeInBytes, &outputHandle, nullptr, 0);
        std::chrono::high_resolution_clock::time_point build_gas_end = std::chrono::high_resolution_clock::now();
        std::cout << "Build gas time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(build_gas_end - build_gas_start).count() << " ms" << std::endl;

        if (results == OPTIX_SUCCESS) {
            std::cout << "Successfully built acceleration structure" << std::endl;
        } else {
            std::cout << "Failed to build acceleration structure" << std::endl;
        }

        gas_handle = outputHandle;
        cudaFree((void*)d_temp);
    }

    void build_module() {
        
        pipeline_compile_options.usesMotionBlur = false;

        // This option is important to ensure we compile code which is optimal
        // for our scene hierarchy. We use a single GAS – no instancing or
        // multi-level hierarchies
        //pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

        // Our device code uses 3 payload registers (r,g,b output value)
        pipeline_compile_options.numPayloadValues = 3;

        // This is the name of the param struct variable in our device code
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        
        std::string ptx = loadPtx("/home/teja/research/optix_splats/_skbuild/linux-x86_64-3.11/cmake-build/ptx/kernels.ptx");
        module = nullptr;
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount =
            OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues = 3;
        pipeline_compile_options.numAttributeValues = 2; // 2 is the minimum
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        optixModuleCreate(context, &module_compile_options,
                                            &pipeline_compile_options, ptx.c_str(),
                                            ptx.size(), nullptr, nullptr, &module);
    }

    void build_pipeline_and_sbt(int image_width, int image_height) {
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

        pipeline = nullptr;
        optixPipelineCreate(
            context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof( program_groups ) / sizeof( program_groups[0] ),
            nullptr,
            nullptr,
            &pipeline );
        
        
        
        
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
        sbt = {};

        // Finally we specify how many records and how they are packed in memory
        sbt.raygenRecord  = raygen_record;
        sbt.missRecordBase  = miss_record;
        sbt.missRecordStrideInBytes = sizeof( MissSbtRecord ); 
        sbt.missRecordCount  = 1;
        sbt.hitgroupRecordBase  = hitgroup_record;
        sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
        sbt.hitgroupRecordCount  = 1;
        
        cudaMalloc( reinterpret_cast<void**>( &d_image ),
            image_width * image_height * sizeof( uchar4 ) );

        cam.setEye( {camera_x, camera_y, camera_z} );
        cam.setLookat( {lookat_x, lookat_y, lookat_z} );
        cam.setUp( {up_x, up_y, up_z} );
        cam.setFovY( 45.0f );
        cam.setAspectRatio( (float)image_width / (float)image_height );

        trackball.set_camera(cam);
    }
};

void init_sdl2(OptixState& state) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    state.window = SDL_CreateWindow( "Splats",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        2560, 1440,
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN
    );

    state.ogl_context = SDL_GL_CreateContext(window);

    int ogl_version = gladLoadGL((GLADloadfunc) SDL_GL_GetProcAddress);
    printf("OpenGL %d.%d\n", GLAD_VERSION_MAJOR(ogl_version), GLAD_VERSION_MINOR(ogl_version));

    // Setup the OpenGL image
    GL_CHECK( glGenBuffers( 1, &state.pbo ) );
    GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, state.pbo ) );
    GL_CHECK( glBufferData( GL_ARRAY_BUFFER, sizeof(uchar4) * state.image_width * state.image_height,
                            nullptr, GL_STREAM_DRAW ) );
    GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) );
    cudaGraphicsGLRegisterBuffer(&state.pbo_cuda, state.pbo, cudaGraphicsMapFlagsWriteDiscard);
}

// Returns true while SDL window is still running.
bool handle_input(OptixState& state) {
    bool running = true;
    SDL_Event event;
    while(SDL_PollEvent(&event)) {
        if (event.type == SDL_WINDOWEVENT) {
            switch(event.window_event) {
            case SDL_WINDOWEVENT_CLOSE:
                window_closed = false;
            }
        }
        if (event.type == SDL_MOUSEMOTION) {
            if (event.state & SDL_BUTTON_LMASK) { // L mouse pressed
                state.trackball.update_tracking(event.x, event.y, state.image_width, state.image_height);
            }
        }
        if (event.type == SDL_MOUSEBUTTONDOWN) {
            if (event.button == SDL_BUTTON_LEFT) {
                state.trackball.start_tracking(event.x, event.y);
            }
        }
    }

    return window_closed;
}


torch::Tensor render_gaussians(OptixState& state) {
    // std::printf("Making image tensor height %d width %d\n", image_height, image_width);
    // create torch tensor with size of image_height x image_width x 3 
    
    int image_width =state.image_width;
    int image_height = state.image_height;
    OptixDeviceContext context = state.context;

    OptixTraversableHandle instanceHandle = state.instanceHandle;
    OptixModule module = state.module;
    
    Params params;
    params.image_width = image_width;
    params.image_height = image_height;
    params.cam_eye      = state.cam.eye();
    params.handle = state.instanceHandle;
    state.cam.UVWFrame( params.cam_u, params.cam_v, params.cam_w );

    cudaGraphicsMapResources(1, &state.pbo_cuda, 0);

    size_t num_bytes_cuda;
    cudaGraphicsResourceGetMappedPointer((void**)&params.image, &num_bytes_cuda, state.pbo_cuda);
    // std::printf("%d bytes accessible in OpenGL buffer of %d expected\n", num_bytes_cuda, image_width * image_height * sizeof(uchar4));

    CUdeviceptr d_param;
    cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) );
    cudaMemcpy( reinterpret_cast<void*>( d_param ),
        &params, sizeof( params ),
        cudaMemcpyHostToDevice );
    
    CUstream stream = 0;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    std::chrono::high_resolution_clock::time_point launch_start = std::chrono::high_resolution_clock::now();

    OPTIX_CHECK(optixLaunch( state.pipeline, 
      stream,   // Default CUDA stream
      d_param,
      sizeof( Params ), 
      &state.sbt,
      image_width,
      image_height,
      1 ));

    torch::Tensor image = torch::zeroes({image_height, image_width}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    cudaMemcpy(image.data_ptr(), params.image, sizeof(uchar4) * image_height * image_width);

    cudaGraphicsUnmapResources(1, &state.pbo_cuda, 0);

    state.display.display(image_width, image_height, image_width, image_height, state.pbo);

    // Stop timer
    std::chrono::high_resolution_clock::time_point launch_end = std::chrono::high_resolution_clock::now();

    cudaFree( (void*)d_param );
    // Compute the difference between the two times in milliseconds
    auto launch_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(launch_end - launch_start).count();
    
    return image;
}

PYBIND11_MODULE(DiffGaussianRenderer, m) {
    py::class_<OptixState>(m, "OptixState")
        .def(py::init<>());

    m.def("render_gaussians", &render_gaussians, "Render gaussians");
    m.def("handle_input", &handle_input, "Handle SDL2 Input");
}
