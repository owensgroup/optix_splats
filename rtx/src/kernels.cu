#include <optix.h>
#include <stdio.h>

// This is a struct used to communicate launch parameters which are constant
// for all threads in a given optixLaunch call. 
struct Params
{
    int*  image;
    unsigned int  image_width;
    unsigned int  image_height;
    float3   cam_eye;
    float3   cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};
struct RayGenData   {};
struct HitGroupData {};
struct MissData     { float3 bg_color;  };

// SBT record with an appropriately aligned and sized data block
template<typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT )
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


extern "C" 
{
__constant__ Params params;
}

//__forceinline__ __device__ float dequantizeUnsigned8Bits( const unsigned char i )
//{
//    enum { N = (1 << 8) - 1 };
//    return min((float)i / (float)N), 1.f)
//}
// __forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x )
// {
//     x = clamp( x, 0.0f, 1.0f );
//     enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
//     return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
// }

// __forceinline__ __device__ uchar4 make_color( const float3& c )
// {
//     // first apply gamma, then convert to unsigned char
//     float3 srgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
//     return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
// }

// __forceinline__ __device__ uchar4 make_color( const float4& c )
// {
//     return make_color( make_float3( c.x, c.y, c.z ) );
// }


// Note the __raygen__ prefix which marks this as a ray-generation
// program function
extern "C" __global__ void __raygen__rg() 
{
    // // Lookup our location within the launch grid
    // const uint3 idx = optixGetLaunchIndex();
    // const uint3 dim = optixGetLaunchDimensions();
    // params.image[3*(dim.y * params.image_width) + dim.x] = 255;// result.x;
    // params.image[3*(dim.y * params.image_width) + dim.x + 1] = 255; //result.y;
    // params.image[3*(dim.y * params.image_width) + dim.x + 2] = 0;
 
    // // Map our launch idx to a screen location and create a ray from 
    // // the camera location through the screen
    // float3 ray_origin, ray_direction;
    // //computeRay( idx, dim, ray_origin, ray_direction );
    // ray_origin = make_float3(__int_as_float(dim.x), __int_as_float(dim.y), -5.0f);
    // ray_direction = make_float3(0.0f, 0.0f, 1.0f);

 
    // // Trace the ray against our scene hierarchy
    // unsigned int p0, p1, p2;
    // optixTrace(
    //     params.handle,
    //     ray_origin,
    //     ray_direction,
    //     0.0f,   // Min intersection distance
    //     1e16f,  // Max intersection distance
    //     0.0f,   // ray-time -- used for motion blur
    //     OptixVisibilityMask( 255 ), // Specify always visible
    //     OPTIX_RAY_FLAG_NONE,
    //     0,      // SBT offset -- See SBT discussion
    //     0,      // SBT stride -- See SBT discussion 
    //     0,      // missSBTIndex -- See SBT discussion
    //     p0, p1, p2 ); // These 32b values are the ray payload
 
    // // Our results were packed into opaque 32b registers
    // float3 result;
    // result.x = int_as_float( p0 );
    // result.y = int_as_float( p1 );
    // result.z = int_as_float( p2 );
    

    // Record results in our output raster
     // result.z;
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();

        // const float2 d = 2.0f * make_float2(
        //     static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
        //     static_cast<float>( idx.y ) / static_cast<float>( dim.y )
        //     ) - 1.0f;

    int result = INT_MAX;

    params.image[idx.y * params.image_width + idx.x] = result;
}

extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental 
    // attributes are provided by the OptiX API, including barycentric 
    // coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();
 
    // Convert to color and assign to our payload outputs.
    const float3 c = make_float3( barycentrics.x, barycentrics.y, 1.0f ); 
    optixSetPayload_0( __float_as_int( c.x ) );
    optixSetPayload_1( __float_as_int( c.y ) );
    optixSetPayload_2( __float_as_int( c.z ) );
}

extern "C" __global__ void __miss__ms()
{
    MissData* miss_data = 
    reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    optixSetPayload_0( __float_as_int( miss_data->bg_color.x ) );
    optixSetPayload_1( __float_as_int( miss_data->bg_color.y ) );
    optixSetPayload_2( __float_as_int( miss_data->bg_color.z ) );
}