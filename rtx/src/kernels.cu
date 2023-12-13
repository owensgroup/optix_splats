#include <optix.h>
#include <stdio.h>

#include "sutil/vec_math.h"

// This is a struct used to communicate launch parameters which are constant
// for all threads in a given optixLaunch call. 
struct Params
{
    uchar4*  image;
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

__forceinline__ __device__ float3 toSRGB( const float3& c )
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}


__forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x )
{
    x = clamp( x, 0.0f, 1.0f );
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color( const float3& c )
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
}

__forceinline__ __device__ uchar4 make_color( const float4& c )
{
    return make_color( make_float3( c.x, c.y, c.z ) );
}

static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    origin    = params.cam_eye;
    direction = normalize( d.x * U + d.y * V + W );
}

// Note the __raygen__ prefix which marks this as a ray-generation
// program function
extern "C" __global__ void __raygen__rg() 
{
    // // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    // params.image[3*(dim.y * params.image_width) + dim.x] = 255;// result.x;
    // params.image[3*(dim.y * params.image_width) + dim.x + 1] = 255; //result.y;
    // params.image[3*(dim.y * params.image_width) + dim.x + 2] = 0;
 
    // // Map our launch idx to a screen location and create a ray from 
    // // the camera location through the screen
    float3 ray_origin, ray_direction;
    computeRay( idx, dim, ray_origin, ray_direction );

    //ray_origin = make_float3(__int_as_float(dim.x), __int_as_float(dim.y), -5.0f);
    //ray_direction = make_float3(0.0f, 0.0f, 1.0f);

    // // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2 );
 
    // // Our results were packed into opaque 32b registers
    float3 result;
    result.x = __int_as_float( p0 );
    result.y = __int_as_float( p1 );
    result.z = __int_as_float( p2 );

    params.image[idx.y * params.image_width + idx.x] = make_color(result);
}



__device__ bool in_ellipsoid(float3 intersect_world, float3 center, float3 scale, const float4* transform)
{
    // Translate the world point back to the origin
    float3 translated = intersect_world - center;

    // // Rotate the point
    // // Get the rotation matrix from the quaternion
    // float3 R0;
    // float3 R1;
    // float3 R2;

    // R0.x = 1.0f - 2.0f * (quaternion.y * quaternion.y + quaternion.z * quaternion.z);
    // R0.y = 2.0f * (quaternion.x * quaternion.y - quaternion.z * quaternion.w);
    // R0.z = 2.0f * (quaternion.x * quaternion.z + quaternion.y * quaternion.w);

    // R1.x = 2.0f * (quaternion.x * quaternion.y + quaternion.z * quaternion.w);
    // R1.y = 1.0f - 2.0f * (quaternion.x * quaternion.x + quaternion.z * quaternion.z);
    // R1.z = 2.0f * (quaternion.y * quaternion.z - quaternion.x * quaternion.w);

    // R2.x = 2.0f * (quaternion.x * quaternion.z - quaternion.y * quaternion.w);
    // R2.y = 2.0f * (quaternion.y * quaternion.z + quaternion.x * quaternion.w);
    // R2.z = 1.0f - 2.0f * (quaternion.x * quaternion.x + quaternion.y * quaternion.y);

    // float3 rotated = make_float3(
    //     R0.x * translated.x + R1.x * translated.y + R2.x * translated.z,
    //     R0.y * translated.x + R1.y * translated.y + R2.y * translated.z,
    //     R0.z * translated.x + R1.z * translated.y + R2.z * translated.z
    // );
    float3 rotated;
    rotated.x = transform[0].x * translated.x + transform[0].y * translated.y + transform[0].z * translated.z + transform[0].w;
    rotated.y = transform[1].x * translated.x + transform[1].y * translated.y + transform[1].z * translated.z + transform[1].w;
    rotated.z = transform[2].x * translated.x + transform[2].y * translated.y + transform[2].z * translated.z + transform[2].w;

    
    // Scale the point
    float3 scaled = make_float3(rotated.x / scale.x, rotated.y / scale.y, rotated.z / scale.z);

    // Check if the point lies within the unit sphere
    
    return (scaled.x * scaled.x + scaled.y * scaled.y + scaled.z * scaled.z) <= 1.0f;
}

extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental 
    // attributes are provided by the OptiX API, including barycentric 
    // coordinates.
    float raytime = optixGetRayTmax();
    float3 ray_direction = optixGetWorldRayDirection();
    float3 ray_origin = optixGetWorldRayOrigin();
    //printf("handle %d\n", params.handle); 

    //printf("transform[0] = [%f, %f, %f, %f]\n", transform[4], transform[5], transform[6], transform[7]);

    unsigned int n_transforms = optixGetTransformListSize();

    const float4* transform = nullptr;
    for (unsigned int i = 0; i < n_transforms; i++) {
        OptixTraversableHandle transform_handle = optixGetTransformListHandle( i );
        if( optixGetTransformTypeFromHandle( transform_handle )==OPTIX_TRANSFORM_TYPE_INSTANCE) {
            transform =
                optixGetInstanceTransformFromHandle( transform_handle );
        }
    }
    
    float3 intersection_point_w = raytime * ray_direction + ray_origin;
   
    float3 center = make_float3(0.0f, 0.0f, 0.0f);
    float3 scale = make_float3(0.5f, 0.5f, 0.5f);
    float4 quaternion = make_float4(0.3f, 0.5f, 0.2f, 1.0f);
    
    
    
    if (in_ellipsoid(intersection_point_w, center, scale, transform)) {
        optixSetPayload_0(__float_as_int(1.0f));
    } else {
        optixSetPayload_0(__float_as_int(0.3f));
    }
    optixSetPayload_1( __float_as_int( 0.0f ) );
    optixSetPayload_2( __float_as_int( 0.0f ) );
}

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}

extern "C" __global__ void __miss__ms()
{
    MissData* miss_data = 
    reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    // optixSetPayload_0( __float_as_int( miss_data->bg_color.x ) );
    // optixSetPayload_1( __float_as_int( miss_data->bg_color.y ) );
    // optixSetPayload_2( __float_as_int( miss_data->bg_color.z ) );
    setPayload(miss_data->bg_color);
}