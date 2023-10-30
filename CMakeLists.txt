cmake_minimum_required(VERSION 3.12)
include(FetchContent)
project(optx_nerf LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

set(CMAKE_CUDA_ARCHITECTURES 70)
find_package(OptiX8 REQUIRED)
include_directories(
    ${OptiX8_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include(PTXUtilities)

set(SOURCES
    main.cu
    saxpy.cu)


add_executable(optx_splat ${SOURCES})
set_target_properties(optx_splat PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(optx_splat PROPERTIES CUDA_ARCHITECTURES 70)

target_link_libraries(optx_splat cuda 
    ${OptiX_LIBRARY}
    ${CUDA_LIBRARIES}   
    ${CMAKE_DL_LIBS})

add_ptx_targets(optx_splat kernels)