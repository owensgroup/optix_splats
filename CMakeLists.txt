cmake_minimum_required(VERSION 3.12)

project(diff_gaussian_renderer LANGUAGES C CXX CUDA)
find_package(PythonExtensions REQUIRED)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(TORCH_CUDA_ARCH_LIST="7.0")
find_package(Torch REQUIRED)
set(CMAKE_CUDA_FLAGS "")
set(CMAKE_CUDA_ARCHITECTURES ON)

set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

set(CMAKE_CUDA_ARCHITECTURES 70)
find_package(OptiX8 REQUIRED)


set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

get_target_property(TORCH_INTERFACE_LIB torch INTERFACE_LINK_LIBRARIES)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

include_directories(
    ${OptiX8_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include(PTXUtilities)


set(SOURCES
    diff_gaussian_renderer/renderer.cu
    )


add_library(DiffGaussianRenderer MODULE ${SOURCES})
add_ptx_targets(DiffGaussianRenderer kernels)
python_extension_module(DiffGaussianRenderer)
# set_target_properties(saxpy_ext PROPERTIES
#                             CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(saxpy_ext PROPERTIES CUDA_ARCHITECTURES 70)

target_link_libraries(DiffGaussianRenderer cuda 
    ${OptiX_LIBRARY}
    ${CUDA_LIBRARIES}   
    ${CMAKE_DL_LIBS}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY})
target_include_directories(DiffGaussianRenderer PRIVATE ${TORCH_INCLUDE_DIRS})

target_compile_options(DiffGaussianRenderer PRIVATE -Wall)

install(TARGETS DiffGaussianRenderer LIBRARY DESTINATION diff_gaussian_renderer)


