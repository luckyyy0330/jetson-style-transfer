cmake_minimum_required(VERSION 3.18)
project(style_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 87)  # Orin (sm_87)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找 pybind11
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(style_cuda
    src/bindings.cpp
    src/prepost.cu
)

target_include_directories(style_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(style_cuda PRIVATE
    CUDA::cudart
)

# 输出到 app/cuda_ext/
set_target_properties(style_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../app/cuda_ext
)
