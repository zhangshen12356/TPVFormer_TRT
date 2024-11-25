cmake_minimum_required(VERSION 3.0.2)

project(trt_plugs)

add_definitions(-std=c++17 -O2 -pthread)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

# set(CUB_ROOT_DIR "${CMAKE_SOURCE_DIR}/cub")
find_package(CUDA REQUIRED)
# cuda和tensorrt的include和link目录, 如果你的不同需要自己修改
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/media/data_12T/zhangshen/TensorRT-8.6.1.6/include)
link_directories(/media/data_12T/zhangshen/TensorRT-8.6.1.6/lib)

# include_directories(CUB_ROOT_DIR)
# include_directories(${PROJECT_SOURCE_DIR}/*.hpp ${PROJECT_SOURCE_DIR}/*.cuh)

# yourplugin为你编译出的动态库名，自己修改
# cuda_add_library(MMCVModulatedDeformConv2d SHARED  ${PROJECT_SOURCE_DIR}/trt_modulated_deform_conv_kernel.cu)
file(GLOB_RECURSE TENSORRT_OPS_SRCS ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.cu)
add_library(MMCVModulatedDeformConv2d SHARED ${TENSORRT_OPS_SRCS})

# cuda_add_library(MMCVModulatedDeformConv2d SHARED ${TENSORRT_OPS_SRCS}/common_cuda_helper.cuh ${TENSORRT_OPS_SRCS}/modulated_deform_conv_cuda.cuh
# ${TENSORRT_OPS_SRCS}/trt_modulated_deform_conv_kernel.cu ${TENSORRT_OPS_SRCS}/trt_modulated_deform_conv.cpp)
# target_link_libraries(MMCVModulatedDeformConv2d nvinfer cudart cublas cudnn)
target_link_libraries(MMCVModulatedDeformConv2d nvinfer ${CUDA_cublas_LIBRARY} cudart cudnn)
# target_link_libraries(MMCVModulatedDeformConv2d nvinfer cudart cudnn)


