# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# compile CUDA with /usr/local/cuda/bin/nvcc
# compile CXX with /usr/bin/g++-9
CUDA_FLAGS =  -Xcompiler=-fPIC,-Wall,-fvisibility=hidden -Xcompiler=-fno-gnu-unique  -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_87,code=sm_87 -O3 -Xcompiler=-fPIC   -std=c++14

CUDA_DEFINES = -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=1

CUDA_INCLUDES = -I/home/zs/Code/TPVFormer_TRT/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/../common -I/home/zs/Code/TPVFormer_TRT/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/common -isystem=/media/data_12T/zhangshen/TensorRT-8.6.1.6/include 

CXX_FLAGS =  -DMMDEPLOY_USE_CUDA=1 -O3 -fPIC   -std=gnu++14

CXX_DEFINES = -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT=1

CXX_INCLUDES = -I/home/zs/Code/TPVFormer_TRT/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/../common -I/home/zs/Code/TPVFormer_TRT/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt/common -I/usr/local/cuda/include -isystem /media/data_12T/zhangshen/TensorRT-8.6.1.6/include 

