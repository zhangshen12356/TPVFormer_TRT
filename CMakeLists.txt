cmake_minimum_required(VERSION 3.0.2)

project(TPVFormer)

add_definitions(-std=c++17 -O2 -pthread)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

find_package(VTK COMPONENTS
  vtkCommonColor
  vtkCommonCore
  vtkFiltersSources
  vtkInteractionStyle
  vtkRenderingContextOpenGL2
  vtkRenderingCore
  vtkRenderingFreeType
  vtkRenderingGL2PSOpenGL2
  vtkRenderingOpenGL2
)

include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        /home/zs/Code/TPVFormer_TRT/plugs
        ${VTK_INCLUDE_DIRS}
)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/media/data_12T/zhangshen/TensorRT-8.6.1.6/include)
link_directories(/media/data_12T/zhangshen/TensorRT-8.6.1.6/lib)
link_directories(/media/data_12T/zhangshen/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib)

# yourplugin为你编译出的动态库名，自己修改
# cuda_add_library(MMCVModulatedDeformConv2d SHARED ${PROJECT_SOURCE_DIR}/plugs/trt_modulated_deform_conv.cpp)
# target_link_libraries(MMCVModulatedDeformConv2d nvinfer cudart)

add_executable(TPVFormer_TRT 
                src/model.cpp
                main.cpp)

add_library(TPVFormer SHARED 
        src/model.cpp 
            )

target_link_libraries(TPVFormer_TRT
        nvinfer 
        nvinfer_plugin 
        # MMCVModulatedDeformConv2d
        nvparsers 
        nvonnxparser 
        cudart 
        cudnn
        ${CUDA_cublas_LIBRARY}
        # /home/zs/Code/TPVFormer_TRT/plugs/build/libMMCVModulatedDeformConv2d.so
        /home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so
        dl
        ${OpenCV_LIBS} 
        ${VTK_LIBRARIES}
)
# vtk_module_autoinit(
#   TARGETS TPVFormer_TRT
#   MODULES ${VTK_LIBRARIES}
# )