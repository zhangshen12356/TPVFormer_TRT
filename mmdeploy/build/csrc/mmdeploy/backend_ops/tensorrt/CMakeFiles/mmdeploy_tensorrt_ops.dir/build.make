# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zs/Code/TPVFormer_TRT/mmdeploy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zs/Code/TPVFormer_TRT/mmdeploy/build

# Include any dependencies generated for this target.
include csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/depend.make

# Include the progress variables for this target.
include csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/progress.make

# Include the compile flags for this target's objects.
include csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/flags.make

# Object files for target mmdeploy_tensorrt_ops
mmdeploy_tensorrt_ops_OBJECTS =

# External object files for target mmdeploy_tensorrt_ops
mmdeploy_tensorrt_ops_EXTERNAL_OBJECTS = \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/batched_bev_nms/trt_batched_bev_nms.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/batched_nms/trt_batched_nms.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/batched_rotated_nms/trt_batched_rotated_nms.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/bicubic_interpolate/trt_bicubic_interpolate.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/bicubic_interpolate/trt_bicubic_interpolate_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/allClassNMS.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/allClassRotatedNMS.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/batched_nms_kernel.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/gatherNMSOutputs.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/permuteData.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/sortScoresPerClass.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/sortScoresPerImage.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/trt_cuda_helper.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/deform_conv/trt_deform_conv.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/deform_conv/trt_deform_conv_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/gather_topk/gather_topk.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/gather_topk/gather_topk_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/grid_priors/trt_grid_priors.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/grid_priors/trt_grid_priors_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/grid_sampler/trt_grid_sampler.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/grid_sampler/trt_grid_sampler_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/instance_norm/trt_instance_norm.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/modulated_deform_conv/trt_modulated_deform_conv.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/modulated_deform_conv/trt_modulated_deform_conv_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/multi_level_roi_align/trt_multi_level_roi_align.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/multi_level_roi_align/trt_multi_level_roi_align_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/multi_level_rotated_roi_align/trt_multi_level_rotated_roi_align.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/multi_level_rotated_roi_align/trt_multi_level_rotated_roi_align_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/roi_align/trt_roi_align.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/roi_align/trt_roi_align_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/scaled_dot_product_attention/scaled_dot_product_attention.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/scaled_dot_product_attention/scaled_dot_product_attention_kernel.cu.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/scatternd/trt_scatternd.cpp.o" \
"/home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/scatternd/trt_scatternd_kernel.cu.o"

lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/batched_bev_nms/trt_batched_bev_nms.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/batched_nms/trt_batched_nms.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/batched_rotated_nms/trt_batched_rotated_nms.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/bicubic_interpolate/trt_bicubic_interpolate.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/bicubic_interpolate/trt_bicubic_interpolate_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/allClassNMS.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/allClassRotatedNMS.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/batched_nms_kernel.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/gatherNMSOutputs.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/permuteData.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/sortScoresPerClass.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/nms/sortScoresPerImage.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/common_impl/trt_cuda_helper.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/deform_conv/trt_deform_conv.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/deform_conv/trt_deform_conv_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/gather_topk/gather_topk.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/gather_topk/gather_topk_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/grid_priors/trt_grid_priors.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/grid_priors/trt_grid_priors_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/grid_sampler/trt_grid_sampler.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/grid_sampler/trt_grid_sampler_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/instance_norm/trt_instance_norm.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/modulated_deform_conv/trt_modulated_deform_conv.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/modulated_deform_conv/trt_modulated_deform_conv_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/multi_level_roi_align/trt_multi_level_roi_align.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/multi_level_roi_align/trt_multi_level_roi_align_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/multi_level_rotated_roi_align/trt_multi_level_rotated_roi_align.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/multi_level_rotated_roi_align/trt_multi_level_rotated_roi_align_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/roi_align/trt_roi_align.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/roi_align/trt_roi_align_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/scaled_dot_product_attention/scaled_dot_product_attention.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/scaled_dot_product_attention/scaled_dot_product_attention_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/scatternd/trt_scatternd.cpp.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops_obj.dir/scatternd/trt_scatternd_kernel.cu.o
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/build.make
lib/libmmdeploy_tensorrt_ops.so: /media/data_12T/zhangshen/TensorRT-8.6.1.6/lib/libnvinfer.so
lib/libmmdeploy_tensorrt_ops.so: /media/data_12T/zhangshen/TensorRT-8.6.1.6/lib/libnvinfer_plugin.so
lib/libmmdeploy_tensorrt_ops.so: /usr/lib/x86_64-linux-gnu/libcudnn.so
lib/libmmdeploy_tensorrt_ops.so: csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zs/Code/TPVFormer_TRT/mmdeploy/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX shared module ../../../../lib/libmmdeploy_tensorrt_ops.so"
	cd /home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mmdeploy_tensorrt_ops.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/build: lib/libmmdeploy_tensorrt_ops.so

.PHONY : csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/build

csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/clean:
	cd /home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt && $(CMAKE_COMMAND) -P CMakeFiles/mmdeploy_tensorrt_ops.dir/cmake_clean.cmake
.PHONY : csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/clean

csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/depend:
	cd /home/zs/Code/TPVFormer_TRT/mmdeploy/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zs/Code/TPVFormer_TRT/mmdeploy /home/zs/Code/TPVFormer_TRT/mmdeploy/csrc/mmdeploy/backend_ops/tensorrt /home/zs/Code/TPVFormer_TRT/mmdeploy/build /home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt /home/zs/Code/TPVFormer_TRT/mmdeploy/build/csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : csrc/mmdeploy/backend_ops/tensorrt/CMakeFiles/mmdeploy_tensorrt_ops.dir/depend

