file(REMOVE_RECURSE
  "../../../../lib/libmmdeploy_tensorrt_ops.pdb"
  "../../../../lib/libmmdeploy_tensorrt_ops.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA CXX)
  include(CMakeFiles/mmdeploy_tensorrt_ops.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
