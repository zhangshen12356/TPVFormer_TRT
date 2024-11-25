#include "model.h"
#include <sys/time.h>

int main()
{
    std::string img_path = "/home/zs/Code/Lane_detect_TensorRT/1697956246.png";
    std::string img_dir = "/home/zs/Code/TPVFormer_TRT/4509/";
    std::string onnx_path = "/home/zs/Code/TPVFormer_TRT/tpvformer_no_points_sim_change.onnx";
    std::string engine_path = "/home/zs/Code/TPVFormer_TRT/tpvformer_no_points_sim_change.engine";
    std::string points_path = "/home/zs/Code/TPVFormer_TRT/points_6.bin";
    std::string lidar2img_path = "/home/zs/Code/TPVFormer_TRT/lidar2img_6.bin";
    std::string img_shape_path = "/home/zs/Code/TPVFormer_TRT/img_shape_6.bin";

    Lane_Model model;
    model.Model_Init(engine_path, onnx_path);
    model.Inference(img_dir, points_path, lidar2img_path, img_shape_path);

    return 0;
}

// export LD_LIBRARY_PATH=/media/data_12T/zhangshen/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib
// CUDA_MODULE_LOADING=LAZY ./TPVFormer_TRT
// /media/data_12T/zhangshen/TensorRT-8.5.3.1/bin/trtexec --onnx=/home/zs/Code/TPVFormer_TRT/tpvformer_sim.onnx --saveEngine=/home/zs/Code/TPVFormer_TRT/tpvformer_sim.engine --fp16 --workspace=100000