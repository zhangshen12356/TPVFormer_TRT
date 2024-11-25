#ifndef MODEL_H
#define MODEL_H

#include "NvInfer.h"
#include "logging.h"
#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include <vector>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include "trt_modulated_deform_conv.hpp"
#include "logging.h"

#include <iostream>
#include <numeric>
#include <cstdio>

struct VoxelResult
{
    float x;
    float y;
    float z;
    int classId;
};

class Lane_Model
{
public:
    Lane_Model();
    bool Model_Init(std::string engine_path, std::string onnx_path);
    ~Lane_Model();
    // std::vector<cv::Mat> load_imgs(std::string img_dir);
    void Inference(std::string img_dir, std::string points_path, std::string lidar2img_path, std::string img_shape_path);
    // void PreProcess(cv::Mat &img, std::vector<float> mean, std::vector<float> scale, bool is_scale);
    void PostProcess(cv::Mat &image, float *outBlob);
    // void RGB2CHW(const cv::Mat *im, float *data);
    void blobFromImages(std::vector<cv::Mat> imgs, float *blob);

private:
    std::vector<float> mean_value = {103.530f, 116.280f, 123.675f};
    std::vector<float> std_value = {1.0f, 1.0f, 1.0f};
  
    const int origin_h_ = 900;
    const int origin_w_ = 1600;
    
    const int resize_h_ = 928;
    const int resize_w_ = 1600;

    const int output_h_ = 100;
    const int output_w_ = 100;
    const int output_z_ = 8;

    const int images_N = 6;

    const int num_classes_ = 18;
    Logger gLogger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::shared_ptr<nvinfer1::IExecutionContext> context;

    const char *INPUT_BLOB_NAME1 = "imgs";
    // const char *INPUT_BLOB_NAME2 = "points";
    const char *INPUT_BLOB_NAME3 = "lidar2img";
    const char *INPUT_BLOB_NAME4 = "img_shape";

    const char *OUTPUT_BLOB_NAME1 = "output1";
    // const char *OUTPUT_BLOB_NAME2 = "output2";

    const int BATCH_SIZE = 1;
};

#endif