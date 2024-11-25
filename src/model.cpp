#include "model.h"
#include <dlfcn.h>


Lane_Model::Lane_Model()
{
}

Lane_Model::~Lane_Model()
{
}

bool loadPluginLibraryAndRegisterPlugin(const std::string &libPath)
{
    void *handle = dlopen(libPath.c_str(), RTLD_LAZY);
    if (!handle)
    {
        std::cerr << "Failed to load plugin library: " << libPath << std::endl;
        return false;
    }
    // You can add additional checks or function calls here if the plugin library requires initialization.
    return true;
}
cv::Mat impadImg(const cv::Mat &img, int divisor)
{
    int pad_h = (img.rows + divisor - 1) / divisor * divisor;
    int pad_w = (img.cols + divisor - 1) / divisor * divisor;
    // std::cout<<"pad_h:"<<pad_h<<" pad_w:"<<pad_w<<std::endl;
    cv::Mat padded_img(pad_h, pad_w, img.type());

    // OpenCV's copyMakeBorder uses top, bottom, left, right order for padding
    int top = 0;
    int bottom = pad_h - img.rows;
    int left = 0;
    int right = pad_w - img.cols;

    cv::copyMakeBorder(img, padded_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
    return padded_img;
}
bool Lane_Model::Model_Init(std::string engine_path, std::string onnx_path)
{
    std::string pluginLibPath = "/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so"; // Update this with the actual path
    if (!loadPluginLibraryAndRegisterPlugin(pluginLibPath))
    {
        return false;
    }
    if (access(engine_path.c_str(), F_OK) == -1)
    {
        std::cout << "engine file does not exist, need to be created!" << std::endl;
        if (access(onnx_path.c_str(), F_OK) == -1)
        {
            std::cout << "onnx path does not exist, can't create a engine from it " << std::endl;
            return false;
        }
        else
        {
            // 1.创建构建器的实例
            nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
            // 2.创建网络定义
            uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            nvinfer1::INetworkDefinition *network = builder->createNetworkV2(flag);

            // -----------------------------------------------------------------------------------
            // auto creator = getPluginRegistry()->getPluginCreator("MMCVModulatedDeformConv2d", "1");
            // if (!creator)
            // {
            //     std::cerr << "Failed to find plugin creator." << std::endl;
            //     return EXIT_FAILURE;
            // }
            // -----------------------------------------------------------------------------------
            // 3.创建一个 ONNX 解析器来填充网络
            nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, gLogger);
            // 4.读取模型文件并处理任何错误
            parser->parseFromFile(onnx_path.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
            for (int32_t i = 0; i < parser->getNbErrors(); ++i)
            {
                std::cout << parser->getError(i)->desc() << std::endl;
                return false;
            }
            // 5.创建一个构建配置，指定 TensorRT 应该如何优化模型
            nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

            nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();

            profile->setDimensions("imgs", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{5, {1, 6, 3, 928, 1600}}); // 设置输入x的动态维度，最小值
            profile->setDimensions("imgs", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{5, {1, 6, 3, 928, 1600}}); // 期望输入的最优值
            profile->setDimensions("imgs", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{5, {1, 6, 3, 928, 1600}}); // 最大值

            // profile->setDimensions("points", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1000, 3));  // 设置输入x的动态维度，最小值
            // profile->setDimensions("points", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 34752, 3)); // 期望输入的最优值
            // profile->setDimensions("points", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 50000, 3)); // 最大值

            profile->setDimensions("lidar2img", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 6, 4, 4)); // 设置输入x的动态维度，最小值
            profile->setDimensions("lidar2img", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 6, 4, 4)); // 期望输入的最优值
            profile->setDimensions("lidar2img", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 6, 4, 4)); // 最大值

            profile->setDimensions("img_shape", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 6, 3)); // 设置输入x的动态维度，最小值
            profile->setDimensions("img_shape", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 6, 3)); // 期望输入的最优值
            profile->setDimensions("img_shape", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 6, 3)); // 最大值

            // builder->setMaxBatchSize(1);
            config->addOptimizationProfile(profile);

            // 7.指定配置后，构建引擎
            nvinfer1::IHostMemory *serializedModel = builder->buildSerializedNetwork(*network, *config);
            // 8.保存TensorRT模型
            std::ofstream p(engine_path, std::ios::binary);
            p.write(reinterpret_cast<const char *>(serializedModel->data()), serializedModel->size());
            // 9.序列化引擎包含权重的必要副本，因此不再需要解析器、网络定义、构建器配置和构建器，可以安全地删除
            delete parser;
            delete network;
            delete config;
            delete builder;
            // 10.将引擎保存到磁盘，并且可以删除它被序列化到的缓冲区
            delete serializedModel;
            std::cout << "create engine file success!" << std::endl;

            std::ifstream file(engine_path, std::ios::binary);
            if (!file.good())
            {
                std::cerr << "read " << engine_path << " error!" << std::endl;
                return false;
            }
            std::vector<char> data;
            try
            {
                file.seekg(0, file.end);
                const auto size = file.tellg();
                file.seekg(0, file.beg);
                data.resize(size);
                file.read(data.data(), size);
            }
            catch (const std::exception &e)
            {
                file.close();
                std::cerr << e.what() << '\n';
                return false;
            }
            file.close();
            runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
            assert(runtime != nullptr);
            engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.data(), data.size()));
            assert(engine != nullptr);
            context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
            assert(context != nullptr);

            data.clear();
            std::cout << "Load engine file success!" << std::endl;
            return true;
        }
    }
    else
    {
        std::cout << engine_path << " already exist!" << std::endl;
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good())
        {
            std::cerr << "read " << engine_path << " error!" << std::endl;
            return false;
        }
        std::vector<char> data;
        try
        {
            file.seekg(0, file.end);
            const auto size = file.tellg();
            file.seekg(0, file.beg);
            data.resize(size);
            file.read(data.data(), size);
        }
        catch (const std::exception &e)
        {
            file.close();
            std::cerr << e.what() << '\n';
            return false;
        }
        file.close();
        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        assert(runtime != nullptr);
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(data.data(), data.size()));
        assert(engine != nullptr);
        context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        assert(context != nullptr);

        data.clear();
        std::cout << "Load engine file success!!!" << std::endl;
        return true;
    }
}
void Lane_Model::blobFromImages(std::vector<cv::Mat> imgs, float *blob)
{
    for (int i = 0; i < images_N; i++)
    {
        cv::Mat img = imgs[i];
        // cv::imwrite("./" + std::to_string(i) + ".jpg", img);
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        int img_h = img.rows;
        int img_w = img.cols;
        for (int h = 0; h < img_h; h++)
        {
            for (int w = 0; w < img_w; w++)
            {
                blob[i * img_w * img_h * 3 + 0 * img_w * img_h + h * img_w + w] = ((float)img.at<cv::Vec3b>(h, w)[0] - mean_value[0]) / std_value[0];
                blob[i * img_w * img_h * 3 + 1 * img_w * img_h + h * img_w + w] = ((float)img.at<cv::Vec3b>(h, w)[1] - mean_value[1]) / std_value[1];
                blob[i * img_w * img_h * 3 + 2 * img_w * img_h + h * img_w + w] = ((float)img.at<cv::Vec3b>(h, w)[2] - mean_value[2]) / std_value[2];
                // std::cout<<blob[i * img_w * img_h * 3 + 0 * img_w * img_h + h * img_w + w]<<" --- "<<blob[i * img_w * img_h * 3 + 1 * img_w * img_h + h * img_w + w]<<" --- "<<blob[i * img_w * img_h * 3 + 2 * img_w * img_h + h * img_w + w]<<std::endl;
            }
        }
    }
}

std::vector<cv::Mat> load_imgs(std::string img_dir)
{

    cv::Mat img_front = cv::imread(img_dir + "n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984245412460.jpg");
    img_front = impadImg(img_front, 32);
    // cv::imwrite("./3.jpg", img_front);

    cv::Mat img_front_right = cv::imread(img_dir + "n015-2018-10-08-15-36-50+0800__CAM_FRONT_RIGHT__1538984245420339.jpg");
    img_front_right = impadImg(img_front_right, 32);

    cv::Mat img_front_left = cv::imread(img_dir + "n015-2018-10-08-15-36-50+0800__CAM_FRONT_LEFT__1538984245404844.jpg");
    img_front_left = impadImg(img_front_left, 32);

    cv::Mat img_back = cv::imread(img_dir + "n015-2018-10-08-15-36-50+0800__CAM_BACK__1538984245437525.jpg");
    img_back = impadImg(img_back, 32);

    cv::Mat img_back_left = cv::imread(img_dir + "n015-2018-10-08-15-36-50+0800__CAM_BACK_LEFT__1538984245447423.jpg");
    img_back_left = impadImg(img_back_left, 32);

    cv::Mat img_back_right = cv::imread(img_dir + "n015-2018-10-08-15-36-50+0800__CAM_BACK_RIGHT__1538984245427893.jpg");
    img_back_right = impadImg(img_back_right, 32);

    std::vector<cv::Mat> imgs{img_front, img_front_right, img_front_left, img_back, img_back_left, img_back_right};
    // std::vector<cv::Mat> imgs{img_front, img_front, img_front, img_front, img_front, img_front};
    return imgs;
}

std::vector<std::vector<float>> get_grid_coords(std::vector<int> dims, std::vector<float> resolution)
{
    std::vector<std::vector<float>> coords_grid;

    std::vector<int> g_xx(dims[0]);
    std::iota(g_xx.begin(), g_xx.end(), 0);

    std::vector<int> g_yy(dims[1]);
    std::iota(g_yy.begin(), g_yy.end(), 0);

    std::vector<int> g_zz(dims[2]);
    std::iota(g_zz.begin(), g_zz.end(), 0);

    for(int i = 0; i < g_yy.size(); i++)
    {
        for(int j = 0; j < g_xx.size(); j++)
        {
            for(int k = 0; k < g_zz.size(); k++)
            {
                std::vector<float> temp;
                temp.push_back(g_xx[j]* resolution[0] + resolution[0] / 2);
                temp.push_back(g_yy[i]* resolution[1] + resolution[1] / 2);
                temp.push_back(g_zz[k]* resolution[2] + resolution[2] / 2);
                coords_grid.push_back(temp);
            }
        }
    }

    return coords_grid;
}

void Lane_Model::Inference(std::string img_dir, std::string points_path, std::string lidar2img_path, std::string img_shape_path)
{
    std::vector<cv::Mat> imgs = load_imgs(img_dir);
    int nbBindings = engine->getNbBindings();
    void *buffers[nbBindings];
    std::cout << "nbBindings: " << nbBindings << std::endl;

    float *inBlob_img = new float[1 * images_N * 3 * resize_h_ * resize_w_];
    blobFromImages(imgs, inBlob_img);

    // for(int i = 0; i < 50; i++)
    // {
    //     std::cout<<"img:"<<inBlob_img[i]<<std::endl;
    // }
    // ----------------------Points File----------------------------
    std::ifstream points_file(points_path.c_str(), std::ios::binary);
    std::streampos points_size = points_file.tellg();
    points_file.seekg(0, std::ios::end);
    points_size = points_file.tellg() - points_size;
    int Num_Points = points_size / sizeof(float);
    std::cout << "Num_Points " << Num_Points << std::endl;
    float *points = new float[Num_Points];
    points_file.seekg(0, std::ios::beg);
    for (int i = 0; i < Num_Points; ++i)
    {
        points_file.read(reinterpret_cast<char *>(&points[i]), sizeof(float));
        // if(i < 50) std::cout<<points[i] << std::endl;
    }
    points_file.close();
    // --------------------------------------------------
    std::ifstream lidar2img_file(lidar2img_path.c_str(), std::ios::binary);
    std::streampos size = lidar2img_file.tellg();
    lidar2img_file.seekg(0, std::ios::end);
    size = lidar2img_file.tellg() - size;
    int num_lidar2img = size / sizeof(float);
    std::cout << "num_lidar2img " << num_lidar2img << std::endl;
    float *lidar2img = new float[num_lidar2img];
    lidar2img_file.seekg(0, std::ios::beg);
    for (int i = 0; i < num_lidar2img; ++i)
    {
        lidar2img_file.read(reinterpret_cast<char *>(&lidar2img[i]), sizeof(float));
        // if(i < 50) std::cout<<lidar2img[i] << std::endl;
    }
    lidar2img_file.close();
    // --------------------------------------------------------------
    std::ifstream img_shape_file(img_shape_path.c_str(), std::ios::binary);
    std::streampos img_shape_size = img_shape_file.tellg();
    img_shape_file.seekg(0, std::ios::end);
    img_shape_size = img_shape_file.tellg() - img_shape_size;
    int numFloats = img_shape_size / sizeof(float);
    std::cout << "num_img_shape " << numFloats << std::endl;
    float *img_shape = new float[numFloats];
    img_shape_file.seekg(0, std::ios::beg);
    for (int i = 0; i < numFloats; ++i)
    {
        img_shape_file.read(reinterpret_cast<char *>(&img_shape[i]), sizeof(float));
        // if (i < 50) std::cout << img_shape[i] << std::endl;
    }
    img_shape_file.close();
    // ---------------------------------------------------------------
    int inputIndex1 = engine->getBindingIndex(INPUT_BLOB_NAME1);
    // int inputIndex2 = engine->getBindingIndex(INPUT_BLOB_NAME2);
    int inputIndex3 = engine->getBindingIndex(INPUT_BLOB_NAME3);
    int inputIndex4 = engine->getBindingIndex(INPUT_BLOB_NAME4);

    int outputIndex1 = engine->getBindingIndex(OUTPUT_BLOB_NAME1);
    // int outputIndex2 = engine->getBindingIndex(OUTPUT_BLOB_NAME2);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex1], 3 * images_N * resize_h_ * resize_w_ * sizeof(float)));
    // CHECK(cudaMalloc(&buffers[inputIndex2], Num_Points * sizeof(float)));
    CHECK(cudaMalloc(&buffers[inputIndex3], images_N * 4 * 4 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[inputIndex4], 3 * images_N * sizeof(float)));

    context->setBindingDimensions(inputIndex1, nvinfer1::Dims{5, {1, images_N, 3, resize_h_, resize_w_}}); // 根据输入图像大小更新输入维度
    // context->setBindingDimensions(inputIndex2, nvinfer1::Dims3(1, Num_Points / 3, 3));                     // 根据输入图像大小更新输入维度
    context->setBindingDimensions(inputIndex3, nvinfer1::Dims4(1, images_N, 4, 4));                        // 根据输入图像大小更新输入维度
    context->setBindingDimensions(inputIndex4, nvinfer1::Dims3(1, images_N, 3));                           // 根据输入图像大小更新输入维度

    // context->setBindingDimensions(outputIndex1, nvinfer1::Dims{5, {1, 18, 100, 100, 8}}); // 根据输入图像大小更新输入维度
    auto out_dims1 = context->getBindingDimensions(outputIndex1);
    int output_size1 = 1;
    for (int j = 0; j < out_dims1.nbDims; j++)
    {
        output_size1 *= out_dims1.d[j];
    }
    float *outBlob1 = new float[output_size1];
    std::cout << "output_size1:" << output_size1 << std::endl;

    // auto out_dims2 = context->getBindingDimensions(outputIndex2);
    // int output_size2 = 1;
    // for (int j = 0; j < out_dims2.nbDims; j++)
    // {
    //     output_size2 *= out_dims2.d[j];
    // }
    // float *outBlob2 = new float[output_size2];
    // std::cout << "output_size2:" << output_size2 << std::endl;

    CHECK(cudaMalloc(&buffers[outputIndex1], output_size1 * sizeof(float)));
    // CHECK(cudaMalloc(&buffers[outputIndex2], output_size2 * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex1], inBlob_img, 3 * images_N * resize_h_ * resize_w_ * sizeof(float), cudaMemcpyHostToDevice, stream));
    // CHECK(cudaMemcpyAsync(buffers[inputIndex2], points, Num_Points * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex3], lidar2img, images_N * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex4], img_shape, 3 * images_N * sizeof(float), cudaMemcpyHostToDevice, stream));

    // do inference
    context->enqueueV2(buffers, stream, nullptr);

    // copy data to cpu from device
    CHECK(cudaMemcpyAsync(outBlob1, buffers[outputIndex1], output_size1 * sizeof(float), cudaMemcpyDeviceToHost, stream)); // torch.Size([1, 18, 100, 100, 8])
    // CHECK(cudaMemcpyAsync(outBlob2, buffers[outputIndex2], output_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream)); // torch.Size([1, 18, 34752, 1, 1])

    cudaStreamSynchronize(stream);
    // Release stream and buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex1]);
    // cudaFree(buffers[inputIndex2]);
    cudaFree(buffers[inputIndex3]);
    cudaFree(buffers[inputIndex4]);
    cudaFree(buffers[outputIndex1]);
    // cudaFree(buffers[outputIndex2]);

    for (int i = 0; i < 10; i++)
    {
        std::cout << "outBlob1[" << std::to_string(i) << "]:" << outBlob1[i] << std::endl; // if ok outBlob1[0]=44.8085
    }
    std::cout << "-------------------------------------" << std::endl;
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << "outBlob2[" << std::to_string(i) << "]:" << outBlob2[i] << std::endl;
    // }

    std::vector<VoxelResult> Voxels;
    Voxels.clear();
    for (int z = 0; z < output_z_; z++)
    {
        for (int h = 0; h < output_h_; h++)
        {
            for (int w = 0; w < output_w_; w++)
            {

                double score = -9999999;
                VoxelResult voxel;
                voxel.x = w;
                voxel.y = h;
                for (int c = 0; c < num_classes_; c++)
                {
                    if (score < outBlob1[output_h_ * output_w_ * output_z_ * c + z * output_h_ * output_w_ + h * output_w_ + w])
                    {
                        score = outBlob1[output_h_ * output_w_ * output_z_ * c + z * output_h_ * output_w_ + h * output_w_ + w];
                        voxel.classId = c;
                    }
                }
                Voxels.push_back(voxel);
            }
        }
    }

    // std::vector<int> points_result(Num_Points / 3);
    // for (int n = 0; n < Num_Points / 3; n++)
    // {
    //     double score = -999999;
    //     for (int c = 0; c < num_classes_; c++)
    //     {
    //         if (score < outBlob2[(Num_Points / 3) * c + n])
    //         {
    //             score = outBlob2[(Num_Points / 3) * c + n];
    //             points_result[n] = c;
    //         }
    //     }
    // }

    std::vector<float> voxel_origin{-51.2, -51.2, -5};
    std::vector<float> resolution{1.024, 1.024, 1.0};
    std::vector<int> dims{output_h_, output_w_, output_z_};
    std::vector<std::vector<float>> coords_grid = get_grid_coords(dims, resolution);
    // for(int i = 0; i < 30; i++)
    // {
    //     std::cout<<std::to_string(i)<<":"<<coords_grid[i][0]<<" "<<coords_grid[i][1]<<" "<<coords_grid[i][2]<<std::endl;
    // }
    for (int i = 0; i < coords_grid.size(); i++)
    {
        // std::cout<<coords_grid[i][0]<<" "<<coords_grid[i][1]<<" "<<coords_grid[i][2]<<std::endl;
        coords_grid[i][0] += voxel_origin[0];
        coords_grid[i][1] += voxel_origin[1];
        coords_grid[i][2] += voxel_origin[2];
    }
    std::cout<<"coords_grid.size:"<<coords_grid.size()<<" Voxels.size():"<<Voxels.size()<<std::endl;
    std::vector<VoxelResult> fov_voxels;
    fov_voxels.clear();
    for(int i = 0; i < coords_grid.size(); i++)
    {
        if(Voxels[i].classId==17)
        {
            Voxels[i].classId = 20;
        }
        if(Voxels[i].classId > 0 && Voxels[i].classId < 20)
        {
            VoxelResult temp_point;
            temp_point.x = coords_grid[i][1];
            temp_point.y = coords_grid[i][0];
            temp_point.z = coords_grid[i][2];
            temp_point.classId = Voxels[i].classId;
            fov_voxels.push_back(temp_point);
        }
    }

    std::ofstream outfile("../fov_voxels.txt");
    if(!outfile.is_open())
    {
        std::cerr << "can not open txt file!"<<std::endl;
        return;
    }

    for(auto voxel:fov_voxels)
    {
        outfile<<voxel.x<<" "<<voxel.y<<" "<<voxel.z<<" "<<voxel.classId<<std::endl;
    }
    outfile.close();
    

    delete[] inBlob_img;
    delete[] points;
    delete[] lidar2img;
    delete[] img_shape;
    delete[] outBlob1;
    // delete[] outBlob2;
}

// export LD_LIBRARY_PATH=/home/zs/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib/
