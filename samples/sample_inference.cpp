//
// Created by xingwg on 21-4-17.
//

#include "trt/net_operator.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <utils/imdecode.h>

using namespace alg::trt;
using namespace alg;
using namespace std::chrono;

int main(int argc, char **argv) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    if (argc != 6) {
        LOG(ERROR) << "input param error, argc must be equal 6";
        return -1;
    }

    const char* imgFile = argv[1];
    const char *engineFile = argv[2];
    const int device_id = std::stoi(argv[3]);
    const int batch_size = std::stoi(argv[4]);
    const int iterations = std::stoi(argv[5]);

    NetParameter param;
    param.input_shape = DimsNCHW{batch_size, 3, 224, 224};
    param.input_node_name = "data";
    param.output_node_names = {"583"};
    param.mean_val[0] = 123.675f;
    param.mean_val[1] = 116.28f;
    param.mean_val[2] = 103.53f;
    param.scale[0] = 0.017125f;
    param.scale[1] = 0.017507f;
    param.scale[2] = 0.017429f;
    param.color_mode = "RGB";
    param.resize_mode = "v1";

    NetOperator engine(engineFile, param, device_id);

    std::vector<alg::Mat> vNvImages;
    vNvImages.clear();
    vNvImages.resize(batch_size);

    alg::nv::ImageDecoder dec;
    alg::Mat img;
    img.create(MAX_FRAME_SIZE);
    if (dec.Decode(imgFile, img) != 0) {
        LOG(ERROR) << "load img failed -> " << imgFile;
        return -1;
    }

    for (auto &nv_image : vNvImages) {
        nv_image.create(img.c(), img.h(), img.w());
        CUDACHECK(cudaMemcpy(nv_image.ptr(), img.data, img.size(), cudaMemcpyDeviceToDevice));
    }

    std::vector<Tensor> vOutputTensors;
    duration<float, std::milli> time_span{0};
    for (int i=0; i<iterations; ++i) {
        vOutputTensors.clear();
        auto t_start = std::chrono::system_clock::now();
        engine.inference(vNvImages, vOutputTensors);
        time_span += high_resolution_clock::now() - t_start;
    }

    engine.printLayerTimes(iterations);

    LOG(INFO) << iterations << " iteration average time " << time_span.count() / iterations << "ms" << std::endl;
}