//
// Created by xingwg on 21-4-20.
//
#include "trt/net_operator.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>

using namespace alg::trt;
using namespace alg;

bool accuracy(const std::vector<int> &v, int gt_idx, int k) {
    LOG_ASSERT(v.size() >= k);
    for (int n = 0; n < k; ++n) {
        if (v[n] == gt_idx)
            return true;
    }
    return false;
}

template<typename T>
std::vector<int> sort_(T *data, size_t size) {
    std::vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return idx;
}

template<typename T>
DimsNCHW read(const char* filepath, T* data, int64_t* label) {
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        LOG(FATAL) << "file open failed -> " << filepath;
    }
    int d[4];
    size_t shapeSize = fread(d, sizeof(int), 4, file);
    LOG_ASSERT(shapeSize == 4);
    size_t labelSize = fread(label, sizeof(int64_t), d[0], file);
    LOG_ASSERT(labelSize == d[0]);
    auto size = d[0]*d[1]*d[2]*d[3];
    size_t dataSize = fread(data, sizeof(T), size, file);
    LOG_ASSERT(dataSize == size);
    fclose(file);
    return DimsNCHW{d[0], d[1], d[2], d[3]};
}

int main(int argc, char **argv) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    const std::string data_dir = "../data/imagenet/batches/";
    const std::string filepath = "../data/imagenet/batch_list.txt";;
    const std::string engineFile = "../engines/resnet50_ibn_a-d9d0bb7b_opt.engine";
    const int device_id = 0;
    const int batch_size = 32;

    NetParameter param;
    param.input_shape = DimsNCHW{batch_size, 3, 224, 224};
    param.input_node_name = "data";
    param.output_node_names = {"583"};

    std::vector<alg::Tensor> vOutputTensors;
    NetOperator engine(engineFile.c_str(), param, device_id);

    DimsNCHW shape = engine.getInputShape();

    float *batch_data{nullptr};
    CUDACHECK(cudaMallocHost((void**)&batch_data,
            shape.n() * shape.c() * shape.h() * shape.w() * sizeof(float)));
    int64_t *label_data{nullptr};
    CUDACHECK(cudaMallocHost((void**)&label_data, shape.n() * sizeof(int64_t)));

    //
    std::ifstream in(filepath);
    if (!in.is_open()) {
        LOG(ERROR) << "file open failed -> " << filepath;
        return -1;
    }
    std::string batch_file;
    int count = 0;
    int top1 = 0, top5 = 0;
    while (getline(in, batch_file)) {
        DimsNCHW data_shape = read((data_dir + batch_file).c_str(), batch_data, label_data);
        LOG_ASSERT(data_shape.n() <= batch_size && data_shape.n() > 0);
        vOutputTensors.clear();
        if (!engine.inference(batch_data, data_shape.n(), vOutputTensors)) {
            LOG(ERROR) << "inference failed";
            return -2;
        }
        LOG_ASSERT(vOutputTensors.size() == 1);
        const int num_classes = vOutputTensors[0].shape.c();
        for (int n = 0; n < data_shape.n(); ++n) {
            const float *output = vOutputTensors[0].data + n * num_classes;
            std::vector<int> v = sort_(output, num_classes);
            if (accuracy(v, label_data[n], 1)) top1++;
            if (accuracy(v, label_data[n], 5)) top5++;
        }
        count += data_shape.n();
    }
    LOG(INFO) << "top1/top5 " << float(top1) / count << "/" << float(top5) / count;

    CUDACHECK(cudaFreeHost(batch_data));
    CUDACHECK(cudaFreeHost(label_data));

    return 0;
}