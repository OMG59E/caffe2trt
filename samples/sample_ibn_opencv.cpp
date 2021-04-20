//
// Created by shiyy on 20-4-9.
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

bool accuracy(const std::vector<int>& v, int gt_idx, int k) {
    LOG_ASSERT(v.size() >= k);
    for (int n=0; n<k; ++n) {
        if (v[n] == gt_idx)
            return true;
    }
    return false;
}

template<typename T>
std::vector<int> sort_(T* data, size_t size) {
    std::vector <int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return idx;
}


int main(int argc, char **argv) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    const std::string img_dir = "../data/imagenet/";
    const std::string filepath = img_dir + "val_list.txt";
    const std::string engineFile = "../engines/resnet50_ibn_a-d9d0bb7b_opt.engine";
    const int device_id = 0;
    const int batch_size = 32;

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

    std::vector<alg::Mat> vNvImages;
    std::vector<alg::Tensor> vOutputTensors;
    NetOperator engine(engineFile.c_str(), param, device_id);

    vNvImages.clear();
    vNvImages.resize(batch_size);
    DimsNCHW shape = engine.getInputShape();
    for (auto &vImage : vNvImages)
        vImage.create(shape.c(), shape.h(), shape.w());

    //
    std::ifstream in(filepath);
    if (!in.is_open()) {
        LOG(ERROR) << "file open failed -> " << filepath;
        return -1;
    }
    std::string line, img_path, gt_idx;
    std::stringstream ss;
    cv::Mat img, img_resize, img_crop;
    int count = 0;
    int idx = 0;
    int top1 = 0, top5 = 0;
    while (getline(in, line)) {
        ss.clear();
        ss << line;
        ss >> img_path >> gt_idx;
        img = cv::imread(img_dir + img_path);
        if (img.empty()) {
            LOG(WARNING) << "load img failed -> " << img_dir + img_path;
            continue;
        }
        cv::resize(img, img_resize, cv::Size(256, 256));
        int x = int((256 - shape.w() + 1) * 0.5f);
        int y = int((256 - shape.h() + 1) * 0.5f);
        img_crop = img_resize(cv::Rect(x, y, shape.w(), shape.h())).clone();
        idx = count % batch_size;
        CUDACHECK(cudaMemcpy(vNvImages[idx].ptr(), img_crop.data, vNvImages[idx].size(), cudaMemcpyHostToDevice));
        if (idx == 0 && count != 0) {
            vOutputTensors.clear();
            if (!engine.inference(vNvImages, vOutputTensors)) {
                LOG(ERROR) << "inference failed";
                return -2;
            }
            LOG_ASSERT(vOutputTensors.size() == 1);
            const int num_classes = vOutputTensors[0].shape.c();
            for (int n=0; n<batch_size; ++n) {
                const float* output = vOutputTensors[0].data + n*num_classes;
                std::vector<int> v = sort_(output, num_classes);
                if (accuracy(v, std::stoi(gt_idx), 1)) top1++;
                if (accuracy(v, std::stoi(gt_idx), 5)) top5++;
            }
        }
        LOG(INFO) << "count: " << count << " idx: " << idx;
        count++;
    }

    //
    if (count % batch_size != 0) {
        vOutputTensors.clear();
        if (!engine.inference(vNvImages, vOutputTensors)) {
            LOG(ERROR) << "inference failed";
            return -2;
        }
        LOG_ASSERT(vOutputTensors.size() == 1);
        const int num_classes = vOutputTensors[0].shape.c();
        for (int n = 0; n <= idx; ++n) {
            const float *output = vOutputTensors[0].data + n * num_classes;
            std::vector<int> v = sort_(output, num_classes);
            if (accuracy(v, std::stoi(gt_idx), 1)) top1++;
            if (accuracy(v, std::stoi(gt_idx), 5)) top5++;
        }
    }
    LOG(INFO) << "top1/top5 " << float(top1) / count << "/" << float(top5) / count;
    return 0;
}