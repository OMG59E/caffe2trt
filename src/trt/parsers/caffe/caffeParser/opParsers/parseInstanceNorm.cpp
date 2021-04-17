//
// Created by xingwg on 21-4-2.
//

#include "trt/parsers/caffe/caffeParser/opParsers/opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1 {
    ILayer* parseInstanceNorm(INetworkDefinition& network, const trtcaffe::LayerParameter& msg,
            CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors) {

        if (!checkBlobs(msg, 1, 1))
            return nullptr;

        const trtcaffe::InstanceNormParameter& p = msg.instance_normalize_param();

        const float eps = p.eps();
        std::vector<Weights> weights;
        weights = weightFactory.getAllWeights(msg.name());
        Weights scales = weights[0];
        Weights bias = weights[1];
        weightFactory.convert(scales);
        weightFactory.convert(bias);

        auto layer_reduce1 = network.addReduce(*tensors[msg.bottom(0)], ReduceOperation::kAVG, 6, true);
        auto layer_elwise1 = network.addElementWise(*tensors[msg.bottom(0)], *layer_reduce1->getOutput(0), ElementWiseOperation::kSUB);

        const static float val1[3]{0.0f, 1.0f, 2.0f};
        Weights shift1{DataType::kFLOAT, val1 + 0, 1};
        Weights scale1{DataType::kFLOAT, val1 + 1, 1};
        Weights power1{DataType::kFLOAT, val1 + 2, 1};
        weightFactory.convert(shift1);
        weightFactory.convert(scale1);
        weightFactory.convert(power1);

        auto layer_scale1 = network.addScale(*layer_elwise1->getOutput(0), ScaleMode::kUNIFORM, shift1, scale1, power1);
        auto layer_reduce2 = network.addReduce(*layer_scale1->getOutput(0), ReduceOperation::kAVG, 6, true);

        const static float val2[3]{eps, 1.0f, 0.5f};
        Weights shift2{DataType::kFLOAT, val2 + 0, 1};
        Weights scale2{DataType::kFLOAT, val2 + 1, 1};
        Weights power2{DataType::kFLOAT, val2 + 2, 1};
        weightFactory.convert(shift2);
        weightFactory.convert(scale2);
        weightFactory.convert(power2);

        auto layer_scale2 = network.addScale(*layer_reduce2->getOutput(0), ScaleMode::kUNIFORM, shift2, scale2, power2);
        auto layer_elwise2 = network.addElementWise(*layer_elwise1->getOutput(0), *layer_scale2->getOutput(0), ElementWiseOperation::kDIV);

        auto val3 = reinterpret_cast<float*>(malloc(sizeof(float) * scales.count));
        std::fill_n(val3, scales.count, 1.0f);
        Weights power3{DataType::kFLOAT, val3, scales.count};
        weightFactory.convert(power3);

        auto layer_scale3 = network.addScale(*layer_elwise2->getOutput(0), ScaleMode::kCHANNEL, bias, scales, power3);

        return layer_scale3;
    }
} //namespace nvcaffeparser1

