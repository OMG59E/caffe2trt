/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>

#include "google/protobuf/text_format.h"

#include "trt/parsers/caffe/caffeMacros.h"
#include "trt/parsers/caffe/caffeParser/caffeParser.h"
#include "trt/parsers/caffe/caffeParser/opParsers/opParsers.h"
#include "trt/parsers/common/parserUtils.h"
#include "trt/parsers/caffe/caffeParser/readProto.h"
#include "trt/parsers/caffe/binaryProtoBlob.h"
#include "trt/parsers/common/half.h"
#include "error_check.h"
#include <NvInferPluginUtils.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;

CaffeParser::~CaffeParser() {
    for (auto v : mTmpAllocs) {
        free(v);
    }
    for (auto p : mNewPlugins) {
        if (p) {
            p->destroy();
        }
    }
    delete mBlobNameToTensor;
}

std::vector<nvinfer1::PluginField> CaffeParser::parseSliceParam(
        const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors) {
    std::vector<nvinfer1::PluginField> f;
    const trtcaffe::SliceParameter& p = msg.slice_param();

    auto* axis = allocMemory<int32_t>();  // default = 1
    *axis = p.axis();
    f.emplace_back("axis", axis, PluginFieldType::kINT32, 1);

    auto* slice_dim = allocMemory<int32_t>();  // default = 1
    *slice_dim = p.slice_dim();
    f.emplace_back("slice_dim", slice_dim, PluginFieldType::kINT32, 1);

    int slice_point_size = p.slice_point_size();

    auto* slice_point = allocMemory<int32_t>(slice_point_size);
    for (int i = 0; i < slice_point_size; ++i)
        slice_point[i] = p.slice_point(i);

    f.emplace_back("slice_point", slice_point, PluginFieldType::kINT32, slice_point_size);

    nvinfer1::DimsCHW inputShape = parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions());

    auto* channel_in = allocMemory<int32_t>();
    *channel_in = inputShape.c();
    f.emplace_back("channel_in", channel_in, PluginFieldType::kINT32, 1);

    auto* height_in = allocMemory<int32_t>();
    *height_in = inputShape.h();
    f.emplace_back("height_in", height_in, PluginFieldType::kINT32, 1);

    auto* width_in = allocMemory<int32_t>();
    *width_in = inputShape.w();
    f.emplace_back("width_in", width_in, PluginFieldType::kINT32, 1);

    return f;
}


const IBlobNameToTensor* CaffeParser::parseBuffers(const char* deployBuffer,
                                                   std::size_t deployLength,
                                                   const char* modelBuffer,
                                                   std::size_t modelLength,
                                                   INetworkDefinition& network,
                                                   DataType weightType) {
    mDeploy = std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
    google::protobuf::io::ArrayInputStream deployStream(deployBuffer, deployLength);
    if (!google::protobuf::TextFormat::Parse(&deployStream, mDeploy.get())) {
        RETURN_AND_LOG_ERROR(nullptr, "Could not parse deploy file");
    }

    if (modelBuffer) {
        mModel = std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
        google::protobuf::io::ArrayInputStream modelStream(modelBuffer, modelLength);
        google::protobuf::io::CodedInputStream codedModelStream(&modelStream);
        codedModelStream.SetTotalBytesLimit(modelLength, -1);

        if (!mModel->ParseFromCodedStream(&codedModelStream)) {
            RETURN_AND_LOG_ERROR(nullptr, "Could not parse model file");
        }
    }

    return parse(network, weightType, modelBuffer != nullptr);
}

const IBlobNameToTensor* CaffeParser::parse(const char* deployFile,
                                            const char* modelFile,
                                            INetworkDefinition& network,
                                            DataType weightType) {
    CHECK_NULL_RET_NULL(deployFile)

    // this is used to deal with dropout layers which have different input and output
    mModel = std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
    if (modelFile && !readBinaryProto(mModel.get(), modelFile, mProtobufBufferSize)) {
        RETURN_AND_LOG_ERROR(nullptr, "Could not parse model file");
    }

    mDeploy = std::unique_ptr<trtcaffe::NetParameter>(new trtcaffe::NetParameter);
    if (!readTextProto(mDeploy.get(), deployFile)) {
        RETURN_AND_LOG_ERROR(nullptr, "Could not parse deploy file");
    }

    return parse(network, weightType, modelFile != nullptr);
}

const IBlobNameToTensor* CaffeParser::parse(INetworkDefinition& network, DataType weightType, bool hasModel) {
    bool ok = true;
    CaffeWeightFactory weights(*mModel.get(), weightType, mTmpAllocs, hasModel);
    mBlobNameToTensor = new (BlobNameToTensor);
    // Get list of all available plugin creators
    int numCreators = 0;
    nvinfer1::IPluginCreator* const* tmpList = getPluginRegistry()->getPluginCreatorList(&numCreators);
    for (int k = 0; k < numCreators; ++k) {
        if (!tmpList[k]) {
            LOG(WARNING) << "Plugin Creator for plugin " << k << " is a nullptr.";
            continue;
        }
        std::string pluginName = tmpList[k]->getPluginName();
        mPluginRegistry[pluginName] = tmpList[k];
    }

    for (int i = 0; i < mDeploy->input_size(); i++) {
        Dims dims{0, 0, 0};
        if (network.hasImplicitBatchDimension()) {
            if (mDeploy->input_shape_size()) {
                dims = DimsCHW{(int) mDeploy->input_shape().Get(i).dim().Get(1),
                               (int) mDeploy->input_shape().Get(i).dim().Get(2),
                               (int) mDeploy->input_shape().Get(i).dim().Get(3)};
            } else {
                // Deprecated, but still used in a lot of networks
                dims = DimsCHW{(int) mDeploy->input_dim().Get(i * 4 + 1),
                               (int) mDeploy->input_dim().Get(i * 4 + 2),
                               (int) mDeploy->input_dim().Get(i * 4 + 3)};
            }
        } else {
            LOG(WARNING) << "Setting batch size to 1. "
                            "Update the dimension after parsing due to using explicit batch size." ;
            if (mDeploy->input_shape_size()) {
                dims = DimsNCHW{
                        1,
                        (int) mDeploy->input_shape().Get(i).dim().Get(1),
                        (int) mDeploy->input_shape().Get(i).dim().Get(2),
                        (int) mDeploy->input_shape().Get(i).dim().Get(3)};
            } else {
                // Deprecated, but still used in a lot of networks
                dims = DimsNCHW{
                        1,
                        (int) mDeploy->input_dim().Get(i * 4 + 1),
                        (int) mDeploy->input_dim().Get(i * 4 + 2),
                        (int) mDeploy->input_dim().Get(i * 4 + 3)};
            }
        }
        ITensor* tensor = network.addInput(mDeploy->input().Get(i).c_str(), DataType::kFLOAT, dims);
        (*mBlobNameToTensor)[mDeploy->input().Get(i)] = tensor;
    }

    for (int i = 0; i < mDeploy->layer_size() && ok; i++) {
        const trtcaffe::LayerParameter& layerMsg = mDeploy->layer(i);
        if (layerMsg.has_phase() && layerMsg.phase() == trtcaffe::TEST)
            continue;

        // If there is a inplace operation and the operation is
        // modifying the input, emit an error as
        for (int j = 0; ok && j < layerMsg.top_size(); ++j) {
            for (int k = 0; ok && k < layerMsg.bottom_size(); ++k) {
                if (layerMsg.top().Get(j) == layerMsg.bottom().Get(k)) {
                    auto iter = mBlobNameToTensor->find(layerMsg.top().Get(j).c_str());
                    if (iter != nullptr && iter->isNetworkInput()) {
                        ok = false;
                        LOG(WARNING) << "TensorRT does not support in-place operations on "
                                        "input tensors in a prototxt file.";
                    }
                }
            }
        }

        // If there is a pluginFactory provided, use layer name matching to handle the plugin construction
        if (mPluginFactory && mPluginFactory->isPlugin(layerMsg.name().c_str())) {
            std::vector<Weights> w = weights.getAllWeights(layerMsg.name());
            IPlugin* plugin = mPluginFactory->createPlugin(
                    layerMsg.name().c_str(), w.empty() ? nullptr : &w[0], w.size());
            std::vector<ITensor*> inputs;
            for (int k = 0, n = layerMsg.bottom_size(); k < n; ++k)
                inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(k)]);

            bool isExt = mPluginFactoryIsExt
                         && static_cast<IPluginFactoryExt*>(mPluginFactory)->isPluginExt(layerMsg.name().c_str());

            ILayer* layer = isExt
                    ? network.addPluginExt(&inputs[0], int(inputs.size()), *static_cast<IPluginExt*>(plugin))
                    : network.addPlugin(&inputs[0], int(inputs.size()), *plugin);

            if (!layer) {
                LOG(ERROR) << "error parsing layer type " << layerMsg.type() << " index " << i;
                ok = false;
            }

            layer->setName(layerMsg.name().c_str());

            if (plugin->getNbOutputs() != layerMsg.top_size()) {
                LOG(ERROR) << " type: " << layerMsg.type()
                          << " plugin: " << plugin->getNbOutputs()
                          << " caffe: " << layerMsg.top_size();
                LOG(ERROR) << "Plugin layer output count is not equal to caffe output count.";
                ok = false;
            }

            for (int k = 0, n = std::min(layer->getNbOutputs(), layerMsg.top_size()); k < n; ++k)
                (*mBlobNameToTensor)[layerMsg.top(k)] = layer->getOutput(k);

            continue;
        }

        if (getInferLibVersion() >= 5000) {
            if (mPluginFactoryV2 && mPluginFactoryV2->isPluginV2(layerMsg.name().c_str())) {
                if (mPluginFactory)
                    RETURN_AND_LOG_ERROR(nullptr, "Both IPluginFactory and IPluginFactoryV2 are set. "
                                                  "If using TensorRT 5.0 or later, switch to IPluginFactoryV2");

                std::vector<Weights> w = weights.getAllWeights(layerMsg.name());

                nvinfer1::IPluginV2* plugin =
                        mPluginFactoryV2->createPlugin(layerMsg.name().c_str(), w.empty()
                        ? nullptr
                        : &w[0], w.size(), mPluginNamespace.c_str());

                std::vector<ITensor*> inputs;
                for (int k = 0, n = layerMsg.bottom_size(); k < n; ++k)
                    inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(k)]);

                ILayer* layer = network.addPluginV2(&inputs[0], int(inputs.size()), *plugin);
                if (!layer) {
                    LOG(ERROR) << "error parsing layer type " << layerMsg.type() << " index " << i;
                    ok = false;
                }

                layer->setName(layerMsg.name().c_str());
                if (plugin->getNbOutputs() != layerMsg.top_size()) {
                    LOG(ERROR) << " type: " << layerMsg.type()
                              << " plugin: " << plugin->getNbOutputs()
                              << " caffe: " << layerMsg.top_size();
                    LOG(ERROR) << "Plugin layer output count is not equal to caffe output count.";
                    ok = false;
                }

                for (int k = 0, n = std::min(layer->getNbOutputs(), layerMsg.top_size()); k < n; ++k)
                    (*mBlobNameToTensor)[layerMsg.top(k)] = layer->getOutput(k);

                continue;
            }

            std::string pluginName;
            nvinfer1::PluginFieldCollection fc;
            std::vector<nvinfer1::PluginField> f;
            if (layerMsg.type() == "Slice") {
                pluginName = "Slice_TRT";
                f = parseSliceParam(layerMsg, weights, *mBlobNameToTensor);
            }

            if (mPluginRegistry.find(pluginName) != mPluginRegistry.end()) {
                // Set fc
                fc.nbFields = f.size();
                fc.fields = f.empty() ? nullptr : f.data();
                nvinfer1::IPluginV2* pluginV2 = mPluginRegistry.at(pluginName)->createPlugin(layerMsg.name().c_str(), &fc);
                LOG_ASSERT(pluginV2);
                mNewPlugins.push_back(pluginV2);

                std::vector<ITensor*> inputs;
                for (int k = 0, n = layerMsg.bottom_size(); k < n; ++k)
                    inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(k)]);

                auto layer = network.addPluginV2(&inputs[0], int(inputs.size()), *pluginV2);
                if (!layer) {
                    LOG(ERROR) << "error parsing layer type " << layerMsg.type() << " index " << i;
                    ok = false;
                }
                layer->setName(layerMsg.name().c_str());
                if (pluginV2->getNbOutputs() != layerMsg.top_size()) {
                    LOG(ERROR) << " type: " << layerMsg.type()
                              << " plugin: " << pluginV2->getNbOutputs()
                              << " caffe: " << layerMsg.top_size();
                    LOG(ERROR) << "Plugin layer output count is not equal to caffe output count.";
                    ok = false;
                }

                for (int k = 0, n = std::min(layer->getNbOutputs(), layerMsg.top_size()); k < n; ++k)
                    (*mBlobNameToTensor)[layerMsg.top(k)] = layer->getOutput(k);

                continue;
            }

        }

        if (layerMsg.type() == "Dropout") {
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            continue;
        }

        if (layerMsg.type() == "ContinuationIndicator") {
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            continue;
        }

        if (layerMsg.type() == "Input") {
            const trtcaffe::InputParameter& p = layerMsg.input_param();
            for (int k = 0; k < layerMsg.top_size(); ++k) {
                const trtcaffe::BlobShape& shape = p.shape().Get(k);
                if (shape.dim_size() != 4) {
                    RETURN_AND_LOG_ERROR(nullptr,
                            "error parsing input layer, TensorRT only supports 4 dimensional input");
                } else {
                    Dims d;
                    if (network.hasImplicitBatchDimension()) {
                        d = DimsCHW{(int) shape.dim().Get(1),
                                    (int) shape.dim().Get(2),
                                    (int) shape.dim().Get(3)};
                    } else {
                        LOG(WARNING) << "Warning, setting batch size to 1. Update the dimension after parsing due to "
                                     "using explicit batch size.";
                        d = DimsNCHW{(int) shape.dim().Get(0),
                                     (int) shape.dim().Get(1),
                                     (int) shape.dim().Get(2),
                                     (int) shape.dim().Get(3)};
                    }
                    ITensor* tensor = network.addInput(layerMsg.top(k).c_str(), DataType::kFLOAT, d);
                    (*mBlobNameToTensor)[layerMsg.top().Get(k)] = tensor;
                }
            }
            continue;
        }

        //if (layerMsg.type() == "Flatten") {
        //    ITensor* tensor = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
        //    (*mBlobNameToTensor)[layerMsg.top().Get(0)] = tensor;
        //    std::cout << "Warning: Flatten layer ignored. TensorRT implicitly"
        //                 " flattens input to FullyConnected layers, but in other"
        //                 " circumstances this will result in undefined behavior."
        //              << std::endl;
        //    continue;
        //}

        // Use parser table to lookup the corresponding parse function to handle the rest of the layers
        auto v = gParseTable.find(layerMsg.type());

        if (v == gParseTable.end()) {
            LOG(ERROR) << "could not parse layer type " << layerMsg.type();
            ok = false;
        } else {
            ILayer* layer = (*v->second)(network, layerMsg, weights, *mBlobNameToTensor);
            if (!layer) {
                LOG(ERROR) << "error parsing layer type " << layerMsg.type() << " index " << i;
                ok = false;
            } else {
                layer->setName(layerMsg.name().c_str());
                (*mBlobNameToTensor)[layerMsg.top(0)] = layer->getOutput(0);
            }
        }
    }

    mBlobNameToTensor->setTensorNames();

    return ok && weights.isOK() && mBlobNameToTensor->isOK() ? mBlobNameToTensor : nullptr;
}

IBinaryProtoBlob* CaffeParser::parseBinaryProto(const char* fileName) {
    CHECK_NULL_RET_NULL(fileName)
    using namespace google::protobuf::io;

    std::ifstream stream(fileName, std::ios::in | std::ios::binary);
    if (!stream)
        RETURN_AND_LOG_ERROR(nullptr, "Could not open file " + std::string{fileName});

    IstreamInputStream rawInput(&stream);
    CodedInputStream codedInput(&rawInput);
    codedInput.SetTotalBytesLimit(INT_MAX, -1);

    trtcaffe::BlobProto blob;
    bool ok = blob.ParseFromCodedStream(&codedInput);
    stream.close();

    if (!ok)
        RETURN_AND_LOG_ERROR(nullptr, "parseBinaryProto: Could not parse mean file");

    DimsNCHW dims{1, 1, 1, 1};
    if (blob.has_shape()) {
        int size = blob.shape().dim_size(), s[4] = {1, 1, 1, 1};
        for (int i = 4 - size; i < 4; i++) {
            LOG_ASSERT(blob.shape().dim(i) < INT32_MAX);
            s[i] = static_cast<int>(blob.shape().dim(i));
        }
        dims = DimsNCHW{s[0], s[1], s[2], s[3]};
    } else {
        dims = DimsNCHW{blob.num(), blob.channels(), blob.height(), blob.width()};
    }

    const int dataSize = dims.n() * dims.c() * dims.h() * dims.w();
    LOG_ASSERT(dataSize > 0);

    const trtcaffe::Type blobProtoDataType = CaffeWeightFactory::getBlobProtoDataType(blob);
    const auto blobProtoData = CaffeWeightFactory::getBlobProtoData(blob, blobProtoDataType, mTmpAllocs);

    if (dataSize != (int) blobProtoData.second) {
        LOG(ERROR) << "CaffeParser::parseBinaryProto: blob dimensions don't match data size.";
        return nullptr;
    }

    const auto dataSizeBytes = dataSize * CaffeWeightFactory::sizeOfCaffeType(blobProtoDataType);
    void* memory = malloc(dataSizeBytes);
    memcpy(memory, blobProtoData.first, dataSizeBytes);
    return new BinaryProtoBlob(memory,
            blobProtoDataType == trtcaffe::FLOAT ? DataType::kFLOAT : DataType::kHALF, dims);
}
