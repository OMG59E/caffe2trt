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
#ifndef TRT_INSTANCE_NORMALIZATION_PLUGIN_H
#define TRT_INSTANCE_NORMALIZATION_PLUGIN_H

#include "trt/plugin/plugin.h"
#include <cuda_fp16.h>
#include <cudnn.h>
#include <vector>
#include <iostream>
#include <string>

namespace nvinfer1 {
    namespace plugin {
        class InstanceNormalizationPlugin : public IPluginV2IOExt {

        public:
            //InstanceNormalizationPlugin(float epsilon, nvinfer1::Weights const &scale, nvinfer1::Weights const &bias);

            InstanceNormalizationPlugin(float epsilon, const DimsCHW &input_shape,
                                        const DataType &dataType, std::vector<float> const &h_scale,
                                        std::vector<float> const &h_bias);

            InstanceNormalizationPlugin(float epsilon,
                                        const DataType &dataType,
                                        std::vector<float> const &h_scale,
                                        std::vector<float> const &h_bias);

            InstanceNormalizationPlugin(void const *serialData, size_t serialLength);

            InstanceNormalizationPlugin() = delete;

            ~InstanceNormalizationPlugin() override;

            int getNbOutputs() const override;

            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

            int initialize() override;

            void terminate() override;

            size_t getWorkspaceSize(int maxBatchSize) const override;

            int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace,
                    cudaStream_t stream) override;

            size_t getSerializationSize() const override;

            void serialize(void *buffer) const override;


            const char *getPluginType() const override;

            const char *getPluginVersion() const override;

            void destroy() override;

            IPluginV2IOExt *clone() const override;

            void setPluginNamespace(const char *pluginNamespace) override;

            const char *getPluginNamespace() const override;

            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const override;

            void attachToContext(cudnnContext *cudnn,
                    cublasContext *cublas, nvinfer1::IGpuAllocator *allocator) override;

            void detachFromContext() override;

            bool isOutputBroadcastAcrossBatch(int outputIndex,
                    const bool *inputIsBroadcasted, int nbInputs) const override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

            void configurePlugin(const PluginTensorDesc *in, int32_t nbInput,
                                  const PluginTensorDesc *out, int32_t nbOutput) override;

            bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                            int32_t nbInputs, int32_t nbOutputs) const override;

        private:
            float _epsilon{0};
            DimsCHW _shape;
            std::vector<float> _h_scale;
            std::vector<float> _h_bias;
            float *_d_scale{nullptr};
            float *_d_bias{nullptr};
            nvinfer1::DataType _data_type;
            cudnnHandle_t _cudnn_handle{nullptr};
            cudnnTensorDescriptor_t _x_desc{nullptr}, _y_desc{nullptr}, _b_desc{nullptr};
            const char *mPluginNamespace;
            std::string mNamespace;
        };

        class InstanceNormalizationPluginCreator : public BaseCreator {
        public:
            InstanceNormalizationPluginCreator();

            ~InstanceNormalizationPluginCreator() override = default;

            const char *getPluginName() const override;

            const char *getPluginVersion() const override;

            const PluginFieldCollection *getFieldNames() override;

            IPluginV2IOExt *createPlugin(const char *name,
                                         const nvinfer1::PluginFieldCollection *fc) override;

            IPluginV2IOExt *deserializePlugin(const char *name,
                                              const void *serialData, size_t serialLength) override;

        private:
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
            std::string mNamespace;
        };
    } //namespace plugin
} //namespace nvinfer1

#endif // TRT_INSTANCE_NORMALIZATION_PLUGIN_H
