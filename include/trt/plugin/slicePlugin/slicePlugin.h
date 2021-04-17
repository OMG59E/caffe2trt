//
// Created by xingwg on 20-4-22.
//

#ifndef ALGORITHMS_SLICEPLUGIN_H
#define ALGORITHMS_SLICEPLUGIN_H

#include <NvInferPlugin.h>
#include <cublas_v2.h>
#include <vector>
#include "trt/plugin/plugin.h"

namespace nvinfer1 {
    namespace plugin {
        class Slice : public IPluginV2IOExt {
        public:
            Slice(int axis, int slice_dim, std::vector<int> slice_points,
                    DataType data_type, int channel_in, int height_in, int width_in);

            Slice(const void *data, size_t length);

            Slice() = delete;

            ~Slice() override;

            int getNbOutputs() const override;

            Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

            int initialize() override;

            void terminate() override;

            size_t getWorkspaceSize(int maxBatchSize) const override;

            int enqueue(
                    int batchSize, const void *const *inputs, void **outputs, void *workspace,
                    cudaStream_t stream) override;

            DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const override;

            size_t getSerializationSize() const override;

            void serialize(void *buffer) const override;

            bool
            isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

//            void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
//                                 const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
//                                 const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;
//            bool supportsFormat(DataType type, PluginFormat format) const override;

            void configurePlugin(const PluginTensorDesc *in, int32_t nbInput,
                                         const PluginTensorDesc *out, int32_t nbOutput) override;
            bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                     int32_t nbInputs, int32_t nbOutputs) const override;

            void detachFromContext() override;

            const char *getPluginType() const override;

            const char *getPluginVersion() const override;

            void destroy() override;

            void attachToContext(
                    cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) override;

            IPluginV2IOExt *clone() const override;

            void setPluginNamespace(const char *pluginNamespace) override;

            const char *getPluginNamespace() const override;

        private:
            int axis_;
            int slice_dim_;
            std::vector<int> slice_points_;
            int num_output_;
            DimsCHW bottom_shape_;
            DataType data_type_{DataType::kFLOAT};
            std::vector<DimsCHW> top_shapes_;
            const char *mPluginNamespace;
        };

        class SlicePluginCreator : public BaseCreator {
        public:
            SlicePluginCreator();

            ~SlicePluginCreator() override = default;

            const char *getPluginName() const override;

            const char *getPluginVersion() const override;

            const PluginFieldCollection *getFieldNames() override;

            IPluginV2IOExt *createPlugin(const char *name, const PluginFieldCollection *fc) override;

            IPluginV2IOExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

        private:
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
        };
    }
}

#endif //ALGORITHMS_SLICEPLUGIN_H
