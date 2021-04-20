//
// Created by xingwg on 20-1-13.
//

#ifndef ALGORITHMS_NET_OPERATOR_H
#define ALGORITHMS_NET_OPERATOR_H

#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>

#include <NvCaffeParser.h>
#include <NvInferPlugin.h>
#include <cuda.h>

#include "base_types.h"
#include "trt_logger.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

#define MAX_OUTPUTS 128

namespace alg {
    namespace trt {
        static Logger gLogger;

        struct Profiler : public IProfiler {
            typedef std::pair<std::string, float> Record;

            std::vector<Record> mProfile;

            void reportLayerTime(const char *layerName, float ms) override {
                auto record = std::find_if(mProfile.begin(), mProfile.end(),
                                           [&](const Record &r) { return r.first == layerName; });

                if (record == mProfile.end()) {
                    mProfile.push_back(std::make_pair(layerName, ms));
                } else {
                    record->second += ms;
                }
            }

            void printLayerTimes(const int iterations) {
                float totalTime = 0;
                for (const auto &profile : mProfile) {
                    LOG(INFO) << profile.first.c_str() << " " << profile.second / iterations << "ms";
                    totalTime += profile.second;
                }
                LOG(INFO) << "Time over all layers: " << totalTime / iterations;
            }
        };

        class NetOperator {
        public:
            NetOperator(const char *engineFile, const NetParameter &para, int device_id = -1);

            ~NetOperator();

            bool inference(const std::vector<alg::Mat> &vGpuImages, std::vector<alg::Tensor> &vOutputTensors);

            bool inference(const float* batch_data, int batch_size, std::vector<alg::Tensor> &vOutputTensors);

            int getNbOutputs() const;

            DimsNCHW getInputShape() const;

            void printLayerTimes(int iterations);

        private:
            bool preprocess_gpu(const std::vector<alg::Mat> &vGpuImages);

            bool prepareInference();

            DimsCHW getTensorDims(const char *name);

            float *allocateMemory(DimsCHW dims, bool host);

        private:

            Profiler profiler_;

            int device_id_;
            DimsNCHW input_shape_;

            cudaStream_t stream_;
            IExecutionContext *ctx_;
            ICudaEngine *engine_;
            IRuntime *runtime_;

            Tensor input_tensor_;
            std::vector<Tensor> output_tensors_;

            float scale_[3];
            float mean_val_[3];

            void *buffers_[MAX_OUTPUTS + 1];

            std::string input_tensor_name_;
            std::vector<std::string> output_tensor_names_;

            std::string color_mode_;
            std::string resize_mode_;
            PaddingMode padding_mode_;
            uint8_t *resize_data_{nullptr};
        };
    }
}

#endif //ALGORITHMS_NET_OPERATOR_H
