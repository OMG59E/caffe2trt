//
// Created by xingwg on 20-1-13.
//
#include <chrono>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "trt/net_operator.h"
#include "utils/color.h"
#include "utils/resize.h"

using namespace std::chrono;

namespace alg {
    namespace trt {
        NetOperator::NetOperator(const char *engineFile, const NetParameter &para, int device_id)
                : ctx_{nullptr}, engine_{nullptr}, runtime_{nullptr}, stream_{nullptr}, device_id_{device_id},
                  resize_data_{nullptr} {

            initLibNvInferPlugins(&gLogger, "");

            // 输入张量形状
            input_shape_ = para.input_shape;

            // 归一化系数
            scale_[0] = para.scale[0];
            scale_[1] = para.scale[1];
            scale_[2] = para.scale[2];

            // 三通道均值
            mean_val_[0] = para.mean_val[0];
            mean_val_[1] = para.mean_val[1];
            mean_val_[2] = para.mean_val[2];

            // 预处理的resize方法
            color_mode_ = para.color_mode;
            resize_mode_ = para.resize_mode;
            padding_mode_ = para.padding_mode;

            // 输入输出张量名字
            input_tensor_name_ = para.input_node_name;
            output_tensor_names_ = para.output_node_names;

            // 设置运行设备
            LOG_ASSERT(device_id_ >= 0) << " The deviceID must be >= 0";
            CUDACHECK(cudaSetDevice(device_id_));

            //
            CUDACHECK(cudaStreamCreate(&stream_));

            // Create inference runtime engine.
            runtime_ = createInferRuntime(gLogger);
            if (!runtime_)
                LOG(FATAL) << "TensorRT failed to create inference runtime.";

            std::stringstream tensorrt_model_stream;
            tensorrt_model_stream.seekg(0, tensorrt_model_stream.beg);

            std::ifstream tensorrt_model_cache_load(engineFile); //model cache to load
            if (!tensorrt_model_cache_load)
                LOG(FATAL) << "TensorRT failed to open model -> " << engineFile;

            DLOG(INFO) << "Cached TensorRT model found, start loading...";

            tensorrt_model_stream << tensorrt_model_cache_load.rdbuf();
            tensorrt_model_cache_load.close();

            // support for stringstream deserialization was deprecated in TensorRT v2
            // instead, read the stringstream into a memory buffer and pass that to TRT.
            tensorrt_model_stream.seekg(0, std::ios::end);
            const size_t modelSize = tensorrt_model_stream.tellg();
            tensorrt_model_stream.seekg(0, std::ios::beg);

            DLOG(INFO) << "Cached model size : " << modelSize;

            void *modelMem = malloc(modelSize);
            if (!modelMem)
                LOG(FATAL) << "TensorRT failed to allocate memory to deserialize model.";

            tensorrt_model_stream.read((char *) modelMem, modelSize);
            engine_ = runtime_->deserializeCudaEngine(modelMem, modelSize, nullptr);
            free(modelMem);
            if (!engine_)
                LOG(FATAL) << "TensorRT failed to deserialize CUDA engine.";

            DLOG(INFO) << "TensorNet deserialize model ok. Number of binding indices " << engine_->getNbBindings();

            tensorrt_model_stream.str("");

            ctx_ = engine_->createExecutionContext();
            if (!ctx_)
                LOG(FATAL) << "TensorRT failed to create context.";

            ctx_->setProfiler(&profiler_);

            if (!prepareInference())
                LOG(FATAL) << "TensorRT failed to prepare inference.";
        }

        NetOperator::~NetOperator() {
            ctx_->destroy();
            engine_->destroy();
            runtime_->destroy();
            CUDACHECK(cudaStreamDestroy(stream_));

            CUDACHECK(cudaFree(input_tensor_.gpu_data));
            for (int i = 0; i < output_tensor_names_.size(); i++) {
                CUDACHECK(cudaFree(output_tensors_[i].gpu_data));
                CUDACHECK(cudaFreeHost(output_tensors_[i].data));
            }

            if (resize_data_) {
                CUDACHECK(cudaFree(resize_data_));
                resize_data_ = nullptr;
            }
        }

        bool NetOperator::inference(const std::vector<alg::Mat> &vGpuImages,
                std::vector<alg::Tensor> &vOutputTensors) {
            vOutputTensors.clear();

            if (resize_mode_ == "v1") {
                if (!preprocess_gpu(vGpuImages)) {
                    LOG(ERROR) << "Net operator preprocess resize failed.";
                    return false;
                }
            } else {
                LOG(ERROR) << "No support resize mode: " << resize_mode_;
                return false;
            }

            // forward
            if (!ctx_->enqueue(int(vGpuImages.size()), buffers_, stream_, nullptr)) {
                LOG(ERROR) << "ctx_->enqueue infer failed.";
                return false;
            }

            for (auto &tensor : output_tensors_) {
                CUDACHECK(cudaMemcpyAsync(tensor.data, tensor.gpu_data,
                                          input_shape_.n() * tensor.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                          stream_));
            }

            CUDACHECK(cudaStreamSynchronize(stream_));

            vOutputTensors.assign(output_tensors_.begin(), output_tensors_.end());

            return true;
        }

        bool NetOperator::prepareInference() {
            DLOG(INFO) << "TensorRT INPUT_WIDTH: " << input_shape_.w()
                       << " INPUT_HEIGHT: " << input_shape_.h();

            if (output_tensor_names_.size() > MAX_OUTPUTS) {
                LOG(ERROR) << "output_tensor_names_.size() must be <= MAX_OUTPUTS.";
                return false;
            }

            /// input
            input_tensor_.name = input_tensor_name_;
            input_tensor_.shape = getTensorDims(input_tensor_name_.c_str());
            input_tensor_.gpu_data = allocateMemory(input_tensor_.shape, false);

            buffers_[0] = input_tensor_.gpu_data;

            /// output
            for (size_t i = 0; i < output_tensor_names_.size(); i++) {
                Tensor tensor;
                tensor.name = output_tensor_names_[i];
                tensor.shape = getTensorDims(output_tensor_names_[i].c_str());
                tensor.data = allocateMemory(tensor.shape, true);
                tensor.gpu_data = allocateMemory(tensor.shape, false);
                output_tensors_.push_back(tensor);
                buffers_[i + 1] = tensor.gpu_data;
            }

            return true;
        }

        DimsCHW NetOperator::getTensorDims(const char *name) {
            for (int n = 0; n < engine_->getNbBindings(); n++) {
                if (!strcmp(name, engine_->getBindingName(n))) {
                    DimsCHW dims = static_cast<DimsCHW &&>(engine_->getBindingDimensions(n));
                    return dims;
                }
            }

            LOG(FATAL) << "The node tensor(" << name << ") does not exist.";
        }

        float *NetOperator::allocateMemory(DimsCHW dims, bool host) {
            float *ptr{nullptr};
            size_t size = input_shape_.n();
            for (int i = 0; i < dims.nbDims; i++)
                size *= dims.d[i];

            if (host)
                CUDACHECK(cudaMallocHost(&ptr, size * sizeof(float)));
            else
                CUDACHECK(cudaMalloc(&ptr, size * sizeof(float)));

            return ptr;
        }

        int NetOperator::getNbOutputs() const {
            return output_tensor_names_.size();
        }

        DimsNCHW NetOperator::getInputShape() const {
            return input_shape_;
        }

        void NetOperator::printLayerTimes(int iterations) {
            profiler_.printLayerTimes(iterations);
        }

        bool NetOperator::preprocess_gpu(const std::vector<Mat> &vGpuImages) {
            high_resolution_clock::time_point t_start = high_resolution_clock::now();

            float *o_ptr = input_tensor_.gpu_data;

            int offset = input_shape_.c() * input_shape_.h() * input_shape_.w();

            for (auto &img : vGpuImages) {
                if (color_mode_ == "RGB")
                    cvtColor_(img.ptr(), img.c(), img.h(), img.w(), alg::nv::NV_BGR2RGB_);

                alg::nv::resizeNormPermuteOp(img.ptr(), img.c(), img.h(), img.w(),
                                             o_ptr, input_shape_.h(), input_shape_.w(), scale_, mean_val_);

                o_ptr += offset;
            }

            duration<float, std::micro> time_span = high_resolution_clock::now() - t_start;
            DLOG(INFO) << "preprocess_gpu: " << time_span.count() * 0.001f << "ms";

            return true;
        }

        bool NetOperator::inference(const float *batch_data,
                const int batch_size, std::vector<alg::Tensor> &vOutputTensors) {
            vOutputTensors.clear();

            CUDACHECK(cudaMemcpy(input_tensor_.gpu_data, batch_data,
                                 batch_size * input_tensor_.size() * sizeof(float), cudaMemcpyHostToDevice));
            // forward
            if (!ctx_->enqueue(batch_size, buffers_, stream_, nullptr)) {
                LOG(ERROR) << "ctx_->enqueue infer failed.";
                return false;
            }

            for (auto &tensor : output_tensors_) {
                CUDACHECK(cudaMemcpyAsync(tensor.data, tensor.gpu_data,
                        input_shape_.n() * tensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
            }

            CUDACHECK(cudaStreamSynchronize(stream_));

            vOutputTensors.assign(output_tensors_.begin(), output_tensors_.end());

            return true;
        }
    }
}