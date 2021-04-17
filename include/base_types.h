//
// Created by xingwg on 20-1-10.
//

#ifndef ALGORITHMS_BASE_TYPE_H
#define ALGORITHMS_BASE_TYPE_H

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "error_check.h"

// CUDA: use 512 threads per block
#define CUDA_NUM_THREADS 1024

// CUDA: number of blocks for threads.
#define CUDA_GET_BLOCKS(N) (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define FLOAT_MAX 3.402823466e+38F        /* max value */

#define MAX_FRAME_H 1920
#define MAX_FRAME_W 1920
#define MAX_FRAME_C 1920


using namespace nvinfer1;

namespace alg {
    struct Mat {
        uint8_t* data{nullptr};
        int channel{0};
        int height{0};
        int width{0};
        bool own{false};
        bool gpu{true};

        const int c() const { return channel; }
        const int h() const { return height; }
        const int w() const { return width; }

        const int size() const { return channel*height*width; }

        uint8_t* ptr() const { return data; }

        void create(int c, int h, int w, bool use_gpu=true) {
            channel = c;
            height = h;
            width = w;
            own = true;
            gpu = use_gpu;

            LOG_ASSERT(!data);
            if (use_gpu)
                CUDACHECK(cudaMalloc((void**)&data, channel*height*width*sizeof(uint8_t)));
            else
                CUDACHECK(cudaMallocHost((void**)&data, channel*height*width*sizeof(uint8_t)));
        }

        void free() {
            if (own && data) {
                if (gpu)
                    CUDACHECK(cudaFree(data));
                else
                    CUDACHECK(cudaFreeHost(data));
                data = nullptr;
                own = false;
            } else {
                data = nullptr;
                own = false;
            }
        }

        bool empty() const { return data == nullptr; }
    };

    enum PaddingMode {
        CENTER=0,
        TOP_LEFT=1,
    };

    struct NetParameter {
        DimsNCHW input_shape;
        float scale[3] = {1, 1, 1};
        float mean_val[3] = {0, 0, 0};

        std::string color_mode;
        std::string resize_mode;
        PaddingMode padding_mode;

        std::string input_node_name;
        std::vector<std::string> output_node_names;

        NetParameter()
            : resize_mode{"v1"}
            , color_mode{"BGR"}
            , padding_mode{PaddingMode::CENTER} {
            output_node_names.clear();
        }
    };

    struct Tensor {
        std::string name;
        DimsCHW shape;
        float* data{nullptr};
        float* gpu_data{nullptr};
        size_t size() const { return (size_t)shape.c()*shape.h()*shape.w(); }
    };
}

#endif //ALGORITHMS_BASE_TYPE_H
