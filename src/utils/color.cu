//
// Created by xingwg on 20-3-14.
//

#include "utils/color.h"
#include "base_types.h"

namespace alg {
    namespace nv {
        template <typename Dtype>
        __global__ void BGRA2BGRKernel(const int nbThreads, const Dtype* input, int width, Dtype* output) {
            CUDA_KERNEL_LOOP(idx, nbThreads) {
                int dx = idx / 3 % width;
                int dy = idx / 3 / width;
                int dc = idx % 3;

                output[idx] = input[(dy*width + dx)*4 + dc];
            }
        }

        template <typename Dtype>
        __global__ void BGRA2BGRKernel_(const int nbThreads, Dtype* input, int width) {
            CUDA_KERNEL_LOOP(idx, nbThreads)
            {
                int dx = idx % width;
                int dy = idx / width;

                uchar3 bgr = { input[(dy*width + dx)*4 + 0],
                               input[(dy*width + dx)*4 + 1],
                               input[(dy*width + dx)*4 + 2] };

                input[(dy*width + dx)*3 + 0] = bgr.x;
                input[(dy*width + dx)*3 + 1] = bgr.y;
                input[(dy*width + dx)*3 + 2] = bgr.z;
            }
        }

        template <typename Dtype>
        __global__ void RGB2BGRKernel(const int nbThreads, const Dtype* input, int width, Dtype* output) {
            CUDA_KERNEL_LOOP(idx, nbThreads) {
                int dx = (idx / 3) % width;
                int dy = ((idx / 3) - dx) / width;
                int dc = idx % 3;

                output[idx] = input[(dy*width + dx)*3 + (2 - dc)];
            }
        }

        template <typename Dtype>
        __global__ void BGR2RGBKernel(const int nbThreads, const Dtype* input, int width, Dtype* output) {
            CUDA_KERNEL_LOOP(idx, nbThreads) {
                int dx = (idx / 3) % width;
                int dy = ((idx / 3) - dx) / width;
                int dc = idx % 3;

                output[idx] = input[(dy*width + dx)*3 + (2 - dc)];
            }
        }

        template <typename Dtype>
        __global__ void BGR2RGBKernel_(const int nbThreads, Dtype* input, int width) {
            CUDA_KERNEL_LOOP(idx, nbThreads) {
                int dx = idx % width;
                int dy = idx / width;
                uchar3 bgr = { input[(dy*width + dx)*3 + 0],
                               input[(dy*width + dx)*3 + 1],
                               input[(dy*width + dx)*3 + 2] };
                input[(dy*width + dx)*3 + 0] = bgr.z;
                input[(dy*width + dx)*3 + 1] = bgr.y;
                input[(dy*width + dx)*3 + 2] = bgr.x;
            }
        }

        template <typename Dtype>
        __global__ void BGR2RGBKernel(const int nbThreads, Dtype* input, int width) {
            CUDA_KERNEL_LOOP(idx, nbThreads) {
                int dx = idx % width;
                int dy = (idx - dx) / width;

                Dtype B = input[(dy*width + dx)*3 + 0];
                Dtype G = input[(dy*width + dx)*3 + 1];
                Dtype R = input[(dy*width + dx)*3 + 2];

                input[(dy*width + dx)*3 + 0] = R;
                input[(dy*width + dx)*3 + 1] = G;
                input[(dy*width + dx)*3 + 2] = B;
            }
        }

        template <typename Dtype>
        __global__ void RGBA2BGRKernel(const int nbThreads, const Dtype* input, int width, Dtype* output) {
            CUDA_KERNEL_LOOP(index, nbThreads) {
                int dx = (index / 3) % width;
                int dy = ((index / 3) - dx) / width;
                int dc = index % 3;

                output[index] = input[(dy*width + dx)*4 + (2 - dc)];
            }
        }

        void cvtColor(const unsigned char* input, const int channels, const int height, const int width,
                      unsigned char* output, int mode) {
            LOG_ASSERT(output);
            if (mode == NV_BGRA2BGR) {
                LOG_ASSERT(channels == 4);
                const int nbThreads = 3*height*width;
                BGRA2BGRKernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS>>>(nbThreads, input, width, output);
            } else if (mode == NV_RGB2BGR) {
                LOG_ASSERT(channels == 3);
                const int nbThreads = 3*height*width;
                RGB2BGRKernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS>>>(nbThreads, input, width, output);
            } else if (mode == NV_RGBA2BGR) {
                LOG_ASSERT(channels == 4);
                const int nbThreads = 3*height*width;
                RGBA2BGRKernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS>>>(nbThreads, input, width, output);
            } else if (mode == NV_BGR2RGB) {
                LOG_ASSERT(channels == 3);
                const int nbThreads = 3 * height * width;
                BGR2RGBKernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (nbThreads, input, width, output);
            } else {
                LOG(FATAL) << "No support mode!";
            }
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaGetLastError());
        }

        void cvtColor(const float* input, const int channels, const int height, const int width,
                      float* output, int mode) {
            LOG_ASSERT(output);
            if (mode == NV_BGRA2BGR) {
                LOG_ASSERT(channels == 4);
                const int nbThreads = 3 * height * width;
                BGRA2BGRKernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (nbThreads, input, width, output);
            } else if (mode == NV_RGB2BGR) {
                LOG_ASSERT(channels == 3);
                const int nbThreads = 3 * height * width;
                RGB2BGRKernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (nbThreads, input, width, output);
            } else if (mode == NV_RGBA2BGR) {
                LOG_ASSERT(channels == 4);
                const int nbThreads = 3 * height * width;
                RGBA2BGRKernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (nbThreads, input, width, output);
            } else if (mode == NV_BGR2RGB) {
                LOG_ASSERT(channels == 3);
                const int nbThreads = 3 * height * width;
                BGR2RGBKernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (nbThreads, input, width, output);
            } else {
                LOG(FATAL) << "No support mode!";
            }
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaGetLastError());
        }


        void cvtColor_(unsigned char* input, const int channels, const int height, const int width, int mode) {
            LOG_ASSERT(input);
            if (mode == NV_BGR2RGB_ || mode == NV_RGB2BGR_) {
                LOG_ASSERT(channels == 3);
                const int nbThreads = height*width;
                BGR2RGBKernel_<uint8_t><<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS>>>(nbThreads, input, width);
            } else if(mode == NV_BGRA2BGR_) {
                LOG_ASSERT(channels == 4);
                const int nbThreads = height*width;
                BGRA2BGRKernel_<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS>>>(nbThreads, input, width);
            } else {
                LOG(FATAL) << "No support mode!";
            }
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaGetLastError());
        }

        void cvtColor_(float* input, const int channels, const int height, const int width, int mode) {
            LOG_ASSERT(input);
            if (mode == NV_BGR2RGB_ || mode == NV_RGB2BGR_) {
                LOG_ASSERT(channels == 3);
                const int nbThreads = height * width;
                BGR2RGBKernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (nbThreads, input, width);
            } else {
                LOG(FATAL) << "No support mode!";
            }
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaGetLastError());
        }
    }
}