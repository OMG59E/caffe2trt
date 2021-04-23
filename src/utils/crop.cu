//
// Created by xingwg on 20-3-14.
//

#include "utils/crop.h"

namespace alg {
    namespace nv {
        template<typename Dtype>
        __global__ void CropKernel(const int nbThreads,
                                   const Dtype *input, const int channels, const int inputWidth,
                                   Dtype *output, const int x, const int y, const int outputWidth) {
            CUDA_KERNEL_LOOP(index, nbThreads) {
                int dx = (index / channels) % outputWidth;
                int dy = ((index / channels) - dx) / outputWidth;
                int dc = index % channels;

                output[index] = input[((dy + y) * inputWidth + (dx + x)) * channels + dc];
            }
        }

        void crop(const unsigned char *input, const int channels, const int height, const int width,
                  unsigned char *output, const int x, const int y, const int crop_width, const int crop_height) {
            LOG_ASSERT(output);
            LOG_ASSERT(x >= 0 && y >= 0 && (x + crop_width - 1) < width && (y + crop_height - 1) < height);

            const int nbThreads = channels * crop_height * crop_width;

            CropKernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (nbThreads,
                    input, channels, width,
                    output, x, y, crop_width);

            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaGetLastError());
        }

        void crop(const float *input, const int channels, const int height, const int width,
                  float *output, const int x, const int y, const int crop_width, const int crop_height) {
            LOG_ASSERT(output);
            LOG_ASSERT(x >= 0 && y >= 0 && (x + crop_width) < width && (y + crop_height) < height);

            const int nbThreads = channels * crop_height * crop_width;

            CropKernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (nbThreads,
                    input, channels, width,
                    output, x, y, crop_width);

            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaGetLastError());
        }

        void crop(const Mat &src, Mat &dst, const Box &box) {
            dst.channel = src.c();
            dst.height = box.h();
            dst.width = box.w();
            crop(src.data, src.c(), src.h(), src.w(), dst.data, box.x1, box.y1, box.w(), box.h());
        }
    }
}