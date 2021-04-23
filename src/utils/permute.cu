
#include "utils/permute.h"

namespace alg {
    namespace nv {
        __global__ void permute_hwc2chw_kernel(const int nbThreads,
                unsigned char* in, unsigned char* out, int channel, int height, int width) {
            CUDA_KERNEL_LOOP(idx, nbThreads) {
                int dh = idx / channel / width;
                int dw = idx / channel % width;
                int dc = idx % channel;
                out[dc*height*width + dh*width + dw] = in[idx];
            }
        }

        __global__ void permute_chw2hwc_kernel(const int nbThreads,
                unsigned char* in, unsigned char* out, int channel, int height, int width) {
            CUDA_KERNEL_LOOP(idx, nbThreads) {
                int dc = idx / width / height;
                int dh = idx / width % height;
                int dw = idx % width;
                out[dh*width*channel + dw*channel + dc] = in[idx];
            }
        }

        void permute(const alg::Mat& src, alg::Mat& dst, permuteEnum_t mode) {
            LOG_ASSERT(src.ptr());

            dst.channel = src.c();
            dst.height = src.h();
            dst.width = src.w();

            LOG_ASSERT(src.c() == 3);
            const int nbThreads = src.size();
            if (mode == NV_HWC2CHW) {
                permute_hwc2chw_kernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (
                        nbThreads, src.ptr(), dst.ptr(), src.c(), src.h(), src.w());
            } else if (mode == NV_CHW2HWC) {
                permute_chw2hwc_kernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (
                        nbThreads, src.ptr(), dst.ptr(), src.c(), src.h(), src.w());
            } else {
                LOG(FATAL) << "No support mode!";
            }
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaGetLastError());
        }
    }
}