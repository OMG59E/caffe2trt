//
// Created by xingwg on 20-3-13.
//

#include "utils/resize.h"

#define clip(x, a, b) x >= a ? (x < b ? x : b-1) : a

namespace alg {
    namespace nv {
        __device__ float calcScaleVal(const unsigned char* input, int channels, int iHeight, int iWidth,
                                      int oHeight, int oWidth, int dx, int dy, int dc) {
            float scale_x = (float) iWidth / oWidth;
            float scale_y = (float) iHeight / oHeight;

            int xmax = oWidth;

            float fx = (dx + 0.5f) * scale_x - 0.5f;
            int sx = floor(fx);
            fx -= sx;

            int isx1 = sx;
            if (isx1 < 0) {
                fx = 0.f;
                isx1 = 0;
            }

            if (isx1 >= (iWidth - 1)) {
                xmax = min(xmax, dy);
                fx = 0;
                isx1 = iWidth - 1;
            }

            float2 cbufx;
            cbufx.x = (1.f - fx);
            cbufx.y = fx;

            float fy = (dy + 0.5f) * scale_y - 0.5f;
            int sy = floor(fy);
            fy -= sy;

            int isy1 = clip(sy + 0, 0, iHeight);
            int isy2 = clip(sy + 1, 0, iHeight);

            float2 cbufy;
            cbufy.x = (1.f - fy);
            cbufy.y = fy;

            int isx2 = isx1 + 1;

            float s11 = input[(isy1 * iWidth + isx1) * channels + dc];
            float s12 = input[(isy1 * iWidth + isx2) * channels + dc];
            float s21 = input[(isy2 * iWidth + isx1) * channels + dc];
            float s22 = input[(isy2 * iWidth + isx2) * channels + dc];

            float h_rst00, h_rst01;

            if (dy > xmax - 1) {
                h_rst00 = s11;
                h_rst01 = s21;
            } else {
                h_rst00 = s11 * cbufx.x + s12 * cbufx.y;
                h_rst01 = s21 * cbufx.x + s22 * cbufx.y;
            }

            float d0 = h_rst00 * cbufy.x + h_rst01 * cbufy.y;

            return d0;
        }

        __global__ void INTER_LINEAR_NORM_Kernel(const int nbThreads,
                const unsigned char* input, const int channels, const int iHeight, const int iWidth,
                float* output, const int oHeight, const int oWidth, const float3 scale, const float3 mean) {
            CUDA_KERNEL_LOOP(idx, nbThreads) {
                int dx = idx / channels % oWidth;
                int dy = idx / channels / oWidth;
                int dc = idx % channels;
                if (iHeight != oHeight || iWidth != oWidth) {
                    float d0 = calcScaleVal(input, channels, iHeight, iWidth, oHeight, oWidth, dx, dy, dc);

                    if (dc == 0)
                        output[dc*oHeight*oWidth + dy*oWidth + dx] = (d0 - mean.x) * scale.x;

                    if (dc == 1)
                        output[dc*oHeight*oWidth + dy*oWidth + dx] = (d0 - mean.y) * scale.y;

                    if (dc == 2)
                        output[dc*oHeight*oWidth + dy*oWidth + dx] = (d0 - mean.z) * scale.z;
                } else {
                    if (dc == 0)
                        output[dc*oHeight*oWidth + dy*oWidth + dx] =
                                (input[(dx + dy * iWidth) * channels + dc] - mean.x) * scale.x;

                    if (dc == 1)
                        output[dc*oHeight*oWidth + dy*oWidth + dx] =
                                (input[(dx + dy * iWidth) * channels + dc] - mean.y) * scale.y;

                    if (dc == 2)
                        output[dc*oHeight*oWidth + dy*oWidth + dx] =
                                (input[(dx + dy * iWidth) * channels + dc] - mean.z) * scale.z;
                }
            }
        }

        void resizeNormPermuteOp(const unsigned char *input, int channels, int iHeight, int iWidth,
                                 float *output, int oHeight, int oWidth, const float *scale, const float *mean,
                                 int mode) {
            float3 scale_ = make_float3(scale[0], scale[1], scale[2]);
            float3 mean_ = make_float3(mean[0], mean[1], mean[2]);
            const int nbThreads = channels*oHeight*oWidth;
            if (mode == PIL_INTER_LINEAR) {
                INTER_LINEAR_NORM_Kernel<<<CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS>>>(nbThreads,
                        input, channels, iHeight, iWidth,
                        output, oHeight, oWidth, scale_, mean_);
            } else {
                LOG(FATAL) << "No support mode: " << mode;
            }
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaGetLastError());
        }

        template<typename Dtype>
        __global__ void INTER_LINEAR_Kernel(const int nbThreads,
                                            const unsigned char *input, int channels, int iHeight, int iWidth,
                                            Dtype *output, int oHeight, int oWidth) {
            CUDA_KERNEL_LOOP(idx, nbThreads) {
                int dx = idx / channels % oWidth;
                int dy = idx / channels / oWidth;
                int dc = idx % channels;

                float d0 = calcScaleVal(input, channels, iHeight, iWidth, oHeight, oWidth, dx, dy, dc);

                output[(dy * oWidth + dx) * channels + dc] = d0;
            }
        }

        void resize(const unsigned char *input, int channels, int iHeight, int iWidth,
                    unsigned char *output, int oHeight, int oWidth, int mode) {

            const int nbThreads = channels * oHeight * oWidth;
            if (mode == PIL_INTER_LINEAR) {
                INTER_LINEAR_Kernel << < CUDA_GET_BLOCKS(nbThreads), CUDA_NUM_THREADS >> > (nbThreads,
                        input, channels, iHeight, iWidth,
                        output, oHeight, oWidth);
            } else {
                LOG(FATAL) << "No support mode: " << mode;
            }
            CUDACHECK(cudaDeviceSynchronize());
            CUDACHECK(cudaGetLastError());
        }

        void resize(const alg::Mat &src, alg::Mat &dst, alg::Size size, int mode) {
            LOG_ASSERT(src.ptr() && dst.ptr());
            LOG_ASSERT((dst.height = size.h) && (dst.width = size.w));
            resize(src.ptr(), src.c(), src.h(), src.w(), dst.ptr(), size.h, size.w, mode);
        }
    }
}