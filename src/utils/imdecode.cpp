//
// Created by xingwg on 21-2-9.
//

#include "utils/imdecode.h"
#include "utils/permute.h"
#include "utils/color.h"

namespace alg {
    namespace nv {
        ImageDecoder::ImageDecoder() {
            CUDACHECK(cudaStreamCreate(&stream_));
            CUDACHECK(cudaMalloc((void**)&data_, MAX_FRAME_SIZE));
            NVJPEGCHECK(nvjpegCreateSimple(&handle_));
            NVJPEGCHECK(nvjpegJpegStateCreate(handle_, &jpeg_state_));
        }

        ImageDecoder::~ImageDecoder() {
            NVJPEGCHECK(nvjpegJpegStateDestroy(jpeg_state_));
            NVJPEGCHECK(nvjpegDestroy(handle_));
            CUDACHECK(cudaFree(data_));
            CUDACHECK(cudaStreamDestroy(stream_));
        }

        int ImageDecoder::Decode(const std::string& filename, alg::Mat& img) {
            std::ifstream input(filename, std::ios::in | std::ios::binary | std::ios::ate);
            if (!input.is_open()) {
                LOG(ERROR) << "Cannot open image";
                return -1;
            }
            std::streamsize file_size = input.tellg();
            input.seekg(0, std::ios::beg);
            std::vector<char> img_data;
            img_data.resize(static_cast<size_t>(file_size));
            input.read(img_data.data(), file_size);
            return Decode(img_data, img);
        }

        int ImageDecoder::Decode(const std::vector<char> &img_data, alg::Mat& img) {

            if (!img.ptr()) {
                LOG(ERROR) << "Input image data is null";
                return -2;
            }

            int widths[NVJPEG_MAX_COMPONENT];
            int heights[NVJPEG_MAX_COMPONENT];
            int channels;

            nvjpegStatus_t status = nvjpegGetImageInfo(handle_, reinterpret_cast<const uint8_t*>(img_data.data()),
                                           img_data.size(), &channels, &subsampling_, widths, heights);
            if (status != 0) {
                LOG(ERROR) << "NVJPEG ERROR: " << NvJpegGetErrorString(status);
                return -3;
            }

            widths[3] = widths[2] = widths[1] = widths[0] ;
            heights[3] = heights[2] = heights[1] = heights[0];

            if (channels * widths[0]* heights[0] > MAX_FRAME_SIZE) {
                LOG(ERROR) << "Image Resolution: " << channels << "x" << heights[0] << "x" << widths[0]
                    << " and Image Size: " << channels * widths[0] * heights[0] << " > " << MAX_FRAME_SIZE;
                return -4;
            }

            nv_image_.channel[0] = data_ + 0*heights[0]*widths[0];
            nv_image_.channel[1] = data_ + 1*heights[0]*widths[0];
            nv_image_.channel[2] = data_ + 2*heights[0]*widths[0];
            nv_image_.channel[3] = data_ + 3*heights[0]*widths[0];

            nv_image_.pitch[0] = static_cast<unsigned int>(widths[0]);
            nv_image_.pitch[1] = static_cast<unsigned int>(widths[0]);
            nv_image_.pitch[2] = static_cast<unsigned int>(widths[0]);
            nv_image_.pitch[3] = static_cast<unsigned int>(widths[0]);

            status = nvjpegDecode(handle_, jpeg_state_, reinterpret_cast<const uint8_t*>(img_data.data()),
                                     img_data.size(), fmt_, &nv_image_, stream_);
            if (status != 0) {
                LOG(ERROR) << "NVJPEG ERROR: " << NvJpegGetErrorString(status);
                return -7;
            }

            if (channels == 1) {
                CUDACHECK(cudaMemcpy(nv_image_.channel[1], nv_image_.channel[0], heights[0] * widths[0],
                                     cudaMemcpyDeviceToDevice));
                CUDACHECK(cudaMemcpy(nv_image_.channel[2], nv_image_.channel[0], heights[0] * widths[0],
                                     cudaMemcpyDeviceToDevice));
            }

            img_temp_.data = data_;
            img_temp_.channel = 3;
            img_temp_.height = heights[0];
            img_temp_.width = widths[0];
            alg::nv::permute(img_temp_, img, NV_CHW2HWC);

            return 0;
        }
    }
}