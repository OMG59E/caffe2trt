//
// Created by xingwg on 21-2-9.
//

#ifndef ALGORITHMS_IMDECODE_H
#define ALGORITHMS_IMDECODE_H

#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include "base_types.h"
#include "error_check.h"

#define MAX_FRAME_C 4
#define MAX_FRAME_H 4000
#define MAX_FRAME_W 4000
#define MAX_FRAME_SIZE MAX_FRAME_C*MAX_FRAME_H*MAX_FRAME_W

namespace alg {
    namespace nv {
        class ImageDecoder {
        public:
            ImageDecoder();
            ~ImageDecoder();

            int Decode(const std::string& filename, alg::Mat& img);

            int Decode(const std::vector<char>& img_data, alg::Mat& img);

        private:
            alg::Mat img_temp_;
            uint8_t *data_{nullptr};
            cudaStream_t stream_;
            nvjpegImage_t nv_image_;
            nvjpegHandle_t handle_;
            nvjpegJpegState_t jpeg_state_;
            nvjpegChromaSubsampling_t subsampling_;
            nvjpegOutputFormat_t fmt_{NVJPEG_OUTPUT_BGR};
        };
    }
}

#endif //ALGORITHMS_IMDECODE_H
