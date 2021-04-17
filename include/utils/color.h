//
// Created by xingwg on 20-3-13.
//

#ifndef ALGORITHMS_COLOR_H
#define ALGORITHMS_COLOR_H

#include "error_check.h"

namespace alg {
    namespace nv {
        enum ColorSpace {
            NV_BGRA2BGR_ = -3,
            NV_RGB2BGR_ = -2,
            NV_BGR2RGB_ = -1,
            NV_NOCOLOR = 0,
            NV_RGB2BGR  = 1,
            NV_RGBA2BGR = 2,
            NV_BGRA2BGR = 3,
            NV_BGR2RGB  = 4,
        };

        void cvtColor(const unsigned char* input, const int channels, const int height, const int width,
                      unsigned char* output, int mode);

        void cvtColor(const float* input, const int channels, const int height, const int width,
                      float* output, int mode);

        void cvtColor_(unsigned char* input, const int channels, const int height, const int width, int mode);

        void cvtColor_(float* input, const int channels, const int height, const int width, int mode);
    }
}

#endif //ALGORITHMS_COLOR_H
