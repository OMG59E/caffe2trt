//
// Created by xingwg on 20-3-13.
//

#ifndef ALGORITHMS_RESIZE_H
#define ALGORITHMS_RESIZE_H

#include "error_check.h"
#include "base_types.h"
#include "utils/color.h"

namespace alg {
    namespace nv {
        enum ResizeMode {
            PIL_INTER_LINEAR = 0,  // bilinear interpolation
            PIL_INTER_NEAREST = 1,  // nearest neighbor interpolation
        };

        void resize(const unsigned char* input, int channels, int iHeight, int iWidth,
                    unsigned char* output, int oHeight, int oWidth, int mode=PIL_INTER_LINEAR);

        void resize(const unsigned char* input, int channels, int iHeight, int iWidth,
                    float* output, int oHeight, int oWidth, int mode=PIL_INTER_LINEAR);

        void resizeNormPermuteOp(const unsigned char *input, int channels, int iHeight, int iWidth,
                                 float *output, int oHeight, int oWidth, const float *scale, const float *mean,
                                 int mode = PIL_INTER_LINEAR);

        void resizeAddCvtColorOp(const unsigned char *input, int channels, int iHeight, int iWidth,
                                 unsigned char *output, int oHeight, int oWidth, int target_h, int target_w,
                                 int offset_y, int offset_x, int mode=NV_NOCOLOR);

        void resizeBGRAToBGROp(const unsigned char* input, int iHeight, int iWidth,
                               unsigned char* output, int oHeight, int oWidth, int mode=PIL_INTER_LINEAR);
    }
}

#endif //ALGORITHMS_RESIZE_H
