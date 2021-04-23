//
// Created by xingwg on 20-3-13.
//

#ifndef ALGORITHMS_CROP_H
#define ALGORITHMS_CROP_H

#include "error_check.h"
#include "base_types.h"

namespace alg
{
    namespace nv
    {
        /**
         *
         * @param input
         * @param channels
         * @param height
         * @param width
         * @param output
         * @param x
         * @param y
         * @param crop_height
         * @param crop_width
         */
        void crop(const unsigned char* input, int channels, int height, int width,
                  unsigned char* output, int x, int y, int crop_width, int crop_height);

        /**
         *
         * @param input
         * @param channels
         * @param height
         * @param width
         * @param output
         * @param x
         * @param y
         * @param crop_height
         * @param crop_width
         */
        void crop(const float* input, int channels, int height, int width,
                  float* output, int x, int y, int crop_width, int crop_height);

        /**
         *
         * @param src
         * @param dst
         * @param box
         */
        void crop(const Mat& src, Mat& dst, const Box& box);
    }
}

#endif //ALGORITHMS_CROP_H
