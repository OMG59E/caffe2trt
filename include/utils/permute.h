//
// Created by xingwg on 21-2-10.
//

#ifndef ALGORITHMS_PERMUTE_H
#define ALGORITHMS_PERMUTE_H

#include "base_types.h"
#include "error_check.h"

namespace alg {
    namespace nv {
        typedef enum {
            NV_HWC2CHW = 1,
            NV_CHW2HWC = 2,
        } permuteEnum_t;

        /**
         *
         * @param src
         * @param dst
         * @param mode
         */
        void permute(const alg::Mat& src, alg::Mat& dst, permuteEnum_t mode);
    }
}
#endif //ALGORITHMS_PERMUTE_H
