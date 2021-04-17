//
// Created by xingwg on 20-1-13.
//

#include "trt/calibrator.h"
#include "trt/batch_stream.h"
#include "error_check.h"

#include <cuda_runtime_api.h>
#include <string.h>
#include <fstream>
#include <iterator>

namespace alg {
    namespace trt {
        Int8EntropyCalibrator::Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache)
                : mStream(stream),
                  mReadCache(readCache) {
            DimsNCHW dims = mStream.getDims();
            mInputCount = mStream.getBatchSize() * dims.c() * dims.h() * dims.w();

            CUDACHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));

            mStream.reset(firstBatch);
        }

        Int8EntropyCalibrator::~Int8EntropyCalibrator() {
            CUDACHECK(cudaFree(mDeviceInput));
        }

        int Int8EntropyCalibrator::getBatchSize() const {
            return mStream.getBatchSize();
        }

        bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) {
            if (!mStream.next())
                return false;

            CUDACHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));

            LOG_ASSERT(!strcmp(names[0], "data"));
            bindings[0] = mDeviceInput;
            return true;
        }

        const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) {
            mCalibrationCache.clear();
            std::ifstream input("calibration.table", std::ios::binary);
            input >> std::noskipws;
            if (mReadCache && input.good())
                std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

            length = mCalibrationCache.size();
            return length ? &mCalibrationCache[0] : nullptr;
        }

        void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) {
            std::ofstream output("calibration.table", std::ios::binary);
            output.write(reinterpret_cast<const char*>(cache), length);
        }
    }
}

