//
// Created by xingwg on 20-1-13.
//

#ifndef ALGORITHMS_CALIBRATOR_H
#define ALGORITHMS_CALIBRATOR_H

#include "trt/batch_stream.h"

using namespace nvinfer1;

namespace alg {
    namespace trt {
        class Int8EntropyCalibrator : public IInt8EntropyCalibrator2 {
        public:
            Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache = true);

            virtual ~Int8EntropyCalibrator();

            int getBatchSize() const override;

            bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

            const void* readCalibrationCache(size_t& length) override;

            void writeCalibrationCache(const void* cache, size_t length) override;

        private:
            BatchStream mStream;
            bool mReadCache{true};

            size_t mInputCount;
            void* mDeviceInput{nullptr};

            std::vector<char> mCalibrationCache;
        };
    }
}

#endif //ALGORITHMS_CALIBRATOR_H