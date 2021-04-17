//
// Created by xingwg on 20-1-13.
//

#include "trt/batch_stream.h"
#include "error_check.h"
#include <algorithm>

namespace alg {
    namespace trt {
        BatchStream::BatchStream(const std::string batchDir, int batchSize, int maxBatches)
                : mBatchSize(batchSize)
                , mMaxBatches(maxBatches)
                , mBatchDir(batchDir) {
            FILE* file = fopen((mBatchDir + "/batch0").c_str(), "rb");
            int d[4];
            size_t readSize = fread(d, sizeof(int), 4, file);
            LOG_ASSERT(readSize == 4);

            mDims = nvinfer1::DimsNCHW{d[0], d[1], d[2], d[3]};
            fclose(file);

            mImageSize = mDims.c() * mDims.h() * mDims.w();
            mBatch.resize(mBatchSize * mImageSize, 0);

            mFileBatch.resize(mDims.n() * mImageSize, 0);

            reset(0);
        }

        BatchStream::~BatchStream() {}

        void BatchStream::reset(int firstBatch) {
            mBatchCount = 0;
            mFileCount = 0;
            mFileBatchPos = mDims.n();
            skip(firstBatch);
        }

        bool BatchStream::next() {
            if (mBatchCount == mMaxBatches)
                return false;

            for (int csize=1, batchPos=0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize) {
                LOG_ASSERT(mFileBatchPos > 0 && mFileBatchPos <= mDims.n());
                if (mFileBatchPos == mDims.n() && !update())
                    return false;

                csize = std::min(mBatchSize - batchPos, mDims.n() - mFileBatchPos);
                std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
            }

            mBatchCount++;

            return true;
        }

        void BatchStream::skip(int skipCount) {
            if (mBatchSize >= mDims.n() && mBatchSize%mDims.n() == 0 && mFileBatchPos == mDims.n()) {
                mFileCount += skipCount * mBatchSize / mDims.n();
                return;
            }

            int x = mBatchCount;
            for(int i = 0; i < skipCount; i++)
                next();

            mBatchCount = x;
        }

        bool BatchStream::update() {
            std::string inputFileName = mBatchDir + "/batch" + std::to_string(mFileCount++);

            FILE* file = fopen(inputFileName.c_str(), "rb");
            if (!file)
                return false;

            int d[4];
            size_t readSize = fread(d, sizeof(int), 4, file);
            LOG_ASSERT(readSize == 4);
            LOG_ASSERT(mDims.n() == d[0] && mDims.c() == d[1] && mDims.h() == d[2] && mDims.w() == d[3]);

            size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.n()*mImageSize, file);

            LOG_ASSERT(readInputCount == size_t(mDims.n() * mImageSize));

            fclose(file);
            mFileBatchPos = 0;

            return true;
        }

        float* BatchStream::getBatch() {
            return &mBatch[0];
        }

        int BatchStream::getBatchesRead() const {
            return mBatchCount;
        }

        int BatchStream::getBatchSize() const {
            return mBatchSize;
        }

        nvinfer1::DimsNCHW BatchStream::getDims() const {
            return mDims;
        }

        float* BatchStream::getFileBatch() {
            return &mFileBatch[0];
        }
    }
}