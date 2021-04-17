//
// Created by xingwg on 20-1-13.
//

#ifndef ALGORITHMS_ERROR_CHECK_H
#define ALGORITHMS_ERROR_CHECK_H

#include <cudnn.h>
#include <glog/logging.h>
#include <nvjpeg.h>

#define CUDACHECK(status)   \
    do  \
    {  \
        auto ret = status;  \
        if (ret != 0)  \
        {  \
            LOG(FATAL) << "CUDA ERROR: " << ret << " - " << cudaGetErrorString(status); \
        }   \
    } while (0)

#define CUDNNCHECK(status)   \
    do  \
    {  \
        auto ret = status;  \
        if (ret != 0)  \
        {  \
            LOG(FATAL) << "CUDNN ERROR: " << ret << " - " << cudnnGetErrorString(status); \
        }   \
    } while (0)

#define CUBLASCHECK(status)   \
    do  \
    {  \
        auto ret = status;  \
        if (ret != 0)  \
        {  \
            LOG(FATAL) << "CUBLAS ERROR: " << ret << " - " << cublasGetErrorString(status); \
        }   \
    } while (0)

#define CUDA_FREE(ptr) \
if (ptr) \
{  \
    CUDACHECK(cudaFree(ptr)); \
    ptr = nullptr; \
}

#define SAFE_FREE(ptr) \
if (ptr) \
{  \
    delete ptr; \
    ptr = nullptr; \
}

#define ZMQ_FREE(ptr) \
if (ptr) \
{  \
    ptr->close(); \
    delete ptr; \
    ptr = nullptr; \
}

#endif //ALGORITHMS_ERROR_CHECK_H
