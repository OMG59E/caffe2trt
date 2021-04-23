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

inline std::string NvJpegGetErrorString(nvjpegStatus_t status) {
    switch (status) {
        case NVJPEG_STATUS_NOT_INITIALIZED:
            return "NVJPEG_STATUS_NOT_INITIALIZED";
        case NVJPEG_STATUS_INVALID_PARAMETER:
            return "NVJPEG_STATUS_INVALID_PARAMETER";
        case NVJPEG_STATUS_BAD_JPEG:
            return "NVJPEG_STATUS_BAD_JPEG";
        case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
            return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
        case NVJPEG_STATUS_ALLOCATOR_FAILURE:
            return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
        case NVJPEG_STATUS_EXECUTION_FAILED:
            return "NVJPEG_STATUS_EXECUTION_FAILED";
        case NVJPEG_STATUS_ARCH_MISMATCH:
            return "NVJPEG_STATUS_ARCH_MISMATCH";
        case NVJPEG_STATUS_INTERNAL_ERROR:
            return "NVJPEG_STATUS_INTERNAL_ERROR";
        case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
            return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
    }
    return "Unknown nvjpeg error";
}

#define NVJPEGCHECK(status)   \
    do  \
    {  \
        auto ret = status;  \
        if (ret != 0)  \
        {  \
            LOG(FATAL) << "NVJPEG ERROR: " << NvJpegGetErrorString(ret); \
        }   \
    } while (0)

#endif //ALGORITHMS_ERROR_CHECK_H
