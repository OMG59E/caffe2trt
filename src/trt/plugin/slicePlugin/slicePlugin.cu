
#include "error_check.h"
#include "base_types.h"
#include "trt/plugin/slicePlugin/slicePlugin.h"
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <iostream>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

template<typename Dtype>
__global__ void SliceKernel(const int nthreads, const Dtype *in_data,
                            const int num_slices, const int slice_size,
                            const int bottom_slice_axis, const int top_slice_axis,
                            const int offset_slice_axis, Dtype *out_data) {
    CUDA_KERNEL_LOOP(idx, nthreads) {
        const int total_slice_size = slice_size * top_slice_axis;
        const int slice_num = idx / total_slice_size;
        const int slice_index = idx % total_slice_size;
        const int bottom_index = slice_index +
                                 (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;

        out_data[idx] = in_data[bottom_index];
    }
}

template<typename Dtype>
void SliceForward(cudaStream_t stream, const Dtype *bottom_data, int num_slices, int slice_axis, int slice_size,
        int bottom_slice_axis, const std::vector<nvinfer1::DimsCHW>& output_shapes, void **outputs) {
    int offset_slice_axis = 0;

    for (int i = 0; i < output_shapes.size(); ++i) {
        Dtype *top_data = reinterpret_cast<Dtype*>(outputs[i]) ;
        const int top_slice_axis = output_shapes[i].d[slice_axis - 1];
        const int top_slice_size = top_slice_axis * slice_size;    // top[i]->shape(axis)
        const int nthreads = top_slice_size * num_slices;
        SliceKernel << < CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, stream >> > (nthreads,
                bottom_data,
                num_slices,
                slice_size,
                bottom_slice_axis,
                top_slice_axis,
                offset_slice_axis,
                top_data);
        CUDACHECK(cudaStreamSynchronize(stream));
        offset_slice_axis += top_slice_axis;
    }
}


int Slice::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
    int num_slices, slice_size, bottom_slice_axis;

    int size = 1;
    for (int i = 0; i < axis_ - 1; ++i)
        size *= bottom_shape_.d[i];
    num_slices = batchSize * size;          // bottom->count(0, axis)

    size = 1;
    for (int i = axis_; i < bottom_shape_.nbDims; ++i)
        size *= bottom_shape_.d[i];
    slice_size = size;                                // bottom->count(axis+1)
    bottom_slice_axis = bottom_shape_.d[axis_ - 1];     // bottom->shape(axis)
    LOG_ASSERT(top_shapes_.size() == num_output_);

    if (data_type_ == DataType::kFLOAT) {
        auto bottom_data = reinterpret_cast<const float*>(inputs[0]);
        SliceForward(stream, bottom_data,
                num_slices, axis_, slice_size, bottom_slice_axis, top_shapes_, outputs);
    } else if (data_type_ == DataType::kHALF) {
        auto bottom_data = reinterpret_cast<const __half*>(inputs[0]);
        SliceForward(stream, bottom_data,
                num_slices, axis_, slice_size, bottom_slice_axis, top_shapes_, outputs);
    } else if (data_type_ == DataType::kINT8) {
        auto bottom_data = reinterpret_cast<const int8_t*>(inputs[0]);
        SliceForward(stream, bottom_data,
                     num_slices, axis_, slice_size, bottom_slice_axis, top_shapes_, outputs);
    };

    return 0;
}
