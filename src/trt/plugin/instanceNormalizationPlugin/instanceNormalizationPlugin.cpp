/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdexcept>
#include "trt/plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.h"
#include "error_check.h"

using namespace nvinfer1;
using nvinfer1::plugin::InstanceNormalizationPlugin;
using nvinfer1::plugin::InstanceNormalizationPluginCreator;

// This is derived from: https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
inline float half_to_float_fast(unsigned short value) {
    union F32 {
        unsigned int u;
        float f;
    };
    static const F32 magic = {(254 - 15) << 23};
    static const F32 was_infnan = {(127 + 16) << 23};
    F32 result;
    result.u = (value & 0x7fff) << 13; // exponent/mantissa bits
    result.f *= magic.f;               // exponent adjust
    if (result.f >= was_infnan.f) { // make sure Inf/NaN survive
        result.u |= 255 << 23;
    }
    result.u |= (value & 0x8000) << 16; // sign bit
    return result.f;
}

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype, cudnnDataType_t *cudnn_dtype) {
    switch (trt_dtype) {
        case nvinfer1::DataType::kFLOAT:
            *cudnn_dtype = CUDNN_DATA_FLOAT;
            break;
        case nvinfer1::DataType::kHALF:
            *cudnn_dtype = CUDNN_DATA_HALF;
            break;
        default:
            return CUDNN_STATUS_BAD_PARAM;
    }
    return CUDNN_STATUS_SUCCESS;
}

static const char *INSTANCE_PLUGIN_VERSION{"1"};
static const char *INSTANCE_PLUGIN_NAME{"InstanceNormalization_TRT"};

PluginFieldCollection InstanceNormalizationPluginCreator::mFC{};
std::vector<PluginField> InstanceNormalizationPluginCreator::mPluginAttributes;

//InstanceNormalizationPlugin::InstanceNormalizationPlugin(
//        float epsilon, nvinfer1::Weights const &scale, nvinfer1::Weights const &bias)
//        : _epsilon(epsilon)
//        , _initialized(false) {
//    LOG_ASSERT(scale.count == bias.count);
//    LOG_ASSERT(scale.type == bias.type);
//
//    _data_type = scale.type;
//
//    if (scale.type == nvinfer1::DataType::kFLOAT) {
//        _h_scale.assign((float *) scale.values, (float *) scale.values + scale.count);
//    } else if (scale.type == nvinfer1::DataType::kHALF) {
//        _h_scale.reserve(scale.count);
//        for (int c = 0; c < scale.count; ++c) {
//            unsigned short value = ((unsigned short *) scale.values)[c];
//            _h_scale.push_back(half_to_float_fast(value));
//        }
//    } else {
//        LOG(FATAL) << "Unsupported scale dtype";
//    }
//
//    if (bias.type == nvinfer1::DataType::kFLOAT) {
//        _h_bias.assign((float *) bias.values, (float *) bias.values + bias.count);
//    } else if (bias.type == nvinfer1::DataType::kHALF) {
//        _h_bias.reserve(bias.count);
//        for (int c = 0; c < bias.count; ++c) {
//            unsigned short value = ((unsigned short *) bias.values)[c];
//            _h_bias.push_back(half_to_float_fast(value));
//        }
//    } else {
//        LOG(FATAL) << "Unsupported bias dtype";
//    }
//}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(float epsilon,
        DataType dataType, std::vector<float> const &h_scale, std::vector<float> const &h_bias)
        : _epsilon(epsilon)
        , _initialized(false) {
    LOG_ASSERT(h_scale.size() == h_bias.size());
    _data_type = dataType;
    _h_scale.assign(h_scale.begin(), h_scale.end());
    _h_bias.assign(h_bias.begin(), h_bias.end());
}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(
        void const *data, size_t length)
        : _initialized(false) {
    const char *d = reinterpret_cast<const char *>(data);
    const char *a = d;

    _epsilon = read<float>(d);
    _maxBatchSize = read<int>(d);
    _shape = read<DimsCHW>(d);
    _data_type = read<DataType>(d);
    if (_data_type == DataType::kFLOAT)
        std::cout << "DataType::kFLOAT" << std::endl;
    else if (_data_type == DataType::kHALF)
        std::cout << "DataType::kHALF" << std::endl;
    else
        std::cout << "DataType::kINT8" << std::endl;
    _h_scale.clear();
    auto scale_size = read<size_t>(d);
    for (int i=0; i<scale_size; ++i)
        _h_scale.push_back(read < float > (d));
    _h_bias.clear();
    auto bias_size = read < size_t > (d);
    for (int i = 0; i < bias_size; ++i)
        _h_bias.push_back(read < float > (d));
    LOG_ASSERT(d == a + length);

    initialize();
}

InstanceNormalizationPlugin::~InstanceNormalizationPlugin() {
    terminate();
}

// InstanceNormalizationPlugin returns one output.
int InstanceNormalizationPlugin::getNbOutputs() const {
    return 1;
}

Dims InstanceNormalizationPlugin::getOutputDimensions(
        int index, const Dims *inputs, int nbInputDims) {
    LOG_ASSERT(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    return inputs[index];
}

int InstanceNormalizationPlugin::initialize() {
    _initialized = true;
    // CUDNNCHECK(cudnnCreate(&_cudnn_handle));
    // CUDNNCHECK(cudnnCreateTensorDescriptor(&_b_desc));
    // CUDNNCHECK(cudnnCreateTensorDescriptor(&_x_desc));
    // CUDNNCHECK(cudnnCreateTensorDescriptor(&_y_desc));
    size_t nchan_bytes = _shape.c() * sizeof(float);
    CUDACHECK(cudaMalloc((void **) &_d_scale, _maxBatchSize * nchan_bytes));
    CUDACHECK(cudaMalloc((void **) &_d_bias, _maxBatchSize * nchan_bytes));
    // Note: We repeat the data for each batch entry so that we can do the full
    //       computation in a single CUDNN call in enqueue().
    for (int i = 0; i < _maxBatchSize; ++i) {
        CUDACHECK(cudaMemcpy(_d_scale + i * _shape.c(), _h_scale.data(), nchan_bytes, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(_d_bias + i * _shape.c(), _h_bias.data(), nchan_bytes, cudaMemcpyHostToDevice));
    }
    return 0;
}

void InstanceNormalizationPlugin::terminate() {
    if (!_initialized)
        return;
    CUDA_FREE(_d_bias);
    CUDA_FREE(_d_scale);
    // CUDNNCHECK(cudnnDestroyTensorDescriptor(_y_desc));
    // CUDNNCHECK(cudnnDestroyTensorDescriptor(_x_desc));
    // CUDNNCHECK(cudnnDestroyTensorDescriptor(_b_desc));
    // CUDNNCHECK(cudnnDestroy(_cudnn_handle));
    _initialized = false;
}

size_t InstanceNormalizationPlugin::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

int InstanceNormalizationPlugin::enqueue(int batchSize,
        const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) {

    int n = batchSize;
    int c = _shape.c();
    int h = _shape.h();
    int w = _shape.w();

    cudnnDataType_t cudnn_dtype;
    CUDNNCHECK(convert_trt2cudnn_dtype(_data_type, &cudnn_dtype));
    LOG_ASSERT(_b_desc && _x_desc && _y_desc);
    CUDNNCHECK(cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, n * c, 1, 1));
    CUDNNCHECK(cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
    CUDNNCHECK(cudnnSetTensor4dDescriptor(_y_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c, h, w));
    float alpha = 1;
    float beta = 0;
    void const *x_ptr = inputs[0];
    void *y_ptr = outputs[0];
    CUDNNCHECK(cudnnSetStream(_cudnn_handle, stream));
    // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
    //       overflows (NaNs) for fp32 data in some circumstances. The lower-
    //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
    //       acceptable.
    CUDNNCHECK(cudnnBatchNormalizationForwardTraining(_cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta,
                                                      _x_desc, x_ptr, _y_desc, y_ptr, _b_desc, _d_scale, _d_bias, 1.,
                                                      nullptr, nullptr, _epsilon, nullptr, nullptr));
    return 0;

}

size_t InstanceNormalizationPlugin::getSerializationSize() const {
     return sizeof(float) + sizeof(int) + sizeof(DimsCHW) + sizeof(DataType)
        + 2*sizeof(size_t) + (_h_bias.size() + _h_scale.size()) * sizeof(float);
}

void InstanceNormalizationPlugin::serialize(void *buffer) const {
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, _epsilon);
    write(d, _maxBatchSize);
    write(d, _shape);
    write(d, _data_type);
    write(d, _h_scale.size());
    for (auto &val : _h_scale)
        write(d, val);

    write(d, _h_bias.size());
    for (auto &val : _h_bias)
        write(d, val);
    LOG_ASSERT(d == a + getSerializationSize());
}

nvinfer1::DataType InstanceNormalizationPlugin::getOutputDataType(
        int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
    LOG_ASSERT(inputTypes && nbInputs == 1 && index == 0);
    return inputTypes[0];
}

bool InstanceNormalizationPlugin::supportsFormat(DataType type, PluginFormat format) const {
    return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW;
}

void InstanceNormalizationPlugin::configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                          const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                          const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) {
    LOG_ASSERT(nbInputs == 1 && nbOutputs == 1);
    LOG_ASSERT(inputDims[0].nbDims == 3 && outputDims[0].nbDims == 3);
    LOG_ASSERT(inputTypes[0] == outputTypes[0] && floatFormat == PluginFormat::kNCHW);
    _shape.c() = inputDims[0].d[0];
    _shape.h() = inputDims[0].d[1];
    _shape.w() = inputDims[0].d[2];
    _maxBatchSize = maxBatchSize;
    _data_type = inputTypes[0];
}

bool InstanceNormalizationPlugin::canBroadcastInputAcrossBatch(int inputIndex) const {
    return false;
}

bool InstanceNormalizationPlugin::isOutputBroadcastAcrossBatch(int outputIndex,
        const bool *inputIsBroadcasted, int nbInputs) const {
    return false;
}

const char *InstanceNormalizationPlugin::getPluginType() const {
    return INSTANCE_PLUGIN_NAME;
}

const char *InstanceNormalizationPlugin::getPluginVersion() const {
    return INSTANCE_PLUGIN_VERSION;
}

void InstanceNormalizationPlugin::destroy() {
    delete this;
}

IPluginV2Ext *InstanceNormalizationPlugin::clone() const {
    auto plugin = new InstanceNormalizationPlugin{_epsilon, _data_type, _h_scale, _h_bias};
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

// Set plugin namespace
void InstanceNormalizationPlugin::setPluginNamespace(const char *pluginNamespace) {
    mPluginNamespace = pluginNamespace;
}

const char *InstanceNormalizationPlugin::getPluginNamespace() const {
    return mPluginNamespace;
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void InstanceNormalizationPlugin::attachToContext(cudnnContext *cudnnContext,
        cublasContext *cublasContext, IGpuAllocator *gpuAllocator) {
    _cudnn_handle = cudnnContext;
    cudnnCreateTensorDescriptor(&_b_desc);
    cudnnCreateTensorDescriptor(&_x_desc);
    cudnnCreateTensorDescriptor(&_y_desc);
}

// Detach the plugin object from its execution context.
void InstanceNormalizationPlugin::detachFromContext() {
    cudnnDestroyTensorDescriptor(_y_desc);
    cudnnDestroyTensorDescriptor(_x_desc);
    cudnnDestroyTensorDescriptor(_b_desc);
}

// InstanceNormalizationPluginCreator methods
InstanceNormalizationPluginCreator::InstanceNormalizationPluginCreator() {
    mPluginAttributes.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("scales", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char *InstanceNormalizationPluginCreator::getPluginName() const {
    return INSTANCE_PLUGIN_NAME;
}

const char *InstanceNormalizationPluginCreator::getPluginVersion() const {
    return INSTANCE_PLUGIN_VERSION;
}

const PluginFieldCollection *InstanceNormalizationPluginCreator::getFieldNames() {
    return &mFC;
}

IPluginV2Ext *InstanceNormalizationPluginCreator::createPlugin(const char *name,
        const nvinfer1::PluginFieldCollection *fc) {
    std::vector<float> scaleValues;
    std::vector<float> biasValues;
    float epsilon{};
    const PluginField *fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
        const char *attrName = fields[i].name;
        if (!strcmp(attrName, "epsilon")) {
            LOG_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            epsilon = *(static_cast<const float *>(fields[i].data));
        } else if (!strcmp(attrName, "scales")) {
            LOG_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            scaleValues.reserve(size);
            const auto *w = static_cast<const float *>(fields[i].data);
            for (int j = 0; j < size; j++) {
                scaleValues.push_back(*w);
                w++;
            }
        } else if (!strcmp(attrName, "bias")) {
            LOG_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            int size = fields[i].length;
            biasValues.reserve(size);
            const auto *w = static_cast<const float *>(fields[i].data);
            for (int j = 0; j < size; j++) {
                biasValues.push_back(*w);
                w++;
            }
        }
    }

    auto obj = new InstanceNormalizationPlugin(epsilon, DataType::kFLOAT, scaleValues, biasValues);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext *InstanceNormalizationPluginCreator::deserializePlugin(const char *name,
        const void *serialData, size_t serialLength) {
    auto obj = new InstanceNormalizationPlugin{serialData, serialLength};
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
