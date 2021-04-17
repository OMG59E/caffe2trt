//
// Created by xingwg on 20-4-22.
//

#include "trt/plugin/slicePlugin/slicePlugin.h"
#include "error_check.h"
#include <algorithm>

using namespace nvinfer1;
using nvinfer1::plugin::Slice;
using nvinfer1::plugin::SlicePluginCreator;

static const char *SLICE_PLUGIN_VERSION{"1"};
static const char *SLICE_PLUGIN_NAME{"Slice_TRT"};

PluginFieldCollection SlicePluginCreator::mFC{};
std::vector<PluginField> SlicePluginCreator::mPluginAttributes;

// 传入 caffe layer 中的变量
Slice::Slice(int axis, int slice_dim, std::vector<int> slice_points,
        DataType data_type, int channel_in, int height_in, int width_in)
        : axis_{axis}
        , slice_dim_{slice_dim}
        , data_type_{data_type} {
    slice_points_.swap(slice_points);
    num_output_ = slice_points_.size() + 1;
    bottom_shape_ = DimsCHW(channel_in, height_in, width_in);

    int prev = 0;
    std::vector<int> slices;
    slices.clear();
    for (int i = 0; i < slice_points_.size(); ++i) {
        CHECK_GT(slice_points_[i], prev);
        slices.push_back(slice_points_[i] - prev);
        prev = slice_points_[i];
    }
    slices.push_back(bottom_shape_.d[axis_ - 1] - prev);

    top_shapes_.clear();
    DimsCHW shape = bottom_shape_;
    for (int i = 0; i < num_output_; ++i) {
        shape.d[axis_ - 1] = slices[i];
        top_shapes_.push_back(shape);
    }
}

Slice::Slice(const void *data, size_t length) {
    const char *d = reinterpret_cast<const char *>(data);
    const char *a = d;

    axis_ = read<int>(d);
    slice_dim_ = read<int>(d);
    num_output_ = read<int>(d);
    slice_points_.clear();
    for (int i = 0; i < num_output_ - 1; ++i)
        slice_points_.push_back(read<int>(d));
    bottom_shape_ = read<DimsCHW>(d);
    data_type_ = read<DataType>(d);
    LOG_ASSERT(d == a + length);

    int prev = 0;
    std::vector<int> slices;
    slices.clear();
    for (int i = 0; i < slice_points_.size(); ++i) {
        CHECK_GT(slice_points_[i], prev);
        slices.push_back(slice_points_[i] - prev);
        prev = slice_points_[i];
    }

    slices.push_back(bottom_shape_.d[axis_ - 1] - prev);

    top_shapes_.clear();
    DimsCHW shape = bottom_shape_;
    for (int i = 0; i < num_output_; ++i) {
        shape.d[axis_ - 1] = slices[i];
        top_shapes_.push_back(shape);
    }
}

Slice::~Slice() {

}

int Slice::getNbOutputs() const {
    return num_output_;
}

Dims Slice::getOutputDimensions(int index, const Dims *inputs, int nbInputDims) {
    LOG_ASSERT(nbInputDims == 1);
    LOG_ASSERT(index >= 0 && index < num_output_);
    // Output dimensions
    return top_shapes_[index];
}

int Slice::initialize() {
    return STATUS_SUCCESS;
}

void Slice::terminate() {

}

size_t Slice::getWorkspaceSize(int maxBatchSize) const {
    return 0;
}

size_t Slice::getSerializationSize() const {
    return sizeof(int) * (3 + num_output_ - 1) + sizeof(DimsCHW) + sizeof(DataType);
}

void Slice::serialize(void *buffer) const {
    char *d = reinterpret_cast<char *>(buffer);
    char *a = d;

    write(d, axis_);
    write(d, slice_dim_);
    write(d, num_output_);
    for (int i = 0; i < num_output_ - 1; ++i)
        write(d, slice_points_[i]);
    write(d, bottom_shape_);
    write(d, data_type_);
    LOG_ASSERT(d == a + getSerializationSize());
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Slice::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) {}

// Detach the plugin object from its execution context.
void Slice::detachFromContext() {}

// Return true if output tensor is broadcast across a batch.
bool Slice::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const {
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool Slice::canBroadcastInputAcrossBatch(int inputIndex) const {
    return false;
}

// Set plugin namespace
void Slice::setPluginNamespace(const char *pluginNamespace) {
    mPluginNamespace = pluginNamespace;
}

const char *Slice::getPluginNamespace() const {
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType Slice::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
    LOG_ASSERT(nbInputs == 1);
    LOG_ASSERT(index >= 0 && index < num_output_);
    return inputTypes[0];
}

void Slice::configurePlugin(const PluginTensorDesc* in, int32_t nbInput,
        const PluginTensorDesc* out, int32_t nbOutput) {
    LOG_ASSERT(nbInput == 1);
    LOG_ASSERT(nbOutput == num_output_);
    LOG_ASSERT(in[0].dims.nbDims == 3);
    LOG_ASSERT(in[0].type == out[0].type);
    LOG_ASSERT(in[0].format == TensorFormat::kLINEAR && out[0].format == TensorFormat::kLINEAR);
    data_type_ = in[0].type;
}

bool Slice::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut,
        int32_t nbInputs, int32_t nbOutputs) const {
    LOG_ASSERT(nbInputs == 1 && nbOutputs == num_output_ && pos < nbInputs + nbOutputs);
    return inOut[pos].type == inOut[0].type && inOut[pos].format == TensorFormat::kLINEAR;
}

const char *Slice::getPluginType() const {
    return "Slice_TRT";
}

const char *Slice::getPluginVersion() const {
    return "1";
}

void Slice::destroy() {
    delete this;
}

IPluginV2IOExt *Slice::clone() const {
    auto plugin = new Slice(axis_, slice_dim_, slice_points_,
            data_type_, bottom_shape_.c(), bottom_shape_.h(), bottom_shape_.w());
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

SlicePluginCreator::SlicePluginCreator() {
    mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("slice_dim", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("slice_point", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("channel_in", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("height_in", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("width_in", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = int(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

const char *SlicePluginCreator::getPluginName() const {
    return SLICE_PLUGIN_NAME;
}

const char *SlicePluginCreator::getPluginVersion() const {
    return SLICE_PLUGIN_VERSION;
}

const PluginFieldCollection *SlicePluginCreator::getFieldNames() {
    return &mFC;
}

IPluginV2IOExt *SlicePluginCreator::createPlugin(const char * /*name*/, const PluginFieldCollection *fc) {
    const PluginField *fields = fc->fields;

    int axis = 0;
    int slice_dim = 0;
    std::vector<int> slice_points;
    int channel_in = 0;
    int height_in = 0;
    int width_in = 0;

    for (int i = 0; i < fc->nbFields; i++) {
        const char *attrName = fields[i].name;
        if (!strcmp(attrName, "axis")) {
            LOG_ASSERT(fields[i].type == PluginFieldType::kINT32);
            axis = *(static_cast<const int *>(fields[i].data));
        } else if (!strcmp(attrName, "slice_dim")) {
            LOG_ASSERT(fields[i].type == PluginFieldType::kINT32);
            slice_dim = *(static_cast<const int *>(fields[i].data));
        } else if (!strcmp(attrName, "slice_point")) {
            LOG_ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            const auto *data = static_cast<const int *>(fields[i].data);
            slice_points.clear();
            for (int k = 0; k < size; ++k)
                slice_points.push_back(data[k]);
        } else if (!strcmp(attrName, "channel_in")) {
            LOG_ASSERT(fields[i].type == PluginFieldType::kINT32);
            channel_in = *(static_cast<const int *>(fields[i].data));
        } else if (!strcmp(attrName, "height_in")) {
            LOG_ASSERT(fields[i].type == PluginFieldType::kINT32);
            height_in = *(static_cast<const int *>(fields[i].data));
        } else if (!strcmp(attrName, "width_in")) {
            LOG_ASSERT(fields[i].type == PluginFieldType::kINT32);
            width_in = *(static_cast<const int *>(fields[i].data));
        }
    }

    auto plugin = new Slice(axis, slice_dim, slice_points, DataType::kFLOAT, channel_in, height_in, width_in);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2IOExt *SlicePluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) {
    // This object will be deleted when the network is destroyed, which will call Concat::destroy()
    auto plugin = new Slice(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
