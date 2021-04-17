//
// Created by xingwg on 21-4-13.
//

#ifndef IBN_TRT_PLUGIN_H
#define IBN_TRT_PLUGIN_H

#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>

// Enumerator for status
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1 {
    namespace plugin {
        class BaseCreator : public IPluginCreator {
        public:
            void setPluginNamespace(const char* libNamespace) override {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const override {
                return mNamespace.c_str();
            }

        protected:
            std::string mNamespace;
        };

        // Write values into buffer
        template <typename T>
        void write(char*& buffer, const T& val) {
            *reinterpret_cast<T*>(buffer) = val;
            buffer += sizeof(T);
        }

        // Read values from buffer
        template <typename T>
        T read(const char*& buffer) {
            T val = *reinterpret_cast<const T*>(buffer);
            buffer += sizeof(T);
            return val;
        }
    } // namespace plugin
} // namespace nvinfer1

#endif //IBN_TRT_PLUGIN_H
