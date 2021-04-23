//
// Created by xingwg on 21-4-13.
//

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

#include "trt/plugin/slicePlugin/slicePlugin.h"
#include "trt/plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.h"

namespace nvinfer1 {
    namespace plugin {
        ILogger* gLogger{};

        // Instances of this class are statically constructed in initializePlugin.
        // This ensures that each plugin is only registered a single time, as further calls to
        // initializePlugin will be no-ops.
        template <typename CreatorType>
        class InitializePlugin {
        public:
            InitializePlugin(void* logger, const char* libNamespace)
                    : mCreator{new CreatorType{}} {
                mCreator->setPluginNamespace(libNamespace);
                bool status = getPluginRegistry()->registerCreator(*mCreator, libNamespace);
                if (logger) {
                    nvinfer1::plugin::gLogger = static_cast<nvinfer1::ILogger*>(logger);
                    if (!status) {
                        std::string errorMsg{"Could not register plugin creator:  " + std::string(mCreator->getPluginName())
                                             + " in namespace: " + std::string{mCreator->getPluginNamespace()}};
                        nvinfer1::plugin::gLogger->log(ILogger::Severity::kERROR, errorMsg.c_str());
                    } else {
                        std::string verboseMsg{
                                 "Plugin Creator registration succeeded - " + std::string{mCreator->getPluginName()}};
                        nvinfer1::plugin::gLogger->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
                    }
                }
            }

            InitializePlugin(const InitializePlugin&) = delete;
            InitializePlugin(InitializePlugin&&) = delete;

        private:
            std::unique_ptr<CreatorType> mCreator;
        };

        template <typename CreatorType>
        void initializePlugin(void* logger, const char* libNamespace) {
            static InitializePlugin<CreatorType> plugin{logger, libNamespace};
        }

    } // namespace plugin
} // namespace nvinfer1

extern "C" {
    bool initLibNvInferPlugins(void* logger, const char* libNamespace) {
        initializePlugin<nvinfer1::plugin::SlicePluginCreator>(logger, libNamespace);
        //initializePlugin<nvinfer1::plugin::InstanceNormalizationPluginCreator>(logger, libNamespace);
        return true;
    }
} // extern "C"