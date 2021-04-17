//
// Created by xingwg on 20-1-13.
//

#ifndef ALGORITHMS_TRT_LOGGER_H
#define ALGORITHMS_TRT_LOGGER_H

#include <NvInfer.h>
#include <iostream>
#include <glog/logging.h>

using namespace nvinfer1;

namespace alg {
    namespace trt {
        class Logger : public ILogger {
        public:
            Logger(Severity severity = Severity::kWARNING)
                    : reportableSeverity(severity) {
            }

            void log(Severity severity, const char* msg) override {
                // suppress messages with severity enum value greater than the reportable
                //if (severity > reportableSeverity)
                //    return;

                switch (severity) {
                    case Severity::kINTERNAL_ERROR:
                        LOG(ERROR) << "INTERNAL_ERROR: " << msg;
                        break;
                    case Severity::kERROR:
                        LOG(ERROR) << "ERROR: " << msg;
                        break;
                    case Severity::kWARNING:
                        LOG(WARNING) << "WARNING: " << msg;
                        break;
                    case Severity::kINFO:
                        LOG(INFO) << "INFO: " << msg;
                        break;
                    case Severity::kVERBOSE:
                        DLOG(INFO) << "VERBOSE: " << msg;
                        break;
                }
            }

            Severity reportableSeverity;
        };
    }
}

#endif //ALGORITHMS_LOGGER_H
