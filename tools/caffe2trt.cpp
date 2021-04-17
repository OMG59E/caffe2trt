//
// Created by xingwg on 20-1-13.
//

#include "trt/batch_stream.h"
#include "trt/calibrator.h"
#include "trt/trt_logger.h"
#include "error_check.h"

#include <NvCaffeParser.h>
#include <NvInferPlugin.h>

#include <fstream>
#include <vector>
#include <cassert>
#include <string.h>

using namespace alg::trt;
using namespace nvcaffeparser1;

static Logger gLogger;

struct Params {
    std::string deployFile;
    std::string modelFile;
    std::string engine;
    std::string dataDir;
    std::string calibrationCache{"CalibrationTable"};
    std::string input;
    std::vector<std::string> outputs;
    DataType dataType{DataType::kFLOAT};
    int device{0};
    int batchSize{1};
    int workspaceSize{512};
    int iterations{1};
    int avgRuns{1};
    int calibBatchSize{8};
    int calibMaxBatches{100};
    int calibFirstBatch{0};
    int DLACore{-1};
} gParams;

bool caffeToTRTModel(IInt8Calibrator *calibrator, nvinfer1::IHostMemory **trtModelStream) {
    // create the builder
    IBuilder *builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition *network = builder->createNetwork();
    ICaffeParser *parser = createCaffeParser();

    if (gParams.dataType == DataType::kINT8 && !builder->platformHasFastInt8()) {
        LOG(ERROR) << "The device for INT8 run since its not supported.";
        return false;
    }

    if (gParams.dataType == DataType::kHALF && !builder->platformHasFastFp16()) {
        LOG(ERROR) << "The device for FP16 run since its not supported.";
        return false;
    }

    const IBlobNameToTensor *blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
                                                              gParams.modelFile.c_str(),
                                                              *network,
                                                              gParams.dataType == DataType::kINT8 ?
                                                              DataType::kFLOAT : gParams.dataType);

    // specify which tensors are outputs
    for (const auto &s : gParams.outputs) {
        if (!blobNameToTensor->find(s.c_str())) {
            LOG(ERROR) << "Could not find output blob " << s;
            return false;
        }
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    // Build the engine
    if (gParams.dataType == DataType::kINT8) {
        builder->setAverageFindIterations(gParams.avgRuns);
        builder->setMinFindIterations(gParams.iterations);
        builder->setDebugSync(true);
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator);
    }

    if (gParams.dataType == DataType::kHALF) {
        builder->setFp16Mode(true);
    }

    if (gParams.DLACore >= 0) {
        LOG_ASSERT(gParams.DLACore < builder->getNbDLACores());
        builder->setDefaultDeviceType(DeviceType::kDLA);
        builder->setDLACore(gParams.DLACore);
        builder->setStrictTypeConstraints(true);
        builder->allowGPUFallback(true);
        if (gParams.batchSize > builder->getMaxDLABatchSize()) {
            LOG(WARNING) << "Requested batch size " << gParams.batchSize
                         << " is greater than the max DLA batch size of "
                         << builder->getMaxDLABatchSize() << ". Reducing batch size accordingly.";

            gParams.batchSize = builder->getMaxDLABatchSize();
        }
    }

    builder->setMaxWorkspaceSize(static_cast<size_t>(gParams.workspaceSize) << 20);
    builder->setMaxBatchSize(gParams.batchSize);
    ICudaEngine *engine = builder->buildCudaEngine(*network);

    LOG_ASSERT(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();

    parser->destroy();

    // serialize the engine, then close everything down
    (*trtModelStream) = engine->serialize();

    engine->destroy();

    builder->destroy();

    shutdownProtobufLibrary();

    return true;
}

void printUsage() {
    printf("\n");
    printf("Mandatory params:\n");
    printf("  --deploy=<file>         Caffe deploy file\n");
    printf("  --model=<file>          Caffe model file (default = no model, random weights used)\n");
    printf("  --engine=<file>         Engine file to serialize to or deserialize from\n");
    printf("  --dataDir=<dir>         Set calibration data dir. For int8 mode.\n");

    printf("\nOptional params:\n");
    printf("  --input=<name>          Input blob name\n");
    printf("  --output=<name>         Output blob name (can be specified multiple times)\n");
    printf("  --dataType=<name>       Run in precision mode (default = fp32). fp32 fp16 int8 kernels\n");
    printf("  --batch=N               Set batch size (default = 1)\n");
    printf("  --device=N              Set cuda device to N (default = 0)\n");
    printf("  --iterations=N          Run N iterations (default = 1)\n");
    printf("  --avgRuns=N             Set the number of averaging iterations used when timing layers (default = 1)\n");
    printf("  --workspace=N           Set workspace size in megabytes (default = 512)\n");
    printf("  --calibBatchSize=N      Set calibBatchSize to N (default = 8)\n");
    printf("  --calibMaxBatches=N     Set calibMaxBatchSize to N (default = 100)\n");
    printf("  --calibFirstBatch=N     Set calibFirstBatch to N (default = 0)\n");
    printf("  --DLACore=N             Set DLACore to N (default = -1)\n\n");
    fflush(stdout);
}

bool parseString(const char *arg, const char *name, std::string &value) {
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match) {
        value = arg + n + 3;
        LOG(INFO) << name << ": " << value;

        if (value == "int8")
            gParams.dataType = DataType::kINT8;

        if (value == "fp16")
            gParams.dataType = DataType::kHALF;

        if (value == "fp32")
            gParams.dataType = DataType::kFLOAT;
    }
    return match;
}

bool parseInt(const char *arg, const char *name, int &value) {
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match) {
        value = atoi(arg + n + 3);
        LOG(INFO) << name << ": " << value;
    }
    return match;
}

bool parseArgs(int argc, char **argv) {
    if (argc < 2) {
        printUsage();
        return false;
    }

    std::string value;
    for (int j = 1; j < argc; j++) {
        if (parseString(argv[j], "model", gParams.modelFile)
            || parseString(argv[j], "deploy", gParams.deployFile)
            || parseString(argv[j], "engine", gParams.engine)
            || parseString(argv[j], "calib", gParams.calibrationCache)
            || parseString(argv[j], "dataDir", gParams.dataDir)
            || parseString(argv[j], "input", gParams.input)
            || parseString(argv[j], "dataType", value))
            continue;

        std::string output;
        if (parseString(argv[j], "output", output)) {
            gParams.outputs.push_back(output);
            continue;
        }

        if (parseInt(argv[j], "batch", gParams.batchSize)
            || parseInt(argv[j], "iterations", gParams.iterations)
            || parseInt(argv[j], "avgRuns", gParams.avgRuns)
            || parseInt(argv[j], "device", gParams.device)
            || parseInt(argv[j], "workspace", gParams.workspaceSize)
            || parseInt(argv[j], "calibBatchSize", gParams.calibBatchSize)
            || parseInt(argv[j], "calibMaxBatches", gParams.calibMaxBatches)
            || parseInt(argv[j], "calibFirstBatch", gParams.calibFirstBatch)
            || parseInt(argv[j], "DLACore", gParams.DLACore))
            continue;

        LOG(ERROR) << "Unknown argument: " << argv[j];
        return false;
    }

    return true;
}

int main(int argc, char **argv) {
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    if (!parseArgs(argc, argv))
        return -1;

    CUDACHECK(cudaSetDevice(gParams.device));

    initLibNvInferPlugins(&gLogger, "");

    if (gParams.outputs.size() == 0 && !gParams.deployFile.empty()) {
        LOG(ERROR) << "At least one network output must be defined";
        return -1;
    }

    BatchStream *calibrationStream{nullptr};
    Int8EntropyCalibrator *calibrator{nullptr};

    if (gParams.dataType == DataType::kINT8) {
        calibrationStream = new BatchStream(gParams.dataDir, gParams.calibBatchSize, gParams.calibMaxBatches);
        calibrator = new Int8EntropyCalibrator(*calibrationStream, gParams.calibFirstBatch);
    }

    IHostMemory *trtModelStream{nullptr};

    if (!caffeToTRTModel(calibrator, &trtModelStream)) {
        LOG(INFO) << "Caffe to trt model failed.";
        return -1;
    }

    std::ofstream trtModelFile(gParams.engine.c_str());

    trtModelFile.write((char *) trtModelStream->data(), trtModelStream->size());

    LOG(INFO) << "Convert model to tensor model cache: " << gParams.engine.c_str() << " completed.";

    trtModelFile.close();

    trtModelStream->destroy();

    if (gParams.dataType == DataType::kINT8) {
        delete calibrationStream;
        calibrationStream = nullptr;

        delete calibrator;
        calibrator = nullptr;
    }

    return 0;
}