#!/usr/bin/env bash

echo "compile trtcaffe proto..."
protoc -I="../src/trt/parsers/caffe/proto" --cpp_out="../src/trt/parsers/caffe/proto" "../src/trt/parsers/caffe/proto/trtcaffe.proto"
if [ ! -d "../include/trt/parsers/caffe/proto" ]; then
    mkdir "../include/trt/parsers/caffe/proto"
fi
echo "mv ../src/trt/parsers/caffe/proto/trtcaffe.pb.h to ../include/trt/parsers/caffe/proto"
mv "../src/trt/parsers/caffe/proto/trtcaffe.pb.h" "../include/trt/parsers/caffe/proto"