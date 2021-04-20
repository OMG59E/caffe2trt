#!/usr/bin/env bash

# TRT_PATH=/usr/local/TensorRT
# export LD_LIBRARY_PATH=${TRT_PATH}/lib:$LD_LIBRARY_PATH

# input shape
batch=32
channels=3
height=224
width=224

#input node name
input='data'

# output node name
output='582'

dataType=fp16

# model
deploy=../models/resnet50_ibn_a-d9d0bb7b_opt.prototxt
model=../models/resnet50_ibn_a-d9d0bb7b_opt.caffemodel
engine=../engines/resnet50_ibn_a-d9d0bb7b_opt.engine

if [ ! -d "../engines" ]; then
    mkdir "../engines"
fi

# only for int8
calibBatchSize=32
calibMaxBatches=1000
calibFirstBatch=10
dataDir=../data/imagenet/batches

../build/caffe2trt  \
--model=${model} \
--deploy=${deploy} \
--engine=${engine} \
--input=${input} \
--output=${output} \
--batch=${batch} \
--dataType=${dataType} \
--dataDir=${dataDir} \
--calibBatchSize=${calibBatchSize} \
--calibMaxBatches=${calibMaxBatches} \
--calibFirstBatch=${calibFirstBatch} \
--workspace=8192 \
--device=0