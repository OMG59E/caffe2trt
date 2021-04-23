# Caffe2TRT

欢迎使用 **caffe2trt**，您可以通过 **caffe2trt** 很方便将 **caffemodel** 转换为 **trt engine**，本仓库的代码基本都是来自 **[tensorrt](https://github.com/NVIDIA/TensorRT.git)** 官方仓库。

# Support Layer

TensorRT 官方目前支持Layer，见[链接](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html)

| Layer | FP32 | FP16 | INT8 |
| ----- | ---- | ---- | ---- |
| Slice |  yes  |  yes  |  yes  |
| InstanceNorm |  yes |  yes  |  yes  |


# Support Model

**Model Performance**

| Model          | Original Model | TRT FP32 | TRT FP16 | TRT INT8 |     |
| -------------- | -------------- | -------- | -------- | -------- | --- |
| ResNet50-IBN-a | 77.118/93.572  | 77.118/93.572 | 77.142/93.574 | 75.676/92.858 | top1/top5 |

**Model Latency**

模型推理时间均是BatchSize 32，且推理1000次的平均前向时间/平均前向时间+预处理, 即T/1000, 单位ms

| Model          | Input Size | Batch | Original Model | TRT FP32 | TRT FP16 | TRT INT8 |  Device   |
| -------------- | ---------- | ----- | -------------- | -------- | -------- | -------- | --- |
| ResNet50-IBN-a |  3x224x224 |   32  | 103.44/106.65  | 72.71/74.81 | 24.24/25.73 | 17.83/19.50 |  Tesla T4 |
| ResNet50-IBN-a |  3x224x224 |   32  | 56.38/58.81  | 28.75/29.43 | 10.33/10.91 | 8.49/9.08 | RTX 2080Ti |

# Build 
 **Dependencies**
 * [CUDA](https://developer.nvidia.com/cuda-toolkit)
    * Recommended versions:
    * cuda-11.x + cuDNN-8.1
    * cuda-10.2 + cuDNN-8.1
 * TensorRT>=5.1.5.0
 * glog, gflags, protobuf

**Compile**
  
``` shell
# gflags
git clone https://github.com/gflags/gflags.git
cd gflags
cmake -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=OFF -DGFLAGS_NAMESPACE=google -DCMAKE_CXX_FLAGS=-fPIC ..
make -j

# glog
git clone https://github.com/google/glog.git
cd glog
sudo apt install libtool automake
./autogen.sh 
./configure CPPFLAGS="-I/usr/local/include -fPIC" LDFLAGS="-L/usr/local/lib" --disable-shared
make -j

# protobuf
git clone --recursive https://github.com/protocolbuffers/protobuf.git
cd protobuf
./autogen.sh 
./configure CXXFLAGS="-fPIC" --prefix=/usr/local --disable-shared
make -j

# caffe2trt
git clone https://github.com/chinasvt/caffe2trt.git
cd caffe2trt
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE ..
make
```

**Usage**

**数据**

测试数据链接: https://pan.baidu.com/s/1tvaGayhYprKroxCXrcsnyw 提取码: yspr 

数据下载完解压后，将数据目录链接到项目的data目录下，如下：
``` shell 
ln -s /your_download_path/ ./data/imagenet 
```
**编译模型**

***INT8量化需要的校准数据，由原始模型仓库[https://gitee.com/SMVD/IBN_PyTorch.git](https://gitee.com/SMVD/IBN_PyTorch.git) 中的 eval.py生成***

运行命令
``` shell
cd build
sh ../scripts/build_model.sh
```

参数说明
``` shell
./caffe2trt  \
--model=${model} \     #设置caffemodel文件路径
--deploy=${deploy} \    #设置prototxt文件路径
--engine=${engine} \    #设置输出引擎路径
--input=${input} \     #设置输入节点名字
--output=${output} \    #设置输出节点名字
--batch=${batch} \     #设置batch大小
--dataType=${dataType} \ #设置转换精度，支持FP32、FP16、INT8
--workspace=$4 \      #分配转换时的显存大小
--device=$3          #设置GPU Id 
```

**性能测试**

``` shell
# 测试模型时延和吞吐量
cd build
# 第一个参数-测试数据
# 第二个参数-模型路径
# 第三个参数-设置显卡id
# 第四个参数-设置推理Batchsize
# 第五个参数-设置迭代次数
./sample_inference ../data/person_0003.jpg ../engines/resnet50_ibn_a-d9d0bb7b_opt_b256_fp16.engine 0 64 1000

## 由于原始模型预处理是通过Pillow实现，预处理差异较大
## 测试模型正确性，无预处理，即预处理好的数据
# 第一个参数-测试数据
# 第二个参数-模型路径
# 第三个参数-设置显卡id
# 第四个参数-设置推理Batchsize，必须=32
./sample_ibn_raw ../data/imagenet ../engines/resnet50_ibn_a-d9d0bb7b_opt_b256_fp16.engine 0 32

## 测试模型正确性，有预处理，通过gpu实现的预处理方法
# 第一个参数-测试数据
# 第二个参数-模型路径
# 第三个参数-设置显卡id
# 第四个参数-设置推理Batchsize
./sample_ibn_raw ../data/imagenet ../engines/resnet50_ibn_a-d9d0bb7b_opt_b256_fp16.engine 0 32

```

