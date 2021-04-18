cmake_minimum_required(VERSION 2.8.7)

project(caffe2trt C CXX)

set(TRT_ROOT_DIR "/usr/local/TensorRT")
set(OPENCV_DIR "/usr/local")

# turn on C++11
if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -w -m64")
endif()

find_package(CUDA 5.5 QUIET)
message(STATUS "CUDA detected: " ${CUDA_VERSION})

set(CUDA_NVCC_FLAGS
        --disable-warnings;
        -O3;
        #-use_fast_math;
        #-gencode arch=compute_52,code=sm_52;
        #-gencode arch=compute_61,code=sm_61;
        #-gencode arch=compute_70,code=sm_70;
        -gencode arch=compute_75,code=sm_75;
        )

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

add_subdirectory(samples)
add_subdirectory(src)
add_subdirectory(tools)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TRT_ROOT_DIR}/include)
include_directories(${OPENCV_DIR}/include)
include_directories(include)
include_directories(include/trt/parsers/caffe/proto)
include_directories(third_party/onnx)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(${TRT_ROOT_DIR}/lib)
link_directories(${OPENCV_DIR}/lib)

list(APPEND LINKER_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_videoio)
list(APPEND LINKER_LIBS nvinfer)
list(APPEND LINKER_LIBS cudnn cublas curand nvrtc nvjpeg nppig nppc)
list(APPEND LINKER_LIBS ${PROTOBUF_LIBRARIES})
list(APPEND LINKER_LIBS "/usr/local/lib/libglog.a")
list(APPEND LINKER_LIBS "/usr/local/lib/libgflags.a")

foreach(TOOL_NAME ${TOOL_SOURCES})
    #unpack the dir "/"
    string(REPLACE "/" ";" SEXY_LIST ${TOOL_NAME})
    list(GET SEXY_LIST -1 TEST_CASE_NAME)
    #get the file name without suffix
    string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
    list(GET SEXY_LIST 0 TEST_CASE_NAME)
    CUDA_ADD_EXECUTABLE(${TEST_CASE_NAME} ${TOOL_NAME} ${TRT_SOURCES})
    target_link_libraries(${TEST_CASE_NAME} ${LINKER_LIBS})
    set_target_properties(${TEST_CASE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY .)
endforeach()

foreach(SAMPLE_NAME ${SAMPLE_SOURCES})
    #unpack the dir "/"
    string(REPLACE "/" ";" SEXY_LIST ${SAMPLE_NAME})
    list(GET SEXY_LIST -1 TEST_CASE_NAME)
    #get the file name without suffix
    string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
    list(GET SEXY_LIST 0 TEST_CASE_NAME)
    CUDA_ADD_EXECUTABLE(${TEST_CASE_NAME}  ${SAMPLE_NAME} ${TRT_SOURCES})
    target_link_libraries(${TEST_CASE_NAME} ${LINKER_LIBS})
    set_target_properties(${TEST_CASE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY .)
endforeach()