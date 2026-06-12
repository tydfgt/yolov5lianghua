cmake_minimum_required(VERSION 3.18)
project(yolov5_trt_int8 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找 TensorRT
find_library(TENSORRT_LIB nvinfer
    PATHS /usr/lib/aarch64-linux-gnu
          /usr/local/cuda-12.6/targets/aarch64-linux/lib)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin
    PATHS /usr/lib/aarch64-linux-gnu
          /usr/local/cuda-12.6/targets/aarch64-linux/lib)
find_library(TENSORRT_ONNX_PARSER_LIB nvonnxparser
    PATHS /usr/lib/aarch64-linux-gnu
          /usr/local/cuda-12.6/targets/aarch64-linux/lib)
find_library(CUDART_LIB cudart
    PATHS /usr/local/cuda-12.6/targets/aarch64-linux/lib)
find_library(CUDA_LIB cuda
    PATHS /usr/local/cuda-12.6/targets/aarch64-linux/lib)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu
          /usr/local/cuda-12.6/include)

# CUDA include
set(CUDA_INCLUDE_DIR /usr/local/cuda-12.6/include)

# OpenCV (系统自编译 CUDA 版)
find_package(OpenCV REQUIRED)

# ==================== 引擎构建程序 ====================
add_executable(build_engine_int8 build_engine_int8.cpp)
target_include_directories(build_engine_int8 PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(build_engine_int8
    ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB} ${TENSORRT_ONNX_PARSER_LIB}
    ${CUDART_LIB} ${CUDA_LIB}
    ${OpenCV_LIBS})

# ==================== 推理程序 ====================
add_executable(infer_engine infer_engine.cpp)
target_include_directories(infer_engine PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(infer_engine
    ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB}
    ${CUDART_LIB} ${CUDA_LIB}
    ${OpenCV_LIBS})

# ==================== 动态形状引擎构建 ====================
add_executable(build_engine_dynamic build_engine_dynamic.cpp)
target_include_directories(build_engine_dynamic PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR})
target_link_libraries(build_engine_dynamic
    ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB} ${TENSORRT_ONNX_PARSER_LIB}
    ${CUDART_LIB} ${CUDA_LIB})

# ==================== 动态形状推理 ====================
add_executable(infer_dynamic infer_dynamic.cpp)
target_include_directories(infer_dynamic PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(infer_dynamic
    ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB}
    ${CUDART_LIB} ${CUDA_LIB}
    ${OpenCV_LIBS})

# ==================== 快速推理 (优化后处理) ====================
add_executable(infer_fast infer_fast.cpp)
target_include_directories(infer_fast PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(infer_fast
    ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB}
    ${CUDART_LIB} ${CUDA_LIB}
    ${OpenCV_LIBS})

# ==================== 全GPU推理 (CUDA后处理) ====================
add_executable(infer_gpu infer_gpu.cpp postprocess_cuda.cu)
target_include_directories(infer_gpu PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(infer_gpu
    ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB}
    ${CUDART_LIB} ${CUDA_LIB}
    ${OpenCV_LIBS})
