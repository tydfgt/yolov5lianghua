/**
 * YOLOv5 后处理 CUDA kernel
 * 包含: objectness 筛选 + 分类 + 边框解码 + NMS
 * 全部在 GPU 上执行
 */

#ifndef YOLOV5_POSTPROCESS_CUDA_H
#define YOLOV5_POSTPROCESS_CUDA_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

struct GpuDetection {
    float x, y, w, h;
    float conf;
    int classId;
};

/**
 * GPU 后处理: 接收 raw FP16 输出，返回检测框列表
 * 
 * @param output      GPU 上的 FP16 模型输出 [numAnchors * (5 + numClasses)]
 * @param numAnchors  anchor 数量 (25200)
 * @param numClasses  类别数 (80)
 * @param confThresh  置信度阈值
 * @param iouThresh   NMS IOU 阈值
 * @param maxDets     最大检测数
 * @param imgW, imgH  原始图片尺寸
 * @param inputW, inputH 模型输入尺寸
 * @param stream      CUDA stream
 * @return            CPU 上的检测结果
 */
std::vector<GpuDetection> yolov5PostprocessGPU(
    const __half *output,
    int numAnchors,
    int numClasses,
    float confThresh,
    float iouThresh,
    int maxDets,
    int imgW, int imgH,
    int inputW, int inputH,
    cudaStream_t stream);

#endif // YOLOV5_POSTPROCESS_CUDA_H
