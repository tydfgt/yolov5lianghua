/**
 * YOLOv5 后处理 CUDA kernel 实现 (v2)
 * GPU解码+筛选 → CPU排序+NMS (候选<200时CPU更高效)
 */

#include "postprocess_cuda.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_CANDIDATES = 1024;

struct CandidateDevice { float x,y,w,h; float score; int classId; };

__global__ void decodeAndFilterKernel(
    const __half *output, CandidateDevice *candidates, int *candidateCount,
    int numAnchors, int numClasses, float confThresh,
    int imgW, int imgH, int inputW, int inputH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAnchors) return;
    int stride = 5 + numClasses;
    const __half *row = output + idx * stride;
    float objConf = __half2float(row[4]);
    if (objConf < confThresh) return;
    half bestClsConf{0}; int bestCls = 0;
    for (int c = 0; c < numClasses; ++c) {
        half clsConf = row[5 + c];
        if (clsConf > bestClsConf) { bestClsConf = clsConf; bestCls = c; }
    }
    float score = objConf * __half2float(bestClsConf);
    if (score < confThresh) return;
    float cx = __half2float(row[0]), cy = __half2float(row[1]);
    float w = __half2float(row[2]), h = __half2float(row[3]);
    float sx = (float)imgW / inputW, sy = (float)imgH / inputH;
    int slot = atomicAdd(candidateCount, 1);
    if (slot < MAX_CANDIDATES) {
        candidates[slot].x = (cx - w/2) * sx;
        candidates[slot].y = (cy - h/2) * sy;
        candidates[slot].w = w * sx;
        candidates[slot].h = h * sy;
        candidates[slot].score = score;
        candidates[slot].classId = bestCls;
    }
}

std::vector<GpuDetection> yolov5PostprocessGPU(
    const __half *output, int numAnchors, int numClasses,
    float confThresh, float iouThresh, int maxDets,
    int imgW, int imgH, int inputW, int inputH, cudaStream_t stream)
{
    CandidateDevice *dCands = nullptr; int *dCount = nullptr;
    cudaMalloc(&dCands, MAX_CANDIDATES * sizeof(CandidateDevice));
    cudaMalloc(&dCount, sizeof(int));
    cudaMemsetAsync(dCount, 0, sizeof(int), stream);

    int grid = (numAnchors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    decodeAndFilterKernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        output, dCands, dCount, numAnchors, numClasses, confThresh,
        imgW, imgH, inputW, inputH);

    int hCount = 0;
    cudaMemcpyAsync(&hCount, dCount, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (hCount == 0) { cudaFree(dCands); cudaFree(dCount); return {}; }
    hCount = std::min(hCount, MAX_CANDIDATES);

    std::vector<CandidateDevice> hCands(hCount);
    cudaMemcpy(hCands.data(), dCands, hCount * sizeof(CandidateDevice), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    cudaFree(dCands); cudaFree(dCount);

    std::sort(hCands.begin(), hCands.end(),
              [](const CandidateDevice &a, const CandidateDevice &b) { return a.score > b.score; });

    std::vector<bool> keep(hCount, true);
    std::vector<GpuDetection> results;
    results.reserve(std::min(hCount, maxDets));

    for (int i = 0; i < hCount && (int)results.size() < maxDets; ++i) {
        if (!keep[i]) continue;
        GpuDetection det; det.x=hCands[i].x; det.y=hCands[i].y;
        det.w=hCands[i].w; det.h=hCands[i].h;
        det.conf=hCands[i].score; det.classId=hCands[i].classId;
        results.push_back(det);

        for (int j = i+1; j < hCount; ++j) {
            if (!keep[j]) continue;
            if (hCands[i].classId != hCands[j].classId) continue;
            float ax1=hCands[i].x, ay1=hCands[i].y, ax2=ax1+hCands[i].w, ay2=ay1+hCands[i].h;
            float bx1=hCands[j].x, by1=hCands[j].y, bx2=bx1+hCands[j].w, by2=by1+hCands[j].h;
            float inter = std::max(0.f, std::min(ax2,bx2)-std::max(ax1,bx1)) *
                          std::max(0.f, std::min(ay2,by2)-std::max(ay1,by1));
            float iou = inter / (hCands[i].w*hCands[i].h + hCands[j].w*hCands[j].h - inter + 1e-6f);
            if (iou > iouThresh) keep[j] = false;
        }
    }
    return results;
}
