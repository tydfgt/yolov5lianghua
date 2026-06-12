/**
 * YOLOv5 TensorRT 快速推理 (优化后处理 + pinned memory + CUDA stream)
 * 用法: ./infer_fast <engine> <input> <output> [--benchmark]
 */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace nvinfer1;
using half = __half;

class Logger : public ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

struct Detection { float x, y, w, h; float conf; int classId; };

inline float fastIou(const Detection &a, const Detection &b) {
    float ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
    float bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
    float inter = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1)) *
                  std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-6f);
}

std::vector<Detection> postprocessFast(const half *output, int numAnchors, int numClasses,
                                        int imgW, int imgH, int inputW, int inputH,
                                        float confThresh = 0.25f, float iouThresh = 0.45f) {
    struct Candidate { int idx; float obj; };
    std::vector<Candidate> candidates;
    candidates.reserve(256);
    for (int i = 0; i < numAnchors; ++i) {
        float obj = __half2float(output[i * (5 + numClasses) + 4]);
        if (obj >= confThresh) candidates.push_back({i, obj});
    }
    std::vector<Detection> dets;
    dets.reserve(candidates.size());
    float scaleX = (float)imgW / inputW, scaleY = (float)imgH / inputH;
    for (const auto &cand : candidates) {
        const half *row = output + cand.idx * (5 + numClasses);
        half bestClsConf{0}; int bestClass = 0;
        for (int c = 0; c < numClasses; ++c) {
            if (row[5 + c] > bestClsConf) { bestClsConf = row[5 + c]; bestClass = c; }
        }
        float score = cand.obj * __half2float(bestClsConf);
        if (score < confThresh) continue;
        float cx = __half2float(row[0]), cy = __half2float(row[1]);
        float w = __half2float(row[2]), h = __half2float(row[3]);
        Detection det;
        det.x = (cx - w / 2) * scaleX; det.y = (cy - h / 2) * scaleY;
        det.w = w * scaleX; det.h = h * scaleY;
        det.conf = score; det.classId = bestClass;
        dets.push_back(det);
    }
    if (dets.empty()) return dets;
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return dets[a].conf > dets[b].conf; });
    std::vector<bool> keep(dets.size(), true);
    std::vector<Detection> result; result.reserve(dets.size());
    for (size_t i = 0; i < order.size(); ++i) {
        int idxI = order[i];
        if (!keep[idxI]) continue;
        result.push_back(dets[idxI]);
        for (size_t j = i + 1; j < order.size(); ++j) {
            int idxJ = order[j];
            if (!keep[idxJ]) continue;
            if (dets[idxI].classId != dets[idxJ].classId) continue;
            if (fastIou(dets[idxI], dets[idxJ]) > iouThresh) keep[idxJ] = false;
        }
    }
    return result;
}

void preprocessFast(const cv::Mat &img, half *gpuInput, int w, int h) {
    cv::Mat rgb, resized, floatImg;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(w, h));
    resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> chs(3); cv::split(floatImg, chs);
    std::vector<float> hostData(3 * h * w);
    for (int c = 0; c < 3; ++c)
        memcpy(hostData.data() + c * h * w, chs[c].data, h * w * sizeof(float));
    std::vector<half> halfData(3 * h * w);
    for (size_t i = 0; i < halfData.size(); ++i)
        halfData[i] = __float2half(hostData[i]);
    cudaMemcpy(gpuInput, halfData.data(), halfData.size() * sizeof(half), cudaMemcpyHostToDevice);
}

const std::vector<std::string> COCO_CLASSES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

void drawDetections(cv::Mat &img, const std::vector<Detection> &dets) {
    const std::vector<cv::Scalar> COLORS = {{0,255,0},{255,0,0},{0,0,255},{255,255,0}};
    for (const auto &det : dets) {
        cv::Scalar color = COLORS[det.classId % COLORS.size()];
        cv::rectangle(img, {(int)det.x, (int)det.y, (int)det.w, (int)det.h}, color, 2);
        std::string label = COCO_CLASSES[det.classId] + " " + cv::format("%.2f", det.conf);
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
        cv::rectangle(img, {(int)det.x, (int)det.y - ts.height - 4, ts.width, ts.height}, color, -1);
        cv::putText(img, label, {(int)det.x, (int)det.y - 4}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "用法: " << argv[0] << " <engine> <input> [output] [--benchmark]\n";
        return 1;
    }
    std::string enginePath = argv[1], inputPath = argv[2];
    std::string outputPath = (argc > 3 && std::string(argv[3]) != "--benchmark") ? argv[3] : "result_fast.jpg";
    bool benchmark = (argc > 3 && std::string(argv[3]) == "--benchmark") ||
                     (argc > 4 && std::string(argv[4]) == "--benchmark");
    initLibNvInferPlugins(&gLogger, "");

    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) { std::cerr << "❌ unable to read engine\n"; return 1; }
    file.seekg(0, std::ios::end); size_t size = file.tellg(); file.seekg(0, std::ios::beg);
    std::vector<char> data(size); file.read(data.data(), size); file.close();

    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(data.data(), size));
    if (!engine) { std::cerr << "❌ engine deserialize failed\n"; return 1; }
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());

    auto inputDims = engine->getTensorShape("images");
    int inputW = inputDims.d[3], inputH = inputDims.d[2];
    auto outputDims = engine->getTensorShape("output0");
    int numAnchors = outputDims.d[1], numClasses = outputDims.d[2] - 5;

    half *gpuInput = nullptr, *gpuOutput = nullptr;
    cudaMalloc((void **)&gpuInput, 3 * inputH * inputW * sizeof(half));
    cudaMalloc((void **)&gpuOutput, numAnchors * (5 + numClasses) * sizeof(half));
    cudaStream_t stream; cudaStreamCreate(&stream);
    context->setTensorAddress("images", gpuInput);
    context->setTensorAddress("output0", gpuOutput);

    half *hostOutput = nullptr;
    cudaMallocHost((void **)&hostOutput, numAnchors * (5 + numClasses) * sizeof(half));

    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) { std::cerr << "❌ cannot read image\n"; return 1; }
    std::cout << "📷 " << inputPath << " (" << img.cols << "x" << img.rows << ")\n";

    auto inferOnce = [&]() {
        preprocessFast(img, gpuInput, inputW, inputH);
        context->enqueueV3(stream);
        cudaMemcpyAsync(hostOutput, gpuOutput, numAnchors * (5 + numClasses) * sizeof(half),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return postprocessFast(hostOutput, numAnchors, numClasses, img.cols, img.rows, inputW, inputH);
    };

    if (benchmark) {
        const int warmup = 10, runs = 100;
        std::cout << "🔥 warmup " << warmup << "...\n";
        for (int i = 0; i < warmup; ++i) inferOnce();
        std::cout << "🔥 benchmark " << runs << "...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i) inferOnce();
        auto t1 = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "========================================\n";
        std::cout << "  Total: " << totalMs << " ms\n";
        std::cout << "  Avg:   " << totalMs / runs << " ms/frame\n";
        std::cout << "  FPS:   " << 1000.0 * runs / totalMs << "\n";
        std::cout << "========================================\n";
    }

    auto dets = inferOnce();
    std::cout << "🎯 detected " << dets.size() << " objects:\n";
    for (const auto &det : dets)
        std::cout << "  [" << COCO_CLASSES[det.classId] << "] conf=" << det.conf
                  << " @(" << (int)det.x << "," << (int)det.y << ")\n";

    drawDetections(img, dets);
    cv::imwrite(outputPath, img);
    std::cout << "💾 " << outputPath << "\n";

    cudaFree(gpuInput); cudaFree(gpuOutput);
    cudaFreeHost(hostOutput);
    cudaStreamDestroy(stream);
    return 0;
}
