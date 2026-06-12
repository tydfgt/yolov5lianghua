/**
 * YOLOv5 全GPU推理 (预处理+推理+后处理全部GPU化)
 * 用法: ./infer_gpu <engine> <input> [output] [--benchmark]
 */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include "postprocess_cuda.cuh"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
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

// COCO 类别
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

void drawDetections(cv::Mat &img, const std::vector<GpuDetection> &dets) {
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

// CPU 预处理 → FP16 buffer (仍然在CPU做像素操作，然后用 cudaMemcpy)
void preprocess(const cv::Mat &img, half *gpuInput, int w, int h, cudaStream_t stream) {
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

    cudaMemcpyAsync(gpuInput, halfData.data(), halfData.size() * sizeof(half),
                    cudaMemcpyHostToDevice, stream);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "用法: " << argv[0] << " <engine> <input> [output] [--benchmark]\n";
        return 1;
    }
    std::string enginePath = argv[1], inputPath = argv[2];
    std::string outputPath = (argc > 3 && std::string(argv[3]) != "--benchmark") ? argv[3] : "result_gpu.jpg";
    bool benchmark = (argc > 3 && std::string(argv[3]) == "--benchmark") ||
                     (argc > 4 && std::string(argv[4]) == "--benchmark");

    initLibNvInferPlugins(&gLogger, "");

    // 加载引擎
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) { std::cerr << "❌ 无法读取引擎\n"; return 1; }
    file.seekg(0, std::ios::end); size_t size = file.tellg(); file.seekg(0, std::ios::beg);
    std::vector<char> data(size); file.read(data.data(), size); file.close();

    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(data.data(), size));
    if (!engine) { std::cerr << "❌ 引擎反序列化失败\n"; return 1; }
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());

    auto inputDims = engine->getTensorShape("images");
    int inputW = inputDims.d[3], inputH = inputDims.d[2];
    auto outputDims = engine->getTensorShape("output0");
    int numAnchors = outputDims.d[1], numClasses = outputDims.d[2] - 5;

    // 分配 GPU 内存
    half *gpuInput = nullptr, *gpuOutput = nullptr;
    cudaMalloc((void **)&gpuInput, 3 * inputH * inputW * sizeof(half));
    cudaMalloc((void **)&gpuOutput, numAnchors * (5 + numClasses) * sizeof(half));

    cudaStream_t stream; cudaStreamCreate(&stream);
    context->setTensorAddress("images", gpuInput);
    context->setTensorAddress("output0", gpuOutput);

    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) { std::cerr << "❌ 无法读取图片\n"; return 1; }
    std::cout << "📷 " << inputPath << " (" << img.cols << "x" << img.rows << ")\n";
    std::cout << "🚀 GPU 后处理启用\n";

    // 全流程推理函数
    auto inferOnce = [&]() {
        preprocess(img, gpuInput, inputW, inputH, stream);
        context->enqueueV3(stream);
        // 后处理直接在 GPU 上执行
        return yolov5PostprocessGPU(
            gpuOutput, numAnchors, numClasses,
            0.25f, 0.45f, 100,
            img.cols, img.rows, inputW, inputH, stream);
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
    cudaStreamDestroy(stream);
    return 0;
}
