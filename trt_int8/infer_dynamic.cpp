/**
 * YOLOv5 TensorRT 动态形状推理程序
 * 
 * 用法: ./infer_dynamic <engine> <image> <width> [height] [--benchmark]
 * 示例: ./infer_dynamic yolov5n_fp16_dynamic.engine ../../data/images/bus.jpg 320 --benchmark
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

float iou(const Detection &a, const Detection &b) {
    float ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
    float bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
    float inter = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1)) *
                  std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-6f);
}

std::vector<Detection> nms(const std::vector<Detection> &dets, float iouThresh) {
    std::vector<Detection> result;
    std::vector<int> indices(dets.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) { return dets[a].conf > dets[b].conf; });
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < indices.size(); ++i) {
        if (suppressed[indices[i]]) continue;
        result.push_back(dets[indices[i]]);
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[indices[j]]) continue;
            if (dets[indices[i]].classId == dets[indices[j]].classId &&
                iou(dets[indices[i]], dets[indices[j]]) > iouThresh)
                suppressed[indices[j]] = true;
        }
    }
    return result;
}

std::vector<Detection> postprocess(const half *output, int numAnchors, int numClasses,
                                    int imgW, int imgH, int inputW, int inputH,
                                    float confThresh = 0.25f, float iouThresh = 0.45f) {
    int totalVals = numAnchors * (5 + numClasses);
    std::vector<float> fout(totalVals);
    for (int i = 0; i < totalVals; ++i) fout[i] = __half2float(output[i]);

    std::vector<Detection> dets;
    float scaleX = (float)imgW / inputW, scaleY = (float)imgH / inputH;
    for (int i = 0; i < numAnchors; ++i) {
        const float *row = fout.data() + i * (5 + numClasses);
        if (row[4] < confThresh) continue;
        int bestClass = 0; float bestClassConf = 0;
        for (int c = 0; c < numClasses; ++c) {
            if (row[5 + c] > bestClassConf) { bestClassConf = row[5 + c]; bestClass = c; }
        }
        float score = row[4] * bestClassConf;
        if (score < confThresh) continue;
        Detection det;
        det.x = (row[0] - row[2] / 2) * scaleX; det.y = (row[1] - row[3] / 2) * scaleY;
        det.w = row[2] * scaleX; det.h = row[3] * scaleY;
        det.conf = score; det.classId = bestClass;
        dets.push_back(det);
    }
    return nms(dets, iouThresh);
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

class YOLOv5DetectorDynamic {
public:
    YOLOv5DetectorDynamic(const std::string &enginePath) {
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) { std::cerr << "❌ 无法读取: " << enginePath << std::endl; return; }
        file.seekg(0, std::ios::end); size_t size = file.tellg(); file.seekg(0, std::ios::beg);
        std::vector<char> data(size); file.read(data.data(), size); file.close();

        mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
        mEngine = std::unique_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(data.data(), size));
        if (!mEngine) { std::cerr << "❌ 反序列化失败" << std::endl; return; }
        mContext = std::unique_ptr<IExecutionContext>(mEngine->createExecutionContext());

        mInputName = mEngine->getIOTensorName(0);
        mOutputName = mEngine->getIOTensorName(1);

        // 最大 I/O 大小 (用于分配)
        auto maxDims = mEngine->getTensorShape(mInputName.c_str());
        mMaxInputW = maxDims.d[3]; mMaxInputH = maxDims.d[2];
        auto maxOutDims = mEngine->getTensorShape(mOutputName.c_str());
        mMaxNumAnchors = maxOutDims.d[1]; mNumClasses = maxOutDims.d[2] - 5;

        cudaMalloc((void **)&mGpuInput, 3 * mMaxInputH * mMaxInputW * sizeof(half));
        cudaMalloc((void **)&mGpuOutput, mMaxNumAnchors * (5 + mNumClasses) * sizeof(half));

        mReady = true;
        std::cout << "✅ 动态引擎加载成功 (MAX: " << mMaxInputW << "x" << mMaxInputH << ")" << std::endl;
    }

    ~YOLOv5DetectorDynamic() { if(mGpuInput) cudaFree(mGpuInput); if(mGpuOutput) cudaFree(mGpuOutput); }

    bool isReady() const { return mReady; }

    std::vector<Detection> detect(const cv::Mat &img, int inputW, int inputH,
                                   float confThresh = 0.25f, float iouThresh = 0.45f) {
        if (!mReady) return {};

        // 设置输入形状
        Dims4 inputShape{1, 3, inputH, inputW};
        mContext->setInputShape(mInputName.c_str(), inputShape);

        // 预处理: BGR→RGB, resize, normalize, FP16
        cv::Mat rgb, resized, floatImg;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(inputW, inputH));
        resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);
        std::vector<cv::Mat> chs(3); cv::split(floatImg, chs);
        std::vector<float> hostData(3 * inputH * inputW);
        for (int c = 0; c < 3; ++c)
            memcpy(hostData.data() + c * inputH * inputW, chs[c].data, inputH * inputW * sizeof(float));
        std::vector<half> halfData(3 * inputH * inputW);
        for (size_t i = 0; i < halfData.size(); ++i) halfData[i] = __float2half(hostData[i]);
        cudaMemcpy(mGpuInput, halfData.data(), halfData.size() * sizeof(half), cudaMemcpyHostToDevice);

        // 设置张量地址
        mContext->setTensorAddress(mInputName.c_str(), mGpuInput);
        mContext->setTensorAddress(mOutputName.c_str(), mGpuOutput);

        // 推理
        mContext->enqueueV3(0);

        // 获取输出形状
        auto outDims = mContext->getTensorShape(mOutputName.c_str());
        int numAnchors = outDims.d[1];
        int totalVals = numAnchors * (5 + mNumClasses);
        std::vector<half> hostOutput(totalVals);
        cudaMemcpy(hostOutput.data(), mGpuOutput, totalVals * sizeof(half), cudaMemcpyDeviceToHost);

        return postprocess(hostOutput.data(), numAnchors, mNumClasses,
                          img.cols, img.rows, inputW, inputH, confThresh, iouThresh);
    }

private:
    std::unique_ptr<IRuntime> mRuntime;
    std::unique_ptr<ICudaEngine> mEngine;
    std::unique_ptr<IExecutionContext> mContext;
    std::string mInputName, mOutputName;
    int mMaxInputW = 640, mMaxInputH = 640, mMaxNumAnchors = 25200, mNumClasses = 80;
    half *mGpuInput = nullptr, *mGpuOutput = nullptr;
    bool mReady = false;
};

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cout << "用法: " << argv[0] << " <engine> <image> <width> [height] [--benchmark]" << std::endl;
        std::cout << "示例: " << argv[0] << " yolov5n_fp16_dynamic.engine bus.jpg 320 --benchmark" << std::endl;
        return 1;
    }

    std::string enginePath = argv[1], inputPath = argv[2];
    int inputW = std::stoi(argv[3]);
    int inputH = (argc >= 5 && argv[4][0] != '-') ? std::stoi(argv[4]) : inputW;
    bool benchmark = (argc >= 5 && std::string(argv[4]) == "--benchmark") ||
                     (argc >= 6 && std::string(argv[5]) == "--benchmark");

    initLibNvInferPlugins(&gLogger, "");

    YOLOv5DetectorDynamic detector(enginePath);
    if (!detector.isReady()) { std::cerr << "❌ 初始化失败" << std::endl; return 1; }

    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) { std::cerr << "❌ 无法读取图片" << std::endl; return 1; }
    std::cout << "📷 " << inputPath << " (" << img.cols << "x" << img.rows << ")"
              << " → 输入 " << inputW << "x" << inputH << std::endl;

    if (benchmark) {
        const int warmup = 10, runs = 100;
        std::cout << "🔥 预热 " << warmup << " 次..." << std::endl;
        for (int i = 0; i < warmup; ++i) detector.detect(img, inputW, inputH);
        std::cout << "🔥 测试 " << runs << " 次..." << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < runs; ++i) detector.detect(img, inputW, inputH);
        auto t1 = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "========================================" << std::endl;
        std::cout << "  总耗时: " << totalMs << " ms" << std::endl;
        std::cout << "  平均: " << totalMs / runs << " ms/张" << std::endl;
        std::cout << "  FPS: " << 1000.0 * runs / totalMs << std::endl;
        std::cout << "========================================" << std::endl;
    }

    auto detections = detector.detect(img, inputW, inputH);
    std::cout << "🎯 检测到 " << detections.size() << " 个目标:" << std::endl;
    for (const auto &det : detections)
        std::cout << "  [" << COCO_CLASSES[det.classId] << "] conf=" << det.conf
                  << " @ (" << (int)det.x << "," << (int)det.y << ")" << std::endl;

    drawDetections(img, detections);
    cv::imwrite("result_dynamic.jpg", img);
    std::cout << "💾 result_dynamic.jpg" << std::endl;
    return 0;
}
