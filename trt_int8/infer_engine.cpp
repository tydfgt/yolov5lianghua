/**
 * YOLOv5 TensorRT 推理程序
 * 
 * 功能: 加载序列化的 TensorRT 引擎，对图片/视频进行目标检测
 * 用法: ./infer_engine <engine_path> <input_image> [output_image]
 * 示例: ./infer_engine yolov5s_int8.engine ../data/images/bus.jpg result.jpg
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

// ==================== Logger ====================
class Logger : public ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

// ==================== 预处理 ====================
void preprocess(const cv::Mat &img, half *gpuInput, int w, int h)
{
    cv::Mat rgb, resized, floatImg;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(w, h));
    resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);
    std::vector<float> hostData(3 * h * w);
    for (int c = 0; c < 3; ++c)
    {
        std::memcpy(hostData.data() + c * h * w, channels[c].data, h * w * sizeof(float));
    }

    // 转换为 FP16
    std::vector<half> halfData(3 * h * w);
    for (int i = 0; i < 3 * h * w; ++i)
        halfData[i] = __float2half(hostData[i]);

    cudaMemcpy(gpuInput, halfData.data(), 3 * h * w * sizeof(half), cudaMemcpyHostToDevice);
}

// ==================== NMS ====================
struct Detection
{
    float x, y, w, h;
    float conf;
    int classId;
};

float iou(const Detection &a, const Detection &b)
{
    float ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
    float bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
    float interX1 = std::max(ax1, bx1);
    float interY1 = std::max(ay1, by1);
    float interX2 = std::min(ax2, bx2);
    float interY2 = std::min(ay2, by2);
    float interArea = std::max(0.0f, interX2 - interX1) * std::max(0.0f, interY2 - interY1);
    float areaA = a.w * a.h;
    float areaB = b.w * b.h;
    return interArea / (areaA + areaB - interArea + 1e-6f);
}

std::vector<Detection> nms(const std::vector<Detection> &dets, float iouThresh)
{
    std::vector<Detection> result;
    std::vector<int> indices(dets.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 按置信度降序排列
    std::sort(indices.begin(), indices.end(),
              [&dets](int a, int b) { return dets[a].conf > dets[b].conf; });

    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (suppressed[indices[i]]) continue;
        result.push_back(dets[indices[i]]);
        for (size_t j = i + 1; j < indices.size(); ++j)
        {
            if (suppressed[indices[j]]) continue;
            if (dets[indices[i]].classId != dets[indices[j]].classId) continue;
            if (iou(dets[indices[i]], dets[indices[j]]) > iouThresh)
                suppressed[indices[j]] = true;
        }
    }
    return result;
}

// ==================== 后处理 ====================
std::vector<Detection> postprocess(const half *output, int numAnchors, int numClasses,
                                    int imgW, int imgH, int inputW, int inputH,
                                    float confThresh = 0.25f, float iouThresh = 0.45f)
{
    // 先转换 FP16 → FP32
    int totalVals = numAnchors * (5 + numClasses);
    std::vector<float> floatOutput(totalVals);
    for (int i = 0; i < totalVals; ++i)
        floatOutput[i] = __half2float(output[i]);

    std::vector<Detection> dets;
    float scaleX = (float)imgW / inputW;
    float scaleY = (float)imgH / inputH;

    for (int i = 0; i < numAnchors; ++i)
    {
        const float *row = floatOutput.data() + i * (5 + numClasses);
        float objConf = row[4];
        if (objConf < confThresh) continue;

        int bestClass = 0;
        float bestClassConf = 0;
        for (int c = 0; c < numClasses; ++c)
        {
            float clsConf = row[5 + c];
            if (clsConf > bestClassConf)
            {
                bestClassConf = clsConf;
                bestClass = c;
            }
        }

        float score = objConf * bestClassConf;
        if (score < confThresh) continue;

        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        Detection det;
        det.x = (cx - w / 2) * scaleX;
        det.y = (cy - h / 2) * scaleY;
        det.w = w * scaleX;
        det.h = h * scaleY;
        det.conf = score;
        det.classId = bestClass;
        dets.push_back(det);
    }

    return nms(dets, iouThresh);
}

// ==================== COCO 类别名 ====================
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"};

// ==================== 绘制检测框 ====================
void drawDetections(cv::Mat &img, const std::vector<Detection> &dets)
{
    const std::vector<cv::Scalar> COLORS = {
        {0, 255, 0}, {255, 0, 0}, {0, 0, 255}, {255, 255, 0},
        {255, 0, 255}, {0, 255, 255}, {128, 255, 0}, {255, 128, 0}};

    for (const auto &det : dets)
    {
        cv::Scalar color = COLORS[det.classId % COLORS.size()];
        cv::Rect rect((int)det.x, (int)det.y, (int)det.w, (int)det.h);
        cv::rectangle(img, rect, color, 2);

        std::string label = COCO_CLASSES[det.classId] + " " + cv::format("%.2f", det.conf);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(img, cv::Point((int)det.x, (int)det.y - textSize.height - 4),
                      cv::Point((int)det.x + textSize.width, (int)det.y), color, -1);
        cv::putText(img, label, cv::Point((int)det.x, (int)det.y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

// ==================== 推理类 ====================
class YOLOv5Detector
{
public:
    YOLOv5Detector(const std::string &enginePath)
    {
        // 反序列化引擎
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good())
        {
            std::cerr << "❌ 无法读取引擎文件: " << enginePath << std::endl;
            return;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();

        mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
        mEngine = std::unique_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(engineData.data(), size));
        if (!mEngine)
        {
            std::cerr << "❌ 引擎反序列化失败" << std::endl;
            return;
        }
        mContext = std::unique_ptr<IExecutionContext>(mEngine->createExecutionContext());

        // 获取输入输出信息 (TRT 10.x: use getTensorShape)
        auto inputDims = mEngine->getTensorShape("images");
        mInputW = inputDims.d[3];
        mInputH = inputDims.d[2];

        auto outputDims = mEngine->getTensorShape("output0");
        mNumAnchors = outputDims.d[1];
        mNumClasses = outputDims.d[2] - 5;

        // 分配 GPU 内存 (FP16)
        cudaMalloc((void **)&mGpuInput, 3 * mInputH * mInputW * sizeof(half));
        cudaMalloc((void **)&mGpuOutput, mNumAnchors * (5 + mNumClasses) * sizeof(half));

        mReady = true;
        std::cout << "✅ 引擎加载成功 (FP16 I/O)" << std::endl;
        std::cout << "   输入: " << mInputW << "x" << mInputH << " (WxH)" << std::endl;
        std::cout << "   输出: " << mNumAnchors << " anchors × " << (5 + mNumClasses) << " values" << std::endl;
    }

    ~YOLOv5Detector()
    {
        if (mGpuInput) cudaFree(mGpuInput);
        if (mGpuOutput) cudaFree(mGpuOutput);
    }

    bool isReady() const { return mReady; }

    std::vector<Detection> detect(const cv::Mat &img, float confThresh = 0.25f, float iouThresh = 0.45f)
    {
        if (!mReady) return {};

        int imgW = img.cols, imgH = img.rows;

        // 预处理
        preprocess(img, mGpuInput, mInputW, mInputH);

        // 推理
        void *bindings[] = {mGpuInput, mGpuOutput};
        mContext->executeV2(bindings);

        // 后处理 (FP16 → FP32 转换在 postprocess 内)
        int totalVals = mNumAnchors * (5 + mNumClasses);
        std::vector<half> hostOutput(totalVals);
        cudaMemcpy(hostOutput.data(), mGpuOutput,
                   totalVals * sizeof(half), cudaMemcpyDeviceToHost);

        return postprocess(hostOutput.data(), mNumAnchors, mNumClasses,
                          imgW, imgH, mInputW, mInputH, confThresh, iouThresh);
    }

private:
    std::unique_ptr<IRuntime> mRuntime;
    std::unique_ptr<ICudaEngine> mEngine;
    std::unique_ptr<IExecutionContext> mContext;
    int mInputW = 640, mInputH = 640;
    int mNumAnchors = 25200, mNumClasses = 80;
    half *mGpuInput = nullptr, *mGpuOutput = nullptr;
    bool mReady = false;
};

// ==================== 主函数 ====================
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cout << "用法: " << argv[0] << " <engine> <input_image> [output_image] [--benchmark]" << std::endl;
        std::cout << "示例: " << argv[0] << " yolov5s_int8.engine ../data/images/bus.jpg result.jpg" << std::endl;
        std::cout << "压测: " << argv[0] << " yolov5s_int8.engine ../data/images/bus.jpg --benchmark" << std::endl;
        return 1;
    }

    std::string enginePath = argv[1];
    std::string inputPath = argv[2];
    std::string outputPath = (argc >= 4 && std::string(argv[3]) != "--benchmark") ? argv[3] : "result.jpg";
    bool benchmark = (argc >= 4 && std::string(argv[3]) == "--benchmark") ||
                     (argc >= 5 && std::string(argv[4]) == "--benchmark");

    initLibNvInferPlugins(&gLogger, "");

    // 加载引擎
    YOLOv5Detector detector(enginePath);
    if (!detector.isReady())
    {
        std::cerr << "❌ 检测器初始化失败" << std::endl;
        return 1;
    }

    // 读取图片
    cv::Mat img = cv::imread(inputPath);
    if (img.empty())
    {
        std::cerr << "❌ 无法读取图片: " << inputPath << std::endl;
        return 1;
    }
    std::cout << "📷 图片: " << inputPath << " (" << img.cols << "x" << img.rows << ")" << std::endl;

    if (benchmark)
    {
        // ========== 性能测试 ==========
        const int warmupRuns = 10;
        const int testRuns = 100;

        std::cout << "🔥 性能测试: 预热 " << warmupRuns << " 次..." << std::endl;
        for (int i = 0; i < warmupRuns; ++i)
            detector.detect(img);

        std::cout << "🔥 正式测试 " << testRuns << " 次..." << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < testRuns; ++i)
            detector.detect(img);
        auto t1 = std::chrono::high_resolution_clock::now();

        double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avgMs = totalMs / testRuns;
        double fps = 1000.0 / avgMs;

        std::cout << "========================================" << std::endl;
        std::cout << "  总耗时: " << totalMs << " ms" << std::endl;
        std::cout << "  平均耗时: " << avgMs << " ms/张" << std::endl;
        std::cout << "  FPS: " << fps << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // 检测并保存结果
    auto detections = detector.detect(img);
    std::cout << "🎯 检测到 " << detections.size() << " 个目标:" << std::endl;
    for (const auto &det : detections)
    {
        std::cout << "  [" << COCO_CLASSES[det.classId] << "] conf=" << det.conf
                  << " @ (" << (int)det.x << "," << (int)det.y
                  << " " << (int)det.w << "x" << (int)det.h << ")" << std::endl;
    }

    drawDetections(img, detections);
    cv::imwrite(outputPath, img);
    std::cout << "💾 结果已保存: " << outputPath << std::endl;

    return 0;
}
