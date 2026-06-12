/**
 * YOLOv5 TensorRT INT8 引擎构建程序
 * 
 * 功能: 加载 ONNX 模型，使用 COCO128 图像进行 INT8 校准，构建 TensorRT 引擎
 * 用法: ./build_engine_int8 <onnx_path> <calib_dir> <output_engine>
 * 示例: ./build_engine_int8 ../yolov5s.onnx ../datasets/coco128/images/train2017 yolov5s_int8.engine
 */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <cuda_fp16.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace nvinfer1;
using half = __half;

// ==================== TensorRT Logger ====================
class Logger : public ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

// ==================== INT8 Entropy 校准器 ====================
class Int8EntropyCalibrator : public IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator(const std::vector<std::string> &imgPaths,
                          const std::string &cacheFile,
                          int batchSize,
                          int inputW, int inputH)
        : mImgPaths(imgPaths),
          mCacheFile(cacheFile),
          mBatchSize(batchSize),
          mInputW(inputW),
          mInputH(inputH),
          mInputCount(batchSize * 3 * inputH * inputW),
          mCurBatch(0)
    {
        mDeviceInput = nullptr;
        cudaMalloc((void **)&mDeviceInput, mInputCount * sizeof(half));
    }

    ~Int8EntropyCalibrator()
    {
        if (mDeviceInput)
            cudaFree(mDeviceInput);
    }

    int getBatchSize() const noexcept override { return mBatchSize; }

    // 校准数据使用 FP16（匹配 ONNX 模型输入精度）
    bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override
    {
        if (mCurBatch + mBatchSize > (int)mImgPaths.size())
            return false;

        std::vector<float> batchData(mInputCount);
        for (int i = 0; i < mBatchSize && (mCurBatch + i) < (int)mImgPaths.size(); ++i)
        {
            cv::Mat img = cv::imread(mImgPaths[mCurBatch + i]);
            if (img.empty())
            {
                int offset = i * 3 * mInputH * mInputW;
                std::fill(batchData.begin() + offset, batchData.begin() + offset + 3 * mInputH * mInputW, 0.0f);
                continue;
            }

            cv::Mat rgb, resized, floatImg;
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
            cv::resize(rgb, resized, cv::Size(mInputW, mInputH));
            resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);

            std::vector<cv::Mat> channels(3);
            cv::split(floatImg, channels);
            int offset = i * 3 * mInputH * mInputW;
            for (int c = 0; c < 3; ++c)
            {
                std::memcpy(batchData.data() + offset + c * mInputH * mInputW,
                            channels[c].data, mInputH * mInputW * sizeof(float));
            }
        }

        // 转换为 FP16
        std::vector<half> halfData(mInputCount);
        for (int i = 0; i < mInputCount; ++i)
            halfData[i] = __float2half(batchData[i]);

        cudaMemcpy(mDeviceInput, halfData.data(), mInputCount * sizeof(half), cudaMemcpyHostToDevice);
        bindings[0] = mDeviceInput;
        mCurBatch += mBatchSize;
        return true;
    }

    const void *readCalibrationCache(size_t &length) noexcept override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCacheFile, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
        {
            std::copy(std::istreambuf_iterator<char>(input),
                      std::istreambuf_iterator<char>(),
                      std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? mCalibrationCache.data() : nullptr;
    }

    void writeCalibrationCache(const void *cache, size_t length) noexcept override
    {
        std::ofstream output(mCacheFile, std::ios::binary);
        output.write(reinterpret_cast<const char *>(cache), length);
        std::cout << "[CALIB] 校准缓存已保存: " << mCacheFile << " (" << length << " bytes)" << std::endl;
    }

private:
    std::vector<std::string> mImgPaths;
    std::string mCacheFile;
    int mBatchSize;
    int mInputW, mInputH;
    int mInputCount;
    int mCurBatch;
    float *mDeviceInput;
    std::vector<char> mCalibrationCache;
};

// ==================== 获取校准图片列表 ====================
std::vector<std::string> getCalibrationImages(const std::string &dirPath, int maxImages = 200)
{
    std::vector<std::string> imgPaths;
    std::vector<cv::String> filesJpg, filesJpeg, filesPng, filesBmp;
    std::cout << "[CALIB] 扫描目录: " << dirPath << std::endl;
    cv::glob(dirPath + "/*.jpg", filesJpg, false);
    cv::glob(dirPath + "/*.jpeg", filesJpeg, false);
    cv::glob(dirPath + "/*.png", filesPng, false);
    cv::glob(dirPath + "/*.bmp", filesBmp, false);
    std::cout << "[CALIB] jpg:" << filesJpg.size() << " jpeg:" << filesJpeg.size()
              << " png:" << filesPng.size() << " bmp:" << filesBmp.size() << std::endl;

    auto addFiles = [&](const std::vector<cv::String> &fs) {
        for (const auto &f : fs)
        {
            if ((int)imgPaths.size() >= maxImages) return;
            imgPaths.push_back(std::string(f.c_str()));
        }
    };
    addFiles(filesJpg);
    addFiles(filesJpeg);
    addFiles(filesPng);
    addFiles(filesBmp);
    std::cout << "[CALIB] 找到 " << imgPaths.size() << " 张校准图片" << std::endl;
    return imgPaths;
}

// ==================== 构建 INT8 引擎 ====================
bool buildInt8Engine(const std::string &onnxPath,
                     const std::string &calibDir,
                     const std::string &enginePath,
                     int maxCalibImages = 200,
                     int calibBatchSize = 8)
{
    // 1. 初始化 TensorRT
    initLibNvInferPlugins(&gLogger, "");

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
    if (!builder)
    {
        std::cerr << "❌ 创建 IBuilder 失败" << std::endl;
        return false;
    }

    // 2. 创建网络 (TRT 10.x: 传 0 即可，kEXPLICIT_BATCH 已废弃)
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        std::cerr << "❌ 创建网络失败" << std::endl;
        return false;
    }

    // 3. ONNX 解析器
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING)))
    {
        std::cerr << "❌ ONNX 解析失败: " << onnxPath << std::endl;
        return false;
    }
    std::cout << "✅ ONNX 模型加载成功: " << onnxPath << std::endl;

    // 4. 构建配置
    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 512 * (1 << 20)); // 512MB

    // 启用 INT8 模式
    if (!builder->platformHasFastInt8())
    {
        std::cerr << "❌ 平台不支持 INT8 推理" << std::endl;
        return false;
    }
    config->setFlag(BuilderFlag::kINT8);
    std::cout << "✅ INT8 模式已启用" << std::endl;

    // 启用 FP16 作为回退（部分层不支持 INT8 时）
    if (builder->platformHasFastFp16())
    {
        config->setFlag(BuilderFlag::kFP16);
        std::cout << "✅ FP16 回退已启用" << std::endl;
    }

    // 5. 校准配置
    auto calibImages = getCalibrationImages(calibDir, maxCalibImages);
    if (calibImages.empty())
    {
        std::cerr << "❌ 校准目录没有图片: " << calibDir << std::endl;
        return false;
    }

    std::string cacheFile = enginePath + ".calib_cache";
    Int8EntropyCalibrator calibrator(calibImages, cacheFile, calibBatchSize, 640, 640);
    config->setInt8Calibrator(&calibrator);
    std::cout << "✅ INT8 校准器已配置 (batch=" << calibBatchSize << ")" << std::endl;

    // 6. 构建引擎
    std::cout << "🔨 开始构建 TensorRT INT8 引擎..." << std::endl;
    std::cout << "   这可能需要几分钟，请耐心等待..." << std::endl;

    auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        std::cerr << "❌ 引擎构建失败" << std::endl;
        return false;
    }

    // 7. 保存引擎
    std::ofstream outFile(enginePath, std::ios::binary);
    outFile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    outFile.close();

    std::cout << "✅ INT8 引擎已保存: " << enginePath
              << " (" << plan->size() / (1 << 20) << " MB)" << std::endl;
    return true;
}

// ==================== 构建 FP16 引擎 (对比用) ====================
bool buildFp16Engine(const std::string &onnxPath, const std::string &enginePath)
{
    initLibNvInferPlugins(&gLogger, "");

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0));

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING)))
    {
        std::cerr << "❌ ONNX 解析失败" << std::endl;
        return false;
    }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 512 * (1 << 20));

    if (builder->platformHasFastFp16())
        config->setFlag(BuilderFlag::kFP16);

    std::cout << "🔨 构建 FP16 引擎 (对比)..." << std::endl;
    auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        std::cerr << "❌ FP16 引擎构建失败" << std::endl;
        return false;
    }

    std::ofstream outFile(enginePath, std::ios::binary);
    outFile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    std::cout << "✅ FP16 引擎已保存: " << enginePath
              << " (" << plan->size() / (1 << 20) << " MB)" << std::endl;
    return true;
}

// ==================== 主函数 ====================
int main(int argc, char **argv)
{
    std::string onnxPath = "../yolov5s.onnx";
    std::string calibDir = "../datasets/coco128/images/train2017";
    std::string enginePath = "yolov5s_int8.engine";
    bool buildFp16Also = true;

    if (argc >= 2) onnxPath = argv[1];
    if (argc >= 3) calibDir = argv[2];
    if (argc >= 4) enginePath = argv[3];

    std::cout << "========================================" << std::endl;
    std::cout << "  YOLOv5 TensorRT INT8 引擎构建" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "ONNX 模型:   " << onnxPath << std::endl;
    std::cout << "校准图片目录: " << calibDir << std::endl;
    std::cout << "输出引擎:   " << enginePath << std::endl;
    std::cout << "========================================" << std::endl;

    // 构建 INT8 引擎
    bool success = buildInt8Engine(onnxPath, calibDir, enginePath);

    // 同时构建 FP16 引擎用于对比
    if (success && buildFp16Also)
    {
        std::string fp16Path = enginePath;
        size_t pos = fp16Path.find("_int8");
        if (pos != std::string::npos)
            fp16Path.replace(pos, 5, "_fp16");
        else
            fp16Path = enginePath + "_fp16";

        buildFp16Engine(onnxPath, fp16Path);
    }

    return success ? 0 : 1;
}
