/**
 * YOLOv5 TensorRT 动态形状 FP16 引擎构建程序 (TensorRT 10.x)
 * 
 * 功能: 加载动态 ONNX 模型，构建 FP16 引擎，支持多分辨率
 * 用法: ./build_engine_dynamic <onnx_path> <output_engine>
 * 示例: ./build_engine_dynamic ../../yolov5n_dynamic.onnx yolov5n_fp16_dynamic.engine
 */

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <cuda_fp16.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <string>

using namespace nvinfer1;

class Logger : public ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

bool buildDynamicEngine(const std::string &onnxPath, const std::string &enginePath,
                         int optH = 320, int optW = 320,
                         int minH = 256, int minW = 256,
                         int maxH = 640, int maxW = 640)
{
    initLibNvInferPlugins(&gLogger, "");

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
    if (!builder) { std::cerr << "❌ IBuilder 创建失败" << std::endl; return false; }

    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) { std::cerr << "❌ 网络创建失败" << std::endl; return false; }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING)))
    {
        std::cerr << "❌ ONNX 解析失败" << std::endl;
        return false;
    }
    std::cout << "✅ ONNX 模型加载成功: " << onnxPath << std::endl;

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 512 * (1 << 20));

    // FP16 模式（Orin 最快）
    if (builder->platformHasFastFp16())
    {
        config->setFlag(BuilderFlag::kFP16);
        std::cout << "✅ FP16 已启用" << std::endl;
    }

    // 动态形状优化配置
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("images", OptProfileSelector::kMIN, Dims4{1, 3, minH, minW});
    profile->setDimensions("images", OptProfileSelector::kOPT, Dims4{1, 3, optH, optW});
    profile->setDimensions("images", OptProfileSelector::kMAX, Dims4{1, 3, maxH, maxW});
    config->addOptimizationProfile(profile);
    std::cout << "✅ 动态形状: MIN=" << minW << "x" << minH
              << " OPT=" << optW << "x" << optH
              << " MAX=" << maxW << "x" << maxH << std::endl;

    std::cout << "🔨 构建 FP16 引擎..." << std::endl;
    auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        std::cerr << "❌ 引擎构建失败" << std::endl;
        return false;
    }

    std::ofstream outFile(enginePath, std::ios::binary);
    outFile.write(reinterpret_cast<const char *>(plan->data()), plan->size());
    std::cout << "✅ 引擎已保存: " << enginePath << " (" << plan->size() / (1 << 20) << " MB)" << std::endl;
    return true;
}

int main(int argc, char **argv)
{
    std::string onnxPath = "../yolov5n_dynamic.onnx";
    std::string enginePath = "yolov5n_fp16_dynamic.engine";
    int optH = 320, optW = 320;

    if (argc >= 2) onnxPath = argv[1];
    if (argc >= 3) enginePath = argv[2];
    if (argc >= 4) optH = optW = std::stoi(argv[3]);

    std::cout << "========================================" << std::endl;
    std::cout << "  YOLOv5n 动态形状 FP16 引擎构建" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "ONNX:    " << onnxPath << std::endl;
    std::cout << "输出:    " << enginePath << std::endl;
    std::cout << "最优分辨率: " << optW << "x" << optH << std::endl;
    std::cout << "========================================" << std::endl;

    return buildDynamicEngine(onnxPath, enginePath, optH, optW) ? 0 : 1;
}
