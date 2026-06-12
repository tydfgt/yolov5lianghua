# Jetson Orin Nano Super 部署 YOLOv5 TensorRT 量化全攻略 (C++ 实现, 304qps+GPU后处理)

> **CSDN 博客文章**  
> 作者: ysdhanji  
> 日期: 2026-06-12  
> GitHub: https://github.com/tydfgt/yolov5lianghua

---

## 前言

在 Jetson Orin Nano Super 上跑 YOLOv5，网上教程大多是 Python + PyTorch 方案，速度感人（5-10 FPS）。本文介绍一套 **纯 C++ + TensorRT FP16 量化**方案，GPU 纯推理 **3.2ms（304 qps）**，含前后处理 **62ms（16 FPS）**，准确率与原版一致。

如果你也被 Jetson 上 PyTorch 装不上、速度慢折磨过，这篇文章应该能帮到你。

---

## 一、硬件环境

| 项目 | 配置 |
|------|------|
| 设备 | NVIDIA Jetson Orin Nano Super Developer Kit |
| 内存 | 8GB 统一内存（CPU/GPU 共享） |
| GPU | Ampere 架构, 1024 CUDA Cores, 32 Tensor Cores |
| 存储 | 256GB NVMe SSD |
| Swap | 8GB SSD swap (编译必需，防 OOM) |
| 功耗模式 | MAXN_SUPER |

> 💡 编译大型项目（OpenCV + CUDA / TensorRT 引擎）必须开 SSD swap，否则 8GB 内存不够。

---

## 二、软件环境

| 组件 | 版本 |
|------|------|
| JetPack / L4T | 6.0 / R36.5.0 |
| CUDA | 12.6.68 |
| TensorRT | 10.3.0 |
| cuDNN | 9.3.0 |
| OpenCV | 4.10.0 (自编译, CUDA 加速) |
| GCC | 11.4.0 |
| CMake | 3.26.4 |
| Python | 3.10.12 |

### 关键踩坑

1. **cuDNN 版本问题**: JetPack 6 自带 cuDNN 9.x，但有些 PyTorch wheel 需要 libcudnn.so.8。解决方案：
   ```bash
   sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9 \
                /usr/lib/aarch64-linux-gnu/libcudnn.so.8
   ```

2. **CUDA 库路径**: CUDA 12.6 的库在 `/usr/local/cuda-12.6/targets/aarch64-linux/lib/`，需要加入 ldconfig。

3. **编译 OOM**: 必须 `make -j2`，决不能 `-j6`。

---

## 三、整体方案

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ ONNX模型  │ → │  TensorRT    │ → │  .engine文件  │
│ (14MB)   │    │  构建引擎     │    │  (6.1MB)     │
└──────────┘    └──────────────┘    └──────────────┘
                                          ↓
┌──────────────────────────────────────────────────┐
│  C++ 推理程序                                     │
│  ① 预处理: BGR→RGB, resize 640, FP16             │
│  ② GPU 推理: enqueueV3 (3.2ms)                   │
│  ③ GPU 后处理: CUDA kernel 解码+筛选 (1ms)       │
│  ④ CPU NMS: 排序+IOU去重 (<1ms)                  │
└──────────────────────────────────────────────────┘
```

为什么不直接用 PyTorch？
- PyTorch for Jetson 版本匹配困难（cuDNN 9.x 不兼容）
- Python 推理慢（100-200ms/张）
- C++ + TensorRT 才是 Jetson 的正确打开方式

---

## 四、步骤 1: 获取 ONNX 模型

YOLOv5 官方提供了预导出的 ONNX 模型：

```bash
# YOLOv5n (nano, 推荐, 3.8MB)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx

# YOLOv5s (small, 15MB)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx
```

验证模型：

```python
import onnx
m = onnx.load('yolov5n.onnx')
# Input:  images, shape=[1, 3, 640, 640], type=FP16
# Output: output0, shape=[1, 25200, 85], type=FP16
```

> ⚠️ 官方 ONNX 输入输出都是 **FP16**！很多教程没提这点，导致推理结果全零。

---

## 五、步骤 2: 构建 TensorRT 引擎

### 方式一: trtexec (推荐, 8分钟)

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov5n.onnx \
    --fp16 \
    --saveEngine=yolov5n_fp16.engine
```

trtexec 输出:
```
GPU Compute Time: min = 3.14819 ms, mean = 3.28105 ms
Throughput: 304.162 qps
```

### 方式二: 自编 C++ 程序 (含 INT8 校准)

如果需要 INT8 量化（精度略低但模型更小）：

```bash
cd trt_int8/build
./build_engine_int8 yolov5s.onnx coco128/images/train2017 yolov5s_int8.engine
```

INT8 需要用 COCO128 数据集做校准（128 张图即可）。

---

## 六、步骤 3: C++ 推理程序

### 核心代码结构

```cpp
// 1. 加载引擎
auto runtime = createInferRuntime(gLogger);
auto engine = runtime->deserializeCudaEngine(engineData, size);
auto context = engine->createExecutionContext();

// 2. 预处理: CPU (OpenCV)
cv::Mat rgb, resized;
cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
cv::resize(rgb, resized, cv::Size(640, 640));
// ... 转 FP16, cudaMemcpy H2D

// 3. GPU 推理
context->setTensorAddress("images", gpuInput);
context->setTensorAddress("output0", gpuOutput);
context->enqueueV3(stream);

// 4. GPU 后处理: CUDA kernel
decodeAndFilterKernel<<<grid, block, 0, stream>>>(...);
// 每个线程处理一个 anchor: obj筛选 + top-1分类 + 边框解码

// 5. CPU NMS (候选框 < 200个)
std::sort(candidates);  // 按置信度排序
for (auto& det : candidates) {  // per-class NMS
    if (iou(det, kept) > 0.45) discard;
}
```

### GPU 后处理 kernel (关键代码)

```cuda
__global__ void decodeAndFilterKernel(
    const __half *output,     // [25200, 85]
    CandidateDevice *candidates,
    int *candidateCount, ...)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. 读 objectness (FP16→FP32)
    float objConf = __half2float(output[idx * 85 + 4]);
    if (objConf < 0.25f) return;  // 提前退出
    
    // 2. Top-1 分类 (80类)
    half bestCls{0}; int bestC = 0;
    for (int c = 0; c < 80; ++c)
        if (output[idx*85 + 5 + c] > bestCls) { bestCls = ...; bestC = c; }
    
    float score = objConf * __half2float(bestCls);
    if (score < 0.25f) return;
    
    // 3. 边框解码
    float cx = __half2float(row[0]), w = __half2float(row[2]);
    // ... 原子写入全局内存
    int slot = atomicAdd(candidateCount, 1);
    candidates[slot] = {x, y, w, h, score, bestC};
}
```

### NMS 策略

候选框通常在 50-150 个之间，直接用 CPU O(n²) + 排序即可，耗时 < 1ms。不需要 GPU NMS。

---

## 七、性能测试

```bash
cd trt_int8/build
make -j2                                  # 编译
./infer_gpu yolov5n_fp16.engine bus.jpg --benchmark
```

### 结果

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 预处理 (CPU) | ~5ms | BGR→RGB + resize 640 + FP16 |
| H2D 拷贝 | ~2ms | 1.2MB(FP16 640×640×3) → GPU |
| **GPU 推理** | **3.2ms** | TensorRT FP16 |
| GPU 后处理 | ~1ms | CUDA kernel 解码+筛选 |
| D2H 拷贝 | ~2ms | 候选框列表 → CPU |
| CPU NMS | <1ms | 排序 + IOU |
| 其他 | ~48ms | OpenCV 开销 + 内存分配 |
| **总计** | **~62ms** | **≈ 16 FPS** |

### trtexec 纯 GPU 推理

```
Throughput: 304 qps
GPU latency: 3.18ms mean
```

> 💡 GPU 本身有 304 FPS 的能力！瓶颈在 CPU 侧，降低分辨率到 320×320 理论上可达 60 FPS。

---

## 八、准确率验证

### bus.jpg (810×1080)

| 方案 | 检测结果 |
|------|---------|
| PyTorch 原版 YOLOv5s | 4 person + 1 bus |
| ONNX Runtime (CPU) | 4 person + 1 bus |
| **TensorRT YOLOv5n FP16** | **4 person + 1 bus** ✅ |
| TensorRT YOLOv5s INT8 | 4 person (bus 漏检) |

### zidane.jpg (1280×720)

| 方案 | 检测结果 |
|------|---------|
| ONNX YOLOv5s | 1 person |
| **TensorRT YOLOv5n FP16** | **1 person** ✅ |

> **FP16 量化几乎无损**。INT8 有轻微精度下降。

---

## 九、完整代码

所有代码已在 GitHub: **[https://github.com/tydfgt/yolov5lianghua](https://github.com/tydfgt/yolov5lianghua)**

### 项目结构

```
yolov5lianghua/
├── trt_int8/
│   ├── CMakeLists.txt              # CMake 构建
│   ├── PROJECT.md                  # 详细项目文档
│   ├── infer_gpu.cpp               # ★ 全GPU推理 (推荐)
│   ├── infer_fast.cpp              # 优化CPU后处理
│   ├── infer_engine.cpp            # 基础推理
│   ├── build_engine_int8.cpp       # INT8 引擎构建
│   ├── postprocess_cuda.cu         # CUDA 后处理 kernel
│   ├── postprocess_cuda.cuh        # CUDA 后处理头文件
│   └── build/                      # 编译输出
│       ├── yolov5n_fp16_trtexec.engine   # ★ 推荐引擎
│       ├── yolov5s_int8.engine           # INT8 引擎
│       ├── infer_gpu
│       ├── infer_fast
│       └── build_engine_int8
├── yolov5s.onnx                   # YOLOv5s 模型
├── yolov5n.onnx                   # YOLOv5n 模型
└── data/images/                   # 测试图片
```

### 快速开始

```bash
git clone https://github.com/tydfgt/yolov5lianghua.git
cd yolov5lianghua/trt_int8/build

# 编译
cmake .. && make -j2

# 推理 (使用预构建的引擎)
./infer_gpu yolov5n_fp16_trtexec.engine ../data/images/bus.jpg output.jpg

# 性能测试
./infer_gpu yolov5n_fp16_trtexec.engine ../data/images/bus.jpg --benchmark
```

### 构建新引擎

```bash
# FP16 (推荐)
/usr/src/tensorrt/bin/trtexec --onnx=yolov5n.onnx --fp16 \
    --saveEngine=yolov5n_fp16.engine

# INT8 (需要校准数据)
./build_engine_int8 yolov5s.onnx ../datasets/coco128/images/train2017 yolov5s_int8.engine
```

---

## 十、常见踩坑

### 1. 模型输出全零 / 检测不到目标

**原因**: ONNX 模型 I/O 是 FP16，但推理时传入了 FP32 数据。

**解决**: 预处理时做 FP32→FP16 转换，后处理时做 FP16→FP32 转换。

```cpp
// 预处理
half halfData = __float2half(floatData);
// 后处理
float floatVal = __half2float(halfData);
```

### 2. trtexec 构建报错 "reshape would change volume"

**原因**: YOLOv5 的 Reshape 层不支持动态形状。

**解决**: 固定 640×640 输入，不要用动态形状。

### 3. 编译报错 "undefined reference to createNvOnnxParser_INTERNAL"

**解决**: CMakeLists.txt 中链接 `nvonnxparser` 库。

### 4. 内存不足 OOM

`make -j2` 而非 `-j6`。确保 8GB SSD swap 已启用。

---

## 十一、总结

| 指标 | 数值 |
|------|------|
| 模型 | YOLOv5n (nano) |
| 量化 | TensorRT FP16 |
| 引擎大小 | 6.1 MB |
| GPU 纯推理 | 3.2ms / 304 qps |
| 端到端 (640×640) | 62ms / 16 FPS |
| 准确率 | 与原版一致 ✅ |

**核心经验**:
1. Jetson 上深度学习 = **C++ + TensorRT**，别用 Python
2. **trtexec** 构建引擎比自编代码快
3. FP16 量化几乎无损精度
4. GPU 后处理 kernel 替代 CPU 循环
5. 瓶颈在预处理，降低分辨率可进一步提升 FPS

---

> 📝 本文所有代码开源在 [GitHub](https://github.com/tydfgt/yolov5lianghua)，欢迎 Star ⭐  
> 📝 详细项目文档见仓库 `trt_int8/PROJECT.md`
