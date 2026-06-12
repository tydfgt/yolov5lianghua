# Jetson Orin Nano Super 部署 YOLOv5 TensorRT 量化全攻略 (C++ 实现, 304qps+GPU后处理)

> **作者**: cedarQ  
> **日期**: 2026-06-12  
> **适用**: Jetson Orin Nano Super / JetPack 6.0 / TensorRT 10.x

---

## 前言

在 Jetson Orin Nano Super 上跑 YOLOv5，网上教程大多是 Python + PyTorch 方案，速度感人（5-10 FPS）。本文介绍一套 **纯 C++ + TensorRT FP16 量化**方案，GPU 纯推理 **3.2ms（304 qps）**，含前后处理 **62ms（16 FPS）**，准确率与原版一致。

与其在 Jetson 上折腾 PyTorch 环境（cuDNN 版本不兼容、ARM 架构 wheel 难找、Python 推理慢），不如直接上 C++ + TensorRT——本文就是这条路线的完整实战记录。完整代码已开源（链接见文末），可以直接 clone 下来编译运行。

---

## 一、硬件环境

| 项目 | 配置 |
|------|------|
| 设备 | NVIDIA Jetson Orin Nano Super Developer Kit |
| 内存 | 8GB 统一内存（CPU/GPU 共享） |
| GPU | Ampere 架构, 1024 CUDA Cores, 32 Tensor Cores |
| 存储 | ~256GB NVMe SSD |
| Swap | 8GB SSD swap (编译必需，防 OOM) |
| 功耗模式 | MAXN_SUPER (15W) |

> 💡 编译 TensorRT 引擎必须开 SSD swap，否则 8GB 内存不够。后面有 swap 创建命令。

---

## 二、软件环境

| 组件 | 版本 | 备注 |
|------|------|------|
| JetPack / L4T | 6.0 / R36.5.0 | `cat /etc/nv_tegra_release` |
| CUDA | 12.6.68 | `nvcc --version` |
| TensorRT | 10.3.0 | `dpkg -l | grep tensorrt` |
| cuDNN | 9.3.0 | JetPack 6 自带 |
| OpenCV | 4.10.0 | 自编译，CUDA 加速，11个 CUDA 模块 |
| GCC | 11.4.0 | ARM 交叉编译注意版本 |
| CMake | 3.26.4 | 需要 3.18+ |
| Python | 3.10.12 | 仅用于 ONNX 操作，不参与推理 |

### 1.1 YOLOv5 变体对比

| 模型 | 参数量 | ONNX 大小 | COCO mAP | GPU 推理 (Orin Nano) |
|------|--------|-----------|----------|---------------------|
| YOLOv5n | 1.9M | 3.8MB | 28.0% | **3.2ms** |
| YOLOv5s | 7.2M | 14MB | 37.4% | ~8ms |
| YOLOv5m | 21.2M | 40MB | 45.4% | ~15ms |
| YOLOv5l | 46.5M | 89MB | 49.0% | ~30ms |

对于边缘设备实时检测，**YOLOv5n 是最佳平衡点**——推理最快、模型最小。高度敏感场景（安防、医疗）可升级到 YOLOv5s。

### 1.2 量化方式对比

| 量化 | 精度损失 | 构建时间 | 引擎大小 | 推理速度 |
|------|---------|---------|---------|---------|
| FP32 | 基准 | ~5分钟 | ~12MB | 1× |
| **FP16** | **几乎无损** | **~8分钟** | **6.1MB** | **~2×** |
| INT8 | 轻微 (<1% mAP) | ~15分钟 | 9.1MB | ~3× |

**推荐 FP16**——简单、快速、精度无损。INT8 需要校准数据集且可能漏检小目标。

### 1.3 环境准备脚本

```bash
# === 确认环境 ===
cat /etc/nv_tegra_release    # 应为 R36.x
nvcc --version               # 12.6
dpkg -l | grep tensorrt      # 10.3.0

# === 创建 8GB SSD swap (编译必需) ===
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# === 修复 cuDNN 兼容性 ===
# (有些库需要 libcudnn.so.8，JetPack 6 只有 9)
sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9 \
             /usr/lib/aarch64-linux-gnu/libcudnn.so.8

# === CUDA 库路径 ===
echo '/usr/local/cuda-12.6/targets/aarch64-linux/lib' | \
    sudo tee /etc/ld.so.conf.d/cuda-12-6.conf
sudo ldconfig

# === 锁定最高频率 ===
sudo jetson_clocks

# === Python venv (仅用于 ONNX 检查) ===
python3 -m venv venv
source venv/bin/activate
pip install onnx onnxruntime opencv-python-headless
```

> ⚠️ 不要在 Jetson 上 `pip install torch`——cuDNN 9.x 不兼容 PyTorch wheel。用 ONNX + TensorRT 替代。

---

## 三、整体方案

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│ ONNX模型  │ → │  TensorRT    │ → │  .engine文件  │
│  (下载)   │    │  构建引擎     │    │  (6.1MB)     │
└──────────┘    └──────────────┘    └──────────────┘
                                          ↓
┌──────────────────────────────────────────────────┐
│  C++ 推理程序 (infer_gpu)                         │
│  ① 预处理 (CPU): BGR→RGB, resize, FP16转         │
│  ② H2D: cudaMemcpy 上传 GPU                       │
│  ③ GPU推理: enqueueV3 (3.2ms)                   │
│  ④ GPU后处理: CUDA kernel 解码+筛选 (1ms)        │
│  ⑤ D2H + CPU NMS: 排序+IOU去重 (<1ms)           │
└──────────────────────────────────────────────────┘
```

**为什么不用 PyTorch？**
- JetPack 6 cuDNN 9.x 与 PyTorch wheel 不兼容
- Python 推理慢（100-200ms/张）
- C++ + TensorRT 才是 Jetson 的性能正确打开方式

---

## 四、步骤 1: 获取 ONNX 模型

### 方式一: 直接下载官方预导出（最简单）

```bash
# YOLOv5n (推荐，3.8MB，3.2ms GPU推理)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx

# YOLOv5s (15MB，~8ms GPU推理)
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx
```

### 方式二: 从 PyTorch 自己导出

在 x86 机器上（有 PyTorch 环境）：

```python
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()

dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy, 'yolov5n.onnx',
    opset_version=17,
    input_names=['images'],
    output_names=['output0'])
```

### 方式三: 用 YOLOv5 的 export.py

```bash
# 在 x86 机器上
python export.py --weights yolov5n.pt --include onnx --half
```

### 验证模型

```python
import onnx
m = onnx.load('yolov5n.onnx')
for inp in m.graph.input:
    t = inp.type.tensor_type
    print(f"Input: {inp.name}, shape={[d.dim_value for d in t.shape.dim]}, type={'FP16' if t.elem_type==10 else 'FP32'}")
for out in m.graph.output:
    t = out.type.tensor_type
    print(f"Output: {out.name}, shape={[d.dim_value for d in t.shape.dim]}, type={'FP16' if t.elem_type==10 else 'FP32'}")
# 预期:
# Input:  images, shape=[1, 3, 640, 640], type=FP16
# Output: output0, shape=[1, 25200, 85], type=FP16
```

### YOLOv5 输出格式详解

```
output shape: [1, 25200, 85]

25200 = 80×80×3 + 40×40×3 + 20×20×3 (3个检测层anchor总数)

85 = 4(bbox) + 1(objectness) + 80(COCO类别)
     ├─ cx, cy, w, h  归一化坐标 (相对于输入尺寸)
     ├─ objectness     sigmoid 后置信度
     └─ 80个类别       原始 logits (后处理取 argmax)
```

> ⚠️ **官方ONNX的I/O是FP16**（elem_type=10）！如果用FP32数据丢进去，结果全是零。必须做FP32↔FP16转换！

---

## 五、步骤 2: 构建 TensorRT 引擎

### 方式一: trtexec（推荐，~8分钟）

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov5n.onnx \
    --fp16 \
    --saveEngine=yolov5n_fp16.engine \
    --workspace=512
```

trtexec 自动运行性能测试：

```
[I] Throughput: 304.162 qps              ← 纯GPU吞吐
[I] Latency: mean = 3.69ms              ← 含H2D/D2H
[I] GPU Compute Time: mean = 3.28ms     ← 纯计算
[I] H2D Latency: mean = 0.18ms
[I] D2H Latency: mean = 0.23ms
```

> 看到 "GPU compute time is unstable"？运行 `sudo jetson_clocks` 锁定频率。

### 方式二: C++ API + INT8 校准

自编的 `build_engine_int8` 支持 INT8 量化。需要 COCO128 校准集。

核心步骤：
1. 创建 `IBuilder` + `INetworkDefinition`
2. 用 `nvonnxparser` 解析 ONNX
3. 设置 `kINT8` flag + 绑定 `IInt8EntropyCalibrator2`
4. `buildSerializedNetwork` → 保存 `.engine`

校准器需要实现：
- `getBatch()` — 每次返回一个 batch 的预处理图像（FP16）
- `readCalibrationCache()` / `writeCalibrationCache()` — 缓存校准结果

完整代码见仓库的 `build_engine_int8.cpp`。

### 方式三: 命令行快速 INT8

```bash
cd trt_int8/build
make build_engine_int8 -j2
./build_engine_int8 yolov5s.onnx path/to/calib/images yolov5s_int8.engine
```

### 引擎对比

| 模型 | 精度 | 大小 | 构建 |
|------|------|------|------|
| YOLOv5n | FP16 | **6.1MB** | 8min |
| YOLOv5s | FP16 | ~10MB | 10min |
| YOLOv5s | INT8 | 9.1MB | 15min |

---

## 六、步骤 3: C++ 推理程序

### 6.1 CMakeLists.txt

Jetson 上 TensorRT 路径与 x86 不同，注意 `aarch64-linux-gnu`：

```cmake
cmake_minimum_required(VERSION 3.18)
project(yolov5_trt_int8 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# TensorRT 库 — 注意 aarch64 路径
find_library(TENSORRT_LIB nvinfer
    PATHS /usr/lib/aarch64-linux-gnu
          /usr/local/cuda-12.6/targets/aarch64-linux/lib)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin ...)
find_library(TENSORRT_ONNX_PARSER_LIB nvonnxparser ...)
find_library(CUDART_LIB cudart ...)

# 头文件
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu)

# OpenCV
find_package(OpenCV REQUIRED)

# 全GPU推理
add_executable(infer_gpu infer_gpu.cpp postprocess_cuda.cu)
target_include_directories(infer_gpu PRIVATE ...)
target_link_libraries(infer_gpu
    ${TENSORRT_LIB} ${TENSORRT_PLUGIN_LIB}
    ${CUDART_LIB} ${CUDA_LIB} ${OpenCV_LIBS})
```

> ⚠️ 在 x86 上把 `aarch64` 改为 `x86_64`。

### 6.2 编译

```bash
cd trt_int8/build
cmake ..
make -j2        # ★ 必须 -j2，-j6 会 OOM！
```

编译产生：
- `build_engine_int8` — INT8 引擎构建
- `infer_engine` — 基础推理 (纯 CPU 后处理)
- `infer_fast` — 优化推理 (pinned memory + CUDA stream)
- **`infer_gpu`** — ★ **全GPU推理** (含 CUDA 后处理)

### 6.3 完整推理代码 (关键部分)

#### 引擎加载

```cpp
// 读取序列化引擎
std::ifstream file(enginePath, std::ios::binary);
file.seekg(0, std::ios::end);
size_t size = file.tellg(); file.seekg(0, std::ios::beg);
std::vector<char> data(size);
file.read(data.data(), size);

// 反序列化
auto runtime = createInferRuntime(gLogger);
auto engine = runtime->deserializeCudaEngine(data.data(), size);
auto context = engine->createExecutionContext();

// 获取 I/O 形状
auto inDims = engine->getTensorShape("images");
int H = inDims.d[2], W = inDims.d[3];  // 640x640
auto outDims = engine->getTensorShape("output0");
int N = outDims.d[1];   // 25200
int C = outDims.d[2] - 5; // 80

// GPU 内存
half *inGPU, *outGPU;
cudaMalloc(&inGPU, 3*H*W*sizeof(half));
cudaMalloc(&outGPU, N*(5+C)*sizeof(half));

// CUDA stream (避免默认流同步)
cudaStream_t stream; cudaStreamCreate(&stream);
context->setTensorAddress("images", inGPU);
context->setTensorAddress("output0", outGPU);
```

#### 预处理

```cpp
// BGR → RGB, resize, normalize, FP32→FP16
cv::Mat rgb, resized, f32;
cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
cv::resize(rgb, resized, cv::Size(640, 640));
resized.convertTo(f32, CV_32F, 1.0/255.0);

// HWC → CHW + FP16 转换
std::vector<cv::Mat> chs(3); cv::split(f32, chs);
std::vector<float> host(3*H*W);
for(int c=0; c<3; ++c)
    memcpy(host.data()+c*H*W, chs[c].data, H*W*sizeof(float));

std::vector<half> hf(3*H*W);
for(size_t i=0; i<hf.size(); ++i)
    hf[i] = __float2half(host[i]);

cudaMemcpyAsync(inGPU, hf.data(), hf.size()*sizeof(half),
                cudaMemcpyHostToDevice, stream);
```

> 提示：这里的 FP32→FP16 转换是必须的。如果跳过这步直接传 FP32 数据，推理结果全是零。读者可以自己试试去掉转换会怎样。

#### GPU 推理

```cpp
// 一行搞定
context->enqueueV3(stream);
```

#### GPU 后处理 CUDA Kernel

```cuda
__global__ void decodeAndFilterKernel(
    const __half *output,     // [N, 85] FP16
    Candidate *cands, int *count,
    int N, int C, float thresh,
    int imgW, int imgH, int inW, int inH)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    const __half *row = output + i*(5+C);

    // 1. 读 objectness
    float obj = __half2float(row[4]);
    if(obj < thresh) return;

    // 2. 找 top-1 类别 (这里有个小坑)
    half best{0}; int cls = 0;
    for(int c=0; c<C; ++c) {
        if(row[5+c] > best) { best = row[5+c]; cls = c; }
    }

    float score = obj * __half2float(best);
    if(score < thresh) return;

    // 3. 解码归一化坐标 → 像素坐标
    float cx = __half2float(row[0]), cy = __half2float(row[1]);
    float w  = __half2float(row[2]), h  = __half2float(row[3]);
    float sx = (float)imgW / inW, sy = (float)imgH / inH;

    int slot = atomicAdd(count, 1);
    if(slot < 1024) {
        cands[slot].x = (cx - w/2) * sx;
        cands[slot].y = (cy - h/2) * sy;
        cands[slot].w = w * sx;
        cands[slot].h = h * sy;
        cands[slot].score = score;
        cands[slot].cls = cls;
    }
}

// 调用
int grid = (25200 + 255) / 256;
decodeAndFilterKernel<<<grid, 256, 0, stream>>>(...);
```

#### CPU NMS

```cpp
// 候选框拷贝回 CPU（通常 <200个，极小）
std::vector<Candidate> cands(count);
cudaMemcpy(cands.data(), dCands, count*sizeof(Candidate), ...);

// 按置信度降序排列
std::sort(cands.begin(), cands.end(),
    [](auto& a, auto& b){ return a.score > b.score; });

// Per-class NMS
std::vector<bool> keep(count, true);
std::vector<Detection> results;
for(int i=0; i<count && results.size()<100; ++i) {
    if(!keep[i]) continue;
    results.push_back({cands[i]});

    for(int j=i+1; j<count; ++j) {
        if(!keep[j]) continue;
        if(cands[i].cls != cands[j].cls) continue;

        // IOU 计算
        float ax1=cands[i].x, ay1=cands[i].y;
        float ax2=ax1+cands[i].w, ay2=ay1+cands[i].h;
        float bx1=cands[j].x, by1=cands[j].y;
        float bx2=bx1+cands[j].w, by2=by1+cands[j].h;
        float inter = max(0.f, min(ax2,bx2)-max(ax1,bx1))
                    * max(0.f, min(ay2,by2)-max(ay1,by1));
        float iou = inter / (cands[i].w*cands[i].h
                   + cands[j].w*cands[j].h - inter + 1e-6f);

        if(iou > 0.45f) keep[j] = false;
    }
}
```

> 这里的 NMS 实现有些刻意保留了冗余变量定义，读者可以自行合并优化。候选框少时 O(n²) 完全够用，不必引入第三方库。

---

## 七、性能测试

### 运行

```bash
cd trt_int8/build
make infer_gpu -j2 && ./infer_gpu yolov5n_fp16.engine bus.jpg --benchmark
```

### 分阶段耗时

| 阶段 | 耗时 | 位置 | 说明 |
|------|------|------|------|
| 预处理 | ~5ms | CPU | BGR→RGB + resize + FP16 |
| H2D 拷贝 | ~2ms | PCIe/DMA | 1.2MB → GPU |
| **GPU 推理** | **3.2ms** | **GPU** | **TensorRT FP16** |
| GPU 后处理 | ~1ms | GPU | CUDA kernel decode+filter |
| D2H 拷贝 | ~2ms | PCIe/DMA | 候选框 → CPU |
| CPU NMS | <1ms | CPU | 排序 + IOU |
| 其他开销 | ~48ms | — | OpenCV + 内存分配 |
| **总计** | **~62ms** | — | **≈ 16 FPS** |

### 各版本对比

| 版本 | 后处理方式 | 延迟 | FPS |
|------|-----------|------|-----|
| `infer_engine` | CPU 全量 FP32 | 132ms | 7.6 |
| `infer_fast` | CPU 优化 (pinned+stream) | 64ms | 15.7 |
| `infer_gpu` | **GPU kernel + CPU NMS** | **62ms** | **16.1** |
| trtexec 纯推理 | 无后处理 | **3.2ms** | **304 qps** |

> 💡 GPU 本身有 304 FPS 的能力！瓶颈在"其他开销"（OpenCV 内存分配、每帧 vector 创建）。降低分辨率到 320×320 预计可达 40-60 FPS。

### GPU 利用率验证

```bash
# 运行时另开终端
sudo tegrastats
# 观察 GR3D_FREQ — 推理时应接近 100%
```

---

## 八、准确率验证

### bus.jpg (810×1080)

| 方案 | 检测结果 |
|------|---------|
| ONNX Runtime YOLOv5s | 4 person + 1 bus |
| **TRT YOLOv5n FP16 (infer_gpu)** | **4 person + 1 bus** ✅ |
| TRT YOLOv5s INT8 | 4 person（bus 漏检）⚠️ |

### zidane.jpg (1280×720)

| 方案 | 检测结果 |
|------|---------|
| ONNX Runtime YOLOv5s | 1 person |
| **TRT YOLOv5n FP16** | **1 person (conf=0.38)** ✅ |

> **FP16 量化几乎无损精度**。INT8 有轻微精度损失，小目标可能漏检。

---

## 九、常见踩坑与解决方案

### 1. 模型输出全零 / 检测不到目标

**原因**: ONNX I/O 是 FP16，推理传入了 FP32。

**解决**: 预处理做 `__float2half()`，后处理做 `__half2float()`。注意检查 `elem_type==10`（FP16）。

### 2. trtexec 报错 "reshape would change volume"

**原因**: YOLOv5 的 Reshape 层不支持动态形状。

**解决**: 固定输入 shape `[1, 3, 640, 640]`，不要设置 `--minShapes/--optShapes/--maxShapes`。

### 3. 编译报错 "undefined reference to createNvOnnxParser_INTERNAL"

**解决**: CMakeLists.txt 没有链接 `nvonnxparser` 库。加上 `find_library(TENSORRT_ONNX_PARSER_LIB nvonnxparser ...)` 并在 target_link_libraries 中加入。

### 4. make -j6 内存耗尽 OOM

Jetson 只有 8GB 共享内存，**必须 `make -j2`**。如果还 OOM，确保 8GB SSD swap 已启用。

### 5. TensorRT 10.x API 变化

TRT 10.x 废弃了 `kEXPLICIT_BATCH` 和 `getBindingDimensions()`，改用：
- `createNetworkV2(0)` 代替 `createNetworkV2(1U << kEXPLICIT_BATCH)`
- `getTensorShape("images")` 代替 `getBindingDimensions(0)`
- `setTensorAddress()` 代替 `setBindingAddress()`
- `enqueueV3()` 代替 `enqueueV2()`

### 6. cuDNN 版本不兼容

JetPack 6 带 cuDNN 9.x，PyTorch 需要 8.x。不用 PyTorch 就没事——本文 ONNX + TRT 方案完全避免此问题。

### 7. 推理结果抖动

`sudo jetson_clocks` 锁定 CPU/GPU 频率可消除。

### 8. FP16 数据精度注意事项

FP16 的数值范围有限（±65504），normalize 到 [0,1] 后精度足够。但如果预处理中用了异常大的值，可能出现 ±inf。建议归一化在 FP32 做，最后一步转 FP16。

---

## 十、项目代码和使用

### 项目结构

```
├── trt_int8/
│   ├── CMakeLists.txt              # CMake 构建配置
│   ├── build_engine_int8.cpp       # INT8 引擎构建 (含校准器)
│   ├── build_engine_dynamic.cpp    # 动态形状构建 (实验)
│   ├── infer_gpu.cpp               # ★ 全GPU推理 (推荐)
│   ├── infer_fast.cpp              # 优化推理 (pinned+stream)
│   ├── infer_engine.cpp            # 基础推理 (学习用)
│   ├── infer_dynamic.cpp           # 动态形状推理 (实验)
│   ├── postprocess_cuda.cuh        # CUDA 后处理头文件
│   ├── postprocess_cuda.cu         # CUDA 后处理 kernel
│   ├── CSDN_ARTICLE.md             # 本文
│   └── build/                      # 编译输出目录
├── yolov5n.onnx / yolov5s.onnx     # 需自行下载 ONNX 模型
└── data/images/                    # 测试图片
```

### 快速开始

```bash
# 1. 编译
cd trt_int8/build && cmake .. && make -j2

# 2. 下载 ONNX (如未下载)
wget -O ../yolov5n.onnx \
  https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx

# 3. 构建引擎 (FP16, ~8分钟)
/usr/src/tensorrt/bin/trtexec --onnx=../yolov5n.onnx --fp16 \
    --saveEngine=yolov5n_fp16.engine

# 4. 推理测试
./infer_gpu yolov5n_fp16.engine ../data/images/bus.jpg output.jpg

# 5. 性能测试
./infer_gpu yolov5n_fp16.engine ../data/images/bus.jpg --benchmark
```

### 构建 INT8 引擎

```bash
# 需要 COCO128 校准集 (128 张图即可)
./build_engine_int8 ../yolov5s.onnx path/to/coco128/images yolov5s_int8.engine
```

---

## 十一、总结

| 指标 | 数值 |
|------|------|
| 模型 | YOLOv5n |
| 量化 | TensorRT FP16 |
| 引擎大小 | 6.1 MB |
| GPU 纯推理 | **3.2ms / 304 qps** |
| 端到端 (640×640) | 62ms / 16 FPS |
| 准确率 | 与原版一致 ✅ |

### 核心经验

1. **Jetson 上的深度学习 = C++ + TensorRT**，Python 只是辅助
2. **trtexec** 是构建引擎最简单可靠的方式
3. FP16 量化几乎无损精度，INT8 有轻微损失
4. GPU 后处理 CPUDA kernel 替代 CPU 循环是最佳实践
5. 预处理和后处理是主要瓶颈，降低分辨率可大幅提升 FPS
6. `make -j2` 不能多，Jetson 内存宝贵
7. 官方 ONNX 的 I/O 是 FP16——这是最容易忽略的点

### 下一步优化方向

- CUDA 预处理 (resize + normalize → GPU)
- 内存池预分配消除动态分配
- 双缓冲流水线 (预处理与推理并行)
- 320×320 分辨率（预计 40-60 FPS）
- DLA 加速器（YOLOv5 改造后可用）

---

> 📝 完整源码开源，详见代码仓库。欢迎交流讨论。
