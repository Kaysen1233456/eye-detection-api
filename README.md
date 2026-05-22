# Eye Detection API | 闭眼检测 API

基于 UniFormerV2 的驾驶员闭眼检测系统，支持单 GPU / 多 GPU 并行推理，提供视频标注工具。

UniFormerV2-based driver eye state detection system with single/multi-GPU inference and video annotation tool.

---

## 目录 | Table of Contents

- [功能特性 | Features](#功能特性--features)
- [检测逻辑 | Detection Logic](#检测逻辑--detection-logic)
- [项目结构 | Project Structure](#项目结构--project-structure)
- [环境依赖 | Dependencies](#环境依赖--dependencies)
- [安装方式 | Installation](#安装方式--installation)
- [快速开始 | Quick Start](#快速开始--quick-start)
- [API 接口说明 | API Reference](#api-接口说明--api-reference)
- [视频标注工具 | Video Annotation Tool](#视频标注工具--video-annotation-tool)
- [性能基准 | Benchmark](#性能基准--benchmark)
- [模型信息 | Model Info](#模型信息--model-info)

---

## 功能特性 | Features

| 功能 | 说明 |
|------|------|
| 闭眼检测 | 输入视频片段，返回睁/闭眼状态及置信度 |
| 多 GPU 并行 | 自动检测 GPU 数量，负载均衡分配推理任务 |
| 视频标注 | 逐帧叠加闭眼概率和状态，输出标注后的视频 |
| 人脸提取 | 可选的人脸裁剪功能，使用 Haar 级联检测器提升精度 |

| Feature | Description |
|---------|-------------|
| Eye Detection | Input video segment, return open/closed eye state with confidence |
| Multi-GPU Inference | Auto-detect GPU count, balanced load distribution |
| Video Annotation | Overlay real-time eye-closed probability on each frame |
| Face Extraction | Optional face cropping using Haar cascade detector for improved accuracy |

---

## 检测逻辑 | Detection Logic

### 整体流程 | Overall Pipeline

```
输入视频 → 时间切片 → 帧采样 → [人脸提取] → 模型推理 → 输出结果
Input Video → Time Slice → Frame Sampling → [Face Crop] → Model Inference → Output
```

### 详细步骤 | Detailed Steps

#### 1. 时间切片 | Time Slicing

根据 `event_st_time` 和 `event_ed_time`（毫秒级时间戳）从视频中截取对应时间段的帧：

Slice frames from video based on `event_st_time` and `event_ed_time` (millisecond timestamps):

```
事件时间范围 = [event_st_time, event_ed_time]
对应视频帧 = 视频起始帧 + (事件时间 - 视频起始时间) × FPS
```

#### 2. 帧采样 | Frame Sampling

采用 **SlowFast** 架构的双路径采样策略：

Uses **SlowFast** architecture dual-path sampling strategy:

- **Slow Path（慢路径）**：低帧率采样，捕获全局语义信息
  - 采样 8 帧，采样间隔 8 帧
  - 覆盖约 128 帧的时间窗口
- **Fast Path（快路径）**：高帧率采样，捕获快速运动信息
  - 采样 32 帧，采样间隔 2 帧
- **时间采样率 (SAMPLING_RATE)**：16，控制整体采样密度
- **集成视图 (NUM_ENSEMBLE_VIEWS)**：4，多次采样取平均提升稳定性
- **空间裁剪 (NUM_SPATIAL_CROPS)**：3，左/中/右三个位置裁剪

```
总推理视图数 = NUM_ENSEMBLE_VIEWS × NUM_SPATIAL_CROPS = 4 × 3 = 12 视图/次
```

- **Slow Path**: Low frame rate sampling for global semantic information
  - 8 frames, sampling interval 8
  - Covers ~128 frame time window
- **Fast Path**: High frame rate sampling for fast motion capture
  - 32 frames, sampling interval 2
- **SAMPLING_RATE**: 16, controls overall sampling density
- **NUM_ENSEMBLE_VIEWS**: 4, multiple sampling for stability
- **NUM_SPATIAL_CROPS**: 3, left/center/right spatial crops

#### 3. 人脸提取（可选）| Face Extraction (Optional)

启用 `enable_face_extraction=True` 时：

When `enable_face_extraction=True`:

1. 使用 OpenCV Haar 级联检测器 (`haarcascade_frontalface_default.xml`) 定位人脸
2. 以人脸中心为基准，按 `face_pad_ratio`（默认 0.5）扩展裁剪区域
3. 扩展后的区域包含额头和下巴，确保眼部区域完整
4. 若检测不到人脸，回退到全图推理

1. Use OpenCV Haar cascade detector to locate face
2. Expand crop region by `face_pad_ratio` (default 0.5) from face center
3. Expanded region includes forehead and chin for complete eye area
4. Fall back to full frame if no face detected

#### 4. 模型推理 | Model Inference

- **模型**：UniFormerV2-B/16（ViT-B/16 架构）
- **输入**：224×224 RGB 图像序列
- **输出**：二分类概率 `[eye_open, eye_close]`
- **预训练**：K400（Kinetics-400）+ K710（Kinetics-710）+ DMS 数据集
- **推理流程**：
  1. 对 12 个视图分别推理
  2. Softmax 计算每个视图的类别概率
  3. 所有视图的概率取平均
  4. 最终概率 > 0.5 则判定为对应状态

- **Model**: UniFormerV2-B/16 (ViT-B/16 architecture)
- **Input**: 224×224 RGB image sequence
- **Output**: Binary classification `[eye_open, eye_close]`
- **Pretraining**: K400 + K710 + DMS datasets
- **Inference**:
  1. Inference on 12 views separately
  2. Softmax for class probabilities per view
  3. Average probabilities across all views
  4. Final probability > 0.5 determines state

#### 5. 多 GPU 并行 | Multi-GPU Parallel

`MultiGPUEyeDetector` 实现多卡并行：

`MultiGPUEyeDetector` implements multi-GPU parallelism:

- 自动检测可用 GPU 数量
- 将视图均匀分配到各 GPU（每卡默认 76 个视图）
- 使用 `ThreadPoolExecutor` 并行执行推理
- 合并所有 GPU 的结果取平均

- Auto-detect available GPU count
- Distribute views evenly across GPUs (default 76 views per GPU)
- Use `ThreadPoolExecutor` for parallel inference
- Merge results from all GPUs by averaging

---

## 项目结构 | Project Structure

```
eye_detect_api/
├── eye_detect_api.py           # 核心检测 API（单 GPU / 多 GPU）
│   ├── EyeDetectionAPI          - 单 GPU 检测器
│   ├── MultiGPUEyeDetector      - 多 GPU 并行检测器
│   ├── create_eye_detector()    - 创建单 GPU 检测器
│   └── create_multi_gpu_detector() - 创建多 GPU 检测器
├── annotate_video.py            # 视频标注工具
├── bench_multi_gpu.py           # 多 GPU 性能基准测试
├── bench_gpu.py                 # 单 GPU 性能基准测试
├── setup.py                     # 安装配置
├── best.pyth                    # 模型权重（UniFormerV2-B/16）
├── exp/                         # 模型配置文件
│   └── humanfactor/dms+k400_k710_b16_f8x224/
│       └── config.yaml          - 模型超参配置
├── slowfast/                    # SlowFast 框架代码
│   ├── config/                  - 配置系统
│   ├── datasets/                - 数据加载
│   ├── models/                  - 模型定义（UniFormerV2）
│   ├── utils/                   - 工具函数
│   └── visualization/           - 可视化工具
├── extract_clip/                # CLIP 特征提取工具
├── tools/                       # 训练/测试脚本
└── test.mp4                     # 测试视频
```

---

## 环境依赖 | Dependencies

### 核心依赖 | Core Dependencies

| 包名 | 版本要求 | 说明 |
|------|----------|------|
| Python | >= 3.10 | 推荐 3.12 |
| PyTorch | >= 2.0 | CUDA 11.8+ |
| torchvision | >= 0.15 | 视频变换 |
| decord | >= 0.6 | 高效视频读取 |
| opencv-python | >= 4.8 | 人脸检测、图像处理 |
| numpy | >= 1.24 | 数值计算 |
| timm | >= 0.9 | PyTorch Image Models |
| yacs | >= 0.1.6 | 配置管理 |
| pyyaml | >= 5.1 | YAML 解析 |

| Package | Version | Description |
|---------|---------|-------------|
| Python | >= 3.10 | Recommended 3.12 |
| PyTorch | >= 2.0 | CUDA 11.8+ |
| torchvision | >= 0.15 | Video transforms |
| decord | >= 0.6 | Efficient video reader |
| opencv-python | >= 4.8 | Face detection, image processing |
| numpy | >= 1.24 | Numerical computing |
| timm | >= 0.9 | PyTorch Image Models |
| yacs | >= 0.1.6 | Configuration management |
| pyyaml | >= 5.1 | YAML parser |

### 完整依赖 | Full Dependencies

```
torch>=2.0
torchvision>=0.15
decord>=0.6
opencv-python>=4.8
numpy>=1.24
timm>=0.9
yacs>=0.1.6
pyyaml>=5.1
av
matplotlib
termcolor>=1.1
simplejson
tqdm
psutil
pandas
pillow
scikit-learn
tensorboard
```

### GPU 环境 | GPU Environment

```
NVIDIA Driver >= 525
CUDA >= 11.8
cuDNN >= 8.6
```

---

## 安装方式 | Installation

```bash
# 克隆仓库 | Clone repository
git clone https://github.com/Kaysen1233456/eye-detection-api.git
cd eye-detection-api

# 安装依赖 | Install dependencies
pip install -e .

# 或手动安装 | Or install manually
pip install torch torchvision decord opencv-python timm yacs pyyaml tqdm psutil
```

---

## 快速开始 | Quick Start

### 1. 单 GPU 检测 | Single GPU Detection

```python
from eye_detect_api import EyeDetectionInput, create_eye_detector

# 创建检测器 | Create detector
detector = create_eye_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    device="cuda:0",
    enable_face_extraction=True,  # 启用人脸提取
    face_pad_ratio=0.5,           # 人脸区域扩展 50%
)

# 构造输入 | Prepare input
inp = EyeDetectionInput(
    video_path="test.mp4",
    video_st_time=1700000000,      # 视频起始时间（秒）
    video_ed_time=1700000011,      # 视频结束时间（秒）
    event_st_time=1700000000000,   # 事件起始时间（毫秒）
    event_ed_time=1700000001000,   # 事件结束时间（毫秒）
)

# 执行检测 | Run detection
result = detector.detect(inp)
print(f"状态 | Status: {result.result}")
print(f"置信度 | Confidence: {result.confidence:.3f}")
print(f"成功 | Success: {result.success}")
```

### 2. 多 GPU 检测 | Multi-GPU Detection

```python
from eye_detect_api import create_multi_gpu_detector

# 自动检测所有可用 GPU | Auto-detect all available GPUs
detector = create_multi_gpu_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    enable_face_extraction=True,
    face_pad_ratio=0.5,
)

result = detector.detect(inp)
```

### 3. 视频标注 | Video Annotation

```bash
# 单 GPU | Single GPU
python annotate_video.py -i input.mp4 -o output.mp4 --gpu

# 指定参数 | Custom parameters
python annotate_video.py -i input.mp4 -o output.mp4 --gpu \
    --window 32 \    # 检测窗口帧数
    --step 16        # 滑窗步长
```

输出视频右上角会叠加：

Output video top-right corner overlay:

```
┌─────────────────────────────┐
│ Eye Detection               │
│ Status: OPEN (绿色/CLOSED 红色) │
│ ████████████░░░░░░░ 65.3%   │
│ Close: 34.7%  Open: 65.3%   │
│ Inference: 45231ms          │
└─────────────────────────────┘
```

---

## API 接口说明 | API Reference

### EyeDetectionInput

| 参数 | 类型 | 说明 |
|------|------|------|
| `video_path` | `str` | 视频文件路径 |
| `video_st_time` | `int` | 视频起始时间（Unix 时间戳，秒） |
| `video_ed_time` | `int` | 视频结束时间（Unix 时间戳，秒） |
| `event_st_time` | `int` | 事件起始时间（Unix 时间戳，毫秒） |
| `event_ed_time` | `int` | 事件结束时间（Unix 时间戳，毫秒） |

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_path` | `str` | Video file path |
| `video_st_time` | `int` | Video start time (Unix timestamp, seconds) |
| `video_ed_time` | `int` | Video end time (Unix timestamp, seconds) |
| `event_st_time` | `int` | Event start time (Unix timestamp, milliseconds) |
| `event_ed_time` | `int` | Event end time (Unix timestamp, milliseconds) |

### EyeDetectionOutput

| 字段 | 类型 | 说明 |
|------|------|------|
| `result` | `str` | `"eye_open"` 或 `"eye_close"` |
| `confidence` | `float` | 置信度（0-1） |
| `success` | `bool` | 是否检测成功 |
| `error_message` | `str | None` | 错误信息（失败时） |

| Field | Type | Description |
|-------|------|-------------|
| `result` | `str` | `"eye_open"` or `"eye_close"` |
| `confidence` | `float` | Confidence score (0-1) |
| `success` | `bool` | Detection success flag |
| `error_message` | `str | None` | Error message (if failed) |

### create_eye_detector()

```python
create_eye_detector(
    config_path: str,           # 配置文件路径
    checkpoint_path: str,       # 模型权重路径
    device: str = "cuda:0",     # 推理设备
    enable_face_extraction: bool = True,  # 是否启用人脸提取
    face_pad_ratio: float = 0.5,          # 人脸扩展比例
) -> EyeDetectionAPI
```

### create_multi_gpu_detector()

```python
create_multi_gpu_detector(
    config_path: str,
    checkpoint_path: str,
    enable_face_extraction: bool = True,
    face_pad_ratio: float = 0.5,
) -> MultiGPUEyeDetector
```

---

## 视频标注工具 | Video Annotation Tool

### 参数 | Arguments

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i, --input` | 必填 | 输入视频路径 |
| `-o, --output` | 必填 | 输出视频路径 |
| `-c, --config` | `exp/.../config.yaml` | 模型配置文件 |
| `-w, --checkpoint` | `best.pyth` | 模型权重 |
| `--gpu` | False | 使用 GPU 加速 |
| `--device` | `cuda:0` | 推理设备 |
| `--window` | 32 | 检测窗口帧数 |
| `--step` | 16 | 滑窗步长 |

| Argument | Default | Description |
|----------|---------|-------------|
| `-i, --input` | Required | Input video path |
| `-o, --output` | Required | Output video path |
| `-c, --config` | `exp/.../config.yaml` | Model config file |
| `-w, --checkpoint` | `best.pyth` | Model weights |
| `--gpu` | False | Use GPU acceleration |
| `--device` | `cuda:0` | Inference device |
| `--window` | 32 | Detection window frames |
| `--step` | 16 | Sliding window step |

### 工作原理 | How It Works

1. 使用 decord 读取视频，获取帧数、FPS、分辨率
2. 按滑动窗口（默认 32 帧窗口、16 帧步长）切分视频
3. 每个窗口调用检测 API，获取睁/闭眼概率
4. 窗口内所有帧继承该窗口的检测结果
5. 未覆盖的帧使用最近有效帧的结果插值
6. 使用 OpenCV 逐帧写入标注后的视频

1. Read video with decord, get frame count, FPS, resolution
2. Split video by sliding window (default 32-frame window, 16-frame step)
3. Call detection API for each window, get open/closed eye probability
4. All frames in window inherit detection result
5. Uncovered frames interpolated from nearest valid frame
6. Write annotated video frame-by-frame with OpenCV

---

## 性能基准 | Benchmark

```bash
# 多 GPU 性能测试 | Multi-GPU benchmark
python bench_multi_gpu.py \
    --config exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml \
    --checkpoint best.pyth

# 单 GPU 性能测试 | Single GPU benchmark
python bench_gpu.py \
    --config exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml \
    --checkpoint best.pyth \
    --device cuda:0
```

### 参考性能 | Reference Performance

| 配置 | 单次检测 | 视频标注（220帧） |
|------|----------|-------------------|
| 2× GPU (A30) | ~45s/窗口 | ~640s |
| 1× GPU (A30) | ~50s/窗口 | ~700s |

| Setup | Single Detection | Video Annotation (220 frames) |
|-------|------------------|-------------------------------|
| 2× GPU (A30) | ~45s/window | ~640s |
| 1× GPU (A30) | ~50s/window | ~700s |

---

## 模型信息 | Model Info

| 属性 | 值 |
|------|-----|
| 模型名 | UniFormerV2-B/16 |
| 架构 | ViT-B/16 (Vision Transformer) |
| 输入尺寸 | 32 帧 × 224 × 224 |
| 输出 | 二分类（睁眼 / 闭眼） |
| 预训练数据集 | Kinetics-400 + Kinetics-710 + DMS |
| 参数量 | ~86M |
| 模型文件 | `best.pyth`（约 1.3GB） |

| Property | Value |
|----------|-------|
| Model Name | UniFormerV2-B/16 |
| Architecture | ViT-B/16 (Vision Transformer) |
| Input Size | 32 frames × 224 × 224 |
| Output | Binary (open/closed eye) |
| Pretraining | Kinetics-400 + Kinetics-710 + DMS |
| Parameters | ~86M |
| Model File | `best.pyth` (~1.3GB) |

---

## 许可证 | License

MIT License
