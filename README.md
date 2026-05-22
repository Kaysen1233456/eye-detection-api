# Eye Detection API

基于 UniFormerV2 的闭眼检测 API，支持单 GPU / 多 GPU 推理，并提供视频标注工具。

---

## 功能特性

- **闭眼检测 API**：输入视频片段，返回睁/闭眼状态及置信度
- **多 GPU 并行推理**：自动检测 GPU 数量，负载均衡分配推理任务
- **视频标注工具**：逐帧叠加闭眼概率和状态，输出标注后的视频
- **人脸提取**：可选的人脸裁剪功能，提升检测精度

---

## 项目结构

```
eye_detect_api/
├── eye_detect_api.py       # 核心检测 API（单 GPU / 多 GPU）
├── annotate_video.py        # 视频标注工具
├── bench_multi_gpu.py       # 多 GPU 性能基准测试
├── bench_gpu.py             # 单 GPU 性能基准测试
├── setup.py                 # 安装配置
├── best.pyth                # 模型权重（UniFormerV2-B/16）
├── exp/                     # 模型配置文件
├── slowfast/                # 模型框架代码
├── extract_clip/            # CLIP 特征提取工具
└── tools/                   # 训练/测试脚本
```

---

## 安装

```bash
pip install -e .
```

依赖：PyTorch, torchvision, decord, opencv-python, timm

---

## 快速开始

### 1. 单 GPU 检测

```python
from eye_detect_api import EyeDetectionInput, create_eye_detector

detector = create_eye_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    device="cuda:0",
    enable_face_extraction=True,
    face_pad_ratio=0.5,
)

inp = EyeDetectionInput(
    video_path="test.mp4",
    video_st_time=1700000000,
    video_ed_time=1700000011,
    event_st_time=1700000000000,
    event_ed_time=1700000001000,
)
result = detector.detect(inp)
print(f"状态: {result.result}, 置信度: {result.confidence:.3f}")
```

### 2. 多 GPU 检测

```python
from eye_detect_api import create_multi_gpu_detector

detector = create_multi_gpu_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    enable_face_extraction=True,
    face_pad_ratio=0.5,
)
result = detector.detect(inp)
```

### 3. 视频标注

```bash
# 单 GPU
python annotate_video.py -i input.mp4 -o output.mp4 --gpu

# 指定参数
python annotate_video.py -i input.mp4 -o output.mp4 --gpu --window 32 --step 16
```

输出视频左上角会叠加：
- 状态：OPEN（绿色）/ CLOSED（红色）
- 闭眼概率条
- 百分比数值
- 推理耗时

---

## API 接口

### EyeDetectionInput

| 参数 | 类型 | 说明 |
|------|------|------|
| `video_path` | str | 视频文件路径 |
| `video_st_time` | int | 视频起始时间（Unix 时间戳，秒） |
| `video_ed_time` | int | 视频结束时间（Unix 时间戳，秒） |
| `event_st_time` | int | 事件起始时间（毫秒） |
| `event_ed_time` | int | 事件结束时间（毫秒） |

### EyeDetectionOutput

| 字段 | 类型 | 说明 |
|------|------|------|
| `result` | str | `"eye_open"` 或 `"eye_close"` |
| `confidence` | float | 置信度（0-1） |
| `success` | bool | 是否检测成功 |
| `error_message` | str | 错误信息（失败时） |

---

## Benchmark

```bash
# 多 GPU 性能测试
python bench_multi_gpu.py --config exp/.../config.yaml --checkpoint best.pyth

# 单 GPU 性能测试
python bench_gpu.py --config exp/.../config.yaml --checkpoint best.pyth --device cuda:0
```

---

## 模型

- **Backbone**: UniFormerV2-B/16（ViT-B/16）
- **输入**: 32 帧视频片段，224x224 分辨率
- **输出**: 二分类（睁眼 / 闭眼）
- **预训练**: K400 + K710 + DMS 数据集

---

# Eye Detection API

UniFormerV2-based eye state detection API with single/multi-GPU inference and video annotation tool.

---

## Features

- **Eye Detection API**: Input video segments, return open/closed eye state with confidence
- **Multi-GPU Inference**: Auto-detect GPU count, balanced load distribution
- **Video Annotation Tool**: Overlay real-time eye-closed probability on each frame
- **Face Extraction**: Optional face cropping for improved detection accuracy

---

## Project Structure

```
eye_detect_api/
├── eye_detect_api.py       # Core detection API (single/multi GPU)
├── annotate_video.py        # Video annotation tool
├── bench_multi_gpu.py       # Multi-GPU benchmark
├── bench_gpu.py             # Single GPU benchmark
├── setup.py                 # Package setup
├── best.pyth                # Model weights (UniFormerV2-B/16)
├── exp/                     # Model config files
├── slowfast/                # Model framework
├── extract_clip/            # CLIP feature extraction
└── tools/                   # Train/test scripts
```

---

## Installation

```bash
pip install -e .
```

Dependencies: PyTorch, torchvision, decord, opencv-python, timm

---

## Quick Start

### 1. Single GPU Detection

```python
from eye_detect_api import EyeDetectionInput, create_eye_detector

detector = create_eye_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    device="cuda:0",
    enable_face_extraction=True,
    face_pad_ratio=0.5,
)

inp = EyeDetectionInput(
    video_path="test.mp4",
    video_st_time=1700000000,
    video_ed_time=1700000011,
    event_st_time=1700000000000,
    event_ed_time=1700000001000,
)
result = detector.detect(inp)
print(f"Status: {result.result}, Confidence: {result.confidence:.3f}")
```

### 2. Multi-GPU Detection

```python
from eye_detect_api import create_multi_gpu_detector

detector = create_multi_gpu_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    enable_face_extraction=True,
    face_pad_ratio=0.5,
)
result = detector.detect(inp)
```

### 3. Video Annotation

```bash
# Single GPU
python annotate_video.py -i input.mp4 -o output.mp4 --gpu

# Custom parameters
python annotate_video.py -i input.mp4 -o output.mp4 --gpu --window 32 --step 16
```

Output video overlay (top-left corner):
- Status: OPEN (green) / CLOSED (red)
- Eye-closed probability bar
- Percentage values
- Inference time

---

## API Reference

### EyeDetectionInput

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_path` | str | Video file path |
| `video_st_time` | int | Video start time (Unix timestamp, seconds) |
| `video_ed_time` | int | Video end time (Unix timestamp, seconds) |
| `event_st_time` | int | Event start time (milliseconds) |
| `event_ed_time` | int | Event end time (milliseconds) |

### EyeDetectionOutput

| Field | Type | Description |
|-------|------|-------------|
| `result` | str | `"eye_open"` or `"eye_close"` |
| `confidence` | float | Confidence score (0-1) |
| `success` | bool | Detection success flag |
| `error_message` | str | Error message (if failed) |

---

## Benchmark

```bash
# Multi-GPU benchmark
python bench_multi_gpu.py --config exp/.../config.yaml --checkpoint best.pyth

# Single GPU benchmark
python bench_gpu.py --config exp/.../config.yaml --checkpoint best.pyth --device cuda:0
```

---

## Model

- **Backbone**: UniFormerV2-B/16 (ViT-B/16)
- **Input**: 32-frame video clip, 224x224 resolution
- **Output**: Binary classification (open/closed eye)
- **Pretraining**: K400 + K710 + DMS datasets
