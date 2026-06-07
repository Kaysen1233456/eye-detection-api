# Eye Detect

基于 UniFormerV2 / SlowFast 视频理解框架的驾驶员闭眼检测项目。项目围绕车载视频中的高风险疲劳驾驶片段，提供单视频检测、批量检测、双 GPU 并行检测、二次核查、训练数据抽取、数据增强、模型微调和重训练等功能。

## 核心功能

- **闭眼检测 API**：`eye_detect_api.py` 提供 `EyeDetectionAPI`、`EyeDetectionInput`、`EyeDetectionOutput`，支持输入视频路径、视频时间范围和事件时间范围，输出 `eye_close` / `eye_open`、置信度和错误信息。
- **人脸区域提取**：可选集成 DeepFace，对视频帧做人脸检测和区域扩展，减少背景对闭眼判断的干扰。
- **视频帧采样与模型推理**：基于 decord 读取视频，根据事件时间窗口抽帧，并调用 UniFormerV2 / SlowFast 模型进行分类推理。
- **批量检测**：`batch_eye_detect.py` 扫描 `close/` 目录中的 `channel7` 视频，解析文件名中的时间戳并批量生成检测结果。
- **双 GPU 并行检测**：`batch_eye_detect_dual_gpu.py` 使用多进程拆分视频任务，支持双 GPU 并行加速。
- **高风险二次核查**：`二次核查_高风险检测.py` 读取高风险 Excel 标注，匹配本地视频并生成模型检测与人工标注的对照结果。
- **训练数据构建**：`extract_training_data.py`、`prepare_training_json.py` 从核查样本中抽取训练片段并生成训练 JSON。
- **数据增强与再训练**：`augment_training_data.py`、`retrain_model.py`、`retrain_simple.py`、`finetune_model.py`、`finetune_simple.py` 支持训练数据增强、分类器重训练和模型微调。
- **评估与校准**：`evaluate_retrained.py`、`finetune_calibrator.py` 用于评估重训练模型并校准输出置信度。

## 主要文件

```text
eye_detect_api.py                # 闭眼检测核心 API
batch_eye_detect.py              # 单 GPU 批量检测
batch_eye_detect_dual_gpu.py     # 双 GPU 并行批量检测
二次核查_高风险检测.py             # 高风险视频二次核查
extract_training_data.py         # 从检测/核查结果抽取训练数据
prepare_training_json.py         # 生成训练 JSON
augment_training_data.py         # 训练数据增强
retrain_model.py                 # 重训练入口
retrain_simple.py                # 简化重训练脚本
finetune_model.py                # 微调入口
finetune_simple.py               # 简化微调脚本
evaluate_retrained.py            # 重训练模型评估
finetune_calibrator.py           # 置信度校准
slowfast/                        # SlowFast / UniFormerV2 模型代码
tools/                           # 训练、测试、可视化工具
exp/humanfactor/                 # 人因闭眼检测相关配置
```

## 不提交到仓库的内容

以下内容属于数据、模型权重或运行产物，不应直接上传到 GitHub：

- `close/` 视频样本目录
- `training_data/`、`training_data_augmented/`
- `best.pyth`、`*.pth`、`*.pyth`
- `core.*` 崩溃转储
- `retrain_output/`、`finetuned_model/`
- 批量检测结果 CSV、Excel 标注文件和测试视频

## 运行示例

单视频 API 使用：

```python
from eye_detect_api import EyeDetectionInput, create_eye_detector

detector = create_eye_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    device="cuda:0",
)

result = detector.detect(EyeDetectionInput(
    video_path="example_channel7.mp4",
    video_st_time=1778603614,
    video_ed_time=1778603621,
    event_st_time=1778603615000,
    event_ed_time=1778603617000,
))

print(result)
```

批量检测：

```bash
python batch_eye_detect.py
```

双 GPU 并行检测：

```bash
python batch_eye_detect_dual_gpu.py
```

## 环境依赖

项目依赖 PyTorch、decord、OpenCV、pandas、DeepFace、SlowFast / UniFormerV2 相关组件。基础依赖可参考 `setup.py`。

模型权重和视频数据需要单独准备，不随代码仓库提交。
