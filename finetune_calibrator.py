#!/usr/bin/env python3
"""
校准器微调脚本

使用原始模型的 detect 方法获取预测结果，
然后训练一个置信度校准器来改善闭眼检测。

用法：
    python finetune_calibrator.py
"""

import os
import sys
import time
import re
import numpy as np
from pathlib import Path

# 禁用 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.chdir('/yanglin/eye_detect')
sys.path.insert(0, '/yanglin/eye_detect')

# 修复 torchvision 兼容性问题
import types
import torchvision.transforms.functional as _tvF
_functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
_functional_tensor.rgb_to_grayscale = _tvF.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

import importlib.machinery


def load_original_api():
    """动态加载 eye_detect_api.1py 原始代码"""
    loader = importlib.machinery.SourceFileLoader(
        "eye_detect_api_original",
        "/yanglin/eye_detect/eye_detect_api.1py"
    )
    spec = importlib.util.spec_from_loader("eye_detect_api_original", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def parse_video_filename(filename: str) -> dict | None:
    """从视频文件名中提取时间戳"""
    pattern = r'_(\d{10})_(\d{10})_(\d{10})__'
    match = re.search(pattern, filename)
    if not match:
        return None

    return {
        'video_st_time': int(match.group(1)),
        'video_ed_time': int(match.group(2)),
        'event_ed_time': int(match.group(3)),
    }


def run_detection(detector, video_path: str) -> dict:
    """对单个视频执行闭眼检测"""
    from decord import VideoReader, cpu

    filename = Path(video_path).name
    timestamps = parse_video_filename(filename)
    if not timestamps:
        return {'success': False, 'error': '无法解析文件名'}

    video_st_time = timestamps['video_st_time']
    video_ed_time = timestamps['video_ed_time']
    event_ed_time_sec = timestamps['event_ed_time']

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
    except Exception as e:
        return {'success': False, 'error': f'读取fps失败: {e}'}

    event_ed_time_ms = event_ed_time_sec * 1000
    frames_back_ms = int(16 / fps * 1000)
    event_st_time_ms = event_ed_time_ms - frames_back_ms

    api_module = load_original_api()
    input_params = api_module.EyeDetectionInput(
        video_path=video_path,
        video_st_time=video_st_time,
        video_ed_time=video_ed_time,
        event_st_time=event_st_time_ms,
        event_ed_time=event_ed_time_ms,
    )

    result = detector.detect(input_params)

    return {
        'success': result.success,
        'result': result.result,
        'confidence': result.confidence,
        'error': result.error_message if not result.success else None,
    }


def main():
    import pandas as pd

    print('=' * 80)
    print('校准器微调 - 基于原始模型输出')
    print('=' * 80)
    print()

    # Step 1: 加载模型
    print('Step 1: 加载预训练模型...')
    api_module = load_original_api()

    detector = api_module.create_eye_detector(
        config_path='exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml',
        checkpoint_path='best.pyth',
        device='cpu',
        enable_face_extraction=True,
        face_pad_ratio=0.5,
    )
    print('  模型加载成功')
    print()

    # Step 2: 加载训练数据
    print('Step 2: 加载训练数据...')
    training_dir = Path('/yanglin/eye_detect/training_data')

    samples = []
    for label_name, label_id in [('close', 1), ('open', 0)]:
        label_dir = training_dir / label_name
        if label_dir.exists():
            for video_dir in label_dir.iterdir():
                if video_dir.is_dir():
                    # 从 video_id 还原视频路径
                    # video_id 格式: 鲁EG7062_2026-05-19_04:50
                    parts = video_dir.name.split('_')
                    if len(parts) >= 3:
                        plate = parts[0]
                        date = parts[1]
                        time_part = parts[2]

                        # 在 close 目录下查找匹配的视频
                        close_dir = Path('/yanglin/eye_detect/close')
                        for sub_dir in close_dir.iterdir():
                            if not sub_dir.is_dir():
                                continue
                            if plate in sub_dir.name and date.replace('-', '_') in sub_dir.name:
                                for video_file in sub_dir.glob('*_channel7.mp4'):
                                    samples.append({
                                        'video_path': str(video_file),
                                        'label': label_id,
                                        'label_name': label_name,
                                        'video_id': video_dir.name,
                                    })

    print(f'  找到 {len(samples)} 个训练样本')
    print(f'    闭眼: {sum(1 for s in samples if s["label"] == 1)}')
    print(f'    睁眼: {sum(1 for s in samples if s["label"] == 0)}')
    print()

    # Step 3: 提取模型预测
    print('Step 3: 提取模型预测...')
    print('-' * 80)

    predictions = []
    for i, sample in enumerate(samples, 1):
        video_path = sample['video_path']
        label = sample['label']

        print(f'  [{i}/{len(samples)}] {sample["video_id"]} ... ', end='', flush=True)

        result = run_detection(detector, video_path)

        if result['success']:
            pred_label = 1 if result['result'] == 'eye_close' else 0
            confidence = result['confidence']
            correct = pred_label == label

            predictions.append({
                'video_id': sample['video_id'],
                'true_label': label,
                'pred_label': pred_label,
                'confidence': confidence,
                'correct': correct,
            })

            status = '✓' if correct else '✗'
            print(f'{result["result"]} ({confidence:.4f}) {status}')
        else:
            print(f'失败: {result["error"][:50]}')

    print()

    # Step 4: 分析预测结果
    print('Step 4: 分析预测结果...')
    print('-' * 80)

    if len(predictions) == 0:
        print('  没有有效的预测结果')
        return

    # 转换为数组
    true_labels = np.array([p['true_label'] for p in predictions])
    pred_labels = np.array([p['pred_label'] for p in predictions])
    confidences = np.array([p['confidence'] for p in predictions])

    # 计算准确率
    accuracy = np.mean(pred_labels == true_labels) * 100
    print(f'  整体准确率: {accuracy:.1f}%')

    # 分类别统计
    for label_id, label_name in [(1, '闭眼'), (0, '睁眼')]:
        mask = true_labels == label_id
        if mask.sum() > 0:
            label_acc = np.mean(pred_labels[mask] == true_labels[mask]) * 100
            label_conf = confidences[mask].mean()
            print(f'  {label_name}: 准确率 {label_acc:.1f}%, 平均置信度 {label_conf:.4f}')

    print()

    # Step 5: 计算校准阈值
    print('Step 5: 计算最优校准阈值...')
    print('-' * 80)

    # 寻找最优阈值
    # 对于闭眼检测，我们希望降低漏检率
    # 可以通过调整阈值来改善

    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.3, 0.8, 0.05):
        # 使用阈值调整预测
        adjusted_preds = np.where(
            (pred_labels == 1) | (confidences < threshold),
            1,  # 如果原预测是闭眼 或 置信度低于阈值，判为闭眼
            0
        )

        # 计算 F1 分数
        tp = np.sum((adjusted_preds == 1) & (true_labels == 1))
        fp = np.sum((adjusted_preds == 1) & (true_labels == 0))
        fn = np.sum((adjusted_preds == 0) & (true_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f'  最优阈值: {best_threshold:.2f}')
    print(f'  最优 F1: {best_f1:.4f}')
    print()

    # 使用最优阈值的性能
    adjusted_preds = np.where(
        (pred_labels == 1) | (confidences < best_threshold),
        1,
        0
    )

    tp = np.sum((adjusted_preds == 1) & (true_labels == 1))
    fp = np.sum((adjusted_preds == 1) & (true_labels == 0))
    fn = np.sum((adjusted_preds == 0) & (true_labels == 1))
    tn = np.sum((adjusted_preds == 0) & (true_labels == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy_adjusted = (tp + tn) / len(true_labels) * 100

    print('  校准后性能:')
    print(f'    准确率: {accuracy_adjusted:.1f}%')
    print(f'    精确率: {precision:.4f}')
    print(f'    召回率: {recall:.4f}')
    print(f'    F1 分数: {f1:.4f}')
    print(f'    真正例 (TP): {tp}')
    print(f'    假正例 (FP): {fp}')
    print(f'    假负例 (FN): {fn}')
    print(f'    真负例 (TN): {tn}')
    print()

    # Step 6: 保存校准参数
    print('Step 6: 保存校准参数...')
    output_dir = Path('/yanglin/eye_detect/finetuned_model')
    output_dir.mkdir(exist_ok=True)

    import json
    calibration_config = {
        'threshold': float(best_threshold),
        'f1_score': float(best_f1),
        'accuracy': float(accuracy_adjusted),
        'precision': float(precision),
        'recall': float(recall),
        'training_samples': len(samples),
        'close_samples': int(np.sum(true_labels == 1)),
        'open_samples': int(np.sum(true_labels == 0)),
    }

    config_path = output_dir / 'calibration_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(calibration_config, f, indent=2, ensure_ascii=False)

    print(f'  校准参数已保存: {config_path}')
    print()

    # 总结
    print('=' * 80)
    print('校准器微调完成！')
    print('=' * 80)
    print()
    print('【改进效果】')
    print(f'  原始准确率: {accuracy:.1f}%')
    print(f'  校准后准确率: {accuracy_adjusted:.1f}%')
    print(f'  改进: {accuracy_adjusted - accuracy:.1f}%')
    print()
    print('【校准规则】')
    print(f'  当模型预测为闭眼 或 置信度 < {best_threshold:.2f} 时，判定为闭眼')
    print()
    print('【下一步】')
    print('  1. 使用校准后的阈值重新运行批量检测')
    print('  2. 对比校准前后的性能')


if __name__ == '__main__':
    main()
