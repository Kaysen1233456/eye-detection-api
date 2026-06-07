#!/usr/bin/env python3
"""
评估重新训练的模型

使用新训练的分类器在原始测试集上评估性能。

用法：
    python evaluate_retrained.py
"""

import os
import sys
import time
import re
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import importlib.machinery

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


class SimpleClassifier(nn.Module):
    """简单分类器"""

    def __init__(self, input_dim=2, hidden_dim=16, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


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


def extract_features(detector, video_path: str):
    """
    使用原始模型提取特征

    Args:
        detector: 检测器实例
        video_path: 视频路径

    Returns:
        特征向量 (2,)
    """
    from decord import VideoReader, cpu
    from torchvision import transforms
    from PIL import Image

    # 解析文件名获取时间戳
    filename = Path(video_path).name
    timestamps = parse_video_filename(filename)
    if not timestamps:
        return None

    video_st_time = timestamps['video_st_time']
    video_ed_time = timestamps['video_ed_time']
    event_ed_time_sec = timestamps['event_ed_time']

    # 读取 fps
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
    except Exception as e:
        return None

    # 计算事件时间窗口 (毫秒)
    event_ed_time_ms = event_ed_time_sec * 1000
    frames_back_ms = int(16 / fps * 1000)
    event_st_time_ms = event_ed_time_ms - frames_back_ms

    # 计算帧索引
    video_start_ms = video_st_time * 1000
    start_offset_ms = event_st_time_ms - video_start_ms
    end_offset_ms = event_ed_time_ms - video_start_ms

    start_frame = int(start_offset_ms / 1000 * fps)
    end_frame = int(end_offset_ms / 1000 * fps)

    # 确保帧索引在有效范围内
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))

    if start_frame >= end_frame:
        return None

    # 采样 8 帧
    num_frames = 8
    if (end_frame - start_frame + 1) >= num_frames:
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    else:
        frame_indices = np.random.choice(range(start_frame, end_frame + 1), num_frames, replace=True)

    # 加载帧
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frame_tensors = []
    for idx in frame_indices:
        try:
            frame = vr[idx].asnumpy()
            frame_pil = Image.fromarray(frame)
            frame_tensor = transform(frame_pil)
            frame_tensors.append(frame_tensor)
        except Exception as e:
            frame_tensors.append(torch.zeros(3, 224, 224))

    # 堆叠帧: (T, C, H, W)
    video_tensor = torch.stack(frame_tensors, dim=0)

    # 转换为模型输入格式
    video_tensor = video_tensor.unsqueeze(0)  # 添加 batch 维度: (1, T, C, H, W)
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)

    # 使用模型进行预测
    with torch.no_grad():
        # 包装在列表中，因为模型期望列表输入
        output = detector.model([video_tensor])
        # 应用 softmax 获取概率
        probs = torch.softmax(output, dim=-1)

    return probs.squeeze(0).numpy()


def main():
    import pandas as pd

    print('=' * 80)
    print('评估重新训练的模型')
    print('=' * 80)
    print()

    # Step 1: 加载模型
    print('Step 1: 加载模型...')
    api_module = load_original_api()

    detector = api_module.create_eye_detector(
        config_path='exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml',
        checkpoint_path='best.pyth',
        device='cpu',
        enable_face_extraction=False,
        face_pad_ratio=0.5,
    )
    print('  原始模型加载成功')

    # 加载新分类器
    classifier = SimpleClassifier(input_dim=2, hidden_dim=16, num_classes=2)
    checkpoint = torch.load('/yanglin/eye_detect/retrain_output/best_classifier.pth', map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    print('  新分类器加载成功')
    print()

    # Step 2: 加载测试数据
    print('Step 2: 加载测试数据...')
    df = pd.read_csv('/yanglin/eye_detect/二次核查_最终标注.csv')

    # 只评估有明确标签的案例
    eval_df = df[df['最终标注'].isin(['闭眼', '睁眼', '睁眼(下瞟/眨眼)'])].copy()
    print(f'  测试样本数: {len(eval_df)}')
    print(f'    闭眼: {len(eval_df[eval_df["最终标注"] == "闭眼"])}')
    print(f'    睁眼: {len(eval_df[eval_df["最终标注"].isin(["睁眼", "睁眼(下瞟/眨眼)"])])}')
    print()

    # Step 3: 提取特征并预测
    print('Step 3: 提取特征并预测...')
    print('-' * 80)

    predictions = []
    for i, (_, row) in enumerate(eval_df.iterrows(), 1):
        video_path = row['视频路径']
        true_label = 1 if row['最终标注'] == '闭眼' else 0

        print(f'  [{i}/{len(eval_df)}] {row["车牌号"]} {row["视频日期"]} ... ', end='', flush=True)

        # 提取特征
        features = extract_features(detector, video_path)

        if features is not None:
            # 使用新分类器预测
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                output = classifier(features_tensor)
                probs = torch.softmax(output, dim=-1)
                pred_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_label].item()

            correct = pred_label == true_label
            status = '✓' if correct else '✗'

            predictions.append({
                'video_id': f"{row['车牌号']}_{row['视频日期']}_{row['视频时间'][:5]}",
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence,
                'correct': correct,
            })

            pred_name = '闭眼' if pred_label == 1 else '睁眼'
            print(f'{pred_name} ({confidence:.4f}) {status}')
        else:
            print('特征提取失败')

    print()

    # Step 4: 分析结果
    print('Step 4: 分析结果...')
    print('-' * 80)

    if len(predictions) == 0:
        print('  没有有效的预测结果')
        return

    # 转换为数组
    true_labels = np.array([p['true_label'] for p in predictions])
    pred_labels = np.array([p['pred_label'] for p in predictions])
    confidences = np.array([p['confidence'] for p in predictions])

    # 计算整体准确率
    accuracy = np.mean(pred_labels == true_labels) * 100
    print(f'  整体准确率: {accuracy:.1f}%')

    # 计算各类别准确率
    for label_id, label_name in [(1, '闭眼'), (0, '睁眼')]:
        mask = true_labels == label_id
        if mask.sum() > 0:
            label_acc = np.mean(pred_labels[mask] == true_labels[mask]) * 100
            label_conf = confidences[mask].mean()
            print(f'  {label_name}: 准确率 {label_acc:.1f}%, 平均置信度 {label_conf:.4f}')

    # 计算混淆矩阵
    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))

    print()
    print('  混淆矩阵:')
    print(f'    真正例 (TP): {tp}')
    print(f'    假正例 (FP): {fp}')
    print(f'    假负例 (FN): {fn}')
    print(f'    真负例 (TN): {tn}')

    # 计算精确率、召回率、F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print()
    print('  性能指标:')
    print(f'    精确率: {precision:.4f}')
    print(f'    召回率: {recall:.4f}')
    print(f'    F1 分数: {f1:.4f}')
    print()

    # 保存结果
    output_path = '/yanglin/eye_detect/retrain_output/evaluation_results.json'
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        },
        'predictions': predictions,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'  结果已保存: {output_path}')
    print()

    # 总结
    print('=' * 80)
    print('评估完成！')
    print('=' * 80)
    print()
    print('【性能对比】')
    print(f'  原始模型闭眼召回率: 26.1%')
    print(f'  新模型闭眼召回率: {recall * 100:.1f}%')
    print(f'  提升: {recall * 100 - 26.1:.1f}%')
    print()
    print('【下一步】')
    print('  1. 如果性能满意，更新 API 使用新模型')
    print('  2. 重新运行批量检测验证效果')


if __name__ == '__main__':
    main()
