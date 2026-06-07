#!/usr/bin/env python3
"""
准备训练 JSON 标注文件

将增强后的训练数据转换为 SlowFast 框架所需的 JSON 格式。

用法：
    python prepare_training_json.py
"""

import os
import sys
import json
import random
from pathlib import Path

os.chdir('/yanglin/eye_detect')
sys.path.insert(0, '/yanglin/eye_detect')


def scan_training_data(data_dir: str, label: int) -> list:
    """
    扫描训练数据目录，生成标注列表

    Args:
        data_dir: 数据目录（close 或 open）
        label: 标签（1=闭眼, 0=睁眼）

    Returns:
        标注列表
    """
    annotations = []
    data_path = Path(data_dir)

    if not data_path.exists():
        return annotations

    for video_dir in sorted(data_path.iterdir()):
        if not video_dir.is_dir():
            continue

        # 统计帧数
        frames = sorted(video_dir.glob('*.jpg'))
        if len(frames) == 0:
            continue

        # 生成帧路径模板
        # 格式: /path/to/video_dir/frame_%03d.jpg
        frame_template = str(video_dir / 'frame_%03d.jpg')

        annotations.append({
            'video_path': frame_template,
            'train_label': label,
            'start_frame': 0,
            'end_frame': len(frames) - 1,
            'video_frame_count': len(frames),
            'video_id': video_dir.name,
        })

    return annotations


def split_dataset(annotations: list, val_ratio: float = 0.2, seed: int = 42) -> tuple:
    """
    划分训练集和验证集

    Args:
        annotations: 标注列表
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        (train_annotations, val_annotations)
    """
    random.seed(seed)
    random.shuffle(annotations)

    val_size = int(len(annotations) * val_ratio)
    val_annotations = annotations[:val_size]
    train_annotations = annotations[val_size:]

    return train_annotations, val_annotations


def main():
    augmented_dir = '/yanglin/eye_detect/training_data_augmented'
    output_dir = '/yanglin/eye_detect/training_json'

    print('=' * 80)
    print('准备训练 JSON 标注文件')
    print('=' * 80)
    print()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 扫描闭眼数据
    print('Step 1: 扫描训练数据...')
    close_annotations = scan_training_data(
        os.path.join(augmented_dir, 'close'),
        label=1
    )
    print(f'  闭眼样本: {len(close_annotations)}')

    open_annotations = scan_training_data(
        os.path.join(augmented_dir, 'open'),
        label=0
    )
    print(f'  睁眼样本: {len(open_annotations)}')

    all_annotations = close_annotations + open_annotations
    print(f'  总计: {len(all_annotations)}')
    print()

    # Step 2: 划分训练集和验证集
    print('Step 2: 划分训练集和验证集...')

    # 分别划分闭眼和睁眼，保持比例
    close_train, close_val = split_dataset(close_annotations, val_ratio=0.2)
    open_train, open_val = split_dataset(open_annotations, val_ratio=0.2)

    train_annotations = close_train + open_train
    val_annotations = close_val + open_val

    # 打乱顺序
    random.shuffle(train_annotations)
    random.shuffle(val_annotations)

    print(f'  训练集: {len(train_annotations)} (闭眼: {len(close_train)}, 睁眼: {len(open_train)})')
    print(f'  验证集: {len(val_annotations)} (闭眼: {len(close_val)}, 睁眼: {len(open_val)})')
    print()

    # Step 3: 保存 JSON 文件
    print('Step 3: 保存 JSON 文件...')

    train_json_path = os.path.join(output_dir, 'train.json')
    val_json_path = os.path.join(output_dir, 'val.json')

    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2, ensure_ascii=False)

    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2, ensure_ascii=False)

    print(f'  训练集 JSON: {train_json_path}')
    print(f'  验证集 JSON: {val_json_path}')
    print()

    # Step 4: 统计信息
    print('=' * 80)
    print('JSON 标注文件准备完成！')
    print('=' * 80)
    print()
    print(f'【训练集】')
    print(f'  样本数: {len(train_annotations)}')
    print(f'  闭眼: {len(close_train)}')
    print(f'  睁眼: {len(open_train)}')
    print()
    print(f'【验证集】')
    print(f'  样本数: {len(val_annotations)}')
    print(f'  闭眼: {len(close_val)}')
    print(f'  睁眼: {len(open_val)}')
    print()
    print(f'【输出文件】')
    print(f'  {train_json_path}')
    print(f'  {val_json_path}')


if __name__ == '__main__':
    main()
