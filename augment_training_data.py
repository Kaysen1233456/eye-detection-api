#!/usr/bin/env python3
"""
数据增强脚本 - 扩充闭眼训练数据

对 training_data/close/ 中的闭眼视频帧进行数据增强：
1. 水平翻转
2. 随机旋转（-15° 到 +15°）
3. 亮度调整（±20%）
4. 对比度调整（±20%）
5. 饱和度调整（±20%）

每个原始视频生成 4 个增强版本，总共扩充 5 倍。

用法：
    python augment_training_data.py
"""

import os
import sys
import random
import numpy as np
import cv2
from pathlib import Path
import shutil

os.chdir('/yanglin/eye_detect')
sys.path.insert(0, '/yanglin/eye_detect')


def augment_frame(frame: np.ndarray, augment_type: str) -> np.ndarray:
    """
    对单帧图像进行数据增强

    Args:
        frame: 输入图像 (BGR 格式)
        augment_type: 增强类型

    Returns:
        增强后的图像
    """
    h, w = frame.shape[:2]

    if augment_type == 'flip':
        # 水平翻转
        return cv2.flip(frame, 1)

    elif augment_type == 'rotate':
        # 随机旋转 -15° 到 +15°
        angle = random.uniform(-15, 15)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    elif augment_type == 'bright':
        # 亮度调整 ±20%
        factor = random.uniform(0.8, 1.2)
        # 转换到 HSV 空间调整亮度
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif augment_type == 'contrast':
        # 对比度调整 ±20%
        factor = random.uniform(0.8, 1.2)
        mean = frame.mean()
        adjusted = (frame - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    elif augment_type == 'saturation':
        # 饱和度调整 ±20%
        factor = random.uniform(0.8, 1.2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif augment_type == 'noise':
        # 高斯噪声
        noise = np.random.normal(0, 5, frame.shape).astype(np.float32)
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    elif augment_type == 'combined':
        # 组合增强：随机应用 2-3 种增强
        augmentations = ['flip', 'rotate', 'bright', 'contrast']
        num_augs = random.randint(2, 3)
        selected = random.sample(augmentations, num_augs)

        result = frame.copy()
        for aug in selected:
            result = augment_frame(result, aug)
        return result

    return frame


def augment_video_frames(
    input_dir: str,
    output_dir: str,
    video_id: str,
    augment_type: str
):
    """
    对一个视频的所有帧进行数据增强

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        video_id: 视频 ID
        augment_type: 增强类型
    """
    input_path = Path(input_dir) / video_id
    output_path = Path(output_dir) / f"{video_id}_{augment_type}"
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取所有帧
    frame_files = sorted(input_path.glob('*.jpg'))

    for frame_file in frame_files:
        # 读取帧
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue

        # 应用增强
        augmented = augment_frame(frame, augment_type)

        # 保存
        output_file = output_path / frame_file.name
        cv2.imwrite(str(output_file), augmented)

    return len(frame_files)


def main():
    input_dir = '/yanglin/eye_detect/training_data/close'
    output_dir = '/yanglin/eye_detect/training_data_augmented/close'

    # 增强类型列表
    augment_types = ['flip', 'rotate', 'bright', 'contrast']

    print('=' * 80)
    print('数据增强 - 扩充闭眼训练数据')
    print('=' * 80)
    print()

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有原始视频
    input_path = Path(input_dir)
    video_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

    print(f'找到 {len(video_dirs)} 个原始闭眼视频')
    print()

    # 复制原始数据
    print('Step 1: 复制原始数据...')
    for video_dir in video_dirs:
        dest_dir = Path(output_dir) / video_dir.name
        if not dest_dir.exists():
            shutil.copytree(video_dir, dest_dir)
    print(f'  已复制 {len(video_dirs)} 个视频')
    print()

    # 应用增强
    print('Step 2: 应用数据增强...')
    print('-' * 80)

    total_augmented = 0
    for i, video_dir in enumerate(video_dirs, 1):
        video_id = video_dir.name
        print(f'  [{i}/{len(video_dirs)}] {video_id}')

        for aug_type in augment_types:
            count = augment_video_frames(
                input_dir=input_dir,
                output_dir=output_dir,
                video_id=video_id,
                augment_type=aug_type,
            )
            total_augmented += count
            print(f'    {aug_type}: {count} 帧')

    print()
    print(f'增强完成！新增 {total_augmented} 帧')
    print()

    # 统计
    print('Step 3: 统计增强后数据...')
    print('-' * 80)

    output_path = Path(output_dir)
    all_dirs = sorted([d for d in output_path.iterdir() if d.is_dir()])

    total_frames = 0
    for d in all_dirs:
        frames = list(d.glob('*.jpg'))
        total_frames += len(frames)

    print(f'  增强后视频数: {len(all_dirs)}')
    print(f'  增强后总帧数: {total_frames}')
    print(f'  扩充倍数: {len(all_dirs) / len(video_dirs):.1f}x')
    print()

    # 同时增强睁眼数据（轻微增强）
    print('Step 4: 增强睁眼数据（轻微）...')
    print('-' * 80)

    open_input_dir = '/yanglin/eye_detect/training_data/open'
    open_output_dir = '/yanglin/eye_detect/training_data_augmented/open'
    os.makedirs(open_output_dir, exist_ok=True)

    # 复制原始睁眼数据
    open_input_path = Path(open_input_dir)
    open_video_dirs = sorted([d for d in open_input_path.iterdir() if d.is_dir()])

    for video_dir in open_video_dirs:
        dest_dir = Path(open_output_dir) / video_dir.name
        if not dest_dir.exists():
            shutil.copytree(video_dir, dest_dir)

    # 对睁眼数据只做轻微增强（翻转）
    for i, video_dir in enumerate(open_video_dirs, 1):
        video_id = video_dir.name
        print(f'  [{i}/{len(open_video_dirs)}] {video_id}')

        # 只做翻转增强
        count = augment_video_frames(
            input_dir=open_input_dir,
            output_dir=open_output_dir,
            video_id=video_id,
            augment_type='flip',
        )
        print(f'    flip: {count} 帧')

    print()

    # 最终统计
    print('=' * 80)
    print('数据增强完成！')
    print('=' * 80)
    print()

    # 统计闭眼数据
    close_output_path = Path(output_dir)
    close_dirs = sorted([d for d in close_output_path.iterdir() if d.is_dir()])
    close_frames = sum(len(list(d.glob('*.jpg'))) for d in close_dirs)

    # 统计睁眼数据
    open_output_path = Path(open_output_dir)
    open_dirs = sorted([d for d in open_output_path.iterdir() if d.is_dir()])
    open_frames = sum(len(list(d.glob('*.jpg'))) for d in open_dirs)

    print(f'【闭眼数据】')
    print(f'  原始: {len(video_dirs)} 个视频, {len(video_dirs) * 17} 帧')
    print(f'  增强后: {len(close_dirs)} 个视频, {close_frames} 帧')
    print(f'  扩充倍数: {len(close_dirs) / len(video_dirs):.1f}x')
    print()
    print(f'【睁眼数据】')
    print(f'  原始: {len(open_video_dirs)} 个视频, {len(open_video_dirs) * 17} 帧')
    print(f'  增强后: {len(open_dirs)} 个视频, {open_frames} 帧')
    print(f'  扩充倍数: {len(open_dirs) / len(open_video_dirs):.1f}x')
    print()
    print(f'【总计】')
    print(f'  视频数: {len(close_dirs) + len(open_dirs)}')
    print(f'  帧数: {close_frames + open_frames}')
    print()
    print(f'输出目录: /yanglin/eye_detect/training_data_augmented/')


if __name__ == '__main__':
    main()
