#!/usr/bin/env python3
"""
提取闭眼帧作为微调训练数据

从 11 个确认的闭眼案例中提取事件结束前 16 帧，
并创建平衡的闭眼/睁眼训练数据集。

用法：
    python extract_training_data.py
"""

import os
import sys
import re
import cv2
import numpy as np
from pathlib import Path
from decord import VideoReader, cpu

os.chdir('/yanglin/eye_detect')
sys.path.insert(0, '/yanglin/eye_detect')


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


def extract_frames_from_video(video_path: str, output_dir: str, label: str, video_id: str):
    """
    从视频中提取事件结束前 16 帧

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        label: 标签 (close/open)
        video_id: 视频 ID
    """
    # 解析时间戳
    filename = Path(video_path).name
    timestamps = parse_video_filename(filename)
    if not timestamps:
        print(f'  [跳过] 无法解析文件名: {filename}')
        return 0

    video_st_time = timestamps['video_st_time']
    video_ed_time = timestamps['video_ed_time']
    event_ed_time_sec = timestamps['event_ed_time']

    # 读取视频
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)
    except Exception as e:
        print(f'  [错误] 读取视频失败: {e}')
        return 0

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
        print(f'  [跳过] 无效的帧范围: {start_frame}-{end_frame}')
        return 0

    # 提取帧
    frame_indices = list(range(start_frame, end_frame + 1))
    frames = vr.get_batch(frame_indices).asnumpy()

    # 保存帧
    output_path = Path(output_dir) / label / video_id
    output_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for i, frame in enumerate(frames):
        frame_path = output_path / f'frame_{i:03d}.jpg'
        # decord 返回 RGB，OpenCV 需要 BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)
        saved_count += 1

    return saved_count


def main():
    # 读取最终标注
    import pandas as pd
    df = pd.read_csv('二次核查_最终标注.csv')

    # 创建输出目录
    output_dir = '/yanglin/eye_detect/training_data'
    os.makedirs(f'{output_dir}/close', exist_ok=True)
    os.makedirs(f'{output_dir}/open', exist_ok=True)

    print('=' * 80)
    print('提取闭眼帧作为微调训练数据')
    print('=' * 80)
    print()

    # 提取闭眼帧
    print('【提取闭眼帧】')
    print('-' * 80)
    close_cases = df[df['最终标注'] == '闭眼']
    close_total = 0

    for i, (_, row) in enumerate(close_cases.iterrows(), 1):
        video_path = row['视频路径']
        video_id = f"{row['车牌号']}_{row['视频日期']}_{row['视频时间'][:5]}"

        print(f'{i:02d}. {video_id} ... ', end='', flush=True)
        count = extract_frames_from_video(video_path, output_dir, 'close', video_id)
        close_total += count
        print(f'{count} 帧')

    print(f'\n闭眼帧总计: {close_total}')
    print()

    # 提取睁眼帧（作为负样本）
    print('【提取睁眼帧（负样本）】')
    print('-' * 80)
    open_cases = df[df['最终标注'].isin(['睁眼', '睁眼(下瞟/眨眼)'])].head(11)  # 取 11 个平衡
    open_total = 0

    for i, (_, row) in enumerate(open_cases.iterrows(), 1):
        video_path = row['视频路径']
        video_id = f"{row['车牌号']}_{row['视频日期']}_{row['视频时间'][:5]}"

        print(f'{i:02d}. {video_id} ... ', end='', flush=True)
        count = extract_frames_from_video(video_path, output_dir, 'open', video_id)
        open_total += count
        print(f'{count} 帧')

    print(f'\n睁眼帧总计: {open_total}')
    print()

    # 统计
    print('=' * 80)
    print('训练数据提取完成')
    print('=' * 80)
    print()
    print(f'输出目录: {output_dir}')
    print(f'  close/: {close_total} 帧')
    print(f'  open/: {open_total} 帧')
    print(f'  总计: {close_total + open_total} 帧')
    print()
    print('下一步：运行微调脚本')


if __name__ == '__main__':
    main()
