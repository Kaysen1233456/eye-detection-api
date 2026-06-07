#!/usr/bin/env python3
"""
批量闭眼检测脚本

对 /yanglin/eye_detect/close/ 下所有 channel7 视频执行 UniFormerV2 闭眼检测。

视频命名规则:
    鲁E27851_1778603614_1778603621_1778603617__63_channel7.mp4
    └─车牌─┘ └video_st┘ └video_ed┘ └event_ed┘ └序┘ └通道┘

时间戳说明:
    第1个时间戳 (1778603614) = 视频开始时间 (秒)
    第2个时间戳 (1778603621) = 视频结束时间 (秒)
    第3个时间戳 (1778603617) = 闭眼事件结束时间 (秒)

检测窗口:
    event_ed_time = 第3个时间戳 (毫秒)
    event_st_time = event_ed_time - 16帧 (毫秒)
    16帧的实际时间 = 16 / fps

输出: eye_detect_results.csv
"""

import os
import sys
import re
import csv
import time
from pathlib import Path

# 禁用 TensorFlow 警告，启动时隐藏 GPU 避免 TF 初始化 segfault
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 启动时隐藏 GPU，DeepFace 用 CPU

# 添加项目路径
os.chdir('/yanglin/eye_detect')
sys.path.insert(0, '/yanglin/eye_detect')

# 先导入 eye_detect_api（会加载 DeepFace，此时 GPU 被隐藏，用 CPU）
from eye_detect_api import create_eye_detector, EyeDetectionInput

# DeepFace 已加载完毕，重新启用 GPU 供 PyTorch 使用
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 再导入 decord（此时 GPU 已启用）
from decord import VideoReader, cpu


def parse_video_filename(filename: str) -> dict | None:
    """
    从视频文件名中提取时间戳。

    文件名格式: 鲁E27851_1778603614_1778603621_1778603617__63_channel7.mp4
    """
    pattern = r'_(\d{10})_(\d{10})_(\d{10})__'
    match = re.search(pattern, filename)
    if not match:
        return None

    video_st_time = int(match.group(1))  # 视频开始 (秒)
    video_ed_time = int(match.group(2))  # 视频结束 (秒)
    event_ed_time = int(match.group(3))  # 事件结束 (秒)

    return {
        'video_st_time': video_st_time,
        'video_ed_time': video_ed_time,
        'event_ed_time': event_ed_time,
    }


def get_video_fps(video_path: str) -> float:
    """读取视频帧率。"""
    vr = VideoReader(video_path, ctx=cpu(0))
    return vr.get_avg_fps()


def find_all_channel7_videos(close_dir: str) -> list[dict]:
    """
    扫描 close 目录下所有 channel7 视频，提取时间戳信息。

    Returns:
        包含视频路径和时间戳信息的字典列表。
    """
    videos = []
    close_path = Path(close_dir)

    for sub_dir in sorted(close_path.iterdir()):
        if not sub_dir.is_dir():
            continue

        for video_file in sorted(sub_dir.glob('*_channel7.mp4')):
            timestamps = parse_video_filename(video_file.name)
            if timestamps is None:
                print(f"  [跳过] 无法解析文件名: {video_file.name}")
                continue

            videos.append({
                'video_path': str(video_file),
                'video_name': video_file.name,
                'sub_dir': sub_dir.name,
                **timestamps,
            })

    return videos


def run_eye_detection(detector, video_info: dict) -> dict:
    """
    对单个视频执行闭眼检测。

    Args:
        detector: EyeDetectionAPI 实例
        video_info: 视频信息字典

    Returns:
        包含检测结果的字典。
    """
    video_path = video_info['video_path']
    video_st_time = video_info['video_st_time']
    video_ed_time = video_info['video_ed_time']
    event_ed_time_sec = video_info['event_ed_time']

    # 读取视频 fps
    try:
        fps = get_video_fps(video_path)
    except Exception as e:
        return {
            **video_info,
            'fps': 0,
            'event_st_time_ms': 0,
            'event_ed_time_ms': 0,
            'result': 'error',
            'confidence': 0.0,
            'success': False,
            'error_message': f'读取fps失败: {e}',
        }

    # 计算事件时间窗口 (毫秒)
    event_ed_time_ms = event_ed_time_sec * 1000
    # 往前推 16 帧
    frames_back_ms = int(16 / fps * 1000)
    event_st_time_ms = event_ed_time_ms - frames_back_ms

    # 构建输入参数
    input_params = EyeDetectionInput(
        video_path=video_path,
        video_st_time=video_st_time,
        video_ed_time=video_ed_time,
        event_st_time=event_st_time_ms,
        event_ed_time=event_ed_time_ms,
    )

    # 执行检测
    result = detector.detect(input_params)

    return {
        **video_info,
        'fps': round(fps, 2),
        'event_st_time_ms': event_st_time_ms,
        'event_ed_time_ms': event_ed_time_ms,
        'frames_back': 16,
        'result': result.result if result.success else 'error',
        'confidence': round(result.confidence, 4) if result.success else 0.0,
        'success': result.success,
        'error_message': result.error_message if not result.success else '',
    }


def main():
    close_dir = '/yanglin/eye_detect/close'
    output_csv = '/yanglin/eye_detect/eye_detect_results.csv'

    print("=" * 60)
    print("批量闭眼检测")
    print("=" * 60)

    # Step 1: 扫描所有视频
    print("\nStep 1: 扫描 channel7 视频...")
    videos = find_all_channel7_videos(close_dir)
    print(f"  找到 {len(videos)} 个 channel7 视频")

    # Step 2: 创建检测器
    print("\nStep 2: 创建检测器 (GPU + DeepFace)...")
    detector = create_eye_detector(
        config_path='exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml',
        checkpoint_path='best.pyth',
        device='cuda',
        enable_face_extraction=True,
        face_pad_ratio=0.5,
    )
    print("  检测器创建成功")

    # Step 3: 逐个检测
    print(f"\nStep 3: 开始检测 {len(videos)} 个视频...")
    results = []
    success_count = 0
    error_count = 0
    eye_close_count = 0
    eye_open_count = 0

    for i, video_info in enumerate(videos, 1):
        video_name = video_info['video_name']
        sub_dir = video_info['sub_dir']

        # 进度显示
        progress = f"[{i}/{len(videos)}]"
        print(f"  {progress} {sub_dir}/{video_name} ... ", end='', flush=True)

        start_time = time.time()
        result = run_eye_detection(detector, video_info)
        elapsed = time.time() - start_time

        if result['success']:
            success_count += 1
            if result['result'] == 'eye_close':
                eye_close_count += 1
                print(f"eye_close ({result['confidence']:.4f}) [{elapsed:.1f}s]")
            else:
                eye_open_count += 1
                print(f"eye_open  ({result['confidence']:.4f}) [{elapsed:.1f}s]")
        else:
            error_count += 1
            print(f"ERROR: {result['error_message'][:50]} [{elapsed:.1f}s]")

        results.append(result)

    # Step 4: 生成 CSV
    print(f"\nStep 4: 生成 CSV: {output_csv}")

    csv_columns = [
        'sub_dir', 'video_name', 'video_path',
        'video_st_time', 'video_ed_time',
        'event_ed_time', 'fps',
        'event_st_time_ms', 'event_ed_time_ms', 'frames_back',
        'result', 'confidence', 'success', 'error_message',
    ]

    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for r in results:
            writer.writerow({col: r.get(col, '') for col in csv_columns})

    # Step 5: 统计
    print(f"\n{'=' * 60}")
    print("检测完成统计:")
    print(f"  总视频数: {len(videos)}")
    print(f"  成功: {success_count}")
    print(f"    eye_close (闭眼): {eye_close_count}")
    print(f"    eye_open  (睁眼): {eye_open_count}")
    print(f"  失败: {error_count}")
    print(f"  CSV 已保存: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
