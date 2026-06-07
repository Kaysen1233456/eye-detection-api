#!/usr/bin/env python3
"""
批量闭眼检测脚本 - 双GPU并行版本

基于 eye_detect_api.1py (原始代码)，使用 2x A30 GPU 并行处理。

用法:
    python batch_eye_detect_dual_gpu.py
"""

import os
import sys
import re
import csv
import time
import importlib.util
from pathlib import Path
from multiprocessing import Process, Queue

# 禁用 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 添加项目路径
os.chdir('/yanglin/eye_detect')
sys.path.insert(0, '/yanglin/eye_detect')

# 修复 torchvision 兼容性问题 (functional_tensor 在新版被移除)
import types
import torchvision.transforms.functional as _tvF
_functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
_functional_tensor.rgb_to_grayscale = _tvF.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor


def load_original_api():
    """动态加载 eye_detect_api.1py 原始代码"""
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader(
        "eye_detect_api_original",
        "/yanglin/eye_detect/eye_detect_api.1py"
    )
    spec = importlib.util.spec_from_loader("eye_detect_api_original", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def parse_video_filename(filename: str) -> dict | None:
    """
    从视频文件名中提取时间戳。

    文件名格式: 鲁E27851_1778603614_1778603621_1778603617__63_channel7.mp4
    """
    pattern = r'_(\d{10})_(\d{10})_(\d{10})__'
    match = re.search(pattern, filename)
    if not match:
        return None

    video_st_time = int(match.group(1))
    video_ed_time = int(match.group(2))
    event_ed_time = int(match.group(3))

    return {
        'video_st_time': video_st_time,
        'video_ed_time': video_ed_time,
        'event_ed_time': event_ed_time,
    }


def find_all_channel7_videos(close_dir: str) -> list[dict]:
    """扫描 close 目录下所有 channel7 视频"""
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


def run_single_video(detector, video_info: dict) -> dict:
    """对单个视频执行闭眼检测"""
    from decord import VideoReader, cpu

    video_path = video_info['video_path']
    video_st_time = video_info['video_st_time']
    video_ed_time = video_info['video_ed_time']
    event_ed_time_sec = video_info['event_ed_time']

    # 读取 fps
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
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
    frames_back_ms = int(16 / fps * 1000)
    event_st_time_ms = event_ed_time_ms - frames_back_ms

    # 构建输入参数 (使用原始 API 的 EyeDetectionInput)
    api_module = load_original_api()
    input_params = api_module.EyeDetectionInput(
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


def gpu_worker(gpu_id: int, video_list: list[dict], result_queue: Queue):
    """
    单个 GPU 工作进程

    Args:
        gpu_id: GPU 编号 (0 或 1)
        video_list: 该 GPU 负责处理的视频列表
        result_queue: 结果队列
    """
    # 设置当前进程使用的 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 动态加载原始 API
    api_module = load_original_api()

    # 创建检测器 (注意：在 CUDA_VISIBLE_DEVICES 设置后，cuda:0 就是对应的物理 GPU)
    print(f"[GPU {gpu_id}] 初始化检测器...")
    detector = api_module.create_eye_detector(
        config_path='exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml',
        checkpoint_path='best.pyth',
        device='cuda:0',  # 在 CUDA_VISIBLE_DEVICES 下，cuda:0 是当前进程的 GPU
        enable_face_extraction=True,
        face_pad_ratio=0.5,
    )
    print(f"[GPU {gpu_id}] 检测器初始化完成，开始处理 {len(video_list)} 个视频")

    # 逐个检测
    for i, video_info in enumerate(video_list, 1):
        video_name = video_info['video_name']
        sub_dir = video_info['sub_dir']

        progress = f"[{i}/{len(video_list)}]"
        print(f"  [GPU {gpu_id}] {progress} {sub_dir}/{video_name} ... ", end='', flush=True)

        start_time = time.time()
        result = run_single_video(detector, video_info)
        elapsed = time.time() - start_time

        if result['success']:
            if result['result'] == 'eye_close':
                print(f"eye_close ({result['confidence']:.4f}) [{elapsed:.1f}s]")
            else:
                print(f"eye_open  ({result['confidence']:.4f}) [{elapsed:.1f}s]")
        else:
            print(f"ERROR: {result['error_message'][:50]} [{elapsed:.1f}s]")

        result_queue.put(result)

    print(f"[GPU {gpu_id}] 处理完成")
    result_queue.put(None)  # 结束信号


def main():
    close_dir = '/yanglin/eye_detect/close'
    output_csv = '/yanglin/eye_detect/eye_detect_results_dual_gpu.csv'

    print("=" * 60)
    print("批量闭眼检测 - 双GPU并行版本")
    print("使用原始代码: eye_detect_api.1py")
    print("=" * 60)

    # Step 1: 扫描所有视频
    print("\nStep 1: 扫描 channel7 视频...")
    videos = find_all_channel7_videos(close_dir)
    print(f"  找到 {len(videos)} 个 channel7 视频")

    if len(videos) == 0:
        print("没有找到视频，退出")
        return

    # Step 2: 分配视频到两个 GPU
    print("\nStep 2: 分配视频到 2x A30 GPU...")
    mid = len(videos) // 2
    gpu0_videos = videos[:mid]
    gpu1_videos = videos[mid:]
    print(f"  GPU 0: {len(gpu0_videos)} 个视频")
    print(f"  GPU 1: {len(gpu1_videos)} 个视频")

    # Step 3: 启动双 GPU 并行处理
    print("\nStep 3: 启动双 GPU 并行处理...")
    result_queue = Queue()

    p0 = Process(target=gpu_worker, args=(0, gpu0_videos, result_queue))
    p1 = Process(target=gpu_worker, args=(1, gpu1_videos, result_queue))

    start_time = time.time()
    p0.start()
    p1.start()

    # 收集结果
    results = []
    finished_count = 0
    while finished_count < 2:
        result = result_queue.get()
        if result is None:
            finished_count += 1
        else:
            results.append(result)

    p0.join()
    p1.join()
    total_time = time.time() - start_time

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
    success_count = sum(1 for r in results if r['success'])
    error_count = sum(1 for r in results if not r['success'])
    eye_close_count = sum(1 for r in results if r.get('result') == 'eye_close')
    eye_open_count = sum(1 for r in results if r.get('result') == 'eye_open')

    print(f"\n{'=' * 60}")
    print("检测完成统计:")
    print(f"  总视频数: {len(videos)}")
    print(f"  成功: {success_count}")
    print(f"    eye_close (闭眼): {eye_close_count}")
    print(f"    eye_open  (睁眼): {eye_open_count}")
    print(f"  失败: {error_count}")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  平均每视频: {total_time/len(videos):.1f}s")
    print(f"  CSV 已保存: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
