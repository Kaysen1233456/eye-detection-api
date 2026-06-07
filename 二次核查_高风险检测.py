#!/usr/bin/env python3
"""
二次核查脚本 - 高风险视频闭眼检测

功能：
1. 读取 高风险0512-0520(1).xlsx 中的高风险视频列表
2. 在 close/ 目录下找到对应的 channel7.mp4 文件
3. 使用 eye_detect_api.1py + DeepFace + UniFormerV2 + 双卡 GPU 进行检测
4. 生成中文结果文件，对比 Excel 标注和模型检测结果

用法：
    python 二次核查_高风险检测.py
"""

import os
import sys
import re
import csv
import time
import importlib.util
import importlib.machinery
from pathlib import Path
from multiprocessing import Process, Queue

import pandas as pd

# 禁用 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 添加项目路径
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


def read_excel_data(excel_path: str) -> list[dict]:
    """
    读取 Excel 高风险文件

    Returns:
        包含 Excel 数据的字典列表
    """
    df = pd.read_excel(excel_path)

    records = []
    for _, row in df.iterrows():
        # 从视频目录路径提取目录名
        video_dir_path = str(row.get('视频目录路径', ''))
        dir_name = video_dir_path.split('\\')[-1] if '\\' in video_dir_path else video_dir_path

        records.append({
            '车牌号': str(row.get('车牌号', '')),
            '视频日期': str(row.get('视频日期', '')),
            '视频时间': str(row.get('视频时间', '')),
            '原始报警时间': str(row.get('原始报警时间', '')),
            '疲劳等级': str(row.get('疲劳等级', '')),
            'alarmId': str(row.get('alarmId', '')),
            '最终风险状态': str(row.get('最终风险状态', '')),
            '报警事件': str(row.get('报警事件', '')),
            '得分': str(row.get('得分', '')),
            '场景': str(row.get('场景', '')),
            '标注人': str(row.get('标注人', '')),
            'Excel标注': str(row.get('Excel标注', '')),
            '视频目录路径': video_dir_path,
            '目录名': dir_name,
        })

    return records


def find_channel7_video(close_dir: str, dir_name: str) -> str | None:
    """
    在 close 目录下查找对应的 channel7.mp4 文件

    Args:
        close_dir: close 目录路径
        dir_name: 子目录名

    Returns:
        视频文件路径，未找到返回 None
    """
    target_dir = Path(close_dir) / dir_name
    if not target_dir.exists():
        return None

    channel7_files = list(target_dir.glob('*_channel7.mp4'))
    if not channel7_files:
        return None

    return str(channel7_files[0])


def run_single_video(detector, video_path: str) -> dict:
    """对单个视频执行闭眼检测"""
    from decord import VideoReader, cpu

    # 解析文件名获取时间戳
    filename = Path(video_path).name
    timestamps = parse_video_filename(filename)
    if not timestamps:
        return {
            'success': False,
            'result': '解析失败',
            'confidence': 0.0,
            'error_message': f'无法解析文件名: {filename}',
        }

    video_st_time = timestamps['video_st_time']
    video_ed_time = timestamps['video_ed_time']
    event_ed_time_sec = timestamps['event_ed_time']

    # 读取 fps
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
    except Exception as e:
        return {
            'success': False,
            'result': '读取失败',
            'confidence': 0.0,
            'error_message': f'读取fps失败: {e}',
        }

    # 计算事件时间窗口 (毫秒)
    event_ed_time_ms = event_ed_time_sec * 1000
    frames_back_ms = int(16 / fps * 1000)
    event_st_time_ms = event_ed_time_ms - frames_back_ms

    # 构建输入参数
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

    # 转换结果为中文
    if result.success:
        result_cn = '闭眼' if result.result == 'eye_close' else '睁眼'
    else:
        result_cn = '检测失败'

    return {
        'success': result.success,
        'result': result_cn,
        'confidence': round(result.confidence, 4) if result.success else 0.0,
        'error_message': result.error_message if not result.success else '',
        'fps': round(fps, 2),
    }


def gpu_worker(gpu_id: int, task_list: list[dict], result_queue: Queue):
    """
    单个 GPU 工作进程

    Args:
        gpu_id: GPU 编号 (0 或 1)
        task_list: 该 GPU 负责处理的任务列表
        result_queue: 结果队列
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    api_module = load_original_api()

    print(f"[GPU {gpu_id}] 初始化检测器...")
    detector = api_module.create_eye_detector(
        config_path='exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml',
        checkpoint_path='best.pyth',
        device='cuda:0',
        enable_face_extraction=True,
        face_pad_ratio=0.5,
    )
    print(f"[GPU {gpu_id}] 检测器初始化完成，开始处理 {len(task_list)} 个视频")

    for i, task in enumerate(task_list, 1):
        dir_name = task['目录名']
        video_path = task['视频路径']
        excel_label = task['Excel标注']

        progress = f"[{i}/{len(task_list)}]"
        print(f"  [GPU {gpu_id}] {progress} {dir_name} ... ", end='', flush=True)

        start_time = time.time()
        detection_result = run_single_video(detector, video_path)
        elapsed = time.time() - start_time

        model_result = detection_result['result']
        confidence = detection_result['confidence']

        # 判断是否一致
        if detection_result['success']:
            # Excel标注可能是：闭眼、睁眼、下瞟、其他
            # 模型结果是：闭眼、睁眼
            if excel_label in ['闭眼', '睁眼']:
                match = '一致' if excel_label == model_result else '不一致'
            else:
                match = '无法对比'  # Excel标注不是闭眼/睁眼

            print(f"{model_result} ({confidence:.4f}) [{elapsed:.1f}s] Excel:{excel_label} {match}")
        else:
            match = '检测失败'
            print(f"失败 [{elapsed:.1f}s]")

        # 组装结果
        result_record = {
            **task,
            '模型检测结果': model_result,
            '模型置信度': confidence,
            '检测耗时(秒)': round(elapsed, 2),
            '是否一致': match,
            '错误信息': detection_result.get('error_message', ''),
        }

        result_queue.put(result_record)

    print(f"[GPU {gpu_id}] 处理完成")
    result_queue.put(None)  # 结束信号


def main():
    excel_path = '/yanglin/eye_detect/高风险0512-0520(1).xlsx'
    close_dir = '/yanglin/eye_detect/close'
    output_csv = '/yanglin/eye_detect/二次核查_高风险检测结果.csv'

    print("=" * 70)
    print("二次核查 - 高风险视频闭眼检测")
    print("使用: eye_detect_api.1py + DeepFace + UniFormerV2 + 双卡 A30 GPU")
    print("=" * 70)

    # Step 1: 读取 Excel
    print("\nStep 1: 读取高风险 Excel 文件...")
    excel_records = read_excel_data(excel_path)
    print(f"  Excel 中共 {len(excel_records)} 条记录")

    # Step 2: 匹配视频文件
    print("\nStep 2: 在 close/ 目录下匹配 channel7.mp4 文件...")
    matched_tasks = []
    not_found = []

    for record in excel_records:
        dir_name = record['目录名']
        video_path = find_channel7_video(close_dir, dir_name)

        if video_path:
            matched_tasks.append({
                **record,
                '视频路径': video_path,
            })
        else:
            not_found.append(record)

    print(f"  匹配成功: {len(matched_tasks)} 个视频")
    print(f"  未找到: {len(not_found)} 个")

    if not_found:
        print("\n  未找到的目录:")
        for r in not_found[:5]:
            print(f"    - {r['目录名']}")
        if len(not_found) > 5:
            print(f"    ... 还有 {len(not_found) - 5} 个")

    if not matched_tasks:
        print("\n没有匹配到视频，退出")
        return

    # Step 3: 分配到双 GPU
    print("\nStep 3: 分配视频到 2x A30 GPU...")
    mid = len(matched_tasks) // 2
    gpu0_tasks = matched_tasks[:mid]
    gpu1_tasks = matched_tasks[mid:]
    print(f"  GPU 0: {len(gpu0_tasks)} 个视频")
    print(f"  GPU 1: {len(gpu1_tasks)} 个视频")

    # Step 4: 启动双 GPU 并行处理
    print("\nStep 4: 启动双 GPU 并行检测...")
    result_queue = Queue()

    p0 = Process(target=gpu_worker, args=(0, gpu0_tasks, result_queue))
    p1 = Process(target=gpu_worker, args=(1, gpu1_tasks, result_queue))

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

    # Step 5: 生成中文 CSV
    print(f"\nStep 5: 生成结果文件: {output_csv}")

    csv_columns = [
        '车牌号', '视频日期', '视频时间', '原始报警时间',
        '疲劳等级', 'alarmId', '最终风险状态', '报警事件',
        '得分', '场景', '标注人',
        'Excel标注', '模型检测结果', '模型置信度',
        '是否一致', '检测耗时(秒)',
        '目录名', '视频路径', '错误信息',
    ]

    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for r in results:
            writer.writerow({col: r.get(col, '') for col in csv_columns})

    # Step 6: 统计
    success_results = [r for r in results if r['模型检测结果'] not in ['检测失败', '读取失败', '解析失败']]
    fail_results = [r for r in results if r['模型检测结果'] in ['检测失败', '读取失败', '解析失败']]

    # 统计一致/不一致
    match_results = [r for r in success_results if r['是否一致'] == '一致']
    mismatch_results = [r for r in success_results if r['是否一致'] == '不一致']
    na_results = [r for r in success_results if r['是否一致'] == '无法对比']

    # 统计 Excel 标注为闭眼/睁眼的
    excel_close = [r for r in success_results if r['Excel标注'] == '闭眼']
    excel_open = [r for r in success_results if r['Excel标注'] == '睁眼']

    # 模型检测结果统计
    model_close = [r for r in success_results if r['模型检测结果'] == '闭眼']
    model_open = [r for r in success_results if r['模型检测结果'] == '睁眼']

    print(f"\n{'=' * 70}")
    print("二次核查统计:")
    print(f"  Excel 总记录: {len(excel_records)}")
    print(f"  匹配到视频: {len(matched_tasks)}")
    print(f"  检测成功: {len(success_results)}")
    print(f"  检测失败: {len(fail_results)}")
    print()
    print("  Excel 标注分布:")
    print(f"    闭眼: {len(excel_close)}")
    print(f"    睁眼: {len(excel_open)}")
    print(f"    其他(下瞟等): {len(na_results)}")
    print()
    print("  模型检测分布:")
    print(f"    闭眼: {len(model_close)}")
    print(f"    睁眼: {len(model_open)}")
    print()
    print("  一致性分析 (仅限 Excel 标注为 闭眼/睁眼):")
    print(f"    一致: {len(match_results)}")
    print(f"    不一致: {len(mismatch_results)}")
    if match_results or mismatch_results:
        accuracy = len(match_results) / (len(match_results) + len(mismatch_results)) * 100
        print(f"    准确率: {accuracy:.1f}%")
    print()
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  平均每视频: {total_time/len(matched_tasks):.1f}s")
    print(f"  结果文件: {output_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
