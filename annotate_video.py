"""
视频闭眼概率标注工具。

读取输入视频，逐段跑闭眼检测模型，在左上角叠加实时概率和状态，
输出标注后的视频。
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
import decord

from eye_detect_api import (
    EyeDetectionInput,
    create_eye_detector,
    create_multi_gpu_detector,
)


def draw_overlay(frame: np.ndarray, prob_close: float, prob_open: float,
                 status: str, fps: float, inference_ms: float) -> np.ndarray:
    """在帧的左上角绘制检测结果叠加层。"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 半透明背景框
    box_w, box_h = 320, 110
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # 状态颜色
    if status == "EYE_CLOSED":
        color = (0, 0, 255)     # 红色
        label = "CLOSED"
    elif status == "EYE_OPEN":
        color = (0, 255, 0)     # 绿色
        label = "OPEN"
    else:
        color = (128, 128, 128) # 灰色
        label = "N/A"

    # 概率条背景
    bar_x, bar_y, bar_w, bar_h = 10, 70, 300, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    # 概率条填充（闭眼概率，红色填充）
    fill_w = int(prob_close * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)

    # 文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Eye Detection", (10, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Status: {label}", (10, 50), font, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Close: {prob_close:.1%}  Open: {prob_open:.1%}",
                (10, 95), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Inference: {inference_ms:.0f}ms",
                (10, box_h - 5), font, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


def run_annotation(args):
    """主流程：读视频 → 滑窗检测 → 叠加概率 → 写视频。"""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"输入视频不存在: {input_path}")
        return

    # 创建检测器
    if args.gpu and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} GPU 并行模式")
        detector = create_multi_gpu_detector(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            enable_face_extraction=True,
            face_pad_ratio=0.5,
        )
    else:
        device = args.device if args.gpu else 'cpu'
        print(f"使用单设备: {device}")
        detector = create_eye_detector(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=device,
            enable_face_extraction=True,
            face_pad_ratio=0.5,
        )

    # 用 decord 读取视频信息
    vr = VideoReader(str(input_path), ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    width, height = vr[0].shape[1], vr[0].shape[0]
    duration_sec = total_frames / fps

    print(f"输入视频: {input_path.name}")
    print(f"  分辨率: {width}x{height}, FPS: {fps:.1f}, 总帧: {total_frames}, 时长: {duration_sec:.1f}s")

    # 构建四舍五入到整秒的视频时间范围（Unix 时间戳）
    now = int(time.time())
    video_st_time = now
    video_ed_time = now + int(duration_sec) + 1

    # 滑窗参数
    window = args.window       # 每次检测覆盖的帧数
    step = args.step           # 滑窗步长

    # 存储每帧的检测结果
    frame_probs = np.zeros((total_frames, 2), dtype=np.float32)  # [prob_open, prob_close]
    frame_valid = np.zeros(total_frames, dtype=bool)

    print(f"开始检测 (窗口={window}, 步长={step})...")
    t_start = time.perf_counter()

    for start_f in range(0, total_frames, step):
        end_f = min(start_f + window, total_frames)
        mid_f = (start_f + end_f) // 2

        # 计算事件时间范围（毫秒）
        event_st_ms = int(video_st_time * 1000 + (start_f / fps) * 1000)
        event_ed_ms = int(video_st_time * 1000 + (end_f / fps) * 1000)

        inp = EyeDetectionInput(
            video_path=str(input_path),
            video_st_time=video_st_time,
            video_ed_time=video_ed_time,
            event_st_time=event_st_ms,
            event_ed_time=event_ed_ms,
        )

        t0 = time.perf_counter()
        result = detector.detect(inp)
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000

        if result.success:
            # 从 result 反推概率
            if result.result == 'eye_close':
                prob = [1 - result.confidence, result.confidence]
            else:
                prob = [result.confidence, 1 - result.confidence]

            # 覆盖窗口内所有帧
            for f in range(start_f, end_f):
                frame_probs[f] = prob
                frame_valid[f] = True

            status = "CLOSED" if result.result == 'eye_close' else "OPEN"
            elapsed = time.perf_counter() - t_start
            progress = (end_f / total_frames) * 100
            print(f"  [{progress:5.1f}%] 帧 {start_f}-{end_f} -> {status} "
                  f"(conf={result.confidence:.3f}, {infer_ms:.0f}ms) "
                  f"[{elapsed:.1f}s elapsed]")

    # 处理未被覆盖的帧（前后插值）
    for f in range(total_frames):
        if not frame_valid[f]:
            # 找最近的有效帧
            valid_idx = np.where(frame_valid)[0]
            if len(valid_idx) > 0:
                nearest = valid_idx[np.argmin(np.abs(valid_idx - f))]
                frame_probs[f] = frame_probs[nearest]
            else:
                frame_probs[f] = [0.5, 0.5]  # 无数据时默认

    # 写输出视频
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"\n写入输出视频: {output_path.name}")
    decord.bridge.set_bridge('torch')

    for f in range(total_frames):
        # 读帧
        frame_torch = vr[f]
        frame_np = frame_torch.numpy()
        if frame_np.ndim == 3 and frame_np.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_np

        prob_open, prob_close = frame_probs[f]
        status = "EYE_CLOSED" if prob_close > prob_open else "EYE_OPEN"

        annotated = draw_overlay(frame_bgr, prob_close, prob_open, status, fps, 0)
        writer.write(annotated)

        if f % 50 == 0:
            print(f"  写入帧 {f}/{total_frames}")

    writer.release()
    total_time = time.perf_counter() - t_start
    print(f"\n完成! 输出: {output_path}")
    print(f"总耗时: {total_time:.1f}s ({total_time/duration_sec:.1f}x 实时速度)")


def main():
    parser = argparse.ArgumentParser(description="视频闭眼概率标注工具")
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument("-o", "--output", required=True, help="输出视频路径")
    parser.add_argument("-c", "--config", default="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
                        help="模型配置文件路径")
    parser.add_argument("-w", "--checkpoint", default="best.pyth", help="模型权重路径")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU 加速")
    parser.add_argument("--device", default="cuda:0", help="推理设备 (默认 cuda:0)")
    parser.add_argument("--window", type=int, default=32, help="检测窗口帧数 (默认 32)")
    parser.add_argument("--step", type=int, default=16, help="滑窗步长 (默认 16)")
    args = parser.parse_args()
    run_annotation(args)


if __name__ == "__main__":
    main()
