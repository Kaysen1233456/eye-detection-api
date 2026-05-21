#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
from tensorflow.python.framework.tensor_util import FastAppendBFloat16ArrayToTensorProto
import torch
from iopath.common.file_io import g_pathmgr

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.models import build_model
from tqdm import tqdm
import json
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from slowfast.datasets import utils as utils
from slowfast.datasets import video_container as container
from slowfast.datasets import decoder as decoder
import decord
from decord import VideoReader
from decord import cpu
import cv2
from deepface import DeepFace
import argparse
from slowfast.utils.misc import launch_job

logger = logging.get_logger(__name__)

def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def save_json(data_path, data):
    with open(data_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, sort_keys=True, ensure_ascii=False)


def extract_face_region(frames, pad=0.1, pad_mode='ratio'):
    """
    从视频帧中提取人脸区域
    参数:
        frames: 视频帧张量，形状为 (T, H, W, C)
        pad: padding值，用于扩展人脸区域
        pad_mode: 'ratio'或'pixels'，padding模式
    返回:
        处理后的帧张量，形状为 (T, H, W, C)
    """
    # 获取帧的尺寸信息
    T, H, W, C = frames.shape
    frames_np = frames.numpy()  # 转换为numpy数组进行处理
    
    # 标记整个视频是否检测到过人脸
    face_detected_in_video = False
    # 保存上一帧有效的bbox信息
    last_valid_bbox = None
    
    processed_frames = []
    
    # 逐帧处理
    for i in range(T):
        current_frame = frames_np[i]
        
        # 将tensor格式转换为OpenCV格式 (H, W, C) -> (H, W, 3)
        if current_frame.shape[2] == 3:
            # 确保数据类型为uint8
            if current_frame.dtype != np.uint8:
                current_frame = (current_frame * 255).astype(np.uint8)
            
            # 转换为BGR格式供OpenCV使用
            frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = current_frame
        
        current_face = None
        current_bbox = None
        
        try:
            # 使用DeepFace检测人脸
            face_objs = DeepFace.extract_faces(
                img_path=frame_bgr,
                detector_backend='yolov8',
                align=True
            )
            
            if len(face_objs) > 0:
                face = face_objs[0]
                area = face['facial_area']
                
                # 获取原始坐标
                x, y, w, h = area['x'], area['y'], area['w'], area['h']
                
                # 计算padding
                if pad_mode == 'ratio':
                    pad_w = int(w * pad)
                    pad_h = int(h * pad)
                else:
                    pad_w = pad_h = pad
                
                # 保存当前bbox信息（包含padding计算值）
                current_bbox = (x, y, w, h, pad_w, pad_h)
                last_valid_bbox = current_bbox  # 更新上一帧有效bbox
                
                # 扩展区域（确保不超出边界）
                new_x = max(0, x - pad_w)
                new_y = max(0, y - pad_h)
                new_w = min(W - new_x, w + 2*pad_w)
                new_h = min(H - new_y, h + 2*pad_h)
                
                # 截取人脸区域
                face_region = frame_bgr[
                    int(new_y):int(new_y+new_h), 
                    int(new_x):int(new_x+new_w)
                ]
                
                # 调整到原始尺寸
                resized_face = cv2.resize(face_region, (W, H))
                
                # 确保数据类型正确
                if resized_face.dtype != np.uint8:
                    resized_face = (resized_face * 255).astype(np.uint8)
                
                # 转换回RGB格式
                current_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
                face_detected_in_video = True  # 标记视频中检测到过人脸
                
        except Exception as e:
            print(f"处理第 {i} 帧时出错: {str(e)}")
            print('使用上一帧的区域')
        
        # 如果当前帧检测不到人脸，但存在上一帧的bbox信息
        if current_face is None and last_valid_bbox is not None:
            x, y, w, h, pad_w, pad_h = last_valid_bbox
            
            # 使用上一帧的bbox信息截取当前帧的区域
            new_x = max(0, x - pad_w)
            new_y = max(0, y - pad_h)
            new_w = min(W - new_x, w + 2*pad_w)
            new_h = min(H - new_y, h + 2*pad_h)
            
            face_region = frame_bgr[
                int(new_y):int(new_y+new_h), 
                int(new_x):int(new_x+new_w)
            ]
            
            resized_face = cv2.resize(face_region, (W, H))
            # 确保数据类型正确
            if resized_face.dtype != np.uint8:
                resized_face = (resized_face * 255).astype(np.uint8)
            current_face = resized_face
            current_face = cv2.cvtColor(current_face, cv2.COLOR_BGR2RGB)
        
        # 决定最终保存的帧
        if face_detected_in_video:
            # 如果视频中曾经检测到过人脸
            if current_face is not None:
                final_frame = current_face
            else:
                # 如果当前帧检测不到人脸，使用黑色填充
                final_frame = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            # 如果整个视频都没检测到人脸，使用原始帧
            final_frame = current_frame
        
        processed_frames.append(final_frame)
    
    if not face_detected_in_video:
        print("警告: 视频中未检测到人脸，将使用原始视频数据")
    
    # 将处理后的帧转换回tensor格式
    processed_frames = np.array(processed_frames)
    processed_frames = torch.from_numpy(processed_frames)
    
    return processed_frames


def visualize_face_extraction_video(original_frames, extracted_frames, output_path, fps=30):
    """
    将人脸提取结果可视化为视频文件
    参数:
        original_frames: 原始视频帧，形状为 (T, H, W, C)
        extracted_frames: 提取人脸后的帧，形状为 (T, H, W, C)
        output_path: 输出视频文件路径
        fps: 输出视频的帧率
    """
    # 确保输入是numpy数组
    if isinstance(original_frames, torch.Tensor):
        original_frames = original_frames.numpy()
    if isinstance(extracted_frames, torch.Tensor):
        extracted_frames = extracted_frames.numpy()
    
    T, H, W, C = original_frames.shape
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建视频写入器 - 使用并排显示原始帧和提取帧
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W * 2, H))
    
    print(f"开始生成人脸提取可视化视频，共 {T} 帧...")
    
    for i in range(T):
        # 处理原始帧
        original_frame = original_frames[i]
        if original_frame.dtype != np.uint8:
            original_frame = (original_frame * 255).astype(np.uint8)
        # 转换为BGR格式
        original_frame_bgr = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
        
        # 处理提取后的帧
        extracted_frame = extracted_frames[i]
        if extracted_frame.dtype != np.uint8:
            extracted_frame = (extracted_frame * 255).astype(np.uint8)
        # 转换为BGR格式
        extracted_frame_bgr = cv2.cvtColor(extracted_frame, cv2.COLOR_RGB2BGR)
        
        # 创建并排显示的画面
        combined_frame = np.hstack([original_frame_bgr, extracted_frame_bgr])
        
        # 添加文字标签
        cv2.putText(combined_frame, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, 'Face Extracted', (W + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined_frame, f'Frame {i+1}/{T}', (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 写入帧
        out.write(combined_frame)
    
    # 释放视频写入器
    out.release()
    print(f"人脸提取可视化视频已保存到: {output_path}")


def extract_face_region_with_visualization(frames, pad=0.1, pad_mode='ratio', enable_visualization=False, vis_output_path=None, fps=30):
    """
    人脸提取函数，可选择是否生成可视化视频
    参数:
        frames: 视频帧张量，形状为 (T, H, W, C)
        pad: padding值，用于扩展人脸区域
        pad_mode: 'ratio'或'pixels'，padding模式
        enable_visualization: 是否启用可视化
        vis_output_path: 可视化视频保存路径
        fps: 可视化视频帧率
    返回:
        处理后的帧张量，形状为 (T, H, W, C)
    """
    # 如果启用可视化，保存原始帧
    if enable_visualization:
        original_frames = frames.clone()
    
    # 执行人脸提取
    processed_frames = extract_face_region(frames, pad, pad_mode)
    
    # 如果启用可视化，生成可视化视频
    if enable_visualization and vis_output_path:
        visualize_face_extraction_video(original_frames, processed_frames, vis_output_path, fps)
    
    return processed_frames



@torch.no_grad()
def perform_test(model, cfg, video_path, video_st_time, video_ed_time, event_st_time, event_ed_time):
    """

    """
    # Enable eval mode.
    model.eval()
    # config
    temporal_sample_index =(0)
    spatial_sample_index = (0)
    min_scale, max_scale, crop_size = ([224] * 3)
    sampling_rate = utils.get_random_sampling_rate(0, 16)
    video_container = VideoReader(video_path, ctx=cpu(0))
    decord.bridge.set_bridge('torch')

    total_frame = len(video_container)
    fps = video_container.get_avg_fps()


    # 2. 统一时间单位为毫秒
    video_start_ms = video_st_time * 1000
    video_end_ms = video_ed_time * 1000
    video_duration_ms = video_end_ms - video_start_ms

    # 3. 计算事件在视频中的时间偏移
    start_offset_ms = event_st_time - video_start_ms
    end_offset_ms = event_ed_time - video_start_ms

    # 检查事件是否在视频时间范围内
    if start_offset_ms < 0 or end_offset_ms > video_duration_ms:
        print(f"事件超出视频时间范围: {event_st_time}-{event_ed_time} vs {video_start_ms}-{video_end_ms}: {video_path}")
        return None

    event_st_sec = start_offset_ms / 1000
    event_ed_sec = end_offset_ms / 1000

    start_frame = int(event_st_sec * fps)
    end_frame = int(event_ed_sec * fps)


    # Decode video. Meta info is used to perform selective decoding.
    frames = decoder.decode(
        video_container,
        sampling_rate,
        cfg.DATA.NUM_FRAMES,
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
        backend=cfg.DATA.DECODING_BACKEND,
        max_spatial_scale=min_scale,
        use_offset=cfg.DATA.USE_OFFSET_SAMPLING,
        sparse=True,
        start_frame=start_frame,
        end_frame=end_frame,
        video_frame_count=total_frame,
        INPUT_CLIP = cfg.INPUT_CLIP,
        VISUAL=cfg.VISUAL,
        EVALUATE=cfg.EVALUATE,
        LAST_FRAMES=cfg.LAST_FRAMES
    )

    # 读取的帧还需要提取人脸区域
    # ========== 可视化开关 ==========
    # 设置为 True 启用可视化，False 禁用可视化
    face_extraction = True
    ENABLE_VISUALIZATION = False  # 修改这里来控制是否生成可视化视频
    VIS_OUTPUT_PATH = "/yszhuo/projects/video_fatigue/workspace2_UniFormerV2/face_extraction_visualization.mp4"  # 可视化视频保存路径
    VIS_FPS = 15  # 可视化视频帧率
    # ================================
    
    if face_extraction and ENABLE_VISUALIZATION:
        # 使用带可视化的版本
        frames = extract_face_region_with_visualization(
            frames, 
            pad=0.5, 
            pad_mode='ratio',
            enable_visualization=True,
            vis_output_path=VIS_OUTPUT_PATH,
            fps=VIS_FPS
        )
    elif face_extraction:
        # 使用普通版本
        frames = extract_face_region(frames, pad=0.5, pad_mode='ratio')
    else:
        pass



    frames = utils.tensor_normalize(
        frames, cfg.DATA.MEAN, cfg.DATA.STD
    )
    # T H W C -> C T H W.
    frames = frames.permute(3, 0, 1, 2)
    # Perform data augmentation.
    frames = utils.spatial_sampling(
        frames,
        spatial_idx=spatial_sample_index,
        min_scale=min_scale,
        max_scale=max_scale,
        crop_size=crop_size,
        random_horizontal_flip=cfg.DATA.RANDOM_FLIP,
        inverse_uniform_sampling=cfg.DATA.INV_UNIFORM_SAMPLE,
    )

    inputs = utils.pack_pathway_output(cfg, frames)[0]
    inputs = [inputs.unsqueeze(0)]  # Add batch dimension.

    
    if cfg.NUM_GPUS:
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        print(inputs)
        preds = model(inputs).softmax(-1)
        preds = preds.cpu()
        print(preds)
        pred_id = torch.argmax(preds[0][1])
        if pred_id == 1:
            return 'eye_close'
        else:
            return 'eye_open'

def test_demo(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    logger.info(f'==========================================')
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        logger.info(f'==========================================')
        misc.log_model_info(model, cfg, use_train_input=False)
    logger.info(f'==========================================')
    cu.load_test_checkpoint(cfg, model)

    logger.info(f"Add softmax after prediction: {cfg.TEST.ADD_SOFTMAX}")

    # Perform multi-view test on the entire dataset.
    video_path = cfg.video_path
    video_st_time = cfg.video_st_time
    video_ed_time = cfg.video_ed_time
    event_st_time = cfg.event_st_time
    event_ed_time = cfg.event_ed_time

    result = perform_test(model, cfg, video_path, video_st_time, video_ed_time, event_st_time, event_ed_time)
    logger.info(f"Result: {result}")