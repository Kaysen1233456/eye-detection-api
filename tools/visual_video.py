#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import cv2
import pickle
import torch
from iopath.common.file_io import g_pathmgr

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm
import json
from slowfast.datasets import utils as utils
from slowfast.datasets import video_container as container
from slowfast.datasets import decoder as decoder
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import datetime


logger = logging.get_logger(__name__)

def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def save_json(data_path, data):
    with open(data_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, sort_keys=True, ensure_ascii=False)

@torch.no_grad()
def perform_test(model, cfg):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    annotation = load_json(cfg.DATA.PATH_TO_DATA_DIR_VAL)    
    
    # config
    temporal_sample_index =(0)
    spatial_sample_index = (0)
    min_scale, max_scale, crop_size = ([224] * 3)
    sampling_rate = utils.get_random_sampling_rate( 0, 16)
    
    for anno in tqdm(annotation):
        
        result = []
        # 滑动窗口参数
        window_size = 8   # 每个窗口8帧
        step = 1          # 每次滑动步长（1帧）
        current_frame = 0  # 起始帧（从0开始）
        start_frames = []
        end_frames = []
        video_frame_count = anno['video_frame_count']  # 视频总帧数
        # if "疲劳" not in anno['label']:
        #     continue
        
        # 滑动窗口循环
        while current_frame + window_size <= anno['video_frame_count']:
            start_frame = current_frame  # 窗口起始帧
            end_frame = current_frame + window_size - 1  # 窗口结束帧
                            
            # 更新下一次起始帧（每次滑动1帧）
            current_frame += step
            start_frames.append(start_frame)
            end_frames.append(end_frame)
        
        for start_frame, end_frame in tqdm(zip(start_frames, end_frames), desc="Processing video frames"):
        
            video_container = container.get_video_container(anno['face_clip_path'], False, 'decord')

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
                video_frame_count=video_frame_count,
                INPUT_CLIP = cfg.INPUT_CLIP,
                VISUAL=cfg.VISUAL,
                LAST_FRAMES=cfg.LAST_FRAMES
            )


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

                # Perform the forward pass.
                if cfg.TEST.ADD_SOFTMAX:
                    preds = model(inputs).softmax(-1)
                else:
                    preds = model(inputs)

                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    preds, labels, video_idx = du.all_gather(
                        [preds, labels, video_idx]
                    )
                if cfg.NUM_GPUS:
                    preds = preds.cpu()
                    result.append(np.round(preds[0][1].numpy(), 2))
        print(1)
        # 输入配置
        video_path = anno['video_path']
        output_path = cfg.OUTPUT_DIR
        label = os.path.basename(os.path.dirname(video_path))
        output_path = os.path.join(output_path, label, video_path.split('/')[-1])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 1. 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video at {video_path}")

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 2. 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 3. 确定当前帧的分数
            if frame_idx < 7:  # 前7帧使用第一个分数
                score = result[0]
            else:  # 后续帧使用result中对应索引的分数
                score_idx = frame_idx - 7
                if score_idx < len(result):
                    score = result[score_idx]
                else:  # 边界情况：使用最后一个可用分数
                    score = result[-1]
            
            # 4. 在帧上绘制分数
            text = f"Close Eye Score: {score:.2f}"  # 保留两位小数
            cv2.putText(frame, text, 
                        (50, 50),  # 文本位置 (左上角x,y)
                        cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                        1,  # 字体大小
                        (0, 255, 0),  # 绿色 (BGR格式)
                        2,  # 线宽
                        cv2.LINE_AA)  # 抗锯齿
            
            # 5. 写入输出视频
            out.write(frame)
            
            frame_idx += 1

        # 6. 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"处理完成! 输出视频已保存至 {output_path}")
        print(f"总处理帧数: {frame_idx}, 使用分数数: {len(result)}")


def visualize(cfg):
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
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    # test_loader = loader.construct_loader(cfg, "test")
    # logger.info("Testing model for {} iterations".format(len(test_loader)))
    logger.info(f"Add softmax after prediction: {cfg.TEST.ADD_SOFTMAX}")


    # # Perform multi-view test on the entire dataset.
    perform_test(model, cfg)