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


class FatigueDataset(Dataset):
    def __init__(self, annotation_path, cfg):
        """
        疲劳检测专用数据集
        :param annotation_path: JSON标注文件路径
        :param cfg: 配置文件对象
        """
        self.cfg = cfg
        
        # 加载并筛选标注
        with open(annotation_path) as f:
            self.annotations = json.load(f)
            
        # 过滤疲劳类型
        self.fatigue_types = ['疲劳_近似人工', '疲劳_人工干预唤醒', '疲劳_TTS干预', '疲劳_人工干预停车']

            
        # 视频加载器缓存
        self.video_cache = {}
        self.window_cache = {}
        
        # 预计算所有滑动窗口
        print("预计算滑动窗口...")
        self.window_infos = []
        # for anno_idx, anno in enumerate(tqdm(random.sample(self.annotations, 50))):
        for anno_idx, anno in enumerate(tqdm(self.annotations)):
            video_path = anno['video_path']
            face_clip_path = anno['face_clip_path']
            total_frames = anno['video_frame_count']
            chinese_label = anno['label']
            if chinese_label in self.fatigue_types:
                fatigue_label = 1
            else:
                fatigue_label = 0       
                   
            window_info = []
            for start in range(0, total_frames - cfg.DATA.NUM_FRAMES + 1, 1):
                end = start + cfg.DATA.NUM_FRAMES - 1
                window_info.append({
                    'video_path': video_path,
                    'face_clip_path':face_clip_path,
                    'start_frame': start,
                    'end_frame': end,
                    'total_frames': total_frames,
                    'fatigue_label': fatigue_label,
                    'video_idx': anno_idx,  # 添加标注索引
                })
                
            self.window_infos.extend(window_info)
    
    def __len__(self):
        return len(self.window_infos)
    
    def __getitem__(self, index):
        
        window_info = self.window_infos[index]
        start_frame = window_info['start_frame']
        end_frame = window_info['end_frame']
        total_frames = window_info['total_frames']
            
        # 加载视频
        video_container = container.get_video_container(window_info['face_clip_path'], False, 'decord')
        fps = video_container.get_avg_fps()
        temporal_sample_index =(0)
        spatial_sample_index = (0)
        min_scale, max_scale, crop_size = ([224] * 3)
        sampling_rate = utils.get_random_sampling_rate(0, 16)

        # Decode video. Meta info is used to perform selective decoding.
        frames = decoder.decode(
            video_container,
            sampling_rate,
            self.cfg.DATA.NUM_FRAMES,
            temporal_sample_index,
            self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
            backend=self.cfg.DATA.DECODING_BACKEND,
            max_spatial_scale=min_scale,
            use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
            sparse=True,
            start_frame=start_frame,
            end_frame=end_frame,
            video_frame_count=total_frames,
            INPUT_CLIP=self.cfg.INPUT_CLIP,
            VISUAL=self.cfg.VISUAL,
            EVALUATE=self.cfg.EVALUATE,
            LAST_FRAMES=self.cfg.LAST_FRAMES
        )


        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
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
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        window_info['fps'] = fps
        return (frames, window_info)
        
    
def collate_fn(batch):
    """自定义批处理函数，过滤无效样本并填充序列"""

    # 解包批次
    frames, window_infos = zip(*batch)

    batch_frames = torch.stack(frames)
    
    return batch_frames, window_infos

@torch.no_grad()
def perform_test(model, loader, cfg):
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
    
    # metric

    results = defaultdict(list)
    fatigue_labels = {}
    video_paths = {}
    fps_videos = {}
    annotations = {}
    
    
    
    for cur_iter, (inputs, window_infos) in tqdm(enumerate(loader), total=len(loader)):
        
        inputs = [inputs]
        
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
                
                for i in range(len(window_infos)):
                    video_idx = window_infos[i]['video_idx']
                    if video_idx not in fatigue_labels:
                        fatigue_labels[video_idx] = window_infos[i]['fatigue_label']
                        video_paths[video_idx] = window_infos[i]['video_path']
                        fps_videos[video_idx] = window_infos[i]['fps']
                        annotations[video_idx] = window_infos[i]

                    results[video_idx].append(preds[i][1].numpy())
                
    
    # 输入配置
    details = []
    predictions = {}
    unsuccess_annos =[]

    close_threshold=0.5
    long_duration=0.3
    short_window=1.0
    perclose_threshold=0.3

    for video_idx, pred_score in results.items():
        assert video_idx in fatigue_labels
        
        fps = fps_videos[video_idx]
        pred_fatigue, reason = detect_fatigue(
            pred_score, 
            fps=fps,
            close_threshold=close_threshold, 
            long_duration=long_duration, 
            short_window=short_window,
            perclose_threshold=perclose_threshold
            )
        predictions[video_idx] = pred_fatigue
        details.append({
            "scores": pred_score,
            "prediction": pred_fatigue,
            "reason": reason
        })
        
        if pred_fatigue != fatigue_labels[video_idx]:
            annotations[video_idx]['prediction'] = pred_fatigue
            unsuccess_annos.append(annotations[video_idx])
            
        
    # 1. 计算性能指标
    metrics = metric_cal(predictions, fatigue_labels)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report = generate_report(metrics)
    unsuccess_annos_path = os.path.join(cfg.OUTPUT_DIR, f"unsuccess-{current_time}.json")
    
    logger.info("Performance metrics:")
    logger.info(report)
    logger.info(f"Unsuccessful annotations saved to {unsuccess_annos_path}")
    save_json(unsuccess_annos_path, unsuccess_annos)
    # 记录信息close_threshold, long_duration, short_window, perclose_threshold
    logger.info(f"Parameters used: close_threshold={close_threshold}, long_duration={long_duration}, short_window={short_window}, perclose_threshold={perclose_threshold}")
        
def video_fatigue_test(cfg):
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
    
    dataset = FatigueDataset(cfg.DATA.PATH_TO_DATA_DIR_VAL, cfg)
    shuffle = False
    drop_last = False  
      
    # Create a sampler for multi-process training
    sampler = utils.create_sampler(dataset, shuffle, cfg)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size= int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS)),
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=24,
        # num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=utils.loader_worker_init_fn(dataset),
        persistent_workers=True
    )
    
    perform_test(model, loader, cfg)
    
def detect_fatigue(eye_scores, fps, 
                  close_threshold=0.5, 
                  long_duration=0.2, 
                  short_window=1.5,
                  perclose_threshold=0.3):
    """
    检测驾驶员疲劳状态
    参数:
    eye_scores: 各帧的闭眼概率列表(0-1), 0表示睁眼, 1表示完全闭眼
    fps: 视频帧率(帧/秒)
    close_threshold: 闭眼判断阈值(0-1)
    long_duration: 长时间闭眼阈值(秒)
    short_window: PERCLOS检测窗口大小(秒)
    perclose_threshold: PERCLOS疲劳阈值(0-1)
    返回: (is_fatigue, reason)
    """
    
    # 1. 预处理: 将概率转为二值状态
    eye_states = [1 if score >= close_threshold else 0 for score in eye_scores]
    
    # 2. 检测长时间闭眼 (长时闭眼标准)
    continuous_threshold = int(long_duration * fps)  # 转换为帧数
    print(f'continuous_threshold:{continuous_threshold}')
    current_close = 0
    for state in eye_states:
        if state == 1:
            current_close += 1
            if current_close >= continuous_threshold:
                return True, "Long eye closure detected ({}s)".format(long_duration)
        else:
            current_close = 0
    
    # 3. 检测频繁眨眼 (PERCLOS标准)
    window_size = int(short_window * fps)  # 转换为帧数
    print(f'window_size:{window_size}')
    for i in range(len(eye_states) - window_size + 1):
        window = eye_states[i:i+window_size]
        close_ratio = sum(window) / window_size  # 闭眼帧占比
        if close_ratio >= perclose_threshold:
            return True, "High PERCLOS detected ({:.1%} closure in window_size{})".format(
                close_ratio, short_window)
    
    return False, "Normal driving state"

def metric_cal(predictions, fatigue_labels):
    """
    评估疲劳检测系统性能，计算两个类别的指标
    :param predictions: 预测结果字典 {视频ID: 预测结果True/False}
    :param fatigue_labels: 真实标签字典 {视频ID: 疲劳状态0/1}
    返回: 包含各项指标的字典（两个类别的指标）
    """
    # 确保两个字典键值一致
    video_ids = sorted(predictions.keys())
    if set(video_ids) != set(fatigue_labels.keys()):
        raise ValueError("预测结果和标签的视频ID不一致")
    
    # 转换为列表
    y_pred = [predictions[vid] for vid in video_ids]  # True/False
    y_true = [fatigue_labels[vid] for vid in video_ids]  # 0/1
    
    # 转换预测结果为二进制 (True->1, False->0)
    y_pred_binary = [1 if pred else 0 for pred in y_pred]
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    # 计算总体准确率
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    # 计算每个类别的召回率
    non_eye_close_recall = tn / (tn + fp) if (tn + fp) > 0 else 0  # 非闭眼召回率（真负率）
    eye_close_recall = tp / (tp + fn) if (tp + fn) > 0 else 0      # 闭眼召回率（真正率）
    
    # 计算每个类别的精确率
    non_eye_close_precision = tn / (tn + fn) if (tn + fn) > 0 else 0  # 非闭眼精确率
    eye_close_precision = tp / (tp + fp) if (tp + fp) > 0 else 0      # 闭眼精确率
    
    # 计算F1分数
    non_eye_close_f1 = (2 * non_eye_close_precision * non_eye_close_recall) / \
                       (non_eye_close_precision + non_eye_close_recall + 1e-8)
    eye_close_f1 = (2 * eye_close_precision * eye_close_recall) / \
                   (eye_close_precision + eye_close_recall + 1e-8)
    
    return {
        "confusion_matrix": conf_matrix.tolist(),
        "overall_accuracy": accuracy,
        "non_eye_close": {
            "precision": non_eye_close_precision,
            "recall": non_eye_close_recall,
            "f1": non_eye_close_f1,
            "support": tn + fp  # 非闭眼样本总数
        },
        "eye_close": {
            "precision": eye_close_precision,
            "recall": eye_close_recall,
            "f1": eye_close_f1,
            "support": tp + fn  # 闭眼样本总数
        },
        "total_samples": len(y_true),
        "raw_values": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp
        }
    }

def generate_report(metrics, file_path=None):
    """
    生成详细性能报告
    :param metrics: 评估指标字典
    :param file_path: 报告保存路径(可选)
    """
    # 提取关键指标
    tn = metrics['raw_values']['true_negative']
    fp = metrics['raw_values']['false_positive']
    fn = metrics['raw_values']['false_negative']
    tp = metrics['raw_values']['true_positive']
    
    non_eye_close = metrics['non_eye_close']
    eye_close = metrics['eye_close']
    
    # 生成详细报告
    report = (
        "疲劳检测系统性能评估报告\n"
        "=================================\n\n"
        f"总样本数: {metrics['total_samples']} 个视频\n\n"
        f"整体准确率: {metrics['overall_accuracy']:.4f} ({(metrics['overall_accuracy']*100):.2f}%)\n\n"
        "非闭眼类指标(类别0):\n"
        f"  精确率: {non_eye_close['precision']:.4f}  |  召回率: {non_eye_close['recall']:.4f}  |  F1分数: {non_eye_close['f1']:.4f}  |  样本数: {non_eye_close['support']}\n\n"
        "闭眼类指标(类别1):\n"
        f"  精确率: {eye_close['precision']:.4f}  |  召回率: {eye_close['recall']:.4f}  |  F1分数: {eye_close['f1']:.4f}  |  样本数: {eye_close['support']}\n\n"
        "混淆矩阵:\n"
        f"          Predicted 0 (非闭眼)  Predicted 1 (闭眼)\n"
        f"Actual 0 |      {tn:<7}        |      {fp:<7}      \n"
        f"Actual 1 |      {fn:<7}        |      {tp:<7}      \n\n"
        "关键指标解释:\n"
        "- 精确率: 系统判断为疲劳/非疲劳的视频中，真正是正确的比例\n"
        "- 召回率: 所有真实的疲劳/非疲劳视频中，被系统成功检测出的比例\n"
        "- F1分数: 精确率和召回率的调和平均数，综合评估指标\n"
        "- 真阴性(TN): 正确识别的非疲劳视频数量\n"
        "- 假阳性(FP): 误判为疲劳的正常视频数量\n"
        "- 假阴性(FN): 漏检的疲劳视频数量\n"
        "- 真阳性(TP): 正确检测出的疲劳视频数量"
    )
    
    return report