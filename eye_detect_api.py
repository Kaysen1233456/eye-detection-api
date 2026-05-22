"""
闭眼检测API - 简洁版本

基于 UniFormerV2 模型的闭眼检测系统，提供简洁的API接口：
1. 删除所有打印信息
2. 支持可配置参数：视频路径，视频时间范围，事件时间范围
3. 只返回闭眼检测结果

作者：基于 test_net_demo.py 改造
版本：API v1.0
日期：2025-11-13
"""

from __future__ import annotations

import os
import sys
import types

# ========== 禁用TensorFlow/CUDA警告 ==========
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# ==========================================

# ========== 兼容性修复：pytorchvideo 需要 functional_tensor ==========
import torchvision.transforms.functional as _F
_functional_tensor = types.ModuleType('torchvision.transforms.functional_tensor')
_functional_tensor.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _functional_tensor
# =====================================================================

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import decord
import numpy as np
import torch
from decord import VideoReader, cpu

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../slowfast'))

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.datasets import decoder
from slowfast.datasets import utils as dataset_utils
from slowfast.utils.parser import load_config
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du

# ========== 检测参数常量 ==========
NUM_ENSEMBLE_VIEWS = 4           # 单GPU 默认集成视图数
NUM_SPATIAL_CROPS = 3            # 空间裁剪数（左/中/右）
RNG_SEED = 6666                  # 随机种子
SAMPLING_RATE = 16               # 时间采样率（覆盖 8*16=128 帧）
SPATIAL_SIZE = 224               # 空间裁剪尺寸
FACE_CASCADE_SCALE = 1.1         # Haar 级联缩放因子
FACE_CASCADE_NEIGHBORS = 5       # Haar 级联最小邻居数
FACE_CASCADE_MIN_SIZE = (30, 30) # Haar 级联最小人脸尺寸
VIEWS_PER_GPU = 76               # 多GPU模式下每卡视图数

# 使用OpenCV人脸检测（避免TensorFlow与PyTorch CUDA冲突）
_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


@dataclass
class EyeDetectionInput:
    """闭眼检测输入参数"""
    video_path: str              # 视频文件路径
    video_st_time: int          # 视频开始时间（秒级Unix时间戳）
    video_ed_time: int          # 视频结束时间（秒级Unix时间戳）
    event_st_time: int          # 事件开始时间（毫秒级Unix时间戳）
    event_ed_time: int          # 事件结束时间（毫秒级Unix时间戳）


@dataclass
class EyeDetectionOutput:
    """闭眼检测输出结果"""
    result: str                  # 'eye_close' 或 'eye_open'
    confidence: float           # 置信度分数 [0-1]
    success: bool               # 检测是否成功
    error_message: str | None = None


class EyeDetectionAPI:
    """
    闭眼检测API - 简洁版本

    提供简洁的API接口用于视频闭眼检测
    """

    def __init__(self,
                 config_path: str,
                 checkpoint_path: str,
                 device: str = 'cpu',
                 enable_face_extraction: bool = True,
                 face_pad_ratio: float = 0.5):
        """
        初始化闭眼检测API

        Args:
            config_path: 配置文件路径（config.yaml）
            checkpoint_path: 模型checkpoint路径
            device: 计算设备 ('cuda:0' 或 'cpu')
            enable_face_extraction: 是否启用人脸提取
            face_pad_ratio: 人脸区域扩展比例（默认0.5表示扩展50%）
        """
        self.device = device
        self.enable_face_extraction = enable_face_extraction
        self.face_pad_ratio = face_pad_ratio


        # 设置 decord bridge 为 torch 模式（全局只设置一次）
        if decord.bridge.current_bridge() != 'torch':
            decord.bridge.set_bridge('torch')

        # 加载配置
        self.cfg = self._load_config(config_path, checkpoint_path)

        # 初始化模型
        self.model = self._build_model()

    def _load_config(self, config_path: str, checkpoint_path: str):
        """加载配置文件"""
        # 创建简单的配置对象
        device = self.device  # 保存外部device变量

        class Args:
            def __init__(self):
                self.cfg_file = config_path
                self.opts = [
                    "NUM_GPUS", "1",
                    "NUM_SHARDS", "1",
                    "TRAIN.ENABLE", "False",
                    "TEST.ENABLE", "True",
                    "TEST.NUM_ENSEMBLE_VIEWS", str(NUM_ENSEMBLE_VIEWS),
                    "TEST.NUM_SPATIAL_CROPS", str(NUM_SPATIAL_CROPS),
                    "TEST.TEST_BEST", "True",
                    "TEST.ADD_SOFTMAX", "True",
                    "TEST.CHECKPOINT_FILE_PATH", checkpoint_path,
                    "RNG_SEED", str(RNG_SEED),
                    "BALANCE_DATA", "False",
                    "DEEPFACE", "False",
                    "INPUT_CLIP", "False",
                    "VISUAL", "False",
                    "LAST_FRAMES", "False",
                    "EVALUATE", "False"
                ]

        args = Args()
        cfg = load_config(args)
        cfg = assert_and_infer_cfg(cfg)
        return cfg

    def _build_model(self):
        """构建并加载模型"""
        # 初始化分布式训练环境（单机单卡模式）
        du.init_distributed_training(self.cfg)

        # 构建模型（CPU上构建）
        from slowfast.models.build import MODEL_REGISTRY
        model = MODEL_REGISTRY.get(self.cfg.MODEL.MODEL_NAME)(self.cfg)

        # 先在CPU上加载checkpoint
        cu.load_test_checkpoint(self.cfg, model)

        # 设置为评估模式
        model.eval()

        # 最后转移到GPU
        if self.cfg.NUM_GPUS:
            model = model.to(self.device)

        return model

    def _detect_face_bbox(self, frames_np: np.ndarray) -> tuple | None:
        """从帧中检测人脸bbox（只检测，不裁剪）。"""
        face_cascade = _get_face_cascade()
        T = frames_np.shape[0]

        for i in range(T):
            frame = frames_np[i]
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=FACE_CASCADE_SCALE,
                minNeighbors=FACE_CASCADE_NEIGHBORS,
                minSize=FACE_CASCADE_MIN_SIZE
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad_w = int(w * self.face_pad_ratio)
                pad_h = int(h * self.face_pad_ratio)
                return (x, y, w, h, pad_w, pad_h)
        return None

    def _apply_face_crop(self, frames: torch.Tensor, bbox: tuple) -> torch.Tensor:
        """使用给定bbox裁剪人脸区域。"""
        T, H, W, C = frames.shape
        frames_np = frames.numpy()
        x, y, w, h, pad_w, pad_h = bbox

        new_x = max(0, x - pad_w)
        new_y = max(0, y - pad_h)
        new_w = min(W - new_x, w + 2 * pad_w)
        new_h = min(H - new_y, h + 2 * pad_h)

        processed = []
        for i in range(T):
            frame = frames_np[i]
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_region = frame_bgr[int(new_y):int(new_y+new_h), int(new_x):int(new_x+new_w)]
            resized = cv2.resize(face_region, (W, H))
            processed.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        return torch.from_numpy(np.array(processed))

    def _extract_face_region(self, frames: torch.Tensor, cached_bbox: tuple | None = None) -> tuple[torch.Tensor, tuple | None]:
        """
        从视频帧中提取人脸区域

        Args:
            frames: 视频帧张量，形状为 (T, H, W, C)
            cached_bbox: 缓存的人脸bbox，如果提供则跳过检测

        Returns:
            (处理后的帧张量, 人脸bbox)
        """
        if not self.enable_face_extraction:
            return frames, None

        if cached_bbox is not None:
            return self._apply_face_crop(frames, cached_bbox), cached_bbox

        frames_np = frames.numpy()
        bbox = self._detect_face_bbox(frames_np)

        if bbox is not None:
            return self._apply_face_crop(frames, bbox), bbox

        # 未检测到人脸，返回原帧
        return frames, None

    def _calculate_frame_indices(self,
                                 video_st_time: int,
                                 video_ed_time: int,
                                 event_st_time: int,
                                 event_ed_time: int,
                                 fps: float) -> tuple[int | None, int | None]:
        """
        计算事件对应的帧索引

        Args:
            video_st_time: 视频开始时间（秒）
            video_ed_time: 视频结束时间（秒）
            event_st_time: 事件开始时间（毫秒）
            event_ed_time: 事件结束时间（毫秒）
            fps: 视频帧率

        Returns:
            (start_frame, end_frame) 或 (None, None)
        """
        # 统一时间单位为毫秒
        video_start_ms = video_st_time * 1000
        video_end_ms = video_ed_time * 1000
        video_duration_ms = video_end_ms - video_start_ms

        # 计算事件在视频中的时间偏移
        start_offset_ms = event_st_time - video_start_ms
        end_offset_ms = event_ed_time - video_start_ms

        # 检查事件是否在视频时间范围内
        if start_offset_ms < 0 or end_offset_ms > video_duration_ms:
            return None, None

        event_st_sec = start_offset_ms / 1000
        event_ed_sec = end_offset_ms / 1000

        start_frame = int(event_st_sec * fps)
        end_frame = int(event_ed_sec * fps)

        return start_frame, end_frame

    @torch.no_grad()
    def detect(self, input_params: EyeDetectionInput,
               view_start: int | None = None,
               view_end: int | None = None) -> EyeDetectionOutput:
        """
        执行闭眼检测（主接口）

        Args:
            input_params: 检测输入参数
            view_start: 起始视图索引（用于多GPU拆分），None表示从0开始
            view_end: 结束视图索引（不含），None表示到最后一个

        Returns:
            EyeDetectionOutput: 检测结果
        """
        try:
            # 检查视频文件是否存在
            if not os.path.exists(input_params.video_path):
                return EyeDetectionOutput(
                    result='error',
                    confidence=0.0,
                    success=False,
                    error_message=f"视频文件不存在: {input_params.video_path}"
                )

            # 打开视频文件
            video_container = VideoReader(input_params.video_path, ctx=cpu(0))

            total_frame = len(video_container)
            fps = video_container.get_avg_fps()

            # 计算帧索引
            start_frame, end_frame = self._calculate_frame_indices(
                input_params.video_st_time,
                input_params.video_ed_time,
                input_params.event_st_time,
                input_params.event_ed_time,
                fps
            )

            if start_frame is None or end_frame is None:
                return EyeDetectionOutput(
                    result='error',
                    confidence=0.0,
                    success=False,
                    error_message="事件时间超出视频时间范围"
                )

            # 多视图batch推理：遍历所有时间视图和空间裁剪
            num_views = self.cfg.TEST.NUM_ENSEMBLE_VIEWS
            num_crops = self.cfg.TEST.NUM_SPATIAL_CROPS
            sampling_rate = dataset_utils.get_random_sampling_rate(0, SAMPLING_RATE)
            batch_inputs = []
            face_bbox_cache = None

            v_start = view_start if view_start is not None else 0
            v_end = view_end if view_end is not None else num_views

            for temporal_idx in range(v_start, v_end):
                frames = decoder.decode(
                    video_container,
                    sampling_rate,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_idx,
                    num_views,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=SPATIAL_SIZE,
                    use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                    sparse=True,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    video_frame_count=total_frame,
                    INPUT_CLIP=False,
                    VISUAL=False,
                    EVALUATE=False,
                    LAST_FRAMES=False
                )

                if self.enable_face_extraction:
                    frames, face_bbox_cache = self._extract_face_region(frames, face_bbox_cache)

                frames = dataset_utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                frames = frames.permute(3, 0, 1, 2)

                for spatial_idx in range(num_crops):
                    spatial_frames = dataset_utils.spatial_sampling(
                        frames,
                        spatial_idx=spatial_idx,
                        min_scale=SPATIAL_SIZE,
                        max_scale=SPATIAL_SIZE,
                        crop_size=SPATIAL_SIZE,
                        random_horizontal_flip=False,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )
                    packed = dataset_utils.pack_pathway_output(self.cfg, spatial_frames)[0]
                    batch_inputs.append(packed)

            # 堆叠成batch：[num_views * num_crops, C, T, H, W]
            batch_tensor = torch.stack(batch_inputs, dim=0)
            del batch_inputs
            inputs = [batch_tensor]

            # 转移到GPU
            if self.cfg.NUM_GPUS:
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(self.device, non_blocking=True)

            # 模型推理
            preds = self.model(inputs).softmax(-1)
            preds = preds.cpu()

            # 所有视图的预测取平均
            avg_probs = preds.mean(dim=0).numpy()
            pred_id = int(avg_probs.argmax())
            confidence = float(avg_probs[pred_id])
            result = 'eye_close' if pred_id == 1 else 'eye_open'

            return EyeDetectionOutput(
                result=result,
                confidence=confidence,
                success=True,
                error_message=None
            )

        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            return EyeDetectionOutput(
                result='error',
                confidence=0.0,
                success=False,
                error_message=error_details
            )


def create_eye_detector(config_path: str,
                       checkpoint_path: str,
                       device: str = 'cuda:0',
                       enable_face_extraction: bool = True,
                       face_pad_ratio: float = 0.5) -> EyeDetectionAPI:
    """
    创建闭眼检测器实例的便捷函数（单GPU版本）

    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型checkpoint路径
        device: 计算设备
        enable_face_extraction: 是否启用人脸提取
        face_pad_ratio: 人脸区域扩展比例

    Returns:
        EyeDetectionAPI: 检测器实例
    """
    return EyeDetectionAPI(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        enable_face_extraction=enable_face_extraction,
        face_pad_ratio=face_pad_ratio
    )


class MultiGPUEyeDetector:
    """
    多GPU并行闭眼检测器

    在每个可用GPU上创建独立的检测器实例，通过线程池并行调度，
    自动将检测任务分配到空闲GPU上，最大化吞吐量。
    """

    def __init__(self,
                 config_path: str,
                 checkpoint_path: str,
                 gpu_ids: list[int] | None = None,
                 enable_face_extraction: bool = True,
                 face_pad_ratio: float = 0.5):
        """
        初始化多GPU检测器

        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型checkpoint路径
            gpu_ids: 使用的GPU ID列表，None则使用所有可用GPU
            enable_face_extraction: 是否启用人脸提取
            face_pad_ratio: 人脸区域扩展比例
        """
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))

        total_views = VIEWS_PER_GPU * len(gpu_ids)

        self.detectors = []
        for gpu_id in gpu_ids:
            detector = EyeDetectionAPI(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device=f'cuda:{gpu_id}',
                enable_face_extraction=enable_face_extraction,
                face_pad_ratio=face_pad_ratio
            )
            # 覆盖为多GPU优化的视图数
            detector.cfg.TEST.NUM_ENSEMBLE_VIEWS = total_views
            self.detectors.append(detector)

        self._current_idx = 0
        self._lock = threading.Lock()
        self._total_views = total_views

    def detect(self, input_params: EyeDetectionInput) -> EyeDetectionOutput:
        """
        执行闭眼检测（自动拆分视图到多GPU并行推理）

        Args:
            input_params: 检测输入参数

        Returns:
            EyeDetectionOutput: 检测结果
        """
        num_gpus = len(self.detectors)
        if num_gpus <= 1:
            return self.detectors[0].detect(input_params)

        num_views = self.detectors[0].cfg.TEST.NUM_ENSEMBLE_VIEWS
        # 拆分视图范围到各GPU
        chunk_size = num_views // num_gpus
        ranges = []
        for i in range(num_gpus):
            start = i * chunk_size
            end = start + chunk_size if i < num_gpus - 1 else num_views
            ranges.append((start, end))

        # 并行推理
        partial_results = [None] * num_gpus
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_idx = {}
            for i, (start, end) in enumerate(ranges):
                future = executor.submit(
                    self.detectors[i].detect, input_params, start, end
                )
                future_to_idx[future] = i
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                partial_results[idx] = future.result()

        # 合并结果：平均所有GPU的预测概率
        all_probs = []
        for r in partial_results:
            if not r.success:
                return r
            # 从confidence和result反推概率
            if r.result == 'eye_close':
                all_probs.append([1 - r.confidence, r.confidence])
            else:
                all_probs.append([r.confidence, 1 - r.confidence])

        avg_probs = np.mean(all_probs, axis=0)
        pred_id = int(avg_probs.argmax())
        confidence = float(avg_probs[pred_id])
        result = 'eye_close' if pred_id == 1 else 'eye_open'

        return EyeDetectionOutput(
            result=result,
            confidence=confidence,
            success=True,
            error_message=None
        )

    def detect_batch(self, inputs: list[EyeDetectionInput]) -> list[EyeDetectionOutput]:
        """
        批量并行检测（多GPU并行）

        Args:
            inputs: 检测输入参数列表

        Returns:
            检测结果列表
        """
        results = [None] * len(inputs)
        with ThreadPoolExecutor(max_workers=len(self.detectors)) as executor:
            future_to_idx = {}
            for i, inp in enumerate(inputs):
                with self._lock:
                    idx = self._current_idx
                    self._current_idx = (self._current_idx + 1) % len(self.detectors)
                future = executor.submit(self.detectors[idx].detect, inp)
                future_to_idx[future] = i
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results


def create_multi_gpu_detector(config_path: str,
                              checkpoint_path: str,
                              gpu_ids: list[int] | None = None,
                              enable_face_extraction: bool = True,
                              face_pad_ratio: float = 0.5) -> MultiGPUEyeDetector:
    """
    创建多GPU并行检测器的便捷函数

    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型checkpoint路径
        gpu_ids: GPU ID列表，None则使用所有可用GPU
        enable_face_extraction: 是否启用人脸提取
        face_pad_ratio: 人脸区域扩展比例

    Returns:
        MultiGPUEyeDetector: 多GPU检测器实例
    """
    return MultiGPUEyeDetector(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        gpu_ids=gpu_ids,
        enable_face_extraction=enable_face_extraction,
        face_pad_ratio=face_pad_ratio
    )


# 使用示例
if __name__ == '__main__':
    # 配置路径
    config_path = "exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml"
    checkpoint_path = "best.pyth"

    # 创建检测器
    detector = create_eye_detector(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device='cpu',
        enable_face_extraction=True,
        face_pad_ratio=0.5
    )

    # 准备输入参数
    input_params = EyeDetectionInput(
        video_path="test.mp4",
        video_st_time=1754610861,
        video_ed_time=1754610872,
        event_st_time=1754610863790,
        event_ed_time=1754610865690
    )

    # 执行检测
    result = detector.detect(input_params)

    # 输出结果
    if result.success:
        print(f"检测结果: {result.result}")
        print(f"置信度: {result.confidence:.4f}")
    else:
        print(f"检测失败: {result.error_message}")
