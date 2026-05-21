#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import json
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from torchvision import transforms
from collections import Counter

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = logging.get_logger(__name__)


def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def save_json(data_path, data):
    with open(data_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, sort_keys=True, ensure_ascii=False)

@DATASET_REGISTRY.register()
class Hf(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
            cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
            cfg.TEST.NUM_SPATIAL_CROPS = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing HF {}...".format(mode))
        self._construct_loader()
        
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # path_to_file = os.path.join(
        #     self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        # )
        if self.mode == 'train':
            path_to_file =self.cfg.DATA.PATH_TO_DATA_DIR_TRAIN
        elif self.mode == 'val' or self.mode == 'test':
            path_to_file = self.cfg.DATA.PATH_TO_DATA_DIR_VAL

        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._path_to_videos_origin = []
        self._labels = []
        self._spatial_temporal_idx = []
        
        ###########################   【20250701】by YiShen, for annotation process ###########################
        self._start_frames = []
        self._end_frames = []
        self._video_frame_count = []
        annotation = load_json(path_to_file)
        labels = [clip_info['train_label'] for clip_info in annotation]
        # 统计标签
        label_counts = Counter(labels)
        print("原始样本分布:", label_counts)

        ######################## model predict remove  ###########################
        # 修改后的代码
        if self.cfg.REMOVE_MODEL_PREDICT and self.mode == 'train':
            assert 'model_preds' in annotation[0]
            # 仅保留模型预测最大值对应类别与标注标签一致的样本
            filtered_annotation = []
            
            for anno in annotation:
                model_preds = anno['model_preds']
                train_label = anno['train_label']
                if max(model_preds) < 0.9:
                    continue
                # 获取模型预测的最大概率索引
                predicted_class = model_preds.index(max(model_preds))
                
                # 仅当预测类别与标注标签一致时保留
                if predicted_class == train_label:
                    filtered_annotation.append(anno)
            
            annotation = filtered_annotation
            labels = [clip_info['train_label'] for clip_info in annotation]
            label_counts = Counter(labels)
            print("模型过滤后样本分布:", label_counts)
        ########################################################################


        if self.cfg.BALANCE_DATA and self.mode == 'train':
            max_count = max(label_counts.values())
            balanced_list = []

            for label, count in label_counts.items():
                samples = [clip for clip in annotation if clip['train_label'] == label]
                if count < max_count:
                    # 复制少数类样本
                    need_copy = max_count - count
                    copied_samples = random.choices(samples, k=need_copy)
                    samples.extend(copied_samples)
                balanced_list.extend(samples)

            # 更新数据
            annotation = balanced_list
            labels = [clip['train_label'] for clip in annotation]
            label_counts = Counter(labels)
            print("平衡后样本分布:", label_counts)
        
        if self.cfg.DEEPFACE:
            assert 'face_clip_path' in annotation[0]
            annotation = [anno for anno in annotation if os.path.exists(anno['face_clip_path'])]
        

        
        self.__annotations__ = annotation
        
        for clip_idx, anno in enumerate(annotation):
            
            label = anno['train_label']
            start_frame, end_frame = anno['start_frame'], anno['end_frame']
            
            # # 86041测试
            # if anno['label'] in ['疲劳_近似人工', '疲劳_人工干预唤醒', '疲劳_TTS干预', '疲劳_人工干预停车']:
            #     label = 1
            # else:
            #     label = 0
            # start_frame, end_frame = 0, anno['video_frame_count']-1
            
            
            if self.cfg.DEEPFACE:
                # if "mta_mid_minus_64_cut" in path:
                #     path = path.replace("mta_mid_minus_64_cut", "mta_mid_minus_64_cut_face_data").replace(".mp4", ".npy")
                # elif "chuzu_mid_minus_64_cut" in path:
                #     path = path.replace("chuzu_mid_minus_64_cut", "chuzu_mid_minus_64_cut_face_data").replace(".mp4", ".npy")
                path = anno['face_clip_path']
            else:
                path = anno['video_path']
                    
            
            for idx in range(self._num_clips):
                self._path_to_videos_origin.append(anno['video_path'])
                self._path_to_videos.append(path)
                self._labels.append(int(label))
                self._spatial_temporal_idx.append(idx)
                self._video_meta[clip_idx * self._num_clips + idx] = {}
                self._start_frames.append(start_frame)
                self._end_frames.append(end_frame)
                self._video_frame_count.append(anno['video_frame_count'])
                
        ############################################################################################################
        
        # with g_pathmgr.open(path_to_file, "r") as f:
        #     for clip_idx, path_label in enumerate(f.read().splitlines()):
        #         assert (
        #             len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
        #             == 2
        #         )
        #         path, label = path_label.split(
        #             self.cfg.DATA.PATH_LABEL_SEPARATOR
        #         )
                
        #         if self.cfg.DEEPFACE:
        #             if "mta_mid_minus_64_cut" in path:
        #                 path = path.replace("mta_mid_minus_64_cut", "mta_mid_minus_64_cut_face_data").replace(".mp4", ".npy")
        #             elif "chuzu_mid_minus_64_cut" in path:
        #                 path = path.replace("chuzu_mid_minus_64_cut", "chuzu_mid_minus_64_cut_face_data").replace(".mp4", ".npy")
                
        #         for idx in range(self._num_clips):
        #             # self._path_to_videos.append(
        #             #     os.path.join(self.cfg.DATA.PATH_PREFIX, path)
        #             # )
        #             self._path_to_videos.append(path)
        #             self._labels.append(int(label))
        #             self._spatial_temporal_idx.append(idx)
        #             self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Hf split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing Hf dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["val", "test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            # min_scale, max_scale, crop_size = (
            #     [self.cfg.DATA.TEST_CROP_SIZE] * 3
            #     if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
            #     else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
            #     + [self.cfg.DATA.TEST_CROP_SIZE]
            # )
            # # The testing is deterministic and no jitter should be performed.
            # # min_scale, max_scale, and crop_size are expect to be the same.
            # assert len({min_scale, max_scale}) == 1

            min_scale, max_scale, crop_size = ([self.cfg.DATA.TEST_CROP_SIZE] * 3)
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                    self.cfg.DATA.DECODING_BACKEND,
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                logger.warning(
                    "Failed to load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                elif self.mode in ["test"] and i_try > self._num_retries // 2:
                    # BUG: should not repeat video
                    logger.info(
                        "Failed to load video idx {} from {}; use idx {}".format(
                            index, self._path_to_videos[index], index - 1
                        )
                    )
                    index = index - 1
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=self.cfg.DATA.TARGET_FPS,
                backend=self.cfg.DATA.DECODING_BACKEND,
                max_spatial_scale=min_scale,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                sparse=True,
                start_frame=self._start_frames[index],
                end_frame=self._end_frames[index],
                video_frame_count=self._video_frame_count[index],
                INPUT_CLIP=self.cfg.INPUT_CLIP,
                VISUAL=self.cfg.VISUAL,
                EVALUATE=self.cfg.EVALUATE,
                LAST_FRAMES=self.cfg.LAST_FRAMES
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:

                    frame_list = []
                    label_list = []
                    index_list = []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                        label = self._labels[index]
                        new_frames = utils.pack_pathway_output(
                            self.cfg, new_frames
                        )
                        frame_list.append(new_frames)
                        label_list.append(label)
                        index_list.append(index)
                    return frame_list, label_list, index_list, {}

                else:
                    frames = self._aug_frame(
                        frames,
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    )

            else:
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

            label = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)
            # return frames, label, index, {}
            return frames, label, index, self._path_to_videos_origin[index]
        else:
            raise RuntimeError(
                "Failed to load video idx {} from {} after {} retries".format(
                    index, self._path_to_videos[index], self._num_retries
                )
            )

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
