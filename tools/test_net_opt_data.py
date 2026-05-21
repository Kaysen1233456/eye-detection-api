#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
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
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import json

logger = logging.get_logger(__name__)

def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def save_json(data_path, data):
    with open(data_path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, sort_keys=True, ensure_ascii=False)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
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
    test_meter.iter_tic()
    for cur_iter, (inputs, labels, video_idx, meta) in tqdm(enumerate(test_loader), total=len(test_loader)):
        meta = {}
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
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
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()


    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with g_pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

        cm = confusion_matrix(test_meter.video_labels, test_meter.video_preds.argmax(dim=1))
        np.set_printoptions(precision=4)
        normalized_cm_row = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        normalized_cm_row = normalized_cm_row * 100
        normalized_cm_column = cm.astype('float') / (cm.sum(axis=0)[np.newaxis,:] + 1e-7)
        normalized_cm_column = normalized_cm_column * 100
        list_diag = np.diag(normalized_cm_row)
        uar = list_diag.mean()
        
        for i,anno in enumerate(test_loader.dataset.__annotations__):
            model_preds =  all_preds[i] / test_loader.dataset._num_clips
            anno['model_preds'] = [round(float(p), 4) for p in model_preds.numpy().tolist()]
        save_json('/yszhuo/projects/video_fatigue/workspace1_DFER-CLIP/data/20250618/process_json/face_clip/dms_eyes_closed_and_non_fatigue_looking_down_train_v1.json',
                  test_loader.dataset.__annotations__)

        logger.info("\n" + "="*60 + " Evaluation Summary " + "="*60)
        logger.info("Confusion Matrix:\n%s", cm)
        logger.info("Confusion Matrix Norm Row:\n%s", normalized_cm_row)
        logger.info("Confusion Matrix Norm Column:\n%s", normalized_cm_column)
        logger.info("UAR: %0.2f" % uar)
            
    test_meter.finalize_metrics(ks=(1, 2))
    return test_meter


def test_opt_data(cfg):
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
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))
    logger.info(f"Add softmax after prediction: {cfg.TEST.ADD_SOFTMAX}")

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    # if writer is not None:
    #     writer.close()

    
    # file_name = f'{cfg.DATA.NUM_FRAMES}x{cfg.DATA.TEST_CROP_SIZE}x{cfg.TEST.NUM_ENSEMBLE_VIEWS}x{cfg.TEST.NUM_SPATIAL_CROPS}.pkl'
    # with g_pathmgr.open(os.path.join(
    #     cfg.OUTPUT_DIR, file_name),
    #     'wb'
    # ) as f:
    #     result = {
    #         'video_preds': test_meter.video_preds.cpu().numpy(),
    #         'video_labels': test_meter.video_labels.cpu().numpy()
    #     }
    #     pickle.dump(result, f)