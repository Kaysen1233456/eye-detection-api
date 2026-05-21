NUM_SHARDS=1
NUM_GPUS=1
BATCH_SIZE=16
BASE_LR=1e-5
CHECKPOINT_FILE_PATH=/yszhuo/projects/video_fatigue/workspace2_UniFormerV2/result/dms_face_clip-k400_k710_pretrain_b16_f8_224-2class/best.pyth
PYTHONPATH=$PYTHONPATH:./slowfast \
CUDA_VISIBLE_DEVICES=0 python tools/run_net_multi_node.py \
  --init_method tcp://localhost:10126 \
  --cfg exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR_TRAIN /yszhuo/projects/video_fatigue/workspace1_DFER-CLIP/data/20250618/process_json/face_clip/dms_eyes_closed_and_non_fatigue_looking_down_train.json \
  DATA.PATH_TO_DATA_DIR_VAL /yszhuo/projects/video_fatigue/workspace2_UniFormerV2/result/test/dms_face_clip-k400_k710_pretrain_b16_f8_224-2class/86041/unsuccess-2025-07-11_15-46-12.json \
  DATA.PATH_PREFIX you_data_path/k400 \
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 100 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.SAVE_LATEST False \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 55 \
  SOLVER.BASE_LR $BASE_LR \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 5. \
  TRAIN.ENABLE False \
  TEST.ENABLE False \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  TEST.TEST_BEST True \
  TEST.ADD_SOFTMAX True \
  TEST.BATCH_SIZE $BATCH_SIZE \
  TEST.CHECKPOINT_FILE_PATH $CHECKPOINT_FILE_PATH \
  RNG_SEED 6666 \
  BALANCE_DATA False \
  DEEPFACE True \
  INPUT_CLIP False \
  VISUAL True \
  LAST_FRAMES False\
  OUTPUT_DIR ./result/visual/dms_face_clip-k400_k710_pretrain_b16_f8_224-2class/86041_face_unsuccess-2025-07-11_15-46-12 \
  # OUTPUT_DIR ./result/test/dms_face_clip-k400_k710_pretrain_b16_f8_224-2class/86041