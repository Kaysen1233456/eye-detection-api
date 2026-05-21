NUM_SHARDS=1
NUM_GPUS=2
BATCH_SIZE=64
BASE_LR=2e-5  # 2e-5
PYTHONPATH=$PYTHONPATH:./slowfast \
CUDA_VISIBLE_DEVICES=0,1 python tools/run_net_multi_node.py \
  --init_method tcp://localhost:10125 \
  --cfg exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR_TRAIN /yszhuo/projects/video_fatigue/workspace1_DFER-CLIP/data/20250618/process_json/face_clip/dms_eyes_closed_and_non_fatigue_looking_down_train_v1.json \
  DATA.PATH_TO_DATA_DIR_VAL /yszhuo/projects/video_fatigue/workspace1_DFER-CLIP/data/20250618/process_json/face_clip/dms_eyes_closed_and_non_fatigue_looking_down_val_balance.json \
  DATA.PATH_PREFIX you_data_path/mit \
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 100 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.SAVE_LATEST False \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 24 \
  SOLVER.BASE_LR $BASE_LR \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 5. \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.TEST_BEST True \
  TEST.ADD_SOFTMAX True \
  TEST.BATCH_SIZE 128 \
  RNG_SEED 6666 \
  BALANCE_DATA True \
  DEEPFACE True \
  INPUT_CLIP True \
  VISUAL False \
  LAST_FRAMES False \
  EVALUATE False \
  REMOVE_MODEL_PREDICT True \
  OUTPUT_DIR ./result/dms_face_clip-k400_k710_pretrain_b16_f8_224-model_filter_data-2class
