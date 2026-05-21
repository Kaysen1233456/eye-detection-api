export PYTHONPATH=$PYTHONPATH:/yszhuo/projects/video_fatigue/workspace2_UniFormerV2_demo
CHECKPOINT_FILE_PATH=/yszhuo/projects/video_fatigue/workspace2_UniFormerV2/result/dms_face_clip-k400_k710_pretrain_b16_f8_224-2class/best.pyth
PYTHONPATH=$PYTHONPATH:./slowfast \
CUDA_VISIBLE_DEVICES=0 python tools/run_net_multi_node.py \
  --cfg /yszhuo/projects/video_fatigue/workspace2_UniFormerV2_demo/exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml \
  --video_path /yszhuo/projects/video_fatigue/workspace2_UniFormerV2/1.mp4 \
  --video_st_time 1754610861 \
  --video_ed_time 1754610872 \
  --event_st_time 1754610865790 \
  --event_ed_time 1754610866690 \
  "NUM_GPUS" "1" \
  "NUM_SHARDS" "1" \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.TEST_BEST True \
  TEST.ADD_SOFTMAX True \
  TEST.CHECKPOINT_FILE_PATH $CHECKPOINT_FILE_PATH \
  RNG_SEED 6666 \
  BALANCE_DATA False \
  DEEPFACE True \
  INPUT_CLIP False \
  VISUAL False \
  LAST_FRAMES False \
  EVALUATE False
