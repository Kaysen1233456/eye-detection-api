import time
import torch
from eye_detect_api import (
    EyeDetectionInput,
    create_eye_detector, create_multi_gpu_detector
)

config_path = "exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml"
checkpoint_path = "best.pyth"
input_params = EyeDetectionInput(
    video_path="test.mp4",
    video_st_time=1754610861,
    video_ed_time=1754610872,
    event_st_time=1754610863790,
    event_ed_time=1754610865690
)

print(f"CUDA devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"  GPU {i}: {name}, {mem:.1f}GB")

# ========== 单GPU ==========
print("\n=== Single GPU (cuda:0) ===")
single = create_eye_detector(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    device='cuda:0',
    enable_face_extraction=True,
    face_pad_ratio=0.5
)
num_views = single.cfg.TEST.NUM_ENSEMBLE_VIEWS
num_crops = single.cfg.TEST.NUM_SPATIAL_CROPS
print(f"  Views={num_views}, Crops={num_crops}, Batch={num_views*num_crops}")

single.detect(input_params)  # warmup
times = []
for i in range(3):
    t0 = time.perf_counter()
    r = single.detect(input_params)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"  Run {i+1}: {t1-t0:.1f}s, {r.result}, conf={r.confidence:.4f}")
avg_single = sum(times) / len(times)
print(f"  Single GPU avg: {avg_single:.1f}s")

del single
torch.cuda.empty_cache()

# ========== 双GPU并行 ==========
if torch.cuda.device_count() > 1:
    print(f"\n=== Multi GPU ({torch.cuda.device_count()} GPUs, batch split) ===")
    multi = create_multi_gpu_detector(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        gpu_ids=None,
        enable_face_extraction=True,
        face_pad_ratio=0.5
    )
    total_views = multi._total_views
    print(f"  Total batch: {total_views*num_crops}, split across {torch.cuda.device_count()} GPUs")

    num_gpus = torch.cuda.device_count()
    multi.detect(input_params)  # warmup
    for i in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(i)
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        r = multi.detect(input_params)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  Run {i+1}: {t1-t0:.1f}s, {r.result}, conf={r.confidence:.4f}")
    avg_multi = sum(times) / len(times)
    print(f"  Multi GPU avg: {avg_multi:.1f}s")

    total_vram = 0
    for i in range(num_gpus):
        mem = torch.cuda.max_memory_allocated(i) / 1024**3
        total_vram += mem
        total_gpu_mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f"  GPU {i} VRAM peak: {mem:.1f}GB ({mem/total_gpu_mem*100:.0f}%)")
    print(f"  Total VRAM: {total_vram:.1f}GB")
