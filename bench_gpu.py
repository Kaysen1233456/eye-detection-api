import time
from eye_detect_api import EyeDetectionInput, create_eye_detector

detector = create_eye_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    device='cuda:0',
    enable_face_extraction=True,
    face_pad_ratio=0.5
)

input_params = EyeDetectionInput(
    video_path="test.mp4",
    video_st_time=1754610861,
    video_ed_time=1754610872,
    event_st_time=1754610863790,
    event_ed_time=1754610865690
)

# 预热
print("Warmup...")
detector.detect(input_params)

# 测速
print("Benchmarking...")
times = []
for i in range(5):
    t0 = time.perf_counter()
    result = detector.detect(input_params)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    print(f"  Run {i+1}: {t1-t0:.3f}s, result={result.result}, conf={result.confidence:.3f}")

avg = sum(times) / len(times)
print(f"\nGPU avg: {avg:.3f}s per detection")
print(f"10s capacity: {int(10/avg)} detections")
