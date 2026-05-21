from eye_detect_api import EyeDetectionInput, create_eye_detector

# 创建检测器 - 使用GPU
detector = create_eye_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    device='cuda:0',
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
    print(f"检测结果：{result.result}")
    print(f"置信度：{result.confidence:.4f}")
else:
    print(f"检测失败：{result.error_message}")
