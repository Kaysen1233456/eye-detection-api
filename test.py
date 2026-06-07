from eye_detect_api import EyeDetectionAPI, EyeDetectionInput, create_eye_detector

# 创建检测器
detector = create_eye_detector(
    config_path="exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml",
    checkpoint_path="best.pyth",
    device='cpu',
    enable_face_extraction=True,
    face_pad_ratio=0.5
)

# 准备输入参数
input_params = EyeDetectionInput(
    video_path="test.mp4",
    video_st_time=1754610861,      # 视频开始时间（秒级时间戳）
    video_ed_time=1754610872,      # 视频结束时间（秒级时间戳）
    event_st_time=1754610863790,   # 事件开始时间（毫秒级时间戳）
    event_ed_time=1754610865690    # 事件结束时间（毫秒级时间戳）
)

# 执行检测
result = detector.detect(input_params)

# 输出结果
if result.success:
    print(f"检测结果：{result.result}")      # 'eye_close' 或 'eye_open'
    print(f"置信度：{result.confidence:.4f}")
else:
    print(f"检测失败：{result.error_message}")