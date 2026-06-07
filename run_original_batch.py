from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from eye_detect_api import EyeDetectionInput, create_eye_detector


ROOT = Path("/yanglin/eye_detect")
EXCEL_PATH = ROOT / "高风险0512-0520(1).xlsx"
OUT_CSV = ROOT / "original_batch_8frames.csv"

VIDEO_RE = re.compile(
    r"^(?P<plate>.+?)_(?P<st>\d+)_(?P<ed>\d+)_(?P<event>\d+)__.*_channel7\.mp4$"
)


def build_folder(row: pd.Series) -> str:
    return f"{str(row['车牌号']).strip()}_{str(row['视频日期']).strip().replace('-', '_')}_{int(row['alarmId'])}"


def parse_video_name(path: str) -> dict[str, int]:
    m = VIDEO_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"bad video name: {path}")
    return {k: int(v) if k != "plate" else v for k, v in m.groupdict().items()}


def main() -> None:
    df = pd.read_excel(EXCEL_PATH)
    detector = create_eye_detector(
        config_path=str(ROOT / "exp/humanfactor/dms+k400_k710_b16_f8x224/config.yaml"),
        checkpoint_path=str(ROOT / "best.pyth"),
        device="cuda:0",
        enable_face_extraction=True,
        face_pad_ratio=0.5,
    )

    rows = []
    for _, row in df.iterrows():
        folder = ROOT / "close" / build_folder(row)
        videos = sorted(folder.glob("*_channel7.mp4")) if folder.exists() else []
        if not videos:
            continue

        video_path = str(videos[0])
        info = parse_video_name(video_path)
        fps = 20.0
        end_frame = int((info["event"] - info["st"]) * fps)
        start_frame = max(0, end_frame - 7)
        start_ts = int(info["st"] * 1000 + (start_frame / fps) * 1000)
        end_ts = int(info["st"] * 1000 + (end_frame / fps) * 1000)

        inp = EyeDetectionInput(
            video_path=video_path,
            video_st_time=info["st"],
            video_ed_time=info["ed"],
            event_st_time=start_ts,
            event_ed_time=end_ts,
        )
        result = detector.detect(inp)

        rows.append(
            {
                "车牌号": row["车牌号"],
                "alarmId": int(row["alarmId"]),
                "报警事件": row["报警事件"],
                "视频路径": video_path,
                "事件结束帧": end_frame,
                "检测窗口": f"{start_frame}-{end_frame}",
                "闭眼开始帧": start_frame,
                "闭眼开始时间戳": start_ts,
                "检测结果": result.result,
                "检测置信度": result.confidence,
                "success": result.success,
                "错误信息": result.error_message or "",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved {OUT_CSV}")
    print(f"rows {len(out)}")


if __name__ == "__main__":
    main()
