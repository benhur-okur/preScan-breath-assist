import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

'''
python tools/clamp_frames.py \
  --labeled-dir data/labeled \
  --videos-root data/raw/videos \
  --out-dir data/labeled_safe

'''

CASE_TO_INDEX = {
    "normal": 1,
    "inhale_hold": 2,
    "exhale_hold": 3,
    "irregular": 4,
}


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def clamp_column(df, col, max_valid):
    before_max = int(df[col].max())
    df[col] = df[col].clip(0, max_valid)
    after_max = int(df[col].max())
    return before_max, after_max


def process_one_csv(csv_path: Path, videos_root: Path, out_root: Path):
    name = csv_path.stem.replace("_labeled", "")
    parts = name.split("_")
    person = parts[0]
    case = "_".join(parts[1:])

    if case not in CASE_TO_INDEX:
        raise RuntimeError(f"Unknown case type in filename: {csv_path.name}")

    case_idx = CASE_TO_INDEX[case]

    front_video = videos_root / person / f"{person}_{case_idx}_front.mp4"
    side_video = videos_root / person / f"{person}_{case_idx}_side.mp4"

    if not front_video.exists():
        raise FileNotFoundError(front_video)
    if not side_video.exists():
        raise FileNotFoundError(side_video)

    front_frames = get_video_frame_count(front_video)
    side_frames = get_video_frame_count(side_video)

    df = pd.read_csv(csv_path)

    report = {
        "front_clipped": False,
        "side_clipped": False,
    }

    # ---- FRONT ----
    if "front_frame" in df.columns:
        before, after = clamp_column(df, "front_frame", front_frames - 1)
        if before != after:
            report["front_clipped"] = True

    # ---- SIDE ----
    if "side_frame" in df.columns:
        before, after = clamp_column(df, "side_frame", side_frames - 1)
        if before != after:
            report["side_clipped"] = True

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / csv_path.name
    df.to_csv(out_path, index=False)

    return report, out_path


def main():
    parser = argparse.ArgumentParser(
        description="Clamp front_frame / side_frame columns to valid video frame ranges."
    )
    parser.add_argument(
        "--labeled-dir",
        type=str,
        default="data/labeled",
        help="Directory containing *_labeled.csv files",
    )
    parser.add_argument(
        "--videos-root",
        type=str,
        default="data/raw/videos",
        help="Root directory of videos",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/labeled_safe",
        help="Output directory for safe CSVs",
    )

    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    videos_root = Path(args.videos_root)
    out_dir = Path(args.out_dir)

    csv_files = sorted(labeled_dir.glob("*_labeled.csv"))

    print("=== FRAME CLAMPING ===")
    print(f"Labeled files : {len(csv_files)}")
    print(f"Videos root  : {videos_root}")
    print(f"Output dir   : {out_dir}")
    print("-" * 80)

    clipped_any = 0

    for csv_path in csv_files:
        report, out_path = process_one_csv(csv_path, videos_root, out_dir)

        if report["front_clipped"] or report["side_clipped"]:
            clipped_any += 1
            print(f"[CLAMPED] {csv_path.name}")
            if report["front_clipped"]:
                print("   - front_frame was clamped")
            if report["side_clipped"]:
                print("   - side_frame was clamped")
        else:
            print(f"[OK]      {csv_path.name}")

    print("-" * 80)
    print(f"Total files       : {len(csv_files)}")
    print(f"Files with clamp  : {clipped_any}")
    print(f"Files untouched  : {len(csv_files) - clipped_any}")
    print("\n[DONE] Safe labeled CSVs are ready for training.")


if __name__ == "__main__":
    main()
