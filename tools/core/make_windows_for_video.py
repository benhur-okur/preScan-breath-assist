# tools/make_windows_for_video.py
from __future__ import annotations

import os
import argparse
from typing import List, Dict, Tuple

import cv2
import pandas as pd


def get_video_info(video_path: str) -> Tuple[int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if n_frames <= 0:
        raise RuntimeError(f"Could not read frame count from: {video_path}")
    if fps <= 0:
        # Some codecs return 0; we still allow it, but min-hold-sec needs fps.
        fps = 0.0

    return n_frames, fps


def build_windows(
    video_path: str,
    window_frames: int,
    stride_frames: int,
    fold_id: str = "inference",
    split: str = "test",
) -> List[Dict]:
    n_frames, fps = get_video_info(video_path)

    rows: List[Dict] = []
    start = 0
    last_start = max(0, n_frames - window_frames)

    while start <= last_start:
        rows.append(
            {
                "front_video_path": video_path.replace("\\", "/"),
                "start_frame": int(start),
                "window_frames": int(window_frames),

                # label unknown for inference; keep placeholder
                "label": 0,
                "person": "inference",
                "case": os.path.basename(video_path),
                "fold_id": fold_id,
                "split": split,

                # NEW: for correct time-based postprocess
                "stride_frames": int(stride_frames),
                "fps": float(fps),
            }
        )
        start += stride_frames

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to front video (mp4)")
    ap.add_argument("--window-frames", type=int, default=15)
    ap.add_argument("--stride-frames", type=int, default=5)
    ap.add_argument("--out-csv", default="data/windows/infer_manifest.csv")
    ap.add_argument("--fold-id", default="inference")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rows = build_windows(
        video_path=args.video,
        window_frames=int(args.window_frames),
        stride_frames=int(args.stride_frames),
        fold_id=str(args.fold_id),
        split=str(args.split),
    )
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] wrote {len(df)} windows -> {args.out_csv}")
    if "fps" in df.columns:
        print(f"[INFO] fps={df['fps'].iloc[0]} stride_frames={df['stride_frames'].iloc[0]}")


if __name__ == "__main__":
    main()
