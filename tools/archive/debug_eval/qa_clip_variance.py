# tools/qa_clip_variance.py
from __future__ import annotations

import os
import argparse
import random
import numpy as np
import pandas as pd

from src.datasets.window_dataset import _read_clip_opencv  # we reuse your exact reader


def mean_abs_diff(clip: np.ndarray) -> float:
    # clip: [T,H,W,3] uint8 RGB
    clip_f = clip.astype(np.float32)
    diffs = np.abs(clip_f[1:] - clip_f[:-1]).mean(axis=(1,2,3))  # [T-1]
    return float(diffs.mean()) if diffs.size else 0.0


def identical_ratio(clip: np.ndarray) -> float:
    # ratio of consecutive frames that are exactly identical
    eq = (clip[1:] == clip[:-1]).all(axis=(1,2,3))  # [T-1]
    return float(eq.mean()) if eq.size else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-loso", required=True)
    p.add_argument("--fold-key", required=True)
    p.add_argument("--split", required=True, help="train/val/test OR fold_key depending on your manifest encoding")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root-dir", default=".")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.manifest_loso)
    df = df[(df["fold_id"] == args.fold_key) & (df["split"] == args.split)].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows for fold_id={args.fold_key} split={args.split}")

    k = min(args.k, len(df))
    sample_df = df.sample(n=k, random_state=args.seed)

    print(f"[QA] fold={args.fold_key} split={args.split} k={k}")

    rows = []
    for i, r in enumerate(sample_df.itertuples(index=False), start=1):
        video_path = os.path.normpath(os.path.join(args.root_dir, getattr(r, "front_video_path")))
        start_frame = int(getattr(r, "start_frame"))
        window_frames = int(getattr(r, "window_frames"))
        label = int(getattr(r, "label"))

        clip = _read_clip_opencv(video_path, start_frame, window_frames)

        mad = mean_abs_diff(clip)
        ir = identical_ratio(clip)

        rows.append((label, mad, ir, video_path, start_frame, window_frames))

        print(
            f"[{i:02d}] label={label} mean_abs_diff={mad:.4f} identical_ratio={ir:.3f} "
            f"start={start_frame} T={window_frames} path={video_path}"
        )

    # quick aggregate (not for decision; just summary)
    mads = [x[1] for x in rows]
    irs = [x[2] for x in rows]
    print("\n[SUMMARY]")
    print(f"mean_abs_diff: min={min(mads):.4f} mean={np.mean(mads):.4f} max={max(mads):.4f}")
    print(f"identical_ratio: min={min(irs):.3f} mean={np.mean(irs):.3f} max={max(irs):.3f}")


if __name__ == "__main__":
    main()
