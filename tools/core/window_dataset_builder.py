# tools/window_dataset_builder.py
import os
import re
import glob
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

'''
python tools/window_dataset_builder.py --labeled-dir data/labeled_safe --videos-root data/raw/videos --out data/windows/manifest.csv --window-sec 0.5 --stride-sec 0.25 --min-coverage 0.60

'''

VIDEO_EXTS = (".mp4", ".mov", ".m4v")

CASE_TO_IDX = {
    "normal": 1,
    "inhale_hold": 2,
    "exhale_hold": 3,
    "irregular": 4,
}

# Cache: path -> (duration_s, fps, total_frames)
_video_meta_cache: Dict[str, Tuple[float, float, int]] = {}


def get_video_meta(path: str, fps_fallback: float) -> Tuple[float, float, int]:
    """
    Returns (duration_s, fps_real, total_frames).
    total_frames is computed as floor(duration * fps_real).
    """
    if path in _video_meta_cache:
        return _video_meta_cache[path]

    with VideoFileClip(path) as clip:
        duration_s = float(clip.duration)
        fps_real = float(clip.fps) if clip.fps else float(fps_fallback)

    total_frames = int(np.floor(duration_s * fps_real))
    _video_meta_cache[path] = (duration_s, fps_real, total_frames)
    return duration_s, fps_real, total_frames


def _find_video_recursive(root_dir: str, base_no_ext: str) -> Optional[str]:
    target = base_no_ext.lower()
    for root, _, files in os.walk(root_dir):
        for f in files:
            b, ext = os.path.splitext(f)
            if ext.lower() in VIDEO_EXTS and b.lower() == target:
                return os.path.join(root, f)
    return None


def infer_video_path(csv_path: str, videos_root: str, view: str) -> str:
    fname = os.path.basename(csv_path)
    m = re.match(
        r"^(?P<person>.+)_(?P<case>normal|inhale_hold|exhale_hold|irregular)_labeled\.csv$",
        fname
    )
    if not m:
        raise ValueError(f"Unexpected labeled CSV name format: {fname}")

    person = m.group("person")
    case_type = m.group("case")
    case_idx = CASE_TO_IDX[case_type]
    base = f"{person}_{case_idx}_{view}"

    # try person folder
    person_dir = None
    for d in os.listdir(videos_root):
        full = os.path.join(videos_root, d)
        if os.path.isdir(full) and d.lower() == person.lower():
            person_dir = full
            break

    if person_dir:
        found = _find_video_recursive(person_dir, base)
        if found:
            return found

    found = _find_video_recursive(videos_root, base)
    if found:
        return found

    raise FileNotFoundError(
        f"Video not found for view='{view}'. Expected base='{base}' under '{videos_root}'."
    )


def majority_vote(labels: np.ndarray) -> int:
    return int(np.mean(labels) >= 0.5)


def build_manifest(
    labeled_dir: str,
    videos_root: str,
    out_path: str,
    fps_expected: float = 30.0,
    window_sec: float = 0.5,
    stride_sec: float = 0.25,
    min_coverage: float = 0.60,
) -> None:

    csv_files = sorted(glob.glob(os.path.join(labeled_dir, "*_labeled.csv")))
    if not csv_files:
        raise RuntimeError(f"No labeled CSV files found in: {labeled_dir}")

    print(f"[INFO] Found {len(csv_files)} labeled files.")

    rows: List[Dict] = []
    window_id_counter = 0

    for csv_path in csv_files:
        base = os.path.basename(csv_path)
        print(f"\n[PROCESS] {base}")

        df = pd.read_csv(csv_path)
        required_cols = {"front_frame", "label"}
        if not required_cols.issubset(df.columns):
            raise RuntimeError(f"{base} missing required columns. Need: {required_cols}")

        m = re.match(
            r"^(?P<person>.+)_(?P<case>normal|inhale_hold|exhale_hold|irregular)_labeled\.csv$",
            base
        )
        if not m:
            raise ValueError(f"Unexpected labeled CSV name format: {base}")

        person = m.group("person")
        case_type = m.group("case")
        case_idx = CASE_TO_IDX[case_type]

        front_path = infer_video_path(csv_path, videos_root, view="front")
        side_path = infer_video_path(csv_path, videos_root, view="side")

        # Get meta for both videos (real fps!)
        front_duration_s, front_fps_real, front_total_frames = get_video_meta(front_path, fps_expected)
        side_duration_s, side_fps_real, side_total_frames = get_video_meta(side_path, fps_expected)

        # Window params derived from FRONT fps (front-only baseline)
        window_frames = int(round(window_sec * front_fps_real))
        stride_frames = int(round(stride_sec * front_fps_real))
        if window_frames < 2:
            raise RuntimeError(f"window_frames too small: {window_frames}. Check window_sec/fps.")
        if stride_frames < 1:
            stride_frames = 1

        frames = df["front_frame"].values.astype(int)
        labels = df["label"].values.astype(int)

        fmin = int(frames.min())
        fmax = int(frames.max())
        last_valid_frame = front_total_frames - 1

        start_min = max(0, fmin)
        start_max = min(fmax, last_valid_frame - (window_frames - 1))

        if start_max < start_min:
            print("  [WARN] No valid window range (video shorter than labeled range). Skipping.")
            continue

        win_count = 0
        for start in range(start_min, start_max + 1, stride_frames):
            end = start + window_frames - 1
            if end > last_valid_frame:
                continue

            mask = (frames >= start) & (frames <= end)
            n_samples = int(mask.sum())
            if n_samples < int(window_frames * min_coverage):
                continue

            window_label = majority_vote(labels[mask])

            start_time = start / front_fps_real
            end_time = end / front_fps_real
            center_time = (start + end) / 2.0 / front_fps_real

            rows.append({
                "window_id": window_id_counter,
                "source_csv": os.path.relpath(csv_path),
                "person": person,
                "case": case_type,
                "case_idx": case_idx,

                "front_video_path": os.path.relpath(front_path),
                "side_video_path": os.path.relpath(side_path),

                # Repro / provenance
                "front_fps": round(front_fps_real, 4),
                "side_fps": round(side_fps_real, 4),
                "window_sec": float(window_sec),
                "stride_sec": float(stride_sec),
                "window_frames": int(window_frames),
                "stride_frames": int(stride_frames),
                "min_coverage": float(min_coverage),

                # Frame/time window
                "start_frame": int(start),
                "end_frame": int(end),
                "start_time_sec": round(start_time, 4),
                "end_time_sec": round(end_time, 4),
                "center_time_sec": round(center_time, 4),

                # Coverage
                "n_samples_in_window": int(n_samples),
                "samples_per_frame": round(n_samples / float(window_frames), 4),

                # Video meta (debug friendly)
                "front_duration_s": round(front_duration_s, 4),
                "front_total_frames": int(front_total_frames),
                "side_duration_s": round(side_duration_s, 4),
                "side_total_frames": int(side_total_frames),

                "label": int(window_label),

                # Split column (blank for now; we fill later with split script)
                "split": "",
                "fold_id": "",
            })

            window_id_counter += 1
            win_count += 1

        print(
            f"  → windows created: {win_count} | front_fps={front_fps_real:.3f} | front_total_frames={front_total_frames}"
        )

    manifest = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    manifest.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print(f"[OK] Window manifest saved → {out_path}")
    print(f"Total windows: {len(manifest)}")
    if len(manifest) > 0:
        print("Label distribution:\n", manifest["label"].value_counts())
        print("Case distribution:\n", manifest["case"].value_counts())
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build window-level dataset manifest from labeled CSVs.")
    parser.add_argument("--labeled-dir", required=True)
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--out", default="data/windows/manifest.csv")
    parser.add_argument("--fps-expected", type=float, default=30.0)
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument("--stride-sec", type=float, default=0.25)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    args = parser.parse_args()

    build_manifest(
        labeled_dir=args.labeled_dir,
        videos_root=args.videos_root,
        out_path=args.out,
        fps_expected=args.fps_expected,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        min_coverage=args.min_coverage,
    )
