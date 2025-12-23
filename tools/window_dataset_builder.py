import os
import re
import glob
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

"""
python tools/window_dataset_builder.py \
  --labeled-dir data/labeled_safe \
  --videos-root data/raw/videos \
  --out data/windows/manifest.csv \
  --window-sec 0.5 \
  --stride-sec 0.25 \
  --min-coverage 0.60
"""

# ------------------------------------------------------------
# DEFAULT WINDOW CONFIG (override via CLI)
# ------------------------------------------------------------
DEFAULT_FPS_EXPECTED = 30.0
DEFAULT_WINDOW_SEC = 0.5
DEFAULT_STRIDE_SEC = 0.25

VIDEO_EXTS = (".mp4", ".mov", ".m4v")

CASE_TO_IDX = {
    "normal": 1,
    "inhale_hold": 2,
    "exhale_hold": 3,
    "irregular": 4,
}

# ------------------------------------------------------------
# VIDEO META CACHE
# ------------------------------------------------------------
# path -> (duration_s, total_frames_for_given_fps)
_video_meta_cache: Dict[Tuple[str, float], Tuple[float, int]] = {}


def get_video_meta(path: str, fps_for_frames: float) -> Tuple[float, int]:
    """
    Returns:
      duration_s (float)
      total_frames (int) computed using fps_for_frames

    Cache key includes fps, because total_frames depends on fps.
    """
    key = (path, float(fps_for_frames))
    if key in _video_meta_cache:
        return _video_meta_cache[key]

    with VideoFileClip(path) as clip:
        duration_s = float(clip.duration)

    # Use round for stability (duration is float with tiny noise)
    total_frames = int(round(duration_s * float(fps_for_frames)))
    if total_frames < 1:
        total_frames = 1

    _video_meta_cache[key] = (duration_s, total_frames)
    return duration_s, total_frames


# ------------------------------------------------------------
# VIDEO PATH INFERENCE
# data/raw/videos/<PersonName>/<person>_<idx>_<front|side>.<ext>
# ------------------------------------------------------------
def _find_video_recursive(root_dir: str, base_no_ext: str) -> Optional[str]:
    target = base_no_ext.lower()
    for root, _, files in os.walk(root_dir):
        for f in files:
            b, ext = os.path.splitext(f)
            if ext.lower() in VIDEO_EXTS and b.lower() == target:
                return os.path.join(root, f)
    return None


def infer_video_path(csv_path: str, videos_root: str, view: str) -> str:
    """
    CSV:  benhur_exhale_hold_labeled.csv
    -> person=benhur, case=exhale_hold -> idx=3
    -> video base: benhur_3_front / benhur_3_side
    """
    fname = os.path.basename(csv_path)

    m = re.match(
        r"^(?P<person>.+)_(?P<case>normal|inhale_hold|exhale_hold|irregular)_labeled\.csv$",
        fname,
    )
    if not m:
        raise ValueError(f"Unexpected labeled CSV name format: {fname}")

    person = m.group("person")
    case_type = m.group("case")
    case_idx = CASE_TO_IDX[case_type]
    base = f"{person}_{case_idx}_{view}"

    # 1) try person folder match (case-insensitive)
    person_dir = None
    if os.path.isdir(videos_root):
        for d in os.listdir(videos_root):
            full = os.path.join(videos_root, d)
            if os.path.isdir(full) and d.lower() == person.lower():
                person_dir = full
                break

    if person_dir:
        found = _find_video_recursive(person_dir, base)
        if found:
            return found

    # 2) fallback: search anywhere under videos_root
    found = _find_video_recursive(videos_root, base)
    if found:
        return found

    raise FileNotFoundError(
        f"Video not found for view='{view}'. Expected base='{base}' under '{videos_root}'."
    )


# ------------------------------------------------------------
# LABEL WINDOWING
# ------------------------------------------------------------
def majority_vote(labels: np.ndarray) -> int:
    """label=1 if >=50% are hold"""
    return int(np.mean(labels) >= 0.5)


def _norm_path(p: str) -> str:
    """
    Normalize paths to be OS-friendly and stable in CSV:
    - keep relative paths
    - use forward slashes (Windows-friendly in Python too)
    """
    return p.replace("\\", "/")


# ------------------------------------------------------------
# MANIFEST BUILDER
# ------------------------------------------------------------
def build_manifest(
    labeled_dir: str,
    videos_root: str,
    out_path: str,
    fps_expected: float = DEFAULT_FPS_EXPECTED,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
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

        # parse person/case from filename robustly
        m = re.match(
            r"^(?P<person>.+)_(?P<case>normal|inhale_hold|exhale_hold|irregular)_labeled\.csv$",
            base,
        )
        if not m:
            raise ValueError(f"Unexpected labeled CSV name format: {base}")

        person = m.group("person")
        case_type = m.group("case")
        case_idx = CASE_TO_IDX[case_type]

        front_path_abs = infer_video_path(csv_path, videos_root, view="front")
        side_path_abs = infer_video_path(csv_path, videos_root, view="side")

        # Read fps from front clip (authoritative)
        with VideoFileClip(front_path_abs) as clip:
            fps_real = float(clip.fps) if clip.fps else float(fps_expected)

        # Derive window params based on fps_real
        window_frames = int(round(window_sec * fps_real))
        stride_frames = int(round(stride_sec * fps_real))
        if window_frames < 2:
            raise RuntimeError(f"window_frames too small: {window_frames}. Check window_sec/fps.")
        if stride_frames < 1:
            stride_frames = 1

        # Video meta (duration + total frames) for both views
        front_duration_s, front_total_frames = get_video_meta(front_path_abs, fps_real)
        side_duration_s, side_total_frames = get_video_meta(side_path_abs, fps_real)

        # Labeled data
        frames = df["front_frame"].values.astype(int)
        labels = df["label"].values.astype(int)

        fmin = int(frames.min())
        fmax = int(frames.max())

        # Use "last valid frame index" consistent with total_frames
        front_last_valid = front_total_frames - 1
        if front_last_valid < 0:
            print("  [WARN] front_total_frames invalid (<1). Skipping.")
            continue

        # Start positions must allow a full window within video bounds
        start_min = max(0, fmin)
        start_max = min(fmax, front_last_valid - (window_frames - 1))

        win_count = 0
        if start_max < start_min:
            print("  [WARN] No valid window range (video shorter than labeled range). Skipping.")
            continue

        # relative paths in manifest (portable)
        front_rel = _norm_path(os.path.relpath(front_path_abs))
        side_rel = _norm_path(os.path.relpath(side_path_abs))
        csv_rel = _norm_path(os.path.relpath(csv_path))

        for start in range(start_min, start_max + 1, stride_frames):
            end = start + window_frames - 1
            if end > front_last_valid:
                continue

            mask = (frames >= start) & (frames <= end)
            n_samples = int(mask.sum())

            # Need enough labeled samples to trust majority voting
            if n_samples < int(window_frames * float(min_coverage)):
                continue

            window_label = majority_vote(labels[mask])

            start_time = start / fps_real
            end_time = end / fps_real
            center_time = (start + end) / 2.0 / fps_real

            rows.append({
                # IDs / provenance
                "window_id": window_id_counter,
                "source_csv": csv_rel,
                "person": person,
                "case": case_type,
                "case_idx": case_idx,

                # Video paths
                "front_video_path": front_rel,
                "side_video_path": side_rel,

                # Reproducibility: windowing
                "fps": round(fps_real, 6),
                "window_sec": float(window_sec),
                "stride_sec": float(stride_sec),
                "window_frames": int(window_frames),
                "stride_frames": int(stride_frames),

                # Frame slice
                "start_frame": int(start),
                "end_frame": int(end),
                "start_time_sec": round(start_time, 6),
                "end_time_sec": round(end_time, 6),
                "center_time_sec": round(center_time, 6),

                # Coverage diagnostics
                "n_samples_in_window": int(n_samples),
                "samples_per_frame": round(n_samples / float(window_frames), 6),

                # NEW: video meta (per-video constant, but repeated for convenience)
                "front_duration_s": round(front_duration_s, 6),
                "front_total_frames": int(front_total_frames),
                "side_duration_s": round(side_duration_s, 6),
                "side_total_frames": int(side_total_frames),

                # Label
                "label": int(window_label),
            })

            window_id_counter += 1
            win_count += 1

        print(
            f"  → windows created: {win_count} | fps={fps_real:.3f} "
            f"| front_frames={front_total_frames} | side_frames={side_total_frames}"
        )

    manifest = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    manifest.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print(f"[OK] Window manifest saved → {out_path}")
    print(f"Total windows: {len(manifest)}")
    if len(manifest) > 0:
        print("Label distribution:")
        print(manifest["label"].value_counts())
        print("Case distribution:")
        print(manifest["case"].value_counts())
    print("=" * 70)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build window-level dataset manifest from labeled CSVs.")
    parser.add_argument("--labeled-dir", required=True)
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--out", default="data/windows/manifest.csv")
    parser.add_argument("--fps-expected", type=float, default=DEFAULT_FPS_EXPECTED)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--stride-sec", type=float, default=DEFAULT_STRIDE_SEC)
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
