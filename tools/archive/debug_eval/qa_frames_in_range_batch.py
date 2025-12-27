import argparse
import os
import sys
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import pandas as pd

# python tools/qa_frames_in_range_batch.py --print-ok

CASE_TO_NUM = {
    "normal": 1,
    "inhale_hold": 2,
    "exhale_hold": 3,
    "irregular": 4,
}

VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".MP4", ".MOV", ".M4V")


@dataclass
class VideoInfo:
    path: str
    frames: int
    fps: float
    duration: float


def get_video_info_cv2(video_path: str) -> VideoInfo:
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python not installed. Install: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) and cap.get(cv2.CAP_PROP_FPS) > 0 else float("nan")
    duration = frame_count / fps if fps == fps and fps > 0 else float("nan")  # fps==fps checks not-nan
    cap.release()
    return VideoInfo(path=video_path, frames=frame_count, fps=fps, duration=duration)


def parse_labeled_name(filename: str) -> Tuple[str, str]:
    """
    Expected: <person>_<case>_labeled.csv
    person may contain underscores? In your examples it's just 'sinan', 'doga', etc.
    We'll parse from the end: ..._<case>_labeled.csv
    """
    base = os.path.basename(filename)
    if not base.lower().endswith("_labeled.csv"):
        raise ValueError(f"Not a labeled CSV name: {base}")

    stem = base[:-len("_labeled.csv")]  # remove suffix
    # case is the last token after '_' (or last two tokens for inhale_hold/exhale_hold)
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse person/case from: {base}")

    # Try match the longest case name (two-part cases first)
    if len(parts) >= 3:
        maybe_case2 = "_".join(parts[-2:])
        if maybe_case2 in CASE_TO_NUM:
            case = maybe_case2
            person = "_".join(parts[:-2])
            return person, case

    maybe_case1 = parts[-1]
    if maybe_case1 in CASE_TO_NUM:
        case = maybe_case1
        person = "_".join(parts[:-1])
        return person, case

    raise ValueError(f"Case token not recognized in: {base}. Got tail parts={parts[-2:]} / {parts[-1]}")


def find_video(videos_root: str, person: str, case_num: int, view: str) -> Optional[str]:
    """
    Find video like: {person}_{case_num}_{view}.mp4  (view: front/side)
    Search recursively in videos_root.
    """
    pattern = os.path.join(videos_root, "**", f"{person}_{case_num}_{view}.*")
    candidates = glob.glob(pattern, recursive=True)
    # filter by extension just in case
    candidates = [p for p in candidates if p.endswith(VIDEO_EXTS)]
    if not candidates:
        return None
    # if multiple, pick the shortest path / deterministic
    candidates.sort(key=lambda p: (len(p), p))
    return candidates[0]


def check_frame_bounds(df: pd.DataFrame, frame_col: str, video_frames: int) -> Tuple[bool, str]:
    if frame_col not in df.columns:
        return False, f"missing column '{frame_col}'"

    fmin = int(df[frame_col].min())
    fmax = int(df[frame_col].max())

    if fmin < 0:
        return False, f"{frame_col}.min={fmin} < 0"
    if fmax > video_frames - 1:
        return False, f"{frame_col}.max={fmax} > last_valid={video_frames - 1} (video_frames={video_frames})"
    return True, f"ok (min={fmin}, max={fmax}, video_frames={video_frames})"


def check_time_bounds(df: pd.DataFrame, t_col: str, video_dur: float, tol_before: float = 0.05, tol_after: float = 0.25) -> Tuple[bool, str]:
    if t_col not in df.columns:
        return True, f"missing '{t_col}' (skip)"

    if not (video_dur == video_dur):  # nan
        return True, "video duration NaN (skip)"

    tmin = float(df[t_col].min())
    tmax = float(df[t_col].max())

    ok = True
    reasons = []
    if tmin < -tol_before:
        ok = False
        reasons.append(f"tmin={tmin:.3f} < -{tol_before}")
    if tmax > video_dur + tol_after:
        ok = False
        reasons.append(f"tmax={tmax:.3f} > dur+{tol_after} (dur≈{video_dur:.3f})")

    if ok:
        return True, f"ok (tmin={tmin:.3f}, tmax={tmax:.3f}, dur≈{video_dur:.3f})"
    return False, "; ".join(reasons)


def main():
    ap = argparse.ArgumentParser( # rearranged the arguments that data/labeled to data/labeled_safe
        description="Batch QA: check front/side frame indices are within bounds for ALL labeled CSVs."
    )
    ap.add_argument("--labeled-dir", default="data/labeled_safe", help="Directory containing *_labeled.csv files")
    ap.add_argument("--videos-dir", default="data/raw/videos", help="Root directory containing videos (recursive search)")
    ap.add_argument("--strict", action="store_true", help="If set, fail when a matching video is missing")
    ap.add_argument("--print-ok", action="store_true", help="If set, also print OK rows (default prints only WARN/FAIL)")
    args = ap.parse_args()

    if not os.path.isdir(args.labeled_dir):
        print(f"[FAIL] labeled-dir not found: {args.labeled_dir}")
        sys.exit(1)
    if not os.path.isdir(args.videos_dir):
        print(f"[FAIL] videos-dir not found: {args.videos_dir}")
        sys.exit(1)

    labeled_files = sorted(glob.glob(os.path.join(args.labeled_dir, "*_labeled.csv")))
    if not labeled_files:
        print(f"[FAIL] No *_labeled.csv found in {args.labeled_dir}")
        sys.exit(1)

    total = 0
    ok_count = 0
    warn_count = 0
    fail_count = 0

    print("=== BATCH FRAME QA ===")
    print(f"Labeled files: {len(labeled_files)}")
    print(f"Videos root  : {args.videos_dir}")
    print("-" * 80)

    for csv_path in labeled_files:
        total += 1
        base = os.path.basename(csv_path)

        try:
            person, case = parse_labeled_name(base)
            case_num = CASE_TO_NUM[case]
        except Exception as e:
            fail_count += 1
            print(f"[FAIL] {base}: cannot parse name -> {e}")
            continue

        front_path = find_video(args.videos_dir, person, case_num, "front")
        side_path = find_video(args.videos_dir, person, case_num, "side")

        missing = []
        if front_path is None:
            missing.append("front")
        if side_path is None:
            missing.append("side")

        if missing:
            msg = f"{base}: missing videos {missing} for pattern {person}_{case_num}_(front|side).*"
            if args.strict:
                fail_count += 1
                print(f"[FAIL] {msg}")
                continue
            else:
                warn_count += 1
                print(f"[WARN] {msg} (skipping frame check for missing ones)")
                # we can still check whichever exists
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            fail_count += 1
            print(f"[FAIL] {base}: cannot read CSV -> {e}")
            continue

        row_status = "OK"
        row_msgs: List[str] = []

        # FRONT checks
        if front_path is not None:
            try:
                finfo = get_video_info_cv2(front_path)
            except Exception as e:
                row_status = "FAIL"
                row_msgs.append(f"front video open fail: {e}")
            else:
                ok_frames, msg_frames = check_frame_bounds(df, "front_frame", finfo.frames)
                ok_time, msg_time = check_time_bounds(df, "t_front_video", finfo.duration)

                if not ok_frames:
                    row_status = "FAIL"
                if not ok_time and row_status != "FAIL":
                    row_status = "WARN"

                row_msgs.append(f"front={os.path.basename(front_path)} | frames: {msg_frames} | time: {msg_time}")

        # SIDE checks
        if side_path is not None:
            try:
                sinfo = get_video_info_cv2(side_path)
            except Exception as e:
                row_status = "FAIL"
                row_msgs.append(f"side video open fail: {e}")
            else:
                ok_frames, msg_frames = check_frame_bounds(df, "side_frame", sinfo.frames)
                ok_time, msg_time = check_time_bounds(df, "t_side_video", sinfo.duration)

                if not ok_frames:
                    row_status = "FAIL"
                if not ok_time and row_status != "FAIL":
                    row_status = "WARN"

                row_msgs.append(f"side={os.path.basename(side_path)} | frames: {msg_frames} | time: {msg_time}")

        if row_status == "OK":
            ok_count += 1
            if args.print_ok:
                print(f"[OK]   {base}  ({person}/{case})")
                for m in row_msgs:
                    print(f"       - {m}")
        elif row_status == "WARN":
            warn_count += 1
            print(f"[WARN] {base}  ({person}/{case})")
            for m in row_msgs:
                print(f"       - {m}")
        else:
            fail_count += 1
            print(f"[FAIL] {base}  ({person}/{case})")
            for m in row_msgs:
                print(f"       - {m}")

    print("-" * 80)
    print("=== SUMMARY ===")
    print(f"Total: {total}")
    print(f"OK   : {ok_count}")
    print(f"WARN : {warn_count}")
    print(f"FAIL : {fail_count}")

    # exit code: fail -> 2, warn -> 1, ok -> 0
    if fail_count > 0:
        sys.exit(2)
    if warn_count > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
