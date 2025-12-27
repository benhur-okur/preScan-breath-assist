#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# python tools/label_from_imu.py data/synced/doga_normal_synced.csv --case-type normal --plot

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def robust_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Simple EMA smoothing. alpha in (0,1]. Lower -> more smoothing."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def rolling_std(t: np.ndarray, x: np.ndarray, win_s: float) -> np.ndarray:
    """
    Rolling std with a time-based window ~ win_s seconds.
    Assumes roughly uniform sampling but uses median dt to estimate window size.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.full(n, np.nan)

    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.01
    win_n = max(5, int(round(win_s / max(dt_med, 1e-6))))
    # center rolling
    half = win_n // 2

    out = np.empty(n, dtype=float)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = float(np.std(x[a:b]))
    return out


def fill_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Fills False gaps inside True regions if the gap length <= max_gap.
    Example: True True False False True  with max_gap>=2 -> True True True True True
    """
    mask = mask.astype(bool).copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        # start of False run
        j = i
        while j < n and not mask[j]:
            j += 1
        gap_len = j - i
        # if surrounded by True and gap small -> fill
        if gap_len <= max_gap and i > 0 and j < n and mask[i - 1] and mask[j]:
            mask[i:j] = True
        i = j
    return mask


def longest_true_run(mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Returns (start_idx, end_idx_inclusive, length) for the longest contiguous True run.
    If none, returns (-1, -1, 0).
    """
    mask = mask.astype(bool)
    best_len = 0
    best_s = -1
    best_e = -1

    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        s = i
        while i < n and mask[i]:
            i += 1
        e = i - 1
        L = e - s + 1
        if L > best_len:
            best_len = L
            best_s = s
            best_e = e
    return best_s, best_e, best_len


def time_to_index(t: np.ndarray, time_s: float) -> int:
    """Nearest index for a timestamp."""
    return int(np.argmin(np.abs(t - float(time_s))))


# -----------------------------
# Core detection logic
# -----------------------------

@dataclass
class HoldDetectionResult:
    status: str           # OK / WARN / FAIL
    reason: str
    hold_start_s: Optional[float]
    hold_end_s: Optional[float]
    th: float
    th_high: float
    core_len_s: float


def detect_hold_window(
    t: np.ndarray,
    a_mag: np.ndarray,
    *,
    win_s: float,
    ema_alpha: float,
    stable_percentile: float,
    th_mult: float,
    th_high_mult: float,
    min_hold_s: float,
    gap_fill_s: float,
) -> HoldDetectionResult:
    """
    Detect a stable 'hold' window from IMU magnitude.

    Steps:
    1) center a_mag (remove mean)
    2) smooth with EMA
    3) compute rolling std (time window win_s)
    4) compute adaptive threshold th using stable_percentile
    5) stable_mask = rollstd <= th
    6) fill small gaps
    7) choose longest stable run as core; must be >= min_hold_s
    8) expand boundaries using th_high (looser) to capture real hold edges
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(a_mag, dtype=float)

    if len(t) < 20:
        return HoldDetectionResult(
            status="FAIL",
            reason="Too few samples.",
            hold_start_s=None,
            hold_end_s=None,
            th=float("nan"),
            th_high=float("nan"),
            core_len_s=0.0
        )

    # Center and smooth
    v_center = v - np.mean(v)
    v_smooth = robust_ema(v_center, alpha=ema_alpha)

    rs = rolling_std(t, v_smooth, win_s=win_s)
    # Ignore NaNs if any
    rs_valid = rs[np.isfinite(rs)]
    if rs_valid.size < 10:
        return HoldDetectionResult(
            status="FAIL",
            reason="Rolling std invalid / too many NaNs.",
            hold_start_s=None,
            hold_end_s=None,
            th=float("nan"),
            th_high=float("nan"),
            core_len_s=0.0
        )

    # Adaptive threshold from low-variance part
    base = float(np.percentile(rs_valid, stable_percentile))
    th = base * float(th_mult)
    th_high = base * float(th_high_mult)

    # Safety: avoid absurdly tiny thresholds that cause false WARNs
    # (This is exactly the issue you hit earlier.)
    floor = float(np.median(rs_valid) * 0.35)
    if th < floor:
        th = floor
    if th_high < th:
        th_high = th * 1.2

    stable = rs <= th

    # Gap fill in samples (convert seconds -> samples)
    dt = np.diff(t)
    dt_med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.01
    gap_fill_n = max(0, int(round(gap_fill_s / max(dt_med, 1e-6))))
    stable_filled = fill_small_gaps(stable, max_gap=gap_fill_n)

    # Longest stable run = core
    s_idx, e_idx, L = longest_true_run(stable_filled)
    if L <= 0:
        return HoldDetectionResult(
            status="WARN",
            reason="No stable run found.",
            hold_start_s=None,
            hold_end_s=None,
            th=th,
            th_high=th_high,
            core_len_s=0.0
        )

    core_len_s = float(t[e_idx] - t[s_idx])
    if core_len_s < float(min_hold_s):
        return HoldDetectionResult(
            status="WARN",
            reason="Stable run exists but too short.",
            hold_start_s=None,
            hold_end_s=None,
            th=th,
            th_high=th_high,
            core_len_s=core_len_s
        )

    # Expand using looser threshold th_high (rs <= th_high)
    looser = rs <= th_high

    # Expand left
    s2 = s_idx
    while s2 > 0 and looser[s2 - 1]:
        s2 -= 1
    # Expand right
    e2 = e_idx
    while e2 < len(looser) - 1 and looser[e2 + 1]:
        e2 += 1

    # Final hold window
    hold_start_s = float(t[s2])
    hold_end_s = float(t[e2])

    # Sanity: ensure non-trivial
    if hold_end_s - hold_start_s < min_hold_s:
        return HoldDetectionResult(
            status="WARN",
            reason="Expanded hold window ended up too short.",
            hold_start_s=None,
            hold_end_s=None,
            th=th,
            th_high=th_high,
            core_len_s=core_len_s
        )

    return HoldDetectionResult(
        status="OK",
        reason="Hold window detected.",
        hold_start_s=hold_start_s,
        hold_end_s=hold_end_s,
        th=th,
        th_high=th_high,
        core_len_s=core_len_s
    )


# -----------------------------
# Labeling
# -----------------------------

def label_dataframe(df: pd.DataFrame, case_type: str, det: Optional[HoldDetectionResult]) -> pd.DataFrame:
    """
    Adds 'label' column:
      1 = hold
      0 = breathing
    """
    out = df.copy()

    if "seconds_elapsed" not in out.columns or "a_mag" not in out.columns:
        raise ValueError("Expected columns missing: 'seconds_elapsed' and/or 'a_mag'.")

    t = out["seconds_elapsed"].astype(float).values

    label = np.zeros(len(out), dtype=int)  # default breathing

    if case_type in ("inhale_hold", "exhale_hold"):
        if det is not None and det.status == "OK" and det.hold_start_s is not None and det.hold_end_s is not None:
            mask = (t >= det.hold_start_s) & (t <= det.hold_end_s)
            label[mask] = 1
        else:
            # if we cannot find hold, do NOT fake it
            # keep all breathing -> safer than injecting wrong labels
            pass

    elif case_type == "normal":
        # all breathing (0)
        pass

    elif case_type == "irregular":
        # default all breathing (0), unless user explicitly wants hold detection
        pass

    else:
        raise ValueError(f"Unknown case_type: {case_type}")

    out["label"] = label
    return out


# -----------------------------
# Plotting
# -----------------------------

def plot_labels(df: pd.DataFrame, title: str, det: Optional[HoldDetectionResult]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; cannot plot.")
        return

    t = df["seconds_elapsed"].astype(float).values
    a = df["a_mag"].astype(float).values
    y = df["label"].astype(int).values

    plt.figure(figsize=(12, 5))
    plt.plot(t, a, label="a_mag")

    # highlight hold region
    if det is not None and det.status == "OK" and det.hold_start_s is not None and det.hold_end_s is not None:
        plt.axvspan(det.hold_start_s, det.hold_end_s, alpha=0.2, label="Detected HOLD window")

        plt.axvline(det.hold_start_s, linestyle="--", alpha=0.7)
        plt.axvline(det.hold_end_s, linestyle="--", alpha=0.7)

    # overlay labels (scaled)
    a_min, a_max = np.min(a), np.max(a)
    span = (a_max - a_min) if (a_max > a_min) else 1.0
    y_overlay = a_min + y * (0.12 * span)
    plt.plot(t, y_overlay, label="label (overlay)", alpha=0.8)

    plt.xlabel("seconds_elapsed (s)")
    plt.ylabel("a_mag")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate labels from synced IMU magnitude (a_mag) for breath-hold dataset."
    )
    parser.add_argument("synced_csv", type=str, help="Path to synced CSV (output of sync_manager.py)")
    parser.add_argument(
        "--case-type",
        required=True,
        choices=["normal", "inhale_hold", "exhale_hold", "irregular"],
        help="Breathing protocol case for this recording."
    )

    # Your validated defaults (new baseline)
    # Your validated defaults (new baseline) - MORE TOLERANT for micro-movements
    parser.add_argument("--win-s", type=float, default=1.00, help="Rolling std window size in seconds.")
    parser.add_argument("--ema-alpha", type=float, default=0.15, help="EMA alpha for smoothing a_mag.")
    parser.add_argument("--stable-percentile", type=float, default=40.0, help="Percentile for baseline stability.")
    parser.add_argument("--th-mult", type=float, default=1.50, help="Multiplier for strict threshold.")
    parser.add_argument("--th-high-mult", type=float, default=2.60, help="Multiplier for expansion threshold.")
    parser.add_argument("--min-hold-s", type=float, default=1.8, help="Minimum hold duration (seconds).")
    parser.add_argument("--gap-fill-s", type=float, default=0.40, help="Fill short unstable gaps (seconds).")

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/labeled",
        help="Output directory for labeled CSV."
    )
    parser.add_argument("--plot", action="store_true", help="Show plot with detected hold window and labels.")

    # Optional: allow irregular hold detection if user insists
    parser.add_argument(
        "--allow-irregular-hold",
        action="store_true",
        help="If set, will attempt hold detection for irregular case too (not recommended)."
    )

    args = parser.parse_args()

    if not os.path.exists(args.synced_csv):
        raise FileNotFoundError(f"File not found: {args.synced_csv}")

    df = pd.read_csv(args.synced_csv)

    # Basic column checks
    for col in ("seconds_elapsed", "a_mag"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.synced_csv}")

    t = df["seconds_elapsed"].astype(float).values
    a = df["a_mag"].astype(float).values

    # Decide whether to detect hold for this case
    do_detect_hold = args.case_type in ("inhale_hold", "exhale_hold")
    if args.case_type == "irregular" and args.allow_irregular_hold:
        do_detect_hold = True

    det = None
    if do_detect_hold:
        det = detect_hold_window(
            t, a,
            win_s=args.win_s,
            ema_alpha=args.ema_alpha,
            stable_percentile=args.stable_percentile,
            th_mult=args.th_mult,
            th_high_mult=args.th_high_mult,
            min_hold_s=args.min_hold_s,
            gap_fill_s=args.gap_fill_s,
        )

        # Print META summary
        if det.status == "OK":
            print(f"[META] case={args.case_type} status=OK")
            print(f"       hold={det.hold_start_s:.2f}s → {det.hold_end_s:.2f}s")
            print(f"       th={det.th:.4f} (adaptive)  th_high={det.th_high:.4f}")
        else:
            print(f"[META] case={args.case_type} status={det.status}")
            print(f"       reason={det.reason}")
            print(f"       th={det.th:.4f} (adaptive)  th_high={det.th_high:.4f}")
            print(f"       core_len_s={det.core_len_s:.2f}s")

    labeled = label_dataframe(df, args.case_type, det)

    # Save
    ensure_dir(args.out_dir)
    base = os.path.splitext(os.path.basename(args.synced_csv))[0]
    out_path = os.path.join(args.out_dir, f"{base}_labeled.csv")
    labeled.to_csv(out_path, index=False)
    print(f"       saved → {out_path}")

    # Plot
    if args.plot:
        title = f"{base} | case={args.case_type}"
        plot_labels(labeled, title=title, det=det)


if __name__ == "__main__":
    main()
