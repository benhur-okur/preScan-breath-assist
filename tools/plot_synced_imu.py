
'''
1) Tek inhale_hold dosyasını çiz (varsayılan %20)
python tools/plot_synced_imu.py data/synced/person1_inhale_hold_synced.csv --case-type inhale_hold

2) %15 edge + %10 ignore ile “hold core”u gör
python tools/plot_synced_imu.py data/synced/person1_inhale_hold_synced.csv \
  --case-type inhale_hold \
  --edge-breath 0.15 \
  --ignore-transition 0.10 \
  --smooth-window 25

3) Klasördeki tüm inhale_hold kayıtlarını PNG olarak kaydet
python tools/plot_synced_imu.py data/synced \
  --case-type inhale_hold \
  --edge-breath 0.15 \
  --ignore-transition 0.10 \
  --smooth-window 25 \
  --save-dir outputs/imu_plots \
  --no-show
'''



import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# matplotlib optional import guard (but you have it usually)
import matplotlib.pyplot as plt


def _ensure_columns(df: pd.DataFrame, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")


def _nice_basename(p: str) -> str:
    return Path(p).stem


def plot_one_csv(
    csv_path: str,
    case_type: str | None = None,
    edge_breath: float = 0.20,
    ignore_transition: float = 0.00,
    show: bool = True,
    save_path: str | None = None,
    center: bool = True,
    smooth_window: int = 0,
):
    df = pd.read_csv(csv_path)
    _ensure_columns(df, ["seconds_elapsed", "a_mag"])

    t = df["seconds_elapsed"].astype(float).to_numpy()
    a = df["a_mag"].astype(float).to_numpy()

    # Sort by time (just in case)
    order = np.argsort(t)
    t = t[order]
    a = a[order]

    # Centering helps visually
    if center:
        a_plot = a - np.mean(a)
    else:
        a_plot = a.copy()

    # Optional smoothing (moving average) for easier visual segmentation
    if smooth_window and smooth_window > 1:
        w = int(smooth_window)
        kernel = np.ones(w) / w
        a_sm = np.convolve(a_plot, kernel, mode="same")
    else:
        a_sm = None

    t0, t1 = float(t[0]), float(t[-1])
    dur = t1 - t0
    if dur <= 0.1:
        raise ValueError(f"Duration too short ({dur:.3f}s). Check timestamps in {csv_path}")

    # Percent boundaries
    edge = float(edge_breath)
    ign = float(ignore_transition)

    if not (0.0 <= edge <= 0.45):
        raise ValueError("--edge-breath must be in [0, 0.45] realistically.")
    if not (0.0 <= ign <= 0.30):
        raise ValueError("--ignore-transition must be in [0, 0.30] realistically.")
    if (2 * edge + 2 * ign) >= 1.0:
        raise ValueError("edge+ignore too large: 2*edge + 2*ignore must be < 1.0")

    # Timeline boundaries (relative to clipped synced duration)
    # Regions:
    # [breath edge] [ignore] [HOLD core] [ignore] [breath edge]
    b1 = t0 + dur * edge
    i1 = b1 + dur * ign
    i2 = t1 - dur * (edge + ign)
    b2 = t1 - dur * edge

    title = f"{_nice_basename(csv_path)}"
    if case_type:
        title += f"  |  case={case_type}"
    title += f"  |  dur={dur:.2f}s"

    plt.figure(figsize=(12, 4))
    plt.plot(t, a_plot, linewidth=1.0, label="a_mag (centered)" if center else "a_mag")

    if a_sm is not None:
        plt.plot(t, a_sm, linewidth=1.0, label=f"smoothed (MA={smooth_window})")

    # Overlay segmentation guides (help you decide %s)
    if case_type in ("inhale_hold", "exhale_hold"):
        # Breath edges (left & right)
        plt.axvline(b1, linestyle="--", alpha=0.8, label=f"edge={edge:.0%}")
        plt.axvline(b2, linestyle="--", alpha=0.8)

        # Ignore transitions
        if ign > 0:
            plt.axvline(i1, linestyle=":", alpha=0.8, label=f"ignore={ign:.0%}")
            plt.axvline(i2, linestyle=":", alpha=0.8)

        # Shade hold core
        plt.axvspan(i1, i2, alpha=0.15, label="HOLD core (proposed)")
    else:
        # For normal/irregular, still show edges if you want, but no hold shading by default.
        pass

    plt.xlabel("seconds_elapsed (s)")
    plt.ylabel("a_mag (centered)" if center else "a_mag")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        outp = Path(save_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outp, dpi=150)
        print(f"[OK] Saved plot -> {outp}")

    if show:
        plt.show()
    else:
        plt.close()


def iter_synced_files(input_path: str):
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return [str(p)]
    if p.is_dir():
        # assume synced files end with _synced.csv but still accept all csv
        files = sorted([str(x) for x in p.glob("*.csv")])
        return files
    raise FileNotFoundError(f"Not found: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot seconds_elapsed vs a_mag from synced CSV files, with optional % segmentation overlays."
    )
    parser.add_argument("input", help="Path to a synced CSV file OR a directory containing synced CSVs.")
    parser.add_argument(
        "--case-type",
        choices=["normal", "inhale_hold", "exhale_hold", "irregular"],
        default=None,
        help="If inhale_hold/exhale_hold, overlays proposed hold-core region.",
    )
    parser.add_argument(
        "--edge-breath",
        type=float,
        default=0.20,
        help="Breathing edge fraction at start & end (default 0.20).",
    )
    parser.add_argument(
        "--ignore-transition",
        type=float,
        default=0.00,
        help="Ignore band fraction between breathing and hold (default 0.00).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show plots (useful with --save-dir).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set, saves each plot as PNG into this directory.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Optional moving average window size (e.g., 25). 0 disables smoothing.",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Do not mean-center a_mag (default centers).",
    )

    args = parser.parse_args()

    files = iter_synced_files(args.input)
    if len(files) == 0:
        print("[WARN] No CSV files found.")
        return

    show = not args.no_show
    center = not args.no_center

    for f in files:
        save_path = None
        if args.save_dir:
            out_name = _nice_basename(f) + f"_case_{args.case_type or 'none'}.png"
            save_path = str(Path(args.save_dir) / out_name)

        try:
            plot_one_csv(
                f,
                case_type=args.case_type,
                edge_breath=args.edge_breath,
                ignore_transition=args.ignore_transition,
                show=show,
                save_path=save_path,
                center=center,
                smooth_window=args.smooth_window,
            )
        except Exception as e:
            print(f"[FAIL] {Path(f).name}: {e}")


if __name__ == "__main__":
    main()
