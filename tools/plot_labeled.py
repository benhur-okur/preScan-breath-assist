#!/usr/bin/env python3
"""
Labeled CSV'yi görselleştir.
Kullanım: python tools/plot_labeled. py data/labeled/kisi1_inhale_hold_synced_labeled.csv
"""



## Tek bir labeled CSV'yi görüntüle
# python tools/plot_labeled.py data/labeled/kisi1_inhale_hold_synced_labeled.csv

# PNG olarak kaydet
# python tools/plot_labeled.py data/labeled/kisi1_inhale_hold_synced_labeled.csv --save labeledPlots/kisi1_inhale.png


import argparse
import sys
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot labeled CSV (a_mag + labels)")
    parser.add_argument("labeled_csv", type=str, help="Path to labeled CSV")
    parser.add_argument("--save", type=str, default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib not installed")
        sys.exit(1)

    df = pd.read_csv(args. labeled_csv)

    # Gerekli kolonlar
    for col in ("seconds_elapsed", "a_mag", "label"):
        if col not in df.columns:
            print(f"[ERROR] Missing column: {col}")
            sys.exit(1)

    t = df["seconds_elapsed"].astype(float).values
    a = df["a_mag"].astype(float).values
    y = df["label"]. astype(int).values

    # İstatistikler
    total_duration = t[-1] - t[0]
    hold_mask = y == 1
    hold_duration = np.sum(np.diff(t)[hold_mask[:-1]]) if np.any(hold_mask) else 0
    breathing_duration = total_duration - hold_duration

    # Plot
    fig, ax = plt. subplots(figsize=(14, 5))

    ax.plot(t, a, label="a_mag", color='blue', alpha=0.8, linewidth=0.8)

    # Hold bölgelerini highlight et
    in_hold = False
    hold_start = 0
    for i, val in enumerate(y):
        if val == 1 and not in_hold:
            hold_start = t[i]
            in_hold = True
        elif val == 0 and in_hold:
            ax.axvspan(hold_start, t[i], alpha=0.25, color='green', label="hold" if i == np.where(y == 0)[0][np.where(y == 0)[0] > np.where(y == 1)[0][0]][0] else "")
            in_hold = False
    # Son hold bölgesi kapanmamışsa
    if in_hold:
        ax.axvspan(hold_start, t[-1], alpha=0.25, color='green')

    # Label overlay
    a_min, a_max = np.min(a), np.max(a)
    span = (a_max - a_min) if (a_max > a_min) else 1.0
    y_overlay = a_min - 0.05 * span + y * (0.08 * span)
    ax.fill_between(t, a_min - 0.05 * span, y_overlay, alpha=0.4, color='orange', label="label=1")

    # Bilgi kutusu
    info_text = (
        f"Total:  {total_duration:.1f}s\n"
        f"Hold: {hold_duration:.1f}s ({100*hold_duration/total_duration:.1f}%)\n"
        f"Breathing: {breathing_duration:.1f}s ({100*breathing_duration/total_duration:.1f}%)"
    )
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel("seconds_elapsed (s)")
    ax.set_ylabel("a_mag")
    ax.set_title(args.labeled_csv. split("/")[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()

    if args.save:
        plt. savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()