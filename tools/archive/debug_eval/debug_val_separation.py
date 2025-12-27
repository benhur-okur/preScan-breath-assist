# tools/debug_val_separation.py
from __future__ import annotations

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="e.g. data/windows/manifest_final_train_v3.csv")
    ap.add_argument("--video-col", default="front_video_path")
    ap.add_argument("--case-col", default="case")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--split-col", default="split")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)

    need = [args.video_col, args.case_col, args.label_col, args.split_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    df[args.video_col] = df[args.video_col].astype(str)
    df[args.case_col] = df[args.case_col].astype(str)
    df[args.split_col] = df[args.split_col].astype(str)

    print("=== BASIC ===")
    print("rows:", len(df))
    print("\ncase counts:\n", df[args.case_col].value_counts(dropna=False))
    print("\nlabel counts:\n", df[args.label_col].value_counts(dropna=False))
    pos_rate = float((df[args.label_col] == 1).mean())
    print(f"\npos_rate(label=1): {pos_rate:.6f}")

    print("\n=== SPLIT CHECK ===")
    print("split counts:\n", df[args.split_col].value_counts(dropna=False))
    print("\nlabel counts by split:\n", df.groupby([args.split_col, args.label_col]).size())

    # video overlap
    train_v = set(df[df[args.split_col] == "train"][args.video_col])
    val_v   = set(df[df[args.split_col] == "val"][args.video_col])
    overlap = train_v & val_v
    print("\n=== VIDEO OVERLAP (train ∩ val) ===")
    print("train_videos:", len(train_v), "val_videos:", len(val_v), "overlap:", len(overlap))
    if len(overlap) > 0:
        print("OVERLAP LIST (first 20):")
        for v in list(sorted(overlap))[:20]:
            print("  -", v)

    # per-video label rate by split
    g = df.groupby([args.split_col, args.video_col])[args.label_col].mean().reset_index()
    print("\n=== PER-VIDEO LABEL MEAN (by split) ===")
    for sp in sorted(df[args.split_col].unique()):
        sub = g[g[args.split_col] == sp][args.label_col]
        if len(sub) == 0:
            continue
        print(f"[{sp}] videos={len(sub)}  min={sub.min():.4f}  p25={sub.quantile(0.25):.4f}  "
              f"median={sub.median():.4f}  p75={sub.quantile(0.75):.4f}  max={sub.max():.4f}")

    print("\n=== CASE DISTRIBUTION BY SPLIT (rows) ===")
    print(df.groupby([args.split_col, args.case_col]).size())

    # “normal” case sanity: should be mostly label=0 (your project assumption)
    normal_mask = df[args.case_col].str.lower().str.contains("normal")
    if normal_mask.any():
        df_norm = df[normal_mask].copy()
        norm_pos = int((df_norm[args.label_col] == 1).sum())
        print("\n=== NORMAL CASE LABEL SANITY ===")
        print("normal rows:", len(df_norm), "normal pos rows(label=1):", norm_pos, "pos_rate:", float((df_norm[args.label_col] == 1).mean()))
        if norm_pos > 0:
            print("normal videos with ANY label=1 (first 30):")
            bad_v = (
                df_norm.groupby(args.video_col)[args.label_col]
                .max()
                .reset_index()
            )
            bad_v = bad_v[bad_v[args.label_col] == 1][args.video_col].tolist()
            for v in bad_v[:30]:
                print("  -", v)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
