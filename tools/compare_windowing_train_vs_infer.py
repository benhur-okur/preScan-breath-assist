# tools/compare_windowing_train_vs_infer.py
from __future__ import annotations

import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--infer-manifest", required=True)
    ap.add_argument("--video", required=True, help="substring match")
    args = ap.parse_args()

    tr = pd.read_csv(args.train_manifest)
    inf = pd.read_csv(args.infer_manifest)

    trv = tr[tr["front_video_path"].astype(str).str.contains(args.video, regex=False)].copy()
    infv = inf[inf["front_video_path"].astype(str).str.contains(args.video, regex=False)].copy()

    if len(trv) == 0:
        raise SystemExit("No train rows matched.")
    if len(infv) == 0:
        raise SystemExit("No infer rows matched.")

    trv = trv.sort_values("start_frame")
    infv = infv.sort_values("start_frame")

    tr_starts = trv["start_frame"].astype(int).to_list()
    inf_starts = infv["start_frame"].astype(int).to_list()

    print("=== TRAIN windows ===")
    print("rows:", len(tr_starts), "min:", min(tr_starts), "max:", max(tr_starts))
    print("unique strides (first 30):", sorted(set([tr_starts[i+1]-tr_starts[i] for i in range(min(len(tr_starts)-1, 30))])))

    print("\n=== INFER windows ===")
    print("rows:", len(inf_starts), "min:", min(inf_starts), "max:", max(inf_starts))
    print("unique strides (first 30):", sorted(set([inf_starts[i+1]-inf_starts[i] for i in range(min(len(inf_starts)-1, 30))])))

    tr_set = set(tr_starts)
    inf_set = set(inf_starts)
    overlap = len(tr_set & inf_set)

    print("\n=== OVERLAP ===")
    print("overlap start_frame count:", overlap)
    print("train_only:", len(tr_set - inf_set))
    print("infer_only:", len(inf_set - tr_set))

if __name__ == "__main__":
    main()
