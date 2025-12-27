# tools/eval_single_video_from_manifest.py
from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.window_dataset import WindowVideoDataset
from src.models.mobilenet_temporal import MobileNetV3Temporal


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    ps = []
    for x, _y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        ps.append(p)
    return np.concatenate(ps).astype(np.float32) if ps else np.array([], dtype=np.float32)


def postprocess_min_consecutive(pred: np.ndarray, min_consecutive: int) -> np.ndarray:
    if min_consecutive <= 1:
        return pred.copy()
    out = pred.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == 1:
            j = i
            while j < n and out[j] == 1:
                j += 1
            if (j - i) < min_consecutive:
                out[i:j] = 0
            i = j
        else:
            i += 1
    return out


def extract_segments(df_preds: pd.DataFrame) -> pd.DataFrame:
    pred = df_preds["pred_hold"].to_numpy(dtype=int)
    starts = df_preds["start_frame"].to_numpy(dtype=int)
    win = df_preds["window_frames"].to_numpy(dtype=int)

    segs = []
    n = len(pred)
    i = 0
    while i < n:
        if pred[i] == 1:
            j = i
            while j < n and pred[j] == 1:
                j += 1
            segs.append({
                "start_window_idx": int(i),
                "end_window_idx": int(j - 1),
                "start_frame": int(starts[i]),
                "end_frame_exclusive": int(starts[j - 1] + win[j - 1]),
                "length_windows": int(j - i),
            })
            i = j
        else:
            i += 1

    return pd.DataFrame(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="data/windows/manifest_final_train_v3.csv")
    ap.add_argument("--video", required=True, help="front_video_path exact match (or substring)")
    ap.add_argument("--ckpt", required=True, help="runs/final_v4/final_best.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--input-mode", choices=["rgb", "diff"], default="rgb")
    ap.add_argument("--pooling", choices=["mean", "max"], default="max")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--min-consecutive", type=int, default=13, help="diagnostic: use same as min_hold_sec mapping")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.manifest)

    # Match by substring OR exact
    mask = df["front_video_path"].astype(str).str.contains(str(args.video), regex=False)
    dfv = df[mask].copy()
    if len(dfv) == 0:
        raise SystemExit(f"No rows matched video filter: {args.video}")

    dfv = dfv.sort_values("start_frame").reset_index(drop=True)

    # Build a mini-manifest compatible with WindowVideoDataset "test" split
    dfm = dfv.copy()
    dfm["fold_id"] = "inference"
    dfm["split"] = "test"
    mini_path = os.path.join(args.out_dir, "mini_manifest_from_train.csv")
    dfm.to_csv(mini_path, index=False, encoding="utf-8")
    print(f"[OK] wrote mini manifest: {mini_path} rows={len(dfm)}")

    ds = WindowVideoDataset(
        manifest_csv=mini_path,
        split="test",
        fold_id="inference",
        img_size=args.img_size,
        train_aug=False,
        input_mode=args.input_mode,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = MobileNetV3Temporal(pooling=args.pooling, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    probs = predict_probs(model, loader, device)
    pred_raw = (probs >= float(args.threshold)).astype(np.int64)
    pred_post = postprocess_min_consecutive(pred_raw, int(args.min_consecutive))

    out_df = pd.DataFrame({
        "start_frame": dfm["start_frame"].astype(int),
        "window_frames": dfm["window_frames"].astype(int),
        "prob_hold": probs.astype(float),
        "pred_hold_raw": pred_raw.astype(int),
        "pred_hold": pred_post.astype(int),
        "video_path": dfm["front_video_path"].astype(str),
        "threshold": float(args.threshold),
        "min_consecutive": int(args.min_consecutive),
        "input_mode": args.input_mode,
    })

    pred_csv = os.path.join(args.out_dir, "predictions_from_train_windows.csv")
    out_df.to_csv(pred_csv, index=False, encoding="utf-8")
    print(f"[OK] saved: {pred_csv}")

    seg_df = extract_segments(out_df)
    seg_csv = os.path.join(args.out_dir, "segments_from_train_windows.csv")
    seg_df.to_csv(seg_csv, index=False, encoding="utf-8")
    print(f"[OK] saved: {seg_csv} segments={len(seg_df)}")

    if args.plot:
        import matplotlib.pyplot as plt
        x = np.arange(len(out_df))
        plt.figure()
        plt.plot(x, out_df["prob_hold"].values)
        plt.axhline(float(args.threshold), linestyle="--")
        plt.title("Hold probability over windows (TRAIN windows)")
        plt.xlabel("Window index")
        plt.ylabel("P(hold)")
        pth = os.path.join(args.out_dir, "probability_plot_from_train_windows.png")
        plt.savefig(pth, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved: {pth}")


if __name__ == "__main__":
    main()
