# tools/eval_final_threshold.py
from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.datasets.window_dataset import WindowVideoDataset
from src.models.mobilenet_temporal import MobileNetV3Temporal


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        y_np = y.detach().cpu().numpy().reshape(-1).astype(np.int64)
        ys.append(y_np)
        ps.append(p)
    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    probs  = np.concatenate(ps) if ps else np.array([], dtype=np.float32)
    return y_true, probs


def sweep_thresholds(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in grid:
        pred = (probs >= t).astype(np.int64)
        rows.append({
            "threshold": float(t),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "pos_rate": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def pick_threshold_for_precision(sweep_df: pd.DataFrame, min_precision: float) -> float:
    ok = sweep_df[sweep_df["precision"] >= float(min_precision)].copy()
    if len(ok) == 0:
        return float(sweep_df.sort_values("precision", ascending=False).iloc[0]["threshold"])
    # among those, pick the best recall (or f1). Here: best F1 under constraint.
    ok = ok.sort_values(["f1", "recall"], ascending=False)
    return float(ok.iloc[0]["threshold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="data/windows/manifest_final_train_v3.csv")
    ap.add_argument("--ckpt", required=True, help="runs/final_model_v3/final_best.pt or path")
    ap.add_argument("--pooling", choices=["mean", "max"], default="max")
    ap.add_argument("--input-mode", choices=["rgb", "diff"], default="diff")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--min-precision", type=float, default=0.90, help="for product-style threshold pick")
    ap.add_argument("--out-dir", default="runs/eval_final_threshold")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # VAL loader
    val_ds = WindowVideoDataset(
        args.manifest,
        split="val",
        fold_id="final",        # final fold id
        img_size=args.img_size,
        train_aug=False,
        input_mode=args.input_mode,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MobileNetV3Temporal(pooling=args.pooling, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    y_true, probs = predict_probs(model, val_loader, device)
    if len(y_true) == 0:
        raise SystemExit("VAL set is empty; check manifest/fold_id/split.")

    pr_auc = float(average_precision_score(y_true, probs))
    try:
        roc_auc = float(roc_auc_score(y_true, probs))
    except Exception:
        roc_auc = float("nan")

    # threshold sweep
    grid = np.linspace(0.0, 1.0, 201)
    sweep_df = sweep_thresholds(y_true, probs, grid)
    best_row = sweep_df.sort_values("f1", ascending=False).iloc[0]
    best_f1_thr = float(best_row["threshold"])

    prec_target_thr = pick_threshold_for_precision(sweep_df, args.min_precision)

    # also PR curve points (optional export)
    prec_curve, rec_curve, thr_curve = precision_recall_curve(y_true, probs)

    out = {
        "ckpt": args.ckpt,
        "pooling": args.pooling,
        "input_mode": args.input_mode,
        "n_val": int(len(y_true)),
        "pos_rate_val": float(y_true.mean()),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "best_f1_threshold": best_f1_thr,
        "best_f1": float(best_row["f1"]),
        "best_f1_precision": float(best_row["precision"]),
        "best_f1_recall": float(best_row["recall"]),
        "precision_target": float(args.min_precision),
        "threshold_for_precision_target": prec_target_thr,
    }

    # save files
    sweep_path = os.path.join(args.out_dir, "threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False, encoding="utf-8")

    pr_path = os.path.join(args.out_dir, "pr_curve.csv")
    pr_df = pd.DataFrame({
        "precision": prec_curve,
        "recall": rec_curve,
        "threshold": np.concatenate([thr_curve, [np.nan]]),  # curve has len-1 thresholds
    })
    pr_df.to_csv(pr_path, index=False, encoding="utf-8")

    out_path = os.path.join(args.out_dir, "eval_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=== FINAL VAL EVAL ===")
    print(json.dumps(out, indent=2))
    print(f"[OK] saved: {out_path}")
    print(f"[OK] saved: {sweep_path}")
    print(f"[OK] saved: {pr_path}")


if __name__ == "__main__":
    main()
