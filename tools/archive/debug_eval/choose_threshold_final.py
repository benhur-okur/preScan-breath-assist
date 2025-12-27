# tools/choose_threshold_final.py
from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from src.datasets.window_dataset import WindowVideoDataset
from src.models.mobilenet_temporal import MobileNetV3Temporal


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ls = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy().reshape(-1)
        y_np = y.detach().cpu().numpy().reshape(-1)
        ys.append(y_np)
        ls.append(logits)
    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_true = y_true.astype(np.int64)
    logits = np.concatenate(ls) if ls else np.array([], dtype=np.float32)
    return y_true, logits


def sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class ThrResult:
    thr: float
    f1: float
    precision: float
    recall: float
    pos_rate: float


def pick_threshold_guarded(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray,
    *,
    min_precision: float,
    min_pos_rate: float,
    max_pos_rate: float,
) -> Tuple[float, List[ThrResult]]:
    results: List[ThrResult] = []
    best_thr = 0.5
    best_f1 = -1.0

    if len(y_true) == 0:
        return best_thr, results

    for t in grid:
        pred = (probs >= t).astype(np.int64)
        pos_rate = float(pred.mean())

        prec = float(precision_score(y_true, pred, zero_division=0))
        rec = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))

        results.append(ThrResult(float(t), f1, prec, rec, pos_rate))

        # guardrails
        if pos_rate < min_pos_rate or pos_rate > max_pos_rate:
            continue
        if prec < min_precision:
            continue

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)

    # fallback: best F1 (if guardrails eliminate all)
    if best_f1 < 0:
        best_thr = 0.5
        best_f1 = -1.0
        for r in results:
            if r.f1 > best_f1:
                best_f1 = r.f1
                best_thr = r.thr

    return best_thr, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True, help="final manifest with split=train/val and fold_id=final")
    ap.add_argument("--ckpt", required=True, help="final_best.pt path")
    ap.add_argument("--fold-id", default="final", help="fold_id value inside the manifest")
    ap.add_argument("--pooling", choices=["mean", "max"], default="max")
    ap.add_argument("--input-mode", choices=["rgb", "diff"], default="diff")

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--grid-start", type=float, default=0.05)
    ap.add_argument("--grid-end", type=float, default=0.99)
    ap.add_argument("--grid-step", type=float, default=0.01)

    # guardrails (professional defaults; adjust if needed)
    ap.add_argument("--min-precision", type=float, default=0.80)
    ap.add_argument("--min-pos-rate", type=float, default=0.02)
    ap.add_argument("--max-pos-rate", type=float, default=0.80)

    ap.add_argument("--out-json", default="runs/final_threshold.json")
    ap.add_argument("--out-csv", default="runs/final_threshold_grid.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Basic sanity on manifest
    df = pd.read_csv(args.manifest_csv)
    if "split" not in df.columns or "fold_id" not in df.columns:
        raise SystemExit("manifest must contain columns: split, fold_id")

    df_fold = df[df["fold_id"].astype(str) == str(args.fold_id)]
    if len(df_fold) == 0:
        raise SystemExit(f"No rows found for fold_id={args.fold_id} in {args.manifest_csv}")

    if not {"train", "val"}.issubset(set(df_fold["split"].unique().tolist())):
        raise SystemExit("final manifest must have both split=train and split=val")

    # VAL loader
    val_ds = WindowVideoDataset(
        manifest_csv=args.manifest_csv,
        split="val",
        fold_id=args.fold_id,
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

    # Load model
    model = MobileNetV3Temporal(pooling=args.pooling, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if "model" not in ckpt:
        raise SystemExit("Checkpoint must contain key 'model' (state_dict).")
    model.load_state_dict(ckpt["model"])

    y_true, logits = predict_logits(model, val_loader, device=device)
    if len(y_true) == 0:
        raise SystemExit("VAL set returned 0 samples. Check manifest split rules.")
    if len(np.unique(y_true)) < 2:
        raise SystemExit("VAL set has only one class. Threshold selection is meaningless. Fix your val split.")

    probs = sigmoid(logits)

    grid = np.arange(args.grid_start, args.grid_end + 1e-9, args.grid_step)
    best_thr, results = pick_threshold_guarded(
        y_true,
        probs,
        grid,
        min_precision=float(args.min_precision),
        min_pos_rate=float(args.min_pos_rate),
        max_pos_rate=float(args.max_pos_rate),
    )

    # Final metrics at chosen threshold
    pred = (probs >= best_thr).astype(np.int64)
    f1 = float(f1_score(y_true, pred, zero_division=0))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    pos_rate = float(pred.mean())

    # Save grid CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    grid_df = pd.DataFrame([r.__dict__ for r in results])
    grid_df.to_csv(args.out_csv, index=False, encoding="utf-8")

    out = {
        "manifest_csv": args.manifest_csv,
        "ckpt": args.ckpt,
        "fold_id": args.fold_id,
        "pooling": args.pooling,
        "input_mode": args.input_mode,
        "device": device,
        "grid": {
            "start": args.grid_start,
            "end": args.grid_end,
            "step": args.grid_step,
            "count": int(len(grid)),
        },
        "guardrails": {
            "min_precision": args.min_precision,
            "min_pos_rate": args.min_pos_rate,
            "max_pos_rate": args.max_pos_rate,
        },
        "best_threshold": float(best_thr),
        "val_metrics_at_best_threshold": {
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "pred_pos_rate": pos_rate,
            "n_val": int(len(y_true)),
            "pos_rate_true": float(y_true.mean()),
        },
        "artifacts": {
            "grid_csv": args.out_csv,
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== FINAL THRESHOLD (VAL) ===")
    print(f"best_threshold={best_thr:.3f}")
    print(f"VAL: f1={f1:.4f}  precision={prec:.4f}  recall={rec:.4f}  pred_pos_rate={pos_rate:.4f}  N={len(y_true)}")
    print(f"saved: {args.out_json}")
    print(f"grid : {args.out_csv}")


if __name__ == "__main__":
    main()
