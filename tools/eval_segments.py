# tools/eval_segments.py
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from src.datasets.window_dataset import WindowVideoDataset
from src.models.mobilenet_temporal import MobileNetV3Temporal


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).view(-1)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).astype(np.float32)
        ys.append(y.detach().cpu().numpy().reshape(-1).astype(np.int64))
        ps.append(p)
    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    probs = np.concatenate(ps) if ps else np.array([], dtype=np.float32)
    return y_true, probs


def segments_from_windows(df: pd.DataFrame, label_col: str) -> List[Tuple[int, int]]:
    """
    Build frame segments [start, end_exclusive] from consecutive windows where label_col==1.
    Expects start_frame, window_frames columns.
    """
    s = df["start_frame"].to_numpy(dtype=int)
    w = df["window_frames"].to_numpy(dtype=int)
    y = df[label_col].to_numpy(dtype=int)

    segs: List[Tuple[int, int]] = []
    n = len(y)
    i = 0
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            start = int(s[i])
            end = int(s[j - 1] + w[j - 1])
            segs.append((start, end))
            i = j
        else:
            i += 1
    return segs


def match_segments(pred: List[Tuple[int, int]], gt: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    A predicted segment is TP if it overlaps ANY unmatched GT segment.
    Greedy matching by order.
    """
    gt_used = [False] * len(gt)
    tp = 0
    fp = 0

    for (ps, pe) in pred:
        matched = False
        for k, (gs, ge) in enumerate(gt):
            if gt_used[k]:
                continue
            # overlap?
            if max(ps, gs) < min(pe, ge):
                gt_used[k] = True
                matched = True
                tp += 1
                break
        if not matched:
            fp += 1

    fn = int(gt_used.count(False))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {"seg_tp": tp, "seg_fp": fp, "seg_fn": fn, "seg_precision": prec, "seg_recall": rec, "seg_f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--final-fold-id", default="final")
    ap.add_argument("--pooling", choices=["mean", "max"], default="max")
    ap.add_argument("--input-mode", choices=["rgb", "diff"], default="rgb")

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--out-dir", default="runs/eval_segments")
    ap.add_argument("--thr-min", type=float, default=0.10)
    ap.add_argument("--thr-max", type=float, default=0.95)
    ap.add_argument("--thr-step", type=float, default=0.05)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (VAL)
    val_ds = WindowVideoDataset(
        manifest_csv=args.manifest,
        split="val",
        fold_id=args.final_fold_id,
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

    # Model
    model = MobileNetV3Temporal(pooling=args.pooling, pretrained=True).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    y_true, probs = predict_probs(model, val_loader, device)

    # Build a VAL df aligned with dataset filtering (same logic as WindowVideoDataset)
    df = pd.read_csv(args.manifest)
    df = df[df["fold_id"] == args.final_fold_id].copy()
    df = df[df["split"] == "val"].copy()
    df = df.reset_index(drop=True)

    if len(df) != len(probs):
        raise RuntimeError(f"Row mismatch: manifest_val_rows={len(df)} vs probs={len(probs)}")

    df["prob"] = probs
    df["y_true"] = y_true

    thresholds = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)

    rows = []
    best = {"thr": 0.5, "seg_f1": -1.0}

    for t in thresholds:
        pred = (probs >= float(t)).astype(np.int64)
        df["y_pred"] = pred

        # window-level
        wf1 = float(f1_score(y_true, pred, zero_division=0))
        wprec = float(precision_score(y_true, pred, zero_division=0))
        wrec = float(recall_score(y_true, pred, zero_division=0))
        cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()

        # segment-level
        gt_segs = segments_from_windows(df, "y_true")
        pr_segs = segments_from_windows(df, "y_pred")
        segm = match_segments(pr_segs, gt_segs)

        out = {
            "thr": float(t),
            "win_f1": wf1,
            "win_precision": wprec,
            "win_recall": wrec,
            "cm00": cm[0][0], "cm01": cm[0][1], "cm10": cm[1][0], "cm11": cm[1][1],
            "n_gt_segments": len(gt_segs),
            "n_pred_segments": len(pr_segs),
            **segm,
            "pred_pos_rate": float(pred.mean()),
        }
        rows.append(out)

        if segm["seg_f1"] > best["seg_f1"]:
            best = {"thr": float(t), "seg_f1": float(segm["seg_f1"])}

    sweep_df = pd.DataFrame(rows)
    sweep_csv = os.path.join(args.out_dir, "threshold_sweep_segments.csv")
    sweep_df.to_csv(sweep_csv, index=False, encoding="utf-8")

    summary = {
        "manifest": args.manifest,
        "ckpt": args.ckpt,
        "pooling": args.pooling,
        "input_mode": args.input_mode,
        "thresholds": thresholds.tolist(),
        "best_by_segment_f1": best,
        "n_val_rows": int(len(df)),
        "val_pos_rate_true": float(df["y_true"].mean()) if len(df) else 0.0,
    }
    summary_json = os.path.join(args.out_dir, "eval_segments_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== EVAL SEGMENTS DONE ===")
    print(f"saved: {sweep_csv}")
    print(f"saved: {summary_json}")
    print(f"best_thr_by_segment_f1: thr={best['thr']:.2f} seg_f1={best['seg_f1']:.4f}")


if __name__ == "__main__":
    main()
