# tools/temperature_scaling_final.py
from __future__ import annotations

import os
import json
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import brier_score_loss, log_loss

from src.datasets.window_dataset import WindowVideoDataset
from src.models.mobilenet_temporal import MobileNetV3Temporal


@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy().reshape(-1)
        y_np = y.detach().cpu().numpy().reshape(-1)
        logits_all.append(logits)
        y_all.append(y_np)
    logits = np.concatenate(logits_all) if logits_all else np.array([], dtype=np.float32)
    y_true = np.concatenate(y_all) if y_all else np.array([], dtype=np.int64)
    y_true = y_true.astype(np.int64)
    return logits, y_true


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    # Expected Calibration Error
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        p_bin = probs[mask].mean()
        acc_bin = y[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - p_bin)
    return float(ece)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """
    Fit T by minimizing NLL on val.
    We optimize a single scalar parameter.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # parameterize T via log_T to keep T>0
    log_T = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = bce(logits_t / T, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_T).detach().cpu().item())
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fold-id", default="final")
    ap.add_argument("--pooling", choices=["mean", "max"], default="max")
    ap.add_argument("--input-mode", choices=["rgb", "diff"], default="diff")

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--out-json", default="runs/final_temperature.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    model.load_state_dict(ckpt["model"])

    logits, y_true = collect_logits_labels(model, val_loader, device=device)
    if len(y_true) == 0:
        raise SystemExit("VAL set returned 0 samples. Check manifest.")
    if len(np.unique(y_true)) < 2:
        raise SystemExit("VAL set has only one class. Temperature scaling is meaningless. Fix val split.")

    # before
    p_before = sigmoid(logits)
    nll_before = float(log_loss(y_true, p_before, labels=[0, 1]))
    brier_before = float(brier_score_loss(y_true, p_before))
    ece_before = ece_score(p_before, y_true)

    # fit T
    T = fit_temperature(logits, y_true)

    # after
    p_after = sigmoid(logits / T)
    nll_after = float(log_loss(y_true, p_after, labels=[0, 1]))
    brier_after = float(brier_score_loss(y_true, p_after))
    ece_after = ece_score(p_after, y_true)

    out = {
        "manifest_csv": args.manifest_csv,
        "ckpt": args.ckpt,
        "fold_id": args.fold_id,
        "pooling": args.pooling,
        "input_mode": args.input_mode,
        "device": device,
        "temperature": float(T),
        "val_metrics": {
            "n_val": int(len(y_true)),
            "pos_rate_true": float(y_true.mean()),
            "nll_before": nll_before,
            "nll_after": nll_after,
            "brier_before": brier_before,
            "brier_after": brier_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== TEMPERATURE SCALING (VAL) ===")
    print(f"T={T:.4f}")
    print(f"NLL:   {nll_before:.4f} -> {nll_after:.4f}")
    print(f"Brier: {brier_before:.4f} -> {brier_after:.4f}")
    print(f"ECE:   {ece_before:.4f} -> {ece_after:.4f}")
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
