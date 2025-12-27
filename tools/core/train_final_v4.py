# tools/train_final_v4.py
from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from src.datasets.window_dataset import WindowVideoDataset
from src.models.mobilenet_temporal import MobileNetV3Temporal


@dataclass
class FinalMeta:
    manifest_csv: str
    out_dir: str
    pooling: str
    input_mode: str
    epochs: int
    batch_size: int
    img_size: int
    num_workers: int
    lr: float
    weight_decay: float
    seed: int
    loss_type: str
    pos_weight: float
    focal_gamma: float
    best_val_f1: float
    best_epoch: int
    best_threshold: float
    train_rows: int
    val_rows: int
    train_pos_rate: float
    val_pos_rate: float
    started_at: float
    finished_at: float


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(train_df: pd.DataFrame) -> float:
    pos = float((train_df["label"] == 1).sum())
    neg = float((train_df["label"] == 0).sum())
    if pos <= 0:
        return 1.0
    return neg / pos


class FocalLossWithLogits(nn.Module):
    """
    Binary focal loss on logits.
    Uses alpha via pos_weight-style (optional) and gamma focusing.
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B] or [B,1], targets: [B] float {0,1}
        logits = logits.view(-1)
        targets = targets.view(-1)

        # BCE with logits but get per-sample
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        # p_t = p if y=1 else (1-p)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal = (1.0 - p_t).pow(self.gamma) * bce
        return focal.mean()


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str, thr: float) -> Dict:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).view(-1)
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        ys.append(y.detach().cpu().numpy().reshape(-1).astype(np.int64))
        ps.append(p.reshape(-1))

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    probs = np.concatenate(ps) if ps else np.array([], dtype=np.float32)

    if len(y_true) == 0:
        return {"f1": 0.0, "cm": [[0, 0], [0, 0]], "pos_rate_pred": 0.0, "pos_rate_true": 0.0}

    pred = (probs >= float(thr)).astype(np.int64)
    f1 = float(f1_score(y_true, pred, zero_division=0))
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    return {
        "f1": f1,
        "cm": cm,
        "pos_rate_pred": float(pred.mean()),
        "pos_rate_true": float(y_true.mean()),
    }


@torch.no_grad()
def pick_best_threshold(model: nn.Module, loader: DataLoader, device: str, grid: np.ndarray) -> Tuple[float, float]:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).view(-1)
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        ys.append(y.detach().cpu().numpy().reshape(-1).astype(np.int64))
        ps.append(p.reshape(-1))

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    probs = np.concatenate(ps) if ps else np.array([], dtype=np.float32)

    if len(y_true) == 0:
        return 0.5, 0.0

    best_thr = 0.5
    best_f1 = -1.0
    for t in grid:
        pred = (probs >= float(t)).astype(np.int64)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return best_thr, float(best_f1)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--final-fold-id", default="final")
    ap.add_argument("--pooling", choices=["mean", "max"], default="max")

    # IMPORTANT: V4 = RGB training (fixes "diff saturates" failure mode)
    ap.add_argument("--input-mode", choices=["rgb", "diff"], default="rgb")

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--focal-gamma", type=float, default=2.0)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.manifest_csv)
    df = df[df["fold_id"] == args.final_fold_id].copy()
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise SystemExit("Train/Val split empty. Check manifest_final_train_v3.csv creation.")

    pos_weight_val = compute_pos_weight(train_df)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    # Datasets
    train_ds = WindowVideoDataset(
        manifest_csv=args.manifest_csv,
        split="train",
        fold_id=args.final_fold_id,
        img_size=args.img_size,
        train_aug=True,
        input_mode=args.input_mode,
    )
    val_ds = WindowVideoDataset(
        manifest_csv=args.manifest_csv,
        split="val",
        fold_id=args.final_fold_id,
        img_size=args.img_size,
        train_aug=False,
        input_mode=args.input_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MobileNetV3Temporal(pooling=args.pooling, pretrained=True).to(device)

    # Loss
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = FocalLossWithLogits(pos_weight=pos_weight, gamma=args.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Threshold search grid for val
    thr_grid = np.arange(0.10, 0.96, 0.05)

    best_val_f1 = -1.0
    best_epoch = -1
    best_thr = 0.5
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x).view(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        # pick threshold on val for this epoch
        thr, val_f1 = pick_best_threshold(model, val_loader, device, thr_grid)
        metrics = eval_epoch(model, val_loader, device, thr)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(
            f"[E{epoch:02d}] train_loss={avg_loss:.4f}  "
            f"val_f1={metrics['f1']:.4f} (thr={thr:.2f})  "
            f"val_pos_pred={metrics['pos_rate_pred']:.3f} val_pos_true={metrics['pos_rate_true']:.3f}  "
            f"cm={metrics['cm']}"
        )

        # Save last
        last_path = os.path.join(args.out_dir, "final_last.pt")
        torch.save({"model": model.state_dict()}, last_path)

        # Save best by val_f1
        if metrics["f1"] > best_val_f1:
            best_val_f1 = float(metrics["f1"])
            best_epoch = int(epoch)
            best_thr = float(thr)
            best_path = os.path.join(args.out_dir, "final_best.pt")
            torch.save({"model": model.state_dict()}, best_path)

    finished = time.time()

    meta = FinalMeta(
        manifest_csv=args.manifest_csv,
        out_dir=args.out_dir,
        pooling=args.pooling,
        input_mode=args.input_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        loss_type=args.loss,
        pos_weight=float(pos_weight_val),
        focal_gamma=float(args.focal_gamma),
        best_val_f1=float(best_val_f1),
        best_epoch=int(best_epoch),
        best_threshold=float(best_thr),
        train_rows=int(len(train_df)),
        val_rows=int(len(val_df)),
        train_pos_rate=float(train_df["label"].mean()),
        val_pos_rate=float(val_df["label"].mean()),
        started_at=float(started),
        finished_at=float(finished),
    )
    meta_path = os.path.join(args.out_dir, "final_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    print("\n=== FINAL TRAIN V4 DONE ===")
    print(f"best_val_f1={best_val_f1:.4f}  best_epoch={best_epoch}  best_thr={best_thr:.2f}")
    print(f"saved: {os.path.join(args.out_dir, 'final_best.pt')}")
    print(f"saved: {os.path.join(args.out_dir, 'final_last.pt')}")
    print(f"saved: {meta_path}")


if __name__ == "__main__":
    main()
