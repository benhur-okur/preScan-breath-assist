# tools/train_loso.py
from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from src.datasets.window_dataset import WindowVideoDataset
from src.models.mobilenet_temporal import MobileNetV3Temporal


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster


@dataclass
class TrainConfig:
    manifest_loso: str
    pooling: str
    input_mode: str            # <-- NEW
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    img_size: int
    num_workers: int
    seed: int
    device: str
    out_dir: str


def compute_pos_weight(train_df: pd.DataFrame) -> float:
    """
    label 1 = hold, label 0 = breathing
    pos_weight = neg/pos
    """
    pos = float((train_df["label"] == 1).sum())
    neg = float((train_df["label"] == 0).sum())
    if pos <= 0:
        return 1.0
    return neg / pos


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: [N] int64
      probs : [N] float32 (sigmoid(logits))
    """
    model.eval()
    ys: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

        y_np = y.detach().cpu().numpy().reshape(-1)
        ys.append(y_np)
        probs.append(p)

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_true = y_true.astype(np.int64)
    p_all = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
    return y_true, p_all


def pick_threshold_guarded(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
    *,
    min_precision: float = 0.15,
    min_pos_rate: float = 0.02,
    max_pos_rate: float = 0.80,
) -> float:
    """
    Choose threshold that maximizes F1 on validation, BUT:
    - avoid degenerate all-positive/all-negative regimes
    - enforce minimal hold precision and predicted positive rate bounds
    """
    best_t = 0.5
    best_f1 = -1.0

    n = len(y_true)
    if n == 0:
        return best_t

    for t in thresholds:
        pred = (probs >= t).astype(np.int64)
        pos_rate = float(pred.mean())

        if pos_rate < min_pos_rate or pos_rate > max_pos_rate:
            continue

        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if precision < min_precision:
            continue

        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    if best_f1 < 0:
        best_t = 0.5
        best_f1 = -1.0
        for t in thresholds:
            pred = (probs >= t).astype(np.int64)
            f1 = float(f1_score(y_true, pred, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

    return best_t


@torch.no_grad()
def evaluate_with_threshold(model: nn.Module, loader: DataLoader, device: str, thr: float) -> Dict:
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob >= thr).long()

        ys.append(y.detach().cpu().numpy().reshape(-1).astype(np.int64))
        ps.append(pred.detach().cpu().numpy().reshape(-1).astype(np.int64))

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=np.int64)

    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
    }


def _make_dataset(
    cfg: TrainConfig,
    split: str,
    fold_key: str,
    train_aug: bool,
) -> WindowVideoDataset:
    """
    Your WindowVideoDataset already supports BOTH test encodings:
      split=="test" OR split==fold_key
    so we always call split="train"/"val"/"test" here.
    """
    return WindowVideoDataset(
        cfg.manifest_loso,
        split=split,
        fold_id=fold_key,
        img_size=cfg.img_size,
        train_aug=train_aug,
        input_mode=cfg.input_mode,
    )


def run_fold(cfg: TrainConfig, fold_idx: int, fold_key: str) -> Dict:
    df = pd.read_csv(cfg.manifest_loso)

    train_df = df[(df["fold_id"] == fold_key) & (df["split"] == "train")].copy()
    val_df   = df[(df["fold_id"] == fold_key) & (df["split"] == "val")].copy()

    test_df_std = df[(df["fold_id"] == fold_key) & (df["split"] == "test")].copy()
    test_df_fk  = df[(df["fold_id"] == fold_key) & (df["split"] == fold_key)].copy()

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError(
            f"Fold {fold_key} has empty train/val. train_rows={len(train_df)} val_rows={len(val_df)}"
        )
    if len(test_df_std) == 0 and len(test_df_fk) == 0:
        raise RuntimeError(
            f"Fold {fold_key} has no test rows (split=='test' OR split==fold_key)."
        )

    train_ds = _make_dataset(cfg, split="train", fold_key=fold_key, train_aug=True)
    val_ds   = _make_dataset(cfg, split="val",   fold_key=fold_key, train_aug=False)
    test_ds  = _make_dataset(cfg, split="test",  fold_key=fold_key, train_aug=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = MobileNetV3Temporal(pooling=cfg.pooling, pretrained=True).to(cfg.device)

    pos_w = compute_pos_weight(train_df)
    pos_weight_t = torch.tensor([pos_w], device=cfg.device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in cfg.device))

    fold_dir = os.path.join(cfg.out_dir, f"fold_{fold_idx:02d}_{fold_key}")
    os.makedirs(fold_dir, exist_ok=True)

    best_val_f1_05 = -1.0
    best_path = os.path.join(fold_dir, "best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        running = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=("cuda" in cfg.device)):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(1, n_batches)
        val_metrics_05 = evaluate_with_threshold(model, val_loader, cfg.device, thr=0.5)

        dt = time.time() - t0
        print(
            f"[fold {fold_idx:02d} | {fold_key}] "
            f"epoch {epoch:02d}/{cfg.epochs} | loss={train_loss:.4f} | "
            f"val_f1@0.5={val_metrics_05['f1']:.4f} | {dt:.1f}s | pos_w={pos_w:.3f} | "
            f"input_mode={cfg.input_mode}"
        )

        if val_metrics_05["f1"] > best_val_f1_05:
            best_val_f1_05 = float(val_metrics_05["f1"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "pooling": cfg.pooling,
                    "img_size": cfg.img_size,
                    "pos_weight": float(pos_w),
                    "fold_key": fold_key,
                    "input_mode": cfg.input_mode,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])

    y_val, p_val = predict_probs(model, val_loader, cfg.device)
    grid = np.arange(0.10, 0.91, 0.05)

    best_thr = pick_threshold_guarded(
        y_val, p_val, grid,
        min_precision=0.15,
        min_pos_rate=0.02,
        max_pos_rate=0.80,
    )

    val_pred_tuned = (p_val >= best_thr).astype(np.int64)
    val_f1_tuned = float(f1_score(y_val, val_pred_tuned, zero_division=0))

    test_050 = evaluate_with_threshold(model, test_loader, cfg.device, thr=0.5)
    test_tuned = evaluate_with_threshold(model, test_loader, cfg.device, thr=best_thr)

    y_test, p_test = predict_probs(model, test_loader, cfg.device)
    pred_050 = (p_test >= 0.5).astype(np.int64)
    pred_tuned = (p_test >= best_thr).astype(np.int64)

    report_050 = classification_report(y_test, pred_050, digits=4, zero_division=0)
    report_tuned = classification_report(y_test, pred_tuned, digits=4, zero_division=0)

    out = {
        "fold_idx": fold_idx,
        "fold_key": fold_key,
        "pooling": cfg.pooling,
        "input_mode": cfg.input_mode,
        "pos_weight": float(pos_w),

        "best_val_f1_at_0.5": float(best_val_f1_05),

        "best_val_threshold": float(best_thr),
        "val_threshold_grid": grid.tolist(),
        "val_f1_tuned": float(val_f1_tuned),

        "test_f1_050": float(test_050["f1"]),
        "test_confusion_050": test_050["confusion"],
        "test_f1_tuned": float(test_tuned["f1"]),
        "test_confusion_tuned": test_tuned["confusion"],
        "test_n": int(test_tuned["n"]),

        "report_050": report_050,
        "report_tuned": report_tuned,
    }

    with open(os.path.join(fold_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(fold_dir, "test_report_tuned.txt"), "w", encoding="utf-8") as f:
        f.write(report_tuned + "\n")

    with open(os.path.join(fold_dir, "test_report_0p5.txt"), "w", encoding="utf-8") as f:
        f.write(report_050 + "\n")

    print(
        f"[fold {fold_idx:02d} | {fold_key}] DONE | "
        f"val_best_thr={best_thr:.2f} val_f1_tuned={val_f1_tuned:.4f} | "
        f"test_f1@0.5={test_050['f1']:.4f} test_f1_tuned={test_tuned['f1']:.4f} | saved={fold_dir}"
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-loso", required=True)
    p.add_argument("--pooling", choices=["mean", "max"], default="mean")
    p.add_argument("--input-mode", choices=["rgb", "diff"], default="rgb")  # <-- NEW
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs/loso")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig(
        manifest_loso=args.manifest_loso,
        pooling=args.pooling,
        input_mode=args.input_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
        out_dir=args.out_dir,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    df = pd.read_csv(cfg.manifest_loso)
    fold_keys: List[str] = sorted(df["fold_id"].dropna().unique().tolist())

    print(f"[INFO] device={device} | folds={fold_keys} | pooling={cfg.pooling} | input_mode={cfg.input_mode} | epochs={cfg.epochs}")

    all_results = []
    for fold_idx, fold_key in enumerate(fold_keys, start=1):
        all_results.append(run_fold(cfg, fold_idx, fold_key))

    test_f1s_tuned = [r["test_f1_tuned"] for r in all_results]
    test_f1s_050   = [r["test_f1_050"] for r in all_results]
    val_f1s_tuned  = [r["val_f1_tuned"] for r in all_results]
    thr_list       = [r["best_val_threshold"] for r in all_results]

    summary = {
        "pooling": cfg.pooling,
        "input_mode": cfg.input_mode,
        "folds": fold_keys,

        "val_f1_tuned_mean": float(np.mean(val_f1s_tuned)) if val_f1s_tuned else 0.0,
        "val_f1_tuned_std": float(np.std(val_f1s_tuned)) if val_f1s_tuned else 0.0,

        "test_f1_tuned_mean": float(np.mean(test_f1s_tuned)) if test_f1s_tuned else 0.0,
        "test_f1_tuned_std": float(np.std(test_f1s_tuned)) if test_f1s_tuned else 0.0,

        "test_f1_050_mean": float(np.mean(test_f1s_050)) if test_f1s_050 else 0.0,
        "test_f1_050_std": float(np.std(test_f1s_050)) if test_f1s_050 else 0.0,

        "best_val_thresholds": thr_list,
        "per_fold": all_results,
        "config": asdict(cfg),
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(cfg.out_dir, f"summary_{cfg.pooling}_{cfg.input_mode}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LOSO SUMMARY (Threshold Guarded) ===")
    print(f"pooling={cfg.pooling}")
    print(f"input_mode={cfg.input_mode}")
    print(f"val_f1_tuned : mean={summary['val_f1_tuned_mean']:.4f} std={summary['val_f1_tuned_std']:.4f}")
    print(f"test_f1_tuned: mean={summary['test_f1_tuned_mean']:.4f} std={summary['test_f1_tuned_std']:.4f}")
    print(f"test_f1@0.5  : mean={summary['test_f1_050_mean']:.4f} std={summary['test_f1_050_std']:.4f}")
    print(f"thresholds   : {['%.2f' % t for t in summary['best_val_thresholds']]}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
