# tools/train_loso.py
from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List

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
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    img_size: int
    num_workers: int
    seed: int
    device: str
    out_dir: str


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    model.eval()
    ys = []
    ps = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        # y may be shape [B] or [B,1]; keep it consistent
        y = y.to(device, non_blocking=True)

        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()

        ys.append(y.long().cpu().numpy().reshape(-1))
        ps.append(pred.cpu().numpy().reshape(-1))

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=np.int64)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "f1": f1,
        "confusion": cm,
        "n": int(len(y_true)),
    }


def compute_pos_weight(train_df: pd.DataFrame) -> float:
    # label 1 = hold, label 0 = breathing
    pos = float((train_df["label"] == 1).sum())
    neg = float((train_df["label"] == 0).sum())
    if pos <= 0:
        return 1.0
    return neg / pos


def run_fold(cfg: TrainConfig, fold_idx: int, fold_key: str) -> Dict:
    """
    fold_key: manifest'teki fold_id string değeri (örn 'fold_1_test_benhur')
    fold_idx: sadece output klasör isimlendirme için 1..N
    """
    df = pd.read_csv(cfg.manifest_loso)

    # Bu fold'a ait satırlar: split ∈ {train,val,fold_key}
    # (manifest'te test split'i fold_key olarak yazılmış)
    df_fold = df[(df["split"].isin(["train", "val"])) | (df["split"] == fold_key)].copy()

    # Datasets
    # NOT: WindowVideoDataset'in fold_id parametresi varsa, artık fold_key (string) veriyoruz.
    train_ds = WindowVideoDataset(cfg.manifest_loso, split="train", fold_id=fold_key, img_size=cfg.img_size, train_aug=True)
    val_ds   = WindowVideoDataset(cfg.manifest_loso, split="val",   fold_id=fold_key, img_size=cfg.img_size, train_aug=False)
    test_ds  = WindowVideoDataset(cfg.manifest_loso, split="test",  fold_id=fold_key, img_size=cfg.img_size, train_aug=False)

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

    # Model
    model = MobileNetV3Temporal(pooling=cfg.pooling, pretrained=True).to(cfg.device)

    # Loss (pos_weight from TRAIN only, in this fold)
    pos_w = compute_pos_weight(df_fold[df_fold["split"] == "train"])
    pos_weight_t = torch.tensor([pos_w], device=cfg.device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=("cuda" in cfg.device))

    fold_dir = os.path.join(cfg.out_dir, f"fold_{fold_idx:02d}_{fold_key}")
    os.makedirs(fold_dir, exist_ok=True)

    best_val_f1 = -1.0
    best_path = os.path.join(fold_dir, "best.pt")

    # Train
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        running = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            # BCEWithLogitsLoss expects float targets
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
        val_metrics = evaluate(model, val_loader, cfg.device)

        dt = time.time() - t0
        print(
            f"[fold {fold_idx:02d} | {fold_key}] "
            f"epoch {epoch:02d}/{cfg.epochs} | loss={train_loss:.4f} | val_f1={val_metrics['f1']:.4f} | {dt:.1f}s"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "pooling": cfg.pooling,
                    "img_size": cfg.img_size,
                    "pos_weight": pos_w,
                    "fold_key": fold_key,
                },
                best_path,
            )

    # Test best
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(model, test_loader, cfg.device)

    # Detailed report
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(cfg.device, non_blocking=True)
            logits = model(x)
            pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().reshape(-1)
            y_true_all.append(y.long().numpy().reshape(-1))
            y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    out = {
        "fold_idx": fold_idx,
        "fold_key": fold_key,
        "pooling": cfg.pooling,
        "best_val_f1": float(best_val_f1),
        "test_f1": float(test_metrics["f1"]),
        "test_confusion": test_metrics["confusion"],
        "test_n": int(test_metrics["n"]),
        "pos_weight": float(pos_w),
        "report": report,
    }

    with open(os.path.join(fold_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(fold_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(
        f"[fold {fold_idx:02d} | {fold_key}] DONE | "
        f"best_val_f1={best_val_f1:.4f} | test_f1={test_metrics['f1']:.4f} | saved={fold_dir}"
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-loso", required=True)
    p.add_argument("--pooling", choices=["mean", "max"], default="mean")
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

    # fold_key'ler string: 'fold_1_test_benhur' gibi
    fold_keys: List[str] = sorted(df["fold_id"].dropna().unique().tolist())

    print(f"[INFO] device={device} | folds={fold_keys} | pooling={cfg.pooling} | epochs={cfg.epochs}")

    all_results = []
    for fold_idx, fold_key in enumerate(fold_keys, start=1):
        all_results.append(run_fold(cfg, fold_idx, fold_key))

    # Summarize
    test_f1s = [r["test_f1"] for r in all_results]
    val_f1s = [r["best_val_f1"] for r in all_results]

    summary = {
        "pooling": cfg.pooling,
        "folds": fold_keys,
        "val_f1_mean": float(np.mean(val_f1s)) if val_f1s else 0.0,
        "val_f1_std": float(np.std(val_f1s)) if val_f1s else 0.0,
        "test_f1_mean": float(np.mean(test_f1s)) if test_f1s else 0.0,
        "test_f1_std": float(np.std(test_f1s)) if test_f1s else 0.0,
        "per_fold": all_results,
        "config": asdict(cfg),
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(cfg.out_dir, f"summary_{cfg.pooling}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LOSO SUMMARY ===")
    print(f"pooling={cfg.pooling}")
    print(f"val_f1 : mean={summary['val_f1_mean']:.4f} std={summary['val_f1_std']:.4f}")
    print(f"test_f1: mean={summary['test_f1_mean']:.4f} std={summary['test_f1_std']:.4f}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
