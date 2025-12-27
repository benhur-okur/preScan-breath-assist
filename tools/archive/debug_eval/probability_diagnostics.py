# tools/probability_diagnostics.py
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
from sklearn.metrics import f1_score, confusion_matrix

from src.datasets.window_dataset import WindowVideoDataset
from src.models.mobilenet_temporal import MobileNetV3Temporal


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: [N] int64
      probs : [N] float32 (sigmoid(logits))
    """
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

        ys.append(y.detach().cpu().numpy().reshape(-1))
        ps.append(p)

    y_true = np.concatenate(ys).astype(np.int64) if ys else np.array([], dtype=np.int64)
    probs = np.concatenate(ps).astype(np.float32) if ps else np.array([], dtype=np.float32)
    return y_true, probs


def bin_histogram(probs: np.ndarray, bins: int = 10) -> Dict:
    """
    Simple histogram counts for probs in [0,1].
    Returns dict with bin edges and counts.
    """
    if probs.size == 0:
        return {"edges": [], "counts": []}

    # edges: 0.0, 0.1, ..., 1.0 (bins=10)
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(probs, bins=edges)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def summarize_probs(y: np.ndarray, p: np.ndarray) -> Dict:
    """
    Summaries overall + per-class.
    """
    out: Dict = {}
    if len(y) == 0:
        return {"n": 0}

    out["n"] = int(len(y))
    out["p_min"] = float(np.min(p))
    out["p_mean"] = float(np.mean(p))
    out["p_max"] = float(np.max(p))
    out["p_p10"] = float(np.percentile(p, 10))
    out["p_p50"] = float(np.percentile(p, 50))
    out["p_p90"] = float(np.percentile(p, 90))

    for cls in [0, 1]:
        mask = (y == cls)
        if mask.sum() == 0:
            out[f"class_{cls}_n"] = 0
            continue
        pc = p[mask]
        out[f"class_{cls}_n"] = int(mask.sum())
        out[f"class_{cls}_p_mean"] = float(np.mean(pc))
        out[f"class_{cls}_p_p50"] = float(np.percentile(pc, 50))
        out[f"class_{cls}_p_p90"] = float(np.percentile(pc, 90))
        out[f"class_{cls}_p_max"] = float(np.max(pc))

    out["hist_10bins"] = bin_histogram(p, bins=10)
    return out


def eval_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> Dict:
    pred = (p >= thr).astype(np.int64)
    f1 = float(f1_score(y, pred, zero_division=0))
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    pos_rate = float(pred.mean()) if pred.size else 0.0
    return {"thr": float(thr), "f1": f1, "confusion": cm, "pred_pos_rate": pos_rate, "n": int(len(y))}


def load_test_dataset(manifest_csv: str, fold_key: str, img_size: int) -> WindowVideoDataset:
    """
    Your project has 2 possible encodings:
    - split == 'test'
    - split == fold_key  (some earlier manifests did this)
    We support both without guessing training logic.
    """
    try:
        return WindowVideoDataset(
            manifest_csv,
            split="test",
            fold_id=fold_key,
            img_size=img_size,
            train_aug=False,
        )
    except RuntimeError:
        # fallback: split==fold_key
        return WindowVideoDataset(
            manifest_csv,
            split=fold_key,
            fold_id=fold_key,
            img_size=img_size,
            train_aug=False,
        )


def main():
    ap = argparse.ArgumentParser(description="Probability diagnostics for a LOSO fold/split.")
    ap.add_argument("--manifest-loso", required=True, help="data/windows/manifest_loso.csv")
    ap.add_argument("--fold-key", required=True, help="e.g., fold_1_test_benhur")
    ap.add_argument("--pooling", choices=["mean", "max"], default="max")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--ckpt", default=None, help="Optional path to best.pt. If not given, auto-resolve under --runs-dir.")
    ap.add_argument("--runs-dir", default="runs/loso_max_guard", help="Base dir where fold folders live")
    ap.add_argument("--out", default=None, help="Optional output json path")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve ckpt path if not provided
    ckpt_path = args.ckpt
    if ckpt_path is None:
        # expected fold folder: fold_??_<fold_key>/best.pt
        # search for matching folder
        if not os.path.isdir(args.runs_dir):
            raise FileNotFoundError(f"runs_dir not found: {args.runs_dir}")

        candidates = []
        for name in os.listdir(args.runs_dir):
            if name.endswith(args.fold_key) and name.startswith("fold_"):
                candidates.append(os.path.join(args.runs_dir, name, "best.pt"))

        candidates = [p for p in candidates if os.path.isfile(p)]
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"Could not auto-find best.pt for fold_key={args.fold_key} under {args.runs_dir}. "
                f"Provide --ckpt explicitly."
            )
        if len(candidates) > 1:
            # deterministic: pick lexicographically smallest
            candidates = sorted(candidates)
        ckpt_path = candidates[0]

    # Dataset/Loader (TEST only)
    ds = load_test_dataset(args.manifest_loso, args.fold_key, img_size=args.img_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = MobileNetV3Temporal(pooling=args.pooling, pretrained=False).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    y, p = predict_probs(model, loader, device=device)

    summary = {
        "fold_key": args.fold_key,
        "pooling": args.pooling,
        "device": device,
        "ckpt_path": ckpt_path,
        "stats": summarize_probs(y, p),
        "metrics": {
            "thr_0.50": eval_at_threshold(y, p, 0.50),
            "thr_0.30": eval_at_threshold(y, p, 0.30),
            "thr_0.20": eval_at_threshold(y, p, 0.20),
            "thr_0.10": eval_at_threshold(y, p, 0.10),
        },
        "top_hold_probs": [],
    }

    # Top hold probabilities (if any hold samples exist)
    hold_idx = np.where(y == 1)[0]
    if hold_idx.size > 0:
        hold_probs = p[hold_idx]
        topk = min(10, hold_probs.size)
        top_vals = np.sort(hold_probs)[-topk:][::-1]
        summary["top_hold_probs"] = [float(x) for x in top_vals]

    # Save or print
    if args.out is None:
        out_path = os.path.join("runs", "diagnostics", f"probs_{args.fold_key}_{args.pooling}.json")
    else:
        out_path = args.out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved: {out_path}")
    print("Key stats:")
    s = summary["stats"]
    print(f"  n={s.get('n',0)} p_min={s.get('p_min',0):.4f} p_mean={s.get('p_mean',0):.4f} p_max={s.get('p_max',0):.4f}")
    print(f"  class0_mean={s.get('class_0_p_mean',0):.4f} class1_mean={s.get('class_1_p_mean',0):.4f}")
    print("Metrics quick:")
    for k, v in summary["metrics"].items():
        print(f"  {k}: f1={v['f1']:.4f} pred_pos_rate={v['pred_pos_rate']:.3f} cm={v['confusion']}")


if __name__ == "__main__":
    main()
