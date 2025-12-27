# tools/predict_video.py
from __future__ import annotations

import os
import glob
import json
import argparse
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.window_dataset import WindowVideoDataset
from src.models.mobilenet_temporal import MobileNetV3Temporal


# -----------------------------
# Utils
# -----------------------------
def _load_temperature(args) -> float:
    """
    Temperature scaling: use T>0.
    Priority:
      1) --temperature-json (expects {"temperature": ...})
      2) --temperature
      3) default 1.0
    """
    if getattr(args, "temperature_json", None):
        with open(args.temperature_json, "r", encoding="utf-8") as f:
            j = json.load(f)
        if "temperature" not in j:
            raise SystemExit("temperature_json does not contain key: temperature")
        T = float(j["temperature"])
        if T <= 0:
            raise SystemExit("temperature must be > 0")
        return T

    if getattr(args, "temperature", None) is not None:
        T = float(args.temperature)
        if T <= 0:
            raise SystemExit("temperature must be > 0")
        return T

    return 1.0


def _postprocess_min_consecutive(pred: np.ndarray, min_consecutive: int) -> np.ndarray:
    """
    Keep predicted hold segments only if they have >= min_consecutive consecutive 1s.
    """
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


def _compute_min_consecutive_from_time(df_manifest: pd.DataFrame, min_hold_sec: float) -> int:
    """
    Requires manifest columns: fps, stride_frames
    """
    if "fps" not in df_manifest.columns or "stride_frames" not in df_manifest.columns:
        raise RuntimeError("min-hold-sec requires manifest columns: fps and stride_frames")

    fps = float(df_manifest["fps"].iloc[0])
    stride_frames = int(df_manifest["stride_frames"].iloc[0])

    if fps <= 0:
        raise RuntimeError("min-hold-sec requires valid fps>0 in manifest.")

    stride_sec = stride_frames / fps
    if stride_sec <= 0:
        raise RuntimeError("Invalid stride_sec computed from stride_frames/fps.")

    min_consecutive = int(np.ceil(float(min_hold_sec) / stride_sec))
    return max(1, min_consecutive)


def _extract_segments(df_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Build segments from per-window pred_hold=1 runs.
    Expects columns: start_frame, window_frames, pred_hold
    """
    pred = df_preds["pred_hold"].to_numpy(dtype=int)
    starts = df_preds["start_frame"].to_numpy(dtype=int)
    win = df_preds["window_frames"].to_numpy(dtype=int)

    segs: List[Dict] = []
    n = len(pred)
    i = 0
    while i < n:
        if pred[i] == 1:
            j = i
            while j < n and pred[j] == 1:
                j += 1

            start_frame = int(starts[i])
            end_frame_exclusive = int(starts[j - 1] + win[j - 1])

            segs.append(
                {
                    "start_window_idx": int(i),
                    "end_window_idx": int(j - 1),
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame_exclusive,
                    "length_windows": int(j - i),
                }
            )
            i = j
        else:
            i += 1

    return pd.DataFrame(segs)


def _resolve_ckpt_path(args) -> str:
    """
    Production rule:
      - Prefer explicit --ckpt
      - Else if --ckpt-dir:
          * If contains final_best.pt -> use it
          * Else if contains best.pt directly -> use it
          * Else error (do NOT autoload multiple folds here)
    """
    if args.ckpt:
        if not os.path.isfile(args.ckpt):
            raise SystemExit(f"Checkpoint not found: {args.ckpt}")
        return args.ckpt

    if args.ckpt_dir:
        cand1 = os.path.join(args.ckpt_dir, "final_best.pt")
        if os.path.isfile(cand1):
            return cand1

        # allow user to point directly to a folder that contains best.pt
        cand2 = os.path.join(args.ckpt_dir, "best.pt")
        if os.path.isfile(cand2):
            return cand2

        # allow pattern if they pass a path like runs/final_model_v3 and file named final_best.pt exists
        pts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
        raise SystemExit(
            "Could not resolve a single checkpoint from --ckpt-dir.\n"
            f"Expected one of:\n"
            f"  - {cand1}\n"
            f"  - {cand2}\n"
            f"Found pt files: {pts}\n"
            "Use --ckpt <path_to_final_best.pt> explicitly."
        )

    raise SystemExit("You must provide --ckpt OR --ckpt-dir")


# -----------------------------
# Core inference
# -----------------------------
@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    """
    Returns logits for each sample in loader, shape [N]
    """
    all_logits: List[np.ndarray] = []
    model.eval()

    for x, _y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)  # [B] or [B,1]
        logits = logits.detach().cpu().numpy().reshape(-1)
        all_logits.append(logits)

    if not all_logits:
        return np.array([], dtype=np.float32)
    return np.concatenate(all_logits).astype(np.float32)


def load_model(ckpt_path: str, pooling: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise SystemExit("Checkpoint must contain key 'model' (state_dict).")
    model = MobileNetV3Temporal(pooling=pooling, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest-csv", required=True, help="mini manifest from make_windows_for_video.py")
    ap.add_argument("--fold-key", default="inference", help="fold_id used in mini manifest (default: inference)")

    ap.add_argument("--pooling", choices=["mean", "max"], default="max")
    ap.add_argument("--input-mode", choices=["rgb", "diff"], default="diff",
                    help="MUST match training preprocessing. Final model uses diff.")

    ap.add_argument("--ckpt", default=None, help="path to final_best.pt (single model)")
    ap.add_argument("--ckpt-dir", default=None, help="directory containing final_best.pt")

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--threshold", type=float, required=True, help="global threshold for hold (after calibration)")
    ap.add_argument("--out-dir", default="runs/infer")
    ap.add_argument("--plot", action="store_true", help="save probability_plot.png")

    # Calibration
    ap.add_argument("--temperature-json", default=None, help="json output of temperature_scaling_final.py")
    ap.add_argument("--temperature", type=float, default=None, help="manual temperature override (if no json)")

    # Postprocess options
    ap.add_argument("--no-postprocess", action="store_true", help="disable postprocess")
    ap.add_argument("--min-consecutive", type=int, default=None,
                    help="minimum consecutive windows to keep a hold segment")
    ap.add_argument("--min-hold-sec", type=float, default=None,
                    help="minimum hold duration in seconds (requires fps+stride_frames in manifest)")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = _resolve_ckpt_path(args)
    T = _load_temperature(args)

    print(f"[INFO] device={device}")
    print(f"[INFO] ckpt={ckpt_path}")
    print(f"[INFO] input_mode={args.input_mode} pooling={args.pooling}")
    print(f"[INFO] threshold={float(args.threshold):.4f}  temperature={T:.4f}")

    # Read manifest first (needed for min-hold-sec)
    df = pd.read_csv(args.manifest_csv)
    df = df[df["fold_id"] == args.fold_key].copy()
    df = df[(df["split"] == "test") | (df["split"] == args.fold_key)].copy()
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise SystemExit(f"No rows found for fold_key={args.fold_key} in manifest={args.manifest_csv}")

    # Decide postprocess parameter
    min_consecutive: Optional[int] = args.min_consecutive
    if not args.no_postprocess:
        if args.min_hold_sec is not None:
            min_consecutive = _compute_min_consecutive_from_time(df, float(args.min_hold_sec))
            print(f"[INFO] min_hold_sec={args.min_hold_sec} => min_consecutive={min_consecutive}")
        if min_consecutive is None:
            raise SystemExit("Postprocess enabled. Provide --min-hold-sec OR --min-consecutive (or use --no-postprocess).")

    # Dataset
    ds = WindowVideoDataset(
        manifest_csv=args.manifest_csv,
        split="test",
        fold_id=args.fold_key,
        img_size=args.img_size,
        train_aug=False,
        input_mode=args.input_mode,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = load_model(ckpt_path, pooling=args.pooling, device=device)

    # Predict logits -> apply temperature -> sigmoid
    logits = predict_logits(model, loader, device=device)
    if len(logits) != len(df):
        raise RuntimeError(f"Row count mismatch: manifest_rows={len(df)} vs logits={len(logits)}")

    logits_cal = logits / float(T)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_cal, -50, 50)))

    pred_raw = (probs >= float(args.threshold)).astype(np.int64)
    pred_post = pred_raw.copy()

    if not args.no_postprocess and min_consecutive is not None:
        pred_post = _postprocess_min_consecutive(pred_raw, int(min_consecutive))

    out_df = pd.DataFrame({
        "start_frame": df["start_frame"].astype(int),
        "window_frames": df["window_frames"].astype(int),
        "logit": logits.astype(float),
        "logit_calibrated": logits_cal.astype(float),
        "prob_hold": probs.astype(float),
        "pred_hold_raw": pred_raw.astype(int),
        "pred_hold": pred_post.astype(int),
        "video_path": df["front_video_path"].astype(str),
        "input_mode": args.input_mode,
        "threshold": float(args.threshold),
        "temperature": float(T),
        "min_consecutive": (int(min_consecutive) if min_consecutive is not None else -1),
        "min_hold_sec": (float(args.min_hold_sec) if args.min_hold_sec is not None else -1.0),
    })

    out_csv = os.path.join(args.out_dir, "predictions.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] saved: {out_csv}")

    seg_df = _extract_segments(out_df)
    seg_csv = os.path.join(args.out_dir, "segments.csv")
    seg_df.to_csv(seg_csv, index=False, encoding="utf-8")
    print(f"[OK] saved: {seg_csv} (segments={len(seg_df)})")

    if args.plot:
        import matplotlib.pyplot as plt

        x = np.arange(len(out_df))
        plt.figure()
        plt.plot(x, out_df["prob_hold"].values)
        plt.axhline(float(args.threshold), linestyle="--")
        plt.title("Hold probability over windows")
        plt.xlabel("Window index")
        plt.ylabel("P(hold)")
        plot_path = os.path.join(args.out_dir, "probability_plot.png")
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved: {plot_path}")


if __name__ == "__main__":
    main()
