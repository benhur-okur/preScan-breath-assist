# src/datasets/window_dataset.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass
class WindowSample:
    video_path: str
    start_frame: int
    window_frames: int
    label: int
    person: str
    case: str
    fold_key: str
    split: str


def _read_clip_opencv(video_path: str, start_frame: int, window_frames: int) -> np.ndarray:
    """
    Returns: uint8 RGB array of shape [T, H, W, 3]
    Robust to short reads: pads by repeating last good frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    frames = []
    last_ok = None

    for _ in range(window_frames):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            if last_ok is None:
                cap.release()
                raise RuntimeError(f"Failed reading frames at start_frame={start_frame} from {video_path}")
            frames.append(last_ok.copy())
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        last_ok = frame_rgb
        frames.append(frame_rgb)

    cap.release()
    return np.stack(frames, axis=0)  # [T,H,W,3]


class WindowVideoDataset(Dataset):
    """
    Front-only baseline dataset.

    LOSO manifest contract (your project):
      - df["fold_id"] is a string key: e.g. "fold_1_test_benhur"
      - df["split"] is one of:
          "train", "val", OR the same fold key for test rows.
        i.e. test rows: split == fold_id

    Each item -> (clip_tensor [T,3,img,img], label float32)
    """
    def __init__(
        self,
        manifest_csv: str,
        split: str,
        fold_id: str,              # <-- IMPORTANT: fold key string
        root_dir: Optional[str] = None,
        img_size: int = 224,
        train_aug: bool = True,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of ('train','val','test'), got: {split}")

        self.manifest_csv = manifest_csv
        self.split = split
        self.fold_key = str(fold_id)
        self.root_dir = root_dir or "."

        df = pd.read_csv(manifest_csv)

        # Keep only rows belonging to this fold key
        df = df[df["fold_id"] == self.fold_key].copy()

        # Split selection rule (robust)
        if split in ("train", "val"):
            df = df[df["split"] == split].copy()
        else:
            # Support BOTH encodings:
            # 1) split == "test"  (your current manifest_loso.csv)
            # 2) split == fold_key (optional future encoding)
            df = df[(df["split"] == "test") | (df["split"] == self.fold_key)].copy()

        if len(df) == 0:
            raise RuntimeError(
                f"No rows for fold_key={self.fold_key} split={split} in {manifest_csv} "
                f"(expected test rows: split==fold_key)"
            )

        self.samples: list[WindowSample] = []
        for _, r in df.iterrows():
            video_path = str(r["front_video_path"])
            video_path = os.path.normpath(os.path.join(self.root_dir, video_path))

            self.samples.append(
                WindowSample(
                    video_path=video_path,
                    start_frame=int(r["start_frame"]),
                    window_frames=int(r["window_frames"]),
                    label=int(r["label"]),
                    person=str(r["person"]),
                    case=str(r["case"]),
                    fold_key=str(r["fold_id"]),
                    split=str(r["split"]),
                )
            )

        # Transforms (minimal, safe)
        if split == "train" and train_aug:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]

        clip = _read_clip_opencv(
            video_path=s.video_path,
            start_frame=s.start_frame,
            window_frames=s.window_frames,
        )  # [T,H,W,3] uint8 RGB

        frames_t = []
        for t in range(clip.shape[0]):
            pil = Image.fromarray(clip[t])
            frames_t.append(self.transform(pil))

        clip_t = torch.stack(frames_t, dim=0)  # [T,3,H,W]
        y = torch.tensor(float(s.label), dtype=torch.float32)

        return clip_t, y
