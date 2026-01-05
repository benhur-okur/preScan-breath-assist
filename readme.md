# Breath-Hold Detection from Front-View Video  
### Temporal Deep Learning Pipeline for Offline Analysis

**Graduation Project (Bitirme Projesi)**  
Department of Computer Engineering – İzmir Ekonomi University  

**Author:** Benhur Rahman Okur  
**Supervisor:** (to be filled)  
**Year:** 2025  

---

## 1. Project Summary

This project presents an **offline, video-based breath-hold detection system** using a **temporal deep learning pipeline**.

The system analyzes **single front-view video recordings** and detects **meaningful breath-hold events** over time.  
Unlike frame-level or naive classification approaches, this project focuses on **segment-level reasoning**, answering:

- When does a breath-hold start?
- How long does it last?
- Is the detected event meaningful or just noise?

The project is developed as an **experimental research prototype** within the scope of a **graduation thesis**, prioritizing **engineering reliability and academic defensibility** over raw performance metrics.

---

## 2. Problem Definition & Motivation

Breath-hold detection from video is inherently challenging because:

- Breath-hold periods involve **minimal visible motion**
- Micro-movements differ significantly across subjects
- Frame-level classifiers produce noisy, unstable predictions
- Simple probability thresholding causes false-positive spikes

Therefore, this is treated as a **temporal detection problem**, not a frame classification task.

### Design Objective

To build a system that:
- Is temporally aware
- Produces stable and interpretable outputs
- Generalizes across subjects
- Is reproducible and engineering-safe
- Can be defended academically

---

## 3. Dataset Description

- **Total videos:** 24  
- **Total subjects:** 6  
- **View:** Single front-view camera  
- **Format:** `.mp4`  
- **Frame rate:** Read dynamically from video metadata  

### Case Types
Each subject includes multiple cases:
- `normal`
- `inhale_hold`
- `exhale_hold`
- `irregular`

> **Important:**  
> A *case* is not a *subject*.  
> Evaluation is performed **subject-wise**.

---

## 4. High-Level System Architecture

Video (.mp4)
↓
Sliding Window Extraction (T = 15 frames)
↓
Temporal CNN (MobileNetV3Temporal)
↓
Window-level probabilities
↓
Thresholding
↓
Temporal post-processing
↓
Breath-hold segments (start / end / duration)



The system outputs **segments**, not just probabilities.

---

## 5. Model Architecture

- **Model:** MobileNetV3Temporal  
- **Version:** v4 (FINAL)  
- **Temporal window:** 15 frames  
- **Pooling:** max  
- **Aggregation:** frame → window  

### Input Mode
- **Final input mode:** `rgb`
- `diff` mode was evaluated but rejected due to instability, especially for `exhale_hold` cases.

The definitive training configuration is stored in:

runs/final_model_v4_rgb_bce/final_meta.json



---

## 6. Training Strategy

### 6.1 Evaluation During Development

- **Leave-One-Subject-Out (LOSO)** evaluation was used extensively.
- LOSO was critical for:
  - Detecting subject leakage
  - Avoiding inflated metrics
  - Ensuring subject-independent behavior

### 6.2 Final Training

- All data used
- Leak-free validation split
- Best model selected based on validation behavior
- Training metadata automatically saved

---

## 7. Thresholding & Post-processing

### 7.1 Threshold Selection

- Validation-optimal threshold (`best_threshold = 0.1`) is recorded in `final_meta.json`
- **Final deployment threshold:** `0.90`

> The final threshold was selected based on **segment-level behavior**, not window-level metrics.

### 7.2 Temporal Filtering

- `min_hold_sec = 2.0`
- Short spikes are removed
- Only temporally consistent segments are retained

---

## 8. Engineering Decisions

All major design decisions are documented using **Engineering Decision Records (EDR)**.

edr/EDR-0001-final-breath-hold-pipeline.yaml


This includes:
- Input mode selection
- Threshold choice
- Segment-level prioritization
- Known risks and limitations

---

## 9. Project Structure

preScan-breath-assist/
│
├─ tools/
│ ├─ core/ # Final pipeline scripts
│ └─ archive/ # Legacy, debug, IMU/audio tools
│
├─ src/ # Model, dataset, utilities
│
├─ data/
│ └─ windows/ # Window manifests (.csv)
│
├─ runs/
│ ├─ final_model_v4_rgb_bce/ # FINAL checkpoint + meta
│ ├─ final_demo/ # Demo inference outputs
│ └─ loso_* # LOSO experiments
│
├─ edr/ # Engineering Decision Records
├─ requirements.txt
├─ requirements.in
└─ README.md


---

## 10. Core Scripts (tools/core)

### `make_windows_for_video.py`
- Converts a video into sliding windows
- Outputs a manifest `.csv`
- Window size: 15 frames

### `train_loso.py`
- Performs LOSO evaluation
- Used only for development and validation
- Not used in final training

### `train_final_v4.py`
- Trains the FINAL model
- Saves:
  - `final_best.pt`
  - `final_last.pt`
  - `final_meta.json`

### `predict_video.py`
- Runs offline inference
- Enforces compatibility with training via `final_meta.json`
- Outputs predictions, segments, and plots

---

## 11. Installation

### 11.1 Create Virtual Environment


python -m venv .venv
Activate:

bash

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
## 11.2 Install Dependencies

pip install -r requirements.txt
# 12. Usage

## 12.1 Generate Windows

python -m tools.core.make_windows_for_video \
  --video "path/to/video" \
  --out-csv "runs/final_demo/manifest.csv" \
  --window-frames 15 \
  --stride-frames 5


## 12.2 Run Final Inference

python -m tools.core.predict_video \
  --manifest-csv runs/final_demo/exhale_hold_manifest.csv \
  --ckpt runs/final_model_v4_rgb_bce/final_best.pt \
  --pooling max \
  --input-mode rgb \
  --threshold 0.90 \
  --min-hold-sec 2.0 \
  --out-dir runs/final_demo/v4_exhale_thr090_min2s \
  --plot

# 13. Reproducibility
All training parameters are saved in final_meta.json

Inference validates compatibility with training configuration

No hard-coded assumptions are used

# 14. Known Limitations
Small dataset (6 subjects)

Offline processing only

Single camera view

No clinical validation

Research prototype only

# 15. Future Work
Real-time inference

Larger datasets

Multi-view analysis

Physiological signal fusion

Automated segment-level metrics

# 16. Disclaimer
This project is developed strictly for academic and research purposes.
It is not intended for medical or clinical use.

# 17. Contact
Benhur Rahman Okur
Computer Engineering
İzmir University Of Economics