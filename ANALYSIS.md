# Repository Analysis: preScan-breath-assist

## 1. Project Overview
This repository implements a research prototype pipeline for synchronizing Inertial Measurement Unit (IMU) data with dual-view video recordings (front and side). The primary goal appears to be constructing a labeled dataset for breath-holding detection, presumably to train machine learning models. The pipeline automates synchronization via "claps", generates labels from IMU signal characteristics, and formats the data into a window-based dataset.

## 2. Pipeline Architecture
The workflow follows a linear sequence of data processing steps:

1.  **Synchronization (`tools/sync_manager.py`)**:
    -   Aligns IMU time series with video timelines.
    -   **Method**: Detects distinct "claps" (peaks) at the start and end of recordings in both IMU accelerometer magnitude and video audio envelopes.
    -   **Scaling**: Calculates a linear time-scaling factor based on the duration between start and end claps to map IMU timestamps to video frame indices.

2.  **Label Generation (`tools/label_from_imu.py`)**:
    -   Infers "Breath Hold" (1) vs. "Breathing" (0) labels solely from IMU data.
    -   **Heuristic**: Uses a rolling standard deviation of the accelerometer magnitude. Periods of low variance (below an adaptive threshold) are classified as "holds".
    -   **Constraint**: Logic varies by "case type" (e.g., `normal`, `inhale_hold`). For `normal`, it defaults to breathing. For hold cases, it searches for a stable window.

3.  **Quality Assurance (`tools/qa_frames_in_range_batch.py`)**:
    -   Verifies that the calculated frame indices from the synchronization step actually fall within the valid frame range of the video files.
    -   Serves as a sanity check against synchronization drift or offset errors.

4.  **Dataset Construction (`tools/window_dataset_builder.py`)**:
    -   Slices the continuous synchronized streams into fixed-length time windows (default 0.5s).
    -   Assigns labels to windows based on a majority vote of the frame-level labels.
    -   Outputs a `manifest.csv` pointing to video paths and specifying start/end frames for each window.

5.  **Modeling (`src/models/mobilenet_temporal.py`)**:
    -   Provides a `MobileNetV3Temporal` model.
    -   Architecture: MobileNetV3-Small backbone extracting features per frame, followed by temporal pooling (mean or max) and a linear classifier.

## 3. Engineering & Code Quality
-   **Dependency Management**: The project uses `pip-tools` (`requirements.in`, `requirements.lock.*.txt`), indicating a disciplined approach to reproducibility.
-   **Code Structure**: Scripts in `tools/` are relatively modular and self-contained. `src/` is reserved for reusable components (like the model).
-   **Error Handling**: The synchronization script includes specific checks for signal length, variance, and "non-positive intervals", suggesting awareness of common failure modes in signal processing.
-   **Hardcoded Constraints**: The code strictly relies on `moviepy==1.0.3` and specific directory structures (`data/raw/videos/<Person>...`), which reduces flexibility but ensures consistency within the specific experimental setup.

## 4. Limitations & Critical Observations
*These points highlight areas requiring caution interpretation of results.*

### A. Ground Truth Validity
-   **Inferred Labels**: The labels are **not** manually annotated by clinical experts. They are algorithmically derived from the IMU stability.
-   **Risk**: If the subject holds their breath but moves their body (high IMU variance), the algorithm may incorrectly label it as "breathing". Conversely, if the subject sits perfectly still while breathing shallowly, it might be mislabeled as "hold".
-   **Implication**: The dataset represents "IMU stability" as a proxy for "breath hold", which may not fully align with physiological reality.

### B. Synchronization Assumptions
-   **Linear Drift**: The `sync_manager` assumes a constant linear scaling factor between IMU and Video clocks. While generally acceptable for short recordings, thermal drift or dropped frames could violate this assumption.
-   **Clap Detection**: Relies on `z-score` peaks. Noisy environments or accidental bumps could trigger false positives, leading to catastrophic sync failures. The script does perform "windowed" search to mitigate this.

### C. Dataset & Model
-   **Data Scope**: The repository contains data for a limited number of subjects (Benhur, Doga, Ece, Ecenaz, Mert, Sinan). Generalization to a broader population is unproven.
-   **Model Simplicity**: The temporal pooling (mean/max) loses the temporal order of features within a window. For a 0.5s window, this is likely acceptable, but it limits the model's ability to detect complex temporal dynamics (e.g., onset of inhale).

## 5. Conclusion
This repository represents a **functional engineering prototype** for multimodal data collection and preliminary analysis. It is well-structured for an academic project but relies on key assumptions (IMU stability = breath hold) that must be validated before any clinical or performance claims are made.
