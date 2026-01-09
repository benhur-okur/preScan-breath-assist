from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import cv2  # opencv-python

# ----------------------------
# Repo paths (run from repo root)
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
APP_RUNS_DIR = RUNS_DIR / "app_runs"

DEFAULT_CKPT = REPO_ROOT / "runs" / "final_model_v4_rgb_bce" / "final_best.pt"
DEFAULT_POOLING = "max"
DEFAULT_INPUT_MODE = "rgb"
DEFAULT_WINDOW_FRAMES = 15
DEFAULT_STRIDE_FRAMES = 5
DEFAULT_THRESHOLD = 0.90
DEFAULT_MIN_HOLD_SEC = 2.0

st.set_page_config(page_title="Breath-Hold Offline Demo", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def run_cmd(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    out = (p.stdout or "")
    if p.stderr:
        out += ("\n" + p.stderr)
    return p.returncode, out.strip()

def safe_filename(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ ."
    return "".join([c if c in keep else "_" for c in name])

def find_project_ckpts() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(RUNS_DIR.glob("**/*.pt"))

def list_app_runs() -> list[Path]:
    """Return demo_* run dirs sorted by newest first."""
    if not APP_RUNS_DIR.exists():
        return []
    runs = [p for p in APP_RUNS_DIR.glob("demo_*") if p.is_dir()]
    # newest first by name timestamp demo_YYYYMMDD_HHMMSS
    return sorted(runs, reverse=True)

def pick_video_in_run(run_dir: Path) -> Path | None:
    """Try to find the input video stored in the run_dir (first .mp4/.mov)."""
    for ext in (".mp4", ".mov", ".m4v"):
        cands = list(run_dir.glob(f"*{ext}"))
        if cands:
            return cands[0]
    return None

def extract_preview_frame(video_path: Path):
    """Extract 1st frame as RGB image array. Returns None if fails."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb

def summarize_segments(segments_csv: Path) -> dict:
    if not segments_csv.exists():
        return {"segments_count": 0}

    df = pd.read_csv(segments_csv)
    if df.empty:
        return {"segments_count": 0}

    summary = {"segments_count": int(len(df))}

    dur_col = None
    for c in df.columns:
        if c.lower() in ("duration_sec", "duration_s", "duration"):
            dur_col = c
            break
    if dur_col:
        summary["total_hold_sec"] = float(df[dur_col].sum())
        summary["longest_segment_sec"] = float(df[dur_col].max())

    start_col = None
    end_col = None
    for c in df.columns:
        if c.lower() in ("start_sec", "start_s"):
            start_col = c
        if c.lower() in ("end_sec", "end_s"):
            end_col = c
    if start_col:
        summary["first_segment_start_sec"] = float(df[start_col].min())
    if end_col:
        summary["last_segment_end_sec"] = float(df[end_col].max())

    return summary

def quick_commentary(summary: dict) -> str:
    n = summary.get("segments_count", 0)
    if n == 0:
        return "No breath-hold segments detected under the current threshold and minimum-duration settings."
    parts = [f"Detected **{n}** segment(s)."]
    if "total_hold_sec" in summary:
        parts.append(f"Total hold duration: **{summary['total_hold_sec']:.2f}s**.")
    if "longest_segment_sec" in summary:
        parts.append(f"Longest segment: **{summary['longest_segment_sec']:.2f}s**.")
    if "first_segment_start_sec" in summary:
        parts.append(f"First segment starts at **{summary['first_segment_start_sec']:.2f}s**.")
    return " ".join(parts)

def read_run_outputs(run_dir: Path):
    out_dir = run_dir / "predict_out"
    return {
        "predictions": out_dir / "predictions.csv",
        "segments": out_dir / "segments.csv",
        "plot": out_dir / "probability_plot.png",
    }

# ----------------------------
# Session state
# ----------------------------
if "last_run_dir" not in st.session_state:
    st.session_state["last_run_dir"] = None
if "last_logs" not in st.session_state:
    st.session_state["last_logs"] = {"make_windows": "", "predict": ""}

# ----------------------------
# UI
# ----------------------------
st.title("Breath-Hold Detection — Offline Demo UI")
st.caption("Local web UI that wraps the existing CLI pipeline (no deployment).")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Inputs")

    uploaded = st.file_uploader("Upload a video (.mp4 recommended)", type=["mp4", "mov", "m4v"])

    # --- Video preview (uploaded)
    st.markdown("### Video Preview")
    if uploaded is not None:
        # save temp to extract frame without waiting for run
        tmp_dir = RUNS_DIR / "app_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_video = tmp_dir / safe_filename(uploaded.name)
        with open(tmp_video, "wb") as f:
            f.write(uploaded.getbuffer())

        frame = extract_preview_frame(tmp_video)
        if frame is not None:
            st.image(frame, caption="First frame preview", use_container_width=True)
        else:
            st.warning("Could not extract preview frame (codec/format issue).")

    ckpts = find_project_ckpts()
    ckpt_paths = [DEFAULT_CKPT] + [p for p in ckpts if p != DEFAULT_CKPT]
    ckpt_paths = [p for p in ckpt_paths if p.exists()]
    ckpt_labels = [str(p.relative_to(REPO_ROOT)) for p in ckpt_paths]

    ckpt_choice = st.selectbox(
        "Checkpoint (.pt)",
        options=ckpt_labels if ckpt_labels else [str(DEFAULT_CKPT.relative_to(REPO_ROOT))],
        index=0,
    )

    st.markdown("### Settings")
    c1, c2 = st.columns(2)
    with c1:
        threshold = st.slider("Threshold", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01)
        pooling = st.selectbox("pooling", ["max", "mean"], index=0 if DEFAULT_POOLING == "max" else 1)
    with c2:
        min_hold_sec = st.number_input("min_hold_sec", 0.0, 30.0, float(DEFAULT_MIN_HOLD_SEC), 0.1)
        input_mode = st.selectbox("input_mode", ["rgb", "diff"], index=0 if DEFAULT_INPUT_MODE == "rgb" else 1)

    st.markdown("### Windowing")
    w1, w2 = st.columns(2)
    with w1:
        window_frames = st.number_input("window_frames", 5, 120, int(DEFAULT_WINDOW_FRAMES), 1)
    with w2:
        stride_frames = st.number_input("stride_frames", 1, 60, int(DEFAULT_STRIDE_FRAMES), 1)

    run_btn = st.button("Run Inference", type="primary", use_container_width=True)

    if run_btn:
        if uploaded is None:
            st.error("Please upload a video first.")
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            run_dir = APP_RUNS_DIR / f"demo_{ts}"
            run_dir.mkdir(parents=True, exist_ok=True)

            video_name = safe_filename(uploaded.name)
            video_path = run_dir / video_name
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())

            manifest_csv = run_dir / "manifest.csv"
            out_dir = run_dir / "predict_out"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd1 = [
                "python", "-m", "tools.core.make_windows_for_video",
                "--video", str(video_path),
                "--out-csv", str(manifest_csv),
                "--window-frames", str(int(window_frames)),
                "--stride-frames", str(int(stride_frames)),
            ]

            cmd2 = [
                "python", "-m", "tools.core.predict_video",
                "--manifest-csv", str(manifest_csv),
                "--ckpt", str(REPO_ROOT / ckpt_choice),
                "--pooling", pooling,
                "--input-mode", input_mode,
                "--threshold", str(float(threshold)),
                "--min-hold-sec", str(float(min_hold_sec)),
                "--out-dir", str(out_dir),
                "--plot",
            ]

            with st.status("Running pipeline...", expanded=True) as status:
                st.write("**1) make_windows_for_video**")
                st.code(" ".join(cmd1), language="bash")
                rc1, out1 = run_cmd(cmd1)
                st.session_state["last_logs"]["make_windows"] = out1
                if rc1 != 0:
                    status.update(label="make_windows_for_video failed", state="error")
                    st.stop()

                st.write("**2) predict_video**")
                st.code(" ".join(cmd2), language="bash")
                rc2, out2 = run_cmd(cmd2)
                st.session_state["last_logs"]["predict"] = out2
                if rc2 != 0:
                    status.update(label="predict_video failed", state="error")
                    st.stop()

                status.update(label="Pipeline finished", state="complete")

            st.session_state["last_run_dir"] = str(run_dir)
            st.success(f"Run saved to: {run_dir.relative_to(REPO_ROOT)}")

    st.divider()
    st.subheader("Logs (latest run)")
    st.text_area("make_windows_for_video log", value=st.session_state["last_logs"]["make_windows"], height=160)
    st.text_area("predict_video log", value=st.session_state["last_logs"]["predict"], height=160)

with right:
    st.subheader("Run History & Outputs")

    runs = list_app_runs()
    if not runs:
        st.info("No previous runs found yet. Run the pipeline once.")
        st.stop()

    # --- Run history dropdown
    run_labels = [r.name for r in runs]  # demo_YYYYMMDD_HHMMSS
    # default: last_run_dir if exists, else newest
    default_label = runs[0].name
    if st.session_state["last_run_dir"]:
        try:
            default_label = Path(st.session_state["last_run_dir"]).name
        except Exception:
            default_label = runs[0].name

    selected_label = st.selectbox(
        "Select a previous run",
        options=run_labels,
        index=run_labels.index(default_label) if default_label in run_labels else 0,
    )
    selected_run = APP_RUNS_DIR / selected_label

    # show which input video was used
    vid = pick_video_in_run(selected_run)
    if vid:
        st.caption(f"Input video: `{vid.name}`")

    outputs = read_run_outputs(selected_run)
    predictions_csv = outputs["predictions"]
    segments_csv = outputs["segments"]
    plot_png = outputs["plot"]

    # Summary + commentary
    st.markdown("### Report Summary")
    summary = summarize_segments(segments_csv)
    st.json(summary)
    st.markdown("### Interpretation")
    st.write(quick_commentary(summary))

    # Plot
    st.markdown("### Probability Plot")
    if plot_png.exists():
        st.image(str(plot_png), use_container_width=True)
    else:
        st.warning("probability_plot.png not found in this run output.")

    # Segments table
    st.markdown("### Segments")
    if segments_csv.exists():
        seg_df = pd.read_csv(segments_csv)
        st.dataframe(seg_df, use_container_width=True)
    else:
        st.warning("segments.csv not found in this run output.")

    # Downloads
    st.markdown("### Downloads")
    if predictions_csv.exists():
        st.download_button(
            "Download predictions.csv",
            data=predictions_csv.read_bytes(),
            file_name=f"{selected_label}_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if segments_csv.exists():
        st.download_button(
            "Download segments.csv",
            data=segments_csv.read_bytes(),
            file_name=f"{selected_label}_segments.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if plot_png.exists():
        st.download_button(
            "Download probability_plot.png",
            data=plot_png.read_bytes(),
            file_name=f"{selected_label}_probability_plot.png",
            mime="image/png",
            use_container_width=True,
        )
