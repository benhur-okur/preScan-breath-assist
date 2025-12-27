import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mp

 #python plot_auido_claps.py Videos/Sinan/sinan_4_front.mp4


def load_audio_mono(path, target_fps=2000.0):
    """
    Video dosyasından audio'yu alır, target_fps ile örnekler,
    mono (tek kanal) numpy array ve zaman ekseni döner.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    clip = mp.VideoFileClip(path)
    audio = clip.audio
    if audio is None:
        raise RuntimeError(f"Video {path} has no audio track")

    # Audio'yu parça parça oku (memory dostu)
    chunks = []
    for chunk in audio.iter_chunks(
        chunksize=2000,       # 2000 örnekli parçalar
        fps=target_fps,
        quantize=False,
        nbytes=2
    ):
        # chunk shape: (N, channels)
        chunks.append(chunk)

    if len(chunks) == 0:
        raise RuntimeError(f"Audio.iter_chunks returned no data for {path}")

    samples = np.concatenate(chunks, axis=0)

    # Mono'ya indir
    if samples.ndim == 2:
        mono = samples.mean(axis=1)
    else:
        mono = samples.astype(float)

    n = len(mono)
    t = np.arange(n) / float(target_fps)

    return t, mono, target_fps


def detect_clap_peaks(t, signal, z_thresh=5.0, min_sep=0.4):
    """
    Clap tespiti:
      - Mutlak genlik / envelope üzerinden z-score hesaplar
      - z > z_thresh olan noktaları alır
      - min_sep (saniye) ile gruplayıp her gruptaki en büyük peak'i seçer

    Dönen:
      - peak_times: [t1, t2, ...]
      - peak_indices: [i1, i2, ...]
      - z_scores: z array
      - envelope: |signal| array
    """
    # Envelope = |signal|
    env = np.abs(signal).astype(float)

    # İstersen hafif smooth (moving average)
    # smoothing_window = 5
    # if len(env) > smoothing_window:
    #     kernel = np.ones(smoothing_window) / smoothing_window
    #     env = np.convolve(env, kernel, mode="same")

    mean = env.mean()
    std = env.std() + 1e-9
    z = (env - mean) / std

    # threshold üstü indexler
    idx = np.where(z > z_thresh)[0]
    if len(idx) == 0:
        return [], [], z, env

    # min_sep saniye ile grupla
    peaks = []
    group = [idx[0]]
    for i in idx[1:]:
        if t[i] - t[group[-1]] < min_sep:
            group.append(i)
        else:
            # grup içindeki en yüksek z'li index'i al
            best_idx = group[np.argmax(z[group])]
            peaks.append(best_idx)
            group = [i]
    # son grubu ekle
    best_idx = group[np.argmax(z[group])]
    peaks.append(best_idx)

    peak_times = [float(t[i]) for i in peaks]
    return peak_times, peaks, z, env


def plot_audio_and_peaks(t, mono, env, peak_times, peak_indices, z, z_thresh, title_suffix=""):
    """
    2 subplot:
      1) raw waveform
      2) envelope + threshold + clap çizgileri
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 1) Raw waveform
    ax1 = axes[0]
    ax1.plot(t, mono, linewidth=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Raw audio waveform" + title_suffix)
    ax1.grid(True, alpha=0.3)

    # clap çizgileri
    for pt in peak_times:
        ax1.axvline(pt, color="r", linestyle="--", alpha=0.7)

    # 2) Envelope + z-score threshold
    ax2 = axes[1]
    ax2.plot(t, env, label="|audio| (envelope)", linewidth=0.8)

    # threshold çizgisi için kabaca mean + z_thresh * std
    env_mean = env.mean()
    env_std = env.std()
    thr_val = env_mean + z_thresh * env_std
    ax2.axhline(thr_val, color="orange", linestyle="--", label=f"threshold (z>{z_thresh})")

    # Clap noktalarını işaretle
    ax2.scatter(
        [t[i] for i in peak_indices],
        [env[i] for i in peak_indices],
        color="red",
        marker="o",
        label="Detected clap peaks"
    )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Envelope")
    ax2.set_title("Audio envelope and detected clap peaks")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "MP4/MOV dosyasından audio çıkarıp clap peaklerini (start/finish) "
            "görsel olarak gösteren script."
        )
    )
    parser.add_argument("video_path", type=str, help="Input video file (mp4/mov)")
    parser.add_argument(
        "--audio-fps",
        type=float,
        default=2000.0,
        help="Audio sampling rate for analysis (default: 2000 Hz).",
    )
    parser.add_argument(
        "--z-thresh",
        type=float,
        default=5.0,
        help="Z-score threshold for clap detection (default: 5.0).",
    )
    parser.add_argument(
        "--min-sep",
        type=float,
        default=0.4,
        help="Minimum separation between claps in seconds (default: 0.4s).",
    )

    args = parser.parse_args()

    print(f"[INFO] Loading audio from {args.video_path} ...")
    try:
        t, mono, fps_audio = load_audio_mono(args.video_path, target_fps=args.audio_fps)
    except Exception as e:
        print(f"[FAIL] Could not load/extract audio: {e}")
        sys.exit(1)

    print(f"[INFO] Audio length: {t[-1]:.3f} s, samples: {len(mono)}, fs: {fps_audio:.1f} Hz")

    # Clap tespiti
    print("[INFO] Detecting clap peaks...")
    peak_times, peak_indices, z, env = detect_clap_peaks(
        t, mono, z_thresh=args.z_thresh, min_sep=args.min_sep
    )

    if len(peak_times) == 0:
        print(f"[WARN] No peaks detected above z>{args.z_thresh}. "
              "Clap sesi çok zayıf veya kayıt sessiz olabilir.")
    else:
        print(f"[INFO] Detected {len(peak_times)} clap peak group(s):")
        for i, pt in enumerate(peak_times):
            label = f"CLAP_{i+1}"
            print(f"  - {label}: t={pt:.3f} s")

        if len(peak_times) >= 2:
            print("\n[INFO] Interpreting:")
            print(f"  - START_CLAP ≈ {peak_times[0]:.3f} s")
            print(f"  - END_CLAP   ≈ {peak_times[-1]:.3f} s")
        else:
            print("\n[WARN] 2 güçlü clap bekleniyordu, ama "
                  f"{len(peak_times)} grup bulundu. Çekim protokolünü kontrol edin.")

    print("[INFO] Plotting waveform and detected peaks...")
    title_suffix = f" ({os.path.basename(args.video_path)})"
    plot_audio_and_peaks(t, mono, env, peak_times, peak_indices, z, args.z_thresh, title_suffix=title_suffix)


if __name__ == "__main__":
    main()
