import zipfile
import numpy as np
import pandas as pd
import moviepy.editor as mp
import argparse


def detect_start_end_claps_windowed(
    t,
    signal,
    z_min=3.0,
    min_group_dt=0.4,
    start_frac=0.3,
    end_frac=0.3,
    label=""
):
    """
    t            : zaman (saniye)
    signal       : 1D numpy array (IMU |a| veya audio envelope)
    z_min        : "aday clap" için minimum z-score
    min_group_dt : aynı clap grubunda sayacağımız örnekler arası max süre (s)
    start_frac   : kaydın ilk yüzde kaçı 'start window' sayılacak (0.3 -> ilk %30)
    end_frac     : kaydın son yüzde kaçı 'end window' sayılacak (0.3 -> son %30)

    Mantık:
      1) z-score normalize et
      2) z >= z_min olan indexleri al
      3) bunları zamana göre grupla (min_group_dt içinde kalanlar aynı clap)
      4) her grup için en büyük z-score'u seç
      5) start window içinden z'si en büyük grubu al → START_CLAP
      6) end window içinden z'si en büyük grubu al → END_CLAP
      7) Eğer pencere boşsa, fallback: en erken / en geç güçlü grupları al

    Döndürür:
      t_start, t_end, max_z_global
    """
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 2:
        raise RuntimeError(f"{label}: Signal too short to detect claps.")

    # Z-score
    mean = signal.mean()
    std = signal.std()
    if std < 1e-9:
        raise RuntimeError(f"{label}: Signal has almost no variation; cannot detect claps.")

    z = (signal - mean) / std
    max_z_global = float(z.max())

    cand_idx = np.where(z >= z_min)[0]
    if len(cand_idx) == 0:
        raise RuntimeError(
            f"{label}: No samples above z_min={z_min:.1f} (max z={max_z_global:.2f})."
        )

    # Gruplama
    groups = []
    current_group = [cand_idx[0]]
    for i in cand_idx[1:]:
        if t[i] - t[current_group[-1]] < min_group_dt:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    if len(groups) == 1:
        g = np.asarray(groups[0])
        best_idx = g[np.argmax(z[g])]
        t_single = float(t[best_idx])
        raise RuntimeError(
            f"{label}: Only one clap-like group detected around t={t_single:.2f}s "
            f"(max z={max_z_global:.2f}). Need two (start & end)."
        )

    # Her grup için peak çıkar
    group_peaks = []
    for g in groups:
        g = np.asarray(g)
        best_idx = g[np.argmax(z[g])]
        group_peaks.append(
            {
                "idx": int(best_idx),
                "time": float(t[best_idx]),
                "z": float(z[best_idx]),
            }
        )

    # DEBUG: tüm grup peaklerini yaz
    print(f"[DEBUG] {label}: candidate clap groups (time, z):")
    for i, gp in enumerate(group_peaks, 1):
        print(f"    Group {i}: t={gp['time']:.3f}s, z={gp['z']:.2f}")

    # Kayıt süresi ve pencereler
    t_min = float(t.min())
    t_max = float(t.max())
    T = t_max - t_min
    start_window_end = t_min + start_frac * T
    end_window_start = t_max - end_frac * T

    start_candidates = [gp for gp in group_peaks if gp["time"] <= start_window_end]
    end_candidates = [gp for gp in group_peaks if gp["time"] >= end_window_start]

    # Fallback: pencere boşsa globalden doldur
    if not start_candidates:
        print(f"[WARN] {label}: no start-window candidates, using earliest strong group instead.")
        start_candidates = [min(group_peaks, key=lambda d: d["time"])]

    if not end_candidates:
        print(f"[WARN] {label}: no end-window candidates, using latest strong group instead.")
        end_candidates = [max(group_peaks, key=lambda d: d["time"])]

    # Start / end seçimi (pencere içindeki en yüksek z)
    start_gp = max(start_candidates, key=lambda d: d["z"])
    end_gp = max(end_candidates, key=lambda d: d["z"])

    # Aynı grubu seçtiysek fallback: global en güçlü 2 farklı grup
    if start_gp["idx"] == end_gp["idx"]:
        print(f"[WARN] {label}: start and end candidates are same group, falling back to global top-2.")
        group_peaks_sorted = sorted(group_peaks, key=lambda d: d["z"], reverse=True)
        if len(group_peaks_sorted) < 2:
            raise RuntimeError(f"{label}: cannot find two distinct clap groups.")
        start_gp, end_gp = sorted(group_peaks_sorted[:2], key=lambda d: d["time"])

    t_start = start_gp["time"]
    t_end = end_gp["time"]

    if t_end <= t_start:
        raise RuntimeError(
            f"{label}: Detected end clap time <= start clap time (start={t_start}, end={t_end})."
        )

    print(
        f"[DEBUG] {label}: selected claps → "
        f"start t={t_start:.3f}s (z={start_gp['z']:.2f}), "
        f"end t={t_end:.3f}s (z={end_gp['z']:.2f})"
    )

    return t_start, t_end, max_z_global


class MultiCameraIMUSync:
    """
    Sync IMU + FRONT VIDEO + SIDE VIDEO using clap peaks in IMU and audio.
    """

    def __init__(self, imu_zip, front_video, side_video,
                 z_min_imu=3.0, z_min_audio=3.0):
        self.imu_zip = imu_zip
        self.front_video = front_video
        self.side_video = side_video
        self.z_min_imu = z_min_imu
        self.z_min_audio = z_min_audio

        self.df_imu = None
        self.t_imu = None
        self.a_mag = None

        # Clap markers
        self.t_imu_start = None
        self.t_imu_end = None
        self.t_front_start = None
        self.t_front_end = None
        self.t_side_start = None
        self.t_side_end = None

        # FPS of each video
        self.fps_front = None
        self.fps_side = None

    # --------------------------------------------------------
    # LOAD IMU
    # --------------------------------------------------------
    def load_imu(self):
        with zipfile.ZipFile(self.imu_zip, "r") as z:
            accel_name = None
            for f in z.namelist():
                if f.lower().endswith("accelerometer.csv") and "uncalibrated" not in f.lower():
                    accel_name = f
                    break

            if accel_name is None:
                raise FileNotFoundError("Accelerometer.csv not found in IMU ZIP.")

            with z.open(accel_name) as f:
                df = pd.read_csv(f)

        # Time column
        if "seconds_elapsed" in df.columns:
            t = df["seconds_elapsed"].astype(float).values
        elif "time" in df.columns:
            t = df["time"].astype(float).values
        else:
            raise ValueError("No usable time column in IMU CSV.")

        ax = df["x"].astype(float).values
        ay = df["y"].astype(float).values
        az = df["z"].astype(float).values
        a_mag = np.sqrt(ax**2 + ay**2 + az**2)

        self.df_imu = df
        self.t_imu = t
        self.a_mag = a_mag

    # --------------------------------------------------------
    # AUDIO CLAP DETECTION (video)
    # --------------------------------------------------------
    def detect_video_claps(self, path, label):
        """
        Video dosyasını açar, audio track'ini alır,
        audio envelope üzerinden start/end clap arar.
        """
        clip = mp.VideoFileClip(path)
        audio = clip.audio
        if audio is None:
            raise RuntimeError(f"Video {path} has no audio track")

        fps_video = clip.fps

        # Clap tespiti için makul bir örnekleme hızı
        target_fps_audio = 2000.0

        chunks = []
        chunk_size = int(target_fps_audio * 0.5)  # 0.5 sn

        for chunk in audio.iter_chunks(
            chunksize=chunk_size,
            fps=target_fps_audio,
            quantize=False,
            nbytes=2
        ):
            chunks.append(chunk)

        if len(chunks) == 0:
            raise RuntimeError(f"{label}: audio iter_chunks returned no data")

        samples = np.concatenate(chunks, axis=0)

        # Mono
        if samples.ndim == 2:
            mono = samples.mean(axis=1)
        else:
            mono = samples

        envelope = np.abs(mono)
        n = len(mono)
        t_audio = np.arange(n) / target_fps_audio

        t_start, t_end, max_z = detect_start_end_claps_windowed(
            t_audio,
            envelope,
            z_min=self.z_min_audio,
            min_group_dt=0.05,
            start_frac=0.3,
            end_frac=0.3,
            label=label + " (AUDIO)"
        )

        print(f"[{label}] audio max z-score ~ {max_z:.2f}")
        return t_start, t_end, fps_video

    # --------------------------------------------------------
    # MASTER SYNC
    # --------------------------------------------------------
    def sync_all(self):
        print("\n=== LOADING IMU ===")
        self.load_imu()

        print("Detecting IMU clap peaks (accelerometer)...")
        self.t_imu_start, self.t_imu_end, max_z_imu = detect_start_end_claps_windowed(
            self.t_imu,
            self.a_mag,
            z_min=self.z_min_imu,
            min_group_dt=0.4,
            start_frac=0.3,
            end_frac=0.3,
            label="IMU"
        )
        print(f"IMU start={self.t_imu_start:.3f}   IMU end={self.t_imu_end:.3f}   (max z={max_z_imu:.2f})")

        if max_z_imu < 4.0:
            print("[WARN] IMU claps are relatively weak (max z < 4.0).")

        print("\n=== FRONT VIDEO ===")
        self.t_front_start, self.t_front_end, self.fps_front = \
            self.detect_video_claps(self.front_video, label="FRONT VIDEO")
        print(f"Front start={self.t_front_start:.3f}   Front end={self.t_front_end:.3f}")

        print("\n=== SIDE VIDEO ===")
        self.t_side_start, self.t_side_end, self.fps_side = \
            self.detect_video_claps(self.side_video, label="SIDE VIDEO")
        print(f"Side start={self.t_side_start:.3f}   Side end={self.t_side_end:.3f}")

        print("\n=== BUILD SYNCED DATA ===")

        # Sadece IMU clap aralığını kullan
        imu_mask = (self.t_imu >= self.t_imu_start) & (self.t_imu <= self.t_imu_end)
        t_imu_clipped = self.t_imu[imu_mask]
        a_mag_clipped = self.a_mag[imu_mask]
        df_imu_clipped = self.df_imu.loc[imu_mask].reset_index(drop=True)

        imu_interval = self.t_imu_end - self.t_imu_start
        if imu_interval <= 0:
            raise RuntimeError("IMU interval is non-positive. Check clap detection on IMU.")

        front_interval = self.t_front_end - self.t_front_start
        side_interval = self.t_side_end - self.t_side_start
        if front_interval <= 0 or side_interval <= 0:
            raise RuntimeError("Video intervals are non-positive. Check clap detection on videos.")

        scale_front = front_interval / imu_interval
        scale_side = side_interval / imu_interval

        t_front_video = self.t_front_start + (t_imu_clipped - self.t_imu_start) * scale_front
        t_side_video = self.t_side_start + (t_imu_clipped - self.t_imu_start) * scale_side

        front_frame = np.round(t_front_video * self.fps_front).astype(int)
        side_frame = np.round(t_side_video * self.fps_side).astype(int)

        df = df_imu_clipped.copy()
        df["a_mag"] = a_mag_clipped
        df["t_front_video"] = t_front_video
        df["t_side_video"] = t_side_video
        df["front_frame"] = front_frame
        df["side_frame"] = side_frame

        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync IMU + 2 Video (Front + Side) using clap peaks.")
    parser.add_argument("--imu", required=True, help="Path to IMU ZIP file")
    parser.add_argument("--front", required=True, help="Front camera video file (mov/mp4)")
    parser.add_argument("--side", required=True, help="Side camera video file (mov/mp4)")
    parser.add_argument("--out", required=True, help="Output synced CSV filename")
    parser.add_argument("--z-min-imu", type=float, default=3.0, help="IMU z_min for clap candidate detection")
    parser.add_argument("--z-min-audio", type=float, default=3.0, help="Audio z_min for clap candidate detection")

    args = parser.parse_args()

    sync = MultiCameraIMUSync(
        imu_zip=args.imu,
        front_video=args.front,
        side_video=args.side,
        z_min_imu=args.z_min_imu,
        z_min_audio=args.z_min_audio,
    )

    df = sync.sync_all()
    df.to_csv(args.out, index=False)
    print(f"\n[OK] Saved synced file → {args.out}")
