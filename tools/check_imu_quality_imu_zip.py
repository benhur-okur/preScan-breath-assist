import argparse
import sys
import os
import zipfile

import numpy as np
import pandas as pd

# Command = python check_imu_quality_imu_zip.py imu_zips/deneme_3_5-10-15.zip --plot

# Sensor Logger örneğine göre default kolon isimleri:
# Accelerometer.csv, Gyroscope.csv, Gravity.csv -> columns: ['time', 'seconds_elapsed', 'z', 'y', 'x']
DEFAULT_TIME_COL = "seconds_elapsed"
DEFAULT_X_COL = "x"
DEFAULT_Y_COL = "y"
DEFAULT_Z_COL = "z"


SENSOR_TO_FILENAME = {
    "accelerometer": "accelerometer.csv",
    "gyroscope": "gyroscope.csv",
    "gravity": "gravity.csv",
}


def load_sensor_dataframe_from_zip(zip_path, sensor_type):
    """
    zip_path: Sensor Logger export .zip
    sensor_type: 'accelerometer' | 'gyroscope' | 'gravity'
    İlgili sensörün CSV dosyasını bulup okur (Uncalibrated olmayan).
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"File not found: {zip_path}")

    target_suffix = SENSOR_TO_FILENAME[sensor_type]

    with zipfile.ZipFile(zip_path, "r") as z:
        selected_name = None
        for name in z.namelist():
            lname = name.lower()
            # Uncalibrated dosyaları istemiyoruz
            if "uncalibrated" in lname:
                continue
            if lname.endswith(target_suffix):
                selected_name = name
                break

        if selected_name is None:
            raise FileNotFoundError(f"No {target_suffix} found inside zip (calibrated).")

        with z.open(selected_name) as f:
            df = pd.read_csv(f)
        source_info = f"{os.path.basename(zip_path)}::{selected_name}"

    return df, source_info


def load_sensor_dataframe(path, sensor_type):
    """
    path: .zip veya tek bir CSV
    sensor_type: 'accelerometer' | 'gyroscope' | 'gravity'
    - .zip ise: ilgili sensör CSV'sini zip içinden bulur.
    - .csv ise: tek dosyayı okur (kullanıcı doğru sensör için çağırmış varsayılır).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.lower().endswith(".zip"):
        return load_sensor_dataframe_from_zip(path, sensor_type)
    else:
        df = pd.read_csv(path)
        source_info = os.path.basename(path)
        return df, source_info


def pick_time_column(df, user_time_col=None):
    # Öncelik: kullanıcı parametresi
    if user_time_col is not None:
        if user_time_col not in df.columns:
            raise ValueError(f"Time column '{user_time_col}' not in CSV. Columns: {df.columns.tolist()}")
        return user_time_col

    # Otomatik seçim: seconds_elapsed varsa onu al, yoksa time
    if "seconds_elapsed" in df.columns:
        return "seconds_elapsed"
    if "time" in df.columns:
        return "time"

    raise ValueError(f"No obvious time column found. Columns: {df.columns.tolist()}")


def check_time_column(df, time_col):
    t = df[time_col].values.astype(float)

    if len(t) < 2:
        return {
            "ok": False,
            "reason": "Too few samples in time column.",
        }

    diffs = np.diff(t)
    non_positive = int(np.sum(diffs <= 0))

    duration = float(t[-1] - t[0])
    mean_dt = float(diffs.mean())
    min_dt = float(diffs.min())
    max_dt = float(diffs.max())
    est_hz = 1.0 / mean_dt if mean_dt > 0 else float("nan")

    result = {
        "ok": True,
        "duration": duration,
        "mean_dt": mean_dt,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "est_hz": est_hz,
        "num_samples": int(len(t)),
        "non_monotonic_steps": non_positive,
    }

    if duration <= 0:
        result["ok"] = False
        result["reason"] = "Non-positive total duration (timestamps may be malformed)."
    elif non_positive > 0:
        result["ok"] = False
        result["reason"] = f"Found {non_positive} non-positive time steps (timestamps not strictly increasing)."

    return result


def check_vector_columns(df, x_col, y_col, z_col, sensor_type):
    """
    x,y,z vektör büyüklüğünü hesaplar.
    sensor_type: accelerometer | gyroscope | gravity
    Aynı hesap ama yorumlar sensör tipine göre değişecek.
    """
    for col in (x_col, y_col, z_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV. Columns: {df.columns.tolist()}")

    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    z = df[z_col].values.astype(float)

    mag = np.sqrt(x**2 + y**2 + z**2)

    mean_mag = float(np.mean(mag))
    std_mag = float(np.std(mag))
    max_mag = float(np.max(mag))

    if std_mag > 1e-9:
        z_peak = (max_mag - mean_mag) / std_mag
    else:
        z_peak = 0.0

    comments = []
    movement_level = "unknown"

    if sensor_type == "accelerometer":
        # m/s^2 -> nefes + clap için
        if std_mag < 0.01:
            movement_level = "very_low"
        elif std_mag < 0.05:
            movement_level = "low_to_medium"
        elif std_mag < 0.2:
            movement_level = "medium"
        else:
            movement_level = "high"

        if std_mag < 0.005:
            comments.append(
                "Acceleration magnitude std is extremely low: signal might be almost constant "
                "(no movement or sensor issue)."
            )
        else:
            comments.append("Acceleration magnitude shows some variation (movement present).")

        if z_peak > 6:
            comments.append(
                "Strong peak detected in acceleration magnitude (likely clap marker present)."
            )
        else:
            comments.append(
                "No very strong peak in acceleration magnitude; clap marker may be weak or missing."
            )

    elif sensor_type == "gyroscope":
        # Gyro -> açısal hız (rad/s veya deg/s) -> std yüksekse gövde/sensör daha fazla dönmüş demektir.
        if std_mag < 0.01:
            movement_level = "very_low"
        elif std_mag < 0.05:
            movement_level = "low"
        elif std_mag < 0.2:
            movement_level = "medium"
        else:
            movement_level = "high"

        comments.append(
            "Gyroscope magnitude variation reflects how much the device rotated during recording."
        )

        if movement_level in ("medium", "high"):
            comments.append(
                "Rotation level is medium/high; device or torso might have moved noticeably. "
                "Check if this matches the intended protocol (patient should mostly stand still)."
            )
        else:
            comments.append(
                "Rotation level is low; device orientation seems relatively stable overall."
            )

        if z_peak > 6:
            comments.append(
                "A strong peak in gyroscope magnitude was detected; there may have been a sudden rotation."
            )

    elif sensor_type == "gravity":
        # Gravity sensörü -> genelde 9.81 civarı, orientation değişince bileşenler değişir.
        if std_mag < 0.01:
            movement_level = "very_low"
        elif std_mag < 0.05:
            movement_level = "low"
        elif std_mag < 0.2:
            movement_level = "medium"
        else:
            movement_level = "high"

        comments.append(
            "Gravity vector magnitude should be close to local g; variation reflects orientation changes."
        )

        if abs(mean_mag - 9.8) > 1.0:
            comments.append(
                "Mean gravity magnitude is far from 9.8; units or scaling might be different than expected."
            )

        if movement_level in ("medium", "high"):
            comments.append(
                "Gravity variation is medium/high; device orientation changed noticeably during recording."
            )
        else:
            comments.append(
                "Gravity variation is low; device orientation seems relatively stable."
            )

    result = {
        "mean_mag": mean_mag,
        "std_mag": std_mag,
        "max_mag": max_mag,
        "z_peak": float(z_peak),
        "movement_level": movement_level,
        "comments": comments,
        "mag_series": mag,
    }

    return result


# ---------------------------------------------------------
# YENİ: Clap peak tespiti (en büyük 2 clap mantığı)
# ---------------------------------------------------------
def detect_clap_peaks_in_signal(t, mag, z_min=3.0, min_group_dt=0.4, max_groups=2):
    """
    t       : zaman dizisi (saniye)
    mag     : vektör büyüklüğü dizisi (ör: |a|)
    z_min   : "aday peak" için minimum z-score eşiği (ör: 3.0)
    min_group_dt: aynı clap grubunda sayacağımız örnekler arası max zaman farkı (s)
    max_groups  : en fazla kaç clap grubu seçeceğiz (ör: 2 -> start ve end)

    Döndürdüğü:
      {
        "all_groups_count": int,
        "selected_count": int,
        "peak_indices": [...],   # seçilen grupların peak index'leri
        "peak_times": [...],     # seçilen peak zamanları
        "peak_zscores": [...],   # seçilen peak z-score'ları
        "max_z": float,          # sinyaldeki global max z-score
      }
    """
    mag = np.asarray(mag, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(mag) < 2:
        return {
            "all_groups_count": 0,
            "selected_count": 0,
            "peak_indices": [],
            "peak_times": [],
            "peak_zscores": [],
            "max_z": 0.0,
        }

    mean = mag.mean()
    std = mag.std()
    if std < 1e-9:
        return {
            "all_groups_count": 0,
            "selected_count": 0,
            "peak_indices": [],
            "peak_times": [],
            "peak_zscores": [],
            "max_z": 0.0,
        }

    z = (mag - mean) / std
    max_z = float(z.max())

    # 1) z >= z_min olanları aday yap
    candidate_idx = np.where(z >= z_min)[0]
    if len(candidate_idx) == 0:
        return {
            "all_groups_count": 0,
            "selected_count": 0,
            "peak_indices": [],
            "peak_times": [],
            "peak_zscores": [],
            "max_z": max_z,
        }

    # 2) Adayları zaman içinde grupla (aynı clap)
    groups = []
    current_group = [candidate_idx[0]]
    for i in candidate_idx[1:]:
        if t[i] - t[current_group[-1]] < min_group_dt:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    all_groups_count = len(groups)

    # 3) Her grup için en güçlü peak'i al
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

    # 4) Grupları z-score'a göre büyükten küçüğe sırala ve en güçlü max_groups tanesini seç
    group_peaks_sorted = sorted(group_peaks, key=lambda d: d["z"], reverse=True)
    selected = group_peaks_sorted[:max_groups]

    # 5) Seçilenleri zamana göre sırala (start & end yorumlamak için)
    selected_sorted_by_time = sorted(selected, key=lambda d: d["time"])

    peak_indices = [g["idx"] for g in selected_sorted_by_time]
    peak_times = [g["time"] for g in selected_sorted_by_time]
    peak_zscores = [g["z"] for g in selected_sorted_by_time]

    return {
        "all_groups_count": all_groups_count,
        "selected_count": len(peak_indices),
        "peak_indices": peak_indices,
        "peak_times": peak_times,
        "peak_zscores": peak_zscores,
        "max_z": max_z,
    }


def plot_magnitude(df, time_col, mag, sensor_type, title_suffix=""):
    """
    |v| vs time plot'u. Clap piki / ani olay varsa onu da işaretler.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; cannot plot.")
        return

    t = df[time_col].values.astype(float)

    if len(t) != len(mag):
        print("[WARN] Time and magnitude length mismatch; skipping plot.")
        return

    # Global peak (bilgi amaçlı)
    peak_idx = int(np.argmax(mag))
    t_peak = t[peak_idx]
    mag_peak = mag[peak_idx]

    plt.figure(figsize=(10, 4))
    plt.plot(t, mag, label=f"|{sensor_type}| magnitude")

    # --- Accelerometer için clap peaklerini işaretle ---
    if sensor_type == "accelerometer":
        clap_info = detect_clap_peaks_in_signal(
            t, mag, z_min=3.0, min_group_dt=0.4, max_groups=4
        )
        for i, tt in enumerate(clap_info["peak_times"]):
            idx = np.argmin(np.abs(t - tt))
            plt.axvline(tt, color="red", linestyle="--", alpha=0.7,
                        label="Clap peak" if i == 0 else None)
            plt.scatter([tt], [mag[idx]], color="red")

    # Global peak çizgisi (daha hafif)
    plt.axvline(t_peak, linestyle=":", alpha=0.3, label=f"Global peak ~ t={t_peak:.2f}s")
    plt.scatter([t_peak], [mag_peak], marker="o")

    plt.xlabel("Time (s)")
    if sensor_type == "accelerometer":
        ylabel = "|a| (m/s²)"
    elif sensor_type == "gyroscope":
        ylabel = "|ω| (angular velocity)"
    else:
        ylabel = "|g| (gravity magnitude)"
    plt.ylabel(ylabel)

    title = f"{sensor_type.capitalize()} magnitude over time"
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_single_sensor(path, sensor_type, time_col_arg, x_col, y_col, z_col, do_plot):
    print("=" * 70)
    print(f"[SENSOR] {sensor_type.upper()}")

    try:
        df, source_info = load_sensor_dataframe(path, sensor_type)
    except Exception as e:
        print(f"[WARN] Could not load {sensor_type} data: {e}")
        return

    print(f"Loaded {sensor_type} data from: {source_info}")
    print(f"Columns: {df.columns.tolist()}")

    # Zaman kolonu
    try:
        time_col = pick_time_column(df, time_col_arg)
    except Exception as e:
        print(f"[FAIL] Time column detection error for {sensor_type}: {e}")
        return

    time_info = check_time_column(df, time_col)
    if not time_info["ok"]:
        print(f"[FAIL] Time axis problem for {sensor_type}:")
        print("  Reason:", time_info.get("reason", "Unknown"))
        return

    print("[OK] Time axis looks valid.")
    print(f"  Number of samples : {time_info['num_samples']}")
    print(f"  Duration          : {time_info['duration']:.3f} s")
    print(f"  mean Δt           : {time_info['mean_dt']:.4f} s  (~{time_info['est_hz']:.1f} Hz)")
    print(f"  min Δt / max Δt   : {time_info['min_dt']:.4f} s / {time_info['max_dt']:.4f} s")
    print(f"  Non-monotonic steps: {time_info['non_monotonic_steps']}")

    # Vektör büyüklüğü
    try:
        vec_info = check_vector_columns(df, x_col, y_col, z_col, sensor_type)
    except Exception as e:
        print(f"[FAIL] Vector column check error for {sensor_type}: {e}")
        return

    print(f"[INFO] {sensor_type.capitalize()} magnitude stats:")
    print(f"  mean |v| : {vec_info['mean_mag']:.4f}")
    print(f"  std  |v| : {vec_info['std_mag']:.4f}")
    print(f"  max  |v| : {vec_info['max_mag']:.4f}")
    print(f"  z_peak   : {vec_info['z_peak']:.2f} (peak vs mean/std)")
    print(f"  movement level (heuristic): {vec_info['movement_level']}")
    print("  Comments:")
    for c in vec_info["comments"]:
        print("   -", c)

    if do_plot:
        print("[PLOT] Showing magnitude over time...")
        plot_magnitude(df, time_col, vec_info["mag_series"], sensor_type, title_suffix=source_info)

    # Final durum
    status = "OK"
    reasons = []

    # --------------------------------------------------
    # Accelerometer için ekstra clap analizi
    # --------------------------------------------------
    if sensor_type == "accelerometer":
        t = df[time_col].values.astype(float)
        mag = vec_info["mag_series"]
        duration = time_info["duration"]

        clap_info = detect_clap_peaks_in_signal(
            t, mag, z_min=3.0, min_group_dt=0.4, max_groups=2
        )

        print("----")
        print("[CLAP DETECTION] (Accelerometer)")
        print(f"  Total clap-like groups above z>=3: {clap_info['all_groups_count']}")
        print(f"  Selected strongest groups        : {clap_info['selected_count']}")

        for i, (tt, zz) in enumerate(zip(clap_info["peak_times"], clap_info["peak_zscores"])):
            rel = (tt - t[0]) / duration if duration > 0 else 0.0
            print(f"   - Selected Peak #{i+1}: t = {tt:.3f} s "
                  f"(relative pos: {rel*100:.1f}% of recording), z = {zz:.2f}")

        # Clap gücü kontrolü
        if clap_info["max_z"] < 4.0:
            status = "WARNING"
            reasons.append(
                f"Global max z-score for acceleration is only {clap_info['max_z']:.2f} (<4.0); "
                "claps are likely too weak. Ask the subject to hit the chest/phone more distinctly."
            )

        # Clap sayısı / konumu kontrolleri
        if clap_info["selected_count"] == 0:
            status = "WARNING"
            reasons.append(
                "No strong clap peaks detected (z>=3) in accelerometer signal. "
                "You may have forgotten the start/end clap, or the chest hit was too soft."
            )
        elif clap_info["selected_count"] == 1:
            status = "WARNING"
            reasons.append(
                "Only one strong clap-like event detected. "
                "Either start or end clap is missing or too weak; consider re-recording."
            )
        else:
            # İki (veya daha fazla) seçilmiş peak var → ilkini START, sonuncuyu END gibi yorumla
            t_start_clap = clap_info["peak_times"][0]
            t_end_clap = clap_info["peak_times"][-1]
            rel_start = (t_start_clap - t[0]) / duration if duration > 0 else 0.0
            rel_end = (t_end_clap - t[0]) / duration if duration > 0 else 0.0

            print(f"  Interpreted START clap ~ t={t_start_clap:.3f}s "
                  f"({rel_start*100:.1f}% of recording).")
            print(f"  Interpreted END clap   ~ t={t_end_clap:.3f}s "
                  f"({rel_end*100:.1f}% of recording).")

            # Clap pozisyonu: kayıt başına/sonuna yakın mı?
            # Ör: start ilk %10 içinde, end son %10 içinde olsun
            if rel_start > 0.10:
                status = "WARNING"
                reasons.append(
                    f"First strong clap (interpreted as START) is relatively late in the recording "
                    f"(~{rel_start*100:.1f}%). Ideally it should be closer to the beginning "
                    "(<= 10%)."
                )

            if rel_end < 0.90:
                status = "WARNING"
                reasons.append(
                    f"Last strong clap (interpreted as END) is too early in the recording "
                    f"(~{rel_end*100:.1f}%). Ideally it should be closer to the end (>= 90%)."
                )

        # Genel hareket seviyesi zaten vec_info'da → ekstra uyarı:
        if vec_info["std_mag"] < 0.005:
            status = "WARNING"
            reasons.append(
                "Very low acceleration variability; check if the phone was really on the chest and the subject was breathing."
            )

    elif sensor_type == "gyroscope":
        if vec_info["movement_level"] in ("medium", "high"):
            status = "WARNING"
            reasons.append(
                "Gyroscope indicates medium/high rotation; ensure the torso/device was not moving too much "
                "during supposed 'standing still' phases."
            )
    elif sensor_type == "gravity":
        # Şimdilik özel FAIL kuralı yok, yorumlar üstte.
        pass

    print("----")
    if status == "OK":
        print(f"[FINAL STATUS] OK: {sensor_type} recording looks usable for analysis.")
    else:
        print(f"[FINAL STATUS] {status}")
        for r in reasons:
            print("  -", r)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "IMU quality checker for breath detection project.\n"
            "Supports Sensor Logger zip exports (Accelerometer/Gyroscope/Gravity CSV) "
            "or a single sensor CSV."
        )
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to .zip (Sensor Logger export) or a single sensor CSV",
    )
    parser.add_argument(
        "--sensor-type",
        type=str,
        default="all",
        choices=["all", "accelerometer", "gyroscope", "gravity"],
        help="Which sensor to analyze (default: all).",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=None,
        help="Name of time column (default: auto-detect 'seconds_elapsed' or 'time').",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default=DEFAULT_X_COL,
        help="Name of X axis column (default: 'x').",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default=DEFAULT_Y_COL,
        help="Name of Y axis column (default: 'y').",
    )
    parser.add_argument(
        "--z-col",
        type=str,
        default=DEFAULT_Z_COL,
        help="Name of Z axis column (default: 'z').",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, plots magnitude over time for visual inspection.",
    )

    args = parser.parse_args()

    if args.sensor_type == "all":
        # Eğer path .zip değilse, all anlamsız → uyarı
        if not args.path.lower().endswith(".zip"):
            print("[FAIL] --sensor-type all only makes sense with a Sensor Logger .zip export.")
            sys.exit(1)

        for sensor in ["accelerometer", "gyroscope", "gravity"]:
            analyze_single_sensor(
                args.path,
                sensor,
                args.time_col,
                args.x_col,
                args.y_col,
                args.z_col,
                args.plot,
            )
    else:
        analyze_single_sensor(
            args.path,
            args.sensor_type,
            args.time_col,
            args.x_col,
            args.y_col,
            args.z_col,
            args.plot,
        )


if __name__ == "__main__":
    main()
