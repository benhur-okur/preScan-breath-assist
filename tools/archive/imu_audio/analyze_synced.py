import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

# python analyze_synced.py outputs/sinan_1.csv  --case-type normal

def _basic_interval_stats(t: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Basit zaman istatistikleri (duration, mean dt, min dt, max dt).
    """
    if t.size < 2:
        return 0.0, np.nan, np.nan, np.nan
    t = t.astype(float)
    dt = np.diff(t)
    return float(t[-1] - t[0]), float(dt.mean()), float(dt.min()), float(dt.max())


def _estimate_fps(t_video: np.ndarray, frames: np.ndarray) -> float:
    """
    Video zaman ekseni ve frame indexlerinden medyan FPS tahmini.
    """
    if t_video.size < 2:
        return float("nan")
    t = t_video.astype(float)
    f = frames.astype(float)
    dt = np.diff(t)
    df = np.diff(f)

    mask = (dt > 1e-6) & (df > 0)
    if not np.any(mask):
        return float("nan")

    fps_samples = df[mask] / dt[mask]
    return float(np.median(fps_samples))


def _analyze_case_pattern(seconds: np.ndarray, amag: np.ndarray, case_type: str) -> dict:
    """
    a_mag sinyaline bakıp case'e göre daha gerçekçi heuristic yorum döner.
    - NORMAL: erken/orta/son benzer hareket mi?
    - INHALE_HOLD / EXHALE_HOLD: orta bölüm belirgin şekilde daha sakin mi?
    - IRREGULAR: genel varyans yüksek, dalgalı mı?

    status: "ok" | "warn" | "fail"
    """
    t = seconds.astype(float)
    v = amag.astype(float)

    if len(t) < 10:
        return {
            "message": "[WARN] IMU sequence too short for case-level analysis.",
            "std_all": np.nan,
            "std_early": np.nan,
            "std_mid": np.nan,
            "std_late": np.nan,
            "status": "warn",
        }

    dur = t[-1] - t[0]

    # Normalize (mean çıkar)
    v_centered = v - v.mean()
    std_all = float(np.std(v_centered))

    # İlk %20, orta %60, son %20 segment
    n = len(v_centered)
    i1 = int(0.2 * n)
    i2 = int(0.8 * n)
    early = v_centered[:i1]
    middle = v_centered[i1:i2]
    late = v_centered[i2:]

    std_early = float(np.std(early))
    std_mid = float(np.std(middle))
    std_late = float(np.std(late))

    eps = 1e-6
    ratio_mid_all = std_mid / (std_all + eps)
    ratio_early_mid = std_early / (std_mid + eps)
    ratio_late_mid = std_late / (std_mid + eps)

    msg_lines = []
    msg_lines.append(f"[CASE] Type = {case_type}, duration ≈ {dur:.2f} s")
    msg_lines.append(
        f"        std_all={std_all:.4f}, std_early={std_early:.4f}, "
        f"std_mid={std_mid:.4f}, std_late={std_late:.4f}"
    )
    msg_lines.append(
        f"        ratios: mid/all={ratio_mid_all:.3f}, "
        f"early/mid={ratio_early_mid:.2f}, late/mid={ratio_late_mid:.2f}"
    )

    status = "info"

    # ---------------- NORMAL ----------------
    if case_type == "normal":
        # Neredeyse hiç hareket yoksa (cihaz yanlış takılmış olabilir)
        if std_all < 0.02:
            msg_lines.append(
                "[FAIL] Normal case için ivme değişimi çok düşük; neredeyse hiç nefes hareketi yok gibi."
            )
            status = "fail"

        else:
            # Tüm segmentler benzer seviyede hareketli mi?
            if (
                0.6 <= ratio_mid_all <= 1.4
                and 0.5 <= ratio_early_mid <= 2.0
                and 0.5 <= ratio_late_mid <= 2.0
            ):
                msg_lines.append(
                    "[OK] Normal case pattern: baş/orta/son segmentlerde benzer seviyede nefes hareketi var."
                )
                status = "ok"
            # Orta segment bariz şekilde daha sakin → muhtemel mini-hold
            elif ratio_mid_all < 0.4 and ratio_early_mid > 2.5 and ratio_late_mid > 2.5:
                msg_lines.append(
                    "[WARN] Normal case içinde orta bölüm belirgin şekilde daha sakin; "
                    "kısa bir hold veya göğüs hareketinin az yakalandığı bir aralık olabilir."
                )
                status = "warn"
            else:
                msg_lines.append(
                    "[WARN] Normal case için baş/orta/son varyansları arasında belirgin farklar var. "
                    "Bu kayıt yine de kullanılabilir ama nefes paterni tamamen 'düz normal' olmayabilir."
                )
                status = "warn"

    # ---------------- INHALE HOLD ----------------
    elif case_type == "inhale_hold":
        # Güzel bir hold için: mid, tüm kayda göre çok sakin olsun
        # ve early/late mid'den çok daha hareketli olsun.
        if (
            ratio_mid_all < 0.25
            and ratio_early_mid > 3.0
            and ratio_late_mid > 3.0
        ):
            msg_lines.append(
                "[OK] Inhale_hold pattern: orta bölüm, baş/son segmentlere göre belirgin şekilde daha sakin "
                "(nefes tutma dönemi net)."
            )
            status = "ok"
        # Borderline: yine mid daha sakin ama fark aşırı değil → kullanılabilir ama dikkatli
        elif (
            ratio_mid_all < 0.45
            and ratio_early_mid > 2.0
            and ratio_late_mid > 2.0
        ):
            msg_lines.append(
                "[WARN] Inhale_hold pattern kısmen var: orta bölüm daha sakin ama fark çok büyük değil. "
                "Hold süresi kısa veya hafif göğüs hareketi devam ediyor olabilir. "
                "Eğitimde kullanılabilir ama isterseniz 'borderline' olarak işaretleyin."
            )
            status = "warn"
        else:
            msg_lines.append(
                "[FAIL] Inhale_hold için orta bölüm, baş/son segmentlere göre yeterince sakin değil. "
                "Hold net görünmüyor; bu kayıt inhale_hold protokolüne uymuyor olabilir."
            )
            status = "fail"

    # ---------------- EXHALE HOLD ----------------
    elif case_type == "exhale_hold":
        if (
            ratio_mid_all < 0.25
            and ratio_early_mid > 3.0
            and ratio_late_mid > 3.0
        ):
            msg_lines.append(
                "[OK] Exhale_hold pattern: orta bölüm, baş/son segmentlere göre belirgin şekilde daha sakin "
                "(nefes tutma dönemi net)."
            )
            status = "ok"
        elif (
            ratio_mid_all < 0.45
            and ratio_early_mid > 2.0
            and ratio_late_mid > 2.0
        ):
            msg_lines.append(
                "[WARN] Exhale_hold pattern kısmen var: orta bölüm daha sakin ama fark çok büyük değil. "
                "Hold süresi kısa veya hafif göğüs hareketi devam ediyor olabilir. "
                "Eğitimde kullanılabilir ama isterseniz 'borderline' olarak işaretleyin."
            )
            status = "warn"
        else:
            msg_lines.append(
                "[FAIL] Exhale_hold için orta bölüm, baş/son segmentlere göre yeterince sakin değil. "
                "Hold net görünmüyor; bu kayıt exhale_hold protokolüne uymuyor olabilir."
            )
            status = "fail"

    # ---------------- IRREGULAR ----------------
    elif case_type == "irregular":
        if std_all < 0.03:
            msg_lines.append(
                "[WARN] Irregular case için sinyal çok sakin görünüyor; "
                "beklenenden daha düzenli bir nefes paterni olabilir."
            )
            status = "warn"
        else:
            msg_lines.append(
                "[OK] Irregular case: genel varyans normalden yüksek ve zaman içinde dalgalı; "
                "düzensiz nefes davranışı için makul görünüyor."
            )
            status = "ok"

    else:
        msg_lines.append("[INFO] Unknown case_type, no pattern heuristic applied.")
        status = "info"

    return {
        "message": "\n  ".join(msg_lines),
        "std_all": std_all,
        "std_early": std_early,
        "std_mid": std_mid,
        "std_late": std_late,
        "status": status,
        "ratio_mid_all": ratio_mid_all,
        "ratio_early_mid": ratio_early_mid,
        "ratio_late_mid": ratio_late_mid,
    }


def analyze_synced_csv(path: str, case_type: str = None) -> None:
    if not os.path.exists(path):
        print(f"[FAIL] File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"=== ANALYZING SYNCED FILE ===")
    print(f"File: {os.path.basename(path)}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 70)

    # ---- IMU TIME AXIS ----
    if "seconds_elapsed" not in df.columns:
        print("[FAIL] 'seconds_elapsed' column not found. This does not look like a synced IMU file.")
        sys.exit(1)

    t_imu = df["seconds_elapsed"].values.astype(float)
    imu_dur, imu_mean_dt, imu_min_dt, imu_max_dt = _basic_interval_stats(t_imu)
    imu_hz = 1.0 / imu_mean_dt if imu_mean_dt > 0 else float("nan")

    print("IMU timeline (seconds_elapsed):")
    print(f"  t_min       : {t_imu[0]:.3f} s")
    print(f"  t_max       : {t_imu[-1]:.3f} s")
    print(f"  duration    : {imu_dur:.3f} s")
    print(f"  mean Δt     : {imu_mean_dt:.4f} s  (~{imu_hz:.1f} Hz)")
    print(f"  min Δt / max Δt : {imu_min_dt:.4f} / {imu_max_dt:.4f} s")
    print()

    # Global QA state
    can_use_for_training = True
    quality_flags = []
    interpretation_lines = []

    # ---- IMU BASIC SANITY CHECKS ----
    if imu_dur < 5.0:
        print("[FAIL] IMU duration < 5s, recording too short for this project.")
        can_use_for_training = False
    if imu_dur > 60.0:
        print("[FAIL] IMU duration > 60s, this is outside expected range.")
        can_use_for_training = False

    if imu_mean_dt <= 0 or np.isnan(imu_mean_dt):
        print("[FAIL] IMU mean Δt invalid (NaN or non-positive).")
        can_use_for_training = False
    if imu_min_dt < 0.001 or imu_max_dt > 0.05:
        print("[WARN] IMU Δt range suspicious (min_dt/max_dt out of [0.001, 0.05] s).")

    if "a_mag" in df.columns:
        std_all_global = float(np.std(df["a_mag"].values))
        if std_all_global < 0.01:
            print(
                "[FAIL] Global IMU magnitude std is extremely low (<0.01). "
                "Sensor may not have captured any meaningful motion."
            )
            can_use_for_training = False

    # ---- FRONT & SIDE VAR MI? ----
    has_front = ("t_front_video" in df.columns) and ("front_frame" in df.columns)
    has_side = ("t_side_video" in df.columns) and ("side_frame" in df.columns)

    front_dur = side_dur = None
    front_fps = side_fps = float("nan")

    # ---- FRONT VIDEO ----
    if has_front:
        t_front = df["t_front_video"].values.astype(float)
        front_dur, _, _, _ = _basic_interval_stats(t_front)
        print("Front video timeline (t_front_video):")
        print(f"  t_min       : {t_front[0]:.3f} s")
        print(f"  t_max       : {t_front[-1]:.3f} s")
        print(f"  duration    : {front_dur:.3f} s")

        front_fps = _estimate_fps(t_front, df["front_frame"].values)
        if not np.isnan(front_fps):
            print(f"  est. FPS    : {front_fps:.2f}")
            if front_fps < 20 or front_fps > 240:
                print("[FAIL] Estimated front FPS is out of realistic range [20, 240].")
                can_use_for_training = False
        else:
            print("  est. FPS    : N/A (could not estimate)")

        if imu_dur > 0:
            ratio = front_dur / imu_dur
            print(f"  duration ratio (front/imu): {ratio:.3f}")
            if abs(ratio - 1.0) < 0.05:
                quality_flags.append("[OK] Front duration ~ IMU duration (within 5%).")
            elif abs(ratio - 1.0) < 0.15:
                quality_flags.append(
                    "[WARN] Front duration differs from IMU by 5–15%. Check claps & mapping."
                )
            else:
                quality_flags.append(
                    "[FAIL] Front duration differs from IMU by >15%. "
                    "Likely wrong clap detection on front."
                )
                can_use_for_training = False
        print()
    else:
        print("[INFO] No front video columns ('t_front_video', 'front_frame') found.\n")

    # ---- SIDE VIDEO ----
    if has_side:
        t_side = df["t_side_video"].values.astype(float)
        side_dur, _, _, _ = _basic_interval_stats(t_side)
        print("Side video timeline (t_side_video):")
        print(f"  t_min       : {t_side[0]:.3f} s")
        print(f"  t_max       : {t_side[-1]:.3f} s")
        print(f"  duration    : {side_dur:.3f} s")

        side_fps = _estimate_fps(t_side, df["side_frame"].values)
        if not np.isnan(side_fps):
            print(f"  est. FPS    : {side_fps:.2f}")
            if side_fps < 20 or side_fps > 240:
                print("[FAIL] Estimated side FPS is out of realistic range [20, 240].")
                can_use_for_training = False
        else:
            print("  est. FPS    : N/A (could not estimate)")

        if imu_dur > 0:
            ratio = side_dur / imu_dur
            print(f"  duration ratio (side/imu): {ratio:.3f}")
            if abs(ratio - 1.0) < 0.05:
                quality_flags.append("[OK] Side duration ~ IMU duration (within 5%).")
            elif abs(ratio - 1.0) < 0.15:
                quality_flags.append(
                    "[WARN] Side duration differs from IMU by 5–15%. Check claps & mapping."
                )
            else:
                quality_flags.append(
                    "[FAIL] Side duration differs from IMU by >15%. "
                    "Likely wrong clap detection on side."
                )
                can_use_for_training = False
        print()
    else:
        print("[INFO] No side video columns ('t_side_video', 'side_frame') found.\n")

    # ---- CASE-SPECIFIC IMU PATTERN ----
    case_analysis = None
    if case_type is not None:
        if "a_mag" not in df.columns:
            print("[WARN] 'a_mag' column not found; cannot perform case-level IMU pattern analysis.\n")
        else:
            print("-" * 70)
            print("CASE-LEVEL IMU PATTERN CHECK:")
            case_analysis = _analyze_case_pattern(
                df["seconds_elapsed"].values,
                df["a_mag"].values,
                case_type=case_type,
            )
            print(" ", case_analysis["message"])
            if case_analysis["status"] == "fail":
                can_use_for_training = False
            print()

    # ------------------------------------------------------------------
    # İNSAN GİBİ YORUM / INTERPRETATION
    # ------------------------------------------------------------------
    print("=" * 70)
    print("INTERPRETATION (INSIGHT LEVEL):")

    # 1) SIDE kamera yorumu
    if has_side and imu_dur > 0 and side_dur is not None:
        ratio_side = side_dur / imu_dur
        if ratio_side < 0.85 or ratio_side > 1.15:
            text = (
                "1) SIDE CAMERA SENKRONİZASYONU SIKINTILI:\n"
                f"   - IMU süresi ≈ {imu_dur:.2f} s\n"
                f"   - Side video süresi ≈ {side_dur:.2f} s (ratio={ratio_side:.3f})\n"
                "   → Side kamera için clap büyük ihtimalle yanlış tespit edildi. "
                "Clap sesi zayıf olabilir, ya da audio'da başka bir peak clap sanılmış olabilir.\n"
                "   → Bu kayıt için side kamera zaman ekseni güvenilir değil; model eğitiminde side view kullanmamalısın."
            )
            interpretation_lines.append(text)
            if not np.isnan(side_fps) and side_fps > 240:
                interpretation_lines.append(
                    f"   - Tahmini SIDE FPS ≈ {side_fps:.2f} fps → gerçekçi değil. "
                    "Bu da clap aralığının yanlış alındığını gösteriyor."
                )
        else:
            interpretation_lines.append(
                "1) SIDE CAMERA SENKRONİZASYONU MAKUL:\n"
                f"   - IMU süresi ≈ {imu_dur:.2f} s\n"
                f"   - Side video süresi ≈ {side_dur:.2f} s (ratio={ratio_side:.3f})\n"
                "   → Side clap tespiti süre bazında IMU ile uyumlu görünüyor."
            )
    elif has_side:
        interpretation_lines.append(
            "1) SIDE CAMERA: Side timeline var ama IMU süresi veya side süresi düzgün hesaplanamamış."
        )
    else:
        interpretation_lines.append("1) SIDE CAMERA: Bu kayıtta side video senkron bilgisi yok.")

    # 2) FRONT kamera yorumu
    if has_front and imu_dur > 0 and front_dur is not None:
        ratio_front = front_dur / imu_dur
        if abs(ratio_front - 1.0) < 0.05:
            interpretation_lines.append(
                "\n2) FRONT CAMERA SENKRONU GÜZEL:\n"
                f"   - IMU süresi ≈ {imu_dur:.2f} s\n"
                f"   - Front video süresi ≈ {front_dur:.2f} s (ratio={ratio_front:.3f})\n"
                f"   - Tahmini FRONT FPS ≈ {front_fps:.2f} fps\n"
                "   → Front kamera için clap detection ve mapping oldukça düzgün görünüyor."
            )
        else:
            interpretation_lines.append(
                "\n2) FRONT CAMERA SENKRONU ŞÜPHELİ:\n"
                f"   - IMU süresi ≈ {imu_dur:.2f} s\n"
                f"   - Front video süresi ≈ {front_dur:.2f} s (ratio={ratio_front:.3f})\n"
                "   → Front tarafında da clap algısı veya mapping sorunlu olabilir."
            )
    else:
        interpretation_lines.append("\n2) FRONT CAMERA: Front senkron bilgisi yok veya eksik.")

    # 3) CASE analizi yorumu
    if case_analysis is not None:
        if case_type == "normal":
            std_e = case_analysis["std_early"]
            std_m = case_analysis["std_mid"]
            std_l = case_analysis["std_late"]
            r_ma = case_analysis["ratio_mid_all"]
            r_em = case_analysis["ratio_early_mid"]
            r_lm = case_analysis["ratio_late_mid"]

            if not any(np.isnan(v) for v in [std_e, std_m, std_l]):
                interpretation_lines.append(
                    "\n3) CASE-LEVEL (NORMAL) PATTERN YORUMU:\n"
                    f"   - Baş (early) std  ≈ {std_e:.4f}\n"
                    f"   - Orta (mid)  std  ≈ {std_m:.4f}\n"
                    f"   - Son (late)  std  ≈ {std_l:.4f}\n"
                    f"   - mid/all ≈ {r_ma:.3f}, early/mid ≈ {r_em:.2f}, late/mid ≈ {r_lm:.2f}\n"
                )
                if case_analysis["status"] == "ok":
                    interpretation_lines.append(
                        "   → Baş/orta/son segmentler benzer hareket seviyesinde; "
                        "bu, stabil bir normal nefes paterniyle uyumlu."
                    )
                elif case_analysis["status"] == "warn":
                    interpretation_lines.append(
                        "   → Normal case için hafif dengesizlikler var (örneğin ortada biraz daha sakin bir bölüm gibi). "
                        "Yine de bu kayıt genelde kullanılabilir, sadece tamamen textbook-normal olmayabilir."
                    )
                elif case_analysis["status"] == "fail":
                    interpretation_lines.append(
                        "   → Normal case etiketiyle uyuşmayan bir pattern tespit edildi "
                        "(örneğin ortada bariz hold veya tüm kayıtta neredeyse hiç hareket olmaması gibi)."
                    )
        else:
            interpretation_lines.append(
                "\n3) CASE-LEVEL PATTERN ÖZETİ:\n"
                f"   {case_analysis['message']}"
            )

    for line in interpretation_lines:
        print(line)

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print("SYNC QUALITY SUMMARY:")
    if not quality_flags:
        print("  [INFO] No video timelines found or no checks performed.")
    else:
        for q in quality_flags:
            print(" ", q)

    if quality_flags:
        if any(q.startswith("[FAIL]") for q in quality_flags):
            print("\n[GLOBAL STATUS] ❌ Problems detected in sync (see FAIL lines above).")
        elif any(q.startswith("[WARN]") for q in quality_flags):
            print("\n[GLOBAL STATUS] ⚠️ Sync mostly OK but has warnings. Review details above.")
        else:
            print("\n[GLOBAL STATUS] ✅ Sync looks consistent across IMU & videos.")
    else:
        print("\n[GLOBAL STATUS] ℹ️ Not enough information to judge sync.")

    # Eğitim için kullanılabilirlik yorumu
    print("\n" + "=" * 70)
    print("DATASET USAGE SUGGESTION:")
    if can_use_for_training:
        print("  ✅ Bu kayıt, genel olarak model eğitiminde kullanılabilir durumda görünüyor.")
        print("  Ama yine de görsel ve sinyal bazlı hızlı bir manuel kontrol yapmak her zaman iyi fikirdir.")
    else:
        print("  ❌ Bu kayıtta ciddi senkron/pattern problemleri var.")
        print("  → Bu versiyonu model eğitiminde kullanmaman daha güvenli.")
        print("  → Side/Front clap, case protokolü veya IMU yerleşimini tekrar gözden geçirip çekimi yenilemen önerilir.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a synced IMU + multi-camera CSV (output of sync_manager.py)."
    )
    parser.add_argument("csv_path", type=str, help="Path to synced CSV file")
    parser.add_argument(
        "--case-type",
        type=str,
        choices=["normal", "inhale_hold", "exhale_hold", "irregular"],
        default=None,
        help="Optional: expected breathing case type for IMU pattern sanity-check.",
    )

    args = parser.parse_args()
    analyze_synced_csv(args.csv_path, case_type=args.case_type)


if __name__ == "__main__":
    main()
