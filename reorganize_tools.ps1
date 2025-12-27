# ==============================
# Reorganize tools/ into core + archive
# Run this from the REPO ROOT (where tools/ lives).
# ==============================

$ErrorActionPreference = "Stop"

# ---- Toggle: if you want a dry-run first, set $DryRun = $true
$DryRun = $true

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Move-IfExists([string]$Source, [string]$DestDir) {
    if (Test-Path $Source) {
        Ensure-Dir $DestDir
        $destPath = Join-Path $DestDir (Split-Path $Source -Leaf)

        if ($DryRun) {
            Write-Host "[DRYRUN] Move $Source -> $destPath"
        } else {
            Move-Item -Path $Source -Destination $destPath -Force
            Write-Host "[OK] Moved $Source -> $destPath"
        }
    } else {
        Write-Host "[SKIP] Not found: $Source"
    }
}

# ---- Paths
$ToolsDir = "tools"
$CoreDir  = Join-Path $ToolsDir "core"
$ArchDir  = Join-Path $ToolsDir "archive"
$ImuAudioDir = Join-Path $ArchDir "imu_audio"
$DebugEvalDir = Join-Path $ArchDir "debug_eval"

# ---- Create dirs (in case)
Ensure-Dir $CoreDir
Ensure-Dir $ImuAudioDir
Ensure-Dir $DebugEvalDir

# ==============================
# 1) CORE (Final pipeline & reproducibility essentials)
# ==============================
$coreFiles = @(
    "predict_video.py",
    "make_windows_for_video.py",
    "train_final_v4.py",
    "train_loso.py",
    "eval_segments.py",
    "eval_single_video_from_manifest.py",
    "window_dataset_builder.py",
    "subject_split_loso.py"
)

# ==============================
# 2) IMU / AUDIO / SYNC (archive)
# ==============================
$imuAudioFiles = @(
    "check_imu_quality_imu_zip.py",
    "label_from_imu.py",
    "plot_synced_imu.py",
    "sync_manager.py",
    "plot_auido_claps.py",
    "analyze_synced.py"
)

# ==============================
# 3) DEBUG / EVAL / THRESHOLD / QA (archive)
# ==============================
$debugEvalFiles = @(
    "choose_threshold_final.py",
    "temperature_scaling_final.py",
    "eval_final_threshold.py",
    "debug_val_separation.py",
    "compare_windowing_train_vs_infer.py",
    "probability_diagnostics.py",
    "plot_labeled.py",
    "clamp_frames.py",
    "qa_clip_variance.py",
    "qa_frames_in_range_batch.py"
)

# ---- Move core first
Write-Host "==== Moving CORE tools -> $CoreDir ===="
foreach ($f in $coreFiles) {
    Move-IfExists (Join-Path $ToolsDir $f) $CoreDir
}

# ---- Move IMU/audio
Write-Host "==== Moving IMU/AUDIO tools -> $ImuAudioDir ===="
foreach ($f in $imuAudioFiles) {
    Move-IfExists (Join-Path $ToolsDir $f) $ImuAudioDir
}

# ---- Move debug/eval
Write-Host "==== Moving DEBUG/EVAL tools -> $DebugEvalDir ===="
foreach ($f in $debugEvalFiles) {
    Move-IfExists (Join-Path $ToolsDir $f) $DebugEvalDir
}

Write-Host ""
Write-Host "==== DONE ===="
Write-Host "Core:        $CoreDir"
Write-Host "Archive IMU: $ImuAudioDir"
Write-Host "Archive DBG: $DebugEvalDir"
Write-Host ""
Write-Host "NOTE: Nothing was deleted. If you want to preview moves, set `$DryRun = `$true and rerun."
