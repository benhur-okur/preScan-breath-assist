# ==============================
# Reorganize project into core vs archive (SAFE: move only, no delete)
# Uses ONLY paths seen in PROJECT_TREE.txt (non-heuristic by name matching FINAL decision)
# Run from REPO ROOT.
# ==============================

$ErrorActionPreference = "Stop"

# ---- Toggle dry-run
$DryRun = $false   # <- first run MUST be true. After you like it, set to $false.

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Move-PathIfExists([string]$Source, [string]$DestDir) {
    if (Test-Path $Source) {
        Ensure-Dir $DestDir
        $leaf = Split-Path $Source -Leaf
        $dest = Join-Path $DestDir $leaf

        if ($DryRun) {
            Write-Host "[DRYRUN] Move $Source -> $dest"
        } else {
            Move-Item -Path $Source -Destination $dest -Force
            Write-Host "[OK] Moved  $Source -> $dest"
        }
    } else {
        Write-Host "[SKIP] Not found: $Source"
    }
}

function Move-GlobIfExists([string]$Glob, [string]$DestDir) {
    $items = Get-ChildItem -Path $Glob -ErrorAction SilentlyContinue
    if ($null -ne $items -and $items.Count -gt 0) {
        foreach ($it in $items) {
            Move-PathIfExists $it.FullName $DestDir
        }
    } else {
        Write-Host "[SKIP] No match: $Glob"
    }
}

# ---- Archive root (timestamped)
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ArchiveRoot = Join-Path "archive" ("legacy_" + $stamp)

# ---- Create base archive dirs
Ensure-Dir $ArchiveRoot
Ensure-Dir (Join-Path $ArchiveRoot "runs")
Ensure-Dir (Join-Path $ArchiveRoot "plots")
Ensure-Dir (Join-Path $ArchiveRoot "debug")
Ensure-Dir (Join-Path $ArchiveRoot "thresholding")

Write-Host "==== ARCHIVE ROOT: $ArchiveRoot ===="
Write-Host "DryRun = $DryRun"
Write-Host ""

# ==============================
# 1) Move obvious generated plot directory (exists in tree)
# ==============================
# labeledPlots is a generated output directory (by name + tree presence)
Move-PathIfExists "labeledPlots" (Join-Path $ArchiveRoot "plots")

# ==============================
# 2) runs/: archive only legacy/debug/diff/v3 artifacts that exist in tree
#    KEEP: runs/final_model_v4_rgb_bce, runs/final_demo, runs/loso_max_guard (core evidence)
# ==============================

# --- 2A) runs root files that are explicitly v3 artifacts (seen in tree)
Move-PathIfExists "runs\final_temperature_v3.json"       (Join-Path $ArchiveRoot "thresholding")
Move-PathIfExists "runs\final_threshold_grid_v3.csv"     (Join-Path $ArchiveRoot "thresholding")
Move-PathIfExists "runs\final_threshold_v3.json"         (Join-Path $ArchiveRoot "thresholding")
Move-PathIfExists "runs\global_threshold_val_merged.json"(Join-Path $ArchiveRoot "thresholding")

# --- 2B) archive run folders that are explicitly v3/diff/debug in name (seen in tree)
Move-PathIfExists "runs\final_model_v3"           (Join-Path $ArchiveRoot "runs")
Move-PathIfExists "runs\loso_max_diff_final"      (Join-Path $ArchiveRoot "runs")
Move-PathIfExists "runs\loso_max_diff_guard"      (Join-Path $ArchiveRoot "runs")
Move-PathIfExists "runs\eval_final_threshold_v3"  (Join-Path $ArchiveRoot "runs")
Move-PathIfExists "runs\debug_doga"               (Join-Path $ArchiveRoot "runs")

# --- 2C) If there is a generic older folder runs/final_model (seen in tree), archive it
Move-PathIfExists "runs\final_model"              (Join-Path $ArchiveRoot "runs")

# --- 2D) Cache inside runs (seen in tree)
Move-PathIfExists "runs\__pycache__"              (Join-Path $ArchiveRoot "debug")

# ==============================
# 3) Python cache everywhere (safe: auto-generated)
# ==============================
# This does NOT delete; it quarantines caches under archive for cleanliness
# If you prefer delete later, you can, but now it stays safe.
Move-GlobIfExists ".\**\__pycache__" (Join-Path $ArchiveRoot "debug")
Move-GlobIfExists ".\**\*.pyc"       (Join-Path $ArchiveRoot "debug")

Write-Host ""
Write-Host "==== DONE ===="
Write-Host "If DryRun looks correct: set `$DryRun = `$false and rerun."
Write-Host "Nothing was deleted; everything moved into: $ArchiveRoot"
