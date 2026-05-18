# =============================================================================
# run_base_model_1.ps1
# End-to-end runner for MSFDroid Base Model 1 (MLP(H)) — Dex header only.
#
# Pipeline (Phases 2 → 6):
#   1. Optional: install Python dependencies
#   2. Optional: verify environment
#   3. Preprocess APKs → dex_header_features.pt
#   4. Train MLP(H) with checkpoint resume
#   5. Evaluate checkpoint (ACC, F1, AUC)
#
# Usage examples (PowerShell):
#   .\run_base_model_1.ps1
#   $env:APK_ROOT = "D:\datasets\apks"; .\run_base_model_1.ps1
#   $env:SKIP_PREPROCESS = "1"; .\run_base_model_1.ps1
#   $env:FRESH_TRAIN = "1"; $env:EPOCHS = "50"; .\run_base_model_1.ps1
# =============================================================================

# --- Stop on first error -------------------------------------------------------
# In PowerShell 7+ this behaves like bash `set -e`
$ErrorActionPreference = "Stop"

# --- Resolve project root (folder containing this script) ----------------------
# $PSScriptRoot is the directory of this .ps1 file (only_base1_model/)
$ROOT = $PSScriptRoot

# Run all steps from the package root so relative config paths resolve correctly
Set-Location -Path $ROOT

# --- Configurable settings (override via environment variables) ----------------

# APK_ROOT: directory tree containing .apk files (benign/ and malware/ subdirs)
if (-not $env:APK_ROOT) {
    $env:APK_ROOT = Join-Path $ROOT "data\apks"
}
$APK_ROOT = $env:APK_ROOT

# PYTHON: which Python executable to use (venv recommended)
if (-not $env:PYTHON) {
    $env:PYTHON = "python"
}
$PYTHON = $env:PYTHON

# CONFIG: YAML config file path
if (-not $env:CONFIG) {
    $env:CONFIG = Join-Path $ROOT "config\default.yaml"
}
$CONFIG = $env:CONFIG

# EPOCHS: if set, overrides training.epochs in YAML (empty = use YAML default)
$EPOCHS = $env:EPOCHS

# INSTALL_DEPS: "1" runs pip install -r requirements.txt
$INSTALL_DEPS = if ($env:INSTALL_DEPS) { $env:INSTALL_DEPS } else { "0" }

# VERIFY_SETUP: "1" runs scripts/verify_setup.py before preprocessing
$VERIFY_SETUP = if ($env:VERIFY_SETUP) { $env:VERIFY_SETUP } else { "1" }

# SKIP_PREPROCESS: "1" skips Phase 2 (expects existing processed .pt)
$SKIP_PREPROCESS = if ($env:SKIP_PREPROCESS) { $env:SKIP_PREPROCESS } else { "0" }

# SKIP_TRAIN: "1" skips Phase 5 training
$SKIP_TRAIN = if ($env:SKIP_TRAIN) { $env:SKIP_TRAIN } else { "0" }

# SKIP_EVAL: "1" skips Phase 6 standalone evaluation
$SKIP_EVAL = if ($env:SKIP_EVAL) { $env:SKIP_EVAL } else { "0" }

# FRESH_TRAIN: "1" passes --fresh (ignore existing checkpoint)
$FRESH_TRAIN = if ($env:FRESH_TRAIN) { $env:FRESH_TRAIN } else { "0" }

# PREPROCESS_LIMIT: if set (e.g. "100"), only process first N APKs
$PREPROCESS_LIMIT = $env:PREPROCESS_LIMIT

# --- Python import path --------------------------------------------------------
# Prepend package root so `import src....` works from any working directory
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$ROOT;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $ROOT
}

# --- Helper: print a visible section banner ------------------------------------
function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "============================================================================="
    Write-Host "  $Title"
    Write-Host "============================================================================="
}

# --- Step 0: Show configuration ------------------------------------------------
Write-Section "Base Model 1 (MLP(H)) - configuration"
Write-Host "ROOT:            $ROOT"
Write-Host "APK_ROOT:        $APK_ROOT"
Write-Host "PYTHON:          $PYTHON"
Write-Host "CONFIG:          $CONFIG"
Write-Host "SKIP_PREPROCESS: $SKIP_PREPROCESS"
Write-Host "SKIP_TRAIN:      $SKIP_TRAIN"
Write-Host "SKIP_EVAL:       $SKIP_EVAL"
Write-Host "FRESH_TRAIN:     $FRESH_TRAIN"
Write-Host "INSTALL_DEPS:    $INSTALL_DEPS"
Write-Host "VERIFY_SETUP:    $VERIFY_SETUP"

# --- Step 1 (optional): Install dependencies -----------------------------------
if ($INSTALL_DEPS -eq "1") {
    Write-Section "Step 1: Installing dependencies"
    # Install torch, sklearn, tqdm, PyYAML, etc.
    & $PYTHON -m pip install -r (Join-Path $ROOT "requirements.txt")
} else {
    Write-Host ""
    Write-Host "(Skipping pip install; set `$env:INSTALL_DEPS = '1' to install requirements.txt)"
}

# --- Step 2 (optional): Verify environment ---------------------------------------
if ($VERIFY_SETUP -eq "1") {
    Write-Section "Step 2: Verifying environment"
    # Phase 1 smoke test: imports, config, artifact directories
    & $PYTHON (Join-Path $ROOT "scripts\verify_setup.py")
}

# --- Step 3: Preprocess APKs (Phase 2) -----------------------------------------
if ($SKIP_PREPROCESS -ne "1") {
    Write-Section "Step 3: Preprocessing APKs (Dex header extraction)"

    if (-not (Test-Path -Path $APK_ROOT -PathType Container)) {
        Write-Host "WARNING: APK_ROOT does not exist: $APK_ROOT"
        Write-Host "         Create it or set `$env:APK_ROOT before running."
        exit 1
    }

    # Build argument list for preprocess_apks module
    $preprocessArgs = @(
        "-m", "src.preprocessing.preprocess_apks",
        "--apk-root", $APK_ROOT
    )
    if ($CONFIG) {
        $preprocessArgs += @("--config", $CONFIG)
    }
    if ($PREPROCESS_LIMIT) {
        # --limit N: only first N APKs (smoke test)
        $preprocessArgs += @("--limit", $PREPROCESS_LIMIT)
    }

    # Extract classes.dex headers → artifacts/processed/dex_header_features.pt
    & $PYTHON @preprocessArgs
} else {
    Write-Host ""
    Write-Host "(Skipping preprocessing; SKIP_PREPROCESS=1)"
}

# Path to preprocessed tensor bundle (required before training)
$PROCESSED_FILE = Join-Path $ROOT "artifacts\processed\dex_header_features.pt"
if (-not (Test-Path -Path $PROCESSED_FILE -PathType Leaf)) {
    Write-Host "ERROR: Preprocessed features not found: $PROCESSED_FILE"
    Write-Host "       Run without SKIP_PREPROCESS=1 or copy features to that path."
    exit 1
}

# --- Step 4: Train MLP(H) (Phases 4–5) -----------------------------------------
if ($SKIP_TRAIN -ne "1") {
    Write-Section "Step 4: Training MLP(H)"

    $trainArgs = @("-m", "src.training.train")
    if ($CONFIG) {
        $trainArgs += @("--config", $CONFIG)
    }
    if ($EPOCHS) {
        $trainArgs += @("--epochs", $EPOCHS)
    }
    if ($FRESH_TRAIN -eq "1") {
        # Do not resume from latest_checkpoint.pth
        $trainArgs += @("--fresh")
    }

    # Train with SGD/BCE; tqdm; per-epoch checkpoint; validation ACC/F1/AUC
    & $PYTHON @trainArgs
} else {
    Write-Host ""
    Write-Host "(Skipping training; SKIP_TRAIN=1)"
}

# Checkpoint written by training (also used for resume)
$CHECKPOINT = Join-Path $ROOT "artifacts\checkpoints\latest_checkpoint.pth"
if (-not (Test-Path -Path $CHECKPOINT -PathType Leaf)) {
    Write-Host "ERROR: Training checkpoint not found: $CHECKPOINT"
    Write-Host "       Train first or place a valid latest_checkpoint.pth there."
    exit 1
}

# --- Step 5: Standalone evaluation (Phase 6) -----------------------------------
if ($SKIP_EVAL -ne "1") {
    Write-Section "Step 5: Evaluation (ACC, F1, AUC)"

    $evalArgs = @(
        "-m", "src.training.evaluate",
        "--split", "val",
        "--checkpoint", $CHECKPOINT
    )
    if ($CONFIG) {
        $evalArgs += @("--config", $CONFIG)
    }

    # Report sklearn metrics on validation split from saved weights
    & $PYTHON @evalArgs
} else {
    Write-Host ""
    Write-Host "(Skipping evaluation; SKIP_EVAL=1)"
}

# --- Done -----------------------------------------------------------------------
Write-Section "Base Model 1 pipeline finished"
Write-Host "Processed features: $PROCESSED_FILE"
Write-Host "Checkpoint:         $CHECKPOINT"
Write-Host "Failed APK log:     $(Join-Path $ROOT 'artifacts\failed_apks.log') (if any failures)"
Write-Host ""
Write-Host "Done."
