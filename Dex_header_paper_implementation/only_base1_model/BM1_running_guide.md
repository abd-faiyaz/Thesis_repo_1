# Base Model 1 — Running Guide

Run the MSFDroid **Base Model 1** pipeline: Dex header features → **MLP(H)** → validation metrics (ACC, F1, AUC).

No prior knowledge of this repo is required. Follow the steps in order.

---

## What this pipeline does

1. **Preprocess** — reads APK files, extracts all `classes*.dex` headers (multi-dex sum-pool), saves one tensor file.
2. **Train** — trains a small MLP on those features (80/20 random train/val split at load time).
3. **Evaluate** — reports metrics on the validation split.

---

## Before you start

### Dataset layout

Point the pipeline at the **root folder** that contains your APK tree. Labels come from folder names — no CSV needed.

Example (HuggingFace-style layout):

```
/path/to/dataset/
├── 2020/
│   ├── benign/    ← label 0
│   └── malware/   ← label 1
├── 2021/
│   ├── benign/
│   └── malware/
...
```

Any nested `benign/` or `malware/` parent folder works. The scanner finds all `*.apk` files recursively.

### Requirements

- **Linux** (recommended) or Windows via `run_base_model_1.ps1`
- **Python 3.10+**
- **~50 MB** free disk for outputs (for ~40k APKs); APK dataset size is separate
- **GPU optional** — default config uses `cuda` if available, else CPU

---

## Step 0 — Go to the project folder

```bash
cd /path/to/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
```

All commands below assume you are in this directory.

---

## Step 1 — Shared `thesis_venv` (BM1 + Pattern A + Pattern B)

**Why:** One virtual environment at the **git repo root** covers all three pipelines. `run_base_model_1.sh` auto-detects `thesis_venv/bin/python` (no per-pipeline `.venv` needed).

**One-time setup** (from repo root — parent of `Dex_header_paper_implementation/`):

```bash
cd /path/to/thesis_vigidroid
./scripts/setup_thesis_venv.sh
```

Or manually:

```bash
cd /path/to/thesis_vigidroid
python3 -m venv thesis_venv
thesis_venv/bin/pip install -r requirements-thesis-all.txt
```

Optional: `export PYTHON=/path/to/thesis_vigidroid/thesis_venv/bin/python` (run scripts set this automatically if `thesis_venv` exists).

On Windows (PowerShell), use a local `.venv` in this folder instead, or point `PYTHON` at your shared venv’s `python.exe`.

---

## Step 2 — Install dependencies (if not done in Step 1)

**Why:** Installs PyTorch, torchvision, scikit-learn, tqdm, PyYAML, etc. into `thesis_venv`.

Either use Step 1’s `setup_thesis_venv.sh`, or from this pipeline folder:

```bash
INSTALL_DEPS=1 ./run_base_model_1.sh
```

(`INSTALL_DEPS=1` installs from repo-root `requirements-thesis-all.txt` when `thesis_venv` is active.)

---

## Step 3 — Verify the environment

**Why:** Confirms imports, config, and multi-dex settings before a long preprocess run.

```bash
export PYTHONPATH=.
# thesis_venv is used automatically if present at repo root:
../../thesis_venv/bin/python scripts/verify_setup.py
```

Or run the full runner (it calls `verify_setup` by default):

```bash
VERIFY_SETUP=1 SKIP_PREPROCESS=1 SKIP_TRAIN=1 SKIP_EVAL=1 ./run_base_model_1.sh
```

You should see `Phase 1 setup verified` (or similar) with `multidex.mode: sum`.

---

## Step 4 — Point at your APK dataset

**Why:** Default config expects `data/apks/` which may not exist on your machine.

Pick **one** option:

**Option A — environment variable (recommended, no file edits):**

```bash
export APK_ROOT=/path/to/your/dataset
```

**Option B — symlink:**

```bash
mkdir -p data
ln -s /path/to/your/dataset data/apks
```

**Option C — edit `config/default.yaml`:**

Set `paths.apk_root` to your dataset path.

---

## Step 5 — Smoke test (strongly recommended)

**Why:** Validates the full pipeline on a tiny subset before processing tens of thousands of APKs.

```bash
chmod +x run_base_model_1.sh
INSTALL_DEPS=0 PREPROCESS_LIMIT=100 EPOCHS=2 APK_ROOT=/path/to/your/dataset ./run_base_model_1.sh
```

If this finishes without errors, proceed to the full run.

---

## Step 6 — Full end-to-end run (one command)

**Why:** Runs preprocess → train → evaluate in order; resumes training if interrupted.

```bash
APK_ROOT=/path/to/your/dataset ./run_base_model_1.sh
```

Optional overrides:

| Variable | Effect |
|----------|--------|
| `INSTALL_DEPS=1` | Run `pip install` before the pipeline |
| `FRESH_TRAIN=1` | Ignore existing checkpoint; train from scratch |
| `EPOCHS=50` | Override epoch count from config |
| `SKIP_PREPROCESS=1` | Reuse existing `artifacts/processed/dex_header_features.pt` |
| `SKIP_TRAIN=1` | Preprocess only |
| `SKIP_EVAL=1` | Skip final metrics step |

On **Windows PowerShell**:

```powershell
$env:APK_ROOT = "D:\path\to\dataset"
.\run_base_model_1.ps1
```

---

## Manual step-by-step (alternative)

Use this if you want to run each stage separately instead of `run_base_model_1.sh`.

### 6a. Preprocess APKs

**Why:** Converts raw APKs into a single training-ready file (~30–50 MB for ~40k APKs).

```bash
export PYTHONPATH=.
export APK_ROOT=/path/to/your/dataset
# Uses thesis_venv when present (or set PYTHON=.../thesis_venv/bin/python):
./scripts/run_preprocess.sh --apk-root "$APK_ROOT"
```

**Output:** `artifacts/processed/dex_header_features.pt`, `artifacts/normalization.json`, `artifacts/failed_apks.log`

### 6b. Train the model

**Why:** Fits MLP(H); saves a checkpoint each epoch for resume.

```bash
./scripts/run_train.sh
```

**Output:** `artifacts/checkpoints/latest_checkpoint.pth`

Resume after interruption:

```bash
./scripts/run_train.sh --resume artifacts/checkpoints/latest_checkpoint.pth
```

Train from scratch:

```bash
./scripts/run_train.sh --fresh
```

### 6c. Evaluate

**Why:** Computes ACC, F1, and AUC on the validation split.

```bash
./scripts/run_evaluate.sh --split val --checkpoint artifacts/checkpoints/latest_checkpoint.pth
```

Metrics are printed to the terminal.

---

## Expected outputs (full run)

| Path | Description |
|------|-------------|
| `artifacts/processed/dex_header_features.pt` | All APK features + labels (main preprocess output) |
| `artifacts/normalization.json` | Min–max stats + multi-dex metadata |
| `artifacts/failed_apks.log` | APKs that failed extraction (should be small) |
| `artifacts/checkpoints/latest_checkpoint.pth` | Trained model + optimizer state |
| Terminal output | Validation ACC, F1, AUC at end of train/eval |

Default training: **50 epochs** (`config/default.yaml` → `training.epochs`).

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| `APK_ROOT does not exist` | Set `APK_ROOT` to the folder containing `2020/`, `2021/`, etc. |
| `No .apk files under ...` | Check path; APKs must end in `.apk` |
| `No label folder found` | APK must live under a folder named `benign` or `malware` |
| CUDA out of memory | Edit `config/default.yaml` → `training.device: cpu` or lower `data.batch_size` |
| Preprocess interrupted | Re-run preprocess; it processes all APKs again (no per-APK resume in BM1) |
| Train interrupted | Re-run `./run_base_model_1.sh` with `SKIP_PREPROCESS=1` or `train --resume` |

---

## Quick reference — minimum commands

```bash
cd /path/to/thesis_vigidroid
./scripts/setup_thesis_venv.sh
cd Dex_header_paper_implementation/only_base1_model
export PYTHONPATH=.
../../thesis_venv/bin/python scripts/verify_setup.py
APK_ROOT=/path/to/dataset ./run_base_model_1.sh
```

---

## CachyOS PC (SSH from Fedora) — path and command overrides

Use this section when the repo and dataset live on the **CachyOS** machine and you connect from your **Fedora laptop** over SSH. Run all commands **on the remote shell** (after `ssh user@cachyos-host`).

### Fixed paths on CachyOS

| What | Path |
|------|------|
| Project (git) root | `/mnt/Files/thesis_vigidroid/thesis_vigidroid` |
| Shared Python venv | `/mnt/Files/thesis_vigidroid/thesis_vigidroid/thesis_venv` |
| This pipeline folder | `/mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model` |
| APK dataset root | `/mnt/Files/thesis_full_dataset` |

Your tree (`2020`–`2023` / `benign` / `malware`) is valid for `label_mode: parent_folder`. Ignore Windows / PowerShell steps in this guide on CachyOS.

### Before the first run (on CachyOS)

```bash
ls /mnt/Files/thesis_full_dataset/2021/benign | head
ls /mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model/run_base_model_1.sh

# One-time shared venv (BM1 + Pattern A + Pattern B):
cd /mnt/Files/thesis_vigidroid/thesis_vigidroid
./scripts/setup_thesis_venv.sh

tmux new -s bm1
```

### Replace generic paths in this guide

| Guide placeholder | Use on CachyOS |
|-------------------|----------------|
| `/path/to/thesis_vigidroid/...` | `/mnt/Files/thesis_vigidroid/thesis_vigidroid/...` |
| `/path/to/your/dataset` or `/path/to/dataset` | `/mnt/Files/thesis_full_dataset` |
| `D:\path\to\dataset` (Windows) | not used on CachyOS |

**Step 0 — project folder:**

```bash
cd /mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
```

**Step 4 — Option A (recommended):**

```bash
export APK_ROOT=/mnt/Files/thesis_full_dataset
```

**Step 4 — Option B (symlink):**

```bash
mkdir -p data
ln -sf /mnt/Files/thesis_full_dataset data/apks
```

**Step 5 — smoke test:**

```bash
chmod +x run_base_model_1.sh
INSTALL_DEPS=0 PREPROCESS_LIMIT=100 EPOCHS=2 APK_ROOT=/mnt/Files/thesis_full_dataset ./run_base_model_1.sh
```

**Step 6 — full run:**

```bash
APK_ROOT=/mnt/Files/thesis_full_dataset ./run_base_model_1.sh
```

**Manual preprocess (6a):**

```bash
export PYTHONPATH=.
export APK_ROOT=/mnt/Files/thesis_full_dataset
python -m src.preprocessing.preprocess_apks --apk-root "$APK_ROOT"
```

Resume training after interrupt (on CachyOS, same paths):

```bash
cd /mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
export PYTHONPATH=.
SKIP_PREPROCESS=1 ./run_base_model_1.sh
```

(`run_base_model_1.sh` picks up `thesis_venv` automatically.)

### GPU on CachyOS

Default `training.device: cuda` with CPU fallback in code. Check on the remote:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Quick reference (copy-paste on CachyOS)

```bash
cd /mnt/Files/thesis_vigidroid/thesis_vigidroid
./scripts/setup_thesis_venv.sh   # once, shared by BM1 / Pattern A / B

cd Dex_header_paper_implementation/only_base1_model
export PYTHONPATH=.
export APK_ROOT=/mnt/Files/thesis_full_dataset
../../thesis_venv/bin/python scripts/verify_setup.py
APK_ROOT=/mnt/Files/thesis_full_dataset ./run_base_model_1.sh
```

### BM1-specific notes on CachyOS

- Preprocess writes a **single** `artifacts/processed/dex_header_features.pt`; a full corpus run can take hours and is **not** per-APK resumable like Pattern A/B shards. Use `PREPROCESS_LIMIT` smoke test first.
- Use the **same** `thesis_venv` for Pattern A and B; only `cd` and `artifacts/` differ per pipeline.
