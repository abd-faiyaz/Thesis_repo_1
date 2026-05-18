# How to Run — Dual-Branch Merge (Pattern B)

This guide is for someone with **no prior context** on the project. It explains what the pipeline does, what you need on disk, and **every command** to install, run, verify, and showcase results.

**Project folder:** `dual_branch_merge_approach/`  
**What it does:** Trains a malware detector on Android APKs using two signals fused together:

1. **Dex header** (structure of `classes.dex`) → small neural network (MLP)  
2. **AndroidManifest** (permissions + intents) → convolutional network (ASCNN)  
3. **Fusion head** combines both → outputs malware probability  

Training does **not** re-read APKs every epoch. APKs are parsed once into cached **shard** files (`.npz`), then training loads those tensors only.

For deeper design notes, see [`implementation_plan.md`](implementation_plan.md) and [`patternB_specifics.md`](patternB_specifics.md).

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Linux** (recommended) | Tested on Fedora-like systems; macOS often works too |
| **Python 3.10+** | Check with `python3 --version` |
| **pip** | To install Python packages |
| **APK dataset** | `.apk` files in two subfolders: benign and malware |
| **Disk space** | Depends on dataset size; ~50k APKs need substantial space for shards |
| **GPU (optional)** | Config defaults to `cuda`; falls back to CPU if no GPU |

**Recommended:** Use a Python virtual environment so dependencies do not pollute system Python.

---

## 2. Dataset layout (required before preprocessing)

APKs must live under a single root directory with **folder names** used as labels (not a separate label CSV).

Example:

```text
data/apks/
  benign/
    app1.apk
    app2.apk
    ...
  malware/
    bad1.apk
    bad2.apk
    ...
```

Accepted folder names (configurable in `config/default.yaml`):

- **Benign:** `benign`, `goodware`, `clean`, `good`, `0`  
- **Malware:** `malware`, `malicious`, `virus`, `bad`, `1`  

You can point to any path using the environment variable `APK_ROOT` (see below).

---

## 3. One-time setup

Open a terminal and go to this project folder:

```bash
cd /path/to/dual_branch_merge_approach
```

Replace `/path/to/` with the real path on your machine.

### 3.1 (Recommended) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

After activation, your prompt usually shows `(.venv)`.

### 3.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs: `torch`, `numpy`, `scikit-learn`, `tqdm`, `PyYAML`, `pyaxmlparser`.

**What it does:** Installs libraries needed for parsing APKs, building tensors, training, and evaluation.

### 3.3 Tell Python where the project code lives

Every command below assumes you export `PYTHONPATH` to the project root (the folder that contains `src/`):

```bash
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
```

Run this **once per terminal session** before Python commands, or use the provided shell scripts (they set it automatically).

### 3.4 Verify the environment (smoke test, no APKs needed)

```bash
python scripts/verify_setup.py
```

**Expected:** Prints `All pip dependencies OK`, `Package layout and config loader OK`, and exits with code 0.

**What it does:** Checks imports, loads `config/default.yaml`, and creates empty artifact directories under `artifacts/`.

---

## 4. Configure paths (optional but important on a remote machine)

Default APK location in config is:

```text
data/apks/
```

(relative to `dual_branch_merge_approach/`).

To use another directory **without editing files**, set when running:

```bash
export APK_ROOT=/absolute/path/to/your/apks
```

Example:

```bash
export APK_ROOT=/data/thesis/apks
```

Other settings live in `config/default.yaml` (epochs, batch size, learning rate, etc.).

---

## 5. Full pipeline — one command (recommended for showcase)

This runs everything in order: preprocess → class balance → train → evaluate → package artifacts.

```bash
cd /path/to/dual_branch_merge_approach
source .venv/bin/activate          # if using venv
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
chmod +x run_pattern_b.sh          # only needed once

APK_ROOT=/path/to/your/apks ./run_pattern_b.sh
```

**What each stage does:**

| Stage | Purpose |
|-------|---------|
| Verify setup | Confirms environment (if enabled) |
| Preprocess | Parses APKs → saves `.npz` shards + manifests |
| Class balance | Counts train labels → writes `pos_weight` for imbalanced data |
| Train | Trains fusion model; saves `latest.pt` and `best.pt` |
| Evaluate | Prints ACC, F1, AUC on validation split |
| Package | Creates `artifacts/pattern_b_bundle.tar.gz` for easy copy-off |

**Log file:** Output is also appended to `artifacts/pipeline.log`.

---

## 6. Full pipeline — step by step (manual commands)

Use this if you want to explain each step in a demo or run stages separately.

All commands assume you are in `dual_branch_merge_approach/` with `PYTHONPATH` set and venv active.

### Step A — Scan APKs and create train/val split

```bash
python -m src.preprocessing.scan_dataset --apk-root /path/to/your/apks
```

**Outputs:**

- `artifacts/dataset_index.csv` — list of every APK with path, label, id  
- `artifacts/splits/train.txt` — 90% of APK ids (stratified)  
- `artifacts/splits/val.txt` — 10% of APK ids  

**Optional — limit APK count for a quick demo:**

```bash
python -m src.preprocessing.scan_dataset --apk-root /path/to/your/apks --limit 200
```

---

### Step B — Build manifest vocabulary (train split only)

```bash
python -m src.preprocessing.build_lexicon
```

**Outputs:** `artifacts/vocab.json` (top 4380 manifest tokens + UNK).

**What it does:** Reads manifests from **train** APKs only, counts permission/intent strings, builds fixed vocabulary for bag-of-words vectors.

---

### Step C — Fit Dex header normalization (train split only)

```bash
python -m src.preprocessing.fit_header_norm
```

**Outputs:** `artifacts/normalization_header.json` (min/max per header dimension).

**What it does:** Extracts raw 104-dim Dex header features from train APKs and computes corpus min–max scaling stats.

---

### Step D — Extract feature shards (train + val)

```bash
python -m src.preprocessing.extract_to_cache --split both
```

**Outputs (main ones):**

- `artifacts/processed/shards/train/<apk_id>.npz` — each file: `header`, `bow`, `label`  
- `artifacts/processed/shards/val/<apk_id>.npz`  
- `artifacts/processed/manifest_train.json` — index for DataLoader  
- `artifacts/processed/manifest_val.json`  
- `artifacts/processed/processed_ids.txt` — resume log  

**Optional — limit for demo:**

```bash
python -m src.preprocessing.extract_to_cache --split both --limit 100
```

**Resume:** Re-run the same command after interrupt; already-finished APKs are skipped.

**Failures:** Logged to `artifacts/failed_apks.log` (APK path + reason).

**Shortcut — run A+B+C+D via shell script:**

```bash
./scripts/run_preprocess.sh
```

With custom APK root, set `APK_ROOT` first or edit `config/default.yaml`.

---

### Step E — Compute class balance (for training loss)

```bash
python scripts/compute_class_balance.py
```

**Outputs:** `artifacts/class_balance.json` with `n_benign`, `n_malware`, `pos_weight`.

**What it does:** Counts labels in the train manifest so training can weight the malware class if counts are imbalanced (`training.auto_pos_weight: true` in config).

---

### Step F — Train the model

```bash
python -m src.training.train
```

**Or via script:**

```bash
./scripts/run_train.sh
```

**Outputs:**

- `artifacts/checkpoints/latest.pt` — updated every epoch (for resume)  
- `artifacts/checkpoints/best.pt` — lowest validation loss so far  

**Common options:**

```bash
# Train only 2 epochs (demo)
python -m src.training.train --epochs 2

# Ignore old checkpoint, start fresh
python -m src.training.train --fresh

# Resume after power loss / interrupt
python -m src.training.train --resume
```

**What it does:** Loads shards via DataLoader, runs forward + `BCEWithLogitsLoss` + backward + SGD. Uses GPU if available (`training.device: cuda` in config).

---

### Step G — Evaluate metrics (showcase numbers)

```bash
python -m src.training.evaluate --split val
```

**Or:**

```bash
./scripts/run_evaluate.sh --split val
```

**Optional — use a specific checkpoint:**

```bash
python -m src.training.evaluate --split val --checkpoint artifacts/checkpoints/best.pt
```

**Expected output (example):**

```text
Evaluation (val) — loss=0.6234 ACC=0.9100 F1=0.8800 AUC=0.9500
```

Metrics: **Accuracy**, **F1**, **ROC-AUC** (paper-style).

---

### Step H — Package artifacts to copy off the machine

```bash
./scripts/package_artifacts.sh
```

**Outputs:** `artifacts/pattern_b_bundle.tar.gz` containing checkpoints, vocab, normalization stats, manifests, class balance, etc.

**Note:** Shard `.npz` files are **not** included (too large). Copy `artifacts/processed/shards/` separately if needed.

---

## 7. Quick demo without a large dataset

If you only have a **small** APK set (or want a fast end-to-end demo):

```bash
cd /path/to/dual_branch_merge_approach
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

APK_ROOT=/path/to/small/apk/set \
  PREPROCESS_LIMIT=100 \
  EXTRACT_LIMIT=100 \
  EPOCHS=3 \
  ./run_pattern_b.sh
```

| Variable | Meaning |
|----------|---------|
| `PREPROCESS_LIMIT=100` | Only index/first 100 APKs at scan |
| `EXTRACT_LIMIT=100` | Cap shard extraction per split |
| `EPOCHS=3` | Short training for demo |

---

## 8. Verify installation without any APKs (developer checks)

These use **synthetic data** and confirm code works; they do **not** prove accuracy on real malware.

```bash
export PYTHONPATH="$(pwd)"

# Phase 1 — environment
python scripts/verify_setup.py

# Phase 3 — DataLoader (synthetic shards if no manifests)
python scripts/verify_dataloader.py

# Phase 4 — model forward pass (random tensors)
python scripts/verify_model.py

# Unit tests
python -m unittest discover -s tests -v
```

---

## 9. What to show in a showcase / presentation

After a successful full run, point to these artifacts:

| Artifact | What to say |
|----------|-------------|
| `artifacts/dataset_index.csv` | “We indexed N APKs with labels from folder names.” |
| `artifacts/processed/shards/train/*.npz` | “Precomputed features per APK — no re-parsing during training.” |
| `artifacts/vocab.json` | “Fixed manifest vocabulary from train set.” |
| `artifacts/checkpoints/best.pt` | “Best model weights by validation loss.” |
| `artifacts/pipeline.log` | “Full run log.” |
| Terminal eval line | “ACC / F1 / AUC on held-out val split.” |

**Optional live demo commands:**

```bash
# Show one training batch shape
python scripts/verify_dataloader.py

# Show model I/O shapes
python scripts/verify_model.py

# Show final metrics again
./scripts/run_evaluate.sh --split val
```

---

## 10. Useful environment variables (`run_pattern_b.sh`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `APK_ROOT` | `data/apks` | Where benign/malware APK folders live |
| `PYTHON` | `python3` | Python executable |
| `CONFIG` | `config/default.yaml` | Config file path |
| `EPOCHS` | from YAML (80) | Override training epochs |
| `INSTALL_DEPS` | `0` | Set `1` to run `pip install -r requirements.txt` |
| `VERIFY_SETUP` | `1` | Set `0` to skip verify step |
| `SKIP_PREPROCESS` | `0` | Set `1` if shards already exist |
| `SKIP_TRAIN` | `0` | Set `1` to skip training |
| `SKIP_EVAL` | `0` | Set `1` to skip evaluation |
| `SKIP_PACKAGE` | `0` | Set `1` to skip tarball |
| `FRESH_TRAIN` | `0` | Set `1` to ignore checkpoint and retrain from scratch |
| `PREPROCESS_LIMIT` | (none) | Max APKs at scan |
| `EXTRACT_LIMIT` | (none) | Max APKs per split during extraction |

**Examples:**

```bash
# Only evaluate an existing model
SKIP_PREPROCESS=1 SKIP_TRAIN=1 ./run_pattern_b.sh

# Retrain without re-preprocessing
SKIP_PREPROCESS=1 FRESH_TRAIN=1 EPOCHS=50 ./run_pattern_b.sh

# Install deps and run on remote path
INSTALL_DEPS=1 APK_ROOT=/data/apks ./run_pattern_b.sh
```

---

## 11. Project layout (reference)

```text
dual_branch_merge_approach/
  howToRun.md                 ← this file
  run_pattern_b.sh            ← main end-to-end entry
  config/default.yaml         ← paths and hyperparameters
  requirements.txt
  scripts/
    verify_setup.py
    run_preprocess.sh
    run_train.sh
    run_evaluate.sh
    compute_class_balance.py
    package_artifacts.sh
    verify_dataloader.py
    verify_model.py
  src/                        ← Python package
    preprocessing/              ← APK → shards
    data/                       ← DataLoaders
    models/                     ← DualBranchNet
    training/                   ← train, evaluate, checkpoints
  data/apks/                    ← put your APKs here (default)
  artifacts/                    ← generated outputs (gitignored)
  tests/                        ← unit tests
```

---

## 12. Troubleshooting

| Problem | What to check |
|---------|----------------|
| `APK_ROOT does not exist` | Create `benign/` and `malware/` under `APK_ROOT`, or set `APK_ROOT` correctly |
| `No .apk files` | APKs must end with `.apk` somewhere under `APK_ROOT` |
| `Manifests not found` | Run preprocessing before training (`./scripts/run_preprocess.sh`) |
| `ModuleNotFoundError: src` | Run `export PYTHONPATH="$(pwd)"` from project root |
| CUDA errors | Set `device: cpu` under `training:` in `config/default.yaml`, or fix GPU drivers |
| Very slow preprocessing | Normal for large datasets; use `EXTRACT_LIMIT` for tests; shards enable fast re-training |
| Many lines in `failed_apks.log` | Some APKs lack valid Dex/manifest; they are skipped, not silently labeled |
| Resume not working | Re-run `extract_to_cache` or `train` — checkpoints/shards are designed to be idempotent |

---

## 13. Minimal command cheat sheet

```bash
# Setup (once per machine)
cd dual_branch_merge_approach
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"

# Sanity check
python scripts/verify_setup.py

# Full run (real APKs required)
APK_ROOT=/path/to/apks ./run_pattern_b.sh

# Fast demo
APK_ROOT=/path/to/apks PREPROCESS_LIMIT=200 EXTRACT_LIMIT=200 EPOCHS=2 ./run_pattern_b.sh

# Evaluate only
SKIP_PREPROCESS=1 SKIP_TRAIN=1 ./run_pattern_b.sh
```

---

## 14. Related documentation

- [`implementation_plan.md`](implementation_plan.md) — architecture and design choices  
- [`patternB_specifics.md`](patternB_specifics.md) — per-phase implementation log  
