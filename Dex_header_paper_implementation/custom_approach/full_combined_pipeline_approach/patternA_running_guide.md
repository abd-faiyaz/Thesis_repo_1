# Pattern A — Running Guide (Full Combined Pipeline)

Run **Pattern A**: Dex header **H** + manifest bag-of-words **I** → concatenate → **ASCNN** → binary malware classifier.

No prior knowledge of this repo is required. Follow the steps in order.

---

## What this pipeline does

1. **Scan** — finds all APKs, infers labels from folder names, creates a 90/10 stratified train/val split.
2. **Build lexicon** — learns manifest vocabulary from **train APKs only**.
3. **Fit header normalization** — min–max stats on **train Dex headers only**.
4. **Extract shards** — one compressed `.npz` per APK (header + BoW + label) for train and val.
5. **Train** — CombinedNet (single ASCNN tower on `concat(H, I)`).
6. **Evaluate** — ACC, F1, AUC on validation; writes JSON metrics.
7. **Package** — tarball of checkpoints + vocab/norm (shards excluded — copy separately if needed).

---

## Before you start

### Dataset layout

Point at the **root** of your APK tree. No label CSV required.

```
/path/to/dataset/
├── 2020/benign/*.apk
├── 2020/malware/*.apk
├── 2021/benign/*.apk
...
```

Labels: nearest parent folder named `benign` (0) or `malware` (1).

### Requirements

- **Linux** recommended
- **Python 3.10+**
- **~100–150 MB** artifact disk (preprocessed shards + checkpoints for ~40k APKs)
- **GPU recommended** for training — default config uses `device: cpu`; switch to `cuda` in `config/default.yaml` if you have a GPU
- Extra dependency: **pyaxmlparser** (manifest parsing)

---

## Step 0 — Go to the project folder

```bash
cd /path/to/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach
```

---

## Step 1 — Shared `thesis_venv` (BM1 + Pattern A + Pattern B)

**Why:** One virtual environment at the **git repo root** covers all pipelines. `run_pattern_a.sh` auto-detects `thesis_venv/bin/python`.

**One-time setup** (from repo root):

```bash
cd /path/to/thesis_vigidroid
./scripts/setup_thesis_venv.sh
```

This installs from `requirements-thesis-all.txt` (includes `pyaxmlparser` for Pattern A/B).

---

## Step 2 — Install dependencies (if not done in Step 1)

```bash
INSTALL_DEPS=1 ./run_pattern_a.sh
```

Or: `thesis_venv/bin/pip install -r /path/to/thesis_vigidroid/requirements-thesis-all.txt`

---

## Step 3 — Verify the environment

**Why:** Catches missing packages and wrong config before hours of preprocessing.

```bash
export PYTHONPATH=.
../../../thesis_venv/bin/python scripts/verify_setup.py
```

(`run_pattern_a.sh` runs verify automatically when `VERIFY_SETUP=1`, the default.)

---

## Step 4 — Point at your APK dataset

**Why:** Default `paths.apk_root` is `data/apks/`, which may not exist.

```bash
export APK_ROOT=/path/to/your/dataset
```

Or symlink: `ln -s /path/to/dataset data/apks`

Or edit `config/default.yaml` → `paths.apk_root`.

**Optional — use GPU for training:**

Edit `config/default.yaml` → `training.device: cuda`

---

## Step 5 — Smoke test (strongly recommended)

**Why:** Confirms scan → lexicon → shards → train on a small subset.

```bash
chmod +x run_pattern_a.sh
PREPROCESS_LIMIT=200 EPOCHS=2 APK_ROOT=/path/to/your/dataset ./run_pattern_a.sh
```

Check `artifacts/failed_apks.log` for unexpected failures.

---

## Step 6 — Full end-to-end run (one command)

**Why:** `run_pattern_a.sh` runs every stage in the correct order and appends logs to `artifacts/pipeline.log`.

```bash
APK_ROOT=/path/to/your/dataset ./run_pattern_a.sh
```

Useful environment variables:

| Variable | Effect |
|----------|--------|
| `INSTALL_DEPS=1` | `pip install -r requirements.txt` first |
| `FRESH_TRAIN=1` | Ignore checkpoints; train from scratch |
| `EPOCHS=80` | Override epoch count |
| `SKIP_PREPROCESS=1` | Train/eval only (requires existing manifests + shards) |
| `SKIP_TRAIN=1` | Preprocess only |
| `SKIP_EVAL=1` | Skip metrics JSON step |
| `SKIP_PACKAGE=1` | Skip tarball creation |
| `SKIP_DEX_STATS=1` | Skip multi-dex histogram script |

---

## Manual step-by-step (alternative)

Run these **in order** if you prefer separate commands over `run_pattern_a.sh`.

### 6a. Scan dataset and create train/val split

**Why:** Builds the master APK index and 90/10 stratified split files used by all later steps.

```bash
export PYTHONPATH=.
./scripts/run_preprocess.sh --apk-root "$APK_ROOT"
```

Or step-by-step with `../../../thesis_venv/bin/python -m src.preprocessing.scan_dataset ...` etc.

**Outputs:** `artifacts/dataset_index.csv`, `artifacts/splits/train.txt`, `artifacts/splits/val.txt`

### 6b. Build manifest lexicon (train only)

**Why:** Fixed vocabulary for bag-of-words; must not peek at validation APKs.

```bash
../../../thesis_venv/bin/python -m src.preprocessing.build_lexicon
```

**Output:** `artifacts/vocab.json`

### 6c. Fit Dex header normalization (train only)

**Why:** Min–max scaling stats computed only on training headers (no val leakage).

```bash
../../../thesis_venv/bin/python -m src.preprocessing.fit_header_norm
```

**Output:** `artifacts/normalization_header.json`

### 6d. Extract feature shards (train + val)

**Why:** Expensive one-time APK parsing; result is resumable per-APK `.npz` files.

```bash
../../../thesis_venv/bin/python -m src.preprocessing.extract_to_cache --split both
```

**Outputs:**
- `artifacts/processed/shards/train/*.npz`
- `artifacts/processed/shards/val/*.npz`
- `artifacts/processed/manifest_train.json`
- `artifacts/processed/manifest_val.json`

If interrupted, re-run the same command — already-written shards are skipped.

### 6e. Compute class balance (train)

**Why:** Used for optional class-weighted loss during training.

```bash
../../../thesis_venv/bin/python scripts/compute_class_balance.py
```

**Output:** `artifacts/class_balance.json`

### 6f. Train CombinedNet

**Why:** Learns weights; saves `best.pt` and `latest.pt` for resume.

```bash
./scripts/run_train.sh
```

Resume:

```bash
./scripts/run_train.sh --resume artifacts/checkpoints/latest.pt
```

### 6g. Evaluate on validation

**Why:** Final ACC / F1 / AUC report written to JSON.

```bash
./scripts/run_evaluate.sh --split val --checkpoint artifacts/checkpoints/best.pt
```

**Output:** `artifacts/checkpoints/metrics_val.json`

### 6h. Package portable artifacts (optional)

**Why:** Small tarball for copying checkpoints + vocab off the training machine. **Does not include shard `.npz` files.**

```bash
./scripts/package_artifacts.sh
```

**Output:** `artifacts/pattern_a_bundle.tar.gz`

---

## Expected outputs (full run)

| Path | Description |
|------|-------------|
| `artifacts/processed/shards/train/` | Per-APK feature files (train) |
| `artifacts/processed/shards/val/` | Per-APK feature files (val) |
| `artifacts/processed/manifest_train.json` | Shard index (train) |
| `artifacts/processed/manifest_val.json` | Shard index (val) |
| `artifacts/vocab.json` | Manifest BoW vocabulary |
| `artifacts/normalization_header.json` | Dex header min–max stats |
| `artifacts/class_balance.json` | Train class counts |
| `artifacts/checkpoints/best.pt` | Best validation-loss checkpoint |
| `artifacts/checkpoints/latest.pt` | Latest epoch (resume) |
| `artifacts/checkpoints/metrics_val.json` | ACC, F1, AUC |
| `artifacts/dex_stats.json` | Multi-dex file-count histogram |
| `artifacts/failed_apks.log` | Failed extractions |
| `artifacts/pipeline.log` | Full run log |
| `artifacts/pattern_a_bundle.tar.gz` | Portable bundle (no shards) |

Default training: **80 epochs** (`config/default.yaml` → `training.epochs`).

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| `APK_ROOT does not exist` | Export `APK_ROOT` before running |
| `Missing manifests` | Complete preprocess steps 6a–6d first |
| Manifest / pyaxmlparser errors | Some APKs may fail; check `failed_apks.log` |
| Extract slow or interrupted | Re-run `extract_to_cache --split both`; resumes automatically |
| Train interrupted | `SKIP_PREPROCESS=1 ./run_pattern_a.sh` or `train --resume` |
| Out of memory | Lower `training.batch_size` in config; use CPU if GPU RAM is tight |

---

## Quick reference — minimum commands

```bash
cd /path/to/thesis_vigidroid && ./scripts/setup_thesis_venv.sh
cd Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach
export PYTHONPATH=.
../../../thesis_venv/bin/python scripts/verify_setup.py
APK_ROOT=/path/to/dataset ./run_pattern_a.sh
```

---

## CachyOS PC (SSH from Fedora) — path and command overrides

Use this section when the repo and dataset live on the **CachyOS** machine and you connect from your **Fedora laptop** over SSH. All commands below run **on the remote shell** (after `ssh user@cachyos-host`).

### Fixed paths on CachyOS

| What | Path |
|------|------|
| Project (git) root | `/mnt/Files/thesis_vigidroid/thesis_vigidroid` |
| Shared Python venv | `/mnt/Files/thesis_vigidroid/thesis_vigidroid/thesis_venv` |
| This pipeline folder | `/mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach` |
| APK dataset root | `/mnt/Files/thesis_full_dataset` |

Your dataset layout (`2020`–`2023` → `benign/` / `malware/`) matches what the scanner expects. No layout changes needed.

### Before the first run (on CachyOS)

```bash
# Confirm mounts are visible on the remote
ls /mnt/Files/thesis_full_dataset/2020/benign | head
ls /mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/run_pattern_a.sh

# One-time shared venv (skip if already created for BM1 or Pattern B):
cd /mnt/Files/thesis_vigidroid/thesis_vigidroid && ./scripts/setup_thesis_venv.sh

tmux new -s pattern_a
```

### Replace generic paths in this guide

| Guide placeholder | Use on CachyOS |
|-------------------|----------------|
| `/path/to/thesis_vigidroid/...` | `/mnt/Files/thesis_vigidroid/thesis_vigidroid/...` |
| `/path/to/your/dataset` or `/path/to/dataset` | `/mnt/Files/thesis_full_dataset` |

**Step 0 — project folder:**

```bash
cd /mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach
```

**Step 4 — dataset (recommended: env var, no YAML edit):**

```bash
export APK_ROOT=/mnt/Files/thesis_full_dataset
```

Optional symlink instead of `APK_ROOT` every time:

```bash
mkdir -p data
ln -sf /mnt/Files/thesis_full_dataset data/apks
```

**Step 5 — smoke test:**

```bash
chmod +x run_pattern_a.sh
PREPROCESS_LIMIT=200 EPOCHS=2 APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_a.sh
```

**Step 6 — full run:**

```bash
APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_a.sh
```

**Manual steps (6a):** same `export APK_ROOT=/mnt/Files/thesis_full_dataset` before `scan_dataset` and other preprocess commands.

### GPU on CachyOS

`config/default.yaml` uses `training.device: cuda`. On the remote, check GPU visibility before a full run:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If no GPU or drivers are missing, set `training.device: cpu` in `config/default.yaml` or expect a slow CPU run.

### Quick reference (copy-paste on CachyOS)

```bash
cd /mnt/Files/thesis_vigidroid/thesis_vigidroid
./scripts/setup_thesis_venv.sh   # once

cd Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach
export PYTHONPATH=.
export APK_ROOT=/mnt/Files/thesis_full_dataset
../../../thesis_venv/bin/python scripts/verify_setup.py
APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_a.sh
```

### Optional — shared pipeline metadata (not required for Pattern A)

If you use `Shared_pipeline_Files/data/dataset_paths.yaml` for documentation or other tools, point `apk_root` and `project_root` at the CachyOS paths above. Pattern A’s `run_pattern_a.sh` still primarily honors `APK_ROOT` (or `data/apks` symlink).
