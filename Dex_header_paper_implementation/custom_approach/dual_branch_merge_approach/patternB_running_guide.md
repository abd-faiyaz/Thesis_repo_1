# Pattern B — Running Guide (Dual-Branch Merge)

Run **Pattern B**: separate **MLP(H)** on Dex header and **ASCNN(I)** on manifest BoW → concatenate embeddings → fusion head → binary classifier.

No prior knowledge of this repo is required. Follow the steps in order.

---

## What this pipeline does

1. **Scan** — finds APKs, labels from `benign/` / `malware/` folders, 90/10 stratified train/val split.
2. **Build lexicon** — manifest vocabulary from **train only**.
3. **Fit header normalization** — min–max on **train Dex headers only**.
4. **Extract shards** — one `.npz` per APK (header + BoW + label); resumable.
5. **Train** — DualBranchNet (two branches + late fusion).
6. **Evaluate** — validation ACC, F1, AUC → JSON.
7. **Package** — tarball of checkpoints + vocab/norm (shards copied separately if needed).

Preprocessing is the same structure as Pattern A; only the **model and training code** differ.

---

## Before you start

### Dataset layout

Same as Pattern A — root folder with nested year/class structure:

```
/path/to/dataset/
├── 2020/benign/*.apk
├── 2020/malware/*.apk
...
```

No CSV labels needed.

### Requirements

- **Linux** recommended
- **Python 3.10+**
- **~100–170 MB** artifact disk (~40k APKs)
- **GPU recommended** — default config uses `training.device: cuda`
- **pyaxmlparser** for manifest parsing

---

## Step 0 — Go to the project folder

```bash
cd /path/to/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach
```

---

## Step 1 — Shared `thesis_venv` (BM1 + Pattern A + Pattern B)

**Why:** One virtual environment at the **git repo root** covers all pipelines. `run_pattern_b.sh` auto-detects `thesis_venv/bin/python`.

**One-time setup** (from repo root):

```bash
cd /path/to/thesis_vigidroid
./scripts/setup_thesis_venv.sh
```

---

## Step 2 — Install dependencies (if not done in Step 1)

```bash
INSTALL_DEPS=1 ./run_pattern_b.sh
```

---

## Step 3 — Verify the environment

**Why:** Validates imports, multi-dex config, and artifact directories before a long run.

```bash
export PYTHONPATH=.
../../../thesis_venv/bin/python scripts/verify_setup.py
```

---

## Step 4 — Point at your APK dataset

**Why:** The pipeline must know where your downloaded APKs live.

```bash
export APK_ROOT=/path/to/your/dataset
```

Alternatives: symlink `data/apks` → your dataset, or edit `config/default.yaml` → `paths.apk_root`.

---

## Step 5 — Smoke test (strongly recommended)

**Why:** Runs the full chain on ~200 APKs and 2 epochs before committing to the full corpus.

```bash
chmod +x run_pattern_b.sh
PREPROCESS_LIMIT=200 EPOCHS=2 APK_ROOT=/path/to/your/dataset ./run_pattern_b.sh
```

---

## Step 6 — Full end-to-end run (one command)

**Why:** Executes all stages in order; logs to `artifacts/pipeline.log`; supports resume on train and shard extraction.

```bash
APK_ROOT=/path/to/your/dataset ./run_pattern_b.sh
```

Environment variables:

| Variable | Effect |
|----------|--------|
| `INSTALL_DEPS=1` | Install requirements first |
| `FRESH_TRAIN=1` | Train from scratch |
| `EPOCHS=80` | Override epoch count |
| `SKIP_PREPROCESS=1` | Skip to training (needs existing shards) |
| `SKIP_TRAIN=1` | Preprocess only |
| `SKIP_EVAL=1` | Skip final evaluation |
| `SKIP_PACKAGE=1` | Skip tarball |
| `EXTRACT_LIMIT=N` | Cap shard extraction count (debug) |

---

## Manual step-by-step (alternative)

### 6a. Scan dataset and split

**Why:** Creates index + stratified 90/10 train/val ID lists.

```bash
export PYTHONPATH=.
./scripts/run_preprocess.sh --apk-root "$APK_ROOT"
```

**Outputs:** `artifacts/dataset_index.csv`, `artifacts/splits/train.txt`, `artifacts/splits/val.txt`

### 6b. Build lexicon (train only)

**Why:** BoW vocabulary must be fit on training APKs only.

```bash
../../../thesis_venv/bin/python -m src.preprocessing.build_lexicon
```

**Output:** `artifacts/vocab.json`

### 6c. Fit header normalization (train only)

**Why:** Dex header scaling without validation leakage.

```bash
../../../thesis_venv/bin/python -m src.preprocessing.fit_header_norm
```

**Output:** `artifacts/normalization_header.json`

### 6d. Extract shards (train + val)

**Why:** One-time feature cache; training never touches raw APKs again.

```bash
../../../thesis_venv/bin/python -m src.preprocessing.extract_to_cache --split both
```

**Outputs:** `artifacts/processed/shards/{train,val}/*.npz`, `manifest_train.json`, `manifest_val.json`

Re-run safely after interruption — existing shards are skipped.

### 6e. Class balance

**Why:** Feeds optional `pos_weight` for imbalanced classes.

```bash
../../../thesis_venv/bin/python scripts/compute_class_balance.py
```

**Output:** `artifacts/class_balance.json`

### 6f. Train DualBranchNet

**Why:** Fits both branches + fusion; checkpoints for resume.

```bash
./scripts/run_train.sh
```

Resume after power loss:

```bash
./scripts/run_train.sh --resume artifacts/checkpoints/latest.pt
```

Fresh start:

```bash
./scripts/run_train.sh --fresh
```

### 6g. Evaluate

**Why:** Produces the final validation metrics file.

```bash
./scripts/run_evaluate.sh --split val --checkpoint artifacts/checkpoints/best.pt
```

**Output:** `artifacts/checkpoints/metrics_val.json`

### 6h. Package artifacts (optional)

**Why:** Copy checkpoints + vocab/norm off the training machine without tarballing ~100 MB of shards.

```bash
./scripts/package_artifacts.sh
```

**Output:** `artifacts/pattern_b_bundle.tar.gz`

---

## Expected outputs (full run)

| Path | Description |
|------|-------------|
| `artifacts/processed/shards/train/` | Train feature shards |
| `artifacts/processed/shards/val/` | Validation feature shards |
| `artifacts/processed/manifest_train.json` | Train shard manifest |
| `artifacts/processed/manifest_val.json` | Val shard manifest |
| `artifacts/vocab.json` | Manifest vocabulary |
| `artifacts/normalization_header.json` | Header min–max stats |
| `artifacts/class_balance.json` | Train class counts |
| `artifacts/checkpoints/best.pt` | Best val-loss model |
| `artifacts/checkpoints/latest.pt` | Resume checkpoint |
| `artifacts/checkpoints/metrics_val.json` | ACC, F1, AUC |
| `artifacts/failed_apks.log` | Extraction failures |
| `artifacts/pipeline.log` | Full run log |
| `artifacts/pattern_b_bundle.tar.gz` | Portable bundle (no shards) |

Default training: **80 epochs** (`config/default.yaml` → `training.epochs`).

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| `APK_ROOT does not exist` | Set `APK_ROOT` to dataset root |
| `Missing manifests` | Run preprocess steps 6a–6d |
| Many entries in `failed_apks.log` | Normal for corrupt/manifest-less APKs; investigate if rate is high |
| CUDA OOM | Lower `training.batch_size` or set `training.device: cpu` |
| Extract interrupted | Re-run `extract_to_cache --split both` |
| Train interrupted | `SKIP_PREPROCESS=1 ./run_pattern_b.sh` or `train --resume` |

---

## Quick reference — minimum commands

```bash
cd /path/to/thesis_vigidroid && ./scripts/setup_thesis_venv.sh
cd Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach
export PYTHONPATH=.
../../../thesis_venv/bin/python scripts/verify_setup.py
APK_ROOT=/path/to/dataset ./run_pattern_b.sh
```

---

## CachyOS PC (SSH from Fedora) — path and command overrides

Use this section when the repo and dataset live on the **CachyOS** machine and you connect from your **Fedora laptop** over SSH. Run all commands **on the remote shell** (after `ssh user@cachyos-host`).

### Fixed paths on CachyOS

| What | Path |
|------|------|
| Project (git) root | `/mnt/Files/thesis_vigidroid/thesis_vigidroid` |
| Shared Python venv | `/mnt/Files/thesis_vigidroid/thesis_vigidroid/thesis_venv` |
| This pipeline folder | `/mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach` |
| APK dataset root | `/mnt/Files/thesis_full_dataset` |

Dataset tree (`2020`–`2023` / `benign` / `malware`) matches Pattern B expectations.

### Before the first run (on CachyOS)

```bash
ls /mnt/Files/thesis_full_dataset/2020/malware | head
ls /mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/run_pattern_b.sh

cd /mnt/Files/thesis_vigidroid/thesis_vigidroid && ./scripts/setup_thesis_venv.sh

tmux new -s pattern_b
```

### Replace generic paths in this guide

| Guide placeholder | Use on CachyOS |
|-------------------|----------------|
| `/path/to/thesis_vigidroid/...` | `/mnt/Files/thesis_vigidroid/thesis_vigidroid/...` |
| `/path/to/your/dataset` or `/path/to/dataset` | `/mnt/Files/thesis_full_dataset` |

**Step 0 — project folder:**

```bash
cd /mnt/Files/thesis_vigidroid/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach
```

**Step 4 — dataset:**

```bash
export APK_ROOT=/mnt/Files/thesis_full_dataset
```

Or symlink:

```bash
mkdir -p data
ln -sf /mnt/Files/thesis_full_dataset data/apks
```

**Step 5 — smoke test:**

```bash
chmod +x run_pattern_b.sh
PREPROCESS_LIMIT=200 EPOCHS=2 APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_b.sh
```

**Step 6 — full run:**

```bash
APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_b.sh
```

**Manual steps (6a–6d):** export `APK_ROOT` as above; run from this pipeline directory with `export PYTHONPATH=.`.

### GPU on CachyOS

Default config uses `training.device: cuda`. Verify on the remote:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

On CUDA OOM, lower `training.batch_size` in `config/default.yaml` or switch to `cpu`.

### Quick reference (copy-paste on CachyOS)

```bash
cd /mnt/Files/thesis_vigidroid/thesis_vigidroid
./scripts/setup_thesis_venv.sh   # once

cd Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach
export PYTHONPATH=.
export APK_ROOT=/mnt/Files/thesis_full_dataset
../../../thesis_venv/bin/python scripts/verify_setup.py
APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_b.sh
```

### Note on Pattern A vs B

Preprocessing layout is the same as Pattern A. Use the **same** `thesis_venv`; **artifacts** are per-folder — do not reuse Pattern A’s `artifacts/` unless you intentionally symlink or copy them.
