# Dual-Branch Merge (Pattern B) — Implementation Log

Per-phase notes: what was done, where, and why. Aligned with [`implementation_plan.md`](implementation_plan.md).

---

## Phase 1: Workspace & Environment Setup

**Status:** Complete  
**Date:** 2026-05-15

### Sequential steps

1. Pin dependencies (torch, numpy, sklearn, tqdm, PyYAML, pyaxmlparser) → `requirements.txt`
2. Define paths, preprocessing defaults, model dims, training hyperparams → `config/default.yaml`
3. Load YAML and resolve artifact paths → `src/config.py`
4. Define Dex/BoW constants (104-d header, lexicon size) → `src/constants.py`
5. Smoke-test imports and create artifact directories → `scripts/verify_setup.py`
6. Exclude large data and artifacts from git → `.gitignore`

### Goal

Establish a modular PyTorch pipeline layout and pinned dependencies before APK/Dex/manifest preprocessing (Phase 2). Separates concerns so GPU training never touches raw APKs at load time.

### Dependencies

| Dependency | Location | Reason |
|------------|----------|--------|
| `torch` | `requirements.txt` | Model, tensors, optional GPU training (Phases 4–5) |
| `numpy` | `requirements.txt` | Feature arrays during preprocessing |
| `scikit-learn` | `requirements.txt` | Optional val AUC / metrics later |
| `tqdm` | `requirements.txt` | Preprocessing and training progress bars |
| `PyYAML` | `requirements.txt` | `config/default.yaml` loader |
| `pyaxmlparser` | `requirements.txt` | Binary `AndroidManifest.xml` → permissions + intents (Phase 2) |
| `zipfile` | Python stdlib | In-memory APK unpack (no pip package) |

Install (from `dual_branch_merge_approach/`):

```bash
pip install -r requirements.txt
python scripts/verify_setup.py
```

### Files created / purpose

| File | Purpose |
|------|---------|
| `config/default.yaml` | Paths, preprocessing, model dims, training hyperparams |
| `src/config.py` | `load_config()`, `PathsConfig`, `ensure_artifact_dirs()` |
| `src/constants.py` | `DEX_MAGIC`, `DEX_HEADER_FEATURE_DIM=104`, `DEFAULT_LEXICON_SIZE=4380` |
| `src/__init__.py` | Package marker |
| `scripts/verify_setup.py` | Confirms imports and config load |
| `.gitignore` | Excludes `data/`, artifacts, checkpoints |

### Verification

```bash
python scripts/verify_setup.py
```

### Not in Phase 1 (deferred)

- Feature extraction, dataset, models, training — Phases 2–5.

---

## Phase 2: APK Preprocessing & Feature Extraction

**Status:** Complete  
**Date:** 2026-05-15

### Sequential steps

1. Walk `apk_root`, infer label from `benign/` / `malware/` folder names, write index + 90/10 stratified split → `src/preprocessing/scan_dataset.py`, `src/preprocessing/labels.py`, `src/preprocessing/common.py`
2. Collect manifest tokens on **train** APKs; build top-4380 lexicon (freq ≥ 2) → `src/preprocessing/build_lexicon.py`, `src/features/manifest_bow.py`
3. Extract raw Dex header bytes on **train** APKs; fit min–max stats → `src/preprocessing/fit_header_norm.py`, `src/features/dex_header.py`, `src/features/normalization.py`
4. For each train/val APK: read `classes.dex` + manifest → normalize `H`, multi-hot `I` → write one `.npz` shard → `src/preprocessing/extract_to_cache.py`, `src/features/apk_extract.py`
5. Log failures; skip bad APKs → `artifacts/failed_apks.log` via `src/preprocessing/common.py`
6. Resume interrupted extraction via `processed_ids.txt` → `src/preprocessing/extract_to_cache.py`
7. Write shard index JSON for train and val → `artifacts/processed/manifest_{train,val}.json` via `src/preprocessing/common.py`
8. Run full pipeline from shell → `scripts/run_preprocess.sh`

### Goal

Precompute **Dex header** (`H`, 104-d) and **manifest BoW** (`I`, 4381-d) per APK as resumable `.npz` shards so Phase 3 never parses APKs during training.

### Manifest parser note

Planned **`axmlparserpy`** is Python 2–only. Implemented with **`pyaxmlparser`** (`config/default.yaml` → `preprocessing.manifest_parser`).

### Output artifacts

| Path | Contents |
|------|----------|
| `artifacts/dataset_index.csv` | `apk_id`, `path`, `label` |
| `artifacts/splits/train.txt`, `val.txt` | One `apk_id` per line |
| `artifacts/vocab.json` | `token_to_index`, `unk_index` |
| `artifacts/normalization_header.json` | `mins`, `maxs` |
| `artifacts/processed/shards/{train,val}/<apk_id>.npz` | `header`, `bow`, `label` |
| `artifacts/processed/manifest_{train,val}.json` | Shard index |
| `artifacts/processed/processed_ids.txt` | Resume log |
| `artifacts/failed_apks.log` | Skipped APKs |

### How to run

```bash
./scripts/run_preprocess.sh
```

### Tests

```bash
PYTHONPATH=. python -m unittest tests.test_dex_header tests.test_manifest_bow -v
```

### Deferred to Phase 3

- Loading shards in PyTorch — implemented in Phase 3.

---

## Phase 3: PyTorch Dataset & DataLoader

**Status:** Complete  
**Date:** 2026-05-15

### Sequential steps

1. Parse `manifest_{train,val}.json` into shard entry list → `src/data/store.py`
2. Load one `.npz` shard (`header`, `bow`, `label`) on demand → `src/data/store.py` → `load_shard_npz()`
3. Wrap entries in `Dataset`; `__getitem__` returns `(header[104], bow[4381], label)` → `src/data/dataset.py` → `DualBranchDataset`
4. Build train `DataLoader` (shuffle) and val `DataLoader` (no shuffle), batch size 16 → `src/data/dataloaders.py`
5. Wire manifests from config (`manifest_train`, `manifest_val`) — **no random re-split** (Phase 2 split is used) → `build_dataloaders_from_config()` in `src/data/dataloaders.py`
6. Add `data` section to config (batch_size, num_workers, pin_memory) → `config/default.yaml`, `src/config.py`
7. Smoke-test one train/val batch → `scripts/verify_dataloader.py`

### Goal

Load Phase 2 shards only during training — no APK/ZIP/Dex/manifest I/O in the training loop. Provide paper batch size (16), shuffled train batches, sequential val batches.

### Components

| File | Role |
|------|------|
| `src/data/store.py` | `load_shard_manifest()`, `load_shard_npz()`, `ShardEntry` |
| `src/data/dataset.py` | `DualBranchDataset` — `(header, bow, label)` per sample |
| `src/data/dataloaders.py` | `build_train_loader`, `build_eval_loader`, `build_dataloaders_from_config` |
| `src/data/__init__.py` | Public exports |

### DualBranchDataset behavior

1. Input: entries from `manifest_train.json` or `manifest_val.json`.
2. `__getitem__(i)` loads `.npz` → `header` `[104]`, `bow` `[4381]`, `label` scalar `0.0` / `1.0`.
3. `from_manifest(path)` — loads manifest JSON written by Phase 2.

### DataLoader behavior

| Loader | `shuffle` | `batch_size` | Source manifest |
|--------|-----------|--------------|-----------------|
| Train | **True** | 16 (`data.batch_size`) | `artifacts/processed/manifest_train.json` |
| Val | **False** | 16 | `artifacts/processed/manifest_val.json` |

Settings from `config/default.yaml`: `data.num_workers`, `data.pin_memory`.

### Usage (after Phase 2 preprocessing)

```python
from src.config import load_config
from src.data import build_dataloaders_from_config

cfg = load_config()
train_loader, val_loader, header_dim, bow_dim = build_dataloaders_from_config(cfg)
# header_dim == 104, bow_dim == 4381
```

### Verification

```bash
PYTHONPATH=. python -m unittest tests.test_dataset -v
PYTHONPATH=. python scripts/verify_dataloader.py
```

`verify_dataloader.py` uses real manifests if present; otherwise synthetic shards.

### Deferred to Phase 4

- Model forward pass consuming `(header, bow)` batches.

---

## Phase 4: Model Architecture (Dual-Branch Forward Pass)

**Status:** Complete  
**Date:** 2026-05-15

### Sequential steps

1. Implement ASU: Conv1d + per-sample gate + soft threshold + BN + ReLU → `src/models/adaptive_shrinkage_unit.py`
2. Implement header branch: `104 → 128 → 128` embedding `e_h` (no sigmoid) → `src/models/mlp_header.py` → `MLPHeaderBranch`
3. Implement manifest ASCNN: BoW `(B,4381)` → `(B,1,L)` → 3× ASU (64→128→128) → `AdaptiveAvgPool1d` → `e_i` → `src/models/ascnn_manifest.py`
4. Implement fusion: `concat(e_h,e_i)` (256-d) → Linear → BN → ReLU → logit → `src/models/fusion_head.py`
5. Wire `DualBranchNet.forward(header, bow)` and `predict_proba()` → `src/models/dual_branch_net.py`
6. Export builders from config → `src/models/__init__.py`
7. Verify with **dummy random batch** (no dataset required) or real manifests if present → `scripts/verify_model.py`

### Goal

Pattern B graph: `MLP(H)` + `ASCNN(I)` + late fusion; forward-only in this phase (training/backprop in Phase 5).

### Architecture

| Component | Input | Output | File |
|-----------|-------|--------|------|
| `MLPHeaderBranch` | `(B, 104)` | `e_h` `(B, 128)` | `mlp_header.py` |
| `ASCNNManifest` | `(B, 4381)` | `e_i` `(B, 128)` | `ascnn_manifest.py` |
| `FusionHead` | `(B, 256)` | logit `(B, 1)` | `fusion_head.py` |
| `DualBranchNet` | header + bow | logit / sigmoid prob | `dual_branch_net.py` |

**ASU stack (paper Fig. 7):** kernel 3, strides 2, 2, 1; channels 64 → 128 → 128.

**No branch sigmoid:** fusion owns the single malware logit (`BCEWithLogitsLoss` in Phase 5).

### API

```python
from src.config import load_config
from src.models import build_dual_branch_net_from_config

model = build_dual_branch_net_from_config(load_config())
logits = model(header, bow)           # (B, 1), raw logit
probs = model.predict_proba(header, bow)  # (B, 1), sigmoid
```

### Verification

```bash
PYTHONPATH=. python -m unittest tests.test_dual_branch_net -v
PYTHONPATH=. python scripts/verify_model.py
```

Uses **random tensors** when Phase 2 manifests are missing (~290k parameters on default config).

---

## Phase 5: Training Loop & Checkpoints

**Status:** Complete  
**Date:** 2026-05-15

### Sequential steps

1. Build `BCEWithLogitsLoss` on fusion logit; optional `pos_weight` / `benign_to_malware_ratio` → `src/training/losses.py`
2. SGD + StepLR from config; resolve CPU/CUDA → `src/training/setup.py`
3. Train loop: forward → loss → `backward()` → `optimizer.step()` with tqdm → `src/training/loops.py` → `train_one_epoch()`
4. Val loop: no grad, val loss tqdm → `src/training/loops.py` → `validate_one_epoch()`
5. Save/load checkpoint (model, optimizer, scheduler, epoch, RNG, best val) → `src/training/checkpoint.py`
6. Orchestrate epochs; write `latest.pt` each epoch, `best.pt` on improved val loss → `src/training/train.py`
7. CLI: `--epochs`, `--fresh`, `--resume [path]` → `src/training/train.py`
8. Shell entry → `scripts/run_train.sh`
9. Synthetic-shard smoke test + resume → `tests/test_training.py`

### Goal

Train `DualBranchNet` on Phase 2 shards with paper-style SGD (lr 0.005, momentum 0.9, StepLR ×0.5), tqdm, and resume after power loss.

### Training flow (`src/training/train.py`)

1. Require `manifest_train.json` + `manifest_val.json` (Phase 2).
2. Build train/val loaders (Phase 3) and model (Phase 4).
3. `criterion = BCEWithLogitsLoss` on logits; labels reshaped to `(B, 1)`.
4. Resume from `artifacts/checkpoints/latest.pt` unless `--fresh`.
5. Each epoch: train → validate → `scheduler.step()` → save checkpoints.

### Hyperparameters (`config/default.yaml`)

| Setting | Value | Module |
|---------|-------|--------|
| Loss | `BCEWithLogitsLoss` | `losses.py` |
| Optimizer | SGD, lr `0.005`, momentum `0.9` | `setup.py` |
| LR decay | StepLR, γ=`0.5`, step_size=`10` | `setup.py` |
| Batch size | `16` | `data.batch_size` |
| Epochs | `80` (override `--epochs`) | `training.epochs` |
| Device | `cuda` if available | `setup.resolve_device` |

### Checkpoint format (`latest.pt` / `best.pt`)

```python
{
  "next_epoch": int,
  "global_step": int,
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "scheduler_state_dict": ...,
  "train_loss": float,
  "val_loss": float,
  "best_val_loss": float,
  "rng_state": { python, numpy, torch, torch_cuda? },
}
```

**Resume:** `python -m src.training.train --resume` or auto-load `latest.pt`. **`--fresh`** ignores existing checkpoint.

### How to run

```bash
# After Phase 2 preprocessing
./scripts/run_train.sh

# Short CPU smoke test
PYTHONPATH=. python -m src.training.train --epochs 2 --fresh

# Resume
PYTHONPATH=. python -m src.training.train --resume
```

### Verification

```bash
PYTHONPATH=. python -m unittest tests.test_training -v
```

Uses **synthetic shards** in a temp dir (no real APK dataset). Train loss decreased in smoke run; resume continues from `next_epoch`.

---

## Phase 6: Full Remote Run & Evaluation

**Status:** Complete  
**Date:** 2026-05-15

### Sequential steps

1. Set `paths.apk_root` on remote (`benign/`, `malware/`) → `config/default.yaml` or `APK_ROOT=...` env
2. Run end-to-end orchestrator → `run_pattern_b.sh` (preprocess → balance → train → eval → package)
3. Scan/index APKs (optional `--limit` for smoke) → `src/preprocessing/scan_dataset.py`
4. Build lexicon, header norm, extract shards → `scripts/run_preprocess.sh` steps inside orchestrator
5. Count train labels; write `pos_weight` → `scripts/compute_class_balance.py` → `artifacts/class_balance.json`
6. Train with auto `pos_weight` when enabled → `src/training/losses.py` + `src/training/train.py`
7. Evaluate `best.pt` (ACC, F1, AUC) → `src/training/evaluate.py`, `scripts/run_evaluate.sh`
8. Tar portable artifacts (checkpoints, vocab, norms, manifests — not shards) → `scripts/package_artifacts.sh`
9. Copy `pattern_b_bundle.tar.gz` + `processed/shards/` off remote → thesis / deployment

### Goal

Single entry point for remote ~50k APK run: preprocess, train, metrics, and artifact export. Runnable locally as smoke test without real APKs via prior phase unit tests.

### Orchestrator (`run_pattern_b.sh`)

| Env variable | Purpose |
|--------------|---------|
| `APK_ROOT` | APK tree (default `data/apks`) |
| `EPOCHS` | Override `training.epochs` |
| `SKIP_PREPROCESS` | Use existing shards |
| `SKIP_TRAIN` / `SKIP_EVAL` / `SKIP_PACKAGE` | Skip stages |
| `FRESH_TRAIN` | Ignore checkpoint |
| `PREPROCESS_LIMIT` | Cap APKs at scan |
| `EXTRACT_LIMIT` | Cap shard extraction per split |
| `INSTALL_DEPS` | `pip install -r requirements.txt` |

Logs append to `artifacts/pipeline.log`.

### Class balance (`scripts/compute_class_balance.py`)

Reads `manifest_train.json`, counts benign vs malware, writes:

```json
{ "n_benign": ..., "n_malware": ..., "pos_weight": n_benign/n_malware }
```

Training uses this when `training.auto_pos_weight: true` and manual `pos_weight` is null.

### Evaluation (`src/training/evaluate.py`)

Paper metrics on val (or train) split: **ACC**, **F1**, **ROC-AUC** at `evaluation.threshold` (0.5).

### Artifact bundle (`scripts/package_artifacts.sh`)

Creates `artifacts/pattern_b_bundle.tar.gz` with checkpoints, vocab, normalization stats, manifests, class balance, config. **Shards excluded** (copy `artifacts/processed/shards/` separately).

### How to run (remote)

```bash
# Full run
APK_ROOT=/data/apks ./run_pattern_b.sh

# Smoke test (small subset)
APK_ROOT=/data/apks PREPROCESS_LIMIT=500 EXTRACT_LIMIT=500 EPOCHS=2 ./run_pattern_b.sh

# Resume training only
SKIP_PREPROCESS=1 ./run_pattern_b.sh

# Evaluate existing checkpoint
SKIP_PREPROCESS=1 SKIP_TRAIN=1 ./run_pattern_b.sh
```

### Verification

```bash
PYTHONPATH=. python -m unittest tests.test_evaluate tests.test_phase6_pipeline -v
```

Synthetic shards: balance → train 1 epoch → evaluate → package tarball.

### Artifacts to bring back

See `implementation_plan.md` §9: `best.pt`, `latest.pt`, `vocab.json`, `normalization_header.json`, manifests, `class_balance.json`, optional full `shards/` tree.

---

## Phase 7: Multi-Dex Handling (`multiple_dex_handling`)

**Status:** Complete  
**Date:** 2026-05-17

### Sequential steps

1. Add Dex discovery, stable sort, and aggregation (`sum` default) → `src/features/multidex.py`
2. Extend APK ZIP reader for all `classes*.dex` → `src/features/apk_extract.py` (`list_dex_entries`, `read_all_dex_from_apk`, `extract_apk_raw_header`)
3. Add per-Dex batch parse helper → `src/features/dex_header.py` (`extract_headers_from_dex_list`)
4. Wire sum-pooled headers into norm fit + shard extract → `src/preprocessing/fit_header_norm.py`, `src/preprocessing/extract_to_cache.py`
5. Record `multidex_mode` in shard manifests + norm stats metadata → `src/preprocessing/common.py`
6. Set default config → `config/default.yaml` (`preprocessing.multidex.mode: sum`)
7. Export new APIs → `src/features/__init__.py`
8. Unit tests → `tests/test_multidex.py`, `tests/test_dex_header.py` (sum case)
9. Config smoke check → `scripts/verify_setup.py`

### Goal

One pipeline for **single- and multi-Dex** APKs: discover every `classes*.dex`, parse each 104-d header, **element-wise sum** into one `H`, then existing min–max → MLP branch. No model/dataset/training code changes (`header` still 104-d).

### Default behavior

| APK | Dex files found | Aggregated `H` |
|-----|-----------------|----------------|
| Single-Dex | `classes.dex` | Sum of one vector (= that vector) |
| Multi-Dex | `classes.dex`, `classes2.dex`, … | Element-wise sum of all per-Dex vectors |

Config ablations (not default): `multidex.mode: mean`, `primary_only`, `concat`.

### Files created / modified

| File | Change |
|------|--------|
| `src/features/multidex.py` | **New** — `multidex_settings`, `aggregate_header_vectors`, `dex_suffix_sort_key` |
| `src/features/apk_extract.py` | List/read all Dex; `extract_apk_raw_header()` |
| `src/features/dex_header.py` | `extract_headers_from_dex_list()` |
| `src/features/__init__.py` | Export multidex APIs |
| `src/preprocessing/fit_header_norm.py` | Uses `extract_apk_raw_header` |
| `src/preprocessing/extract_to_cache.py` | Uses `extract_apk_raw_header` |
| `src/preprocessing/common.py` | `multidex_mode` in manifest JSON |
| `config/default.yaml` | `preprocessing.multidex` block; removed `dex_entry_name` |
| `tests/test_multidex.py` | **New** — ZIP fixtures, sum/single-Dex/mean/primary_only |
| `tests/test_dex_header.py` | Sum aggregation test |
| `scripts/verify_setup.py` | Asserts `multidex.mode: sum` |

### Unchanged (by design)

`src/models/*`, `src/data/*`, `src/training/*`, manifest BoW path, shard layout (`header`, `bow`, `label`).

### Remote re-run required

Old shards used **primary Dex only**. After Phase 7:

1. Delete or replace `artifacts/processed/shards/` and `processed_ids.txt`
2. Re-run `fit_header_norm.py` → new `normalization_header.json`
3. Re-run `extract_to_cache.py` → new shards
4. Retrain with `--fresh` (checkpoints trained on old `H` are invalid)

### How to run tests

```bash
cd dual_branch_merge_approach
PYTHONPATH=. python -m unittest tests.test_multidex tests.test_dex_header -v
```

### Verification

```bash
PYTHONPATH=. python -m unittest tests.test_multidex tests.test_dex_header -v
PYTHONPATH=. python scripts/verify_setup.py
```

---
