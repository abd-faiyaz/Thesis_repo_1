# Full Combined Pipeline (Pattern A) — Implementation Log

Per-phase notes aligned with [`implementation_plan.md`](implementation_plan.md).

**Python environment:** `/run/media/abd-faiyaz/Files/thesis_vigidroid/thesis_venv`

```bash
cd Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach
/run/media/abd-faiyaz/Files/thesis_vigidroid/thesis_venv/bin/python scripts/verify_setup.py
```

---

## Phase 1: Workspace & Environment Setup

**Status:** Complete  
**Date:** 2026-05-17

### Sequential steps

1. Pin dependencies to **thesis_venv** versions + `pyaxmlparser` → `requirements.txt`
2. Pattern A config (multidex, combined ASCNN dims, training) → `config/default.yaml`
3. YAML loader, paths, artifact dirs, model dim validation → `src/config.py`
4. Dex/BoW/combined constants → `src/constants.py`
5. `multidex_settings()` stub (full aggregation in Phase 2) → `src/features/multidex.py`
6. Package markers → `src/__init__.py`, `src/features/`, etc.
7. Smoke test → `scripts/verify_setup.py`
8. Gitignore artifacts → `.gitignore`

### Goal

Importable package and config before APK preprocessing (Phase 2). Training must never parse APKs at load time.

### Dependencies (thesis_venv)

| Package | Version in venv | Notes |
|---------|-----------------|-------|
| `torch` | 2.11.0+cpu | CPU build in thesis venv |
| `numpy` | 2.4.3 | |
| `scikit-learn` | 1.8.0 | Phase 6 metrics |
| `tqdm` | 4.67.3 | |
| `PyYAML` | 6.0.3 | |
| `pyaxmlparser` | 0.3.31 | Installed during Phase 1 (was missing from venv) |

`requirements.txt` pins the above. Reinstall if you recreate the venv:

```bash
/run/media/abd-faiyaz/Files/thesis_vigidroid/thesis_venv/bin/pip install -r requirements.txt
```

### Files created

| File | Purpose |
|------|---------|
| `requirements.txt` | Pinned deps matching thesis_venv |
| `config/default.yaml` | Paths, multidex `sum`, `combined_input_len: 4485`, `device: cpu` |
| `src/config.py` | `load_config`, `ensure_artifact_dirs`, `validate_model_dims` |
| `src/constants.py` | `DEX_HEADER_FEATURE_DIM=104`, lexicon/combined lengths |
| `src/features/multidex.py` | `multidex_settings()` stub (expanded in Phase 2) |
| `scripts/verify_setup.py` | Phase 1 gate |
| `.gitignore` | Excludes `data/`, `artifacts/` |

### Verification

```bash
/run/media/abd-faiyaz/Files/thesis_vigidroid/thesis_venv/bin/python scripts/verify_setup.py
```

### Not in Phase 1 (deferred)

- Feature extraction, preprocessing scripts, model, training — Phases 2–6.

---

## Phase 2: Multi-Dex Features & Preprocessing

**Status:** Complete  
**Date:** 2026-05-17

### Sequential steps

1. Per-Dex header parse (magic, bytes 8–111 → 104-d) → `src/features/dex_header.py`
2. Multi-dex discovery, sort, **sum/mean/primary_only** aggregation → `src/features/multidex.py`
3. ZIP: list/read all `classes*.dex`, `extract_apk_raw_header()` → `src/features/apk_extract.py`
4. Corpus min–max fit/transform → `src/features/normalization.py`
5. Manifest permissions + intents → multi-hot BoW → `src/features/manifest_bow.py`
6. Index APKs, stratified 90/10 split, CSV + split files → `src/preprocessing/scan_dataset.py`, `labels.py`, `common.py`
7. Lexicon from **train** manifests (freq ≥ 2, top-N) → `src/preprocessing/build_lexicon.py`
8. Header norm on **train** sum-pooled headers → `src/preprocessing/fit_header_norm.py`
9. Per-APK `.npz` shards + resume log + manifest JSON → `src/preprocessing/extract_to_cache.py`
10. End-to-end shell → `scripts/run_preprocess.sh`
11. Unit tests → `tests/test_multidex.py`, `test_dex_header.py`, `test_manifest_bow.py`

**Source:** Ported from tested `dual_branch_merge_approach/` (same feature contract; Pattern A shards are identical for fair ablation vs Pattern B).

### Goal

Precompute **H** (104-d, sum of all Dex headers) and **I** (4381-d BoW) per APK once. Phase 3+ loads shards only — no APK parsing during training.

### Sequential pipeline (run order)

| Step | Script | What runs | Why this order |
|------|--------|-----------|----------------|
| **1** | `scan_dataset.py` | Walk `apk_root` → CSV + train/val split files | Labels and splits must exist before any train-only stats |
| **2** | `build_lexicon.py` | Count manifest tokens on **train** → `vocab.json` | Vocabulary must not see val/test APKs |
| **3** | `fit_header_norm.py` | Sum-pool headers on **train** → `normalization_header.json` | Min–max stats must not see val/test APKs |
| **4** | `extract_to_cache.py` | For each train/val APK: `H` + `I` → `.npz` shard | Uses frozen vocab + norm from steps 2–3 |

Within **step 4**, per APK (inside `extract_to_cache.py`):

1. `extract_apk_raw_header()` — all `classes*.dex` → sum → 104-d raw **H**
2. `transform_minmax()` — normalize **H** with corpus stats
3. `extract_manifest_tokens()` + `build_multihot_vector()` — **I** (4381-d)
4. Write `shards/{split}/{apk_id}.npz`; append `processed_ids.txt` for resume

### Multi-dex flow (default `sum`)

```
APK → classes.dex, classes2.dex, …
  → Hᵢ per file (104-d raw)
  → H_raw = sum(Hᵢ)
  → H = minmax(H_raw, corpus stats)
  → I = manifest multi-hot
  → save shards/{split}/{apk_id}.npz
```

### Output artifacts

| Path | Contents |
|------|----------|
| `artifacts/dataset_index.csv` | `apk_id`, `path`, `label` |
| `artifacts/splits/train.txt`, `val.txt` | One `apk_id` per line |
| `artifacts/vocab.json` | `token_to_index`, `unk_index` |
| `artifacts/normalization_header.json` | `mins`, `maxs`, `multidex_mode` |
| `artifacts/processed/shards/{train,val}/<apk_id>.npz` | `header` (104,), `bow` (4381,), `label` |
| `artifacts/processed/manifest_{train,val}.json` | Shard index for DataLoader |
| `artifacts/processed/processed_ids.txt` | Resume log |
| `artifacts/failed_apks.log` | Skipped APKs |

### How to run (remote / local with APKs under `data/apks/`)

```bash
cd Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach
export PYTHONPATH=.
# optional: point config apk_root to your 50k tree
./scripts/run_preprocess.sh
# or stepwise with limits for smoke:
python -m src.preprocessing.scan_dataset --limit 100
python -m src.preprocessing.build_lexicon
python -m src.preprocessing.fit_header_norm
python -m src.preprocessing.extract_to_cache --limit 50
```

### Tests

```bash
PYTHONPATH=. /run/media/abd-faiyaz/Files/thesis_vigidroid/thesis_venv/bin/python -m unittest \
  tests.test_multidex tests.test_dex_header tests.test_manifest_bow -v
```

**Result:** 15 tests OK (synthetic multi-Dex ZIPs; no real APKs required for CI).

### Deferred to Phase 3

- ~~`CombinedPipelineDataset` / DataLoaders loading shards.~~ → Phase 3 below.

---

## Phase 3: PyTorch Dataset & DataLoader

**Status:** Complete  
**Date:** 2026-05-17

### Sequential steps

1. Load `manifest_{train,val}.json` and open `.npz` shards → `src/data/store.py`
2. `CombinedPipelineDataset` returns `(header, bow, label)` per sample → `src/data/dataset.py`
3. Train loader (shuffle) + val loader (sequential), batch size 16 → `src/data/dataloaders.py`
4. Wire manifests from config (no re-split) → `build_dataloaders_from_config()`
5. Smoke script + unit tests → `scripts/verify_dataloader.py`, `tests/test_dataset.py`

### Goal

Training loop reads **only** Phase 2 shards — no APK/ZIP/Dex/manifest I/O. Batches are `(B, 104)`, `(B, 4381)`, `(B,)` for Phase 4 `CombinedNet` to `concat` → ASCNN.

### Data flow

```
manifest_train.json / manifest_val.json
  → CombinedPipelineDataset.from_manifest()
  → __getitem__: load_shard_npz() → header (104,), bow (4381,), label
  → DataLoader batch → (B, 104), (B, 4381), (B,)
```

### API

| Function / class | Role |
|------------------|------|
| `load_shard_manifest()` | Parse JSON index written by Phase 2 |
| `load_shard_npz()` | One `.npz` → torch tensors |
| `CombinedPipelineDataset` | `Dataset` over manifest entries |
| `build_dataloaders_from_config(cfg)` | Train + val loaders from `config/default.yaml` |

### Verification

```bash
PYTHONPATH=. thesis_venv/bin/python -m unittest tests.test_dataset -v
PYTHONPATH=. thesis_venv/bin/python scripts/verify_dataloader.py
```

**Result:** 4 unit tests OK; verify script uses synthetic shards when real manifests are absent.

### Usage (after Phase 2 preprocessing)

```python
from src.config import load_config
from src.data.dataloaders import build_dataloaders_from_config

cfg = load_config()
train_loader, val_loader, header_dim, bow_dim = build_dataloaders_from_config(cfg)
header, bow, labels = next(iter(train_loader))
# header.shape == (16, 104), bow.shape == (16, 4381)
```

### Deferred to Phase 4

- ~~`CombinedNet` forward on batched `(header, bow)`; concat inside model.~~ → Phase 4 below.

---

## Phase 4: Model Architecture (CombinedNet Forward Pass)

**Status:** Complete  
**Date:** 2026-05-17

### Sequential steps

1. ASU building block (dynamic gate + soft threshold) → `src/models/adaptive_shrinkage_unit.py`
2. Three-layer ASCNN on **concat(H,I)** with right-pad to 4488 → `src/models/ascnn_combined.py`
3. MLP head on 128-d embedding → `src/models/classifier_head.py`
4. `CombinedNet`: `concat` → ASCNN → classifier → logit → `src/models/combined_net.py`
5. Forward smoke (dummy or dataloader batch) → `scripts/verify_model.py`
6. Unit tests → `tests/test_combined_net.py`

### Goal

Forward-only graph for Pattern A (`ASCNN(C)`): header and manifest mix at **conv layer 1**, not after separate encoders.

### Forward path

```
header (B, 104) ──┐
                  ├─ concat → (B, 4485) → pad → (B, 1, 4488)
bow (B, 4381) ────┘         ASCNN (3× ASU + AvgPool) → (B, 128)
                            ClassifierHead → logit (B, 1)
```

| Module | Input | Output |
|--------|--------|--------|
| `CombinedNet._concat_features` | H, I | `(B, 4485)` |
| `ASCNNCombined` | combined seq | `(B, 128)` |
| `ClassifierHead` | embedding | `(B, 1)` logit |

**Parameters (default config):** **153,921** (smaller than Pattern B dual-branch ~290k).

### Verification

```bash
PYTHONPATH=. thesis_venv/bin/python -m unittest tests.test_combined_net -v
PYTHONPATH=. thesis_venv/bin/python scripts/verify_model.py
```

**Result:** 10 unit tests OK; verify script OK on dummy batch.

### Deferred to Phase 5

- ~~`train.py`, BCE loss, SGD, checkpoints, tqdm, `--resume`.~~ → Phase 5 below.

---

## Phase 5: Training Loop & Checkpoints

**Status:** Complete  
**Date:** 2026-05-17

### Sequential steps

1. Checkpoint save/load + RNG state → `src/training/checkpoint.py`
2. `BCEWithLogitsLoss` + optional `pos_weight` → `src/training/losses.py`
3. SGD + StepLR from config → `src/training/setup.py`
4. Train/val loops with tqdm → `src/training/loops.py`
5. CLI `train.py` (`--resume`, `--fresh`, `--epochs`) → `src/training/train.py`
6. Shell entrypoint → `scripts/run_train.sh`
7. Integration tests (2 epochs + resume) → `tests/test_training.py`

### Goal

Train `CombinedNet` on Phase 2 shards with paper-style **SGD** (lr 0.005, momentum 0.9, StepLR ×0.5), tqdm, and resume after power loss.

### Training flow (per epoch)

1. `build_dataloaders_from_config()` — load manifests (Phase 3).
2. `build_combined_net_from_config()` — model (Phase 4).
3. **Train loop:** forward → BCE → backward → SGD step (tqdm per batch).
4. **Val loop:** forward → val loss (no grad).
5. `scheduler.step()` — LR decay every `lr_step_size` epochs.
6. Save `best.pt` if val loss improved; save `latest.pt` each epoch.

### Checkpoint contents

`latest.pt` / `best.pt`: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `next_epoch`, `global_step`, `best_val_loss`, RNG states.

### CLI

```bash
./scripts/run_train.sh                    # train (resume if latest.pt exists)
./scripts/run_train.sh --fresh            # ignore checkpoint
./scripts/run_train.sh --resume           # explicit resume from latest.pt
./scripts/run_train.sh --epochs 10        # override epoch count
```

Requires Phase 2 manifests before real APK training.

### Verification

```bash
PYTHONPATH=. thesis_venv/bin/python -m unittest tests.test_training -v
```

**Result:** 3 tests OK — checkpoint roundtrip, pos_weight, 2-epoch train + resume to epoch 3.

### Deferred to Phase 6

- ~~Full 50k remote train, `evaluate.py`, class balance, packaging.~~ → Phases 6–7 below.

---

## Phase 6: Full Remote Run & Evaluation

**Status:** Complete  
**Date:** 2026-05-17

### Sequential steps

1. End-to-end orchestrator → `run_pattern_a.sh`
2. Class balance from train manifest → `scripts/compute_class_balance.py`
3. Val metrics ACC / F1 / AUC → `src/training/evaluate.py`
4. Portable tarball (no shards) → `scripts/package_artifacts.sh`
5. Eval shell → `scripts/run_evaluate.sh`
6. Tests → `tests/test_evaluate.py`, `tests/test_phase6_pipeline.py`

### Goal

Single entry point for remote ~50k run: preprocess (multi-dex sum) → balance → train → metrics → export.

### Orchestrator (`run_pattern_a.sh`)

| Env variable | Purpose |
|--------------|---------|
| `APK_ROOT` | APK tree (`benign/`, `malware/`) |
| `EPOCHS` | Override training epochs |
| `SKIP_PREPROCESS` | Use existing shards |
| `SKIP_DEX_STATS` | Skip Dex histogram |
| `SKIP_TRAIN` / `SKIP_EVAL` / `SKIP_PACKAGE` | Skip stages |
| `FRESH_TRAIN` | Ignore checkpoint |
| `PREPROCESS_LIMIT` / `EXTRACT_LIMIT` | Smoke-test caps |
| `INSTALL_DEPS` | `pip install -r requirements.txt` |

Logs append to `artifacts/pipeline.log`.

### Evaluation (`src/training/evaluate.py`)

Loads `best.pt` (or `latest.pt`), runs val split, prints **ACC / F1 / ROC-AUC**, writes `artifacts/checkpoints/metrics_val.json`.

```bash
./scripts/run_evaluate.sh
./scripts/run_evaluate.sh --checkpoint artifacts/checkpoints/best.pt
```

### Artifact bundle

`artifacts/pattern_a_bundle.tar.gz` — checkpoints, vocab, norm, manifests, metrics, config. **Shards excluded** (copy `processed/shards/` separately).

### How to run (remote)

```bash
# Full run
APK_ROOT=/data/apks ./run_pattern_a.sh

# Smoke test
APK_ROOT=/data/apks PREPROCESS_LIMIT=500 EXTRACT_LIMIT=500 EPOCHS=2 ./run_pattern_a.sh

# Resume training only
SKIP_PREPROCESS=1 ./run_pattern_a.sh

# Evaluate existing checkpoint
SKIP_PREPROCESS=1 SKIP_TRAIN=1 ./run_pattern_a.sh
```

### Verification

```bash
PYTHONPATH=. thesis_venv/bin/python -m unittest tests.test_evaluate tests.test_phase6_pipeline -v
```

**Result:** 5 tests OK (synthetic shards: train → eval → package).

---

## Phase 7: Multi-Dex Production Policy (Built Into Phase 2)

**Status:** Complete  
**Date:** 2026-05-17

Pattern A ships **multi-dex from day one** (unlike Pattern B where Phase 7 was a later add-on). No separate model/dataset changes — `header` stays **104-d**.

### Sequential policy (already in codebase)

1. Discover all `classes*.dex` in APK → `src/features/apk_extract.py`
2. Per-Dex header → 104-d → `src/features/dex_header.py`
3. **Sum-pool** → single `H` → `src/features/multidex.py`
4. Min–max + shard extract → `fit_header_norm.py`, `extract_to_cache.py`
5. Config default `preprocessing.multidex.mode: sum` → `config/default.yaml`

### Phase 7 additions (operations)

| Step | Tool | Purpose |
|------|------|---------|
| Dex histogram | `scripts/compute_dex_stats.py` | `% multi-dex` on train/val for thesis |
| Ablation config | `multidex.mode: primary_only` | Reproduce paper’s classes.dex-only baseline |
| Verification tests | `tests/test_multidex_phase7.py` | Policy + sum vs primary_only |

```bash
python scripts/compute_dex_stats.py --split train
# → artifacts/dex_stats.json
```

### Remote re-run if changing multidex mode

1. Delete `artifacts/processed/shards/` + `processed_ids.txt`
2. Re-run `fit_header_norm.py` + `extract_to_cache.py`
3. Retrain with `./scripts/run_train.sh --fresh`

### Verification

```bash
PYTHONPATH=. thesis_venv/bin/python -m unittest tests.test_multidex tests.test_multidex_phase7 -v
PYTHONPATH=. thesis_venv/bin/python scripts/verify_setup.py
```
