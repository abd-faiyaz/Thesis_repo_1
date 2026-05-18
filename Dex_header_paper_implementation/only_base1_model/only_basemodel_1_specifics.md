# Base Model 1 (MLP(H)) — Implementation Log

Per-phase notes: what was done, where, and why. Aligned with `only_base1_implementation_gemini.md`.

---

## Phase 1: Workspace & Environment Setup

**Status:** Complete  
**Date:** 2026-05-15

### Goal

Establish a modular PyTorch pipeline layout and pinned dependencies before APK/Dex preprocessing (Phase 2). Separates concerns so GPU training never touches raw APKs at load time.

### Dependencies

| Dependency | Location | Reason |
|------------|----------|--------|
| `torch`, `torchvision` | `requirements.txt` | Model, tensors, optional GPU training |
| `numpy` | `requirements.txt` | Feature arrays during preprocessing |
| `scikit-learn` | `requirements.txt` | Phase 6 metrics (ACC, F1, AUC) |
| `tqdm` | `requirements.txt` | Phase 5 training progress bars |
| `PyYAML` | `requirements.txt` | `config/default.yaml` loader |
| `zipfile` | Python stdlib | Phase 2 in-memory APK unpack (no pip package) |

Install (from `only_base1_model/`):

```bash
pip install -r requirements.txt
python scripts/verify_setup.py
```

### Directory layout

```
only_base1_model/
├── config/default.yaml          # paths + hyperparams (paper-aligned defaults)
├── requirements.txt
├── scripts/verify_setup.py      # Phase 1 smoke test
├── artifacts/
│   ├── processed/               # Phase 2 output tensors
│   └── checkpoints/             # Phase 5 resume checkpoints
├── src/
│   ├── config.py                # YAML load + path resolution
│   ├── constants.py             # DEX_MAGIC, header size (Phase 2)
│   ├── features/dex_header.py   # Phase 2 — parse DexHeader
│   ├── preprocessing/preprocess_apks.py  # Phase 2 — batch APK → tensors
│   ├── data/
│   │   ├── store.py             # load .pt / .npy bundle
│   │   ├── dataset.py           # DexDataset
│   │   └── dataloaders.py       # train/val DataLoaders
│   ├── models/mlp_header.py     # Phase 4 — MLP(H)
│   └── training/
│       ├── checkpoint.py        # save/load (partial impl for Phase 5)
│       ├── train.py             # Phase 5
│       └── evaluate.py          # Phase 6
└── only_basemodel_1_specifics.md  # this file
```

**Why this split:** Matches the plan’s pipeline stages — preprocessing (CPU-heavy, once), dataloader (fast I/O), model (static graph), training/eval (GPU + metrics). Mirrors the dual-branch plan under `custom_approach/` for consistency across thesis experiments.

### Files created / purpose

| File | Purpose |
|------|---------|
| `config/default.yaml` | Single source for APK root, artifact paths, batch size 16, SGD lr 0.005, momentum 0.9, lr decay 0.5, BCELoss — values from paper/plan for later phases |
| `src/config.py` | `load_config()` resolves relative paths against package root; `ensure_artifact_dirs()` creates `artifacts/processed` and `artifacts/checkpoints` |
| `src/constants.py` | `DEX_MAGIC = b"dex\n035\x00"` and `DEX_HEADER_SIZE = 0x70` for Phase 2 parser |
| `src/training/checkpoint.py` | Minimal `torch.save` / `torch.load` helpers (Phase 5 will wire epoch/optimizer/scheduler) |
| Phase 5 (`setup.py`, `loops.py`, `train.py`, `checkpoint.py`) | SGD training, tqdm, resume checkpoints |
| Phase 6 (`evaluate.py`) | ACC, F1, AUC; wired into train + standalone CLI |
| Phase 4 (`mlp_header.py`) | MLP(H): two FC+BN+ReLU blocks + sigmoid head |
| Phase 3 modules (`store.py`, `dataset.py`, `dataloaders.py`) | Load preprocessed tensors; train/val DataLoaders |
| Phase 2 modules (`dex_header.py`, `normalization.py`, `apk_extract.py`, `labels.py`, `preprocess_apks.py`) | Full APK → tensor preprocessing pipeline |
| `scripts/verify_setup.py` | Confirms imports and config load without running training |
| `.gitignore` | Excludes large `data/`, processed tensors, and `.pth` checkpoints from git |

### Config defaults (reasoning)

- **`paths.apk_root: data/apks`** — placeholder; on the remote 50k-APK machine, override via env or a local YAML copy.
- **`preprocessing.dex_entry_name: classes.dex`** — primary Dex only (per guidelines and custom plan §3.4).
- **`data.batch_size: 16`** — paper specification (Phase 3).
- **`training.learning_rate: 0.005`, `momentum: 0.9`, `lr_decay_factor: 0.5`** — paper SGD settings (Phase 5).
- **`model.hidden_dim: 128`** — placeholder; **input dim is 104** after Phase 2 (set `input_dim` in Phase 4 from `feature_dim` in saved `.pt`).

### Verification

Run from `only_base1_model/`:

```bash
python scripts/verify_setup.py
```

Expected when fully installed: all pip deps OK, package/config OK, `zipfile` OK, exit code 0.

### Not in Phase 1 (deferred)

- Dex parsing, APK iteration, `DexDataset`, `MLPHeader` forward pass, training loop, sklearn metrics — Phases 2–6.

---

## Phase 2: APK Preprocessing & DEX Feature Extraction

**Status:** Complete  
**Date:** 2026-05-15

### Goal

Precompute Dex-header feature tensors from APKs so Phase 3 `DataLoader` never unpacks ZIPs or parses Dex during training. Matches the plan: in-memory `zipfile` read → magic check → header bytes → byte-level hex-style encoding → dataset-wide min-max → aggregate `.pt` file + labels.

### Feature vector design

| Step | Where | Reasoning |
|------|-------|-----------|
| Magic check | `src/features/dex_header.py` → `validate_magic()` | First 8 bytes must be `dex\n` + 3-digit version + `\0` (e.g. `035`, `037`); rejects non-Dex APKs early |
| Struct parse | `parse_dex_header_fields()` | Unpacks checksum, 20-byte SHA-1 signature, link segment (`link_size`/`link_off`), `map_off`, and all ID section sizes/offsets (string/type/proto/field/method/class_defs/data) for validation and debugging |
| 1D encoding | `extract_header_features()` | Bytes **8–111** (104 bytes post-magic) divided by 255 → values in `[0, 1]` (“hexadecimal equivalents” / gray 1D vector per paper) |
| Min-max | `src/features/normalization.py` | Second normalization across the **training corpus**: per-dimension `(x - min) / (max - min)`; constants map to 0; stats saved to `artifacts/normalization.json` |
| **Feature dim** | **104** | `DEX_HEADER_SIZE (112) - DEX_MAGIC_LEN (8)` |

Parsed field order (for `parse_dex_header_fields`): checksum, signature, file_size, header_size, endian_tag, link_size, link_off, map_off, string_ids_*, type_ids_*, proto_ids_*, field_ids_*, method_ids_*, class_defs_*, data_size, data_off.

### APK pipeline

| Component | File | Role |
|-----------|------|------|
| ZIP extract | `src/preprocessing/apk_extract.py` | `read_classes_dex()` opens APK, reads `classes.dex` (primary Dex only; also matches `foo/classes.dex`) |
| Labels | `src/preprocessing/labels.py` | `parent_folder`: walk parents for `benign`/`malware` folder names; `csv`: manifest with `path` + `label` columns |
| Batch job | `src/preprocessing/preprocess_apks.py` | Discovers `**/*.apk`, tqdm loop, failed paths → `artifacts/failed_apks.log`, saves aggregate output |

### Output artifacts

| Path | Contents |
|------|----------|
| `artifacts/processed/dex_header_features.pt` | `features` `[N, 104]`, `labels` `[N]`, `paths`, `normalization_mins/maxs`, metadata |
| `artifacts/normalization.json` | `mins`, `maxs`, `feature_dim`, `num_samples` (for inference consistency) |
| `artifacts/failed_apks.log` | Tab-separated: APK path, failure reason |

`output_format: npy` in config writes separate `.features.npy`, `.labels.npy`, `.paths.npy` instead of a single `.pt` bundle.

### Config additions (`config/default.yaml`)

```yaml
preprocessing:
  label_mode: parent_folder   # or csv
  labels_csv: null
  benign_names: [benign, goodware, clean, good, "0"]
  malicious_names: [malware, malicious, virus, bad, "1"]
```

On the remote 50k-APK machine: set `paths.apk_root` to the dataset directory (or pass `--apk-root`). Folder layout example:

```
data/apks/benign/*.apk
data/apks/malware/*.apk
```

Or use `label_mode: csv` with `data/labels.csv`.

### How to run

From `only_base1_model/`:

```bash
# Full dataset
./scripts/run_preprocess.sh
# or
PYTHONPATH=. python -m src.preprocessing.preprocess_apks

# Override APK root / smoke test
PYTHONPATH=. python -m src.preprocessing.preprocess_apks --apk-root /path/to/apks --limit 100
```

### Tests

```bash
PYTHONPATH=. python -m unittest tests.test_dex_header -v
```

Synthetic minimal Dex header (no APK): validates magic, struct unpack, 104-dim vector, min-max pipeline.

### Failure handling

- Missing/invalid Dex, bad ZIP, unknown label → logged to `failed_apks.log`, sample skipped (not silently labeled).
- If **zero** APKs succeed → `RuntimeError` (no empty training file).

### Deferred to later phases

- Train/val split at APK level (separate roots) can still be added; Phase 3 adds index-level split for convenience.

---

## Phase 3: PyTorch Dataset & DataLoader

**Status:** Complete  
**Date:** 2026-05-15

### Goal

Load Phase 2 tensors only during training — no APK/ZIP/Dex I/O in the training loop. Provide paper batch size (16), shuffled training batches, and sequential validation batches.

### Components

| File | Role |
|------|------|
| `src/data/store.py` | `load_processed_bundle()` — reads `.pt` (or `.npy` triplet from Phase 2) into `ProcessedBundle` |
| `src/data/dataset.py` | `DexDataset` — `__getitem__` returns `(x, y)` float tensors; optional `indices` for subsets |
| `src/data/dataloaders.py` | `build_train_loader` (shuffle=True), `build_eval_loader` (shuffle=False), `build_dataloaders_from_config` |

### DexDataset behavior

1. Input: `features` `[N, 104]`, `labels` `[N]` from preprocessed file (or in-memory bundle).
2. `__getitem__(i)` → `x` shape `[104]`, `y` scalar `0.0` / `1.0`.
3. `from_processed_file(path)` — loads `artifacts/processed/dex_header_features.pt` via config filename.
4. `from_bundle(bundle, indices=...)` — supports train/val subsets without duplicating tensors.

### DataLoader behavior

| Loader | `shuffle` | `batch_size` | Use |
|--------|-----------|--------------|-----|
| `build_train_loader` | **True** | 16 (config) | Training |
| `build_eval_loader` | **False** | 16 (config) | Validation / test |

Other settings from `config/default.yaml`: `num_workers: 4`, `pin_memory: true` (effective when CUDA is used in Phase 5).

### Train / val split

- `split_train_val_indices(N, val_fraction=0.2, seed=42)` — random permutation split.
- `build_dataloaders_from_bundle()` / `build_dataloaders_from_config()` return `(train_loader, val_loader, feature_dim)`.
- **Reasoning:** Phase 2 does not split data; Phase 3 provides a default 80/20 index split so Phase 5 can train and validate immediately. Override by passing custom `indices` to `DexDataset` if you pre-split APKs on the remote machine.

### Config additions

```yaml
data:
  batch_size: 16
  val_fraction: 0.2
  random_seed: 42
```

### Usage (after Phase 2 preprocessing)

```python
from src.config import load_config
from src.data import build_dataloaders_from_config

cfg = load_config()
train_loader, val_loader, feature_dim = build_dataloaders_from_config(cfg)
# feature_dim == 104
```

### Verification

```bash
PYTHONPATH=. python -m unittest tests.test_dataset -v
PYTHONPATH=. python scripts/verify_dataloader.py
```

`verify_dataloader.py` uses the real `.pt` if present; otherwise a synthetic 32-sample bundle.

### Deferred to Phase 5+

- Training loop consumes `train_loader` / `val_loader` and this model.

---

## Phase 4: Model Architecture (Base Model 1)

**Status:** Complete  
**Date:** 2026-05-15

### Goal

Implement MSFDroid **MLP(H)**: a shallow feed-forward network on the 104-dim Dex header vector with binary sigmoid output. Input size is dynamic (tied to `feature_dim` from Phase 2/3).

### Architecture (`src/models/mlp_header.py`)

| Layer | Configuration | Reason |
|-------|---------------|--------|
| **Input** | `input_dim` = 104 (from preprocessed features) | Matches Phase 2 header byte vector length |
| **Hidden block 1** | `Linear(104 → 128)` → `BatchNorm1d(128)` → `ReLU` | Paper: FC + BN + ReLU |
| **Hidden block 2** | `Linear(128 → 128)` → `BatchNorm1d(128)` → `ReLU` | Second hidden block, same width |
| **Output** | `Linear(128 → 1)` → `Sigmoid` | Binary malware probability ∈ [0, 1] |

`hidden_dim` defaults to **128** via `config/default.yaml` → `model.hidden_dim`.

### API

| Symbol | Purpose |
|--------|---------|
| `MLPHeader(input_dim, hidden_dim)` | `nn.Module` class |
| `build_mlp_header(input_dim, hidden_dim)` | Explicit factory |
| `build_mlp_header_from_config(cfg, input_dim)` | Reads `hidden_dim` from YAML |
| `forward(x)` | `x`: `(B, input_dim)` or `(input_dim,)` → `(B, 1)` probabilities |

### Wiring with Phase 3

```python
from src.config import load_config
from src.data import build_dataloaders_from_config
from src.models import build_mlp_header_from_config

cfg = load_config()
train_loader, val_loader, feature_dim = build_dataloaders_from_config(cfg)
model = build_mlp_header_from_config(cfg, input_dim=feature_dim)
```

### Parameter count (default)

With `input_dim=104`, `hidden_dim=128`: **30,593** trainable parameters (two hidden blocks + output head).

### Verification

```bash
PYTHONPATH=. python -m unittest tests.test_mlp_header -v
PYTHONPATH=. python scripts/verify_model.py
```

Tests cover batch/single forward shapes, sigmoid range [0,1], wrong-dim error, and config factory.

### Metrics (Phase 6)

- Validation reports ACC, F1, AUC each epoch via `validation_epoch`.

---

## Phase 5: Training Loop with Resiliency & Progression

**Status:** Complete  
**Date:** 2026-05-15

### Goal

Train `MLPHeader` on preprocessed Dex features with paper hyperparameters, live `tqdm` progress, and resume-safe checkpoints after power loss or interruption.

### Training flow (`src/training/train.py`)

1. Load config → build train/val `DataLoader`s (Phase 3) and `MLPHeader` (Phase 4).
2. Build **BCELoss**, **SGD** (lr `0.005`, momentum `0.9`), **StepLR** (γ=`0.5`, step every `lr_step_size` epochs).
3. If `artifacts/checkpoints/latest_checkpoint.pth` exists and not `--fresh` → restore model, optimizer, scheduler, `next_epoch`.
4. For each epoch:
   - **Train** (`loops.train_one_epoch`): `tqdm` bar with batch loss, running avg loss, samples/s.
   - **Validate** (`loops.validate_one_epoch`): sequential val loader, val loss in `tqdm`.
   - `scheduler.step()`; print epoch summary (train/val loss, current lr).
   - Save checkpoint with `next_epoch`, state dicts, `current_loss` (= train loss), `val_loss`.

### Hyperparameters (from `config/default.yaml`)

| Setting | Value | Module |
|---------|-------|--------|
| Loss | `nn.BCELoss` | `setup.build_training_objects` |
| Optimizer | SGD, lr `0.005`, momentum `0.9` | same |
| LR decay | StepLR, γ=`0.5`, step_size=`10` | same |
| Batch size | `16` | Phase 3 `data.batch_size` |
| Epochs | `50` (override `--epochs`) | `training.epochs` |
| Device | `cuda` if available, else CPU | `setup.resolve_device` |

Labels are reshaped to `(batch, 1)` to match sigmoid output for BCELoss.

### Checkpoint format (`latest_checkpoint.pth`)

```python
{
  "next_epoch": int,           # epoch index to run next (0-based)
  "model_state_dict": ...,
  "optimizer_state_dict": ...,
  "scheduler_state_dict": ...,
  "current_loss": float,       # last train loss (plan name)
  "train_loss": float,
  "val_loss": float,
  "feature_dim": 104,
}
```

**Resume:** `restore_from_checkpoint()` loads all state; training continues from `next_epoch`. Use `--fresh` to ignore an existing checkpoint.

### Files

| File | Role |
|------|------|
| `src/training/setup.py` | criterion, optimizer, scheduler, device |
| `src/training/loops.py` | `train_one_epoch`, `validate_one_epoch` + tqdm |
| `src/training/checkpoint.py` | save/load/build/restore helpers |
| `src/training/train.py` | CLI + `run_training()` |
| `scripts/run_train.sh` | Wrapper |

### How to run

```bash
# After Phase 2 preprocessing
./scripts/run_train.sh

# Short smoke test (CPU, 2 epochs)
PYTHONPATH=. python -m src.training.train --epochs 2

# Restart from scratch
PYTHONPATH=. python -m src.training.train --fresh
```

### Verification

```bash
PYTHONPATH=. python -m unittest tests.test_training -v
```

Tests checkpoint round-trip and a 2-epoch synthetic run with resume.

### Integrated with Phase 6

- Each epoch now prints and checkpoints `val_metrics` (ACC, F1, AUC).

---

## Phase 6: Evaluation

**Status:** Complete  
**Date:** 2026-05-15

### Goal

Compute paper validation metrics — **Accuracy (ACC)**, **F1-Score**, **ROC-AUC** — using `scikit-learn` on held-out predictions, integrated into training and available as a standalone eval command.

### Metrics (`src/training/evaluate.py`)

| Metric | sklearn | Inputs |
|--------|---------|--------|
| ACC | `accuracy_score` | `y_true`, `y_pred` (thresholded at 0.5) |
| F1 | `f1_score` | binary precision/recall harmonic mean |
| AUC | `roc_auc_score` | `y_true`, `y_score` (sigmoid probabilities) |

`y_pred = (y_score >= threshold)` with `threshold` from `config/default.yaml` → `evaluation.threshold: 0.5`.

If validation has only one class, AUC is `nan` (logged as `n/a`).

### Functions

| Symbol | Role |
|--------|------|
| `compute_metrics(y_true, y_pred, y_score)` | Core metric dict |
| `format_metrics(metrics)` | `ACC=… F1=… AUC=…` string for logging |
| `collect_predictions(model, loader, device, threshold)` | Gather numpy arrays |
| `validation_epoch(...)` | Val loss + metrics in one tqdm pass (used by training) |
| `run_evaluation(cfg, checkpoint_path, split)` | Standalone eval on val or train split |

### Training integration (Phase 5 + 6)

After each epoch, `train.py` calls `validation_epoch` instead of loss-only validation:

```
Epoch 3/50 — train_loss=0.62 val_loss=0.71 lr=0.005000 ACC=0.8500 F1=0.8200 AUC=0.9100
```

Checkpoints now include `val_metrics: {accuracy, f1, roc_auc}`.

### Standalone evaluation

```bash
# After training (uses latest_checkpoint.pth)
./scripts/run_evaluate.sh

# Custom checkpoint / train split
PYTHONPATH=. python -m src.training.evaluate --checkpoint artifacts/checkpoints/latest_checkpoint.pth --split val
```

### Files

| File | Role |
|------|------|
| `src/training/evaluate.py` | Metrics + validation epoch + CLI |
| `scripts/run_evaluate.sh` | Shell wrapper |

### Verification

```bash
PYTHONPATH=. python -m unittest tests.test_evaluate -v
```

### Full pipeline (Phases 2 → 6)

```bash
PYTHONPATH=. python -m src.preprocessing.preprocess_apks --apk-root /path/to/apks
./scripts/run_train.sh
./scripts/run_evaluate.sh
```
