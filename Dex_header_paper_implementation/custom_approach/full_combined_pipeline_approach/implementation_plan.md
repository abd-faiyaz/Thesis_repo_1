# Implementation Plan: Full Combined Pipeline (Pattern A)

**Pattern:** Single tower — **`concat(H, I)`** → **ASCNN** → **MLP head** → malware probability. Closest to MSFDroid **Base Model 4** (`ASCNN(C)`): combined Dex header + manifest BoW without a separate entropy/MEM-PSD branch.

**Context:** Pattern A from [`../detailed_custom_implementation_plan.md`](../detailed_custom_implementation_plan.md). Sibling [`../dual_branch_merge_approach/`](../dual_branch_merge_approach/) implements **Pattern B** (late fusion: `MLP(H)` + `ASCNN(I)`). Read the parent doc for omissions (no `MLP(M)`).

**Multi-dex (Vigidroid requirement):** Modern APKs routinely ship `classes.dex`, `classes2.dex`, … The paper’s `classes.dex`-only policy is a known blind spot — see [`../../dex_related_instruction.md`](../../dex_related_instruction.md). **This plan treats multi-dex as default from Phase 2:** discover all `classes*.dex`, parse each header, **sum-pool** to a single **104-d** `H`, then concat with manifest BoW. Same aggregation policy as Pattern B §12.

**Reuse:** Per-Dex parsing matches [`../../only_base1_model/`](../../only_base1_model/) (bytes 8–111 → 104 floats, corpus min–max). Feature/preprocessing layout mirrors Pattern B where possible (`multidex.py`, shard format) so you can port or diff files; **models and training differ** (one ASCNN on `concat(H,I)`).

**Training environment:** ~**50,000 APKs** on a **remote PC**. This document covers **preprocessing → training → validation metrics**. Formal held-out **test** reporting is optional (Phase 6); use **train/val** for checkpointing.

**Prerequisite reading:** `../detailed_custom_implementation_plan.md`, [`../../dex_related_instruction.md`](../../dex_related_instruction.md).

---

## 0. Quick Picture

```
APK (zipfile)
 ├─ classes.dex, classes2.dex, …
 │     ├─ per-Dex: magic + bytes 8–111 → Hᵢ (104-d each)
 │     └─ sum-pool ─────────────────────► H (104-d) ──┐
 └─ AndroidManifest.xml ──► BoW I (N+1) ─────────────┤
                                                      ▼
                              X = concat(H, I)  (104 + N+1)
                                                      │
                                                      ▼
                                    ASCNN(X)  →  embedding (128-d)
                                                      │
                                                      ▼
                                    MLP head  →  sigmoid → P(malware)
```

No structural-entropy / MEM-PSD branch (`MLP(M)`). No separate header MLP — header and manifest mix at **layer 1** of the conv stack.

---

## 1. What Each Part Does (Plain Language)

| Symbol | Meaning |
|--------|---------|
| **Hᵢ** | Raw header vector for one Dex file (104-d, `/255` before corpus norm). |
| **H** | **Aggregated** header for the whole APK (default: **element-wise sum** of all **Hᵢ** → still 104-d). |
| **I** | Manifest **bag-of-words** over permissions + intent keywords (length **N+1** with UNK). |
| **X** | **`concat(H, I)`** — fixed order **header first** (`[H \|\| I]`). |
| **ASCNN(X)** | Three adaptive-shrinkage conv blocks + pool (paper Fig. 7–8), input length **L = 104 + (N+1)**. |
| **MLP head** | FC + BN + ReLU on pooled embedding → single logit. |

**Why Pattern A vs B?** Fewer parameters, one forward path, direct `ASCNN(C)` alignment. Trade-off: harder to inspect “header-only vs manifest-only” scores without ablation models.

---

## 2. Feature Extraction

### 2.1 Dex header branch (`H`) — multi-dex default

Align with [`../../dex_related_instruction.md`](../../dex_related_instruction.md), `only_base1_model`, and Pattern B §2.1.

1. Open APK (`zipfile`); discover **all** entries whose **basename** matches `^classes(\d*)\.dex$` (`classes.dex`, `classes2.dex`, …).
2. **Sort:** `classes.dex` first, then numeric suffix (`classes2.dex` → 2, …).
3. For **each** Dex:
   - **Magic** check: `dex\n` + version + `\0`.
   - Encode bytes **8–111** as floats in `[0, 1]` via `/255.0` → **Hᵢ** (104-d).
4. **Aggregate (default):** `H_raw = sum(H₀, H₁, …)` element-wise → **104-d**. Single-Dex APK: sum equals that one vector.
5. **Corpus min–max** on aggregated **H_raw** (training split only) → normalized **H**; save `artifacts/normalization_header.json`.
6. Final model input **`d_h = 104`** (unchanged vs paper single-Dex).

| `preprocessing.multidex.mode` | Behavior | `d_h` |
|-------------------------------|----------|-------|
| **`sum`** (default) | Element-wise sum of all **Hᵢ** | 104 |
| `mean` | Element-wise mean | 104 |
| `primary_only` | **H₀** only (`classes.dex`) — paper/ablation baseline | 104 |
| `concat` | Pad to `max_dex`, concat vectors | **104 × max_dex** — **not default**; requires `model.header_dim` and ASCNN input width change |

**Failures:** No `classes*.dex`, or **any** Dex fails validation → log `artifacts/failed_apks.log`, **skip** APK (no synthetic label).

**Thesis note:** Sum-pooling captures **total structural footprint** across Dex files without growing the ASCNN input past 104 header dims — important when malware may live in `classes2.dex+`.

### 2.2 Manifest branch (`I`)

Same contract as Pattern B / parent plan §3.2:

1. Decode **`AndroidManifest.xml`** via **`pyaxmlparser`** (binary XML → permissions + intents).
2. **Lexicon (train only):** freq ≥ 2, top **N** (default **4380**), **UNK** bucket.
3. **BoW:** binary **multi-hot** → **`I`** shape **`(N+1,)`** `float32`.
4. Save `artifacts/vocab.json`.

Manifest is **one per APK** — multi-dex does not change **I**.

### 2.3 Concatenation layout (model input)

| Rule | Value |
|------|--------|
| Order | **`X = [H \|\| I]`** (header first) — persist in config |
| Length | **`L = d_h + (N+1)`** → default **104 + 4381 = 4485** |
| Caching | Shards store **`header`**, **`bow`**, **`label`** separately; **concat in Dataset or model** (recommended). Optional precomputed **`combined`** in `.npz` for I/O speed (rebuild if `N` or multidex mode changes). |
| Scale | BoW is 0/1; header is normalized floats. **v1:** concat raw **H** and **I**. **Optional ablation:** `LayerNorm` on each branch inside `CombinedNet` before concat if training is unstable. |

### 2.4 ASCNN input shape

- Reshape **X** to **`(batch, 1, L)`** for `Conv1d`.
- Paper strides: kernel **3**, strides **2, 2, 1** → sequence length may not divide evenly; set fixed **`model.combined_padded_len`** (≥ L) in config, **pad right with zeros**, same at train and inference.
- Default target: compute minimal pad from **4485** (document exact value in `patternA_specifics.md` after Phase 4 forward test).

### 2.5 Explicitly omitted

- Full-Dex Shannon entropy, Burg PSD, `MLP(M)`.

---

## 3. Preprocessing Pipeline (~50k APKs, Remote)

CPU-heavy, run **once** (resumable). Training loads **cached shards only**.

### 3.1 Labels and index

APKs under **`apk_root`** with class folders, e.g. `benign/*.apk`, `malware/*.apk`.

**`scan_dataset.py` is source of truth:** labels from **parent folder names** (`benign` → 0, `malware` → 1). External CSVs are **not** used unless you change config.

Output: `artifacts/dataset_index.csv` (`path`, `label`, `apk_id` optional SHA-256).

### 3.2 Stages and scripts

| Stage | Script | Output |
|-------|--------|--------|
| **1. Index + split** | `preprocessing/scan_dataset.py` | `dataset_index.csv`, `artifacts/splits/{train,val}.txt` (90/10 stratified, seed 42) |
| **2. Lexicon** | `preprocessing/build_lexicon.py` | `artifacts/vocab.json` (train only) |
| **3. Header norm** | `preprocessing/fit_header_norm.py` | `artifacts/normalization_header.json` (train only, **aggregated** multi-dex headers) |
| **4. Extract cache** | `preprocessing/extract_to_cache.py` | per-APK `.npz` shards + manifests |

**Order:** split → lexicon + header norm on **train** → extract **train + val** with frozen vocab and norm.

### 3.3 Feature cache (per-APK shards)

```text
artifacts/processed/
  shards/
    train/
      <apk_id>.npz    # header (104,), bow (N+1,), label; optional n_dex
    val/
      <apk_id>.npz
  manifest_train.json
  manifest_val.json
  processed_ids.txt
```

1. `extract_to_cache.py`: for each APK → `extract_apk_raw_header()` + manifest BoW → write shard.
2. Append `apk_id` to `processed_ids.txt` after successful write.
3. Resume: skip ids in `processed_ids.txt` or existing shard.
4. **`tqdm`** on extraction loops.
5. Atomic writes: `*.npz.tmp` then rename.

**Training:** `CombinedPipelineDataset` loads shards via `manifest_{split}.json` — **no APK parsing in the training loop**.

### 3.4 Shell entrypoints

```text
scripts/run_preprocess.sh    # index → lexicon → norm → extract
scripts/run_train.sh         # train with optional --resume
scripts/verify_setup.py      # Phase 1
scripts/verify_model.py      # Phase 4
scripts/verify_dataloader.py # Phase 3
```

---

## 4. Model Architecture (Pattern A)

| Stage | Input | Output |
|--------|--------|--------|
| **Concat** | `H` (104), `I` (N+1) | `X` length **L** |
| **Pad** (if needed) | `X` | length `combined_padded_len` |
| **ASCNN** | `(B, 1, L_pad)` | pooled embedding **128-d** |
| **MLP head** | 128-d | logit → **sigmoid** at inference |

### 4.1 ASCNN stack (over combined `X`)

Mirror paper Fig. 7 (manifest path), applied to **combined** length:

| Layer | Kernel | Stride | Out channels (typical) |
|-------|--------|--------|-------------------------|
| ASU block 1 | 3 | 2 | 64 |
| ASU block 2 | 3 | 2 | 128 |
| ASU block 3 | 3 | 1 | 128 |
| AvgPool1d | — | — | → length 1 |

**ASCU:** `models/adaptive_shrinkage_unit.py` — dynamic conv + soft threshold (paper §3.6, Fig. 8).

### 4.2 Classifier head

- **Linear → BatchNorm1d → ReLU → Linear(1)** on 128-d embedding.
- Loss: **`BCEWithLogitsLoss`** on fusion logit (no AdaSV — single head).
- Inference: **`sigmoid(logit)`**.

### 4.3 Modules

| File | Role |
|------|------|
| `models/adaptive_shrinkage_unit.py` | ASU building block |
| `models/ascnn_combined.py` | ASCNN over padded `X` |
| `models/combined_net.py` | `forward(H, I)` → concat → ASCNN → head |

**No** `mlp_header.py`, **no** `fusion_head.py` (Pattern B only).

### 4.4 Optional later

- `LayerNorm` on `H` and `I` before concat.
- Auxiliary losses / single-modality ablation heads.
- INT8 PTQ (parent plan §5.1) — post-training phase.

---

## 5. Training (Remote)

### 5.1 Hyperparameters (starting point)

MSFDroid §4.2 / `only_base1_model` / Pattern B:

| Parameter | Default |
|-----------|---------|
| Optimizer | **SGD**, momentum **0.9** |
| Learning rate | **0.005** |
| LR decay | ×**0.5** (StepLR) |
| Batch size | **16** |
| Epochs | **80** (tune on val) |
| Loss | `BCEWithLogitsLoss` |
| Imbalance | `training.pos_weight` or `benign_to_malware_ratio` |

### 5.2 `tqdm`

- Outer: epochs (train/val loss, optional val AUC).
- Inner: batches (`leave=False`).

### 5.3 Resume after outage

`artifacts/checkpoints/latest.pt` each epoch:

- `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` (AMP),
- `epoch`, `global_step`, `best_val_loss`,
- RNG: `random`, `numpy`, `torch`, `torch.cuda`.

```bash
python -m src.training.train --resume artifacts/checkpoints/latest.pt
```

### 5.4 DataLoader

- `CombinedPipelineDataset`: `(header_tensor, bow_tensor, label)` or single `combined_tensor`.
- `build_dataloaders_from_config()`: train shuffle, val sequential, `pin_memory=True` if CUDA.

---

## 6. Repository Layout

```text
full_combined_pipeline_approach/
  implementation_plan.md
  patternA_specifics.md              # per-phase implementation log (create in Phase 1)
  requirements.txt
  config/default.yaml
  scripts/
    verify_setup.py
    verify_dataloader.py
    verify_model.py
    run_preprocess.sh
    run_train.sh
    run_pattern_a.sh               # optional: preprocess + train orchestration
    compute_class_balance.py       # Phase 6
    package_artifacts.sh             # Phase 6
  artifacts/                       # gitignored on remote
  src/
    config.py
    constants.py
    features/
      dex_header.py                # per-Dex parse
      multidex.py                  # aggregate classes*.dex → H
      manifest_bow.py
      apk_extract.py               # list/read all Dex entries
      normalization.py
    preprocessing/
      scan_dataset.py
      build_lexicon.py
      fit_header_norm.py
      extract_to_cache.py
      common.py
      labels.py
    models/
      adaptive_shrinkage_unit.py
      ascnn_combined.py
      combined_net.py
    data/
      dataset.py
      dataloaders.py
      store.py
    training/
      checkpoint.py
      train.py
      loops.py
      setup.py
      losses.py
      evaluate.py                  # Phase 6: ACC, F1, AUC on val
  tests/
    test_multidex.py
    test_dex_header.py
    test_manifest_bow.py
    test_combined_net.py
    test_dataset.py
    test_training.py
```

---

## 7. Implementation Phases

### Phase 1 — Workspace and environment

**Goal:** Importable package, config, artifact dirs — no APK I/O yet.

| Step | Action | File(s) |
|------|--------|---------|
| 1.1 | Pin dependencies (`torch`, `numpy`, `pyyaml`, `tqdm`, `scikit-learn`, `pyaxmlparser`) | `requirements.txt` |
| 1.2 | Paths, preprocessing (incl. **`multidex`** block), model dims, training | `config/default.yaml` |
| 1.3 | Load YAML, resolve paths, `ensure_artifact_dirs()` | `src/config.py` |
| 1.4 | `DEX_HEADER_FEATURE_DIM=104`, `DEFAULT_LEXICON_SIZE=4380`, dex pattern constants | `src/constants.py` |
| 1.5 | Smoke test imports + assert `multidex.mode: sum` | `scripts/verify_setup.py` |
| 1.6 | Gitignore `artifacts/`, `data/` | `.gitignore` |
| 1.7 | Start implementation log | `patternA_specifics.md` |

**Done when:** `python scripts/verify_setup.py` exits 0.

---

### Phase 2 — Multi-dex features and preprocessing

**Goal:** Resumable shards with **sum-pooled** `H` and manifest `I` for train/val.

| Step | Action | File(s) |
|------|--------|---------|
| 2.1 | Per-Dex header parse (magic, bytes 8–111) | `src/features/dex_header.py` |
| 2.2 | `aggregate_header_vectors`, `dex_suffix_sort_key`, `multidex_settings` | `src/features/multidex.py` |
| 2.3 | `list_dex_entries`, `read_all_dex_from_apk`, `extract_apk_raw_header` | `src/features/apk_extract.py` |
| 2.4 | Corpus min–max fit/transform | `src/features/normalization.py` |
| 2.5 | Manifest tokenize + multi-hot BoW | `src/features/manifest_bow.py` |
| 2.6 | Walk `apk_root`, labels, 90/10 split | `src/preprocessing/scan_dataset.py`, `labels.py`, `common.py` |
| 2.7 | Build lexicon (train only) | `src/preprocessing/build_lexicon.py` |
| 2.8 | Fit norm on **aggregated** train headers | `src/preprocessing/fit_header_norm.py` |
| 2.9 | Extract shards + `processed_ids` resume + manifests | `src/preprocessing/extract_to_cache.py` |
| 2.10 | End-to-end preprocess script | `scripts/run_preprocess.sh` |
| 2.11 | Unit tests: multidex sum, single-Dex, manifest BoW | `tests/test_multidex.py`, `test_dex_header.py`, `test_manifest_bow.py` |

**Done when:** `manifest_train.json` / `manifest_val.json` exist; sample `.npz` has `header.shape == (104,)`, `bow.shape == (4381,)`; multi-Dex synthetic ZIP passes sum test.

**Porting note:** You may **copy** Phase-7-tested modules from `dual_branch_merge_approach/src/features/` and `preprocessing/` then adjust package imports — behavior should match Pattern B §12.

---

### Phase 3 — PyTorch Dataset and DataLoader

**Goal:** Training loop reads shards only; builds batches for `CombinedNet`.

| Step | Action | File(s) |
|------|--------|---------|
| 3.1 | Load `manifest_{split}.json`, open `.npz` shards | `src/data/store.py` |
| 3.2 | `CombinedPipelineDataset` → `(H, I, label)` float tensors | `src/data/dataset.py` |
| 3.3 | `build_dataloaders_from_config()` — batch 16, no re-split | `src/data/dataloaders.py` |
| 3.4 | Verify on real or synthetic manifests | `scripts/verify_dataloader.py` |
| 3.5 | Dataset tests | `tests/test_dataset.py` |

**Done when:** One batch shape `(B, 104)`, `(B, 4381)`, `(B,)` from cached data.

---

### Phase 4 — Model (combined ASCNN forward pass)

**Goal:** `CombinedNet` runs forward on GPU/CPU; output scalar logit per sample.

| Step | Action | File(s) |
|------|--------|---------|
| 4.1 | Implement ASCU | `src/models/adaptive_shrinkage_unit.py` |
| 4.2 | ASCNN over `combined_padded_len` | `src/models/ascnn_combined.py` |
| 4.3 | `concat(H,I)` → pad → ASCNN → MLP head | `src/models/combined_net.py` |
| 4.4 | Set `combined_padded_len` after shape trace (strides 2,2,1) | `config/default.yaml` |
| 4.5 | Forward smoke (random or dataloader batch) | `scripts/verify_model.py` |
| 4.6 | Unit tests | `tests/test_combined_net.py` |

**Done when:** `verify_model.py` prints OK; parameter count documented in `patternA_specifics.md`.

---

### Phase 5 — Training loop and checkpoints

**Goal:** SGD + BCE, tqdm, resume after power loss.

| Step | Action | File(s) |
|------|--------|---------|
| 5.1 | Checkpoint save/load (full training state) | `src/training/checkpoint.py` |
| 5.2 | Train/val loops, StepLR | `src/training/loops.py`, `setup.py` |
| 5.3 | CLI: `--resume`, `--fresh`, config path | `src/training/train.py` |
| 5.4 | `pos_weight` from config or `class_balance.json` | `src/training/losses.py` |
| 5.5 | Train shell entrypoint | `scripts/run_train.sh` |
| 5.6 | Training + resume tests (small synthetic shards) | `tests/test_training.py` |

**Done when:** Loss decreases on a small subset; `--resume` continues from same epoch.

---

### Phase 6 — Full remote run, metrics, artifact sync

**Goal:** Full ~50k preprocess + train; val metrics; bundle artifacts for local machine.

| Step | Action | File(s) |
|------|--------|---------|
| 6.1 | Full `run_preprocess.sh` on remote (days; resumable) | — |
| 6.2 | `compute_class_balance.py` → set `pos_weight` if needed | `scripts/compute_class_balance.py` |
| 6.3 | Full train → `best.pt`, `latest.pt` | `scripts/run_train.sh` |
| 6.4 | Val **ACC / F1 / AUC** (no separate test set required here) | `src/training/evaluate.py` |
| 6.5 | Package vocab, norm, config snapshot, checkpoints | `scripts/package_artifacts.sh` |
| 6.6 | Optional orchestration | `scripts/run_pattern_a.sh` |

**Done when:** `artifacts/checkpoints/best.pt` trained on multi-dex shards; `evaluate.py` reports val metrics; artifacts copied off remote.

---

### Phase 7 — Multi-dex operations & verification

**Note:** Multi-dex **sum-pool** is default from **Phase 2** (not a late add-on). Phase 7 adds telemetry and ablation docs.

| Step | Action | File(s) |
|------|--------|---------|
| 7.1 | Dex count histogram on train/val manifests | `scripts/compute_dex_stats.py` → `artifacts/dex_stats.json` |
| 7.2 | Ablation: `multidex.mode: primary_only` in config (paper baseline) | `config/default.yaml` |
| 7.3 | Policy tests (sum vs primary_only) | `tests/test_multidex_phase7.py` |

**Done when:** `dex_stats.json` written on remote; `test_multidex_phase7` passes; thesis can cite multi-dex coverage %.

---

### Phase summary table

| Phase | Deliverable | Done when |
|-------|-------------|-----------|
| **1** | Config, deps, verify_setup | imports + dirs OK |
| **2** | Multi-dex preprocess + shards | `manifest_*.json`, sum-pooled `H` in `.npz` |
| **3** | Dataset + DataLoaders | batch `(104, 4381, label)` |
| **4** | `CombinedNet` forward | `verify_model.py` OK |
| **5** | `train.py` + resume | loss ↓ on subset; resume works |
| **6** | Full remote run + eval | `best.pt` + val metrics + `run_pattern_a.sh` |
| **7** | Multi-dex telemetry + policy tests | `dex_stats.json`, ablation documented |

---

## 8. Multi-Dex Policy (Locked)

Reference: [`../../dex_related_instruction.md`](../../dex_related_instruction.md).

| Decision | Choice |
|----------|--------|
| Discovery | All ZIP entries matching `^classes(\d*)\.dex$` on **basename** |
| Aggregation | **`sum`** (default) → **104-d** `H` |
| Shard `header` | Always **`(104,)`** for default mode |
| ASCNN input | **`concat(H, I)`** — header still 104-d; multi-dex affects **H** only |
| Manifest | Unchanged (one BoW per APK) |
| Ablation | `primary_only` (paper baseline), `mean`; `concat` only with explicit `header_dim` change |

### Failure handling

| Case | Behavior |
|------|----------|
| No `classes*.dex` | Log → `failed_apks.log`; skip |
| Any Dex fails magic/size | **Strict v1:** skip entire APK |
| Single-Dex APK | `sum` = that one **Hᵢ** |
| Malware in `classes2.dex` only | Included via sum-pool (fixes paper blind spot) |

### Cache invalidation

If you change `multidex.mode` or re-fit norm:

- Delete `artifacts/processed/shards/`, clear `processed_ids.txt`.
- Re-run `fit_header_norm.py` + `extract_to_cache.py`.
- Retrain checkpoints **`--fresh`** (old weights saw different `H` distribution).

---

## 9. Config Snippet (`config/default.yaml`)

```yaml
paths:
  apk_root: /data/apks              # override on remote
  dataset_index: artifacts/dataset_index.csv
  processed_dir: artifacts/processed
  checkpoint_dir: artifacts/checkpoints

preprocessing:
  label_mode: parent_folder
  benign_names: [benign, goodware, clean, good, "0"]
  malicious_names: [malware, malicious, virus, bad, "1"]
  multidex:
    mode: sum
    dex_pattern: "^classes(\\d*)\\.dex$"
    max_dex: 3                      # concat ablation only
  lexicon_size: 4380
  min_token_freq: 2
  bow_encoding: multihot
  manifest_parser: pyaxmlparser
  cache_format: shard_npz
  train_ratio: 0.9
  seed: 42

model:
  header_dim: 104
  lexicon_size: 4380                # N; bow dim = N+1
  combined_input_len: 4485          # header_dim + (N+1); verify in Phase 4
  combined_padded_len: 4488         # set after conv shape trace
  ascnn_embed_dim: 128
  ascnn_channels: [64, 128, 128]

training:
  batch_size: 16
  learning_rate: 0.005
  momentum: 0.9
  lr_decay_factor: 0.5
  lr_decay_epochs: 30
  epochs: 80
  num_workers: 4
  benign_to_malware_ratio: null
  pos_weight: null
  auto_pos_weight: true
```

---

## 10. Pattern A vs Pattern B

| Topic | Pattern A (this folder) | Pattern B (`dual_branch_merge_approach/`) |
|--------|-------------------------|-------------------------------------------|
| Topology | **One ASCNN** on `concat(H,I)` | **MLP(H)** + **ASCNN(I)** + fusion |
| Multi-dex | Sum-pool → **104-d** `H` (same) | Same |
| Joint mixing | Layer 1 of conv stack | Only after 128+128 embeddings |
| Params / speed | Usually **smaller / faster** | Usually larger / slower |
| Interpretability | Weaker per-modality | Branch embeddings inspectable |
| Feature code | Can port from B | Reference implementation |

**Thesis ablation row:** train both patterns on **same shards** (same `H`, `I`) to isolate fusion architecture.

---

## 11. Resolved Choices

| # | Topic | Decision |
|---|--------|----------|
| 1 | **Labels** | `scan_dataset.py` + `parent_folder`; no external CSV default |
| 2 | **Multi-dex** | **Default `sum`** on all `classes*.dex` per [`dex_related_instruction.md`](../../dex_related_instruction.md) |
| 3 | **BoW** | Multi-hot; **N = 4380**; UNK included |
| 4 | **Manifest parser** | `pyaxmlparser` |
| 5 | **Feature cache** | Per-APK `.npz` shards + `processed_ids.txt` resume |
| 6 | **Concat** | **`[H \|\| I]`** in Dataset/model; separate shard fields |
| 7 | **Class balance** | `pos_weight` / `auto_pos_weight` after Phase 6 counts |
| 8 | **Optimizer** | **SGD** defaults; AdamW optional ablation |
| 9 | **LayerNorm pre-concat** | **Off** in v1; optional experiment |

---

## 12. Artifacts for Remote → Local Sync

- `vocab.json`, `normalization_header.json`
- `processed/shards/`, `manifest_train.json`, `manifest_val.json`
- `checkpoints/best.pt`, `latest.pt`
- `failed_apks.log`, `class_balance.json` (if used)
- Copy of `config/default.yaml` used on remote
- Optional: `dex_stats.json` (histogram of Dex count per APK)

---

## 13. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Paper ignored secondary Dex | **Sum-pool all `classes*.dex`** (this plan) |
| Weak header-only signal (~83% ACC in paper) | Manifest BoW + joint ASCNN; compare Pattern B on same cache |
| Scale mismatch H vs BoW | Monitor val loss; try LayerNorm ablation |
| Very long combined sequence (~4485) | Padding + ASCNN; profile batch time on remote GPU |
| Obfuscated / missing manifest | `failed_apks.log`; report skip rate in thesis |
| Old single-Dex shards | Delete shards + re-preprocess when enabling multi-dex |

---

## 14. Optional Post-v1 Work

- `primary_only` multidex ablation (reproduce paper limitation).
- Pattern A vs B table on identical `artifacts/processed/`.
- INT8 quantization + on-device latency (parent plan §5).
- Formal held-out **test** split when dataset protocol is fixed.
