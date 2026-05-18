# Implementation Plan: Dual-Branch Merge (Pattern B)

**Pattern:** Late fusion — **Branch 1** = Dex header → **MLP(H)**; **Branch 2** = manifest BoW → **ASCNN(I)**; **Fusion** = concatenate branch embeddings → small MLP → malware probability.

**Context:** This is **Pattern B** from [`../detailed_custom_implementation_plan.md`](../detailed_custom_implementation_plan.md). The sibling folder [`../full_combined_pipeline_approach/`](../full_combined_pipeline_approach/) implements **Pattern A** (single tower: `concat(H, BoW)` → one ASCNN). Read the parent doc for *why* entropy/MEM-PSD is dropped and what MSFDroid components we keep.

**Reuse:** Per-Dex header parsing matches [`../../only_base1_model/`](../../only_base1_model/) (104 bytes → 104 floats, min–max normalization). Pattern B **aggregates all `classes*.dex` headers** into one 104-dim `H` by default (Phase 7, §12) — see [`../../dex_related_instruction.md`](../../dex_related_instruction.md). Manifest BoW + ASCNN + fusion sit on top of that header pipeline.

**Training environment:** ~**50,000 APKs** on a **remote PC** (not on your local machine). This document covers **preprocessing → training** only. Formal held-out **test** evaluation is out of scope here; use a **train/val** split for checkpointing and early stopping.

---

## 0. Quick Picture

```
APK
 ├─ classes.dex, classes2.dex, … ──► per-Dex Hᵢ (104 each)
 │         └─ sum-pool ───────────► H (104-dim) ──► MLP(H) ──► e_h  (128-dim)
 └─ AndroidManifest.xml ──────────► BoW I (N+1) ──► ASCNN(I) ──► e_i (128-dim)
                                      │
                                      ▼
                         concat(e_h, e_i) ──► Fusion MLP ──► sigmoid → P(malware)
```

No structural-entropy / MEM-PSD branch (`MLP(M)`).

---

## 1. What Each Part Does (Plain Language)

| Symbol | Meaning |
|--------|---------|
| **H** | Numbers describing the **Dex file header** (sizes/offsets of sections)—cheap to read, no full Dex scan. |
| **I** | **Bag-of-words** vector over manifest **permissions** and **intent** strings (fixed vocabulary + UNK). |
| **Lexicon size N** | How many manifest keywords you keep (paper: **4380**); vector length = **N+1** with UNK. |
| **MLP(H)** | Small fully connected network on header only (paper Base Model 1 style). |
| **ASCNN(I)** | Convolution stack with **adaptive shrinkage** on the sparse BoW (paper Base Model 3 / Fig. 7–8). |
| **Fusion** | Learns how to combine “structure” (header) and “behavior hints” (manifest). |

**Why two branches instead of one combined ASCNN (Pattern A)?** Easier to inspect each stream, optional staged use (header-only cheap pass), and matches the Gemini / late-fusion design. Trade-off: usually **more parameters** than Pattern A.

---

## 2. Feature Extraction

### 2.1 Dex header branch (`H`)

Align with parent plan §3.1, `only_base1_model`, and [`../../dex_related_instruction.md`](../../dex_related_instruction.md). **Target behavior (Phase 7, §12):**

1. Open APK (ZIP); discover **all** entries matching `classes(\d*)\.dex` (`classes.dex`, `classes2.dex`, …); sort `classes.dex` first, then numeric suffix order.
2. For **each** Dex: **magic** check (`dex\n` + version + `\0`); encode bytes **8–111** as floats in `[0, 1]` via `/255.0` → vector **Hᵢ** (104-d).
3. **Default aggregation:** **sum** of all **Hᵢ** → single **H** (104-d). Single-Dex APKs: one vector, identical to sum of one file. Corpus min–max follows to rescale.
4. **Corpus min–max** normalization on aggregated **H** (training split); save stats to `artifacts/normalization_header.json`.
5. Final **`d_h = 104`** (unchanged model input).

**Implemented in Phase 7** — see `patternB_specifics.md`.

**Failures:** No matching Dex, or any Dex fails validation → log to `artifacts/failed_apks.log`, skip sample (do not invent a label).

### 2.2 Manifest branch (`I`)

Align with parent plan §3.2:

1. Decode **`AndroidManifest.xml`** (binary XML → XML or element tree).
2. Collect tokens:
   - full **permission** names (`android.permission.*`, custom permissions),
   - **intent** keywords: `action`, `category`, and related string attributes used in the paper’s “intent keywords.”
3. **Lexicon (training APKs only):**
   - Count token frequencies across the **train split**.
   - Drop tokens with count **&lt; 2**.
   - Keep top **N** by frequency (default **N = 4380**).
   - Map unknown tokens at inference to **UNK** index.
4. **BoW encoding:** binary **multi-hot** (1 if token present, else 0); see `bow_encoding: multihot` in config.
5. Output vector **`I`** of shape **`(N+1,)`** as `float32`.

Save `artifacts/vocab.json` (`token → index`, `N`, `unk_index`).

### 2.3 Explicitly omitted

- 256-byte Shannon entropy over full Dex, Burg PSD, `MLP(M)` — not in this pipeline.

---

## 3. Preprocessing Pipeline (~50k APKs)

Preprocessing is **CPU-heavy** and run **once** (or resumed) on the remote machine. Training should **only load cached tensors**, not re-parse APKs every epoch.

### 3.1 Inputs and labeling

APKs live under **`apk_root`** in class folders, e.g. `benign/*.apk` and `malware/*.apk` (names configurable).

**`scan_dataset.py` is the label source of truth:** it walks the directory tree and assigns labels from **parent folder names** (`benign` → `0`, `malware` → `1`). It does **not** read an external CSV for labels. Any existing label CSV is ignored unless you change config.

The script **writes** a normalized index for downstream steps:

| column | description |
|--------|-------------|
| `path` | path to `.apk` (relative to `apk_root` or absolute) |
| `label` | `0` = benign, `1` = malware |
| `apk_id` | optional stable id (e.g. SHA-256) for cache filenames |

Output: `artifacts/dataset_index.csv`. Optional: **SHA-256** per file for deduplication and stable shard keys.

### 3.2 Stages and scripts

| Stage | Script | Output |
|-------|--------|--------|
| **1. Index** | `preprocessing/scan_dataset.py` | `artifacts/dataset_index.csv`, optional `artifacts/splits/{train,val}.txt` |
| **2. Split** | same or `make_splits.py` | 90% train / 10% val, **stratified** by label, fixed seed |
| **3. Lexicon** | `preprocessing/build_lexicon.py` | `artifacts/vocab.json` (train split only) |
| **4. Header stats** | `preprocessing/fit_header_norm.py` | `artifacts/normalization_header.json` (train split only) |
| **5. Extract cache** | `preprocessing/extract_to_cache.py` | per-APK or sharded feature store (see §3.3) |

**Order:** split → lexicon + header norm on **train** → extract **train + val** using frozen vocab and norm.

### 3.3 Feature cache (per-APK shards)

**Feature cache** = saved **`H`**, **`I`**, and **`label`** per APK after extraction.

- **Why:** 50k × many epochs would re-unzip and re-parse manifests without cache; training should load tensors only.
- **Format (chosen):** one **shard file per APK**, not one giant `.pt` per split.

Layout:

```text
artifacts/processed/
  shards/
    train/
      <apk_id>.npz    # H (104,), I (N+1,), label (scalar)
    val/
      <apk_id>.npz
  manifest_train.json   # list of shard paths + labels (built after scan)
  manifest_val.json
  processed_ids.txt       # apk_ids already extracted (append-only log)
```

**How it works:**

1. **`extract_to_cache.py`** loops over `dataset_index.csv` for the current split.
2. For each APK: unzip → compute `H`, `I` → write `shards/{split}/{apk_id}.npz`.
3. Append `apk_id` to `processed_ids.txt` only after a successful write.
4. On **interrupt or crash**, re-run the same command: skip any `apk_id` already in `processed_ids.txt` or with an existing shard file.
5. When extraction finishes (or incrementally), refresh `manifest_{split}.json` so `DualBranchDataset` can load shards without scanning 50k files every epoch.

**Training load:** `src/data/store.py` reads `manifest_{split}.json`, opens only the shard paths for the current batch (optionally memory-map `.npz`). No APK parsing during training.

Optional later: `scripts/merge_shards_to_pt.py` to pack shards into monolithic `.pt` for faster I/O on a machine with enough RAM.

### 3.4 Resume during extraction

- **`processed_ids.txt`** + existing shard files define “done”; idempotent re-runs are safe.
- **`tqdm`** over remaining APKs in `extract_to_cache.py`.
- Partially written shards: write to `*.npz.tmp` then rename, or delete corrupt shard and re-extract that id.

### 3.5 Shell entrypoints

```text
scripts/run_preprocess.sh   # index → lexicon → norm → extract
scripts/run_train.sh        # train with optional --resume
```

---

## 4. Model Architecture (Pattern B)

### 4.1 Branch 1 — `MLP(H)`

Match `only_base1_model` / paper Fig. 3 style:

- Input: **`104`**
- Two blocks: **Linear → BatchNorm1d → ReLU**
- Hidden dim: **`128`** (configurable)
- Output embedding **`e_h`**: **`128`** (take last hidden layer **before** any branch-specific sigmoid; the **fusion head** owns the final logit)

Do **not** attach a standalone training-only sigmoid on this branch unless you add an auxiliary loss; v1 uses **one BCE** on the fusion output only.

### 4.2 Branch 2 — `ASCNN(I)`

Input: BoW reshaped to **`(batch, 1, N+1)`** for `Conv1d`.

Stack (paper Fig. 7, manifest-only path):

| Layer | Kernel | Stride | Out channels (typical) |
|-------|--------|--------|-------------------------|
| ASU block 1 | 3 | 2 | 64 |
| ASU block 2 | 3 | 2 | 128 |
| ASU block 3 | 3 | 1 | 128 |
| AvgPool1d | — | — | length → 1 |

**ASU (adaptive shrinkage unit):** dynamic conv weights + per-sample soft threshold (parent plan §3.6, paper Fig. 8). Implement in `models/adaptive_shrinkage_unit.py`.

Output embedding **`e_i`**: **`128`**.

If `N+1` does not divide cleanly through strides, **pad** BoW to `bow_padded_len` in config (fixed at train and inference).

### 4.3 Fusion head

- Input: **`concat(e_h, e_i)`** → dim **`256`**
- **Linear → BN → ReLU → Linear(1)** → logit
- Loss: **`BCEWithLogitsLoss`**
- Inference: **`sigmoid(logit)`** = malware probability

### 4.4 Optional later

- Auxiliary BCE on each branch (multi-task) for debugging.
- Simplified **two-branch soft voting** (paper AdaSV) — not required for v1.

---

## 5. Training (Remote)

### 5.1 Hyperparameters (starting point)

From MSFDroid §4.2 / `only_base1_model` config:

| Parameter | Default |
|-----------|---------|
| Optimizer | SGD, momentum **0.9** |
| Learning rate | **0.005** |
| LR decay | multiply by **0.5** (StepLR or manual) |
| Batch size | **16** |
| Epochs | e.g. **50–100** (tune on val loss) |
| Loss | BCEWithLogits on fusion logit |
| Class imbalance | `pos_weight` from `training.pos_weight` (set manually) or derived from `training.benign_to_malware_ratio` when you know the split counts |

### 5.2 `tqdm`

- Outer bar: **epochs** (train loss, val loss, val AUC optional).
- Inner bar: **batches** per epoch (`leave=False` on inner).

### 5.3 Resume after power loss

Save **`artifacts/checkpoints/latest.pt`** each epoch (and optionally every N minutes) containing:

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict` (if used)
- `scaler_state_dict` (if AMP)
- `epoch`, `global_step`, `best_val_loss`
- RNG: `random`, `numpy`, `torch`, `torch.cuda`

CLI: `python -m src.training.train --resume artifacts/checkpoints/latest.pt`

### 5.4 DataLoader

- `src/data/dataset.py`: returns `(header_tensor, bow_tensor, label)`.
- `src/data/dataloaders.py`: train/val loaders, `pin_memory=True` if CUDA.
- Shuffle train; no shuffle val.

---

## 6. Repository Layout

```text
dual_branch_merge_approach/
  implementation_plan.md          # this file
  requirements.txt                  # torch, numpy, pyyaml, tqdm, scikit-learn, pyaxmlparser (manifest)
  config/default.yaml
  scripts/
    verify_setup.py
    run_preprocess.sh
    run_train.sh
  artifacts/                        # gitignored on remote
    vocab.json
    normalization_header.json
    processed/
    checkpoints/
    failed_apks.log
  src/
    config.py
    constants.py
    features/
      dex_header.py                 # per-Dex parse (only_base1_model logic)
      multidex.py                   # Phase 7: aggregate classes*.dex → H
      manifest_bow.py
      apk_extract.py                # Phase 7: list/read all Dex entries
      normalization.py
    preprocessing/
      scan_dataset.py
      build_lexicon.py
      fit_header_norm.py
      extract_to_cache.py
    models/
      adaptive_shrinkage_unit.py
      mlp_header.py
      ascnn_manifest.py
      fusion_head.py
      dual_branch_net.py
    data/
      dataset.py
      dataloaders.py
      store.py
    training/
      checkpoint.py
      train.py
      losses.py
```

---

## 7. Implementation Phases (Suggested Order)

| Phase | Deliverable | Done when |
|-------|-------------|-----------|
| **1** | `requirements.txt`, `config/default.yaml`, `verify_setup.py` | imports + dirs OK |
| **2** | `dex_header.py`, `manifest_bow.py`, `preprocess` scripts | train/val shards + `manifest_*.json` on remote |
| **3** | `DualBranchDataset`, DataLoaders | batch loads `(104, N+1, label)` |
| **4** | `dual_branch_net.py` forward pass | dummy batch runs on GPU/CPU |
| **5** | `train.py` + checkpoint + **tqdm** + **--resume** | loss decreases on small subset |
| **6** | Full 50k preprocess + full train | `best.pt` / `latest.pt` on remote |
| **7** | **`multiple_dex_handling`** (§12) — **default pipeline** | sum-pooled `H` from all `classes*.dex`; full re-preprocess + retrain; §12 checklist complete |

---

## 12. Phase: `multiple_dex_handling`

**Status:** Implemented (see `patternB_specifics.md` Phase 7). **Multi-dex is the default** — no separate “enable” path required for production runs.  
**Prerequisite:** Phases 1–6 complete.  
**Policy (locked for Pattern B):**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Discovery | All ZIP entries matching `^classes(\d*)\.dex$` on **basename** | Covers `classes.dex`, `classes2.dex`, … per [`dex_related_instruction.md`](../../dex_related_instruction.md) |
| Aggregation | **`sum`** (default `preprocessing.multidex.mode`) | Keeps `d_h = 104`; emphasizes total structural footprint across Dex files; min–max rescales; no MLP/dataset edits |
| Model / shards | `header` still `(104,)` in `.npz` | Seamless single- + multi-Dex through same `DualBranchNet` |
| Ablation only | `mode: primary_only`, `mode: mean` | `primary_only` = old `classes.dex`-only; `mean` = average instead of sum |

Manifest branch (`I`) is **unchanged** — one `AndroidManifest.xml` per APK.

---

### 12.1 Default behavior (target)

```
APK (zipfile)
  ├─ list entries → [classes.dex, classes2.dex, …]   # sorted
  ├─ for each: extract_header_features(bytes) → Hᵢ     # 104-d, /255
  ├─ H_raw = sum(H₀, H₁, …)                           # 104-d
  ├─ H = minmax(H_raw, corpus stats)                  # same as today
  └─ I = manifest BoW                                 # unchanged
```

---

### 12.2 Complete change list (Pattern B)

Every row is required unless marked **optional**. Implement in order §12.5.

#### A. Feature layer (core)

| # | File | Change |
|---|------|--------|
| A1 | `src/features/apk_extract.py` | **Rewrite module docstring** — “all `classes*.dex`”, not only `classes.dex`. |
| A2 | `src/features/apk_extract.py` | Add `_dex_basename(name: str) -> str` — strip directory prefix from ZIP entry names. |
| A3 | `src/features/apk_extract.py` | Add `list_dex_entries(zf: zipfile.ZipFile, *, pattern: str) -> list[str]` — match basename against config regex; return full ZIP paths sorted (`classes.dex` suffix `""` → 0, `"2"` → 2, …). |
| A4 | `src/features/apk_extract.py` | Add `read_all_dex_from_apk(apk_path: Path, *, pattern: str) -> list[tuple[str, bytes]]` — open ZIP once; read each matched entry; raise `ApkExtractError` if zero matches. |
| A5 | `src/features/apk_extract.py` | Add `extract_apk_raw_header(apk_path: Path, *, mode: str, pattern: str, max_dex: int) -> np.ndarray` — orchestrates A4 + aggregation (calls into multidex/dex_header); single public entry for preprocessing. |
| A6 | `src/features/apk_extract.py` | Keep `read_classes_dex()` as thin wrapper: `read_all_dex…` with `mode=primary_only` or first entry only — used by tests / ablation. |
| A7 | `src/features/multidex.py` | **New file.** `aggregate_header_vectors(vectors: list[np.ndarray], mode: str, *, max_dex: int) -> np.ndarray` — implement `sum` (default), `mean`, `primary_only` (first vector only), `concat` (optional; see §12.4). |
| A8 | `src/features/multidex.py` | `dex_suffix_sort_key(basename: str) -> tuple` — stable ordering helper. |
| A9 | `src/features/dex_header.py` | Update module docstring — per-Dex parse unchanged; aggregation lives in `multidex.py`. |
| A10 | `src/features/dex_header.py` | Add `extract_headers_from_dex_list(dex_bytes_list: list[bytes]) -> list[np.ndarray]` — map `extract_header_features` over list; fail fast on first `DexHeaderError`. |
| A11 | `src/features/__init__.py` | Export `extract_apk_raw_header`, `list_dex_entries`, `aggregate_header_vectors` (or names chosen in A5/A7). |

#### B. Preprocessing

| # | File | Change |
|---|------|--------|
| B1 | `src/preprocessing/extract_to_cache.py` | Remove `dex_entry` parameter from `_extract_features`; call `extract_apk_raw_header(row.apk_path, mode=…, pattern=…)` instead of `read_classes_dex` + single `extract_header_features`. |
| B2 | `src/preprocessing/extract_to_cache.py` | In `extract_split()`, read `pre["multidex"]` (not `dex_entry_name`) for mode/pattern/max_dex. |
| B3 | `src/preprocessing/extract_to_cache.py` | **Optional:** `np.savez_compressed(..., n_dex=np.int32(len(dex_list)))` for debugging. |
| B4 | `src/preprocessing/fit_header_norm.py` | Same as B1–B2: loop train APKs via `extract_apk_raw_header`; stack aggregated raw vectors before `fit_minmax_stats`. |
| B5 | `src/preprocessing/common.py` | **Optional:** extend `write_shard_manifest()` metadata with `"multidex_mode": "sum"` so loaders know how shards were built. |

**No changes:** `scan_dataset.py`, `build_lexicon.py`, `labels.py` (manifest / indexing unaffected).

#### C. Configuration

| # | File | Change |
|---|------|--------|
| C1 | `config/default.yaml` | **Remove** `dex_entry_name: classes.dex` as primary path (or comment “ablation only”). |
| C2 | `config/default.yaml` | Set default block: `multidex.mode: sum`, `multidex.dex_pattern: "^classes(\\d*)\\.dex$"`, `multidex.max_dex: 3` (for concat ablation only). |
| C3 | `config/default.yaml` | **Optional:** `paths.processed_dir: artifacts/processed_multidex` (or `cache_version: 2`) to avoid mixing old shards. |
| C4 | `src/config.py` | **Optional:** add `def multidex_settings(cfg) -> dict` helper; not required if preprocessing reads `cfg.preprocessing["multidex"]` directly. |

#### D. Model, data, training (no code changes for default `sum`)

| # | File | Change |
|---|------|--------|
| D1 | `src/models/mlp_header.py` | **No change** — `input_dim=104` stays valid. |
| D2 | `src/models/dual_branch_net.py` | **No change**. |
| D3 | `src/data/dataset.py`, `store.py`, `dataloaders.py` | **No change** — still load `header` shape 104. |
| D4 | `src/training/*` | **No change** — retrains on new shards only. |
| D5 | `src/constants.py` | **No change** — `DEX_HEADER_FEATURE_DIM = 104`. |

**Only if `multidex.mode: concat` (ablation, not default):** update `model.header_dim`, `MLPHeaderBranch`, manifests’ `header_dim`, and tests — see §12.4.

#### E. Tests

| # | File | Change |
|---|------|--------|
| E1 | `tests/test_multidex.py` | **New.** Synthetic ZIP: `classes.dex` + `classes2.dex`; assert `sum` equals hand-computed element-wise sum; assert single-Dex APK matches one-vector sum; assert sort order; assert zero Dex → error. |
| E2 | `tests/test_dex_header.py` | Keep per-Dex tests; add import of `aggregate_header_vectors` sum case. |
| E3 | `tests/test_dataset.py` | **Optional:** one shard built with multi-Dex mock path (if E1 covers aggregation, may skip). |
| E4 | `tests/test_phase6_pipeline.py` | Re-run smoke after config/shard path change; no logic change if dims stay 104. |
| E5 | `scripts/verify_setup.py` | Import `multidex` module in smoke check. |

#### F. Scripts & orchestration

| # | File | Change |
|---|------|--------|
| F1 | `scripts/run_preprocess.sh` | Document: **delete** old `artifacts/processed/shards/` (or use new `processed_dir`) before full re-extract; norm must run before extract. |
| F2 | `run_pattern_b.sh` | Same cache-bust note; ensure preprocess stage does not `SKIP_PREPROCESS` with stale single-Dex shards. |
| F3 | `scripts/package_artifacts.sh` | **Optional:** include `multidex_mode` in copied config snapshot. |

#### G. Documentation (keep plan + runbook in sync)

| # | File | Change |
|---|------|--------|
| G1 | `implementation_plan.md` | §0, §2.1, §8, §11 — already target multi-dex default; update after code lands. |
| G2 | `patternB_specifics.md` | Add **Phase 7** log: files touched, `sum` default, re-preprocess commands. |
| G3 | `howToRun_dualBranchThenMerge.md` | Update Dex header step text: “all `classes*.dex`, sum-pooled to 104-d”. |

---

### 12.3 New public API (summary)

```python
# src/features/apk_extract.py
list_dex_entries(zf, *, pattern: str) -> list[str]
read_all_dex_from_apk(apk_path, *, pattern: str) -> list[tuple[str, bytes]]
extract_apk_raw_header(apk_path, *, mode: str, pattern: str, max_dex: int) -> np.ndarray  # (104,)

# src/features/multidex.py
aggregate_header_vectors(vectors: list[np.ndarray], mode: str, *, max_dex: int) -> np.ndarray
```

Preprocessing calls **only** `extract_apk_raw_header()` then existing `transform_minmax()`.

---

### 12.4 Config (default after Phase 7)

```yaml
preprocessing:
  multidex:
    mode: sum                         # default; production
    dex_pattern: "^classes(\\d*)\\.dex$"
    max_dex: 3                        # concat ablation only
  # Ablation — not default:
  # multidex.mode: primary_only      # old classes.dex-only behavior
  # multidex.mode: mean              # average instead of sum
  # multidex.mode: concat            # header_dim → 104 * max_dex; requires model changes (D5-alt)
```

---

### 12.5 Implementation order

| Step | Action | Validates |
|------|--------|-----------|
| 1 | A7, A8, A10 — `multidex.py` + dex_header list helper | unit: aggregation math |
| 2 | A2–A6, A11 — `apk_extract.py` discovery + `extract_apk_raw_header` | unit: ZIP listing |
| 3 | C1–C2 — `config/default.yaml` defaults | config load |
| 4 | B4 — `fit_header_norm.py` | new `normalization_header.json` |
| 5 | B1–B3 — `extract_to_cache.py` | shards on multi-Dex APKs |
| 6 | E1–E5 — tests + `verify_setup` | CI / local |
| 7 | F1–F2 — full re-preprocess on remote (clear old shards) | shard count ≈ index |
| 8 | Retrain Phase 5–6 (`run_pattern_b.sh` / `run_train.sh`) | `best.pt` on multi-dex features |
| 9 | G2–G3 — docs | thesis reproducibility |

---

### 12.6 Cache, artifacts, and remote re-run

| Artifact | Action |
|----------|--------|
| `artifacts/processed/shards/{train,val}/*.npz` | **Delete or new directory** — old shards used single-Dex `H`; incompatible. |
| `artifacts/processed/processed_ids.txt` | Clear or new file alongside new shard dir. |
| `artifacts/normalization_header.json` | **Recompute** on train split (aggregated headers differ). |
| `artifacts/processed/manifest_{train,val}.json` | Regenerated by extract. |
| `artifacts/checkpoints/*.pt` | **Retrain from scratch** (`--fresh`) — weights trained on old `H` distribution. |
| `artifacts/vocab.json` | **Unchanged** (manifest-only). |
| `artifacts/failed_apks.log` | May grow slightly (APKs with no `classes*.dex`). |

**Optional telemetry:** `artifacts/dex_stats.json` — per-split histogram of `n_dex` per APK.

---

### 12.7 Failure handling

| Case | Behavior |
|------|----------|
| No `classes*.dex` in APK | Log → `failed_apks.log`; skip |
| Any Dex fails magic / size check | **Strict (v1):** skip entire APK |
| Single-Dex APK | One vector → sum = that vector (same numeric result as primary-only for one file) |
| Entry in `assets/` or subdirs named `classes2.dex` | Match on **basename** only if under APK root; ignore `foo/classes.dex` unless you widen pattern later |

---

### 12.8 Files explicitly unchanged

`src/features/manifest_bow.py`, `normalization.py` (API unchanged), `src/preprocessing/scan_dataset.py`, `build_lexicon.py`, `src/models/*` (default sum), `src/data/*`, `src/training/*`, `requirements.txt`.

---

### 12.9 Done when

- [x] Default config uses `multidex.mode: sum` (no `dex_entry_name`-only path in normal runs).
- [x] `extract_apk_raw_header` used in both `fit_header_norm.py` and `extract_to_cache.py`.
- [x] Multi-Dex test APK (synthetic) produces same 104-d `H` as manual element-wise sum of per-Dex vectors.
- [x] Single-Dex APK sum equals one-vector path (unit tests).
- [ ] Full remote re-preprocess + retrain completed (required before using new checkpoints).
- [x] Inference path: any APK with 1 or N Dex files → one malware probability via unchanged `DualBranchNet`.

---

## 8. Config Snippet (`config/default.yaml`)

```yaml
paths:
  apk_root: /data/apks              # override on remote; expect benign/ and malware/ subdirs
  dataset_index: artifacts/dataset_index.csv   # written by scan_dataset.py
  processed_dir: artifacts/processed
  checkpoint_dir: artifacts/checkpoints

preprocessing:
  label_mode: parent_folder         # scan_dataset walks apk_root; do not trust external CSV
  benign_names: [benign, goodware, clean, good, "0"]
  malicious_names: [malware, malicious, virus, bad, "1"]
  multidex:
    mode: sum                       # default: sum-pool all classes*.dex → 104-d H
    dex_pattern: "^classes(\\d*)\\.dex$"
    max_dex: 3                      # concat ablation only
  lexicon_size: 4380
  min_token_freq: 2
  bow_encoding: multihot
  manifest_parser: pyaxmlparser     # lightweight binary XML; permissions + intents only
  cache_format: shard_npz           # one .npz per APK under processed/shards/
  train_ratio: 0.9
  seed: 42

model:
  header_dim: 104
  hidden_dim: 128
  ascnn_embed_dim: 128
  bow_padded_len: 4381              # N+1 or padded

training:
  batch_size: 16
  learning_rate: 0.005
  momentum: 0.9
  lr_decay_factor: 0.5
  epochs: 80
  num_workers: 4
  # Set after you know class counts on the train split (see §11):
  benign_to_malware_ratio: null     # e.g. 1.0 = equal counts; used to set pos_weight if pos_weight is null
  pos_weight: null                    # BCE pos_weight for malware class; overrides ratio when set
```

---

## 9. Artifacts to Bring Back From Remote

- `vocab.json`, `normalization_header.json`
- `processed/shards/`, `manifest_train.json`, `manifest_val.json`
- `checkpoints/best.pt`, `latest.pt`
- `failed_apks.log`, training log / config copy used

---

## 10. Differences vs Pattern A (Same Parent Plan)

| | **This folder (B)** | **`full_combined_pipeline_approach` (A)** |
|---|---------------------|-------------------------------------------|
| Header | Dedicated **MLP** | Mixed into **one ASCNN** input with BoW |
| Manifest | Dedicated **ASCNN** | Same single ASCNN |
| Fusion | After **128+128** embeddings | N/A (single tower) |
| Params / speed | Usually larger / slower | Usually smaller / faster |

---

## 11. Resolved Choices

| # | Topic | Decision |
|---|--------|----------|
| 1 | **Labels** | **`scan_dataset.py` walks `apk_root`** and infers labels from folder names (`benign/` vs `malware/`). Writes `dataset_index.csv` for downstream steps. External label CSVs are **not** used as source of truth. Align with `only_base1_model` `label_mode: parent_folder`. |
| 2 | **Class balance** | Deferred. Config exposes **`training.benign_to_malware_ratio`** and **`training.pos_weight`** (both default `null`). After counting the train split, set one of them for `BCEWithLogitsLoss` (e.g. `pos_weight = n_benign / n_malware`). |
| 3 | **BoW** | **Multi-hot** (`bow_encoding: multihot`). |
| 4 | **Manifest parser** | **`pyaxmlparser`** (implemented; replaces legacy `axmlparserpy` which is Python 2–only). Lightweight binary `AndroidManifest.xml` → permissions + intents for BoW. |
| 5 | **Feature cache** | **Per-APK `.npz` shards** under `artifacts/processed/shards/{train,val}/`, plus `processed_ids.txt` for resume. See §3.3–3.4. |
| 6 | **Multi-dex** | **Default after Phase 7:** discover all `classes*.dex`, **sum-pool** to 104-d `H`. Phases 1–6 code still primary-Dex-only until §12 ships. Ablation: `primary_only`, `mean`. See [`../../dex_related_instruction.md`](../../dex_related_instruction.md). |

Implementation can proceed with these defaults; override only via `config/default.yaml`.
