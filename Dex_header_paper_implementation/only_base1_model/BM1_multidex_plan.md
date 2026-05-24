# Base Model 1 — Multi-Dex Handling Plan

**Status:** Not implemented (plan only)  
**Date:** 2026-05-24  
**Guideline:** [`../dex_related_instruction.md`](../dex_related_instruction.md)  
**Reference implementation:** [`../custom_approach/full_combined_pipeline_approach/`](../custom_approach/full_combined_pipeline_approach/) (Phase 7 / `multidex.py`, `apk_extract.py`)

---

## Current state (audit)

Multi-dex APK handling has **not** been done in Base Model 1. The pipeline matches the MSFDroid paper’s limitation: **primary `classes.dex` only**.

| Area | Current behavior | Guideline requirement |
|------|------------------|------------------------|
| APK discovery | `read_classes_dex()` reads a single entry (`classes.dex`) | Discover **all** entries matching `classes.*\.dex` |
| Config | `preprocessing.dex_entry_name: classes.dex` | Configurable multidex mode + pattern |
| Feature extraction | One Dex header → 104-d vector | Parse **every** Dex header, then aggregate |
| Model / dataset | `feature_dim = 104` | Unchanged if using **sum** or **mean** pooling |
| Tests | Single-Dex synthetic header only | Need multi-Dex ZIP fixtures |
| Docs | `only_basemodel_1_specifics.md` documents primary-Dex-only | Must document multidex path |

**Evidence in code:**

- `src/preprocessing/apk_extract.py` — docstring: *"Matches primary Dex only (no classes2.dex)"*
- `config/default.yaml` — `dex_entry_name: classes.dex # primary Dex only`
- `src/preprocessing/preprocess_apks.py` — calls `read_classes_dex()` once per APK
- No `src/features/multidex.py` module exists under `only_base1_model/`

**Risk if left as-is:** Modern APKs often place substantial code in `classes2.dex`, `classes3.dex`, etc. Malware can hide payloads in secondary Dex files while `classes.dex` holds only the multidex loader — the scanner would miss that signal entirely.

---

## Design decision for BM1

Per [`dex_related_instruction.md`](../dex_related_instruction.md), three aggregation strategies are listed. For Base Model 1 (MLP on a fixed 104-d header vector), the recommended default is:

| Mode | Output dim | Model change? | Use |
|------|------------|---------------|-----|
| **`sum`** (default) | 104 | **No** | Holistic footprint across all Dex files; aligns with Pattern A / dual-branch Phase 7 |
| `mean` | 104 | No | Ablation: average instead of sum |
| `primary_only` | 104 | No | Ablation: reproduce current paper-style baseline |
| `concat` | `104 × max_dex` | **Yes** (MLP `input_dim`, tests, checkpoints) | Optional later ablation only |

**Default pipeline:** discover all `classes*.dex` → per-Dex 104-d raw header vector → **element-wise sum** → existing corpus min-max → MLP(H). No changes to Phases 3–6 (dataset, model, training, eval) when `mode: sum` or `mode: mean`.

---

## Phase 1 — Multidex module & config

**Goal:** Add aggregation logic and YAML knobs without touching the training loop yet.

### Tasks

1. **Create** `src/features/multidex.py` (adapt from custom approach):
   - `multidex_settings(preprocessing) -> dict` — resolve `mode`, `dex_pattern`, `max_dex`
   - `dex_suffix_sort_key(basename)` — sort `classes.dex` → `classes2.dex` → … → `classes10.dex`
   - `aggregate_header_vectors(vectors, mode, *, max_dex)` — implement `sum`, `mean`, `primary_only`, `concat`
   - `MultidexError` for invalid config / empty vector list

2. **Update** `config/default.yaml`:
   ```yaml
   preprocessing:
     multidex:
       mode: sum
       dex_pattern: "^classes(\\d*)\\.dex$"
       max_dex: 3
     # dex_entry_name: classes.dex   # deprecated; use multidex.mode: primary_only for ablation
   ```

3. **Update** `src/features/__init__.py` (if present) to export multidex helpers.

4. **Optional:** `src/config.py` helper `multidex_settings(cfg)` mirroring custom approach.

### Verification

- Import smoke: `from src.features.multidex import aggregate_header_vectors`
- Unit test aggregation math in isolation (Phase 5)

### Files touched

| File | Action |
|------|--------|
| `src/features/multidex.py` | **New** |
| `config/default.yaml` | Add `multidex` block; comment out primary-only default |
| `src/config.py` | Optional helper |

---

## Phase 2 — APK extract: discover all Dex entries

**Goal:** Replace single-entry read with full APK Dex discovery.

### Tasks

1. **Extend** `src/preprocessing/apk_extract.py`:
   - `_dex_basename(zip_entry_name)` — normalize path separators
   - `list_dex_entries(zf, *, pattern)` — regex match on **basename**; sorted list of ZIP paths
   - `read_all_dex_from_apk(apk_path, *, pattern)` — `list[tuple[str, bytes]]`; error if zero matches
   - `extract_apk_raw_header(apk_path, *, mode, pattern, max_dex)` — orchestrate read → parse each header → aggregate

2. **Keep** `read_classes_dex()` as a thin legacy wrapper (single entry / ablation) — do not remove; tests may depend on it.

3. **Parsing loop** inside `extract_apk_raw_header`:
   ```text
   for each (name, dex_bytes) in sorted dex_list:
       parse_dex_header_fields(dex_bytes)   # validate
       vectors.append(extract_header_features(dex_bytes))
   return aggregate_header_vectors(vectors, mode, max_dex=max_dex)
   ```

### Edge cases

| Case | Behavior |
|------|----------|
| No `classes*.dex` in APK | `ApkExtractError` → log to `failed_apks.log`, skip sample |
| One Dex only (`classes.dex`) | Sum/mean equals that single vector |
| Invalid secondary Dex | Fail entire APK (consistent with current strict parsing) |
| `classes10.dex` vs `classes2.dex` | Numeric suffix sort, not lexicographic |
| Nested paths (`foo/classes2.dex`) | Match on basename only (same as custom approach) |

### Verification

- Manual: synthetic ZIP with `classes.dex` + `classes2.dex` returns sum of two known vectors

### Files touched

| File | Action |
|------|--------|
| `src/preprocessing/apk_extract.py` | **Rewrite / expand** |

---

## Phase 3 — Wire preprocessing pipeline

**Goal:** Batch job uses multidex extraction instead of `read_classes_dex()`.

### Tasks

1. **Update** `src/preprocessing/preprocess_apks.py`:
   - Read `multidex_settings(pre)` instead of `dex_entry_name`
   - Replace:
     ```python
     dex_bytes = read_classes_dex(apk_path, entry_name=dex_entry_name)
     parse_dex_header_fields(dex_bytes)
     vector = extract_header_features(dex_bytes)
     ```
     with:
     ```python
     vector = extract_apk_raw_header(
         apk_path,
         mode=md["mode"],
         pattern=md["dex_pattern"],
         max_dex=md["max_dex"],
     )
     ```
   - If `mode == "concat"`: set `FEATURE_DIM` dynamically to `104 * max_dex` before save (see Phase 4 note)

2. **Update** CLI description and `scripts/run_preprocess.sh` comments.

3. **Backward compatibility:** Support legacy config that only has `dex_entry_name` — map to `multidex.mode: primary_only` with a deprecation warning in stdout.

### Output artifacts (unchanged paths, richer metadata)

| Path | Change |
|------|--------|
| `artifacts/processed/dex_header_features.pt` | Same schema; `feature_dim` still 104 for sum/mean/primary_only |
| `artifacts/normalization.json` | Add `"multidex_mode": "sum"` |
| `artifacts/failed_apks.log` | May grow (APKs with no Dex matches) |

### Verification

```bash
PYTHONPATH=. python -m src.preprocessing.preprocess_apks --limit 10
# Confirm feature_dim: 104 in summary
```

### Files touched

| File | Action |
|------|--------|
| `src/preprocessing/preprocess_apks.py` | Use `extract_apk_raw_header` |
| `scripts/run_preprocess.sh` | Comment update |
| `run_base_model_1.sh` / `.ps1` | Comment update |

---

## Phase 4 — Normalization & metadata

**Goal:** Ensure min-max stats and saved bundles record how features were built.

### Tasks

1. **Update** `src/features/normalization.py` — `save_normalization_stats()` extra dict: include `multidex_mode`, optional `num_dex_files_seen` histogram (future).

2. **Update** `_save_aggregate()` in `preprocess_apks.py` — bundle metadata:
   ```python
   "multidex_mode": md["mode"],
   "dex_pattern": md["dex_pattern"],
   ```

3. **Cache versioning (recommended):** Either:
   - Change `aggregate_filename` to `dex_header_features_multidex.pt`, **or**
   - Add `preprocessing.cache_version: 2` and document that old `.pt` files are primary-only

   Prevents accidentally training on old primary-only tensors while believing multidex is active.

4. **`concat` mode only (deferred):** If enabled, `feature_dim` becomes `104 * max_dex`; must propagate to Phase 4 model (`MLPHeader.input_dim`) and checkpoint `feature_dim`. Not in default path.

### Verification

- Inspect `artifacts/normalization.json` for `multidex_mode`
- Load bundle in Python; confirm metadata keys

### Files touched

| File | Action |
|------|--------|
| `src/features/normalization.py` | Extra metadata |
| `src/preprocessing/preprocess_apks.py` | Bundle metadata |
| `config/default.yaml` | Optional `cache_version` / renamed output file |

---

## Phase 5 — Tests

**Goal:** Lock behavior with synthetic multi-Dex APKs (no real malware needed).

### Tasks

1. **Create** `tests/test_multidex.py` (port from custom approach):
   - `dex_suffix_sort_key` ordering (`classes.dex` < `classes2.dex` < `classes10.dex`)
   - `list_dex_entries` on synthetic ZIP
   - `read_all_dex_from_apk` returns all entries
   - `aggregate_header_vectors`: sum equals hand-computed; mean; primary_only; empty list raises
   - Single-Dex APK: sum equals one vector

2. **Create** `tests/test_multidex_preprocess.py` (optional integration):
   - Build temp APK with two synthetic Dex headers
   - Run extraction path; assert output shape `(104,)`

3. **Update** `tests/test_dex_header.py` — no change to per-Dex parsing; optionally add note in module docstring.

4. **Update** `scripts/verify_setup.py` — import multidex; print `multidex.mode` from config.

### Run

```bash
PYTHONPATH=. python -m unittest tests.test_multidex tests.test_dex_header -v
```

### Files touched

| File | Action |
|------|--------|
| `tests/test_multidex.py` | **New** |
| `tests/test_multidex_preprocess.py` | **New** (optional) |
| `scripts/verify_setup.py` | Multidex smoke check |

---

## Phase 6 — Documentation & full re-run

**Goal:** Document the change and define the remote 50k-APK re-preprocess workflow.

### Tasks

1. **Update** `only_basemodel_1_specifics.md` — new section **Phase 2b: Multi-Dex Header Aggregation** with:
   - Rationale (link to `dex_related_instruction.md`)
   - Default `sum` mode
   - Ablation `primary_only` for paper baseline comparison

2. **Update** `BM1_multidex_plan.md` — mark phases complete as work lands.

3. **Re-run checklist** (remote machine):
   ```bash
   cd only_base1_model
   pip install -r requirements.txt
   PYTHONPATH=. python -m unittest tests.test_multidex tests.test_dex_header -v
   PYTHONPATH=. python -m src.preprocessing.preprocess_apks --apk-root /path/to/50k/apks
   ./scripts/run_train.sh --fresh
   ./scripts/run_evaluate.sh
   ```

4. **Thesis note:** Record in experiment log:
   - Old baseline: primary Dex only (current artifacts)
   - New default: sum-pooled all Dex headers
   - Compare ACC/F1/AUC if both runs exist

### Files touched

| File | Action |
|------|--------|
| `only_basemodel_1_specifics.md` | New Phase 2b section |
| `BM1_multidex_plan.md` | Status updates |

---

## Phase 7 — Ablation & optional concat mode (later)

**Goal:** Support experiment comparisons without blocking the default sum path.

### Tasks

1. Config switches (no code change beyond Phase 1):
   ```yaml
   preprocessing:
     multidex:
       mode: primary_only   # reproduce MSFDroid paper behavior
   # or
       mode: mean
   ```

2. **Concat ablation** (only if thesis requires fixed-size concatenation per guideline):
   - Set `mode: concat`, `max_dex: 3` → `input_dim = 312`
   - Update `src/models/mlp_header.py` factory to read dynamic dim from bundle
   - Retrain from scratch; old checkpoints incompatible

3. **Optional stats mode** (guideline “Primary Focus” lightweight variant): sum only ID-section **sizes** (`string_ids_size`, `method_ids_size`, …) across Dex files into a small fixed vector — **not** recommended for BM1 unless byte-level sum proves too noisy; would change feature semantics.

### Verification

- Run preprocess with `primary_only` on same `--limit 100` APKs; diff feature rows vs old pipeline (should match for single-Dex APKs; differ for multi-Dex)

---

## Implementation order (summary)

```text
Phase 1  multidex.py + config
   ↓
Phase 2  apk_extract.py (list / read all / extract_apk_raw_header)
   ↓
Phase 3  preprocess_apks.py wiring
   ↓
Phase 4  normalization metadata + cache versioning
   ↓
Phase 5  tests + verify_setup
   ↓
Phase 6  docs + full dataset re-preprocess + retrain
   ↓
Phase 7  ablations (primary_only, mean, concat) — optional
```

---

## What does **not** change (sum/mean/primary_only)

| Component | Reason |
|-----------|--------|
| `src/models/mlp_header.py` | Input stays 104-d |
| `src/data/dataset.py`, `dataloaders.py` | Still load `[N, 104]` |
| `src/training/train.py`, `evaluate.py` | Same loops and metrics |
| Checkpoint format | `feature_dim: 104` unchanged |

---

## Acceptance criteria

- [ ] APK with `classes.dex` + `classes2.dex` produces one 104-d feature row (sum mode)
- [ ] APK with only `classes.dex` behaves identically to pre-multidex pipeline
- [ ] APK with no matching Dex entries is logged and skipped
- [ ] `config/default.yaml` defaults to `multidex.mode: sum`
- [ ] `primary_only` mode reproduces current single-Dex extraction
- [ ] Unit tests pass; full preprocess + train smoke test passes
- [ ] `normalization.json` and `.pt` bundle record `multidex_mode`
- [ ] `only_basemodel_1_specifics.md` documents the new behavior

---

## Code reuse note

The custom approach under `full_combined_pipeline_approach/` already implements Phases 1–2 logic (`multidex.py`, expanded `apk_extract.py`, tests). For BM1, **copy and trim** (drop manifest-related imports; keep constants aligned with BM1’s `FEATURE_DIM = 104` in `dex_header.py`). Avoid duplicating divergent implementations long-term — consider a shared `dex_common/` package only if both pipelines must stay in sync for the thesis.
