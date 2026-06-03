# BM1 Post-Run Guide — Outputs, Usage, and What’s Next

You completed **Base Model 1 (D3 / MLP(H))** training per `BM1_running_guide.md`. This document lists every artifact, explains how to use each one, and maps the next steps from `PIPELINE_IMPLEMENTATION_PLAN.md` at the repo root.

**Your environment (CachyOS):**

| Item | Path |
|------|------|
| BM1 folder | `/mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model` |
| APK dataset | `/mnt/Files/thesis_full_dataset` |
| Shared venv | `/mnt/Files/thesis_vigidroid/thesis_venv` |
| Pipeline plan | `/mnt/Files/thesis_vigidroid/PIPELINE_IMPLEMENTATION_PLAN.md` |

**Corpus processed:** 13,528 APKs (all `*.apk` under `APK_ROOT`; matches `find … -name '*.apk' | wc -l`).

**Reference validation metrics (50 epochs, random 80/20 split, seed 42):**

| Metric | Value |
|--------|--------|
| ACC | 0.9649 |
| F1 | 0.9298 |
| ROC-AUC | 0.9827 |
| val_loss (BCE) | 0.1261 |

Treat these as a **baseline on your current split**, not final thesis numbers until you align splits with Pattern A/B and optional temporal holdout (see §5).

---

## 1. Full list of BM1 outputs

### 1.1 Files on disk (under `only_base1_model/artifacts/`)

| Path | Always created? | Size (typical) | What it contains |
|------|-----------------|----------------|------------------|
| `artifacts/processed/dex_header_features.pt` | Yes (after preprocess) | ~7 MB for 13.5k APKs | PyTorch bundle: `features` `[N,104]`, `labels` `[N]`, `paths` (APK paths), `feature_dim`, `num_samples`, `normalization_mins/maxs`, multidex metadata (`multidex_mode`, `dex_pattern`, `cache_version`, `dex_file_counts`, …) |
| `artifacts/normalization.json` | Yes | ~4 KB | Min–max stats + metadata for **inference parity** (Java/on-device must use same mins/maxs) |
| `artifacts/checkpoints/latest_checkpoint.pth` | Yes (after train) | ~250 KB | `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `next_epoch`, `train_loss`, `val_loss`, `val_metrics`, `feature_dim`, `hidden_dim` |
| `artifacts/failed_apks.log` | Only if failures | 0 bytes if empty | Tab-separated: `apk_path<TAB>reason` for APKs skipped during preprocess |

**Optional / config-dependent (not used with default `output_format: pt`):**

| Path | When |
|------|------|
| `artifacts/processed/*.features.npy`, `*.labels.npy`, `*.paths.npy`, `*.meta.json` | If `preprocessing.output_format: npy` in `config/default.yaml` |

**Not produced by BM1 today (planned in `PIPELINE_IMPLEMENTATION_PLAN.md`):**

| Path | Phase | Purpose |
|------|-------|---------|
| `artifacts/export/mlp_header/model.onnx` | P7 | VigiDroid / ONNX Runtime |
| `artifacts/export/mlp_header/export_manifest.json` | P7 | Input shape, opset, preprocessing version |
| `artifacts/export/mlp_header/features/normalization_header.json` | P7 | Copy of norm stats for assets bundle |
| `artifacts/checkpoints/metrics_val.json` | P6 | Offline metrics JSON for plotting (Pattern A already writes this; BM1 prints to terminal only) |
| `results/offline/mlp_header_val.json` | P6 / §7 | Repo-wide standardized metrics path (recommended to add) |

### 1.2 Terminal-only outputs

| Output | Where | Use |
|--------|-------|-----|
| Per-epoch line: `train_loss`, `val_loss`, `lr`, `ACC`, `F1`, `AUC` | Training (Step 4) | Monitor convergence; copy best epoch for notes |
| Final `Evaluation (val) — …` | Step 5 | Same metrics as last epoch on val split |
| Preprocess summary: `total_apks`, `successful`, `failed`, `dex_file_counts` | Step 3 | Corpus QA, thesis “dataset statistics” table |
| `Device: cuda` / `cpu` | Training start | Confirm GPU run |

### 1.3 Config inputs (not outputs, but required to interpret artifacts)

| File | Role |
|------|------|
| `config/default.yaml` | Paths, multidex `sum`, batch 16, 50 epochs, `device: cuda`, val_fraction 0.2, seed 42 |
| Command env: `APK_ROOT` | Dataset root used for this run |

### 1.4 What is *not* an output

| Item | Note |
|------|------|
| APK files under `/mnt/Files/thesis_full_dataset` | Unchanged; only read |
| Train/val index lists on disk | BM1 splits **in memory** at DataLoader build time (not `train.txt` / `val.txt`) |
| Per-epoch checkpoint files | Only `latest_checkpoint.pth` is kept (overwritten each epoch) |

---

## 2. Where to use what (and how)

### 2.1 `dex_header_features.pt`

| Use case | How |
|----------|-----|
| **Resume / retrain** | Loaded automatically by `build_dataloaders_from_config()` when you run `run_base_model_1.sh` or `run_train.sh` |
| **Inspect corpus** | `torch.load(..., weights_only=False)` — check `labels`, `paths`, `dex_file_counts` |
| **Ablation / error analysis** | Map misclassified APKs: run eval, compare predictions to `paths` in the bundle |
| **Not for Pattern A/B directly** | Pattern A/B re-extract per-APK shards with manifest BoW; they share Dex header *logic* but not this single `.pt` file |

Example — label balance:

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
../../thesis_venv/bin/python -c "
import torch
b = torch.load('artifacts/processed/dex_header_features.pt', weights_only=False)
y = b['labels'].int()
print('N=', len(y), 'benign=', (y==0).sum().item(), 'malware=', (y==1).sum().item())
"
```

### 2.2 `normalization.json`

| Use case | How |
|----------|-----|
| **On-device D3 (future)** | Ship alongside ONNX in VigiDroid assets; Java `DexHeaderExtractor` applies same min–max as Python |
| **Re-preprocess consistency** | If you change APK set, re-run preprocess — new JSON overwrites old stats |
| **Parity checks** | Compare fields to `normalization_mins` / `normalization_maxs` inside `.pt` (should match) |

### 2.3 `latest_checkpoint.pth`

| Use case | How |
|----------|-----|
| **Resume training** | `./run_base_model_1.sh` without `FRESH_TRAIN=1` (or `run_train.sh` without `--fresh`) |
| **Evaluation only** | `SKIP_PREPROCESS=1 SKIP_TRAIN=1 ./run_base_model_1.sh` or `./scripts/run_evaluate.sh --checkpoint artifacts/checkpoints/latest_checkpoint.pth` |
| **ONNX export (when implemented)** | Load `model_state_dict` + `feature_dim` / `hidden_dim` → trace `MLPHeader` → `.onnx` |
| **Thesis tables** | Read `val_metrics` from checkpoint: `torch.load(...)['val_metrics']` |

Example — read stored metrics:

```bash
../../thesis_venv/bin/python -c "
import torch
ckpt = torch.load('artifacts/checkpoints/latest_checkpoint.pth', map_location='cpu', weights_only=False)
print('epochs done:', ckpt.get('next_epoch'))
print('val_metrics:', ckpt.get('val_metrics'))
"
```

### 2.4 `failed_apks.log`

| Use case | How |
|----------|-----|
| **Data quality** | `wc -l artifacts/failed_apks.log` — should be 0 or small |
| **Thesis limitations** | Quote failure reasons (bad zip, no dex, label unknown) for feasibility / evasion discussion |
| **Feasibility (Task 6)** | Input to “how many APKs are statically analyzable?” |

### 2.5 Terminal metrics (ACC / F1 / AUC)

| Use case | How |
|----------|-----|
| **Quick baseline** | Copy from log or `1stRunOutputs_BM1.txt` into thesis draft |
| **Compare models later** | Same split policy → compare D3 vs D4 vs D5 (after you align evaluation protocol) |
| **Multistep thresholds (Task 2)** | Use validation score distribution to pick `t_low` / `t_high` for cascade |

**Caveat:** BM1 uses a **random 80/20 split** (`data.val_fraction: 0.2`, `random_seed: 42`). Pattern A uses **90/10 stratified split files**. Numbers are **not directly comparable** until you use a shared split strategy.

---

## 3. BM1 phases vs pipeline plan (status)

From `PIPELINE_IMPLEMENTATION_PLAN.md` §3.1 (per-model phases):

| Phase | Name | BM1 status |
|-------|------|------------|
| P0 | Config & environment | Done |
| P1 | Dataset indexing | **Partial** — labels from folders; no `dataset_index.csv` / year holdout in BM1 |
| P2 | Feature extraction | Done → `.pt` + `normalization.json` |
| P3 | DataLoaders | Done |
| P4 | Model | Done → `MLPHeader` |
| P5 | Training | Done → `latest_checkpoint.pth` |
| P6 | Offline evaluation | **Partial** — metrics in terminal + checkpoint dict; **no `metrics_val.json` yet** |
| P7 | ONNX export | **Not implemented** — `scripts/export_onnx.py` missing |
| P8 | Parity check | **Blocked** until P7 + Java `DexHeaderExtractor` |

**Domain ID in master plan:** **D3** — Dex header only, MSFDroid Base Model 1. Status moves from “Planned” to **“Trained offline”**; on-device still **Planned** until P7–P8.

---

## 4. What to do next (ordered roadmap)

Aligned with `PIPELINE_IMPLEMENTATION_PLAN.md` §8 (Phase B → C) and §11 (immediate actions).

### Step A — Record this run (do now, ~15 minutes)

1. **Archive the log** (if not already):
   ```bash
   cp /path/to/your/terminal/log.txt \
      /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model/artifacts/bm1_full_run.log
   ```
2. **Snapshot metrics** from checkpoint (command in §2.3).
3. **Note run metadata** in your thesis lab notebook:
   - Date, commit hash, `APK_ROOT`, N=13528, split=random 80/20 seed 42, 50 epochs, CUDA, multidex `sum`.

Optional: re-run eval only after code updates:

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
SKIP_PREPROCESS=1 SKIP_TRAIN=1 APK_ROOT=/mnt/Files/thesis_full_dataset ./run_base_model_1.sh
```

### Step B — Close BM1 gaps before VigiDroid (Phase B / P7–P8)

| Priority | Task | Why |
|----------|------|-----|
| **B1** | Add `scripts/export_onnx.py` for MLP(H) | Required for D3 on device (`PIPELINE_IMPLEMENTATION_PLAN.md` §3.4) |
| **B2** | Export bundle under `artifacts/export/mlp_header/` | `model.onnx`, `export_manifest.json`, copy `normalization.json` |
| **B3** | Implement `--export-json` in BM1 `evaluate.py` | Matches §7.1 / Phase C; Pattern A already has this |
| **B4** | Java `DexHeaderExtractor` + parity | `Shared_pipeline_Files/tools/parity_check.py` when ONNX exists |

Until B1–B2 exist, `Shared_pipeline_Files/tools/export_all_onnx.sh` will **SKIP** D3.

### Step C — Train Pattern A and Pattern B (same dataset, Phase B)

BM1 is only **D3**. The plan’s full static portfolio includes:

| Model | Folder | Guide |
|-------|--------|-------|
| **D4 Pattern A** | `custom_approach/full_combined_pipeline_approach/` | `patternA_running_guide.md` |
| **D5 Pattern B** | `custom_approach/dual_branch_merge_approach/` | `patternB_running_guide.md` (if present) |

**Suggested command** (after Pattern A verify step):

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach
# Edit config: training.device: cuda
PREPROCESS_LIMIT=200 EPOCHS=2 APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_a.sh   # smoke
APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_a.sh   # full
```

Pattern A differences vs BM1:

- Manifest BoW + header; **90/10 stratified** split on disk
- Per-APK `.npz` shards (larger artifact footprint)
- Writes **`metrics_val.json`** automatically
- Packaging tarball step for checkpoints + vocab

You do **not** need to re-run BM1 preprocess for Pattern A; it runs its own pipeline.

### Step D — Shared dataset hygiene (Phase B / §3.2) — recommended before final thesis numbers

| Task | Tool / location |
|------|-----------------|
| Build APK manifest + splits | `Shared_pipeline_Files/tools/build_apk_manifest.py`, `split_dataset.py` |
| Point paths | `Shared_pipeline_Files/data/dataset_paths.yaml` → set `apk_root: /mnt/Files/thesis_full_dataset` |
| Temporal test (2023 holdout) | Document in thesis; re-train with split files when integrated |
| Deduplicate SHA-256 | Plan §3.2 — avoid inflated ACC from duplicates |

BM1 can keep its random split for development; for **thesis tables**, prefer a **documented split** shared across D3–D5.

### Step E — Metrics & thesis figures (Phase C / §7)

| Task | Action |
|------|--------|
| Offline JSON | Add/run `--export-json results/offline/mlp_header_val.json` (after B3) |
| Compare models | Bar chart: D3 vs D4 vs D5 ACC/F1/AUC from each pipeline’s JSON |
| Plotting | `Shared_pipeline_Files/tools/plot_metrics.py` (when populated) |

### Step F — VigiDroid deployment (Project 2)

After **B1–B4**:

1. Copy export bundle → `vigidroid/app/src/main/assets/models/mlp_header/` (exact path per your app layout).
2. Register in `ModelRegistry` / `ScanService` refactor (plan §4.2).
3. Use D3 as **Step 1** in multistep cascade (plan §Task 2) — fast filter before ByteCNN / fusion.

### Step G — Research tasks (after multiple models exported)

| Task | Depends on |
|------|------------|
| **Task 1** — Resource optimization | D3–D5 on-device timings |
| **Task 2** — Multistep cascade | D3 thresholds + orchestrator |
| **Task 3** — Smart schedule | VigiDroid scheduler |
| **Task 4** — vs dynamic | Labeled subset + external sandbox |
| **Task 5** — Tradeoffs | All `metrics_*.json` + device JSON |
| **Task 6** — Feasibility | `failed_apks.log` + latency SLOs |

---

## 5. Decision tree: “What should I run tomorrow?”

```
BM1 training finished?
        │
        ├─► Need thesis numbers only on header baseline?
        │       └─► Archive metrics (Step A); optional temporal split later (Step D)
        │
        ├─► Need on-device malware scanner with header model?
        │       └─► Step B (ONNX + parity) → Step F
        │
        ├─► Need full MSFDroid-style comparison (header + manifest)?
        │       └─► Step C (Pattern A, then Pattern B)
        │
        └─► Need ensemble / multistep / battery study?
                └─► Step C + B + F, then Step G
```

**Practical recommendation for your thesis timeline:**

1. **Step A** — archive BM1 results (today).  
2. **Step C** — start **Pattern A** full run on same `APK_ROOT` (parallel research track; higher accuracy expected).  
3. **Step B** — implement BM1 ONNX export (unblocks D3 on phone).  
4. **Step D** — unify splits when you freeze experiment protocol for the thesis chapter.

---

## 6. Quick command reference

| Goal | Command |
|------|---------|
| Re-evaluate only | `SKIP_PREPROCESS=1 SKIP_TRAIN=1 APK_ROOT=/mnt/Files/thesis_full_dataset ./run_base_model_1.sh` |
| Train more epochs (resume) | `EPOCHS=80 SKIP_PREPROCESS=1 APK_ROOT=... ./run_base_model_1.sh` |
| Retrain from scratch (keep `.pt`) | `FRESH_TRAIN=1 SKIP_PREPROCESS=1 APK_ROOT=... ./run_base_model_1.sh` |
| Full redo (wipe artifacts) | See `BM1_freshRunCmds.md` |
| Pattern A full run | `cd .../full_combined_pipeline_approach && APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_a.sh` |

---

## 7. Related docs

| Document | Purpose |
|----------|---------|
| `BM1_running_guide.md` | How to run BM1 |
| `BM1_freshRunCmds.md` | Wipe artifacts + clean 50-epoch run |
| `only_basemodel_1_specifics.md` | Architecture and phase details |
| `PIPELINE_IMPLEMENTATION_PLAN.md` | Full thesis pipeline (D1–D6, tasks, ONNX, VigiDroid) |
| `patternA_running_guide.md` | Next model (D4) |
| `Shared_pipeline_Files/data/dataset_paths.yaml` | Shared paths for manifests/splits |

---

*Generated for post–first-full-run BM1. Update validation metrics in §顶部 if you re-ran with `FRESH_TRAIN=1` or a different split.*
