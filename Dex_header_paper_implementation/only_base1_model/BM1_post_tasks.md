# BM1 Post-Run Tasks — Phase-Wise Implementation Plan

**Scope:** Base Model 1 (`only_base1_model`) only. Pattern A / B are out of scope here.

**Status:** Phase 0 manual steps partially automated — training/preprocess/eval now write `artifacts/metrics/` and mirror to `output_archives/` when `BM1_ARCHIVE=1`. Run a fresh logged training (see `BM1_freshRunCmds.md`), then ask the agent to complete Phase 0 archive/figures.

**Prerequisites (already done):**

- Full preprocess on `/mnt/Files/thesis_full_dataset` → 13,528 APKs
- 50-epoch training on CUDA → `artifacts/checkpoints/latest_checkpoint.pth`
- Reference val metrics: ACC 0.9649, F1 0.9298, AUC 0.9827 (random 80/20, seed 42)

**Related docs:** `BM1_postRunGuide.md`, `BM1_running_guide.md`, `PIPELINE_IMPLEMENTATION_PLAN.md` (D3 / P6–P8).

---

## Archive root: `output_archives/`

All post-run **logs, JSON, figures, and run manifests** live under one tree (separate from mutable `artifacts/` used for training).

**Base path:**

```
/mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model/output_archives/
```

**Recommended layout (one folder per run):**

```
output_archives/
└── run_<YYYYMMDD>_<short_label>/          # e.g. run_20260524_full_corpus_v1
    ├── RUN_MANIFEST.json                  # single source of truth for this archive
    ├── config/
    │   └── default.yaml.snapshot          # copy of config used
    ├── logs/
    │   ├── pipeline_full.log              # tee of run_base_model_1.sh
    │   └── preprocess_summary.txt         # optional excerpt
    ├── metrics/
    │   ├── metrics_val.json               # final val metrics (schema § Phase 2)
    │   ├── metrics_train.json             # optional train-split eval
    │   ├── checkpoint_summary.json        # extracted from .pth
    │   └── epochs.jsonl                   # per-epoch train/val loss + metrics (if captured)
    ├── corpus_stats/
    │   ├── label_distribution.json
    │   ├── dex_file_counts.json
    │   └── year_folder_counts.json        # optional, from APK paths
    ├── figures/
    │   ├── loss_curves.png
    │   ├── metrics_vs_epoch.png           # ACC, F1, AUC
    │   ├── label_distribution.png
    │   ├── dex_count_histogram.png
    │   ├── roc_curve_val.png              # Phase 3
    │   └── confusion_matrix_val.png
    ├── export/                            # Phase 4 — after ONNX exists
    │   ├── model.onnx
    │   ├── export_manifest.json
    │   └── normalization_header.json      # copy from artifacts/
    └── parity/                            # Phase 5
        ├── parity_report.json
        └── sample_vectors.npz
```

**Naming convention:** `run_<date>_<label>` — never overwrite; new training → new subfolder.

**Git:** Add `output_archives/run_*/` to `.gitignore` if archives are large; keep a small `output_archives/README.md` in repo describing layout.

---

## Phase overview

| Phase | Name | Type | Depends on |
|-------|------|------|------------|
| **0** | Manual archive of current run | You run commands | Full BM1 run complete |
| **1** | `output_archives` bootstrap + RUN_MANIFEST | You + light scripting | Phase 0 |
| **2** | Metrics JSON + checkpoint / corpus exports | Code + run | Phase 1 |
| **3** | Visualizations & figures | Code + run | Phase 2 (needs epoch history or re-log) |
| **4** | ONNX export bundle (D3 deployment) | Code + run | Trained checkpoint |
| **5** | ONNX parity vs PyTorch | Code + run | Phase 4 |
| **6** | Optional BM1 experiments | You run | Phase 1 |
| **7** | Thesis-ready summary pack | You write | Phases 0–3 minimum |

---

## Phase 0 — Archive of the completed run

**Goal:** Preserve the logged run under `output_archives/` (mostly automatic if you used `BM1_ARCHIVE=1`; agent completes copies/checksums/figures after your fresh run).

**You first:** Run `BM1_freshRunCmds.md` (wipe + `FRESH_TRAIN=1` + `BM1_ARCHIVE=1`).

**Agent after your run:** Finish Phase 0 checklist (checksums, any missing copies, optional plots).

**Tasks (checklist):**

- [ ] **0.1** Create archive directory:
  ```bash
  cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
  RUN_ID="run_$(date +%Y%m%d)_full_corpus_v1"
  mkdir -p "output_archives/${RUN_ID}"/{logs,metrics,corpus_stats,figures,config,export,parity}
  echo "$RUN_ID" > output_archives/LATEST_RUN.txt
  ```

- [ ] **0.2** Copy terminal log into `logs/`:
  - If saved: `cp /path/to/1stRunOutputs_BM1.txt "output_archives/${RUN_ID}/logs/pipeline_full.log"`
  - Else: paste final Konsole output into `logs/pipeline_full.log`

- [ ] **0.3** Snapshot config:
  ```bash
  cp config/default.yaml "output_archives/${RUN_ID}/config/default.yaml.snapshot"
  ```

- [ ] **0.4** Copy immutable artifacts (read-only record; keep originals in `artifacts/`):
  ```bash
  cp artifacts/normalization.json "output_archives/${RUN_ID}/metrics/"
  cp artifacts/failed_apks.log "output_archives/${RUN_ID}/logs/" 2>/dev/null || true
  # Optional: checksum only (7MB) instead of full .pt duplicate:
  sha256sum artifacts/processed/dex_header_features.pt >> "output_archives/${RUN_ID}/RUN_MANIFEST.sha256"
  sha256sum artifacts/checkpoints/latest_checkpoint.pth >> "output_archives/${RUN_ID}/RUN_MANIFEST.sha256"
  ```

- [ ] **0.5** Extract checkpoint summary to JSON (one-off command):
  ```bash
  ../../thesis_venv/bin/python -c "
  import json, torch
  from pathlib import Path
  import os
  run = Path('output_archives') / Path(open('output_archives/LATEST_RUN.txt').read().strip())
  ckpt = torch.load('artifacts/checkpoints/latest_checkpoint.pth', map_location='cpu', weights_only=False)
  summary = {
      'next_epoch': ckpt.get('next_epoch'),
      'train_loss': ckpt.get('train_loss'),
      'val_loss': ckpt.get('val_loss'),
      'val_metrics': ckpt.get('val_metrics'),
      'feature_dim': ckpt.get('feature_dim'),
      'hidden_dim': ckpt.get('hidden_dim'),
  }
  (run / 'metrics/checkpoint_summary.json').write_text(json.dumps(summary, indent=2))
  print('wrote', run / 'metrics/checkpoint_summary.json')
  "
  ```

- [ ] **0.6** Write minimal `RUN_MANIFEST.json` by hand or script (template in Phase 1).

**Deliverables:** Populated `output_archives/run_*` with logs, config snapshot, `checkpoint_summary.json`.

**Acceptance:** You can point to one folder that documents the first full run without re-running training.

---

## Phase 1 — Archive bootstrap & run manifest standard

**Goal:** Define a repeatable archive contract so every future BM1 run is self-describing.

**Tasks (implementation later — not started):**

- [ ] **1.1** Add `output_archives/README.md` (layout + naming rules).
- [ ] **1.2** Define `RUN_MANIFEST.json` schema (minimum fields):

  ```json
  {
    "run_id": "run_20260524_full_corpus_v1",
    "model_id": "mlp_header",
    "domain": "dex_header_d3",
    "created_at": "ISO-8601",
    "git_commit": "<hash or null>",
    "apk_root": "/mnt/Files/thesis_full_dataset",
    "n_apks_discovered": 13528,
    "n_apks_processed": 13528,
    "n_failed": 0,
    "preprocessing": {
      "multidex_mode": "sum",
      "cache_version": 2,
      "feature_dim": 104
    },
    "training": {
      "epochs_configured": 50,
      "epochs_completed": 50,
      "fresh_train": true,
      "device": "cuda",
      "val_fraction": 0.2,
      "random_seed": 42,
      "batch_size": 16,
      "hidden_dim": 128
    },
    "final_val_metrics": {
      "accuracy": 0.9649,
      "f1": 0.9298,
      "roc_auc": 0.9827,
      "loss": 0.1261
    },
    "artifact_paths": {
      "features_pt": "artifacts/processed/dex_header_features.pt",
      "checkpoint": "artifacts/checkpoints/latest_checkpoint.pth",
      "normalization": "artifacts/normalization.json"
    },
    "archive_paths": {
      "log": "logs/pipeline_full.log",
      "metrics_val": "metrics/metrics_val.json"
    },
    "notes": "Random 80/20 split; not comparable to Pattern A 90/10 until unified."
  }
  ```

- [ ] **1.3** Add script `scripts/archive_run.sh` (optional):
  - Args: `--run-id`, `--log-file`, `--from-artifacts`
  - Creates folder tree, copies files, writes manifest template, sets `LATEST_RUN.txt`

- [ ] **1.4** Update `run_base_model_1.sh` (optional later): `tee` stdout/stderr to `output_archives/${RUN_ID}/logs/pipeline_full.log` when `ARCHIVE_RUN_ID` is set.

**Deliverables:** Documented schema + optional archive script.

**Acceptance:** Phase 0 folder matches schema; `LATEST_RUN.txt` points to canonical run.

---

## Phase 2 — Metrics JSON & structured exports

**Goal:** Match `PIPELINE_IMPLEMENTATION_PLAN.md` §7.1 (offline metrics JSON); stop relying on terminal copy-paste.

**Tasks (implementation later):**

- [ ] **2.1** Add `write_metrics_json()` helper (or reuse pattern from shared pipeline if extracted):
  - Output path default: `output_archives/<run_id>/metrics/metrics_val.json` when env `BM1_ARCHIVE_RUN` set, else `artifacts/checkpoints/metrics_val.json`

- [ ] **2.2** Extend `src/training/evaluate.py`:
  - `--metrics-out PATH`
  - `--export-json` flag
  - Payload: `run_id`, `model_id`, `split`, `n_samples`, `metrics`, `threshold`, `confusion_matrix`, `checkpoint`, `hardware`, `timestamp`

- [ ] **2.3** Add `scripts/export_corpus_stats.py`:
  - Reads `dex_header_features.pt`
  - Writes to `output_archives/<run>/corpus_stats/`:
    - `label_distribution.json`
    - `dex_file_counts.json` (from bundle metadata)
    - `paths_sample.txt` (first/last N paths, optional)

- [ ] **2.4** Parse epoch history from `logs/pipeline_full.log` → `metrics/epochs.jsonl`:
  - Regex lines like `Epoch 12/50 — train_loss=... val_loss=... ACC=...`
  - One-off script `scripts/parse_training_log.py` if training doesn’t log JSON per epoch yet

- [ ] **2.5** Optional: log per-epoch metrics inside `train.py` to `artifacts/training_history.jsonl` during training (future runs).

**Commands you will run (after implementation):**

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
export BM1_ARCHIVE_RUN="$(cat output_archives/LATEST_RUN.txt)"
SKIP_PREPROCESS=1 SKIP_TRAIN=1 APK_ROOT=/mnt/Files/thesis_full_dataset \
  ./run_base_model_1.sh
# Or:
./scripts/run_evaluate.sh --split val \
  --metrics-out "output_archives/${BM1_ARCHIVE_RUN}/metrics/metrics_val.json"
../../thesis_venv/bin/python scripts/export_corpus_stats.py \
  --archive-dir "output_archives/${BM1_ARCHIVE_RUN}"
```

**Deliverables:** `metrics_val.json`, `corpus_stats/*.json`, `epochs.jsonl` (from log parse).

**Acceptance:** All thesis numbers reproducible from JSON under `output_archives/`.

---

## Phase 3 — Graphs & visualizations

**Goal:** Figures for understanding and thesis (loss curves, class balance, ROC, confusion matrix).

**Dependencies:** Phase 2 (`epochs.jsonl` or `training_history.jsonl`); checkpoint + `.pt` for ROC/CM.

**Tasks (implementation later):**

- [ ] **3.1** Add `scripts/plot_bm1_results.py` (matplotlib, save to `figures/`):
  - **loss_curves.png** — train_loss & val_loss vs epoch
  - **metrics_vs_epoch.png** — ACC, F1, AUC vs epoch (twin axis or subplots)
  - **label_distribution.png** — bar chart benign vs malware
  - **dex_count_histogram.png** — APKs binned by number of DEX files

- [ ] **3.2** Add evaluation plots (requires inference pass on val split):
  - **roc_curve_val.png** — `sklearn.metrics.roc_curve`
  - **confusion_matrix_val.png** — heatmap at threshold 0.5 (config)

- [ ] **3.3** Optional:
  - Score histogram (malware probability on val)
  - Calibration plot (reliability diagram)

- [ ] **3.4** Save `figures/figure_index.json` listing each PNG + caption for thesis.

**Commands you will run (after implementation):**

```bash
RUN="$(cat output_archives/LATEST_RUN.txt)"
../../thesis_venv/bin/python scripts/plot_bm1_results.py \
  --archive-dir "output_archives/${RUN}" \
  --checkpoint artifacts/checkpoints/latest_checkpoint.pth \
  --features artifacts/processed/dex_header_features.pt
```

**Deliverables:** PNGs under `output_archives/<run>/figures/`.

**Acceptance:** Can explain training convergence and final performance without opening terminal logs.

---

## Phase 4 — ONNX export (D3 deployment artifact)

**Goal:** `PIPELINE_IMPLEMENTATION_PLAN.md` P7 — export MLP(H) for VigiDroid / ORT.

**Tasks (implementation later):**

- [ ] **4.1** Add `scripts/export_onnx.py`:
  - Load `latest_checkpoint.pth` → `MLPHeader(feature_dim, hidden_dim)`
  - `torch.onnx.export` — input `[1, 104]` float32, output `[1, 1]` malware probability
  - Opset 14 (per master plan)
  - Write to `artifacts/export/mlp_header/` and copy to `output_archives/<run>/export/`

- [ ] **4.2** Write `export_manifest.json`:
  - `model_id`: `mlp_header`
  - `inputs` / `outputs` shapes and dtypes
  - `preprocessing_version`, `multidex_mode`, `normalization` path reference

- [ ] **4.3** Copy `artifacts/normalization.json` → `export/normalization_header.json` in both export dir and archive.

- [ ] **4.4** Document copy target for Android:
  - `vigidroid/app/src/main/assets/models/mlp_header/` (when app integration starts — not in this phase’s scope)

**Commands you will run (after implementation):**

```bash
./scripts/export_onnx.py \
  --checkpoint artifacts/checkpoints/latest_checkpoint.pth \
  --out-dir artifacts/export/mlp_header
RUN="$(cat output_archives/LATEST_RUN.txt)"
cp -r artifacts/export/mlp_header/* "output_archives/${RUN}/export/"
```

**Deliverables:** `model.onnx`, `export_manifest.json`, normalization copy in archive.

**Acceptance:** ONNX loads in Python ORT; input/output shapes match manifest.

---

## Phase 5 — Parity check (PyTorch vs ONNX)

**Goal:** P8 — ≤1e-4 probability delta on sample vectors (master plan §3.4).

**Tasks (implementation later):**

- [ ] **5.1** Add `scripts/parity_check_onnx.py`:
  - Sample N rows from `dex_header_features.pt` (or fixed seeds)
  - Compare `MLPHeader` forward vs ONNX Runtime
  - Write `parity/parity_report.json` with max/mean abs diff

- [ ] **5.2** Save `parity/sample_vectors.npz` for Java spot-checks later.

- [ ] **5.3** Wire into `Shared_pipeline_Files/tools/parity_check.py` when repo-wide runner exists.

**Deliverables:** `parity_report.json` under archive.

**Acceptance:** `max_abs_diff < 1e-4` (or documented tolerance if BatchNorm eval mode differs).

---

## Phase 6 — Optional BM1-only experiments (no Pattern A/B)

**Goal:** Strengthen thesis claims or ablations; each experiment → **new** `output_archives/run_*` folder.

| ID | Experiment | Action | New archive? |
|----|------------|--------|--------------|
| 6a | Train-split eval (overfit check) | `run_evaluate.sh --split train` | Yes — metrics_train.json |
| 6b | Longer training | `EPOCHS=80 SKIP_PREPROCESS=1` resume | Yes |
| 6c | Fresh 50-epoch (same `.pt`) | `FRESH_TRAIN=1 SKIP_PREPROCESS=1` | Yes |
| 6d | Multidex ablation `primary_only` | Change config + `cache_version` + full preprocess | Yes — new label e.g. `_primary_only` |
| 6e | Threshold sweep | Script: F1 vs threshold on val | JSON + plot in figures/ |

**You run these only if needed for the thesis; not required for minimum D3 completion.**

---

## Phase 7 — Thesis-ready summary pack

**Goal:** One-page + figure pack for BM1 chapter from `output_archives/` only.

**Tasks (you write / compile):**

- [ ] **7.1** `output_archives/<run>/THESIS_SNIPPET.md`:
  - Dataset: N APKs, years layout, label method
  - Features: 104-d Dex header, multidex sum
  - Model: MLP(H) architecture summary
  - Training: hyperparameters table
  - Results: val ACC/F1/AUC + caveats (random split, 13.5k not 40k)
  - Limitations: `failed_apks.log`, split policy

- [ ] **7.2** Copy selected figures to thesis `figures/` directory (manual).

- [ ] **7.3** Link `RUN_MANIFEST.json` as reproducibility artifact in appendix.

**Acceptance:** External reader can reproduce claims from archive + `BM1_running_guide.md`.

---

## Suggested execution order (for you)

```
Phase 0  ──►  You: manual archive (today, no code)
   │
Phase 1  ──►  Approve schema; optional archive_run.sh
   │
Phase 2  ──►  Implement metrics JSON + corpus stats → re-run eval once
   │
Phase 3  ──►  Implement plotting → generate figures
   │
Phase 4  ──►  Implement ONNX export
   │
Phase 5  ──►  Parity check
   │
Phase 6  ──►  Optional experiments (as needed)
   │
Phase 7  ──►  Thesis snippet
```

**Minimum viable post-run (BM1 only, no device):** Phase **0 → 2 → 3 → 7**.  
**Minimum for VigiDroid D3:** Phase **0 → 2 → 4 → 5** (+ Java extractor later, outside BM1 folder).

---

## What is explicitly out of scope here

- Pattern A / Pattern B training or archives
- Shared `train.txt` / `val.txt` splits (Phase D in `BM1_postRunGuide.md`)
- VigiDroid `ScanService` / Java `DexHeaderExtractor`
- Multistep cascade, ensemble, device metrics JSON
- Re-downloading or expanding APK corpus beyond current 13,528

---

## Implementation gate

| Who | When |
|-----|------|
| **You** | Complete **Phase 0** checklist manually first |
| **You** | Review this plan; adjust `RUN_MANIFEST` fields or figure list |
| **Agent / dev** | Implement Phases 1–5 only after you say “start Phase N” |

**Do not start code changes or re-training until Phase 0 is done and you approve the next phase.**

---

*Last updated: planning doc for post–first-full-run BM1. Align metrics in Phase 0 manifest with your latest checkpoint if you re-ran with `FRESH_TRAIN=1`.*
