# Visualization & CSV fixes — task list

Applied after full model runs, Scan A/B pulls, and `run_e2e_plotting_pipeline.sh`.

## 1. Model display names (`model_plot_registry.json`)

| Task | Status |
|------|--------|
| Rename **Pattern A** → **Dex+Manifest ASCNN** (MSFDroid ASCNN(C) single fused tower) | done |
| Rename **Pattern B** → **Dex+Manifest Dual** (dual-branch late fusion) | done |
| Rename **BM1 (MLP-H)** → **MLP-H** | done |
| Keep canonical `model_id`: `early_fusion_dex_manifest`, `dual_branch_dex_manifest`, `mlp_header` | done |

## 2. Features column (`model_plot_registry.json`)

Brief feature-domain wording per `model_feature_domains_list.md` (signal source · encoding).

| Task | Status |
|------|--------|
| Update all 10 `features` strings to short domain descriptions | done |

## 3. Battery column (`device_metrics_lib.py`)

**Root cause:** Per-APK `capacity_pct_delta` stays 0 when Android reports integer % (99→99) on fast scans; session `charge_counter_uah_used` holds the real drain (~18.8 mAh on POCO Scan A).

| Task | Status |
|------|--------|
| Session-level allocation: model share = session drain × (model stage-ms / total stage-ms) | done |
| Fallback: `charge_counter_uah_used / charge_counter_uah_start × capacity_pct_start` when % delta is 0 | done |
| CSV/plots read `device_scan_a.battery_pct_delta` from session allocation | done |

## 4. Comments column (`model_plot_registry.json` + `plot_registry_lib.py`)

| Task | Status |
|------|--------|
| Add per-model `comment` (5–6 words, pros/cons, fits `temp.tex` table width) | done |
| Remove split notes, file paths, and cascade wall from Comments | done |

## 5. Device feasibility — low / medium / high

| Task | Status |
|------|--------|
| Rank 10 models: quality (Acc, F1, ROC-AUC) + Scan A cost + cascade tier | done |
| Top 3 → **high**, middle 4 → **medium**, bottom 3 → **low** | done |
| Document rankings in `model_feasibility_how.md` | done |
| Replace old Feasible/Marginal/Not feasible rule in aggregation + CSV | done |

## 6. Regenerate outputs

| Task | Status |
|------|--------|
| Re-run `aggregate_plot_metrics.py` | done |
| Re-run `run_all_thesis_plots.sh` | done |
| Re-run `build_extended_abstract_csv.py` | done |

## 7. CSV device-column footgun (fixed)

| Task | Status |
|------|--------|
| `run_offline_plotting_eval.sh` no longer passes `--offline-only` to CSV builder | done |
| `build_extended_abstract_csv.py` auto-merges device cols when `plot_metrics_table.json` exists | done |
| Unit test writes offline CSV to temp dir (not production path) | done |

## Files touched

- `Shared_pipeline_Files/data/model_plot_registry.json`
- `Shared_pipeline_Files/tools/plot_registry_lib.py`
- `Shared_pipeline_Files/tools/device_metrics_lib.py`
- `Shared_pipeline_Files/tools/build_extended_abstract_csv.py`
- `Shared_pipeline_Files/tools/aggregate_plot_metrics.py`
- `model_feasibility_how.md` (new)
- `visualization_fixes.md` (this file)
