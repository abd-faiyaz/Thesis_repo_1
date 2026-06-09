# All Model Current Stats

**Generated:** 2026-06-09 (runtime fix P0–P6)  
**Corpus:** `/mnt/Files/thesis_full_dataset` — **13,528 APKs** (2020–2023, benign + malware)  
**Pipeline reference:** [`Pipeline_full_concept.html`](Pipeline_full_concept.html) (P0–P8 offline, A1–A4 Android)

---

## Summary matrix

| Model | `model_id` | Folder | Full corpus P1–P8 | Split policy | Android A1–A4 |
|-------|------------|--------|-------------------|--------------|---------------|
| BM1 (Dex header MLP) | `mlp_header` | `Dex_header_paper_implementation/only_base1_model/` | **P1–P8 complete** | Train 2020–21; val 10% from train years; test = all 2022+2023 | Implemented; bulk device eval pending |
| Pattern A (concat ASCNN) | `early_fusion_dex_manifest` | `Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/` | **P1–P6 done**; P7–P8 missing locally | Same as BM1 | Code + assets staged; PC export stale |
| Pattern B (dual branch) | `dual_branch_dex_manifest` | `Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/` | **Not run** (no `artifacts/`) | Config matches BM1 (not exercised) | Code + stale assets only |
| Broadcast + MLDP hybrid | `broadcast_mldp_hybrid` | `broadcast_mldp_hybrid/` | **P1–P8 complete** | Train 2020–21; **disjoint val/test** from 2022+2023 (50/50) | Implemented + staged |
| LinRegDroid | `linregdroid_permission` | `linear/` | **P1–P8 complete** (smoke/quick train) | Train/val/dev_test from 2020–21; test = temporal_holdout 2022+2023 | Complete (A1–A4) |
| MLDP-pruned permissions | `mldp_pruned_permission` | `permission_extractor/` | **P1–P8 complete** (smoke/quick train) | Same as LinRegDroid | Complete (A1–A4) |
| MLDP + Dex header cascade | `mldp_dexheader_cascade` | `mldp_dexheader_cascade/` | **P1–P8 complete** (quick train: 3 epochs) | Same as BM1 | A4 green; P1 exit 11/11 on device (2026-06-09) |
| Dex header + broadcast fusion | `dexheader_broadcast_fusion` | `dexheader_broadcast_fusion/` | **P1–P8 complete** (quick train) | Same as BM1 | A4 gate + `DexheaderBroadcastFusionA4ParityTest`; device bulk eval pending |

---

## 1) Outputs per model

### BM1 — `mlp_header` (`only_base1_model/`)

| Output | Location | Notes |
|--------|----------|-------|
| Split lists | `artifacts/splits/{train,val,test}.txt` | 10,800 / 1,200 / 1,528 |
| Preprocessed features | `artifacts/processed/dex_header_features.pt` | 13,528 × 104-dim header |
| Normalization stats | `artifacts/normalization.json` | Train-only min–max |
| Checkpoint | `artifacts/checkpoints/latest_checkpoint.pth` | 50-epoch full train |
| Metrics | `artifacts/metrics/{test_results,epochs,preprocess_summary,training_run_info}.json` | Test ACC 0.966, F1 0.944 |
| ONNX export bundle | `artifacts/export/mlp_header/` | `model.onnx`, `thresholds.json`, `export_manifest.json`, `features/normalization_header.json`, `parity_samples/` |
| P8 parity | `artifacts/parity/parity_report.json` | **PASS** (max Δ ≈ 8.9e-8, 8 samples) |
| Thesis archive | `output_archives/run_20260607_val_test_fix/` | Figures, manifest, export copy, `THESIS_SNIPPET.md` |
| Android assets | `vigidroid/app/src/main/assets/models/mlp_header/` | Staged from PC export |

---

### Pattern A — `early_fusion_dex_manifest` (`full_combined_pipeline_approach/`)

| Output | Location | Notes |
|--------|----------|-------|
| Dataset index | `artifacts/dataset_index.csv` | Full corpus indexed |
| Split lists | `artifacts/splits/{train,val,test}.txt` | 10,800 / 1,200 / 1,528 |
| Vocab + norm | `artifacts/vocab.json`, `artifacts/normalization_header.json` | Train-only |
| Processed shards | `artifacts/processed/shards/{train,val,test}/*.npz` | ~13,531 shard files + manifests |
| Checkpoints | `artifacts/checkpoints/{best,latest}.pt` | 80-epoch train completed 2026-06-06 |
| P6 metrics (informal) | `artifacts/checkpoints/test_results.json` | Test ACC 0.961, F1 0.936, n=1,528 |
| Dex stats | `artifacts/dex_stats.json` | Multidex histogram |
| Package tarball | `artifacts/pattern_a_bundle.tar.gz` | Checkpoints + vocab + norm (no shards) |
| Pipeline log | `artifacts/pipeline.log` | Full run log |
| Shared offline JSON | `Shared_pipeline_Files/results/offline/early_fusion_dex_manifest_test_20260606T211430Z.json` | Thesis aggregate export |
| **Missing locally** | `artifacts/export/early_fusion_dex_manifest/` | P7 not re-run after latest train |
| **Missing locally** | `artifacts/metrics/parity_report.json` | P8 not run on latest checkpoint |
| Android assets | `vigidroid/app/src/main/assets/models/early_fusion_dex_manifest/` | **Stale** — exported 2026-06-03 (predates latest train) |

---

### Pattern B — `dual_branch_dex_manifest` (`dual_branch_merge_approach/`)

| Output | Location | Notes |
|--------|----------|-------|
| Source + scripts | `src/`, `scripts/`, `run_pattern_b.sh` | Implementation present |
| **All pipeline artifacts** | `artifacts/` | **Directory does not exist** — pipeline not run in this workspace |
| Shared offline JSON (old) | `Shared_pipeline_Files/results/offline/dual_branch_dex_manifest_val_*.json` | Val-only runs from 2026-06-03 |
| Android assets | `vigidroid/app/src/main/assets/models/dual_branch_dex_manifest/` | **Stale** — exported 2026-06-03; checkpoint path in manifest points to missing local file |

---

### Broadcast + MLDP hybrid — `broadcast_mldp_hybrid`

| Output | Location | Notes |
|--------|----------|-------|
| APK index | `artifacts/manifests/apk_index.csv`, `apk_index_summary.json` | 13,528 APKs |
| Split lists | `artifacts/splits/{train,val,test}.txt` | 12,000 / 764 / 764 |
| Feature tensors | `artifacts/processed/features_{train,val,test}.pt` | S=22 MLDP perms + R=70 receiver actions |
| MLDP trace | `artifacts/processed/mldp_trace.json`, vocabs | Train-only frozen S |
| Checkpoints | `artifacts/checkpoints/{best,ablation_*,svm_rbf,decision_tree}.*` | Tiny MLP + ablations + paper baselines |
| Metrics | `artifacts/metrics/{test_results,epochs,training_run_info,parity_report,thresholds}.json` | Test F1 0.862 (n=764) |
| ONNX export | `artifacts/export/broadcast_mldp_hybrid/` | `model.onnx`, feature vocabs, 10 parity samples |
| P8 parity | `artifacts/metrics/parity_report.json` | **PASS** (max Δ ≈ 6.0e-8) |
| Android assets | `vigidroid/app/src/main/assets/models/broadcast_mldp_hybrid/` | Staged |

---

### LinRegDroid — `linregdroid_permission` (`linear/`)

| Output | Location | Notes |
|--------|----------|-------|
| Dataset index | `artifacts/dataset_index.csv` | Full corpus |
| Split lists | `artifacts/splits/{train,val,dev_test,temporal_holdout}.txt` | 8,400 / 1,800 / 1,800 / 1,528 |
| Permission vocab | `artifacts/permission_vocab.json` | Train-only, 173 dims |
| Processed shards | `artifacts/processed/shards/` | ~13,534 `.npz` files |
| Checkpoint | `artifacts/checkpoints/linregdroid.pth` | MLR weights |
| Metrics | `artifacts/metrics/{test_results,evaluation_results,parity_report}.json` | Test F1 0.709 (temporal_holdout, n=1,528) |
| ONNX export | `artifacts/export/linregdroid_permission/` | Full bundle + parity samples |
| P8 parity | `artifacts/metrics/parity_report.json` | **PASS** (max Δ = 0.0) |
| Shared offline JSON | `Shared_pipeline_Files/results/offline/linregdroid_permission_test_*.json` | |
| Android assets | `vigidroid/app/src/main/assets/models/linregdroid_permission/` | Staged |

---

### MLDP-pruned permissions — `mldp_pruned_permission` (`permission_extractor/`)

| Output | Location | Notes |
|--------|----------|-------|
| Dataset index | `artifacts/dataset_index.csv` | Full corpus |
| Split lists | `artifacts/splits/{train,val,dev_test,temporal_holdout}.txt` | 8,400 / 1,800 / 1,800 / 1,528 |
| MLDP selection | `artifacts/mldp/{selected_permissions,selection_validation,prnr,spr,pmar}.json` | Frozen set S (≤40 perms) |
| Transactions | `artifacts/transactions/` | 13,528 transaction files (P2) |
| Processed shards | `artifacts/processed/shards/` | Pruned 40-dim vectors |
| Checkpoint | `artifacts/checkpoints/mldp_pruned.pth` | Selected: `tiny_mlp` |
| Metrics | `artifacts/metrics/{test_results,evaluation_results,parity_report}.json` | Test F1 0.844 (n=1,528) |
| ONNX export | `artifacts/export/mldp_pruned_permission/` | Full bundle |
| P8 parity | `artifacts/metrics/parity_report.json` | **PASS** (max Δ ≈ 6.0e-8) |
| Shared offline JSON | `Shared_pipeline_Files/results/offline/mldp_pruned_permission_test_*.json` | |
| Android assets | `vigidroid/app/src/main/assets/models/mldp_pruned_permission/` | Staged |

---

### MLDP + Dex header cascade — `mldp_dexheader_cascade`

| Output | Location | Notes |
|--------|----------|-------|
| APK index | `artifacts/manifests/apk_index.csv`, `apk_index_summary.json` | 13,528 APKs |
| Split lists | `artifacts/splits/{train,val,test}.txt` | 10,800 / 1,200 / 1,528 |
| Feature tensors | `artifacts/processed/features_{train,val,test}.pt` | x_S (22) + H (104) fused dim 126 |
| MLDP + dex norm | `artifacts/processed/{mldp_permission_vocab,normalization_header,mldp_trace}.json` | Train-only |
| Checkpoints | `artifacts/checkpoints/{mode_a_best,stage1_best,ablation_*}.pt` | Mode A + Mode B Stage-1 |
| Metrics | `artifacts/metrics/{test_results,epochs,training_run_info,thresholds,parity_report}.json` | Mode A F1 0.911; Mode B e2e F1 0.938 |
| ONNX export | `artifacts/export/mldp_dexheader_cascade/` | `mode_a/model.onnx`, `mode_b/{stage1_mldp,stage2_mlp_header}.onnx`, features, 10 parity samples |
| P8 parity | `artifacts/metrics/parity_report.json` | **PASS** (max Δ ≈ 1.2e-7) |
| Android assets | `vigidroid/app/src/main/assets/models/mldp_dexheader_cascade/` | `mode_a/`, `mode_b/`, `features/`, `parity_samples/` |
| Android scripts | `Android_Works/run_mldp_dexheader_a{1,2,4}.sh`, `stage_mldp_dexheader_cascade.sh` | |

---

## 2) P1–P8 on full dataset & temporal split audit

**Full dataset criterion:** preprocessing/indexing covers **13,528 APKs** from `thesis_full_dataset`.

### Per-phase status

| Phase | BM1 | Pattern A | Pattern B | Broadcast | LinReg | MLDP-pruned | Cascade |
|-------|-----|-----------|-----------|-----------|--------|-------------|---------|
| **P1** Index | ✓ 13,528 | ✓ 13,528 | ✗ not run | ✓ 13,528 | ✓ 13,528 | ✓ 13,528 | ✓ 13,528 |
| **P2** Preprocess | ✓ 13,528 | ✓ 13,531 shards | ✗ | ✓ | ✓ ~13,534 shards | ✓ ~13,533 shards | ✓ |
| **P3** DataLoader | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| **P4** Model def | ✓ | ✓ | ✓ (code) | ✓ | ✓ | ✓ | ✓ |
| **P5** Train | ✓ 50 ep | ✓ 80 ep | ✗ | ✓ (1 ep quick) | ✓ | ✓ | ✓ (3 ep quick) |
| **P6** Evaluate | ✓ | ✓ (informal path) | ✗ | ✓ | ✓ | ✓ | ✓ |
| **P7** ONNX export | ✓ | ✗ local | ✗ | ✓ | ✓ | ✓ | ✓ |
| **P8** Parity | ✓ | ✗ local | ✗ | ✓ | ✓ | ✓ | ✓ |

### Train / validation / test separation (2020–2021 vs 2022–2023)

| Model | Train (2020–2021) | Validation | Test (2022–2023) | Train/test year leakage? | Val vs test disjoint? | 2022 vs 2023 separate? |
|-------|-------------------|------------|------------------|--------------------------|----------------------|------------------------|
| **BM1** | 10,800 (90% of dev years) | 1,200 (10% stratified from 2020–21) | 1,528 (all 2022+2023) | **No** — test years excluded from train | **Yes** — val ⊂ train years, test ⊂ holdout years | **No** — single merged test pool |
| **Pattern A** | 10,800 | 1,200 (from 2020–21) | 1,528 (all 2022+2023) | **No** | **Yes** (same policy as BM1) | **No** |
| **Pattern B** | — | — | — | Not exercised | — | — |
| **Broadcast** | 12,000 (all 2020–21) | 764 (50% of holdout) | 764 (other 50% of holdout) | **No** | **Yes** — stratified partition of 2022+2023 | **No** — years pooled before 50/50 split |
| **LinRegDroid** | 8,400 (70% of 2020–21) | 1,800 (15% of 2020–21) | 1,528 (`temporal_holdout` = all 2022+2023) | **No** | **Yes** — val ⊂ dev years, test ⊂ holdout years | **No** — merged holdout |
| **MLDP-pruned** | 8,400 | 1,800 | 1,528 (`temporal_holdout`) | **No** | **Yes** | **No** |
| **Cascade** | 10,800 | 1,200 (10% from train years) | 1,528 (all 2022+2023) | **No** | **Yes** | **No** — but per-year counts logged in `apk_index_summary.json` (2022: 1,010 test; 2023: 518 test) |

### Split policy notes

1. **Correct temporal holdout (train 2020–2021, evaluate on 2022–2023):** All models that completed P6 respect this at the year level. No model trains on 2022 or 2023 APKs.

2. **Validation source:**
   - **BM1 / Pattern A / Cascade:** val is a **random 10% holdout from train years** (early stopping + threshold tuning).
   - **Broadcast:** val and test are both drawn from **2022+2023 only**, split 50/50 — the only model with a **disjoint val/test within the holdout years**.
   - **LinReg / MLDP-pruned:** val is from **2020–21 development pool** (15%); primary thesis test is `temporal_holdout` (all 2022+2023). An extra `dev_test` split (15% of 2020–21) exists for in-distribution checks only.

3. **Phase 1 tracker gap ([`ongoing_task_tracker_thesis.md`](ongoing_task_tracker_thesis.md)):** None of the models currently expose **separate 2022-only vs 2023-only** validation and test partitions. Holdout years are either merged into one test set (BM1, Pattern A, Cascade, LinReg, MLDP) or merged then re-split into val/test halves (Broadcast).

4. **Train-only feature fitting:** Vocab, MLDP set S, permission vocab, and dex min–max normalization are built from **train split only** (verified in configs and `apk_index_summary.json` notes).

5. **Quick-train caveat:** Broadcast (`epochs_configured: 1`) and Cascade (`epochs_configured: 3`) used abbreviated training runs — metrics exist on full preprocessed data but may not reflect converged models.

---

## 3) A1–A4 Android integration progress

| Model | A1 Extractor | A2 ONNX runner | A3 Scan registration | A4 Parity test | Assets in app | Device verified |
|-------|--------------|----------------|------------------------|----------------|---------------|-----------------|
| **BM1** `mlp_header` | `DexHeaderFeatureExtractor` ✓ | `MlpHeaderOnnxRunner` ✓ | `ScanService.initMlpHeaderPipeline()` ✓ | `MlpHeaderParityTest` ✓ | ✓ | Bulk eval pending |
| **Pattern A** `early_fusion_dex_manifest` | `DexHeaderFeatureExtractor` + `ManifestBowExtractor` ✓ | `PatternAOnnxRunner` ✓ | `ScanService.initPatternAPipeline()` ✓ | `PatternAParityTest` ✓ | ✓ (stale ONNX) | Not confirmed |
| **Pattern B** `dual_branch_dex_manifest` | Reuses A extractors ✓ | `PatternBOnnxRunner` ✓ | `ScanService.initPatternBPipeline()` ✓ | `PatternBParityTest` ✓ | ✓ (stale ONNX) | Not confirmed |
| **Broadcast** `broadcast_mldp_hybrid` | `BroadcastMldpHybridExtractor` ✓ | `BroadcastMldpHybridOnnxRunner` ✓ | `ScanService.initBroadcastMldpHybridPipeline()` ✓ | `BroadcastMldpHybridA4ParityTest` ✓ | ✓ | PC parity pass; device gate script exists |
| **LinRegDroid** `linregdroid_permission` | `LinRegPermissionExtractor` ✓ | `LinRegDroidOnnxRunner` ✓ | `ScanService.initLinRegPermissionPipeline()` ✓ | `LinRegDroidParityTest` ✓ | ✓ | PC parity pass (Δ=0); bulk device eval pending |
| **MLDP-pruned** `mldp_pruned_permission` | `MldpPrunedPermissionExtractor` ✓ | `MldpPrunedOnnxRunner` ✓ | `ScanService.initMldpPrunedPermissionPipeline()` ✓ | `MldpPrunedParityTest` ✓ | ✓ | PC parity pass; bulk device eval pending |
| **Cascade Mode A** `mldp_dexheader_cascade_mode_a` | `MldpDexHeaderExtractor` ✓ | `MldpDexHeaderModeAOnnxRunner` ✓ | `ScanService.initMldpDexHeaderCascadePipeline()` ✓ | `MldpDexHeaderA4ParityTest` ✓ | ✓ | Dedicated A1/A2/A4 scripts |
| **Cascade Mode B** `mldp_dexheader_cascade_mode_b` | Same extractor (perm + optional dex) ✓ | `MldpDexHeaderModeBOnnxRunner` ✓ | Cascade orchestration in `ScanService.runMldpDexHeaderCascade()` ✓ | Covered in `MldpDexHeaderA4ParityTest` ✓ | ✓ | Two-stage thresholds in `mode_b/thresholds.json` |
| **Fusion** `dexheader_broadcast_fusion` | `DexheaderBroadcastFusionExtractor` ✓ | `DexheaderBroadcastFusionOnnxRunner` ✓ | `ScanService.initDexheaderBroadcastFusionPipeline()` ✓ | `DexheaderBroadcastFusionA4ParityTest` ✓ | ✓ | Fixed in P1 (`f != java.lang.Long`); A4 in `run_all_a4_gates.sh` |

**Registry:** All models above are listed in `vigidroid/app/src/main/java/com/msh/vigidroid/ModelRegistry.java`.

**Runtime fix (2026-06-09):** See [`app_runtime_fixing.md`](app_runtime_fixing.md). P1 exit scan: 11/11 legacy stages OK on `scan_1514_malware.apk`. Cascade is now the default scan mode in the UI. Debug **Model health** screen runs ONNX parity for Mode A, broadcast hybrid, and fusion.

**JUnit / instrumented tests:**

| Test class | Scope |
|------------|-------|
| `DexHeaderFeatureExtractorTest` | BM1 / shared header A1 (JVM) |
| `ManifestBowExtractorTest` | Pattern A/B BoW A1 (JVM) |
| `LinRegPermissionExtractorTest`, `MldpPrunedPermissionExtractorTest`, `BroadcastMldpHybridExtractorTest`, `MldpDexHeaderExtractorTest` | Per-model A1 (JVM) |
| `MldpDexHeaderA1ParityTest` | Cascade extraction parity (instrumented) |
| `MldpDexHeaderA2ParityTest` | Cascade ONNX vectors (instrumented) |
| `*ParityTest` / `*A4ParityTest` | End-to-end score parity ±1e-4 (instrumented) |

**Outstanding Android work (cross-model):**

- [ ] Re-export and re-stage **Pattern A** and **Pattern B** after full-corpus train (current app assets predate latest Pattern A checkpoint).
- [ ] Run **Pattern B** PC pipeline P1–P8 (no local `artifacts/` today).
- [ ] Bulk **A3 device scans** on `device_eval_manifest.csv` (~400 APKs) for thesis latency/memory plots.
- [ ] Confirm all **A4 instrumented tests** pass on target device (POCO F3) before trusting on-device metrics.
- [ ] Implement **year-specific 2022 vs 2023 val/test** splits if Phase 1 tracker requirement is adopted repo-wide.

---

## Quick reference — primary test metrics (temporal holdout)

| Model | n (test) | Accuracy | F1 | ROC-AUC |
|-------|----------|----------|-----|---------|
| BM1 `mlp_header` | 1,528 | 0.966 | 0.944 | 0.963 |
| Pattern A `early_fusion_dex_manifest` | 1,528 | 0.961 | 0.936 | 0.982 |
| Pattern B | — | — | — | — |
| Broadcast `broadcast_mldp_hybrid` | 764 | 0.923 | 0.862 | 0.923 |
| LinRegDroid | 1,528 | 0.855 | 0.709 | 0.731 |
| MLDP-pruned | 1,528 | 0.914 | 0.844 | 0.902 |
| Cascade Mode A | 1,528 | 0.946 | 0.911 | 0.931 |
| Cascade Mode B (e2e) | 1,528 | 0.962 | 0.938 | — |

*Broadcast test n=764 reflects its 50% holdout test split, not the full 1,528 merged holdout used by BM1/Pattern A/Cascade/LinReg/MLDP.*

---

## Related docs

- Tracker: [`ongoing_task_tracker_thesis.md`](ongoing_task_tracker_thesis.md)
- MoMo completion log: [`momo_corrections.md`](momo_corrections.md)
- Android step guides: [`Android_Works/BM1_android_steps.md`](Android_Works/BM1_android_steps.md), [`patternA_android_steps.md`](Android_Works/patternA_android_steps.md), [`patternB_android_steps.md`](Android_Works/patternB_android_steps.md)
- Pattern A eval crash postmortem: [`full_combined_pipeline_approach/pipelineA-runfix1.md`](Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/pipelineA-runfix1.md)
