# BM1 — Remaining Tasks (running list)

**Scope:** `only_base1_model/` only. No D1/D2, Pattern A/B, or VigiDroid Java.

**Canonical dataset:** `/mnt/Files/FromLaptop/thesis_full_dataset` (13,528 APKs)

**Canonical run (current):** `output_archives/run_20260524_fresh_logged` — see `output_archives/LATEST_RUN.txt`

**Already complete:** `BM1_running_guide.md` Steps 0–6; P0, P2–P6; `artifacts/metrics/`; 50-epoch fresh CUDA train (ACC 0.9678, F1 0.9364, AUC 0.9829 on val).

---

## Phase 1 — Archive completion ✅

**Goal:** Lock the finished run under `output_archives/` with checksums and a reproducible manifest.

**Completed:** 2026-05-24 for `run_20260524_fresh_logged`

| ID | Task | Status |
|----|------|--------|
| 1.1 | `RUN_MANIFEST.sha256` for core artifacts (`.pt`, `.pth`, key JSON) | ✅ |
| 1.2 | Update `RUN_MANIFEST.json` (canonical `apk_root`, archive index, phase notes) | ✅ |
| 1.3 | `scripts/archive_run.sh` + `scripts/finalize_bm1_archive.py` | ✅ |
| 1.4 | `output_archives/README.md` — layout + how to use `BM1_ARCHIVE=1` | ✅ |
| 1.5 | Placeholders: `figures/README.md`, `export/README.md`, `parity/README.md` | ✅ |

**Verify:**

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
sha256sum -c output_archives/run_20260524_fresh_logged/RUN_MANIFEST.sha256
```

**Acceptance:** Anyone can verify artifact integrity via `sha256sum -c RUN_MANIFEST.sha256`.

---

## Phase 2 — Figures & analysis ✅

**Goal:** Thesis-ready plots from `epochs.jsonl` and final checkpoint.

**Completed:** 2026-06-02 for `run_20260524_fresh_logged`

| ID | Task | Status |
|----|------|--------|
| 2.1 | `scripts/plot_bm1_results.py` | ✅ |
| 2.2 | `figures/loss_curves.png` | ✅ |
| 2.3 | `figures/metrics_vs_epoch.png` | ✅ |
| 2.4 | `figures/label_distribution.png`, `dex_count_histogram.png` | ✅ |
| 2.5 | `figures/roc_curve_val.png`, `confusion_matrix_val.png` | ✅ |
| 2.6 | `figures/figure_index.json` | ✅ |

**Verify:**

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
RUN="$(cat output_archives/LATEST_RUN.txt)"
../../thesis_venv/bin/python scripts/plot_bm1_results.py \
  --archive-dir "output_archives/${RUN}" \
  --checkpoint artifacts/checkpoints/latest_checkpoint.pth
ls -la "output_archives/${RUN}/figures/"
```

---

## Phase 3 — ONNX export (P7) ✅

**Goal:** Deployment bundle under `artifacts/export/mlp_header/`.

**Completed:** 2026-06-02 for `run_20260524_fresh_logged`

| ID | Task | Status |
|----|------|--------|
| 3.1 | `scripts/export_onnx.py` | ✅ |
| 3.2 | `model.onnx` (input `[1,104]` float32, opset 14) | ✅ |
| 3.3 | `export_manifest.json` | ✅ |
| 3.4 | `thresholds.json` | ✅ |
| 3.5 | `features/normalization_header.json` | ✅ |
| 3.6 | `parity_samples/` (vectors + expected scores) | ✅ |
| 3.7 | Copy bundle → `output_archives/<run>/export/` | ✅ |

**Verify:**

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
../../thesis_venv/bin/python scripts/export_onnx.py \
  --checkpoint artifacts/checkpoints/latest_checkpoint.pth
ls -la artifacts/export/mlp_header/
```

---

## Phase 4 — Parity (P8) ✅

**Goal:** PyTorch vs ONNX agreement on sample vectors.

**Completed:** 2026-06-02 for `run_20260524_fresh_logged`

| ID | Task | Status |
|----|------|--------|
| 4.1 | `scripts/parity_check_onnx.py` | ✅ |
| 4.2 | `parity/parity_report.json` (max diff &lt; 1e-4) | ✅ |
| 4.3 | `parity/sample_vectors.npz` | ✅ |

**Verify:**

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
../../thesis_venv/bin/python scripts/parity_check_onnx.py \
  --bundle artifacts/export/mlp_header
```

---

## Phase 5 — Thesis pack ✅

**Goal:** Human-readable summary for the thesis chapter.

**Completed:** 2026-06-02 for `run_20260524_fresh_logged`

| ID | Task | Status |
|----|------|--------|
| 5.1 | `output_archives/<run>/THESIS_SNIPPET.md` | ✅ |
| 5.2 | Copy figure PNGs to thesis `figures/` (manual) | ✅ documented in snippet |
| 5.3 | Appendix pointer to `RUN_MANIFEST.json` | ✅ |

**Verify:**

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
../../thesis_venv/bin/python scripts/generate_thesis_snippet.py
cat output_archives/$(cat output_archives/LATEST_RUN.txt)/THESIS_SNIPPET.md
../../thesis_venv/bin/python scripts/render_bm1_remaining_html.py
```

---

## Phase 6 — Optional (only if needed)

| ID | Task |
|----|------|
| 6.1 | Re-run pipeline with `APK_ROOT=/mnt/Files/FromLaptop/thesis_full_dataset` (new archive id) |
| 6.2 | `metrics_train.json` (`run_evaluate.sh --split train`) |
| 6.3 | Saved train/val index files in `artifacts/splits/` |
| 6.4 | Temporal holdout (e.g. test year 2023) |
| 6.5 | Multidex ablation `primary_only` |
| 6.6 | Threshold sweep script + plot |

---

## Execution order

```text
Phase 1  →  archive + checksums     [THIS FILE: complete first]
Phase 2  →  figures
Phase 3  →  ONNX export
Phase 4  →  parity
Phase 5  →  thesis snippet
Phase 6  →  optional
```

---

*Update Status column as phases complete.*
