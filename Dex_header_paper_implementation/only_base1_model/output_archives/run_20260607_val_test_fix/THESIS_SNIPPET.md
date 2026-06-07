# Base Model 1 (MLP-H) — Thesis snippet

**Run:** `run_20260607_val_test_fix` · **Generated:** 2026-06-06  
**Archive:** `output_archives/run_20260607_val_test_fix/`  
**Reproducibility:** see [Appendix: run manifest](#appendix-run-manifest) (`RUN_MANIFEST.json`).

---

## 1. Dataset

| Item | Value |
|------|-------|
| APK root (canonical) | `/mnt/Files/FromLaptop/thesis_full_dataset` |
| APKs preprocessed | 13,528 (failed: 0) |
| Benign / malware | 10,050 / 3,478 |
| Labeling | Parent folder name (`benign` vs `malware`) |
| Year folders (APK path) | See table below |

| Year | APK count |
|------|-----------|
| 2020 | 8,000 |
| 2021 | 4,000 |
| 2022 | 1,010 |
| 2023 | 518 |

**Caveats:** Corpus size is **13,528** APKs, not the full ~40k MSFDroid-scale set cited in the paper. Results apply to this corpus only. Split matches Pattern A/B: **train 2020–2021**, **val holdout (~10% from 2020–2021)**, **test 2022–2023** (reported metrics are on test only).

---

## 2. Features (Dex header, D3)

| Item | Value |
|------|-------|
| Domain | `dex_header_d3` |
| Raw feature dim | 104 (Dex header bytes 8–111, min–max normalized) |
| Multidex aggregation | `sum` |
| Preprocessing cache version | 2 |
| Artifact | `dex_header_features.pt` |

Per-APK: extract all `classes*.dex` headers → normalize → aggregate by **sum** across DEX files → one 104-D vector per APK.

---

## 3. Model — MLP(H)

| Layer | Configuration |
|-------|----------------|
| Input | 104 |
| Block 1 | Linear(104→128) → BatchNorm → ReLU |
| Block 2 | Linear(128→128) → BatchNorm → ReLU |
| Output | Linear(128→1) → Sigmoid (malware probability) |
| Hidden dim | 128 |

Deployment: `model.onnx` opset 14, input `[1, 104]` float32 → malware probability.

---

## 4. Training

| Hyperparameter | Value |
|----------------|-------|
| Loss | BCE |
| Optimizer | SGD (lr=0.005, momentum 0.9) |
| LR schedule | StepLR, ×0.5 every 10 epochs |
| Batch size | 16 |
| Epochs | 50 |
| Train / val samples | 10,800 / 1,200 |
| Device | cuda (NVIDIA GeForce RTX 3060 Ti) |
| Checkpoint | `artifacts/checkpoints/latest_checkpoint.pth` |

Full config snapshot: `output_archives/run_20260607_val_test_fix/config/default.yaml.snapshot`.

---

## 5. Test results (test split — 2022–2023 holdout)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9660 |
| F1 (malware) | 0.9444 |
| ROC-AUC | 0.9631 |
| BCE loss | 0.2080 |
| Decision threshold | 0.5 |
| Test samples | 1,528 |

**Confusion matrix** (rows=true, cols=predicted; benign first):

| | Pred benign | Pred malware |
|---|-------------|--------------|
| True benign | 1034 | 16 |
| True malware | 36 | 442 |

Figures: `output_archives/run_20260607_val_test_fix/figures/` (`loss_curves.png`, `metrics_vs_epoch.png`, `roc_curve_val.png`, `confusion_matrix_val.png`, corpus plots).

---

## 6. Export & parity

| Check | Result |
|-------|--------|
| ONNX bundle | `output_archives/run_20260607_val_test_fix/export/` |
| Parity (PyTorch vs ONNX) | PyTorch vs ONNX max abs diff = 8.94e-08 (PASS, tolerance 0.0001). |

---

## 7. Limitations & honesty notes

- **Split policy:** Temporal train 2020–2021; val ~10% stratified from train years; test 2022–2023 (never used for training or checkpoint selection).
- **Class imbalance:** ~74% benign; F1 and threshold 0.5 should be reported together.
- **Corpus scope:** 13.5k APKs; do not claim full-dataset paper numbers without retraining.
- **Failed APKs:** 0 in this run (`artifacts/failed_apks.log` if any).
- **Git commit at archive:** `0fc5152a7b84`

---

## 8. Figures for thesis (manual copy)

Copy selected PNGs from the archive into your thesis `figures/` directory:

```bash
RUN="run_20260607_val_test_fix"
SRC="Dex_header_paper_implementation/only_base1_model/output_archives/${RUN}/figures"
# Example (adjust THESIS_FIGS to your LaTeX tree):
# cp "$SRC/loss_curves.png" "$THESIS_FIGS/bm1_loss_curves.png"
# cp "$SRC/roc_curve_val.png" "$THESIS_FIGS/bm1_roc_val.png"
# cp "$SRC/confusion_matrix_val.png" "$THESIS_FIGS/bm1_confusion_val.png"
```

See `figures/figure_index.json` in the archive for captions.

---

## Appendix: run manifest

The authoritative reproducibility record for this run is:

**`output_archives/run_20260607_val_test_fix/RUN_MANIFEST.json`**

It links preprocessing counts, training hyperparameters, final test metrics, artifact paths, and (after Phase 1) SHA-256 checksums. Verify integrity:

```bash
cd Dex_header_paper_implementation/only_base1_model
sha256sum -c output_archives/run_20260607_val_test_fix/RUN_MANIFEST.sha256
```

Related docs: `BM1_running_guide.md`, `BM1_remaining.md`, `output_archives/README.md`.
