#!/usr/bin/env python3
"""Phase 5: generate THESIS_SNIPPET.md from output_archives/<run_id>/."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_snippet(archive_dir: Path) -> str:
    archive_dir = archive_dir.resolve()
    manifest_path = archive_dir / "RUN_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}")

    manifest = _load(manifest_path)
    labels = _load(archive_dir / "corpus_stats" / "label_distribution.json")
    val_metrics = _load(archive_dir / "metrics" / "metrics_val.json")
    metrics = val_metrics.get("metrics", {})
    cm = val_metrics.get("confusion_matrix", [[0, 0], [0, 0]])

    pre = manifest.get("preprocessing", {})
    train = manifest.get("training", {})
    run_id = manifest.get("run_id", archive_dir.name)
    apk_root = manifest.get("canonical_apk_root") or manifest.get("apk_root", "—")
    git_commit = (manifest.get("git_commit") or "—")[:12]

    parity_path = archive_dir / "parity" / "parity_report.json"
    parity_line = "Parity not run."
    if parity_path.is_file():
        parity = _load(parity_path)
        parity_line = (
            f"PyTorch vs ONNX max abs diff = {parity['pytorch_vs_onnx']['max_abs_diff']:.2e} "
            f"({'PASS' if parity.get('passed') else 'FAIL'}, tolerance {parity.get('tolerance')})."
        )

    export_manifest_path = archive_dir / "export" / "export_manifest.json"
    onnx_line = "ONNX export not found in archive."
    if export_manifest_path.is_file():
        exp = _load(export_manifest_path)
        onnx_line = (
            f"`model.onnx` opset {exp.get('opset_version')}, "
            f"input `{exp['inputs'][0]['shape']}` float32 → malware probability."
        )

    year_counts = {}
    year_path = archive_dir / "corpus_stats" / "year_folder_counts.json"
    if year_path.is_file():
        year_counts = _load(year_path)

    year_rows = "\n".join(f"| {y} | {c:,} |" for y, c in sorted(year_counts.items()))

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return f"""# Base Model 1 (MLP-H) — Thesis snippet

**Run:** `{run_id}` · **Generated:** {generated}  
**Archive:** `output_archives/{run_id}/`  
**Reproducibility:** see [Appendix: run manifest](#appendix-run-manifest) (`RUN_MANIFEST.json`).

---

## 1. Dataset

| Item | Value |
|------|-------|
| APK root (canonical) | `{apk_root}` |
| APKs preprocessed | {pre.get('successful', labels.get('total', '—')):,} (failed: {pre.get('failed', 0)}) |
| Benign / malware | {labels['benign']:,} / {labels['malware']:,} |
| Labeling | Parent folder name (`benign` vs `malware`) |
| Year folders (APK path) | See table below |

| Year | APK count |
|------|-----------|
{year_rows or "| — | — |"}

**Caveats:** Corpus size is **13,528** APKs, not the full ~40k MSFDroid-scale set cited in the paper. Results apply to this corpus only. Train/val split matches Pattern A/B: **temporal year holdout** (train 2020–2021, test 2022–2023).

---

## 2. Features (Dex header, D3)

| Item | Value |
|------|-------|
| Domain | `dex_header_d3` |
| Raw feature dim | 104 (Dex header bytes 8–111, min–max normalized) |
| Multidex aggregation | `{pre.get('multidex_mode', 'sum')}` |
| Preprocessing cache version | {pre.get('cache_version', 2)} |
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
| Hidden dim | {train.get('hidden_dim', 128)} |

Deployment: {onnx_line}

---

## 4. Training

| Hyperparameter | Value |
|----------------|-------|
| Loss | BCE |
| Optimizer | SGD (lr=0.005, momentum 0.9) |
| LR schedule | StepLR, ×0.5 every 10 epochs |
| Batch size | {train.get('batch_size', 16)} |
| Epochs | {train.get('total_epochs', 50)} |
| Train / val samples | {train.get('train_samples', '—'):,} / {train.get('val_samples', '—'):,} |
| Device | {train.get('device', 'cuda')} ({train.get('gpu_name', '—')}) |
| Checkpoint | `artifacts/checkpoints/latest_checkpoint.pth` |

Full config snapshot: `output_archives/{run_id}/config/default.yaml.snapshot`.

---

## 5. Validation results (held-out 20%)

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 0):.4f} |
| F1 (malware) | {metrics.get('f1', 0):.4f} |
| ROC-AUC | {metrics.get('roc_auc', 0):.4f} |
| BCE loss | {val_metrics.get('loss', 0):.4f} |
| Decision threshold | {val_metrics.get('threshold', 0.5)} |
| Val samples | {val_metrics.get('n_samples', 0):,} |

**Confusion matrix** (rows=true, cols=predicted; benign first):

| | Pred benign | Pred malware |
|---|-------------|--------------|
| True benign | {cm[0][0]} | {cm[0][1]} |
| True malware | {cm[1][0]} | {cm[1][1]} |

Figures: `output_archives/{run_id}/figures/` (`loss_curves.png`, `metrics_vs_epoch.png`, `roc_curve_val.png`, `confusion_matrix_val.png`, corpus plots).

---

## 6. Export & parity

| Check | Result |
|-------|--------|
| ONNX bundle | `output_archives/{run_id}/export/` |
| Parity (PyTorch vs ONNX) | {parity_line} |

---

## 7. Limitations & honesty notes

- **Split policy:** Random 80/20 on preprocessed APKs; no held-out years or families.
- **Class imbalance:** ~74% benign; F1 and threshold 0.5 should be reported together.
- **Corpus scope:** 13.5k APKs; do not claim full-dataset paper numbers without retraining.
- **Failed APKs:** {pre.get('failed', 0)} in this run (`artifacts/failed_apks.log` if any).
- **Git commit at archive:** `{git_commit}`

---

## 8. Figures for thesis (manual copy)

Copy selected PNGs from the archive into your thesis `figures/` directory:

```bash
RUN="{run_id}"
SRC="Dex_header_paper_implementation/only_base1_model/output_archives/${{RUN}}/figures"
# Example (adjust THESIS_FIGS to your LaTeX tree):
# cp "$SRC/loss_curves.png" "$THESIS_FIGS/bm1_loss_curves.png"
# cp "$SRC/roc_curve_val.png" "$THESIS_FIGS/bm1_roc_val.png"
# cp "$SRC/confusion_matrix_val.png" "$THESIS_FIGS/bm1_confusion_val.png"
```

See `figures/figure_index.json` in the archive for captions.

---

## Appendix: run manifest

The authoritative reproducibility record for this run is:

**`output_archives/{run_id}/RUN_MANIFEST.json`**

It links preprocessing counts, training hyperparameters, final val metrics, artifact paths, and (after Phase 1) SHA-256 checksums. Verify integrity:

```bash
cd Dex_header_paper_implementation/only_base1_model
sha256sum -c output_archives/{run_id}/RUN_MANIFEST.sha256
```

Related docs: `BM1_running_guide.md`, `BM1_remaining.md`, `output_archives/README.md`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate THESIS_SNIPPET.md for an archive run.")
    parser.add_argument("--archive-dir", type=Path, default=None)
    args = parser.parse_args()

    archive_dir = args.archive_dir
    if archive_dir is None:
        latest = ROOT / "output_archives" / "LATEST_RUN.txt"
        if not latest.is_file():
            raise SystemExit("Set --archive-dir or create output_archives/LATEST_RUN.txt")
        archive_dir = ROOT / "output_archives" / latest.read_text(encoding="utf-8").strip()

    text = generate_snippet(archive_dir)
    out = archive_dir / "THESIS_SNIPPET.md"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
