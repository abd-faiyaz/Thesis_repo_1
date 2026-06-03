#!/usr/bin/env python3
"""Phase 2: thesis-ready figures from BM1 training logs and validation checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib required for Phase 2 figures: pip install matplotlib"
    ) from exc

from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_epochs_path(archive_dir: Path, cfg_metrics: Path) -> Path:
    candidates = [
        archive_dir / "metrics" / "epochs.jsonl",
        cfg_metrics / "epochs.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "epochs.jsonl not found under archive metrics/ or artifacts/metrics/"
    )


def plot_loss_curves(epochs: list[dict[str, Any]], out_path: Path) -> None:
    xs = [int(r["epoch"]) for r in epochs]
    train = [float(r["train_loss"]) for r in epochs]
    val = [float(r["val_loss"]) for r in epochs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, train, label="Train loss", color="#1f77b4", linewidth=1.8)
    ax.plot(xs, val, label="Val loss", color="#ff7f0e", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("BM1 (MLP-H): training and validation loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_vs_epoch(epochs: list[dict[str, Any]], out_path: Path) -> None:
    xs = [int(r["epoch"]) for r in epochs]
    acc = [float(r["accuracy"]) for r in epochs]
    f1 = [float(r["f1"]) for r in epochs]
    auc = [float(r["roc_auc"]) for r in epochs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, acc, label="Accuracy", color="#2ca02c", linewidth=1.8)
    ax.plot(xs, f1, label="F1", color="#d62728", linewidth=1.8)
    ax.plot(xs, auc, label="ROC-AUC", color="#9467bd", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("BM1 (MLP-H): validation metrics per epoch")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_label_distribution(label_dist: dict[str, int], out_path: Path) -> None:
    labels = ["Benign", "Malware"]
    counts = [int(label_dist["benign"]), int(label_dist["malware"])]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, counts, color=["#4c78a8", "#e45756"])
    ax.set_ylabel("APK count")
    ax.set_title("Corpus label distribution (preprocessed)")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dex_histogram(dex_counts: dict[str, int], out_path: Path) -> None:
    pairs = sorted((int(k), int(v)) for k, v in dex_counts.items())
    xs = [str(k) for k, _ in pairs]
    ys = [v for _, v in pairs]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xs, ys, color="steelblue")
    ax.set_xlabel("DEX files per APK")
    ax.set_ylabel("APK count")
    ax.set_title("Distribution of multidex APKs in corpus")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Validation ROC curve (MLP-H)")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return auc


def plot_confusion_matrix(
    cm: list[list[int]],
    out_path: Path,
    *,
    threshold: float,
) -> None:
    arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=arr,
        display_labels=["Benign", "Malware"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Validation confusion matrix (threshold = {threshold})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def collect_val_predictions(
    checkpoint: Path,
    *,
    config_path: Path | None,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    import torch

    from src.config import load_config
    from src.data.dataloaders import build_dataloaders_from_config
    from src.models.mlp_header import build_mlp_header
    from src.training.checkpoint import load_checkpoint, restore_from_checkpoint
    from src.training.evaluate import collect_predictions
    from src.training.setup import build_training_objects

    cfg = load_config(config_path)
    _, val_loader, feature_dim = build_dataloaders_from_config(cfg)

    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    hidden_dim = int(ckpt.get("hidden_dim", cfg.model.get("hidden_dim", 128)))
    feature_dim = int(ckpt.get("feature_dim", feature_dim))
    model = build_mlp_header(input_dim=feature_dim, hidden_dim=hidden_dim)
    _, optimizer, scheduler, device = build_training_objects(cfg, model)
    ckpt = load_checkpoint(checkpoint, map_location=device)
    assert ckpt is not None
    restore_from_checkpoint(ckpt, model, optimizer, scheduler)

    threshold = float(cfg.evaluation.get("threshold", 0.5))
    if num_workers >= 0:
        val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    return (*collect_predictions(model, val_loader, device, threshold=threshold), threshold)


def write_figure_index(
    out_path: Path,
    entries: list[dict[str, Any]],
    *,
    archive_dir: Path,
    run_id: str,
) -> None:
    payload = {
        "generated_at": _utc_now(),
        "run_id": run_id,
        "archive_dir": str(archive_dir),
        "figures": entries,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_figures(
    archive_dir: Path,
    *,
    checkpoint: Path,
    config_path: Path | None = None,
    skip_inference: bool = False,
    num_workers: int = 0,
) -> list[Path]:
    archive_dir = archive_dir.resolve()
    figures_dir = archive_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = ROOT / "artifacts" / "metrics"
    epochs_path = resolve_epochs_path(archive_dir, metrics_dir)
    epochs = load_jsonl(epochs_path)

    label_path = archive_dir / "corpus_stats" / "label_distribution.json"
    dex_path = archive_dir / "corpus_stats" / "dex_file_counts.json"
    if not label_path.is_file():
        corpus = metrics_dir / "corpus_stats.json"
        if corpus.is_file():
            blob = load_json(corpus)
            label_dist = blob.get("label_distribution", blob)
            dex_counts = blob.get("dex_file_counts", {})
        else:
            raise FileNotFoundError("label_distribution.json not found in archive or metrics")
    else:
        label_dist = load_json(label_path)
        dex_counts = load_json(dex_path) if dex_path.is_file() else {}

    written: list[Path] = []
    specs: list[tuple[str, str, str, Path]] = []

    def emit(name: str, caption: str, source: str, plot_fn) -> None:
        path = figures_dir / name
        plot_fn(path)
        written.append(path)
        specs.append((name, caption, source, path))

    emit(
        "loss_curves.png",
        "Training and validation BCE loss across 50 epochs.",
        str(epochs_path.relative_to(ROOT)),
        lambda p: plot_loss_curves(epochs, p),
    )
    emit(
        "metrics_vs_epoch.png",
        "Validation accuracy, F1, and ROC-AUC recorded at each epoch.",
        str(epochs_path.relative_to(ROOT)),
        lambda p: plot_metrics_vs_epoch(epochs, p),
    )
    emit(
        "label_distribution.png",
        "Benign vs malware APK counts in the preprocessed corpus.",
        "corpus_stats/label_distribution.json",
        lambda p: plot_label_distribution(label_dist, p),
    )
    if dex_counts:
        emit(
            "dex_count_histogram.png",
            "Histogram of APKs binned by number of DEX files.",
            "corpus_stats/dex_file_counts.json",
            lambda p: plot_dex_histogram(dex_counts, p),
        )

    metrics_val_path = archive_dir / "metrics" / "metrics_val.json"
    threshold = 0.5
    if skip_inference and metrics_val_path.is_file():
        val_metrics = load_json(metrics_val_path)
        threshold = float(val_metrics.get("threshold", 0.5))
        cm = val_metrics["confusion_matrix"]
        emit(
            "confusion_matrix_val.png",
            f"Validation confusion matrix at decision threshold {threshold}.",
            str(metrics_val_path.relative_to(archive_dir)),
            lambda p: plot_confusion_matrix(cm, p, threshold=threshold),
        )
    else:
        y_true, y_pred, y_score, threshold = collect_val_predictions(
            checkpoint,
            config_path=config_path,
            num_workers=num_workers,
        )
        auc = plot_roc_curve(y_true, y_score, figures_dir / "roc_curve_val.png")
        written.append(figures_dir / "roc_curve_val.png")
        specs.append(
            (
                "roc_curve_val.png",
                f"Validation ROC curve (AUC = {auc:.4f}) from checkpoint inference.",
                str(checkpoint.relative_to(ROOT)),
                figures_dir / "roc_curve_val.png",
            )
        )

        from src.training.run_logging import build_confusion_matrix

        cm = build_confusion_matrix(y_true, y_pred)
        emit(
            "confusion_matrix_val.png",
            f"Validation confusion matrix at threshold {threshold}.",
            str(checkpoint.relative_to(ROOT)),
            lambda p: plot_confusion_matrix(cm, p, threshold=threshold),
        )

    run_id = archive_dir.name
    index_entries = [
        {
            "file": name,
            "path": f"figures/{name}",
            "caption": caption,
            "source": source,
        }
        for name, caption, source, _ in specs
    ]
    index_path = figures_dir / "figure_index.json"
    write_figure_index(
        index_path,
        index_entries,
        archive_dir=archive_dir,
        run_id=run_id,
    )
    written.append(index_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate BM1 Phase 2 thesis figures.")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="output_archives/<run_id>/ (default: LATEST_RUN.txt)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "checkpoints" / "latest_checkpoint.pth",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip ROC (and re-inferred CM); use metrics_val.json CM only",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for val inference (0 is safest for plotting)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.archive_dir is not None:
        archive_dir = args.archive_dir
    else:
        latest = ROOT / "output_archives" / "LATEST_RUN.txt"
        if not latest.is_file():
            raise SystemExit("Set --archive-dir or create output_archives/LATEST_RUN.txt")
        run_id = latest.read_text(encoding="utf-8").strip()
        archive_dir = ROOT / "output_archives" / run_id

    if not archive_dir.is_dir():
        raise SystemExit(f"Archive directory not found: {archive_dir}")

    paths = generate_figures(
        archive_dir,
        checkpoint=args.checkpoint.resolve(),
        config_path=args.config,
        skip_inference=args.skip_inference,
        num_workers=args.num_workers,
    )
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
