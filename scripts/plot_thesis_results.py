#!/usr/bin/env python3
"""Thesis-ready figures (BM1 style) for any archived model profile."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from thesis_run_archive import archive_dir_for, resolve_run_id  # noqa: E402
from thesis_run_logging import ARCHIVE_PROFILES, ArchiveProfile  # noqa: E402

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib required: pip install matplotlib") from exc

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve


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


def metrics_dir_for(root: Path, profile: ArchiveProfile) -> Path:
    sub = profile.metrics_subdir
    if sub == "metrics":
        return root / "artifacts" / "metrics"
    return root / "artifacts" / sub


def find_metrics_file(
    root: Path,
    archive_dir: Path,
    profile: ArchiveProfile,
    name: str,
) -> Path | None:
    candidates = [
        archive_dir / "metrics" / name,
        metrics_dir_for(root, profile) / name,
        root / "artifacts" / "metrics" / name,
        root / "artifacts" / "checkpoints" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def plot_loss_curves(
    epochs: list[dict[str, Any]],
    out_path: Path,
    *,
    display_name: str,
) -> None:
    xs = [int(r["epoch"]) for r in epochs]
    train = [
        float(r.get("train_loss", r.get("loss", 0.0)))
        for r in epochs
    ]
    val = [float(r["val_loss"]) for r in epochs if "val_loss" in r]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, train, label="Train loss", color="#1f77b4", linewidth=1.8)
    if val:
        ax.plot(xs[: len(val)], val, label="Val loss", color="#ff7f0e", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{display_name}: training and validation loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_vs_epoch(
    epochs: list[dict[str, Any]],
    out_path: Path,
    *,
    display_name: str,
) -> None:
    xs = [int(r["epoch"]) for r in epochs]
    acc = [float(r["accuracy"]) for r in epochs if "accuracy" in r]
    f1 = [float(r["f1"]) for r in epochs if "f1" in r]
    auc = [float(r["roc_auc"]) for r in epochs if "roc_auc" in r]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if acc:
        ax.plot(xs[: len(acc)], acc, label="Accuracy", color="#2ca02c", linewidth=1.8)
    if f1:
        ax.plot(xs[: len(f1)], f1, label="F1", color="#d62728", linewidth=1.8)
    if auc:
        ax.plot(xs[: len(auc)], auc, label="ROC-AUC", color="#9467bd", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"{display_name}: validation metrics per epoch")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_label_distribution(label_dist: dict[str, int], out_path: Path) -> None:
    labels = ["Benign", "Malware"]
    counts = [int(label_dist.get("benign", 0)), int(label_dist.get("malware", 0))]

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


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path,
    *,
    display_name: str,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{display_name}: validation ROC curve")
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
    display_name: str,
    split: str = "val",
) -> None:
    arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=arr,
        display_labels=["Benign", "Malware"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(
        f"{display_name}: {split} confusion matrix (threshold = {threshold:.3f})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_label_distribution(
    root: Path,
    archive_dir: Path,
    profile: ArchiveProfile,
) -> dict[str, int]:
    label_path = archive_dir / "corpus_stats" / "label_distribution.json"
    if label_path.is_file():
        return load_json(label_path)

    corpus_path = find_metrics_file(root, archive_dir, profile, "corpus_stats.json")
    if corpus_path is not None:
        blob = load_json(corpus_path)
        return blob.get("label_distribution", blob)

    raise FileNotFoundError(
        "label_distribution not found in archive corpus_stats/ or artifacts metrics"
    )


def load_dex_counts(
    root: Path,
    archive_dir: Path,
    profile: ArchiveProfile,
) -> dict[str, int]:
    dex_path = archive_dir / "corpus_stats" / "dex_file_counts.json"
    if dex_path.is_file():
        return load_json(dex_path)

    corpus_path = find_metrics_file(root, archive_dir, profile, "corpus_stats.json")
    if corpus_path is not None:
        blob = load_json(corpus_path)
        counts = blob.get("dex_file_counts", {})
        if counts:
            return {str(k): int(v) for k, v in counts.items()}

    preprocess = find_metrics_file(root, archive_dir, profile, "preprocess_summary.json")
    if preprocess is not None:
        blob = load_json(preprocess)
        counts = blob.get("dex_file_counts", {})
        if counts:
            return {str(k): int(v) for k, v in counts.items()}

    return {}


def load_val_prediction_arrays(
    root: Path,
    archive_dir: Path,
    profile: ArchiveProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    val_scores_path = find_metrics_file(root, archive_dir, profile, "val_scores.json")
    if val_scores_path is not None:
        data = load_json(val_scores_path)
        labels = np.array([int(row["label"]) for row in data["rows"]], dtype=np.int64)
        scores = np.array([float(row["score"]) for row in data["rows"]], dtype=np.float64)
        threshold = float(data.get("threshold", 0.5))
        preds = (scores >= threshold).astype(np.int64)
        return labels, preds, scores, threshold

    metrics_val_path = find_metrics_file(root, archive_dir, profile, "metrics_val.json")
    if metrics_val_path is not None:
        data = load_json(metrics_val_path)
        cm = data.get("confusion_matrix")
        threshold = float(data.get("threshold", 0.5))
        if cm is not None:
            return None  # caller uses CM-only path

    return None


def confusion_matrix_from_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> list[list[int]]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm.astype(int).tolist()


def write_figure_index(
    out_path: Path,
    entries: list[dict[str, Any]],
    *,
    archive_dir: Path,
    run_id: str,
    model_id: str,
) -> None:
    payload = {
        "generated_at": _utc_now(),
        "run_id": run_id,
        "model_id": model_id,
        "archive_dir": str(archive_dir),
        "figures": entries,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_figures(
    root: Path,
    profile: ArchiveProfile,
    run_id: str,
) -> list[Path]:
    root = root.resolve()
    archive_dir = archive_dir_for(root, run_id)
    figures_dir = archive_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    display_name = profile.display_name
    written: list[Path] = []
    index_entries: list[dict[str, Any]] = []

    def emit(
        name: str,
        caption: str,
        source: str,
        plot_fn,
    ) -> None:
        path = figures_dir / name
        plot_fn(path)
        written.append(path)
        index_entries.append(
            {
                "file": name,
                "path": f"figures/{name}",
                "caption": caption,
                "source": source,
            }
        )

    epochs_path = find_metrics_file(root, archive_dir, profile, "epochs.jsonl")
    if epochs_path is not None:
        epochs = load_jsonl(epochs_path)
        if epochs:
            rel = str(epochs_path.relative_to(root)) if epochs_path.is_relative_to(root) else str(epochs_path)
            emit(
                "loss_curves.png",
                "Training and validation loss across epochs.",
                rel,
                lambda p: plot_loss_curves(epochs, p, display_name=display_name),
            )
            if any("f1" in row or "accuracy" in row for row in epochs):
                emit(
                    "metrics_vs_epoch.png",
                    "Validation accuracy, F1, and ROC-AUC recorded at each epoch.",
                    rel,
                    lambda p: plot_metrics_vs_epoch(epochs, p, display_name=display_name),
                )

    label_dist = load_label_distribution(root, archive_dir, profile)
    emit(
        "label_distribution.png",
        "Benign vs malware APK counts in the preprocessed corpus.",
        "corpus_stats/label_distribution.json",
        lambda p: plot_label_distribution(label_dist, p),
    )

    dex_counts = load_dex_counts(root, archive_dir, profile)
    if dex_counts:
        emit(
            "dex_count_histogram.png",
            "Histogram of APKs binned by number of DEX files.",
            "corpus_stats/dex_file_counts.json",
            lambda p: plot_dex_histogram(dex_counts, p),
        )

    val_arrays = load_val_prediction_arrays(root, archive_dir, profile)
    if val_arrays is not None:
        y_true, y_pred, y_score, threshold = val_arrays
        auc = plot_roc_curve(
            y_true,
            y_score,
            figures_dir / "roc_curve_val.png",
            display_name=display_name,
        )
        written.append(figures_dir / "roc_curve_val.png")
        index_entries.append(
            {
                "file": "roc_curve_val.png",
                "path": "figures/roc_curve_val.png",
                "caption": f"Validation ROC curve (AUC = {auc:.4f}).",
                "source": "metrics/val_scores.json",
            }
        )

        cm = confusion_matrix_from_arrays(y_true, y_pred)
        emit(
            "confusion_matrix_val.png",
            f"Validation confusion matrix at threshold {threshold:.3f}.",
            "metrics/val_scores.json",
            lambda p: plot_confusion_matrix(
                cm,
                p,
                threshold=threshold,
                display_name=display_name,
            ),
        )
    else:
        metrics_val_path = find_metrics_file(root, archive_dir, profile, "metrics_val.json")
        if metrics_val_path is not None:
            val_metrics = load_json(metrics_val_path)
            threshold = float(val_metrics.get("threshold", 0.5))
            cm = val_metrics.get("confusion_matrix")
            if cm is not None:
                rel = str(metrics_val_path.relative_to(archive_dir))
                emit(
                    "confusion_matrix_val.png",
                    f"Validation confusion matrix at threshold {threshold:.3f}.",
                    rel,
                    lambda p, cm=cm, threshold=threshold: plot_confusion_matrix(
                        cm,
                        p,
                        threshold=threshold,
                        display_name=display_name,
                    ),
                )

    index_path = figures_dir / "figure_index.json"
    write_figure_index(
        index_path,
        index_entries,
        archive_dir=archive_dir,
        run_id=run_id,
        model_id=profile.model_id,
    )
    written.append(index_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate BM1-style thesis figures for a model profile."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(ARCHIVE_PROFILES),
        required=True,
        help="Model archive profile key",
    )
    parser.add_argument("--root", type=Path, default=None, help="Model package root")
    parser.add_argument("--run-id", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    profile = ARCHIVE_PROFILES[args.profile]
    root = (args.root or Path.cwd()).resolve()
    run_id = resolve_run_id(root, profile, args.run_id)

    paths = generate_figures(root, profile, run_id)
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
