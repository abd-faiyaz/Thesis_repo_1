"""P5 — train fusion net with early stopping on val F1."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.data.dataloaders import build_dataloaders, compute_pos_weight
from src.training.checkpoint import (
    build_best_checkpoint,
    load_frozen_layout,
    save_best_checkpoint,
)
from src.training.loops import train_one_epoch, validation_epoch
from src.training.metrics import format_metrics
from src.training.setup import build_fusion_model, build_training_objects
from src.training.svm_baseline import run_paper_svm_baseline

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def train_fusion(cfg: PipelineConfig, *, epochs: int | None = None) -> dict:
    ensure_artifact_dirs(cfg)
    train_loader, val_loader, _, dex_dim, receiver_dim, balance_stats = build_dataloaders(cfg)

    pos_weight = float(balance_stats.get("pos_weight", {}).get("value", 0))
    if pos_weight <= 0:
        pos_weight = compute_pos_weight(train_loader.dataset.y)  # type: ignore[attr-defined]

    model = build_fusion_model(cfg, dex_dim=dex_dim, receiver_dim=receiver_dim)
    warm_started = bool(cfg.model.get("header_warm_start", True))

    criterion, optimizer, device = build_training_objects(cfg, model, pos_weight=pos_weight)

    total_epochs = int(epochs if epochs is not None else cfg.training.get("epochs", 60))
    patience = int(cfg.training.get("early_stop_patience", 6))
    threshold = float(cfg.evaluation.get("threshold", 0.5))

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_val_metrics: dict[str, float] = {}
    best_epoch = 0
    stale_epochs = 0

    metrics_dir = cfg.paths.metrics
    (metrics_dir / "epochs.jsonl").write_text("", encoding="utf-8")

    for epoch in range(total_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        val_loss, val_metrics = validation_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
            "ts": _utc_now(),
        }
        with (metrics_dir / "epochs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        val_f1 = float(val_metrics.get("f1", 0.0))
        print(f"Epoch {epoch + 1}/{total_epochs}  train_loss={train_loss:.4f}  val {format_metrics(val_metrics)}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stop at epoch {epoch + 1} (patience={patience})")
                break

    model.load_state_dict(best_state)
    receiver_vocab, layout = load_frozen_layout(cfg.paths.processed)
    payload = build_best_checkpoint(
        cfg,
        model,
        val_metrics=best_val_metrics,
        epochs_trained=best_epoch,
        receiver_vocab=receiver_vocab,
        feature_layout=layout,
        receiver_dim=receiver_dim,
        warm_started=warm_started,
    )
    ckpt_path = cfg.paths.checkpoints / "best.pt"
    save_best_checkpoint(ckpt_path, payload)

    run_info = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_metrics": best_val_metrics,
        "epochs_configured": total_epochs,
        "warm_started": warm_started,
        "pos_weight": pos_weight,
        "trained_at": _utc_now(),
    }
    (metrics_dir / "training_run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved checkpoint → {ckpt_path}")
    return run_info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P5 fusion training.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (smoke)")
    parser.add_argument("--skip-svm", action="store_true")
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    epochs = args.epochs
    if epochs is None and int(cfg.training.get("smoke_epochs", 0)) > 0:
        import os

        if os.environ.get("SMOKE", "0") == "1":
            epochs = int(cfg.training.get("smoke_epochs", 2))

    train_fusion(cfg, epochs=epochs)
    if not args.skip_svm:
        run_paper_svm_baseline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
