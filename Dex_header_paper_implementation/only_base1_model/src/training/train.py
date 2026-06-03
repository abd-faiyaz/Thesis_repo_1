"""Training loop with SGD, BCELoss, LR decay, tqdm, checkpoint resume (Phase 5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import ensure_artifact_dirs, load_config
from src.data.dataloaders import build_dataloaders_from_config
from src.models.mlp_header import build_mlp_header
from src.training.checkpoint import (
    build_checkpoint_state,
    load_checkpoint,
    restore_from_checkpoint,
    save_checkpoint,
)
from src.training.evaluate import format_metrics, validation_epoch
from src.training.loops import train_one_epoch
from src.training.run_logging import (
    finalize_run_manifest,
    log_checkpoint_summary,
    log_epoch,
    log_training_run_info,
    reset_epochs_log,
)
from src.training.setup import build_training_objects, resolve_device

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def run_training(
    cfg,
    *,
    epochs_override: int | None = None,
    fresh_start: bool = False,
) -> None:
    ensure_artifact_dirs(cfg)
    train_loader, val_loader, feature_dim = build_dataloaders_from_config(cfg)
    hidden_dim = int(cfg.model.get("hidden_dim", 128))

    checkpoint_path = cfg.paths.latest_checkpoint
    start_epoch = 0
    existing = None if fresh_start else load_checkpoint(checkpoint_path, map_location="cpu")

    if existing is not None:
        hidden_dim = int(existing.get("hidden_dim", hidden_dim))
        feature_dim = int(existing.get("feature_dim", feature_dim))

    model = build_mlp_header(input_dim=feature_dim, hidden_dim=hidden_dim)
    criterion, optimizer, scheduler, device = build_training_objects(cfg, model)

    if existing is not None:
        existing = load_checkpoint(checkpoint_path, map_location=device)
        if existing is not None:
            start_epoch = restore_from_checkpoint(existing, model, optimizer, scheduler)
            print(
                f"Resumed from {checkpoint_path} — starting at epoch {start_epoch + 1} "
                f"(last train_loss={existing.get('current_loss', existing.get('train_loss', 'n/a'))})"
            )

    total_epochs = int(epochs_override or cfg.training.get("epochs", 50))
    save_each_epoch = bool(cfg.training.get("checkpoint_every_epoch", True))
    threshold = float(cfg.evaluation.get("threshold", 0.5))
    resolved_device = resolve_device(str(cfg.training.get("device", "cuda")))

    reset_epochs_log(cfg, fresh=fresh_start)
    log_training_run_info(
        cfg,
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset),
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        start_epoch=start_epoch,
        total_epochs=total_epochs,
        fresh_start=fresh_start,
        device=str(resolved_device),
    )

    print(f"Device: {device} ({resolved_device})")
    print(f"Samples: train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    print(f"Epochs: {start_epoch + 1} → {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
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
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{total_epochs} — "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr:.6f} "
            f"{format_metrics(val_metrics)}"
        )
        log_epoch(
            cfg,
            epoch=epoch + 1,
            total_epochs=total_epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=lr,
            val_metrics=val_metrics,
        )

        if save_each_epoch:
            save_checkpoint(
                checkpoint_path,
                build_checkpoint_state(
                    next_epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                ),
            )

    print(f"Training complete. Latest checkpoint: {checkpoint_path}")
    log_checkpoint_summary(cfg, checkpoint_path)
    manifest = finalize_run_manifest(cfg)
    if manifest is not None:
        print(f"Run manifest updated: {manifest}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Base Model 1 (MLP(H)) on Dex header features.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing latest_checkpoint.pth and train from scratch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_training(cfg, epochs_override=args.epochs, fresh_start=args.fresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
