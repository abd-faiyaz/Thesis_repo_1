"""Training loop: SGD, BCEWithLogitsLoss, LR decay, tqdm, checkpoint resume (Phase 5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import ensure_artifact_dirs, load_config
from src.data.dataloaders import build_dataloaders_from_config
from src.models.dual_branch_net import build_dual_branch_net_from_config
from src.training.checkpoint import (
    build_checkpoint_state,
    load_checkpoint,
    restore_from_checkpoint,
    save_checkpoint,
)
from src.training.losses import build_criterion
from src.training.loops import train_one_epoch, validate_one_epoch
from src.training.setup import build_training_objects, resolve_device

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _require_manifests(cfg) -> None:
    if not cfg.paths.manifest_train.is_file() or not cfg.paths.manifest_val.is_file():
        raise FileNotFoundError(
            "Training requires Phase 2 manifests:\n"
            f"  {cfg.paths.manifest_train}\n"
            f"  {cfg.paths.manifest_val}\n"
            "Run ./scripts/run_preprocess.sh first (or use tests with synthetic shards)."
        )


def run_training(
    cfg,
    *,
    epochs_override: int | None = None,
    fresh_start: bool = False,
    resume_path: Path | None = None,
) -> None:
    ensure_artifact_dirs(cfg)
    _require_manifests(cfg)

    train_loader, val_loader, header_dim, bow_dim = build_dataloaders_from_config(cfg)
    model = build_dual_branch_net_from_config(cfg)
    optimizer, scheduler, device = build_training_objects(cfg, model)
    criterion = build_criterion(cfg, device)

    latest_path = resume_path or cfg.paths.latest_checkpoint
    best_path = cfg.paths.best_checkpoint

    start_epoch = 0
    global_step = 0
    best_val_loss: float | None = None
    existing = None if fresh_start else load_checkpoint(latest_path, map_location="cpu")

    if existing is not None:
        start_epoch, global_step, best_val_loss = restore_from_checkpoint(
            existing, model, optimizer, scheduler
        )
        model.to(device)
        print(
            f"Resumed from {latest_path} — epoch {start_epoch + 1} "
            f"(last train_loss={existing.get('train_loss', 'n/a')})"
        )

    total_epochs = int(epochs_override or cfg.training.get("epochs", 80))
    save_each_epoch = bool(cfg.training.get("checkpoint_every_epoch", True))

    print(f"Device: {device}")
    print(f"Feature dims: header={header_dim} bow={bow_dim}")
    print(f"Samples: train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    print(f"Epochs: {start_epoch + 1} → {total_epochs}")

    for epoch in range(start_epoch, total_epochs):
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=total_epochs,
            global_step=global_step,
        )
        val_loss = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        scheduler.step()

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_path,
                build_checkpoint_state(
                    next_epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    global_step=global_step,
                ),
            )

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{total_epochs} — "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"best_val={best_val_loss:.4f} lr={lr:.6f}"
        )

        if save_each_epoch:
            save_checkpoint(
                latest_path,
                build_checkpoint_state(
                    next_epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    global_step=global_step,
                ),
            )

        try:
            from src.thesis_archive import log_epoch

            log_epoch(
                epoch=epoch + 1,
                total_epochs=total_epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=float(lr),
            )
        except ImportError:
            pass

    print(f"Training complete.\n  latest: {latest_path}\n  best:   {best_path}")
    try:
        from src.thesis_archive import after_train

        after_train(
            best_path,
            {
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset),
                "total_epochs": total_epochs,
                "header_dim": header_dim,
                "bow_dim": bow_dim,
            },
        )
    except ImportError:
        pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Pattern B DualBranchNet on cached shards.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument(
        "--resume",
        type=Path,
        nargs="?",
        const=True,
        default=None,
        help="Resume from checkpoint (default: paths.latest_checkpoint)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and train from scratch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)

    resume_path: Path | None = None
    if args.resume is not None:
        if args.resume is True:
            resume_path = cfg.paths.latest_checkpoint
        else:
            resume_path = Path(args.resume)

    run_training(
        cfg,
        epochs_override=args.epochs,
        fresh_start=args.fresh,
        resume_path=resume_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
