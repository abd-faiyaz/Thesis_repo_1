"""P5 — train deployment model with ablations and early stopping on val F1."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.data.dataloaders import _loader_settings, compute_pos_weight
from src.data.store import load_split_shards
from src.models.factory import build_deployment_model_from_config
from src.training.ablation import ABLATION_MODES, build_ablation_loaders
from src.training.checkpoint import (
    build_best_checkpoint,
    load_frozen_vocabs,
    save_best_checkpoint,
)
from src.training.loops import train_one_epoch, validation_epoch
from src.training.metrics import format_metrics
from src.training.setup import build_training_objects
from src.training.svm_baseline import run_paper_baselines

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

from src.training.ablation import ABLATION_MODES


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_epoch_log(metrics_dir: Path, record: dict[str, Any]) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_path = metrics_dir / "epochs.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    try:
        from src.thesis_archive import get_archive

        arc = get_archive()
        if arc.enabled:
            arc.mirror_file(log_path, "metrics/epochs.jsonl")
    except ImportError:
        pass


def _reset_epoch_log(metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "epochs.jsonl").write_text("", encoding="utf-8")


def train_ablation_model(
    cfg: PipelineConfig,
    *,
    mode: str,
    train_loader,
    val_loader,
    input_dim: int,
    pos_weight: float,
    total_epochs: int,
    patience: int,
    threshold: float,
    log_epochs: bool,
    metrics_dir: Path,
) -> dict[str, Any]:
    model = build_deployment_model_from_config(cfg, input_dim)
    criterion, optimizer, device = build_training_objects(cfg, model, pos_weight=pos_weight)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_val_metrics: dict[str, float] = {}
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(total_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=total_epochs,
            desc=f"Train[{mode}]",
        )
        val_loss, val_metrics = validation_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold,
            epoch=epoch,
            total_epochs=total_epochs,
            desc=f"Val[{mode}]",
        )

        val_f1 = float(val_metrics["f1"])
        print(
            f"[{mode}] Epoch {epoch + 1}/{total_epochs} — "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"{format_metrics(val_metrics)}"
        )

        if log_epochs and mode == "full_fusion":
            _append_epoch_log(
                metrics_dir,
                {
                    "epoch": epoch + 1,
                    "mode": mode,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **val_metrics,
                },
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"[{mode}] Early stop at epoch {epoch + 1} (patience={patience})")
                break

    model.load_state_dict(best_state)
    return {
        "mode": mode,
        "input_dim": input_dim,
        "best_epoch": best_epoch,
        "val_f1": best_val_f1,
        "val_metrics": best_val_metrics,
        "model_state": best_state,
    }


def run_training(
    cfg: PipelineConfig,
    *,
    epochs_override: int | None = None,
    skip_baselines: bool = False,
    skip_ablations: bool = False,
    fresh_log: bool = True,
) -> Path:
    ensure_artifact_dirs(cfg)
    shards = load_split_shards(cfg)
    s_tokens, a_tokens, layout = load_frozen_vocabs(cfg.paths.processed)
    s_size = int(layout["S"])

    batch_size, num_workers, pin_memory = _loader_settings(cfg)
    pos_weight = compute_pos_weight(shards["train"].y)
    total_epochs = int(epochs_override or cfg.training.get("epochs", 60))
    patience = int(cfg.training.get("early_stop_patience", 6))
    threshold = float(cfg.evaluation.get("threshold", 0.5))

    metrics_dir = cfg.paths.metrics
    if fresh_log:
        _reset_epoch_log(metrics_dir)

    if not skip_baselines and bool(cfg.classifier.get("paper_baseline_svm", True)):
        print("Running paper RBF-SVM + Decision Tree baselines...")
        run_paper_baselines(cfg, save=True)

    modes = ["full_fusion"] if skip_ablations else list(ABLATION_MODES)
    ablation_results: dict[str, dict[str, Any]] = {}
    full_result: dict[str, Any] | None = None

    for mode in modes:
        train_loader, val_loader, input_dim = build_ablation_loaders(
            shards["train"],
            shards["val"],
            mode=mode,
            s_size=s_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        result = train_ablation_model(
            cfg,
            mode=mode,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            pos_weight=pos_weight,
            total_epochs=total_epochs,
            patience=patience,
            threshold=threshold,
            log_epochs=(mode == "full_fusion"),
            metrics_dir=metrics_dir,
        )
        ablation_results[mode] = {
            "val_f1": result["val_f1"],
            "val_metrics": result["val_metrics"],
            "best_epoch": result["best_epoch"],
            "input_dim": result["input_dim"],
        }
        ablation_ckpt = cfg.paths.checkpoints / f"ablation_{mode}.pt"
        torch.save(
            {
                "mode": mode,
                "model_state": result["model_state"],
                "input_dim": result["input_dim"],
                "val_f1": result["val_f1"],
                "val_metrics": result["val_metrics"],
                "best_epoch": result["best_epoch"],
            },
            ablation_ckpt,
        )
        if mode == "full_fusion":
            full_result = result

    if full_result is None:
        raise RuntimeError("full_fusion training did not run")

    perm_f1 = ablation_results.get("mldp_perms_only", {}).get("val_f1", 0.0)
    recv_f1 = ablation_results.get("receiver_actions_only", {}).get("val_f1", 0.0)
    full_f1 = float(ablation_results["full_fusion"]["val_f1"])
    fusion_wins = full_f1 >= max(perm_f1, recv_f1)
    if not skip_ablations:
        print(
            f"Ablation val F1: full={full_f1:.4f}  "
            f"perms={perm_f1:.4f}  receivers={recv_f1:.4f}  "
            f"combined≥solo={fusion_wins}"
        )

    deploy_model = build_deployment_model_from_config(cfg, full_result["input_dim"])
    deploy_model.load_state_dict(full_result["model_state"])

    best_path = cfg.paths.checkpoints / "best.pt"
    payload = build_best_checkpoint(
        cfg,
        deploy_model,
        val_metrics=full_result["val_metrics"],
        ablations=ablation_results,
        epochs_trained=int(full_result["best_epoch"]),
        s_tokens=s_tokens,
        a_tokens=a_tokens,
        feature_layout=layout,
        input_dim=full_result["input_dim"],
    )
    save_best_checkpoint(best_path, payload)

    run_info = {
        "model_id": cfg.model_id,
        "deployment": payload["deployment"],
        "d": payload["d"],
        "S": len(s_tokens),
        "R": len(a_tokens),
        "pos_weight": pos_weight,
        "epochs_configured": total_epochs,
        "best_epoch": full_result["best_epoch"],
        "val_metrics": full_result["val_metrics"],
        "ablations": ablation_results,
        "fusion_beats_solo": fusion_wins,
        "trained_at": _utc_now(),
    }
    (metrics_dir / "training_run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        from src.thesis_archive import after_train

        after_train(best_path, run_info)
    except ImportError:
        pass

    print(f"Best checkpoint → {best_path}")
    print(f"Training log → {metrics_dir / 'epochs.jsonl'}")
    return best_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train broadcast+MLDP hybrid classifier (P5).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true", help="Train full fusion only")
    parser.add_argument(
        "--append-log",
        action="store_true",
        help="Append to epochs.jsonl instead of resetting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_training(
        cfg,
        epochs_override=args.epochs,
        skip_baselines=args.skip_baselines,
        skip_ablations=args.skip_ablations,
        fresh_log=not args.append_log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
