"""P5 — train Mode A, Mode B Stage 1, ablations, and paper baselines."""

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
from torch.utils.data import DataLoader

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.data.dataloaders import _loader_settings, compute_pos_weight
from src.data.store import load_split_shards
from src.models import (
    build_fused_mlp,
    build_fused_mlp_from_config,
    build_mldp_logistic,
)
from src.models.mldp_logistic import MldpStage1TinyMlp
from src.training.ablation import ABLATION_MODES, build_ablation_loaders
from src.training.checkpoint import (
    build_mode_a_checkpoint,
    build_stage1_checkpoint,
    load_frozen_artifacts,
    save_checkpoint,
)
from src.training.dex_header_eval import eval_deployed_dex_header_from_config
from src.training.loops import train_one_epoch, validation_epoch
from src.training.metrics import format_metrics
from src.training.setup import build_training_objects
from src.training.svm_baseline import run_paper_baselines

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_epoch_log(metrics_dir: Path, record: dict[str, Any]) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_path = metrics_dir / "epochs.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _reset_epoch_log(metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "epochs.jsonl").write_text("", encoding="utf-8")


def train_model(
    cfg: PipelineConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    pos_weight: float,
    total_epochs: int,
    patience: int,
    threshold: float,
    desc: str,
    log_epochs: bool,
    metrics_dir: Path,
) -> dict[str, Any]:
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
            desc=f"Train[{desc}]",
        )
        val_loss, val_metrics = validation_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold,
            epoch=epoch,
            total_epochs=total_epochs,
            desc=f"Val[{desc}]",
        )

        val_f1 = float(val_metrics["f1"])
        print(
            f"[{desc}] Epoch {epoch + 1}/{total_epochs} — "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"{format_metrics(val_metrics)}"
        )

        if log_epochs:
            _append_epoch_log(
                metrics_dir,
                {
                    "epoch": epoch + 1,
                    "model": desc,
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
                print(f"[{desc}] Early stop at epoch {epoch + 1} (patience={patience})")
                break

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "val_f1": best_val_f1,
        "val_metrics": best_val_metrics,
        "model_state": best_state,
    }


def train_ablation_model(
    cfg: PipelineConfig,
    *,
    mode: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    pos_weight: float,
    total_epochs: int,
    patience: int,
    threshold: float,
    log_epochs: bool,
    metrics_dir: Path,
) -> dict[str, Any]:
    if mode == "mldp_perms_only":
        hidden = int(cfg.model.get("mode_b_stage1_mlp_hidden", 32))
        model = build_fused_mlp(input_dim, hidden_dim=hidden)
    else:
        model = build_fused_mlp_from_config(cfg, input_dim)

    result = train_model(
        cfg,
        model,
        train_loader,
        val_loader,
        pos_weight=pos_weight,
        total_epochs=total_epochs,
        patience=patience,
        threshold=threshold,
        desc=mode,
        log_epochs=log_epochs and mode == "mode_a_fusion",
        metrics_dir=metrics_dir,
    )
    result["mode"] = mode
    result["input_dim"] = input_dim
    return result


def train_stage1_candidates(
    cfg: PipelineConfig,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    s_dim: int,
    pos_weight: float,
    total_epochs: int,
    patience: int,
    threshold: float,
    metrics_dir: Path,
) -> dict[str, Any]:
    promote_delta = float(cfg.model.get("stage1_mlp_promote_min_f1_delta", 0.02))
    hidden = int(cfg.model.get("mode_b_stage1_mlp_hidden", 32))

    candidates: dict[str, nn.Module] = {
        "logistic": build_mldp_logistic(s_dim),
        "tiny_mlp": MldpStage1TinyMlp(s_dim, hidden_dim=hidden),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, model in candidates.items():
        print(f"\n=== Mode B Stage 1 candidate: {name} ===")
        results[name] = train_model(
            cfg,
            model,
            train_loader,
            val_loader,
            pos_weight=pos_weight,
            total_epochs=total_epochs,
            patience=patience,
            threshold=threshold,
            desc=f"stage1_{name}",
            log_epochs=False,
            metrics_dir=metrics_dir,
        )
        results[name]["head"] = name

    logistic_f1 = float(results["logistic"]["val_f1"])
    tiny_f1 = float(results["tiny_mlp"]["val_f1"])
    if tiny_f1 > logistic_f1 + promote_delta:
        winner = "tiny_mlp"
        print(
            f"Stage 1 winner: tiny_mlp (val F1 {tiny_f1:.4f} > logistic {logistic_f1:.4f} + {promote_delta})"
        )
    else:
        winner = "logistic"
        print(
            f"Stage 1 winner: logistic (val F1 {logistic_f1:.4f}; tiny_mlp {tiny_f1:.4f})"
        )

    winner_result = results[winner]
    challenger = "tiny_mlp" if winner == "logistic" else "logistic"
    winner_result["winner"] = winner
    winner_result["challenger_metrics"] = results[challenger]["val_metrics"]
    winner_result["candidates"] = {
        name: {
            "val_f1": res["val_f1"],
            "val_metrics": res["val_metrics"],
            "best_epoch": res["best_epoch"],
        }
        for name, res in results.items()
    }
    return winner_result


def run_training(
    cfg: PipelineConfig,
    *,
    epochs_override: int | None = None,
    skip_baselines: bool = False,
    skip_ablations: bool = False,
    skip_stage1: bool = False,
    fresh_log: bool = True,
) -> tuple[Path, Path]:
    ensure_artifact_dirs(cfg)
    shards = load_split_shards(cfg)
    s_tokens, layout = load_frozen_artifacts(cfg.paths.processed)
    s_size = int(layout["S"])

    batch_size, num_workers, pin_memory = _loader_settings(cfg)
    pos_weight = compute_pos_weight(shards["train"].y)
    total_epochs = int(epochs_override or cfg.training.get("epochs", 60))
    patience = int(cfg.training.get("early_stop_patience", 6))
    threshold = float(cfg.evaluation.get("threshold", 0.5))

    metrics_dir = cfg.paths.metrics
    if fresh_log:
        _reset_epoch_log(metrics_dir)

    if not skip_baselines and bool(cfg.baseline.get("paper_svm", True)):
        print("Running paper RBF-SVM + Decision Tree baselines on x_S...")
        run_paper_baselines(cfg, save=True)

    print("\nEvaluating deployed dex-header-only reference on val...")
    dex_val_metrics = eval_deployed_dex_header_from_config(
        cfg, shards["val"], threshold=threshold
    )
    print(f"  dex_header_only val: {format_metrics(dex_val_metrics)}")

    modes = ["mode_a_fusion"] if skip_ablations else list(ABLATION_MODES)
    ablation_results: dict[str, dict[str, Any]] = {
        "dex_header_only": {
            "val_f1": float(dex_val_metrics["f1"]),
            "val_metrics": dex_val_metrics,
            "source": "deployed_mlp_header",
        }
    }
    fusion_result: dict[str, Any] | None = None

    for mode in modes:
        train_loader, val_loader, input_dim = build_ablation_loaders(
            shards["train"],
            shards["val"],
            mode=mode,
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
            log_epochs=(mode == "mode_a_fusion"),
            metrics_dir=metrics_dir,
        )
        ablation_results[mode] = {
            "val_f1": result["val_f1"],
            "val_metrics": result["val_metrics"],
            "best_epoch": result["best_epoch"],
            "input_dim": result["input_dim"],
        }
        ckpt_path = cfg.paths.checkpoints / f"ablation_{mode}.pt"
        save_checkpoint(
            ckpt_path,
            {
                "mode": mode,
                "model_state": result["model_state"],
                "input_dim": result["input_dim"],
                "val_f1": result["val_f1"],
                "val_metrics": result["val_metrics"],
                "best_epoch": result["best_epoch"],
            },
        )
        if mode == "mode_a_fusion":
            fusion_result = result

    if fusion_result is None:
        raise RuntimeError("mode_a_fusion training did not run")

    perm_f1 = float(ablation_results.get("mldp_perms_only", {}).get("val_f1", 0.0))
    dex_f1 = float(ablation_results["dex_header_only"]["val_f1"])
    fusion_f1 = float(ablation_results["mode_a_fusion"]["val_f1"])
    fusion_wins = fusion_f1 >= max(perm_f1, dex_f1)
    if not skip_ablations:
        print(
            f"\nAblation val F1: fusion={fusion_f1:.4f}  "
            f"mldp={perm_f1:.4f}  dex_header={dex_f1:.4f}  "
            f"fusion≥solo={fusion_wins}"
        )

    mode_a_model = build_fused_mlp_from_config(cfg, fusion_result["input_dim"])
    mode_a_model.load_state_dict(fusion_result["model_state"])
    mode_a_path = cfg.paths.checkpoints / "mode_a_best.pt"
    save_checkpoint(
        mode_a_path,
        build_mode_a_checkpoint(
            cfg,
            mode_a_model,
            val_metrics=fusion_result["val_metrics"],
            ablations=ablation_results,
            best_epoch=int(fusion_result["best_epoch"]),
            s_tokens=s_tokens,
            feature_layout=layout,
            input_dim=fusion_result["input_dim"],
        ),
    )

    stage1_path = cfg.paths.checkpoints / "stage1_best.pt"
    stage1_summary: dict[str, Any] | None = None
    if not skip_stage1:
        perm_train_loader, perm_val_loader, _ = build_ablation_loaders(
            shards["train"],
            shards["val"],
            mode="mldp_perms_only",
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        stage1_result = train_stage1_candidates(
            cfg,
            train_loader=perm_train_loader,
            val_loader=perm_val_loader,
            s_dim=s_size,
            pos_weight=pos_weight,
            total_epochs=total_epochs,
            patience=patience,
            threshold=threshold,
            metrics_dir=metrics_dir,
        )
        winner_head = str(stage1_result["winner"])
        if winner_head == "logistic":
            stage1_model = build_mldp_logistic(s_size)
        else:
            stage1_model = MldpStage1TinyMlp(
                s_size,
                hidden_dim=int(cfg.model.get("mode_b_stage1_mlp_hidden", 32)),
            )
        stage1_model.load_state_dict(stage1_result["model_state"])
        save_checkpoint(
            stage1_path,
            build_stage1_checkpoint(
                cfg,
                stage1_model,
                head=winner_head,
                s_dim=s_size,
                val_metrics=stage1_result["val_metrics"],
                best_epoch=int(stage1_result["best_epoch"]),
                s_tokens=s_tokens,
                challenger_metrics=stage1_result.get("challenger_metrics"),
            ),
        )
        stage1_summary = {
            "winner": winner_head,
            "val_f1": stage1_result["val_f1"],
            "val_metrics": stage1_result["val_metrics"],
            "candidates": stage1_result.get("candidates", {}),
        }

    run_info = {
        "model_id": cfg.model_id,
        "pos_weight": pos_weight,
        "epochs_configured": total_epochs,
        "mode_a": {
            "best_epoch": fusion_result["best_epoch"],
            "val_metrics": fusion_result["val_metrics"],
            "checkpoint": str(mode_a_path),
        },
        "ablations": ablation_results,
        "fusion_beats_solo": fusion_wins,
        "stage1": stage1_summary,
        "trained_at": _utc_now(),
    }
    (metrics_dir / "training_run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nMode A checkpoint → {mode_a_path}")
    if not skip_stage1:
        print(f"Stage 1 checkpoint → {stage1_path}")
    print(f"Training log → {metrics_dir / 'epochs.jsonl'}")
    return mode_a_path, stage1_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train mldp_dexheader_cascade models (P5).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--append-log", action="store_true")
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
        skip_stage1=args.skip_stage1,
        fresh_log=not args.append_log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
