#!/usr/bin/env python3
"""Recompute tuned_val + cascade bands on val without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import ensure_artifact_dirs, load_config
from src.data.dataloaders import build_dataloaders_from_config, resolve_processed_path
from src.data.store import load_processed_bundle
from src.models.mlp_header import build_mlp_header
from src.training.checkpoint import load_checkpoint, restore_from_checkpoint
from src.training.evaluate import write_val_thresholds
from src.training.setup import build_training_objects


def _load_model_for_val(cfg, checkpoint_path: Path | None = None):
    _, val_loader, feature_dim = build_dataloaders_from_config(cfg)
    ckpt_path = checkpoint_path or cfg.paths.latest_checkpoint
    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    feature_dim = int(checkpoint.get("feature_dim", feature_dim))
    hidden_dim = int(checkpoint.get("hidden_dim", cfg.model.get("hidden_dim", 128)))
    model = build_mlp_header(input_dim=feature_dim, hidden_dim=hidden_dim)
    _criterion, optimizer, scheduler, device = build_training_objects(cfg, model)
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    assert checkpoint is not None
    restore_from_checkpoint(checkpoint, model, optimizer, scheduler)
    return model, val_loader, device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate tuned_val and cascade t_low/t_high on val only."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-tune-threshold", action="store_true")
    parser.add_argument("--no-cascade-bands", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    bundle = load_processed_bundle(resolve_processed_path(cfg))
    model, val_loader, device = _load_model_for_val(cfg, args.checkpoint)
    payload = write_val_thresholds(
        cfg,
        model,
        val_loader,
        device,
        bundle_paths=bundle.paths,
        tune_on_val=not args.no_tune_threshold,
        calibrate_bands=not args.no_cascade_bands,
        out_path=args.out,
    )
    print(f"tuned_val={payload['tuned_val']:.4f}")
    if "cascade" in payload:
        c = payload["cascade"]
        print(
            f"t_low={c['t_low']:.4f} t_high={c['t_high']:.4f} "
            f"val_step1_exit_rate={c['val_step1_exit_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
