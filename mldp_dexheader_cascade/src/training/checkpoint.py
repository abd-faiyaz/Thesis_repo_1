"""Checkpoint I/O for Mode A and Mode B Stage 1."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.config import PipelineConfig


def config_hash(cfg: PipelineConfig) -> str:
    blob = json.dumps(cfg.raw, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_frozen_artifacts(processed_dir: Path) -> tuple[list[str], dict[str, Any]]:
    s_payload = json.loads((processed_dir / "mldp_permission_vocab.json").read_text(encoding="utf-8"))
    layout = json.loads((processed_dir / "feature_layout.json").read_text(encoding="utf-8"))
    return list(s_payload["tokens"]), layout


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def restore_model_weights(model: nn.Module, payload: dict[str, Any]) -> None:
    model.load_state_dict(payload["model_state"])


def build_mode_a_checkpoint(
    cfg: PipelineConfig,
    model: nn.Module,
    *,
    val_metrics: dict[str, float],
    ablations: dict[str, dict[str, Any]],
    best_epoch: int,
    s_tokens: list[str],
    feature_layout: dict[str, Any],
    input_dim: int,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "model_id": cfg.model_id,
        "domain": cfg.domain,
        "mode": "A",
        "S": s_tokens,
        "d": int(input_dim),
        "feature_layout": feature_layout,
        "config_hash": config_hash(cfg),
        "val_metrics": val_metrics,
        "ablations": ablations,
        "best_epoch": int(best_epoch),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def build_stage1_checkpoint(
    cfg: PipelineConfig,
    model: nn.Module,
    *,
    head: str,
    s_dim: int,
    val_metrics: dict[str, float],
    best_epoch: int,
    s_tokens: list[str],
    challenger_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "model_id": cfg.model_id,
        "domain": cfg.domain,
        "mode": "B",
        "stage": 1,
        "head": head,
        "S_dim": int(s_dim),
        "S": s_tokens,
        "config_hash": config_hash(cfg),
        "val_metrics": val_metrics,
        "challenger_val_metrics": challenger_metrics,
        "best_epoch": int(best_epoch),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
