"""Checkpoint I/O for fusion model (best.pt)."""

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


def load_frozen_layout(processed_dir: Path) -> tuple[list[str], dict[str, Any]]:
    a_payload = json.loads((processed_dir / "receiver_action_vocab.json").read_text(encoding="utf-8"))
    layout = json.loads((processed_dir / "feature_layout.json").read_text(encoding="utf-8"))
    return list(a_payload["tokens"]), layout


def build_best_checkpoint(
    cfg: PipelineConfig,
    model: nn.Module,
    *,
    val_metrics: dict[str, float],
    epochs_trained: int,
    receiver_vocab: list[str],
    feature_layout: dict[str, Any],
    receiver_dim: int,
    warm_started: bool,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "model_id": cfg.model_id,
        "domain": cfg.domain,
        "A": receiver_vocab,
        "R": len(receiver_vocab),
        "receiver_dim": int(receiver_dim),
        "dex_dim": int(feature_layout.get("dex_header", 104)),
        "receiver_embed_dim": int(cfg.model.get("receiver_embed_dim", 32)),
        "feature_layout": feature_layout,
        "config_hash": config_hash(cfg),
        "val_metrics": val_metrics,
        "epochs_trained": int(epochs_trained),
        "warm_started": warm_started,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def save_best_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_best_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def restore_model_weights(model: nn.Module, payload: dict[str, Any]) -> None:
    model.load_state_dict(payload["model_state"])
