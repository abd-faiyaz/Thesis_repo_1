"""Evaluate deployed MLP(H) on preprocessed H vectors (dex-header-only ablation)."""

from __future__ import annotations

import numpy as np

from src.config import PipelineConfig
from src.data.store import CascadeFeatureShard
from src.models.mlp_header_ref import DeployedMlpHeaderRef
from src.training.metrics import compute_metrics


def eval_deployed_dex_header(
    shard: CascadeFeatureShard,
    ref: DeployedMlpHeaderRef,
    *,
    threshold: float = 0.5,
    batch_size: int = 512,
) -> dict[str, float]:
    h = shard.h.numpy().astype(np.float32)
    y_true = shard.y.numpy().astype(int).ravel()
    scores_list: list[np.ndarray] = []

    for start in range(0, h.shape[0], batch_size):
        end = min(start + batch_size, h.shape[0])
        batch_scores = ref.score(h[start:end])
        scores_list.append(np.asarray(batch_scores, dtype=np.float64).ravel())

    y_score = np.concatenate(scores_list)
    y_pred = (y_score >= threshold).astype(int)
    return compute_metrics(y_true, y_pred, y_score)


def eval_deployed_dex_header_from_config(
    cfg: PipelineConfig,
    shard: CascadeFeatureShard,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    ref = DeployedMlpHeaderRef.from_config(cfg)
    return eval_deployed_dex_header(shard, ref, threshold=threshold)
