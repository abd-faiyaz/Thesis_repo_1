"""Pattern A: concat(H, I) → ASCNN → classifier → malware logit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.models.ascnn_combined import ASCNNCombined, build_ascnn_combined_from_config
from src.models.classifier_head import ClassifierHead, build_classifier_head_from_config

if TYPE_CHECKING:
    from src.config import PipelineConfig


class CombinedNet(nn.Module):
    """
    Single-tower ASCNN(C)-style classifier.

    forward(header, bow) -> logit (B, 1)
    predict_proba(header, bow) -> probability (B, 1)
    """

    def __init__(
        self,
        ascnn: ASCNNCombined,
        classifier: ClassifierHead,
        *,
        header_dim: int = 104,
        bow_dim: int = 4381,
    ) -> None:
        super().__init__()
        self.ascnn = ascnn
        self.classifier = classifier
        self.header_dim = header_dim
        self.bow_dim = bow_dim
        self.combined_dim = header_dim + bow_dim

    def _concat_features(self, header: torch.Tensor, bow: torch.Tensor) -> torch.Tensor:
        if header.dim() == 1:
            header = header.unsqueeze(0)
        if bow.dim() == 1:
            bow = bow.unsqueeze(0)

        if header.shape[-1] != self.header_dim:
            raise ValueError(
                f"Expected header last dim {self.header_dim}, got {header.shape[-1]}"
            )
        if bow.shape[-1] != self.bow_dim:
            raise ValueError(f"Expected bow last dim {self.bow_dim}, got {bow.shape[-1]}")
        if header.shape[0] != bow.shape[0]:
            raise ValueError(
                f"Batch size mismatch: header {header.shape[0]} vs bow {bow.shape[0]}"
            )

        return torch.cat([header, bow], dim=-1)

    def forward(self, header: torch.Tensor, bow: torch.Tensor) -> torch.Tensor:
        combined = self._concat_features(header, bow)
        embedding = self.ascnn(combined)
        return self.classifier(embedding)

    def predict_proba(self, header: torch.Tensor, bow: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(header, bow))


def build_combined_net(
    ascnn: ASCNNCombined | None = None,
    classifier: ClassifierHead | None = None,
    *,
    cfg: PipelineConfig | None = None,
) -> CombinedNet:
    if cfg is None:
        if ascnn is None or classifier is None:
            raise ValueError("Provide cfg or both ascnn and classifier")
        return CombinedNet(
            ascnn,
            classifier,
            header_dim=104,
            bow_dim=4381,
        )

    model_cfg = cfg.model
    return CombinedNet(
        ascnn or build_ascnn_combined_from_config(cfg),
        classifier or build_classifier_head_from_config(cfg),
        header_dim=int(model_cfg.get("header_dim", 104)),
        bow_dim=int(model_cfg.get("bow_dim", 4381)),
    )


def build_combined_net_from_config(cfg: PipelineConfig) -> CombinedNet:
    return build_combined_net(cfg=cfg)
