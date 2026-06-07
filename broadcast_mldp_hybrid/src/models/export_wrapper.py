"""ONNX export wrapper — sigmoid logits → malware_prob."""

from __future__ import annotations

import torch
import torch.nn as nn


class MalwareProbExport(nn.Module):
    """Wrap deployment head; ONNX graph includes sigmoid (P7)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.model(features)
        return torch.sigmoid(logits)
