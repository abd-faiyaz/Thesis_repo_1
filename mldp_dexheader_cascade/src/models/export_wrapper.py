"""ONNX export wrappers — logits → probability outputs in graph."""

from __future__ import annotations

import torch
import torch.nn as nn


class MalwareProbExport(nn.Module):
    """Wrap a logits head; ONNX graph includes sigmoid (P7 Mode A)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.model(features)
        return torch.sigmoid(logits)


class Stage1ProbExport(nn.Module):
    """Wrap Stage-1 head; ONNX graph includes sigmoid (P7 Mode B)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.model(features)
        return torch.sigmoid(logits)
