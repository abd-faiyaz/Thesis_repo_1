"""ONNX export wrapper — two inputs, sigmoid output malware_prob."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionMalwareProbExport(nn.Module):
  def __init__(self, model: nn.Module) -> None:
    super().__init__()
    self.model = model

  def forward(self, dex_header: torch.Tensor, receiver: torch.Tensor) -> torch.Tensor:
    logits = self.model(dex_header, receiver)
    return torch.sigmoid(logits)
