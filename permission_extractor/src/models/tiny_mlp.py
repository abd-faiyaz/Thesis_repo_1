"""Tiny MLP for pruned permission vectors."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TinyMlpModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_tiny_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dim: int = 32,
    learning_rate: float = 0.01,
    epochs: int = 100,
    batch_size: int = 256,
    seed: int = 42,
) -> TinyMlpModule:
    torch.manual_seed(seed)
    model = TinyMlpModule(X.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


class LinearSigmoidModule(nn.Module):
    """Export wrapper for calibrated linear SVM decision boundary."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

    @classmethod
    def from_linear_svc(cls, svc_model, input_dim: int) -> "LinearSigmoidModule":
        base = svc_model.calibrated_classifiers_[0].estimator
        module = cls(input_dim)
        coef = base.coef_.reshape(-1).astype(np.float32)
        bias = float(base.intercept_.reshape(-1)[0])
        with torch.no_grad():
            module.linear.weight.copy_(torch.from_numpy(coef).view(1, -1))
            module.linear.bias.copy_(torch.tensor([bias], dtype=torch.float32))
        return module
