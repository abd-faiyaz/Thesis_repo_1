"""Training loop, checkpointing, and evaluation (Phases 5–6)."""

from src.training.evaluate import compute_metrics, format_metrics, run_evaluation
from src.training.train import run_training

__all__ = [
    "run_training",
    "run_evaluation",
    "compute_metrics",
    "format_metrics",
]
