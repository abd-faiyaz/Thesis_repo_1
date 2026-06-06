"""Stage 1 — PRNR ranking on train transactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PRNRResult:
    scores: dict[str, float]
    ranked: list[str]
    malware_counts: dict[str, int]
    benign_counts: dict[str, int]


def compute_prnr(
    transactions: list[set[str]],
    labels: list[int],
    *,
    epsilon: float = 1e-6,
    min_rate_delta: float = 0.02,
    top_k: int = 80,
) -> PRNRResult:
    malware_counts: dict[str, int] = {}
    benign_counts: dict[str, int] = {}
    n_m = sum(1 for y in labels if y == 1)
    n_b = sum(1 for y in labels if y == 0)
    if n_m == 0 or n_b == 0:
        raise ValueError("PRNR requires both malware and benign samples in train split")

    for tokens, label in zip(transactions, labels):
        target = malware_counts if label == 1 else benign_counts
        for token in tokens:
            target[token] = target.get(token, 0) + 1

    all_perms = set(malware_counts) | set(benign_counts)
    scores: dict[str, float] = {}
    for perm in all_perms:
        rho_m = malware_counts.get(perm, 0) / n_m
        rho_b = benign_counts.get(perm, 0) / n_b
        if abs(rho_m - rho_b) < min_rate_delta:
            continue
        scores[perm] = rho_m / (rho_b + epsilon)

    ranked = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)[:top_k]
    return PRNRResult(
        scores=scores,
        ranked=ranked,
        malware_counts=malware_counts,
        benign_counts=benign_counts,
    )


def prnr_to_json(result: PRNRResult) -> dict:
    return {
        "scores": result.scores,
        "ranked": result.ranked,
        "malware_counts": result.malware_counts,
        "benign_counts": result.benign_counts,
    }
