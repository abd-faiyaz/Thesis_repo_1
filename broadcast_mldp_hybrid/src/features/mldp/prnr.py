"""PRNR — Permission Ranking with Negative Rate (paper #7, plan M4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PRNRResult:
    r_scores: dict[str, float]
    survivors: list[str]
    dropped: list[str]
    malware_support: dict[str, float]
    benign_support: dict[str, float]
    malware_counts: dict[str, int]
    benign_counts: dict[str, int]


def compute_prnr(
    transactions: list[set[str]],
    labels: list[int],
    *,
    drop_abs_threshold: float = 0.05,
) -> PRNRResult:
    n_m = sum(1 for y in labels if y == 1)
    n_b = sum(1 for y in labels if y == 0)
    if n_m == 0 or n_b == 0:
        raise ValueError("PRNR requires both malware and benign samples in train split")

    malware_counts: dict[str, int] = {}
    benign_counts: dict[str, int] = {}
    for tokens, label in zip(transactions, labels):
        target = malware_counts if label == 1 else benign_counts
        for token in tokens:
            target[token] = target.get(token, 0) + 1

    all_perms = set(malware_counts) | set(benign_counts)
    r_scores: dict[str, float] = {}
    malware_support: dict[str, float] = {}
    benign_support: dict[str, float] = {}
    survivors: list[str] = []
    dropped: list[str] = []

    for perm in sorted(all_perms):
        s_m = malware_counts.get(perm, 0) / n_m
        s_b = benign_counts.get(perm, 0) / n_b
        malware_support[perm] = s_m
        benign_support[perm] = s_b
        denom = s_m + s_b
        r_val = (s_m - s_b) / denom if denom > 0 else 0.0
        r_scores[perm] = r_val
        if abs(r_val) <= drop_abs_threshold:
            dropped.append(perm)
        else:
            survivors.append(perm)

    return PRNRResult(
        r_scores=r_scores,
        survivors=survivors,
        dropped=dropped,
        malware_support=malware_support,
        benign_support=benign_support,
        malware_counts=malware_counts,
        benign_counts=benign_counts,
    )


def prnr_to_json(result: PRNRResult) -> dict:
    return {
        "survivor_count": len(result.survivors),
        "dropped_count": len(result.dropped),
        "survivors": result.survivors,
        "dropped": result.dropped,
        "r_scores": result.r_scores,
        "malware_support": result.malware_support,
        "benign_support": result.benign_support,
        "malware_counts": result.malware_counts,
        "benign_counts": result.benign_counts,
    }
