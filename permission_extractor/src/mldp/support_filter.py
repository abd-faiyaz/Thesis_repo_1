"""Stage 2 — support-based filtering."""

from __future__ import annotations


def filter_by_support(
    candidates: list[str],
    transactions: list[set[str]],
    *,
    min_support: float = 0.01,
    max_support: float = 0.95,
) -> tuple[list[str], dict[str, float]]:
    n = len(transactions)
    if n == 0:
        return [], {}

    support_stats: dict[str, float] = {}
    kept: list[str] = []
    for perm in candidates:
        count = sum(1 for tx in transactions if perm in tx)
        supp = count / n
        support_stats[perm] = supp
        if supp >= min_support and supp <= max_support:
            kept.append(perm)
    return kept, support_stats
