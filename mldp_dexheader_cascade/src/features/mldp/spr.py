"""SPR — Support-based Permission Ranking (stage 2)."""

from __future__ import annotations


def rank_by_support(
    candidates: list[str],
    transactions: list[set[str]],
    *,
    keep_top: int,
) -> tuple[list[str], dict[str, float]]:
    n = len(transactions)
    if n == 0:
        return [], {}

    support_stats: dict[str, float] = {}
    for perm in candidates:
        count = sum(1 for tx in transactions if perm in tx)
        support_stats[perm] = count / n

    ranked = sorted(candidates, key=lambda p: (-support_stats[p], p))
    kept = ranked[:keep_top]
    return kept, support_stats
