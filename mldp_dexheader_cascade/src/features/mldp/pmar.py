"""PMAR — association-rule collapse (paper #7).

Uses explicit pairwise counting instead of mlxtend Apriori/FP-Growth.
Dense boolean matrices over ~25 candidates × thousands of malware APKs can
OOM-kill the process on real corpora.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def _count_item_and_pair_supports(
    malware_transactions: list[set[str]],
    candidate_set: set[str],
) -> tuple[int, dict[str, int], dict[frozenset[str], int]]:
    n_tx = 0
    item_counts: dict[str, int] = defaultdict(int)
    pair_counts: dict[frozenset[str], int] = defaultdict(int)

    for tx in malware_transactions:
        present = sorted(tx & candidate_set)
        if not present:
            continue
        n_tx += 1
        for item in present:
            item_counts[item] += 1
        if len(present) > 1:
            for a, b in combinations(present, 2):
                pair_counts[frozenset((a, b))] += 1

    return n_tx, dict(item_counts), dict(pair_counts)


def mine_association_rules(
    malware_transactions: list[set[str]],
    candidates: list[str],
    *,
    min_support: float = 0.10,
    min_confidence: float = 0.965,
    min_lift: float = 1.0,
    max_stored_rules: int = 200,
) -> tuple[list[dict], list[tuple[str, str]]]:
    candidate_set = set(candidates)
    if not malware_transactions or not candidates:
        return [], []

    n_tx, item_counts, pair_counts = _count_item_and_pair_supports(
        malware_transactions,
        candidate_set,
    )
    if n_tx == 0:
        return [], []

    min_pair_count = min_support * n_tx
    frequent_items = {
        item for item, count in item_counts.items() if count >= min_pair_count
    }

    rule_records: list[dict] = []
    implications: list[tuple[str, str]] = []

    for pair, co_count in pair_counts.items():
        if co_count < min_pair_count:
            continue
        a, b = tuple(pair)
        if a not in frequent_items or b not in frequent_items:
            continue

        pair_support = co_count / n_tx
        support_a = item_counts[a] / n_tx
        support_b = item_counts[b] / n_tx

        for ant, cons, ant_count, cons_support in (
            (a, b, item_counts[a], support_b),
            (b, a, item_counts[b], support_a),
        ):
            confidence = co_count / ant_count
            lift = confidence / cons_support if cons_support > 0 else 0.0
            if confidence < min_confidence or lift < min_lift:
                continue
            rule_records.append(
                {
                    "antecedents": [ant],
                    "consequents": [cons],
                    "support": pair_support,
                    "confidence": confidence,
                    "lift": lift,
                }
            )
            if ant in candidate_set and cons in candidate_set and ant != cons:
                implications.append((ant, cons))

    rule_records.sort(key=lambda r: r["lift"], reverse=True)
    if len(rule_records) > max_stored_rules:
        rule_records = rule_records[:max_stored_rules]

    # Deduplicate directed implications while preserving first-seen order.
    seen_impl: set[tuple[str, str]] = set()
    unique_impl: list[tuple[str, str]] = []
    for item in implications:
        if item in seen_impl:
            continue
        seen_impl.add(item)
        unique_impl.append(item)

    return rule_records, unique_impl


def collapse_by_implications(
    candidates: list[str],
    implications: list[tuple[str, str]],
    r_scores: dict[str, float],
) -> tuple[list[str], list[str]]:
    candidate_set = set(candidates)
    to_remove: set[str] = set()

    for ant, cons in implications:
        if ant not in candidate_set or cons not in candidate_set:
            continue
        if ant in to_remove:
            continue
        to_remove.add(cons)

    kept = [p for p in candidates if p not in to_remove]
    return kept, sorted(to_remove)
