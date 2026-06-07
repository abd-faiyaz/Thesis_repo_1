"""Stage 3 — association rule mining on malware-only transactions.

Paper describes explicit X ⇒ malware rules. This implementation mines frequent
pairwise co-occurrences on malware-only transactions and keeps permissions
appearing in high-confidence/lift rules among PRNR/support-filtered candidates.
That is an acceptable simplification: consequents are implicit malware context,
not a separate class label column.

Uses explicit counting instead of mlxtend FP-Growth. With |candidates| ≤ ~80 the
full itemset lattice is tiny; FP-Growth on dense boolean matrices can explode
memory/time and get OOM-killed on real corpora.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations


def _count_item_and_pair_supports(
    malware_transactions: list[set[str]],
    candidate_set: set[str],
) -> tuple[int, dict[str, int], dict[frozenset[str], int]]:
    """Return (n_nonempty_tx, item_counts, pair_counts) over filtered transactions."""
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


def mine_rule_permissions(
    malware_transactions: list[set[str]],
    candidates: list[str],
    *,
    min_support: float = 0.05,
    min_confidence: float = 0.70,
    min_lift: float = 1.2,
    max_stored_rules: int = 200,
) -> tuple[set[str], list[dict]]:
    candidate_set = set(candidates)
    if not malware_transactions or not candidates:
        return set(), []

    n_tx, item_counts, pair_counts = _count_item_and_pair_supports(
        malware_transactions,
        candidate_set,
    )
    if n_tx == 0:
        return set(), []

    min_pair_count = min_support * n_tx
    frequent_items = {
        item for item, count in item_counts.items() if count >= min_pair_count
    }

    selected: set[str] = set()
    rule_records: list[dict] = []

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
            selected.add(ant)
            selected.add(cons)
            rule_records.append(
                {
                    "antecedents": [ant],
                    "consequents": [cons],
                    "support": pair_support,
                    "confidence": confidence,
                    "lift": lift,
                }
            )

    rule_records.sort(key=lambda r: r["lift"], reverse=True)
    if len(rule_records) > max_stored_rules:
        rule_records = rule_records[:max_stored_rules]

    return selected, rule_records
