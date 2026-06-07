"""PMAR — Apriori association-rule collapse (paper #7, plan M2)."""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def mine_association_rules(
    malware_transactions: list[set[str]],
    candidates: list[str],
    *,
    min_support: float = 0.10,
    min_confidence: float = 0.965,
) -> tuple[list[dict], list[tuple[str, str]]]:
    candidate_set = set(candidates)
    if not malware_transactions or not candidates:
        return [], []

    rows: list[dict[str, bool]] = []
    for tx in malware_transactions:
        filtered = tx & candidate_set
        if not filtered:
            continue
        rows.append({perm: (perm in filtered) for perm in candidates})

    if not rows:
        return [], []

    df = pd.DataFrame(rows).astype(bool)
    freq = apriori(df, min_support=min_support, use_colnames=True)
    if freq.empty:
        return [], []

    rules_df = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    if rules_df.empty:
        return [], []

    rule_records: list[dict] = []
    implications: list[tuple[str, str]] = []
    for _, row in rules_df.iterrows():
        antecedents = frozenset(str(x) for x in row["antecedents"])
        consequents = frozenset(str(x) for x in row["consequents"])
        rule_records.append(
            {
                "antecedents": sorted(antecedents),
                "consequents": sorted(consequents),
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
            }
        )
        for a in antecedents:
            for c in consequents:
                if a in candidate_set and c in candidate_set and a != c:
                    implications.append((a, c))
    return rule_records, implications


def collapse_by_implications(
    candidates: list[str],
    implications: list[tuple[str, str]],
    r_scores: dict[str, float],
) -> tuple[list[str], list[str]]:
    """Drop rule consequents when the antecedent is retained (directed collapse)."""
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
