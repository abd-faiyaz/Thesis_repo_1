"""Stage 3 — association rule mining on malware-only transactions.

Paper describes explicit X ⇒ malware rules. This implementation mines frequent
itemsets on malware-only transactions via FP-Growth and keeps permissions appearing
in high-confidence/lift rules among PRNR/support-filtered candidates. That is an
acceptable simplification: consequents are implicit malware context, not a separate
class label column.
"""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth


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

    rows: list[dict[str, bool]] = []
    for tx in malware_transactions:
        filtered = tx & candidate_set
        if not filtered:
            continue
        row = {perm: (perm in filtered) for perm in candidates}
        rows.append(row)

    if not rows:
        return set(), []

    df = pd.DataFrame(rows).astype(bool)
    freq = fpgrowth(df, min_support=min_support, use_colnames=True)
    if freq.empty:
        return set(), []

    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        return set(), []

    rules = rules[rules["lift"] >= min_lift]
    selected: set[str] = set()
    rule_records: list[dict] = []

    for _, row in rules.iterrows():
        antecedents = frozenset(row["antecedents"])
        consequents = frozenset(row["consequents"])
        for perm in antecedents | consequents:
            if perm in candidate_set:
                selected.add(perm)
        rule_records.append(
            {
                "antecedents": sorted(antecedents),
                "consequents": sorted(consequents),
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
            }
        )

    rule_records.sort(key=lambda r: r["lift"], reverse=True)
    if len(rule_records) > max_stored_rules:
        rule_records = rule_records[:max_stored_rules]

    return selected, rule_records
