"""Freeze receiver-action vocabulary A and feature layout metadata."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def build_receiver_vocab(train_actions: list[list[str]]) -> list[str]:
    counts: Counter[str] = Counter()
    for actions in train_actions:
        counts.update(actions)
    return sorted(counts.keys())


def save_receiver_vocab(path: Path, tokens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"tokens": tokens, "size": len(tokens)}, indent=2) + "\n",
        encoding="utf-8",
    )


def save_feature_layout(
    path: Path,
    *,
    s_size: int,
    r_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "order": ["mldp_perms", "receiver_actions"],
        "S": s_size,
        "R": r_size,
        "total": s_size + r_size,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_vocab_tokens(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["tokens"])
