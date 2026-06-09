"""Freeze receiver-action vocabulary A and fusion feature layout metadata."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def build_receiver_vocab(
    train_actions: list[list[str]],
    *,
    min_doc_freq: int = 1,
) -> list[str]:
    counts: Counter[str] = Counter()
    for actions in train_actions:
        counts.update(actions)
    return sorted(token for token, count in counts.items() if count >= min_doc_freq)


def save_receiver_vocab(path: Path, tokens: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"tokens": tokens, "size": len(tokens)}, indent=2) + "\n",
        encoding="utf-8",
    )


def save_feature_layout(
    path: Path,
    *,
    dex_dim: int,
    r_size: int,
    receiver_embed_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "order": ["dex_header", "receiver_actions"],
        "dex_header": dex_dim,
        "receiver": r_size,
        "receiver_embed_dim": receiver_embed_dim,
        "fused_embed": 128 + receiver_embed_dim,
        "header_hidden": 128,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_vocab_tokens(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["tokens"])
