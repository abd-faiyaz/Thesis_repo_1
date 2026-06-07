"""Freeze MLDP vocab and feature layout metadata."""

from __future__ import annotations

import json
from pathlib import Path


def save_feature_layout(
    path: Path,
    *,
    s_size: int,
    h_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "order": ["mldp_perms", "dex_header"],
        "S": s_size,
        "H": h_size,
        "d": s_size + h_size,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_vocab_tokens(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["tokens"])
