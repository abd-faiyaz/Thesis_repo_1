"""Static receiver system-action filtering (M3)."""

from __future__ import annotations

import json
from pathlib import Path


def load_system_actions(path: Path) -> frozenset[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actions = payload.get("actions") or []
    if not actions:
        raise ValueError(f"system_actions.json is empty: {path}")
    return frozenset(str(a).strip() for a in actions if str(a).strip())


def filter_receiver_system_actions(
    receiver_actions: list[str] | tuple[str, ...],
    system_actions: frozenset[str],
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for action in receiver_actions:
        name = action.strip()
        if not name or name not in system_actions:
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out
