"""Permission extraction and VigiDroid-compatible normalization."""

from __future__ import annotations

from src.constants import PERMISSION_PREFIX, PERMISSION_TOKEN_PREFIX


def normalize_permission(raw: str) -> str | None:
    p = raw.strip().lower()
    if not p:
        return None
    if p.startswith(PERMISSION_PREFIX):
        p = p[len(PERMISSION_PREFIX) :]
    if not p:
        return None
    return PERMISSION_TOKEN_PREFIX + p.replace(".", "_")


def normalize_permissions(raw_names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in raw_names:
        if not name:
            continue
        token = normalize_permission(name)
        if token is None or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out
