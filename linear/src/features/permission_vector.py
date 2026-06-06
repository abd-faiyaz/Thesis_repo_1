"""APK permission extraction with VigiDroid-compatible token names."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from pyaxmlparser import APK


class PermissionExtractError(ValueError):
    """Manifest could not be parsed or encoded."""


def normalize_permission(raw: str) -> str:
    """
    Match VigiDroid ScanService.normalizePermission:
    android.permission.SEND_SMS -> permissions::send_sms
    """
    p = raw.strip().lower()
    prefix = "android.permission."
    if p.startswith(prefix):
        p = p[len(prefix) :]
    return "permissions::" + p.replace(".", "_")


def extract_permission_tokens(apk_path: Path) -> list[str]:
    """Return unique normalized permission tokens from the manifest."""
    try:
        apk = APK(str(apk_path))
    except Exception as exc:
        raise PermissionExtractError(f"Failed to parse APK: {apk_path}") from exc

    tokens: list[str] = []
    seen: set[str] = set()
    for perm in apk.get_permissions():
        if not perm:
            continue
        token = normalize_permission(perm)
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def build_binary_vector(
    tokens: Iterable[str],
    token_to_index: Mapping[str, int],
    *,
    vector_size: int,
) -> np.ndarray:
    vec = np.zeros(vector_size, dtype=np.float32)
    for token in tokens:
        idx = token_to_index.get(token)
        if idx is not None and 0 <= idx < vector_size:
            vec[idx] = 1.0
    return vec


def load_vocab(path: Path) -> tuple[list[str], dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    permissions: list[str] = list(data["permissions"])
    token_to_index = {name: i for i, name in enumerate(permissions)}
    return permissions, token_to_index


def save_vocab(path: Path, permissions: Sequence[str], *, version: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "permissions": list(permissions),
        "M": len(permissions),
        "version": version,
        "token_normalization": "vigidroid",
        "description": "permissions::name tokens; unknown permissions ignored at inference",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
