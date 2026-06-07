"""APK permission extraction with VigiDroid-compatible token names."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from pyaxmlparser import APK

_ANDROGUARD_LOGGERS = (
    "androguard",
    "androguard.core",
    "androguard.core.axml",
    "androguard.core.bytecodes",
    "androguard.core.bytecodes.apk",
    "androguard.core.bytecodes.axml",
)


def _suppress_manifest_parser_noise() -> None:
    """pyaxmlparser uses androguard; packed/obfuscated manifests spam warnings."""
    for name in _ANDROGUARD_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


class PermissionExtractError(ValueError):
    pass


def normalize_permission(raw: str) -> str:
    p = raw.strip().lower()
    prefix = "android.permission."
    if p.startswith(prefix):
        p = p[len(prefix) :]
    return "permissions::" + p.replace(".", "_")


def extract_permission_tokens(apk_path: Path) -> list[str]:
    _suppress_manifest_parser_noise()
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


def load_selected_permissions(path: Path) -> tuple[list[str], dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    permissions: list[str] = list(data["permissions"])
    return permissions, {name: i for i, name in enumerate(permissions)}


def save_selected_permissions(
    path: Path,
    permissions: Sequence[str],
    *,
    metadata: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "permissions": list(permissions),
        "S": len(permissions),
        "mldp_version": 1,
        "token_normalization": "vigidroid",
    }
    if metadata:
        payload.update(metadata)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
