"""APK manifest decode: permissions + static receiver actions (pyaxmlparser / lxml)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pyaxmlparser import APK

from src.features.permissions import normalize_permissions

ANDROID_NS = "http://schemas.android.com/apk/res/android"
ANDROID_NS_ATTR = f"{{{ANDROID_NS}}}name"

_PYAXML_LOGGERS = (
    "pyaxmlparser",
    "pyaxmlparser.axmlparser",
    "pyaxmlparser.axmlprinter",
    "pyaxmlparser.core",
)


class ManifestDecodeError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestFeatures:
    permissions: tuple[str, ...]
    receiver_actions: tuple[str, ...]


@contextmanager
def _quiet_pyaxmlparser():
    saved: list[tuple[logging.Logger, int]] = []
    for name in _PYAXML_LOGGERS:
        logger = logging.getLogger(name)
        saved.append((logger, logger.level))
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, level in saved:
            logger.setLevel(level)


def _attr_name(element) -> str | None:
    value = element.get(ANDROID_NS_ATTR) or element.get("name")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _tag_local_name(tag) -> str:
    text = tag if isinstance(tag, str) else str(tag)
    return text.split("}")[-1] if "}" in text else text


def _is_permission_tag(tag) -> bool:
    local = _tag_local_name(tag)
    return local == "uses-permission" or local.startswith("uses-permission-sdk")


def _extract_permissions_from_apk(apk, root) -> list[str]:
    raw: list[str] = []
    try:
        raw.extend(apk.get_permissions() or [])
    except Exception:
        pass
    for element in root.iter():
        if _is_permission_tag(element.tag):
            name = _attr_name(element)
            if name:
                raw.append(name)
    return normalize_permissions(raw)


def _extract_receiver_actions_from_xml(root) -> list[str]:
    seen: set[str] = set()
    actions: list[str] = []
    for element in root.iter():
        if _tag_local_name(element.tag) != "receiver":
            continue
        for action_el in element.iter("action"):
            name = _attr_name(action_el)
            if name and name not in seen:
                seen.add(name)
                actions.append(name)
    return actions


def decode_manifest(apk_path: Path) -> ManifestFeatures:
    try:
        with _quiet_pyaxmlparser():
            apk = APK(str(apk_path))
            root = apk.get_android_manifest_xml()
    except Exception as exc:
        raise ManifestDecodeError(f"Failed to parse manifest: {apk_path}") from exc

    permissions = _extract_permissions_from_apk(apk, root)
    receiver_actions = _extract_receiver_actions_from_xml(root)
    return ManifestFeatures(
        permissions=tuple(permissions),
        receiver_actions=tuple(receiver_actions),
    )
