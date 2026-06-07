"""APK manifest decode — declared permissions only (pyaxmlparser)."""

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


def _extract_permissions_from_apk(apk, root, *, include_sdk_23: bool) -> list[str]:
    raw: list[str] = []
    try:
        raw.extend(apk.get_permissions() or [])
    except Exception:
        pass
    for element in root.iter():
        if not include_sdk_23 and _tag_local_name(element.tag).startswith("uses-permission-sdk"):
            continue
        if _is_permission_tag(element.tag):
            name = _attr_name(element)
            if name:
                raw.append(name)
    return normalize_permissions(raw)


def decode_manifest(apk_path: Path, *, include_sdk_23: bool = True) -> ManifestFeatures:
    try:
        with _quiet_pyaxmlparser():
            apk = APK(str(apk_path))
            root = apk.get_android_manifest_xml()
    except Exception as exc:
        raise ManifestDecodeError(f"Failed to parse manifest: {apk_path}") from exc

    permissions = _extract_permissions_from_apk(apk, root, include_sdk_23=include_sdk_23)
    return ManifestFeatures(permissions=tuple(permissions))
