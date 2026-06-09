"""Build the 2500-d XGB manifest+DEX vector (matches VigiDroid XgbFeatureBuilder)."""

from __future__ import annotations

import gzip
import json
import struct
import zipfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path

FEATURE_DIM = 2500


def load_feature_index(features_gzip: Path) -> dict[str, int]:
    with gzip.open(features_gzip, "rt", encoding="utf-8") as handle:
        columns: list[str] = json.load(handle)
    return {name: idx for idx, name in enumerate(columns)}


def normalize_permission(raw: str) -> str:
    p = raw.strip().lower()
    prefix = "android.permission."
    if p.startswith(prefix):
        p = p[len(prefix) :]
    return "permissions::" + p.replace(".", "_")


def normalize_xgb_intent(intent: str) -> str:
    value = intent.lower()
    prefix = "android.intent.action."
    if value.startswith(prefix):
        value = value[len(prefix) :]
    return "intents::" + value.replace(".", "_")


def parse_axml_manifest_tokens(manifest_bytes: bytes) -> set[str]:
    """Port of AxmlReader.parse() — scan string pool for permission/intent strings."""
    buf = manifest_bytes
    if len(buf) < 36:
        return set()

    def read_int(offset: int) -> int:
        return struct.unpack_from("<I", buf, offset)[0]

    def get_string(offset: int) -> str:
        length = struct.unpack_from("<H", buf, offset)[0]
        return buf[offset + 2 : offset + 2 + length * 2].decode("utf-16-le", errors="ignore")

    tokens: set[str] = set()
    num_strings = read_int(16)
    strings_start = read_int(28) + 8
    string_pool_offset = 36
    for i in range(num_strings):
        rel = read_int(string_pool_offset + i * 4)
        value = get_string(strings_start + rel)
        if value.startswith("android.permission."):
            tokens.add(normalize_permission(value))
        elif value.startswith("android.intent.action."):
            tokens.add(normalize_xgb_intent(value))
    return tokens


def _read_uleb128(data: bytes, pos: int) -> tuple[int, int]:
    result = 0
    count = 0
    while True:
        cur = data[pos + count]
        result |= (cur & 0x7F) << (count * 7)
        count += 1
        if (cur & 0x80) == 0:
            break
    return result, count


def _read_mutf8(data: bytes, offset: int) -> str:
    pos = offset
    _, skip = _read_uleb128(data, pos)
    pos += skip
    chars: list[str] = []
    while True:
        b = data[pos]
        pos += 1
        if b == 0:
            break
        if (b & 0x80) == 0:
            chars.append(chr(b))
        elif (b & 0xE0) == 0xC0:
            b2 = data[pos] & 0x3F
            pos += 1
            chars.append(chr(((b & 0x1F) << 6) | b2))
        else:
            b2 = data[pos] & 0x3F
            pos += 1
            b3 = data[pos] & 0x3F
            pos += 1
            chars.append(chr(((b & 0x0F) << 12) | (b2 << 6) | b3))
    return "".join(chars)


def parse_dex_api_tokens(dex_bytes: bytes) -> set[str]:
    """Port of MinimalDexParser — emit apicalls:: tokens for android/* methods."""
    if len(dex_bytes) < 112:
        return set()

    string_ids_size = struct.unpack_from("<I", dex_bytes, 56)[0]
    string_ids_off = struct.unpack_from("<I", dex_bytes, 60)[0]
    type_ids_size = struct.unpack_from("<I", dex_bytes, 64)[0]
    type_ids_off = struct.unpack_from("<I", dex_bytes, 68)[0]
    method_ids_size = struct.unpack_from("<I", dex_bytes, 88)[0]
    method_ids_off = struct.unpack_from("<I", dex_bytes, 92)[0]

    strings: list[str] = []
    for i in range(string_ids_size):
        off = struct.unpack_from("<I", dex_bytes, string_ids_off + i * 4)[0]
        strings.append(_read_mutf8(dex_bytes, off))

    types: list[str] = []
    for i in range(type_ids_size):
        idx = struct.unpack_from("<I", dex_bytes, type_ids_off + i * 4)[0]
        types.append(strings[idx])

    tokens: set[str] = set()
    for i in range(method_ids_size):
        base = method_ids_off + i * 8
        class_idx = struct.unpack_from("<H", dex_bytes, base)[0]
        name_idx = struct.unpack_from("<I", dex_bytes, base + 4)[0]
        class_desc = types[class_idx]
        if not class_desc.startswith("Landroid/"):
            continue
        cls = class_desc[1:-1].lower()
        method = strings[name_idx].lower()
        tokens.add(f"apicalls::l{cls}.{method}")
    return tokens


def _vectorize(tokens: Iterable[str], feature_index: dict[str, int]) -> list[float]:
    vec = [0.0] * FEATURE_DIM
    for token in tokens:
        idx = feature_index.get(token)
        if idx is not None and 0 <= idx < FEATURE_DIM:
            vec[idx] = 1.0
    return vec


def _or_pool(master: list[float], candidate: list[float]) -> None:
    for i, value in enumerate(candidate):
        if value > 0.0:
            master[i] = 1.0


def build_xgb_vector(apk_path: Path, feature_index: dict[str, int]) -> list[float]:
    aggregated = [0.0] * FEATURE_DIM
    with zipfile.ZipFile(apk_path, "r") as zf:
        manifest_bytes = zf.read("AndroidManifest.xml")
        manifest_tokens = parse_axml_manifest_tokens(manifest_bytes)
        _or_pool(aggregated, _vectorize(manifest_tokens, feature_index))

        dex_names = sorted(
            name for name in zf.namelist() if name.startswith("classes") and name.endswith(".dex")
        )
        for dex_name in dex_names:
            dex_bytes = zf.read(dex_name)
            dex_tokens = parse_dex_api_tokens(dex_bytes)
            _or_pool(aggregated, _vectorize(dex_tokens, feature_index))
    return aggregated
