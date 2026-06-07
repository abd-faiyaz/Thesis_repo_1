"""Parse classes.dex DexHeader into a normalized 1D feature vector."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from src.constants import DEX_HEADER_SIZE, DEX_MAGIC_LEN

HEADER_BODY_SIZE = DEX_HEADER_SIZE - DEX_MAGIC_LEN
FEATURE_DIM = HEADER_BODY_SIZE

_HEADER_STRUCT = struct.Struct("<I20s20I")
_FIELD_NAMES: tuple[str, ...] = (
    "checksum",
    "signature",
    "file_size",
    "header_size",
    "endian_tag",
    "link_size",
    "link_off",
    "map_off",
    "string_ids_size",
    "string_ids_off",
    "type_ids_size",
    "type_ids_off",
    "proto_ids_size",
    "proto_ids_off",
    "field_ids_size",
    "field_ids_off",
    "method_ids_size",
    "method_ids_off",
    "class_defs_size",
    "class_defs_off",
    "data_size",
    "data_off",
)


class DexHeaderError(ValueError):
    """Raised when Dex bytes fail validation or parsing."""


@dataclass(frozen=True)
class DexHeaderFields:
    checksum: int
    signature: bytes
    file_size: int
    header_size: int
    endian_tag: int
    link_size: int
    link_off: int
    map_off: int
    string_ids_size: int
    string_ids_off: int
    type_ids_size: int
    type_ids_off: int
    proto_ids_size: int
    proto_ids_off: int
    field_ids_size: int
    field_ids_off: int
    method_ids_size: int
    method_ids_off: int
    class_defs_size: int
    class_defs_off: int
    data_size: int
    data_off: int


def validate_magic(dex_bytes: bytes) -> bool:
    if len(dex_bytes) < DEX_MAGIC_LEN:
        return False
    if dex_bytes[:4] != b"dex\n":
        return False
    version = dex_bytes[4:7]
    if not (version.isdigit() and len(version) == 3):
        return False
    return dex_bytes[7:8] == b"\x00"


def parse_dex_header_fields(dex_bytes: bytes) -> DexHeaderFields:
    if not validate_magic(dex_bytes):
        raise DexHeaderError("Invalid DEX magic (expected dex\\nNNN\\0)")
    if len(dex_bytes) < DEX_HEADER_SIZE:
        raise DexHeaderError(
            f"DEX buffer too small: {len(dex_bytes)} bytes, need >= {DEX_HEADER_SIZE}"
        )

    body = dex_bytes[DEX_MAGIC_LEN:DEX_HEADER_SIZE]
    if len(body) != _HEADER_STRUCT.size:
        raise DexHeaderError("DexHeader body size mismatch")

    values = _HEADER_STRUCT.unpack(body)
    named = dict(zip(_FIELD_NAMES, values, strict=True))
    return DexHeaderFields(**named)  # type: ignore[arg-type]


def extract_raw_byte_features(dex_bytes: bytes) -> np.ndarray:
    """Bytes 8–111 as float64 in [0, 1] via /255 (no corpus min–max)."""
    if not validate_magic(dex_bytes):
        raise DexHeaderError("Invalid DEX magic")
    if len(dex_bytes) < DEX_HEADER_SIZE:
        raise DexHeaderError("DEX buffer too small for header extraction")

    raw = np.frombuffer(
        dex_bytes, dtype=np.uint8, count=HEADER_BODY_SIZE, offset=DEX_MAGIC_LEN
    )
    return raw.astype(np.float64) / 255.0


def extract_header_features(dex_bytes: bytes) -> np.ndarray:
    return extract_raw_byte_features(dex_bytes)


def extract_headers_from_dex_list(dex_bytes_list: list[bytes]) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for dex_bytes in dex_bytes_list:
        parse_dex_header_fields(dex_bytes)
        vectors.append(extract_header_features(dex_bytes))
    return vectors
