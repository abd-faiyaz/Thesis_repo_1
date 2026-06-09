"""Parse classes.dex DexHeader into a normalized 1D feature vector."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.constants import DEX_HEADER_SIZE, DEX_MAGIC_LEN

# Bytes 8..111 of the Dex file (checksum + signature + header uint32 fields).
HEADER_BODY_SIZE = DEX_HEADER_SIZE - DEX_MAGIC_LEN
FEATURE_DIM = HEADER_BODY_SIZE  # 104

# Little-endian: checksum, 20-byte signature, then 20 uint32 header fields.
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
    """Parsed DexHeader fields (post-magic)."""

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
    """
    Verify the first 8 bytes are a valid Dex magic.
    Accepts dex\\n035\\0 and other standard version tags (037, 038, 039).
    """
    if len(dex_bytes) < DEX_MAGIC_LEN:
        return False
    if dex_bytes[:4] != b"dex\n":
        return False
    version = dex_bytes[4:7]
    if not (version.isdigit() and len(version) == 3):
        return False
    return dex_bytes[7:8] == b"\x00"


def parse_dex_header_fields(dex_bytes: bytes) -> DexHeaderFields:
    """Unpack DexHeader into named fields; raises DexHeaderError on failure."""
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
    """
    Hex-style 1D encoding: each header byte (after magic) as float in [0, 1] via /255.
    Covers checksum, SHA-1 signature, link segment, map_off, and ID section sizes/offsets.
    """
    if not validate_magic(dex_bytes):
        raise DexHeaderError("Invalid DEX magic")
    if len(dex_bytes) < DEX_HEADER_SIZE:
        raise DexHeaderError("DEX buffer too small for header extraction")

    raw = np.frombuffer(
        dex_bytes, dtype=np.uint8, count=HEADER_BODY_SIZE, offset=DEX_MAGIC_LEN
    )
    return raw.astype(np.float64) / 255.0


def extract_header_features(dex_bytes: bytes) -> np.ndarray:
    """Alias for the primary feature extractor used by preprocessing."""
    return extract_raw_byte_features(dex_bytes)


def extract_headers_from_dex_list(dex_bytes_list: list[bytes]) -> list[np.ndarray]:
    """Parse each Dex buffer to a 104-d header vector; fail on first invalid Dex."""
    vectors: list[np.ndarray] = []
    for dex_bytes in dex_bytes_list:
        parse_dex_header_fields(dex_bytes)
        vectors.append(extract_header_features(dex_bytes))
    return vectors


def fields_to_hex_strings(fields: DexHeaderFields) -> dict[str, str]:
    """Debug helper: uint32 fields as hex strings (signature as hex blob)."""
    out: dict[str, str] = {"signature": fields.signature.hex()}
    for name in _FIELD_NAMES:
        if name == "signature":
            continue
        out[name] = f"{getattr(fields, name):08x}"
    return out
