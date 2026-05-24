"""Dex header feature extraction."""

from src.features.dex_header import (
    FEATURE_DIM,
    DexHeaderError,
    DexHeaderFields,
    extract_header_features,
    extract_headers_from_dex_list,
    parse_dex_header_fields,
    validate_magic,
)
from src.features.multidex import (
    MultidexError,
    aggregate_header_vectors,
    dex_suffix_sort_key,
    multidex_settings,
)

__all__ = [
    "FEATURE_DIM",
    "DexHeaderError",
    "DexHeaderFields",
    "MultidexError",
    "aggregate_header_vectors",
    "dex_suffix_sort_key",
    "extract_header_features",
    "extract_headers_from_dex_list",
    "multidex_settings",
    "parse_dex_header_fields",
    "validate_magic",
]
