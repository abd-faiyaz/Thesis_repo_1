"""Feature extraction: Dex header and manifest BoW."""

from src.features.apk_extract import extract_apk_raw_header, list_dex_entries, read_all_dex_from_apk
from src.features.dex_header import (
    DEX_HEADER_FEATURE_DIM,
    DexHeaderError,
    extract_header_features,
    extract_headers_from_dex_list,
    parse_dex_header_fields,
    validate_magic,
)
from src.features.manifest_bow import (
    ManifestBoWError,
    build_multihot_vector,
    extract_manifest_tokens,
    load_vocab,
    save_vocab,
)
from src.features.multidex import aggregate_header_vectors, multidex_settings

__all__ = [
    "DEX_HEADER_FEATURE_DIM",
    "DexHeaderError",
    "ManifestBoWError",
    "aggregate_header_vectors",
    "build_multihot_vector",
    "extract_apk_raw_header",
    "extract_header_features",
    "extract_headers_from_dex_list",
    "extract_manifest_tokens",
    "list_dex_entries",
    "load_vocab",
    "multidex_settings",
    "parse_dex_header_fields",
    "read_all_dex_from_apk",
    "save_vocab",
    "validate_magic",
]
