"""Shared constants for Dex header and manifest BoW (Phase 2+)."""

# Valid Dex magic: b"dex\n035\0"
DEX_MAGIC = b"dex\n035\x00"
DEX_MAGIC_LEN = 8
DEX_HEADER_SIZE = 0x70

# Feature bytes 8–111 after magic → 104 floats (/255)
DEX_HEADER_FEATURE_START = 8
DEX_HEADER_FEATURE_END = 112
DEX_HEADER_FEATURE_DIM = 104

# Manifest BoW (paper default)
DEFAULT_LEXICON_SIZE = 4380
DEFAULT_UNK_INDEX = 4380

# Combined ASCNN input: header_dim + (lexicon_size + 1)
DEFAULT_COMBINED_INPUT_LEN = DEX_HEADER_FEATURE_DIM + DEFAULT_LEXICON_SIZE + 1

# Multi-dex discovery (basename regex)
DEFAULT_DEX_PATTERN = r"^classes(\d*)\.dex$"
DEFAULT_MULTIDEX_MODE = "sum"
