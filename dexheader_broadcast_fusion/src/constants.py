"""Shared constants for dexheader_broadcast_fusion."""

MODEL_ID = "dexheader_broadcast_fusion"
DOMAIN_ID = "dex_header_receiver_actions"

# Valid Dex magic: b"dex\n035\0" (bytes 64 65 78 0A 30 33 35 00)
DEX_MAGIC = b"dex\n035\x00"
DEX_MAGIC_LEN = 8

# DexHeader is 0x70 (112) bytes on standard Dex; used when validating buffer size.
DEX_HEADER_SIZE = 0x70

# Manifest permission normalization (pyaxmlparser path in manifest_decode.py)
PERMISSION_PREFIX = "android.permission."
PERMISSION_TOKEN_PREFIX = "permissions::"
