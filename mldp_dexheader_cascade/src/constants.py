"""Model identifiers, Dex header constants, and MLDP token conventions."""

MODEL_ID = "mldp_dexheader_cascade"
DOMAIN_ID = "manifest_mldp_perm_dex_header"

# Valid Dex magic: b"dex\n035\0"
DEX_MAGIC = b"dex\n035\x00"
DEX_MAGIC_LEN = 8  # magic field length; feature bytes start at offset 8
DEX_HEADER_SIZE = 0x70
DEX_FEATURE_DIM = 104

PERMISSION_PREFIX = "android.permission."
PERMISSION_TOKEN_PREFIX = "permissions::"

# Paper #7 Table I fallback (22 permissions), normalized tokens.
PUBLISHED_MLDP_PERMISSIONS: tuple[str, ...] = (
    "permissions::send_sms",
    "permissions::read_sms",
    "permissions::receive_sms",
    "permissions::read_phone_state",
    "permissions::call_phone",
    "permissions::read_contacts",
    "permissions::write_contacts",
    "permissions::read_call_log",
    "permissions::write_call_log",
    "permissions::camera",
    "permissions::record_audio",
    "permissions::access_fine_location",
    "permissions::access_coarse_location",
    "permissions::read_external_storage",
    "permissions::write_external_storage",
    "permissions::get_accounts",
    "permissions::receive_boot_completed",
    "permissions::install_packages",
    "permissions::request_install_packages",
    "permissions::system_alert_window",
    "permissions::disable_keyguard",
    "permissions::wake_lock",
)

LABEL_BENIGN = 0
LABEL_MALWARE = 1
