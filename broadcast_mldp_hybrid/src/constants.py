"""Model identifiers and manifest token conventions."""

MODEL_ID = "broadcast_mldp_hybrid"
DOMAIN_ID = "manifest_mldp_perm_receiver_actions"

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
