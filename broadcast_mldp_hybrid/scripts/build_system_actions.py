#!/usr/bin/env python3
"""Compile Android OS broadcast system-action allow-list (M3).

Union of broadcast-related Intent / Telephony / connectivity constants for
API 21 through target SDK 36 (matches VigiDroid compileSdk). The output JSON
is the single source of truth for Python training and Android inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Curated broadcast/system actions from Android SDK constants (API 21–36).
# Excludes custom app actions; only OS-defined strings Mohsen #12 would count.
SYSTEM_BROADCAST_ACTIONS: tuple[str, ...] = (
    # --- android.intent.action.* (lifecycle / device / package) ---
    "android.intent.action.ACTION_AIRPLANE_MODE_CHANGED",
    "android.intent.action.ACTION_BATTERY_LOW",
    "android.intent.action.ACTION_BATTERY_OKAY",
    "android.intent.action.ACTION_BOOT_COMPLETED",
    "android.intent.action.ACTION_CAMERA_BUTTON",
    "android.intent.action.ACTION_CLOSE_SYSTEM_DIALOGS",
    "android.intent.action.ACTION_CONFIGURATION_CHANGED",
    "android.intent.action.ACTION_DEVICE_STORAGE_LOW",
    "android.intent.action.ACTION_DEVICE_STORAGE_OK",
    "android.intent.action.ACTION_DOCK_EVENT",
    "android.intent.action.ACTION_EXTERNAL_APPLICATIONS_AVAILABLE",
    "android.intent.action.ACTION_EXTERNAL_APPLICATIONS_UNAVAILABLE",
    "android.intent.action.ACTION_HEADSET_PLUG",
    "android.intent.action.ACTION_HDMI_AUDIO_PLUG",
    "android.intent.action.ACTION_INPUT_METHOD_CHANGED",
    "android.intent.action.ACTION_LOCALE_CHANGED",
    "android.intent.action.ACTION_MANAGE_PACKAGE_STORAGE",
    "android.intent.action.ACTION_MEDIA_BAD_REMOVAL",
    "android.intent.action.ACTION_MEDIA_CHECKING",
    "android.intent.action.ACTION_MEDIA_EJECT",
    "android.intent.action.ACTION_MEDIA_MOUNTED",
    "android.intent.action.ACTION_MEDIA_NOFS",
    "android.intent.action.ACTION_MEDIA_REMOVED",
    "android.intent.action.ACTION_MEDIA_SCANNER_FINISHED",
    "android.intent.action.ACTION_MEDIA_SCANNER_STARTED",
    "android.intent.action.ACTION_MEDIA_SHARED",
    "android.intent.action.ACTION_MEDIA_UNMOUNTABLE",
    "android.intent.action.ACTION_MEDIA_UNMOUNTED",
    "android.intent.action.ACTION_MY_PACKAGE_REPLACED",
    "android.intent.action.ACTION_MY_PACKAGE_SUSPENDED",
    "android.intent.action.ACTION_MY_PACKAGE_UNSUSPENDED",
    "android.intent.action.ACTION_NEW_OUTGOING_CALL",
    "android.intent.action.ACTION_PACKAGE_ADDED",
    "android.intent.action.ACTION_PACKAGE_CHANGED",
    "android.intent.action.ACTION_PACKAGE_DATA_CLEARED",
    "android.intent.action.ACTION_PACKAGE_FULLY_REMOVED",
    "android.intent.action.ACTION_PACKAGE_NEEDS_VERIFICATION",
    "android.intent.action.ACTION_PACKAGE_REMOVED",
    "android.intent.action.ACTION_PACKAGE_REPLACED",
    "android.intent.action.ACTION_PACKAGE_RESTARTED",
    "android.intent.action.ACTION_PACKAGE_VERIFIED",
    "android.intent.action.ACTION_POWER_CONNECTED",
    "android.intent.action.ACTION_POWER_DISCONNECTED",
    "android.intent.action.ACTION_PROVIDER_CHANGED",
    "android.intent.action.ACTION_SHUTDOWN",
    "android.intent.action.ACTION_TIMEZONE_CHANGED",
    "android.intent.action.ACTION_TIME_CHANGED",
    "android.intent.action.ACTION_UID_REMOVED",
    "android.intent.action.ACTION_USER_ADDED",
    "android.intent.action.ACTION_USER_BACKGROUND",
    "android.intent.action.ACTION_USER_FOREGROUND",
    "android.intent.action.ACTION_USER_INFO_CHANGED",
    "android.intent.action.ACTION_USER_INITIALIZE",
    "android.intent.action.ACTION_USER_PRESENT",
    "android.intent.action.ACTION_USER_REMOVED",
    "android.intent.action.ACTION_USER_STARTED",
    "android.intent.action.ACTION_USER_STOPPED",
    "android.intent.action.ACTION_USER_SWITCHED",
    "android.intent.action.ACTION_USER_UNLOCKED",
    "android.intent.action.ACTION_WALLPAPER_CHANGED",
    "android.intent.action.BOOT_COMPLETED",
    "android.intent.action.DEVICE_STORAGE_LOW",
    "android.intent.action.DEVICE_STORAGE_OK",
    "android.intent.action.DOWNLOAD_COMPLETE",
    "android.intent.action.DOWNLOAD_NOTIFICATION_CLICKED",
    "android.intent.action.DREAMING_STARTED",
    "android.intent.action.DREAMING_STOPPED",
    "android.intent.action.LOCKED_BOOT_COMPLETED",
    "android.intent.action.NEW_OUTGOING_CALL",
    "android.intent.action.PHONE_STATE",
    "android.intent.action.PRE_BOOT_COMPLETED",
    "android.intent.action.QUERY_PACKAGE_RESTART",
    "android.intent.action.REBOOT",
    "android.intent.action.REPORT_APP_SCORE",
    "android.intent.action.SCREEN_OFF",
    "android.intent.action.SCREEN_ON",
    "android.intent.action.SIM_STATE_CHANGED",
    "android.intent.action.SUBSCRIPTION_INFO_RECORDED",
    "android.intent.action.USER_PRESENT",
    # --- Telephony / SMS ---
    "android.provider.Telephony.ACTION_DEFAULT_SMS_PACKAGE_CHANGED",
    "android.provider.Telephony.SIM_FULL",
    "android.provider.Telephony.SMS_CB_RECEIVED",
    "android.provider.Telephony.SMS_DELIVER",
    "android.provider.Telephony.SMS_EMERGENCY_CB_RECEIVED",
    "android.provider.Telephony.SMS_RECEIVED",
    "android.provider.Telephony.SMS_REJECTED",
    "android.provider.Telephony.SMS_SERVICE_CATEGORY_PROGRAM_DATA_RECEIVED",
    "android.provider.Telephony.WAP_PUSH_DELIVER",
    "android.provider.Telephony.WAP_PUSH_RECEIVED",
    # --- Connectivity / Wi‑Fi / Bluetooth ---
    "android.bluetooth.a2dp.profile.action.CONNECTION_STATE_CHANGED",
    "android.bluetooth.a2dp.profile.action.PLAYING_STATE_CHANGED",
    "android.bluetooth.adapter.action.CONNECTION_STATE_CHANGED",
    "android.bluetooth.adapter.action.STATE_CHANGED",
    "android.bluetooth.avrcp-controller.profile.action.BROWSE_CONNECTION_STATE_CHANGED",
    "android.bluetooth.avrcp-controller.profile.action.CONNECTION_STATE_CHANGED",
    "android.bluetooth.device.action.ACL_CONNECTED",
    "android.bluetooth.device.action.ACL_DISCONNECTED",
    "android.bluetooth.device.action.BOND_STATE_CHANGED",
    "android.bluetooth.device.action.FOUND",
    "android.bluetooth.device.action.NAME_CHANGED",
    "android.bluetooth.device.action.UUID",
    "android.bluetooth.headset.action.AUDIO_STATE_CHANGED",
    "android.bluetooth.headset.action.VENDOR_SPECIFIC_HEADSET_EVENT",
    "android.bluetooth.headset.profile.action.CONNECTION_STATE_CHANGED",
    "android.net.conn.CAPTIVE_PORTAL",
    "android.net.conn.CONNECTIVITY_CHANGE",
    "android.net.conn.RESTRICT_BACKGROUND_CHANGED",
    "android.net.wifi.LINK_CONFIGURATION_CHANGED",
    "android.net.wifi.NETWORK_IDS_CHANGED",
    "android.net.wifi.NETWORK_STATE_CHANGED",
    "android.net.wifi.RSSI_CHANGED",
    "android.net.wifi.SCAN_RESULTS",
    "android.net.wifi.STATE_CHANGE",
    "android.net.wifi.WIFI_AP_STATE_CHANGED",
    "android.net.wifi.WIFI_STATE_CHANGED",
    "android.net.wifi.action.WIFI_NETWORK_SUGGESTION_POST_CONNECTION",
    "android.net.wifi.p2p.CONNECTION_STATE_CHANGE",
    "android.net.wifi.p2p.PEERS_CHANGED",
    "android.net.wifi.p2p.STATE_CHANGED",
    "android.net.wifi.p2p.THIS_DEVICE_CHANGED",
    "android.net.wifi.supplicant.CONNECTION_CHANGE",
    "android.net.wifi.supplicant.STATE_CHANGE",
    # --- Location / USB / NFC ---
    "android.hardware.usb.action.USB_ACCESSORY_ATTACHED",
    "android.hardware.usb.action.USB_ACCESSORY_DETACHED",
    "android.hardware.usb.action.USB_DEVICE_ATTACHED",
    "android.hardware.usb.action.USB_DEVICE_DETACHED",
    "android.hardware.usb.action.USB_PORT_CHANGED",
    "android.hardware.usb.action.USB_STATE",
    "android.location.MODE_CHANGED",
    "android.location.PROVIDERS_CHANGED",
    "android.nfc.action.ADAPTER_STATE_CHANGED",
    "android.nfc.action.NDEF_DISCOVERED",
    "android.nfc.action.TAG_DISCOVERED",
    "android.nfc.action.TECH_DISCOVERED",
    # --- Power / idle / battery ---
    "android.os.action.CHARGING",
    "android.os.action.DEVICE_IDLE_MODE_CHANGED",
    "android.os.action.DISCHARGING",
    "android.os.action.LIGHT_DEVICE_IDLE_MODE_CHANGED",
    "android.os.action.POWER_SAVE_MODE_CHANGED",
    # --- Storage ---
    "android.os.storage.action.DISK_DESTROYED",
    "android.os.storage.action.DISK_SCANNED",
    "android.os.storage.action.VOLUME_EJECTED",
    "android.os.storage.action.VOLUME_REMOVED",
    "android.os.storage.action.VOLUME_STATE_CHANGED",
    # --- App / admin / alarms ---
    "android.app.action.APP_BLOCK_STATE_CHANGED",
    "android.app.action.NEXT_ALARM_CLOCK_CHANGED",
    "android.app.action.NOTIFICATION_CHANNEL_BLOCK_STATE_CHANGED",
    "android.app.action.NOTIFICATION_CHANNEL_GROUP_BLOCK_STATE_CHANGED",
    "android.app.action.SCHEDULE_EXACT_ALARM_PERMISSION_STATE_CHANGED",
    "android.app.action.ACTION_DEVICE_ADMIN_DISABLED",
    "android.app.action.ACTION_DEVICE_ADMIN_ENABLED",
    "android.app.action.ACTION_PASSWORD_CHANGED",
    "android.app.action.ACTION_PASSWORD_FAILED",
    "android.app.action.ACTION_PASSWORD_SUCCEEDED",
    "android.app.action.ACTION_PROFILE_PROVISIONING_COMPLETE",
    # --- Media / camera ---
    "android.media.AUDIO_BECOMING_NOISY",
    "android.media.STREAM_MUTE_CHANGED_ACTION",
    "android.media.VOLUME_CHANGED_ACTION",
    "android.media.action.CLOSE_AUDIO_EFFECT_CONTROL_SESSION",
    "android.media.action.HDMI_AUDIO_PLUG",
    "android.media.action.OPEN_AUDIO_EFFECT_CONTROL_SESSION",
    "android.hardware.action.CAMERA_CLOSED",
    "android.hardware.action.CAMERA_DISCONNECTED",
    "android.hardware.action.NEW_PICTURE",
    "android.hardware.action.NEW_VIDEO",
    # --- Telecom / settings ---
    "android.telecom.action.DEFAULT_DIALER_CHANGED",
    "android.telecom.action.PHONE_ACCOUNT_REGISTERED",
    "android.telecom.action.PHONE_ACCOUNT_UNREGISTERED",
    "android.telecom.action.SHOW_MISSED_CALLS_NOTIFICATION",
    "android.settings.action.MANUFACTURER_APPLICATION_SETTING_DELETED",
    "android.settings.action.MANUFACTURER_APPLICATION_SETTING_INSERTED",
    "android.settings.action.MANUFACTURER_APPLICATION_SETTING_UPDATED",
)


def build_payload(*, min_api: int = 21, max_api: int = 36) -> dict:
    actions = sorted(set(SYSTEM_BROADCAST_ACTIONS))
    return {
        "version": 1,
        "description": "Android OS broadcast system actions (Mohsen #12 allow-list, M3)",
        "api_range": {"min_sdk": min_api, "max_sdk": max_api},
        "count": len(actions),
        "actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write assets/system_actions.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "assets" / "system_actions.json",
    )
    parser.add_argument("--min-api", type=int, default=21)
    parser.add_argument("--max-api", type=int, default=36)
    args = parser.parse_args()

    payload = build_payload(min_api=args.min_api, max_api=args.max_api)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {payload['count']} system actions → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
