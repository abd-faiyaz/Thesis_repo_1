#!/usr/bin/env bash
set -euo pipefail
# Push all 2022-2023 thesis device-eval APKs (1528 total) to /sdcard/Download/Scanable.
# Skips files already present on the phone so the run can be resumed after adb errors.
# Requires: adb, USB debugging, phone connected.
DEST="/sdcard/Download/Scanable"
MANIFEST='/mnt/Files/thesis_vigidroid/Android_Works/device_eval_manifest.csv'
adb shell "mkdir -p \"$DEST\""
total=1528
pushed=0
skipped=0
failed=0
echo "Pushing up to $total APKs to $DEST (skipping existing) ..."
# Use process substitution so adb push cannot consume the manifest on stdin.
while IFS=, read -r apk_id phone_filename source_path _rest; do
  dest="$DEST/$phone_filename"
  if adb shell "[ -f \"$dest\" ]" </dev/null >/dev/null 2>&1; then
    skipped=$((skipped + 1))
    continue
  fi
  if adb push "$source_path" "$dest" </dev/null; then
    pushed=$((pushed + 1))
  else
    echo "adb push failed for $phone_filename" >&2
    failed=$((failed + 1))
    exit 1
  fi
done < <(tail -n +2 "$MANIFEST")

echo "Done. APKs are in $DEST"
adb shell ls -1 "$DEST"/*.apk | wc -l
