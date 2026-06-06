from pathlib import Path


class LabelError(ValueError):
    pass


def _normalize_name(value: str) -> str:
    return value.strip().lower()


def infer_label_from_parent(
    apk_path: Path,
    *,
    benign_names: set[str],
    malicious_names: set[str],
) -> int:
    for parent in apk_path.parents:
        name = _normalize_name(parent.name)
        if name in benign_names:
            return 0
        if name in malicious_names:
            return 1
    raise LabelError(f"No label folder found for {apk_path}")
