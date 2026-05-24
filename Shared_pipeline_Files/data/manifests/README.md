# APK manifest

This folder holds **metadata only** — not APK binaries.

## Build the index

From repo root:

```bash
python Shared_pipeline_Files/tools/build_apk_manifest.py \
  --config Shared_pipeline_Files/data/dataset_paths.yaml
```

## Output

`apk_index.csv` columns:

| Column | Description |
|--------|-------------|
| `apk_path` | Path relative to `apk_root` (or absolute) |
| `sha256` | File hash |
| `label` | `benign` or `malware` |
| `year` | Optional; parsed from path/metadata if present |
| `split` | Empty until `split_dataset.py` runs |

## Label inference

Default: **parent folder name** (`benign/`, `malware/`, etc.). Override with `--labels-csv` for explicit labels.

`apk_index.csv` is gitignored when large; keep a small sample or regenerate from `apk_root`.
