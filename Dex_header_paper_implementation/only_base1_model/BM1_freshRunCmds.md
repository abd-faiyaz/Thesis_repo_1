# BM1 — Fresh full run (temporal split, same as Pattern A/B)

Train on **2020–2021**, validate/test on **2022–2023** (`preprocessing.split_mode: temporal_year`).

## 1) Remove all previous BM1 outputs

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model

rm -rf artifacts/processed artifacts/checkpoints artifacts/metrics
rm -rf artifacts/export artifacts/parity artifacts/splits
rm -f artifacts/normalization.json artifacts/failed_apks.log
rm -rf output_archives
```

## 2) Full pipeline from scratch (logged + archived)

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model

export BM1_ARCHIVE=1
export BM1_RUN_ID="run_$(date +%Y%m%d)_temporal_split"
export APK_ROOT=/mnt/Files/thesis_full_dataset

FRESH_TRAIN=1 INSTALL_DEPS=0 ./run_base_model_1.sh
```

## 3) Step-by-step (only if debugging)

`run_base_model_1.sh` runs the same steps (P0–P8 + archive). Use individual scripts only when skipping phases, e.g. `SKIP_PREPROCESS=1 ./run_base_model_1.sh`.

## 4) After the run — verify metrics and splits

```bash
RUN_ID="$(cat output_archives/LATEST_RUN.txt)"
ls -la artifacts/metrics/
ls -la artifacts/splits/
wc -l artifacts/splits/train.txt artifacts/splits/val.txt
head -3 artifacts/splits/train.txt
head -3 artifacts/splits/val.txt
cat artifacts/metrics/training_run_info.json | head -30
```

Expected:

| Path | Content |
|------|---------|
| `artifacts/splits/train.txt` | APK paths under `2020/` and `2021/` |
| `artifacts/splits/val.txt` | APK paths under `2022/` and `2023/` |
| `artifacts/metrics/training_run_info.json` | `split_mode: temporal_year`, year lists |
| `artifacts/metrics/metrics_val.json` | Metrics on 2022–2023 holdout |

## 5) Quick unit tests (before long run)

```bash
cd /mnt/Files/thesis_vigidroid/Dex_header_paper_implementation/only_base1_model
export PYTHONPATH=.
python -m unittest tests.test_splits tests.test_dataset -v
```
