# MLDP Permissions + Dex Header Cascade

Cross-paper hybrid: MLDP-pruned manifest permissions (#7 Ghasempour et al.) +
Dex header structural bytes + MLP(H) (MSFDroid-style, deployed in VigiDroid).

**Papers:** IJACSA 2020 (MLDP) · Dex header grounded on deployed `mlp_header` bundle

**Deployment modes:**
- **Mode A** — early fusion `x = [x_S ‖ H]` → tiny MLP (`d→64→1`)
- **Mode B** — two-stage cascade: MLDP logistic → (if uncertain) deployed MLP(H) on `H`

## Setup (P0)

```bash
cd mldp_dexheader_cascade
export ROOT="$PWD"
source scripts/activate_thesis_env.sh
pip install -r ../requirements-thesis-all.txt

python scripts/verify_setup.py
```

Edit `config/default.yaml` → `paths.apk_root` if needed (default: `/mnt/Files/thesis_full_dataset`).

## Full run (recommended)

```bash
cd mldp_dexheader_cascade
export ROOT="$PWD"
source scripts/activate_thesis_env.sh

# Full corpus P0–P8 (+ optional archive + Android staging)
./run_mldp_dexheader_cascade.sh

# Logged archive for thesis snippet
MDH_ARCHIVE=1 ./run_mldp_dexheader_cascade.sh

# PC pipeline + copy bundle into vigidroid assets
STAGE_ANDROID=1 ./run_mldp_dexheader_cascade.sh
```

## Run order (step-by-step)

| Phase | Command |
|-------|---------|
| **All** | `./run_mldp_dexheader_cascade.sh` |
| P0 | `python scripts/verify_setup.py` |
| P1 | `bash scripts/run_index.sh` |
| P2 | `bash scripts/run_mldp.sh` · `bash scripts/run_preprocess.sh` |
| P2 (smoke) | `PREPROCESS_LIMIT=200 bash scripts/run_preprocess.sh` |
| P3 | `bash scripts/run_verify_dataloader.sh` |
| P4 | `bash scripts/run_verify_model.sh` |
| P5 | `bash scripts/run_train.sh` |
| P5 (quick) | `QUICK=1 bash scripts/run_train.sh` |
| P6 | `bash scripts/run_evaluate.sh` |
| P7 | `python scripts/export_onnx.py` |
| P8 | `bash scripts/run_parity.sh` |
| Stage Android | `bash ../Android_Works/stage_mldp_dexheader_cascade.sh` |
| A1–A4 | VigiDroid integration (after P8 green) |

## Split policy

- **Train:** 2020 + 2021
- **Val:** 10% stratified holdout from train years (early stopping + threshold calibration)
- **Test:** all 2022 + 2023
- MLDP set `S` and dex min/max are built from **train only** (P2)

## Related docs

- `../detailed_implementation_plans/mldp_dexheader_cascade_full_impl_opus.md`
- `../detailed_implementation_plans/opus_answers_mldp_dexheader_cascade.md`
