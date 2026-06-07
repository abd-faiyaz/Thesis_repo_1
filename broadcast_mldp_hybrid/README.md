# Broadcast + MLDP Permission Hybrid

Cross-paper manifest model: MLDP-pruned permissions (#7 Ghasempour et al.) +
static broadcast receiver system actions (#12 Mohsen et al.), early-fusion tiny MLP.

**Papers:** IJACSA 2020 (MLDP) · IEEE CIC 2017 (broadcast receivers)

## Setup (P0)

```bash
cd broadcast_mldp_hybrid
source scripts/activate_thesis_env.sh
pip install -r requirements.txt   # or repo-root requirements-thesis-all.txt

python scripts/build_system_actions.py
python scripts/verify_setup.py
```

Edit `config/default.yaml` → `paths.apk_root` if needed (default: `/mnt/Files/thesis_full_dataset`).

## Run order

| Phase | Command |
|-------|---------|
| P0 | `python scripts/build_system_actions.py` · `python scripts/verify_setup.py` |
| P1 | `python scripts/index_dataset.py` |
| P2 | `bash scripts/run_preprocess.sh` |
| P2 (smoke) | `PREPROCESS_LIMIT=200 bash scripts/run_preprocess.sh` |
| P3 | `python scripts/verify_dataloader.py` |
| P4 | `python scripts/verify_model.py` |
| P5 | `bash scripts/run_train.sh` |
| P5 (quick) | `QUICK=1 bash scripts/run_train.sh` |
| P6 | `bash scripts/run_evaluate.sh` |
| P7 | `python scripts/export_onnx.py` |
| P7 (deploy) | `python scripts/export_onnx.py --deploy-vigidroid` |
| P8 | `bash scripts/run_parity.sh` |

## Split policy

- **Train:** 2020 + 2021
- **Val / test:** stratified split of 2022 + 2023 only (disjoint, seed 42, 50/50 default)
- MLDP set `S` and receiver vocab `A` are built from **train only** (P2)

## Related docs

- `../detailed_implementation_plans/broadcast_mldp_hybrid_full_impl_opus.md`
- `../detailed_implementation_plans/opus_answers.md`
