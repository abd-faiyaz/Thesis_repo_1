# Dex Header + Broadcast Receiver Fusion (`dexheader_broadcast_fusion`)

Hybrid model: **MLP(H) trunk** (104-d Dex header, BM1 normalization) + **receiver system-action BoW** → embedding fusion → malware score.

- Plan: `detailed_implementation_plans/dexheader_brdcst_rec_fusion_full_impl_opus.md`
- Answers: `detailed_implementation_plans/opus_answers_dxh_bc_fusion.md`

## Decisions (locked)

| Item | Choice |
|------|--------|
| Split | Broadcast style: train 2020–2021; val/test 50/50 of 2022+2023 |
| Manifest decoder | `pyaxmlparser` |
| Dex normalization | Reuse shipped BM1 `normalization_header.json` |
| `system_actions.json` | Reuse `broadcast_mldp_hybrid` (172 actions) |
| Fusion | MLP head, `d_R=32`, multidex `sum`, two-input ONNX |
| Training default | `SMOKE=1` → 2 epochs (set `SMOKE=0` for full train) |

## Run order

```bash
cd dexheader_broadcast_fusion
source scripts/activate_thesis_env.sh

# P0
python scripts/verify_setup.py

# P1–P8 (smoke train by default)
./run_dexheader_broadcast_fusion.sh

# Full corpus preprocess, smoke train (staging to VigiDroid assets is on by default)
SMOKE=1 ./run_dexheader_broadcast_fusion.sh

# Full training, skip Android staging
SMOKE=0 EPOCHS=60 STAGE_ANDROID=0 ./run_dexheader_broadcast_fusion.sh
```

## Android integration status

| Phase | Status |
|-------|--------|
| **A1** `DexheaderBroadcastFusionExtractor` | Implemented |
| **A2** `DexheaderBroadcastFusionOnnxRunner` + `ModelRegistry` | Implemented |
| **A3** ScanService / cascade stage | **TODO-future** — wire after cascade policy lands (`vigidroid_cascading_tasks.md`) |
| **A4** Device parity test | **TODO-future** — after export bundle on device |

## Artifacts

- Processed shards: `artifacts/processed/features_{train,val,test}.pt` (`H`, `R`, `y`)
- Checkpoint: `artifacts/checkpoints/best.pt`
- Export: `artifacts/export/dexheader_broadcast_fusion/`
- VigiDroid assets: `vigidroid/app/src/main/assets/models/dexheader_broadcast_fusion/`
