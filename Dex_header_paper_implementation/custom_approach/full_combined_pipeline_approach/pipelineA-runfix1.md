# Pattern A — Run Error Analysis & Fix Plan (runfix1)

Source log: `pipelineA-runError1.txt`

Training completed all 80 epochs and wrote checkpoints, but **evaluation crashed** and the log was **flooded with RuntimeWarnings** every epoch. Loss also stayed flat (~1.03), which is a separate quality concern.

---

## Issue 1 — `RuntimeWarning` spam during training/eval

### Symptom

Repeated every train/val epoch (and at eval start):

```
RuntimeWarning: 'src.training.train' found in sys.modules after import of package
'src.training', but prior to execution of 'src.training.train'; this may result in
unpredictable behaviour
```

Four warnings appear at each train/val epoch start because `num_workers=4` in the DataLoader forks worker processes that re-import the package.

### Root cause

`src/training/__init__.py` **eagerly imports** submodules:

```python
from src.training.train import run_training
from src.training.evaluate import run_evaluation, ...
```

When you run `python -m src.training.train`, Python loads the `src.training` package first (which registers `src.training.train` in `sys.modules`), then `runpy` tries to execute `train` as `__main__` — triggering the warning.

BM1 already fixed this pattern: `only_base1_model/src/training/__init__.py` documents the issue and **does not** import `train`/`evaluate` at package init.

### Fix steps

1. **Edit** `src/training/__init__.py` — remove eager imports of `train`, `evaluate`, and other submodules; keep only a short docstring (mirror BM1).
2. **Verify** callers import submodules directly (`from src.training.train import run_training`) — already the case in tests/scripts.
3. **Run** `python -m src.training.train --fresh --epochs 1` and confirm warnings are gone (or reduced to zero on main process).

---

## Issue 2 — Evaluation crash: `TypeError: RNG state must be a torch.ByteTensor`

### Symptom

Pipeline fails at **Evaluate (ACC, F1, AUC)** after successful training:

```
File src/training/checkpoint.py, restore_rng_state
  torch.set_rng_state(state["torch"])
TypeError: RNG state must be a torch.ByteTensor
```

Checkpoints exist at `artifacts/checkpoints/best.pt` and `latest.pt`.

### Root cause

Pattern A checkpoints include `rng_state` (CPU tensors from `torch.get_rng_state()`).

In `src/training/evaluate.py`, `run_evaluation` loads the checkpoint **twice**:

1. `load_checkpoint(..., map_location="cpu")` — OK
2. `load_checkpoint(..., map_location=device)` where `device=cuda` — **moves all tensors in the checkpoint to GPU**, including `rng_state["torch"]`

`torch.set_rng_state()` requires a **CPU** `ByteTensor`. Passing a CUDA tensor causes the crash.

Same pattern exists in `src/training/train.py` (resume path, line ~59) — resume-after-outage on CUDA could hit the same bug.

BM1 checkpoints **do not store RNG state**, so BM1 never hit this.

### Fix steps

1. **Edit** `src/training/checkpoint.py`:
   - In `restore_rng_state`, coerce RNG tensors to CPU before restore:
     ```python
     t = state["torch"]
     if not isinstance(t, torch.Tensor):
         t = torch.tensor(t)
     torch.set_rng_state(t.cpu())
     ```
   - Same for `torch_cuda` entries if present (already handled by CUDA API, but guard types).

2. **Edit** `src/training/evaluate.py`:
   - Load checkpoint **once** on CPU only.
   - Add `restore_from_checkpoint(..., restore_rng=False)` **or** a slim `load_model_from_checkpoint()` that only loads `model_state_dict` (eval does not need optimizer/scheduler/RNG).
   - Remove redundant second `load_checkpoint(..., map_location=device)`.

3. **Edit** `src/training/checkpoint.py` — add optional flag:
   ```python
   def restore_from_checkpoint(..., *, restore_rng: bool = True) -> ...
   ```
   Set `restore_rng=False` in evaluation; keep `True` for training resume.

4. **Edit** `src/training/train.py`:
   - On resume, load checkpoint on **CPU**, call `restore_from_checkpoint`, then `model.to(device)` — avoid `map_location=cuda` for full checkpoint dict.

5. **Verify** with existing checkpoints (no retrain needed):
   ```bash
   cd Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach
   export PYTHONPATH=.
   APK_ROOT=/mnt/Files/thesis_full_dataset \
     ../../../thesis_venv/bin/python -m src.training.evaluate \
       --checkpoint artifacts/checkpoints/best.pt --split val
   ```

6. **Re-run packaging only** if eval succeeds:
   ```bash
   SKIP_PREPROCESS=1 SKIP_TRAIN=1 SKIP_DEX_STATS=1 \
     APK_ROOT=/mnt/Files/thesis_full_dataset ./run_pattern_a.sh
   ```

---

## Issue 3 — Flat loss (~1.03) across all 80 epochs (non-fatal, investigate)

### Symptom

```
Epoch 1/80  — train_loss=1.0299 val_loss=1.0300
Epoch 80/80 — train_loss=1.0293 val_loss=1.0296
```

LR decayed (0.005 → 0.00002) but loss barely moved — model likely **did not learn**.

### Likely causes (needs confirmation after Issue 1–2 fixes)

| Hypothesis | Why |
|------------|-----|
| Class-weighted BCE baseline ~1.0 | `pos_weight≈2.88` (from `class_balance.json`); constant predictor can sit near 1.0 |
| Model/gradient issue | ASCNN on sparse 4485-dim input — gradients may be weak; needs one-batch gradient check |
| Feature pipeline OK but signal weak | Full vocab ≈2172 tokens; preprocessing succeeded (12154 train / 1335 val shards) |
| Not caused by eval bug | Training loop ran; checkpoints saved |

### Diagnostic steps (after eval fix)

1. Run eval on saved checkpoint — check **accuracy / F1 / AUC** (may be ~random).
2. Run `scripts/verify_model.py` — forward pass + param count.
3. One-batch overfit test: 1 epoch on 32 samples, loss should drop sharply if gradients flow.
4. Compare with BM1 learning curve on header-only (ablation reference).
5. If confirmed broken: inspect `ASCNNCombined` + `AdaptiveShrinkageUnit` init/forward, learning rate, and whether BoW multihot sparsity needs different tower handling.

**Defer major architecture/hyperparameter changes** until Issues 1–2 are fixed and diagnostics run.

---

## Implementation checklist (awaiting confirmation)

| # | Task | File(s) |
|---|------|---------|
| 1 | Slim `src/training/__init__.py` (no eager train/eval imports) | `src/training/__init__.py` |
| 2 | CPU-safe `restore_rng_state` | `src/training/checkpoint.py` |
| 3 | Add `restore_rng` flag to `restore_from_checkpoint` | `src/training/checkpoint.py` |
| 4 | Eval: load ckpt on CPU, skip RNG/optimizer restore | `src/training/evaluate.py` |
| 5 | Train resume: load ckpt on CPU, then move model to device | `src/training/train.py` |
| 6 | Re-run eval + package on existing checkpoints | manual / `run_pattern_a.sh` |
| 7 | (Optional) Flat-loss diagnostics | scripts / one-off test |

---

## Expected outcome after fixes

- No `RuntimeWarning` spam during `python -m src.training.train`.
- `python -m src.training.evaluate` succeeds on existing `best.pt`.
- Pipeline completes through **Evaluate** and **Package** without re-preprocessing or retraining.
- Flat loss tracked as follow-up if metrics are near chance.

**Status:** Plan only — **awaiting confirmation** before applying code changes.
