# shared_calibration

Canonical threshold tuning and cascade band calibration used by all thesis model pipelines.

## Install

From the repo root (included in `requirements-thesis-all.txt`):

```bash
thesis_venv/bin/pip install -e ./shared_calibration
```

## API

```python
from shared_calibration import (
    tune_threshold,
    calibrate_cascade_thresholds,
    build_thresholds_payload,
    write_thresholds,
)

tuned = tune_threshold(y_val, scores_val)
bands = calibrate_cascade_thresholds(
    y_val,
    scores_val,
    target_false_omission_rate=0.02,
    target_false_alarm_at_thigh=0.02,
)
payload = build_thresholds_payload(
    model_id="mldp_pruned_permission",
    default=0.5,
    tuned_val=tuned,
    cascade=bands,
)
write_thresholds(Path("artifacts/metrics/thresholds.json"), payload)
```

`build_thresholds_payload` writes the canonical `thresholds.json` shape and duplicates
`tuned_val` as `malware_threshold` for existing Android loaders.
