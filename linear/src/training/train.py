"""Train LinRegDroid MLR on permission vectors."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch

from src.config import ensure_artifact_dirs, load_config
from src.constants import DOMAIN_ID, MODEL_ID
from src.data.dataset import stack_split_arrays
from src.models.mlr import LinRegDroidModule, fit_mlr

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def train_model(cfg, *, config_path: Path | None = None) -> Path:
    processed = cfg.paths.processed
    X_train, y_train = stack_split_arrays(processed, "train")
    fit_intercept = bool(cfg.model.get("fit_intercept", True))
    fit = fit_mlr(X_train, y_train, fit_intercept=fit_intercept)

    checkpoint_dir = cfg.paths.checkpoints
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if cfg.training.get("save_sklearn", True):
        joblib.dump(fit.sklearn_model, checkpoint_dir / "linregdroid.joblib")

    torch_module = LinRegDroidModule.from_sklearn(fit.sklearn_model, fit.feature_dim)
    checkpoint = {
        "model_state_dict": torch_module.state_dict(),
        "feature_dim": fit.feature_dim,
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "variant": cfg.model.get("variant", "linregdroid1"),
        "trained_at": _utc_now(),
        "n_train": int(X_train.shape[0]),
        "intercept": fit.intercept,
    }
    ckpt_path = cfg.paths.latest_checkpoint
    torch.save(checkpoint, ckpt_path)

    coef_payload = {
        "intercept": fit.intercept,
        "coefficients": [
            {"permission": name, "beta": float(beta)}
            for name, beta in zip(_load_permission_names(cfg), fit.coefficients)
        ],
    }
    coef_path = checkpoint_dir / "coefficients.json"
    coef_path.write_text(json.dumps(coef_payload, indent=2) + "\n", encoding="utf-8")

    meta = {
        "model_id": MODEL_ID,
        "n_train": int(X_train.shape[0]),
        "M": fit.feature_dim,
        "trained_at": _utc_now(),
        "config": str(config_path or (cfg.root / "config" / "default.yaml")),
    }
    (checkpoint_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return ckpt_path


def _load_permission_names(cfg) -> list[str]:
    vocab = json.loads(cfg.paths.permission_vocab.read_text(encoding="utf-8"))
    return list(vocab["permissions"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit LinRegDroid MLR on train split.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    ckpt = train_model(cfg, config_path=args.config)
    print(f"Training complete → {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
