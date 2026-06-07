"""Mode B Stage 2 — deployed MLP(H) ONNX reference (no retrain)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import PipelineConfig
from src.constants import DEX_FEATURE_DIM


class DeployedMlpHeaderRef:
    """ONNX Runtime wrapper around the shipped mlp_header bundle."""

    def __init__(
        self,
        session: Any,
        *,
        input_name: str,
        output_name: str,
        feature_dim: int = DEX_FEATURE_DIM,
        bundle_dir: Path,
    ) -> None:
        self.session = session
        self.input_name = input_name
        self.output_name = output_name
        self.feature_dim = feature_dim
        self.bundle_dir = bundle_dir

    @classmethod
    def from_bundle(cls, bundle_dir: Path | str) -> DeployedMlpHeaderRef:
        import onnxruntime as ort

        root = Path(bundle_dir).resolve()
        onnx_path = root / "model.onnx"
        manifest_path = root / "export_manifest.json"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Missing deployed ONNX: {onnx_path}")

        input_name = "features"
        output_name = "malware_probability"
        feature_dim = DEX_FEATURE_DIM
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            inputs = manifest.get("inputs") or []
            outputs = manifest.get("outputs") or []
            if inputs:
                input_name = str(inputs[0].get("name", input_name))
            if outputs:
                output_name = str(outputs[0].get("name", output_name))
            feature_dim = int(manifest.get("feature_dim", feature_dim))

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        return cls(
            session,
            input_name=input_name,
            output_name=output_name,
            feature_dim=feature_dim,
            bundle_dir=root,
        )

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> DeployedMlpHeaderRef:
        return cls.from_bundle(cfg.paths.deployed_mlp_header_bundle)

    def score(self, h: np.ndarray | torch.Tensor) -> np.ndarray:
        arr = np.asarray(h, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected H dim {self.feature_dim}, got {arr.shape[-1]}"
            )
        out = self.session.run(
            [self.output_name],
            {self.input_name: arr},
        )[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def score_scalar(self, h: np.ndarray | torch.Tensor) -> float:
        return float(self.score(h)[0])
