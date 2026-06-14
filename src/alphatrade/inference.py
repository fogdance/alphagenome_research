"""Reusable inference interface for AlphaTrade model bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints as flax_ckpt

from alphatrade.core import model as model_lib
from alphatrade.core import schemas
from alphatrade.prediction_schema import build_prediction_frame


def load_bundle_metadata(bundle_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate the minimal bundle metadata files."""
    bundle_path = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle_path / "bundle_manifest.json"
    model_config_path = bundle_path / "model_config.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle_manifest.json not found in {bundle_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"model_config.json not found in {bundle_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    with model_config_path.open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    best_dir = bundle_path / "best"
    if not best_dir.exists():
        raise FileNotFoundError(f"best/ checkpoint directory not found in {bundle_path}")

    return {
        "bundle_dir": str(bundle_path),
        "manifest": manifest,
        "model_config": model_config,
        "checkpoint_dir": str(best_dir),
    }


def config_from_model_config(model_config: dict[str, Any]) -> schemas.AlphaTradeConfig:
    """Build AlphaTradeConfig from a bundle model_config dict."""
    allowed = set(schemas.AlphaTradeConfig.__dataclass_fields__.keys())
    config_kwargs = {k: v for k, v in model_config.items() if k in allowed}
    return schemas.AlphaTradeConfig(**config_kwargs)


def make_forward(config: schemas.AlphaTradeConfig) -> hk.TransformedWithState:
    """Create the Haiku transformed forward used by training and inference."""

    def forward(x):
        model = model_lib.AlphaTrade(config)
        return model(x)

    return hk.transform_with_state(lambda x: forward(x))


class AlphaTradeBundlePredictor:
    """Loads an M9 model bundle and serves batch or single-window inference."""

    def __init__(self, bundle_dir: str | os.PathLike[str]):
        metadata = load_bundle_metadata(bundle_dir)
        self.bundle_dir = metadata["bundle_dir"]
        self.manifest = metadata["manifest"]
        self.model_config_dict = metadata["model_config"]
        self.config = config_from_model_config(self.model_config_dict)
        self.model_version = self.manifest["model_version"]
        self.forward = make_forward(self.config)

        rng = jax.random.PRNGKey(42)
        dummy_x = jnp.zeros(
            (1, self.config.lookback_length, self.config.num_features),
            dtype=jnp.float32,
        )
        params, state = self.forward.init(rng, dummy_x)
        opt_state = optax.adam(1e-3).init(params)
        ckpt_state = {"params": params, "state": state, "opt_state": opt_state, "step": 0}

        restored = flax_ckpt.restore_checkpoint(metadata["checkpoint_dir"], ckpt_state)
        if int(restored["step"]) <= 0:
            raise ValueError(f"No checkpoint found at {metadata['checkpoint_dir']}")

        self.params = restored["params"]
        self.state = restored["state"]
        self.checkpoint_step = int(restored["step"])
        self._predict_fn = jax.jit(self._predict_arrays)

    @property
    def horizons(self) -> list[int]:
        return list(self.config.horizons)

    @property
    def quantiles(self) -> list[float]:
        return list(self.config.quantiles)

    def _predict_arrays(self, features: jax.Array):
        output, _ = self.forward.apply(
            self.params,
            self.state,
            jax.random.PRNGKey(0),
            features,
        )
        return output.log_return_quantiles

    def predict_arrays(self, windows: np.ndarray) -> dict[int, np.ndarray]:
        """Predict quantile arrays for windows shaped [N, lookback, features]."""
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim != 3:
            raise ValueError(f"windows must be 3D [N,L,F], got shape {windows.shape}")
        expected_shape = (self.config.lookback_length, self.config.num_features)
        if tuple(windows.shape[1:]) != expected_shape:
            raise ValueError(
                f"expected window shape [N,{expected_shape[0]},{expected_shape[1]}], "
                f"got {windows.shape}"
            )
        predictions = self._predict_fn(jnp.array(windows))
        return {int(h): np.asarray(values) for h, values in predictions.items()}

    def predict_frame(self, *, symbols: list[str], eobs: list[Any], windows: np.ndarray):
        """Predict and return the stable wide prediction DataFrame."""
        predictions = self.predict_arrays(windows)
        return build_prediction_frame(
            symbols=symbols,
            eobs=eobs,
            model_version=self.model_version,
            predictions_by_horizon=predictions,
            horizons=self.horizons,
            quantiles=self.quantiles,
        )

    def predict_one(
        self,
        *,
        instrument_id: str,
        asof_bar_end: str,
        features: np.ndarray,
        request_id: str | None = None,
    ) -> schemas.PredictionResponse:
        """Predict a single lookback window using the public response schema."""
        preds = self.predict_arrays(np.asarray(features, dtype=np.float32)[None, ...])
        log_return_quantiles = {
            str(h): [float(x) for x in preds[h][0]]
            for h in self.horizons
            if h in preds
        }
        return schemas.PredictionResponse(
            model_version=self.model_version,
            instrument_id=instrument_id,
            asof_bar_end=asof_bar_end,
            lookback_L=self.config.lookback_length,
            horizons=self.horizons,
            quantiles=self.quantiles,
            log_return_quantiles=log_return_quantiles,
            request_id=request_id,
        )
