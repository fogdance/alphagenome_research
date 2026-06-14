# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""API service for AlphaTrade predictions."""

import time
import pickle
from typing import Any, Dict, List, Optional
import json

from alphatrade.core import model as model_lib
from alphatrade.core import preprocessing
from alphatrade.core import schemas
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class AlphaTradeService:
  """Prediction service for AlphaTrade model."""

  def __init__(
      self,
      config: schemas.AlphaTradeConfig,
      train_state: Dict[str, Any],
      scaler: preprocessing.RobustScaler | None = None,
      target_stats: Dict[int, tuple[float, float]] | None = None,
      model_version: str = 'alphatrade_v0.2',
  ):
    """Initializes the service.

    Args:
      config: Model configuration
      train_state: Trained model state (params, state, forward function)
      scaler: Optional fitted scaler for feature normalization
      target_stats: Optional target normalization stats {horizon: (mean, std)}
                    for denormalizing predictions
      model_version: Version string for the model
    """
    self.config = config
    self.train_state = train_state
    self.scaler = scaler
    self.target_stats = target_stats
    self.model_version = model_version
    self.preprocessor = preprocessing.FeaturePreprocessor()

    # JIT compile prediction function
    self._predict_fn = jax.jit(self._predict_internal)

  def _predict_internal(
      self, features: jax.Array
  ) -> schemas.AlphaTradeOutput:
    """Internal prediction function (JIT-compiled).

    Args:
      features: Input features [1, L, 8]

    Returns:
      AlphaTradeOutput with predictions.
    """
    predictions, _ = self.train_state['forward'].apply(
        self.train_state['params'],
        self.train_state['state'],
        jax.random.PRNGKey(0),  # Deterministic inference
        features,
    )
    return predictions

  def predict(
      self, request: schemas.PredictionRequest
  ) -> schemas.PredictionResponse:
    """Makes a prediction for the given request.

    Args:
      request: Prediction request with features and metadata

    Returns:
      Prediction response with quantile predictions.
    """
    start_time = time.time()

    # Validate and prepare features
    features = np.array(request.features)
    if features.ndim == 2:
      features = features[np.newaxis, ...]  # Add batch dimension

    # Validate feature shape
    lookback_L = request.lookback_L or self.config.lookback_length
    if features.shape[1] != lookback_L:
      raise ValueError(
          f'Expected lookback_L={lookback_L}, got {features.shape[1]}'
      )
    if features.shape[2] != 8:
      raise ValueError(f'Expected 8 features, got {features.shape[2]}')

    # Validate raw features before normalization
    preprocessing.validate_features(jnp.array(features[0]), normalized=False)

    # Normalize features if scaler is available
    if self.scaler is not None:
      features = self.scaler.transform(features)

    # Validate normalized features (skip range checks)
    preprocessing.validate_features(jnp.array(features[0]), normalized=True)

    # Convert to JAX array
    features_jax = jnp.array(features)

    # Make prediction
    predictions = self._predict_fn(features_jax)

    # Extract quantiles for requested horizons
    horizons = request.horizons or self.config.horizons
    quantiles = request.quantiles or self.config.quantiles

    # Validate that requested horizons/quantiles match model config
    if request.horizons is not None:
      unsupported_horizons = set(request.horizons) - set(self.config.horizons)
      if unsupported_horizons:
        raise ValueError(
            f'Unsupported horizons: {unsupported_horizons}. '
            f'Model only supports: {self.config.horizons}'
        )

    if request.quantiles is not None:
      if request.quantiles != self.config.quantiles:
        raise ValueError(
            f'Model does not support dynamic quantiles. '
            f'Requested: {request.quantiles}, '
            f'Model config: {self.config.quantiles}'
        )

    log_return_quantiles = {}
    for horizon in horizons:
      if horizon in predictions.log_return_quantiles:
        # Convert to numpy array
        quantile_values = predictions.log_return_quantiles[horizon][0]  # [Q]

        # Denormalize if target_stats available
        if self.target_stats is not None and horizon in self.target_stats:
          mean, std = self.target_stats[horizon]
          quantile_values = quantile_values * std + mean

        # Convert to list of floats
        log_return_quantiles[str(horizon)] = [
            float(q) for q in quantile_values
        ]

    # Compute latency
    latency_ms = (time.time() - start_time) * 1000

    return schemas.PredictionResponse(
        model_version=self.model_version,
        instrument_id=request.instrument_id,
        asof_bar_end=request.asof_bar_end,
        lookback_L=lookback_L,
        horizons=horizons,
        quantiles=quantiles,
        log_return_quantiles=log_return_quantiles,
        latency_ms=latency_ms,
        request_id=request.request_id,
    )

  def health_check(self) -> Dict[str, Any]:
    """Returns service health status.

    Returns:
      Dictionary with health status and model metadata.
    """
    return {
        'status': 'healthy',
        'model_version': self.model_version,
        'config': {
            'lookback_length': self.config.lookback_length,
            'horizons': self.config.horizons,
            'quantiles': self.config.quantiles,
            'd_model': self.config.d_model,
            'num_transformer_layers': self.config.num_transformer_layers,
        },
        'scaler_fitted': self.scaler is not None and self.scaler.fitted_,
    }


class AlphaTradeHTTPHandler:
  """HTTP request handler for AlphaTrade API."""

  def __init__(self, service: AlphaTradeService):
    """Initializes the handler.

    Args:
      service: AlphaTrade service instance
    """
    self.service = service

  def handle_predict(self, request_json: Dict[str, Any]) -> Dict[str, Any]:
    """Handles /v0.1/predict endpoint.

    Args:
      request_json: JSON request body

    Returns:
      JSON response body.
    """
    try:
      # Parse request
      instrument_id = request_json['instrument_id']
      asof_bar_end = request_json['asof_bar_end']
      features = np.array(request_json['inputs']['features'])

      # Optional fields
      lookback_L = request_json.get('lookback_L')
      horizons = request_json.get('horizons')
      quantiles = request_json.get('quantiles')
      request_id = request_json.get('request_id')
      debug = request_json.get('debug', False)

      # Create request object
      request = schemas.PredictionRequest(
          instrument_id=instrument_id,
          asof_bar_end=asof_bar_end,
          features=features,
          lookback_L=lookback_L,
          horizons=horizons,
          quantiles=quantiles,
          request_id=request_id,
          debug=debug,
      )

      # Make prediction
      response = self.service.predict(request)

      # Convert to JSON-serializable dict
      response_dict = {
          'model_version': response.model_version,
          'instrument_id': response.instrument_id,
          'asof_bar_end': response.asof_bar_end,
          'lookback_L': response.lookback_L,
          'horizons': response.horizons,
          'quantiles': response.quantiles,
          'pred': {
              'log_return_quantiles': response.log_return_quantiles,
          },
          'latency_ms': response.latency_ms,
      }

      if response.request_id:
        response_dict['request_id'] = response.request_id

      return response_dict

    except KeyError as e:
      return {
          'code': 'INVALID_ARGUMENT',
          'message': f'Missing required field: {e}',
          'request_id': request_json.get('request_id'),
      }
    except ValueError as e:
      return {
          'code': 'INVALID_ARGUMENT',
          'message': str(e),
          'request_id': request_json.get('request_id'),
      }
    except Exception as e:
      return {
          'code': 'INTERNAL_ERROR',
          'message': f'Internal error: {str(e)}',
          'request_id': request_json.get('request_id'),
      }

  def handle_health(self) -> Dict[str, Any]:
    """Handles /v0.1/health endpoint.

    Returns:
      JSON response with health status.
    """
    try:
      return self.service.health_check()
    except Exception as e:
      return {
          'status': 'unhealthy',
          'error': str(e),
      }


def create_service_from_checkpoint(
    checkpoint_path: str,
    config: schemas.AlphaTradeConfig | None = None,
    scaler_path: str | None = None,
) -> AlphaTradeService:
  """Creates a service from a saved checkpoint.

  Args:
    checkpoint_path: Path to saved model checkpoint (pickle format)
    config: Optional model configuration (if not saved in checkpoint)
    scaler_path: Optional path to saved scaler parameters (JSON format)

  Returns:
    AlphaTradeService instance ready for inference.

  Note:
    Pickle is unsafe with untrusted inputs. Only load checkpoints you created.
  """
  # Load checkpoint
  with open(checkpoint_path, "rb") as f:
    ckpt = pickle.load(f)

  if not isinstance(ckpt, dict):
    raise ValueError(f"Checkpoint must be a dict, got {type(ckpt)}")

  # ---- Config ----
  if config is None:
    if "config" not in ckpt:
      raise ValueError(
          "Checkpoint missing 'config' and no config override provided."
      )
    cfg_obj = ckpt["config"]
    if isinstance(cfg_obj, schemas.AlphaTradeConfig):
      config = cfg_obj
    elif isinstance(cfg_obj, dict):
      config = schemas.AlphaTradeConfig(**cfg_obj)
    else:
      raise ValueError(f"Unsupported checkpoint config type: {type(cfg_obj)}")

  # ---- Forward transform ----
  def _forward_fn(features: jax.Array) -> schemas.AlphaTradeOutput:
    model = model_lib.AlphaTrade(config)
    return model(features)

  forward = hk.transform_with_state(lambda x: _forward_fn(x))

  # ---- Params/state ----
  if "params" not in ckpt or "state" not in ckpt:
    raise ValueError(
        "Checkpoint missing required keys: 'params' and/or 'state'."
    )

  # Convert to JAX arrays (handle numpy/list/device arrays)
  params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
  state = jax.tree_util.tree_map(jnp.asarray, ckpt["state"])

  # Validate params structure matches expected forward transform
  # This catches mismatches early with a clear error message
  try:
    dummy_input = jnp.zeros((1, config.lookback_length, config.num_features))
    expected_params, _ = forward.init(jax.random.PRNGKey(0), dummy_input)

    # Check top-level keys match
    expected_keys = set(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda _: None, expected_params, is_leaf=lambda x: isinstance(x, dict))
    ))
    actual_keys = set(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda _: None, params, is_leaf=lambda x: isinstance(x, dict))
    ))

    # Simple structure check: compare flattened key paths
    expected_flat = jax.tree_util.tree_flatten(expected_params)[0]
    actual_flat = jax.tree_util.tree_flatten(params)[0]

    if len(expected_flat) != len(actual_flat):
      raise ValueError(
          f"Checkpoint params structure mismatch: "
          f"expected {len(expected_flat)} parameters, got {len(actual_flat)}. "
          f"This usually means the checkpoint was saved with a different model architecture."
      )
  except Exception as e:
    if "structure mismatch" in str(e).lower():
      raise
    # If validation fails for other reasons, warn but continue
    import warnings
    warnings.warn(
        f"Could not validate checkpoint params structure: {e}. "
        f"Proceeding anyway, but inference may fail."
    )

  train_state = {
      "params": params,
      "state": state,
      "forward": forward,
  }

  # ---- Scaler ----
  scaler: preprocessing.RobustScaler | None = None

  def _build_scaler_from_medians_iqrs(medians, iqrs) -> preprocessing.RobustScaler:
    s = preprocessing.RobustScaler()
    s.medians_ = np.asarray(medians, dtype=np.float32)
    s.iqrs_ = np.asarray(iqrs, dtype=np.float32)
    # Avoid division by near-zero
    s.iqrs_ = np.where(s.iqrs_ < 1e-6, 1.0, s.iqrs_)
    s.fitted_ = True
    return s

  def _parse_feature_dict(sp: dict) -> tuple[list, list]:
    """Parse feature_i dict format into medians and iqrs lists.

    Handles non-contiguous indices and validates structure.
    """
    # Extract and sort feature indices
    feature_keys = [k for k in sp.keys() if k.startswith("feature_")]
    if not feature_keys:
      raise ValueError("No feature_* keys found in scaler dict")

    # Parse indices and sort
    try:
      indices = sorted([int(k.split("_")[1]) for k in feature_keys])
    except (ValueError, IndexError) as e:
      raise ValueError(f"Invalid feature key format: {e}")

    # Check for gaps
    expected_indices = list(range(len(indices)))
    if indices != expected_indices:
      raise ValueError(
          f"Feature indices must be contiguous starting from 0. "
          f"Expected {expected_indices}, got {indices}"
      )

    # Extract values in order
    med = []
    iqr = []
    for i in indices:
      key = f"feature_{i}"
      val = sp[key]
      if not isinstance(val, (list, tuple)) or len(val) != 2:
        raise ValueError(
            f"Expected {key} to be [median, iqr], got {val}"
        )
      med.append(val[0])
      iqr.append(val[1])

    return med, iqr

  if scaler_path is not None:
    with open(scaler_path, "r", encoding="utf-8") as f:
      sp = json.load(f)
    # Support either {medians, iqrs} or {"feature_0": [median, iqr], ...}
    if "medians" in sp and "iqrs" in sp:
      scaler = _build_scaler_from_medians_iqrs(sp["medians"], sp["iqrs"])
    elif any(k.startswith("feature_") for k in sp.keys()):
      med, iqr = _parse_feature_dict(sp)
      scaler = _build_scaler_from_medians_iqrs(med, iqr)
    else:
      raise ValueError(f"Unrecognized scaler file format: {scaler_path}")
  else:
    sc = ckpt.get("scaler", None)
    if isinstance(sc, dict):
      if "medians" in sc and "iqrs" in sc:
        scaler = _build_scaler_from_medians_iqrs(sc["medians"], sc["iqrs"])
      elif any(k.startswith("feature_") for k in sc.keys()):
        med, iqr = _parse_feature_dict(sc)
        scaler = _build_scaler_from_medians_iqrs(med, iqr)

  # ---- Target stats (for denormalization) ----
  target_stats = ckpt.get("target_stats", None)
  if target_stats is not None:
    # Convert keys to int if they're strings
    if isinstance(target_stats, dict):
      target_stats = {
          int(k) if isinstance(k, str) else k: v
          for k, v in target_stats.items()
      }

  return create_service_from_train_state(
      train_state=train_state,
      config=config,
      scaler=scaler,
      target_stats=target_stats,
      model_version=ckpt.get("model_version", "alphatrade_v0.2"),
  )


def create_service_from_train_state(
    train_state: Dict[str, Any],
    config: schemas.AlphaTradeConfig,
    scaler: preprocessing.RobustScaler | None = None,
    target_stats: Dict[int, tuple[float, float]] | None = None,
    model_version: str = 'alphatrade_v0.2',
) -> AlphaTradeService:
  """Creates a service from an in-memory training state.

  Args:
    train_state: Training state with params, state, and forward function
    config: Model configuration
    scaler: Optional fitted scaler
    target_stats: Optional target normalization stats {horizon: (mean, std)}
    model_version: Version string

  Returns:
    AlphaTradeService instance.
  """
  return AlphaTradeService(
      config=config,
      train_state=train_state,
      scaler=scaler,
      target_stats=target_stats,
      model_version=model_version,
  )


def create_service_from_bundle(bundle_dir: str) -> AlphaTradeService:
  """Creates a prediction service from an M9 model bundle.

  Args:
    bundle_dir: Path to a directory containing bundle_manifest.json,
      model_config.json, and best/ Flax checkpoint files.

  Returns:
    AlphaTradeService instance ready for single-window API inference.
  """
  from alphatrade import inference

  predictor = inference.AlphaTradeBundlePredictor(bundle_dir)
  train_state = {
      "params": predictor.params,
      "state": predictor.state,
      "forward": predictor.forward,
  }
  return create_service_from_train_state(
      train_state=train_state,
      config=predictor.config,
      model_version=predictor.model_version,
  )


# Example usage functions
def example_request() -> Dict[str, Any]:
  """Returns an example prediction request.

  Returns:
    Example request dictionary.
  """
  # Generate dummy features
  L = 4096
  features = np.random.randn(L, 8).tolist()

  return {
      'instrument_id': 'IF2403',
      'asof_bar_end': '2026-02-13T10:07:00+08:00',
      'lookback_L': L,
      'horizons': [1, 5, 20, 60],
      'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
      'request_id': 'req-example-001',
      'inputs': {
          'features': features,
      },
  }


def example_response() -> Dict[str, Any]:
  """Returns an example prediction response.

  Returns:
    Example response dictionary.
  """
  return {
      'model_version': 'alphatrade_v0.2',
      'instrument_id': 'IF2403',
      'asof_bar_end': '2026-02-13T10:07:00+08:00',
      'lookback_L': 4096,
      'horizons': [1, 5, 20, 60],
      'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
      'pred': {
          'log_return_quantiles': {
              '1': [-0.0008, -0.0003, 0.0001, 0.0004, 0.0009],
              '5': [-0.0019, -0.0008, 0.0002, 0.0010, 0.0021],
              '20': [-0.0042, -0.0017, 0.0005, 0.0022, 0.0046],
              '60': [-0.0089, -0.0038, 0.0011, 0.0042, 0.0094],
          }
      },
      'latency_ms': 12.7,
      'request_id': 'req-example-001',
  }
