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
from typing import Any, Dict, List, Optional
import json

from alphagenome_research.alphatrade import model as model_lib
from alphagenome_research.alphatrade import preprocessing
from alphagenome_research.alphatrade import schemas
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
      model_version: str = 'alphatrade_v0.2',
  ):
    """Initializes the service.

    Args:
      config: Model configuration
      train_state: Trained model state (params, state, forward function)
      scaler: Optional fitted scaler for feature normalization
      model_version: Version string for the model
    """
    self.config = config
    self.train_state = train_state
    self.scaler = scaler
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
        # Convert to list of floats
        quantile_values = predictions.log_return_quantiles[horizon][0]
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
    checkpoint_path: Path to saved model checkpoint
    config: Optional model configuration (if not saved in checkpoint)
    scaler_path: Optional path to saved scaler parameters

  Returns:
    AlphaTradeService instance ready for inference.
  """
  # TODO: Implement checkpoint loading using orbax or similar
  # This is a placeholder for the actual implementation
  raise NotImplementedError(
      'Checkpoint loading not yet implemented. '
      'Use create_service_from_train_state for now.'
  )


def create_service_from_train_state(
    train_state: Dict[str, Any],
    config: schemas.AlphaTradeConfig,
    scaler: preprocessing.RobustScaler | None = None,
    model_version: str = 'alphatrade_v0.2',
) -> AlphaTradeService:
  """Creates a service from an in-memory training state.

  Args:
    train_state: Training state with params, state, and forward function
    config: Model configuration
    scaler: Optional fitted scaler
    model_version: Version string

  Returns:
    AlphaTradeService instance.
  """
  return AlphaTradeService(
      config=config,
      train_state=train_state,
      scaler=scaler,
      model_version=model_version,
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
