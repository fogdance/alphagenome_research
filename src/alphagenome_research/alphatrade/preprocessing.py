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

"""Feature preprocessing for AlphaTrade."""

import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Dict, Tuple
import numpy as np


class FeaturePreprocessor:
  """Preprocessor for AlphaTrade input features.

  Handles computation of the 8 required features from OHLCV data:
    0: lr_close - log(C_i / C_{i-1})
    1: hl_range - log(H_i - L_i + epsilon)
    2: oc_return - log(C_i / O_i)
    3: log_vol - log(V_i + 1)
    4: pos_in_range - (C_i - L_i) / (H_i - L_i + epsilon)
    5: time_sin - sin(2π * minute_index / period)
    6: time_cos - cos(2π * minute_index / period)
    7: is_session_open - 0/1 flag
  """

  def __init__(
      self,
      epsilon: float = 1e-9,
      session_period_minutes: int = 1440,
  ):
    """Initializes the preprocessor.

    Args:
      epsilon: Small constant to avoid division by zero
      session_period_minutes: Period for time encoding (default: 1440 = 1 day)
    """
    self.epsilon = epsilon
    self.session_period_minutes = session_period_minutes

  def compute_features(
      self,
      open_prices: Float[Array, 'L'],
      high_prices: Float[Array, 'L'],
      low_prices: Float[Array, 'L'],
      close_prices: Float[Array, 'L'],
      volumes: Float[Array, 'L'],
      minute_indices: Float[Array, 'L'] | None = None,
      is_session_open: Float[Array, 'L'] | None = None,
  ) -> Float[Array, 'L 8']:
    """Computes all 8 features from OHLCV data.

    Args:
      open_prices: Open prices [L]
      high_prices: High prices [L]
      low_prices: Low prices [L]
      close_prices: Close prices [L]
      volumes: Volumes [L]
      minute_indices: Optional minute indices for time encoding [L]
      is_session_open: Optional session open flags [L]

    Returns:
      Feature array [L, 8]
    """
    L = close_prices.shape[0]

    # Feature 0: lr_close - log return of close
    lr_close = jnp.zeros(L)
    lr_close = lr_close.at[1:].set(
        jnp.log(close_prices[1:] / (close_prices[:-1] + self.epsilon))
    )

    # Feature 1: hl_range - log of high-low range
    hl_range = jnp.log(high_prices - low_prices + self.epsilon)

    # Feature 2: oc_return - log return from open to close
    oc_return = jnp.log(close_prices / (open_prices + self.epsilon))

    # Feature 3: log_vol - log of volume
    log_vol = jnp.log(volumes + 1)

    # Feature 4: pos_in_range - position within the bar's range
    pos_in_range = (close_prices - low_prices) / (
        high_prices - low_prices + self.epsilon
    )

    # Feature 5 & 6: time_sin and time_cos
    if minute_indices is None:
      minute_indices = jnp.arange(L, dtype=jnp.float32)

    time_angle = 2 * jnp.pi * minute_indices / self.session_period_minutes
    time_sin = jnp.sin(time_angle)
    time_cos = jnp.cos(time_angle)

    # Feature 7: is_session_open
    if is_session_open is None:
      is_session_open = jnp.ones(L)

    # Stack all features
    features = jnp.stack(
        [
            lr_close,
            hl_range,
            oc_return,
            log_vol,
            pos_in_range,
            time_sin,
            time_cos,
            is_session_open,
        ],
        axis=-1,
    )

    return features

  def normalize_features(
      self,
      features: Float[Array, 'L 8'],
      scaler_params: Dict[str, Tuple[float, float]] | None = None,
  ) -> Tuple[Float[Array, 'L 8'], Dict[str, Tuple[float, float]]]:
    """Normalizes features using robust scaling (median/IQR).

    Args:
      features: Input features [L, 8]
      scaler_params: Optional pre-computed scaler parameters.
        If None, computes from the input features.

    Returns:
      Tuple of (normalized_features, scaler_params)
    """
    if scaler_params is None:
      # Compute scaler parameters from data
      scaler_params = {}
      for i in range(8):
        median = float(jnp.median(features[:, i]))
        q75 = float(jnp.percentile(features[:, i], 75))
        q25 = float(jnp.percentile(features[:, i], 25))
        iqr = q75 - q25
        if iqr < 1e-6:
          iqr = 1.0  # Avoid division by zero
        scaler_params[f'feature_{i}'] = (median, iqr)

    # Apply normalization
    normalized = jnp.zeros_like(features)
    for i in range(8):
      median, iqr = scaler_params[f'feature_{i}']
      normalized = normalized.at[:, i].set(
          (features[:, i] - median) / iqr
      )

    return normalized, scaler_params


class RobustScaler:
  """Robust scaler using median and IQR for normalization."""

  def __init__(self):
    self.medians_ = None
    self.iqrs_ = None
    self.fitted_ = False

  def fit(self, features: np.ndarray) -> 'RobustScaler':
    """Fits the scaler on training data.

    Args:
      features: Training features [N, L, 8] or [L, 8]

    Returns:
      Self for chaining.
    """
    if features.ndim == 3:
      # Flatten batch dimension
      features = features.reshape(-1, features.shape[-1])

    self.medians_ = np.median(features, axis=0)
    q75 = np.percentile(features, 75, axis=0)
    q25 = np.percentile(features, 25, axis=0)
    self.iqrs_ = q75 - q25
    # Avoid division by zero
    self.iqrs_ = np.where(self.iqrs_ < 1e-6, 1.0, self.iqrs_)
    self.fitted_ = True

    return self

  def transform(self, features: np.ndarray) -> np.ndarray:
    """Transforms features using fitted parameters.

    Args:
      features: Features to transform [N, L, 8] or [L, 8]

    Returns:
      Normalized features with same shape as input.
    """
    if not self.fitted_:
      raise ValueError('Scaler must be fitted before transform.')

    return (features - self.medians_) / self.iqrs_

  def fit_transform(self, features: np.ndarray) -> np.ndarray:
    """Fits and transforms in one step.

    Args:
      features: Features to fit and transform

    Returns:
      Normalized features.
    """
    return self.fit(features).transform(features)

  def inverse_transform(self, features: np.ndarray) -> np.ndarray:
    """Inverse transforms normalized features back to original scale.

    Args:
      features: Normalized features

    Returns:
      Features in original scale.
    """
    if not self.fitted_:
      raise ValueError('Scaler must be fitted before inverse_transform.')

    return features * self.iqrs_ + self.medians_

  def get_params(self) -> Dict[str, Tuple[float, float]]:
    """Gets scaler parameters for serialization.

    Returns:
      Dictionary mapping feature index to (median, iqr).
    """
    if not self.fitted_:
      raise ValueError('Scaler must be fitted before getting params.')

    params = {}
    for i in range(len(self.medians_)):
      params[f'feature_{i}'] = (
          float(self.medians_[i]),
          float(self.iqrs_[i]),
      )
    return params

  def set_params(self, params: Dict[str, Tuple[float, float]]) -> 'RobustScaler':
    """Sets scaler parameters from dictionary.

    Args:
      params: Dictionary mapping feature index to (median, iqr)

    Returns:
      Self for chaining.
    """
    num_features = len(params)
    self.medians_ = np.zeros(num_features)
    self.iqrs_ = np.zeros(num_features)

    for i in range(num_features):
      median, iqr = params[f'feature_{i}']
      self.medians_[i] = median
      self.iqrs_[i] = iqr

    self.fitted_ = True
    return self


def validate_features(features: Float[Array, 'L 8'], normalized: bool = False) -> bool:
  """Validates that features are well-formed.

  Args:
    features: Feature array to validate
    normalized: If True, skip range checks for normalized features

  Returns:
    True if valid, raises ValueError otherwise.
  """
  if features.shape[-1] != 8:
    raise ValueError(f'Expected 8 features, got {features.shape[-1]}')

  # Check for NaN or Inf
  if jnp.any(jnp.isnan(features)):
    raise ValueError('Features contain NaN values')

  if jnp.any(jnp.isinf(features)):
    raise ValueError('Features contain Inf values')

  # Only check ranges for non-normalized features
  if not normalized:
    # Check pos_in_range is in [0, 1]
    pos_in_range = features[:, 4]
    if jnp.any((pos_in_range < -0.1) | (pos_in_range > 1.1)):
      raise ValueError('pos_in_range feature out of expected range [0, 1]')

    # Check is_session_open is binary
    is_session_open = features[:, 7]
    if not jnp.all((is_session_open == 0) | (is_session_open == 1)):
      raise ValueError('is_session_open must be binary (0 or 1)')

  return True
