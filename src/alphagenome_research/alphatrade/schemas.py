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

"""Data schemas for AlphaTrade."""

import dataclasses
from jaxtyping import Array, Float, Int
from typing import Dict, List


@dataclasses.dataclass
class AlphaTradeConfig:
  """Configuration for AlphaTrade model."""

  # Input configuration
  lookback_length: int = 4096  # L: number of 1-minute bars
  num_features: int = 8  # F: fixed feature count

  # Architecture configuration
  stem_channels: int = 128  # C0
  num_encoder_stages: int = 6  # Downsample to /64 (2^6)
  channel_increment: int = 64  # Channel growth per stage

  # Transformer configuration
  d_model: int = 512
  num_transformer_layers: int = 6
  num_heads: int = 8
  head_dim: int = 64
  mlp_expansion: int = 4
  logits_soft_cap: float = 5.0
  max_position: int = 8192

  # Output configuration
  horizons: List[int] = dataclasses.field(
      default_factory=lambda: [1, 5, 20, 60]
  )
  quantiles: List[float] = dataclasses.field(
      default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9]
  )

  # Optional heads
  predict_range: bool = False
  predict_volume: bool = False
  predict_regime: bool = False
  num_regime_classes: int = 3  # {up, down, range}

  def __post_init__(self):
    """Validate configuration after initialization."""
    # Validate quantiles
    if len(self.quantiles) == 0:
      raise ValueError("quantiles cannot be empty")

    # Check strictly increasing
    for i in range(len(self.quantiles) - 1):
      if self.quantiles[i] >= self.quantiles[i + 1]:
        raise ValueError(
            f"quantiles must be strictly increasing, got {self.quantiles}"
        )

    # Check all values in (0, 1)
    for q in self.quantiles:
      if not (0 < q < 1):
        raise ValueError(f"quantiles must be in (0, 1), got {q}")

    # Check contains 0.5 (required by QuantileHead implementation)
    if 0.5 not in self.quantiles:
      raise ValueError(
          "quantiles must contain 0.5 (median) for QuantileHead to work"
      )

    # Check odd number (required by QuantileHead median+deltas approach)
    if len(self.quantiles) % 2 == 0:
      raise ValueError(
          f"quantiles must have odd length for QuantileHead, "
          f"got {len(self.quantiles)}"
      )

    # Check 0.5 is in the middle
    median_idx = len(self.quantiles) // 2
    if self.quantiles[median_idx] != 0.5:
      raise ValueError(
          f"quantiles[{median_idx}] must be 0.5 (median must be centered), "
          f"got {self.quantiles[median_idx]}"
      )

    # Validate horizons
    if len(self.horizons) == 0:
      raise ValueError("horizons cannot be empty")

    for h in self.horizons:
      if h <= 0:
        raise ValueError(f"horizons must be positive, got {h}")


@dataclasses.dataclass
class AlphaTradeInput:
  """Input data for AlphaTrade model.

  Features are ordered as:
    0: lr_close - log(C_i / C_{i-1})
    1: hl_range - log(H_i - L_i + epsilon)
    2: oc_return - log(C_i / O_i)
    3: log_vol - log(V_i + 1)
    4: pos_in_range - (C_i - L_i) / (H_i - L_i + epsilon)
    5: time_sin - sin(2π * minute_index / period)
    6: time_cos - cos(2π * minute_index / period)
    7: is_session_open - 0/1 flag
  """

  features: Float[Array, 'B L 8']  # [batch, lookback_length, num_features]


@dataclasses.dataclass
class AlphaTradeTargets:
  """Target labels for training AlphaTrade."""

  # Required: log returns for each horizon
  log_returns: Dict[int, Float[Array, 'B']]  # horizon -> log(C_{t+h}/C_t)

  # Optional targets
  log_ranges: Dict[int, Float[Array, 'B']] | None = None  # horizon -> log(range)
  log_volumes: Dict[int, Float[Array, 'B']] | None = None  # horizon -> log(volume)
  regime_labels: Int[Array, 'B'] | None = None  # regime class indices


@dataclasses.dataclass
class AlphaTradeOutput:
  """Output predictions from AlphaTrade model."""

  # Required: quantile predictions for log returns
  # Keys are horizon values (e.g., 1, 5, 20, 60)
  # Values are arrays of shape [B, Q] where Q is number of quantiles
  log_return_quantiles: Dict[int, Float[Array, 'B Q']]

  # Optional outputs
  log_range_quantiles: Dict[int, Float[Array, 'B Q']] | None = None
  log_volume_quantiles: Dict[int, Float[Array, 'B Q']] | None = None
  regime_probs: Float[Array, 'B K'] | None = None  # K = num_regime_classes

  # Embeddings for analysis
  trunk_embedding: Float[Array, 'B D'] | None = None  # Final readout embedding


@dataclasses.dataclass
class TrainingBatch:
  """A batch of training data."""

  inputs: AlphaTradeInput
  targets: AlphaTradeTargets

  # Optional metadata
  instrument_ids: List[str] | None = None
  timestamps: List[str] | None = None  # asof_bar_end timestamps


@dataclasses.dataclass
class PredictionRequest:
  """API request for prediction."""

  instrument_id: str
  asof_bar_end: str  # RFC3339 timestamp
  features: Float[Array, 'L 8']  # [lookback_length, num_features]

  # Optional overrides
  lookback_L: int | None = None
  horizons: List[int] | None = None
  quantiles: List[float] | None = None
  request_id: str | None = None
  debug: bool = False


@dataclasses.dataclass
class PredictionResponse:
  """API response for prediction."""

  model_version: str
  instrument_id: str
  asof_bar_end: str
  lookback_L: int
  horizons: List[int]
  quantiles: List[float]

  # Predictions: horizon (as string) -> list of quantile values
  log_return_quantiles: Dict[str, List[float]]

  # Optional fields
  latency_ms: float | None = None
  request_id: str | None = None
  quality: Dict[str, float] | None = None
