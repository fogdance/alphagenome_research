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

"""Loss functions for AlphaTrade."""

from alphagenome import typing
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree


@typing.jaxtyped
def quantile_pinball_loss(
    y_true: Float[Array, 'B'],
    y_pred: Float[Array, 'B Q'],
    quantiles: Float[Array, 'Q'],
    quantile_weights=None,
) -> Float[Array, '']:
  """Quantile pinball loss.

  Args:
    y_true: True values [batch_size]
    y_pred: Predicted quantiles [batch_size, num_quantiles]
    quantiles: Quantile levels [num_quantiles]

  Returns:
    Scalar loss value.
  """
  # Expand y_true to match y_pred shape
  y_true = y_true[:, None]  # [B, 1]
  quantiles = quantiles[None, :]  # [1, Q]

  # Compute residuals
  residuals = y_true - y_pred  # [B, Q]

  # Pinball loss: max(q * residual, (q-1) * residual)
  loss = jnp.where(
      residuals >= 0,
      quantiles * residuals,
      (quantiles - 1) * residuals,
  )

  if quantile_weights is not None:
    weights = quantile_weights / jnp.maximum(jnp.sum(quantile_weights), 1e-12)
    return jnp.sum(jnp.mean(loss, axis=0) * weights)

  return jnp.mean(loss)


@typing.jaxtyped
def quantile_crossing_penalty(
    y_pred: Float[Array, 'B Q'],
) -> Float[Array, '']:
  """Penalty for quantile crossing (non-monotonic quantiles).

  Ensures q_i <= q_{i+1} for all i.

  Args:
    y_pred: Predicted quantiles [batch_size, num_quantiles]

  Returns:
    Scalar penalty value.
  """
  # Compute differences between consecutive quantiles
  diffs = y_pred[:, 1:] - y_pred[:, :-1]  # [B, Q-1]

  # Penalize negative differences (crossings)
  penalties = jnp.maximum(0, -diffs)

  return jnp.mean(penalties)


@typing.jaxtyped
def multi_horizon_quantile_loss(
    targets: dict[int, Float[Array, 'B']],
    predictions: dict[int, Float[Array, 'B Q']],
    quantiles: Float[Array, 'Q'],
    horizon_weights: dict[int, float] | None = None,
    crossing_penalty_weight: float = 1.0,
) -> tuple[Float[Array, ''], PyTree[Float[Array, '']]]:
  """Multi-horizon quantile loss with crossing penalty.

  Args:
    targets: Dictionary mapping horizon -> true values
    predictions: Dictionary mapping horizon -> predicted quantiles
    quantiles: Quantile levels
    horizon_weights: Optional weights for each horizon
    crossing_penalty_weight: Weight for crossing penalty

  Returns:
    Tuple of (total_loss, loss_dict) where loss_dict contains individual losses.
  """
  if horizon_weights is None:
    # Default weights from spec
    horizon_weights = {1: 1.0, 5: 1.0, 20: 0.8, 60: 0.6}

  # Use JAX array for loss accumulation to avoid tracer issues
  total_loss = jnp.zeros((), dtype=jnp.float32)
  loss_dict = {}

  # Compute pinball loss for each horizon
  for horizon in targets.keys():
    if horizon not in predictions:
      continue

    y_true = targets[horizon]
    y_pred = predictions[horizon]
    weight = horizon_weights.get(horizon, 1.0)

    pinball = quantile_pinball_loss(y_true, y_pred, quantiles)
    crossing = quantile_crossing_penalty(y_pred)

    horizon_loss = pinball + crossing_penalty_weight * crossing
    total_loss += weight * horizon_loss

    loss_dict[f'pinball_h{horizon}'] = pinball
    loss_dict[f'crossing_h{horizon}'] = crossing
    loss_dict[f'total_h{horizon}'] = horizon_loss

  loss_dict['total_loss'] = total_loss
  return total_loss, loss_dict


@typing.jaxtyped
def regime_classification_loss(
    y_true: Int[Array, 'B'],
    y_pred: Float[Array, 'B K'],
) -> Float[Array, '']:
  """Cross-entropy loss for regime classification.

  Args:
    y_true: True regime class indices [batch_size]
    y_pred: Predicted logits [batch_size, num_classes]

  Returns:
    Scalar loss value.
  """
  # Compute log softmax
  log_probs = jax.nn.log_softmax(y_pred, axis=-1)

  # Gather log probabilities for true classes
  batch_size = y_true.shape[0]
  batch_indices = jnp.arange(batch_size)
  true_log_probs = log_probs[batch_indices, y_true]

  # Return negative log likelihood
  return -jnp.mean(true_log_probs)


@typing.jaxtyped
def combined_loss(
    targets: dict,
    predictions: dict,
    quantiles: Float[Array, 'Q'],
    config: dict | None = None,
) -> tuple[Float[Array, ''], PyTree[Float[Array, '']]]:
  """Combined loss for all AlphaTrade outputs.

  Args:
    targets: Dictionary containing all target values
    predictions: Dictionary containing all predictions
    quantiles: Quantile levels
    config: Optional configuration for loss weights

  Returns:
    Tuple of (total_loss, loss_dict).
  """
  if config is None:
    config = {
        'horizon_weights': {1: 1.0, 5: 1.0, 20: 0.8, 60: 0.6},
        'crossing_penalty_weight': 1.0,
        'range_weight': 0.5,
        'volume_weight': 0.3,
        'regime_weight': 0.2,
    }

  # Use JAX array for loss accumulation to avoid tracer issues
  total_loss = jnp.zeros((), dtype=jnp.float32)
  loss_dict = {}

  # Main return quantile loss
  return_loss, return_dict = multi_horizon_quantile_loss(
      targets=targets['log_returns'],
      predictions=predictions['log_return_quantiles'],
      quantiles=quantiles,
      horizon_weights=config['horizon_weights'],
      crossing_penalty_weight=config['crossing_penalty_weight'],
  )
  total_loss += return_loss
  loss_dict.update({f'return_{k}': v for k, v in return_dict.items()})

  # Optional: range quantile loss
  if 'log_ranges' in targets and 'log_range_quantiles' in predictions:
    range_loss, range_dict = multi_horizon_quantile_loss(
        targets=targets['log_ranges'],
        predictions=predictions['log_range_quantiles'],
        quantiles=quantiles,
        horizon_weights=config['horizon_weights'],
        crossing_penalty_weight=config['crossing_penalty_weight'],
    )
    total_loss += config['range_weight'] * range_loss
    loss_dict.update({f'range_{k}': v for k, v in range_dict.items()})

  # Optional: volume quantile loss
  if 'log_volumes' in targets and 'log_volume_quantiles' in predictions:
    volume_loss, volume_dict = multi_horizon_quantile_loss(
        targets=targets['log_volumes'],
        predictions=predictions['log_volume_quantiles'],
        quantiles=quantiles,
        horizon_weights=config['horizon_weights'],
        crossing_penalty_weight=config['crossing_penalty_weight'],
    )
    total_loss += config['volume_weight'] * volume_loss
    loss_dict.update({f'volume_{k}': v for k, v in volume_dict.items()})

  # Optional: regime classification loss
  if 'regime_labels' in targets and 'regime_probs' in predictions:
    regime_loss = regime_classification_loss(
        targets['regime_labels'],
        predictions['regime_probs'],
    )
    total_loss += config['regime_weight'] * regime_loss
    loss_dict['regime_loss'] = regime_loss

  loss_dict['combined_total_loss'] = total_loss
  return total_loss, loss_dict
