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

"""Training utilities for AlphaTrade."""

import functools
from typing import Any, Callable, Dict, Tuple

from alphagenome_research.alphatrade import model as model_lib
from alphagenome_research.alphatrade import schemas
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree


def create_train_state(
    config: schemas.AlphaTradeConfig,
    rng: jax.Array,
    learning_rate: float = 1e-4,
    sample_batch: schemas.TrainingBatch | None = None,
) -> Dict[str, Any]:
  """Creates initial training state.

  Args:
    config: Model configuration
    rng: Random number generator
    learning_rate: Learning rate for optimizer
    sample_batch: Optional sample batch for initialization

  Returns:
    Dictionary containing params, state, and optimizer state.
  """
  # Create model function
  def forward_fn(features: Float[Array, 'B L 8']) -> schemas.AlphaTradeOutput:
    model = model_lib.AlphaTrade(config)
    return model(features)

  # Transform to pure function
  forward = hk.transform_with_state(lambda x: forward_fn(x))

  # Initialize parameters
  if sample_batch is None:
    # Create dummy batch for initialization
    dummy_features = jnp.zeros(
        (1, config.lookback_length, config.num_features)
    )
  else:
    dummy_features = sample_batch.inputs.features

  params, state = forward.init(rng, dummy_features)

  # Create optimizer
  optimizer = optax.adam(learning_rate)
  opt_state = optimizer.init(params)

  return {
      'params': params,
      'state': state,
      'opt_state': opt_state,
      'optimizer': optimizer,
      'forward': forward,
  }


def create_loss_fn(
    config: schemas.AlphaTradeConfig,
) -> Callable:
  """Creates loss function for training.

  Args:
    config: Model configuration

  Returns:
    Loss function that takes (params, state, rng, batch) and returns
    (loss, (state, metrics)).
  """
  def loss_fn(
      params: PyTree,
      state: PyTree,
      rng: jax.Array,
      batch: schemas.TrainingBatch,
  ) -> Tuple[Float[Array, ''], Tuple[PyTree, Dict[str, Float[Array, '']]]]:
    """Computes loss for a batch.

    Args:
      params: Model parameters
      state: Model state
      rng: Random number generator
      batch: Training batch

    Returns:
      Tuple of (loss, (new_state, metrics)).
    """
    def model_fn(features):
      model = model_lib.AlphaTrade(config)
      return model.loss(
          schemas.TrainingBatch(
              inputs=schemas.AlphaTradeInput(features=features),
              targets=batch.targets,
          )
      )

    forward = hk.transform_with_state(lambda x: model_fn(x))
    (loss, metrics), new_state = forward.apply(
        params, state, rng, batch.inputs.features
    )

    return loss, (new_state, metrics)

  return loss_fn


def train_step(
    train_state: Dict[str, Any],
    batch: schemas.TrainingBatch,
    rng: jax.Array,
    config: schemas.AlphaTradeConfig,
) -> Tuple[Dict[str, Any], Dict[str, Float[Array, '']]]:
  """Performs a single training step.

  Args:
    train_state: Current training state
    batch: Training batch
    rng: Random number generator
    config: Model configuration

  Returns:
    Tuple of (updated_train_state, metrics).
  """
  loss_fn = create_loss_fn(config)

  # Compute gradients
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_state, metrics)), grads = grad_fn(
      train_state['params'],
      train_state['state'],
      rng,
      batch,
  )

  # Update parameters
  updates, new_opt_state = train_state['optimizer'].update(
      grads, train_state['opt_state'], train_state['params']
  )
  new_params = optax.apply_updates(train_state['params'], updates)

  # Update training state
  new_train_state = {
      **train_state,
      'params': new_params,
      'state': new_state,
      'opt_state': new_opt_state,
  }

  # Add gradient norm to metrics
  grad_norm = optax.global_norm(grads)
  metrics['grad_norm'] = grad_norm
  metrics['loss'] = loss

  return new_train_state, metrics


def eval_step(
    train_state: Dict[str, Any],
    batch: schemas.TrainingBatch,
    rng: jax.Array,
    config: schemas.AlphaTradeConfig,
) -> Dict[str, Float[Array, '']]:
  """Performs a single evaluation step.

  Args:
    train_state: Current training state
    batch: Evaluation batch
    rng: Random number generator
    config: Model configuration

  Returns:
    Dictionary of evaluation metrics.
  """
  loss_fn = create_loss_fn(config)

  loss, (_, metrics) = loss_fn(
      train_state['params'],
      train_state['state'],
      rng,
      batch,
  )

  metrics['loss'] = loss
  return metrics


def predict(
    train_state: Dict[str, Any],
    features: Float[Array, 'B L 8'],
    rng: jax.Array,
) -> schemas.AlphaTradeOutput:
  """Makes predictions on input features.

  Args:
    train_state: Training state containing params and state
    features: Input features [batch, lookback_length, 8]
    rng: Random number generator

  Returns:
    AlphaTradeOutput with predictions.
  """
  predictions, _ = train_state['forward'].apply(
      train_state['params'],
      train_state['state'],
      rng,
      features,
  )
  return predictions


# JIT-compiled versions for performance
train_step_jit = jax.jit(train_step, static_argnums=(3,))
eval_step_jit = jax.jit(eval_step, static_argnums=(3,))
predict_jit = jax.jit(predict, static_argnums=())


def compute_quantile_coverage(
    y_true: Float[Array, 'N'],
    y_pred_quantiles: Float[Array, 'N Q'],
    quantile_levels: list[float],
) -> Dict[str, float]:
  """Computes empirical coverage for predicted quantiles.

  Args:
    y_true: True values [N]
    y_pred_quantiles: Predicted quantiles [N, Q]
    quantile_levels: List of quantile levels (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])

  Returns:
    Dictionary mapping quantile level to empirical coverage.
  """
  coverage = {}
  for i, q in enumerate(quantile_levels):
    # Count how many true values are below this quantile
    below = jnp.mean(y_true <= y_pred_quantiles[:, i])
    coverage[f'coverage_q{int(q*100)}'] = float(below)

  return coverage


def compute_calibration_error(
    y_true: Float[Array, 'N'],
    y_pred_quantiles: Float[Array, 'N Q'],
    quantile_levels: list[float],
) -> float:
  """Computes expected calibration error (ECE) for quantiles.

  Args:
    y_true: True values [N]
    y_pred_quantiles: Predicted quantiles [N, Q]
    quantile_levels: List of quantile levels

  Returns:
    Mean absolute calibration error.
  """
  errors = []
  for i, q in enumerate(quantile_levels):
    empirical_coverage = jnp.mean(y_true <= y_pred_quantiles[:, i])
    error = jnp.abs(empirical_coverage - q)
    errors.append(error)

  return float(jnp.mean(jnp.array(errors)))
