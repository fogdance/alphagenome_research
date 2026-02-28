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

"""AlphaTrade model architecture."""

from alphagenome import typing
from alphatrade.core import causal_attention
from alphatrade.core import causal_layers
from alphatrade.core import schemas
from alphagenome_research.model import layers
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


class TemporalEncoder(hk.Module):
  """Encodes temporal features with causal downsampling."""

  def __init__(
      self,
      num_stages: int = 6,
      channel_increment: int = 64,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._num_stages = num_stages
    self._channel_increment = channel_increment

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D']
  ) -> tuple[Float[Array, 'B S//64 D_out'], dict[str, Array]]:
    """Encodes temporal features with multi-scale downsampling.

    Args:
      x: Input features [batch, sequence, channels]

    Returns:
      Tuple of (encoded_features, intermediates) where intermediates
      contains skip connections for each scale.

    Note:
      With num_stages=6, downsampling is 2^6 = 64x (not 128x).
    """
    intermediates = {}
    intermediates['scale_1'] = x

    # Downsample through multiple stages
    for stage_idx in range(self._num_stages):
      scale = 2 ** (stage_idx + 1)

      # Pool first (causal)
      x = causal_layers.causal_pool(x, by=2)

      # Apply residual block
      x = causal_layers.CausalDownResBlock(
          channel_increment=self._channel_increment,
          name=f'downres_stage_{stage_idx}',
      )(x)

      intermediates[f'scale_{scale}'] = x

    return x, intermediates


class TemporalDecoder(hk.Module):
  """Decodes temporal features with causal upsampling."""

  def __init__(
      self,
      num_stages: int = 6,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._num_stages = num_stages

  @typing.jaxtyped
  def __call__(
      self,
      x: Float[Array, 'B S D'],
      intermediates: dict[str, Array],
  ) -> Float[Array, 'B S_final D_final']:
    """Decodes temporal features with multi-scale upsampling.

    Args:
      x: Encoded features from transformer
      intermediates: Skip connections from encoder

    Returns:
      Decoded features at original resolution.
    """
    # Upsample through stages in reverse order
    # Start at coarsest (scale = 2^num_stages), then upsample to the next finer
    # scale each step: 2^(N-1) -> ... -> 2^0 (=1).
    for stage_idx in range(self._num_stages - 1, -1, -1):
      target_scale = 2 ** stage_idx  # Target scale after upsampling
      skip = intermediates[f'scale_{target_scale}']

      x = causal_layers.CausalUpResBlock(
          name=f'upres_to_scale_{target_scale}'
      )(x, skip)

    return x


class QuantileHead(hk.Module):
  """Prediction head for quantile regression."""

  def __init__(
      self,
      num_quantiles: int,
      hidden_dim: int = 256,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._num_quantiles = num_quantiles
    self._hidden_dim = hidden_dim

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B D']
  ) -> Float[Array, 'B {self._num_quantiles}']:
    """Predicts quantiles from embedding.

    Args:
      x: Input embedding [batch, d_model]

    Returns:
      Predicted quantiles [batch, num_quantiles].
    """
    h = layers.LayerNorm(rms_norm=True)(x)
    h = hk.Linear(self._hidden_dim)(h)
    h = jnp.tanh(h)
    h = hk.Linear(self._hidden_dim // 2)(h)
    h = jnp.tanh(h)

    # Output quantiles
    # Use a strategy to prevent crossing: predict median + deltas
    q_median = hk.Linear(1, name='q_median')(h)  # [B, 1]

    # Predict positive deltas for upper quantiles
    num_upper = self._num_quantiles // 2
    deltas_upper = hk.Linear(num_upper, name='deltas_upper')(h)
    deltas_upper = jax.nn.softplus(deltas_upper)  # Ensure positive
    deltas_upper = jnp.cumsum(deltas_upper, axis=-1)  # Cumulative for ordering

    # Predict negative deltas for lower quantiles
    num_lower = self._num_quantiles - num_upper - 1
    deltas_lower = hk.Linear(num_lower, name='deltas_lower')(h)
    deltas_lower = -jax.nn.softplus(deltas_lower)  # Ensure negative
    deltas_lower = jnp.cumsum(deltas_lower[:, ::-1], axis=-1)[:, ::-1]

    # Concatenate: [lower quantiles, median, upper quantiles]
    quantiles = jnp.concatenate(
        [q_median + deltas_lower, q_median, q_median + deltas_upper],
        axis=-1,
    )

    return quantiles


class RegimeHead(hk.Module):
  """Classification head for market regime prediction."""

  def __init__(
      self,
      num_classes: int = 3,
      hidden_dim: int = 128,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._num_classes = num_classes
    self._hidden_dim = hidden_dim

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B D']
  ) -> Float[Array, 'B {self._num_classes}']:
    """Predicts regime probabilities.

    Args:
      x: Input embedding [batch, d_model]

    Returns:
      Logits for regime classes [batch, num_classes].
    """
    h = layers.LayerNorm(rms_norm=True)(x)
    h = hk.Linear(self._hidden_dim)(h)
    h = jnp.tanh(h)
    logits = hk.Linear(self._num_classes)(h)
    return logits


class AlphaTrade(hk.Module):
  """AlphaTrade: Multi-horizon financial time series predictor.

  Architecture:
    1. Stem: Feature embedding
    2. Encoder: Multi-scale causal downsampling (6 stages -> /64)
    3. Transformer: Causal attention at coarse resolution
    4. Decoder: Multi-scale causal upsampling
    5. Readout: Extract final timestep embedding
    6. Heads: Multi-horizon quantile prediction
  """

  def __init__(
      self,
      config: schemas.AlphaTradeConfig | None = None,
      name: str | None = None,
  ):
    super().__init__(name=name or 'alphatrade')
    self._config = config or schemas.AlphaTradeConfig()

  @typing.jaxtyped
  def __call__(
      self, features: Float[Array, 'B L 8']
  ) -> schemas.AlphaTradeOutput:
    """Forward pass through AlphaTrade model.

    Args:
      features: Input features [batch, lookback_length, 8]

    Returns:
      AlphaTradeOutput containing predictions.
    """
    # 1. Stem: Embed features
    x = causal_layers.FeatureEmbedder(
        output_channels=self._config.stem_channels
    )(features)

    # 2. Encoder: Downsample to coarse resolution
    x, intermediates = TemporalEncoder(
        num_stages=self._config.num_encoder_stages,
        channel_increment=self._config.channel_increment,
    )(x)

    # Project to transformer dimension
    x = hk.Linear(self._config.d_model, name='proj_to_transformer')(x)

    # 3. Transformer: Model long-range dependencies at coarse scale
    x = causal_attention.CausalTransformerTower(
        num_layers=self._config.num_transformer_layers,
        num_heads=self._config.num_heads,
        head_dim=self._config.head_dim,
        mlp_expansion=self._config.mlp_expansion,
        logits_soft_cap=self._config.logits_soft_cap,
        max_position=self._config.max_position,
    )(x)

    # Project back for decoder
    final_channels = intermediates[f'scale_{2**self._config.num_encoder_stages}'].shape[-1]
    x = hk.Linear(final_channels, name='proj_from_transformer')(x)

    # 4. Decoder: Upsample back to original resolution
    x = TemporalDecoder(
        num_stages=self._config.num_encoder_stages,
    )(x, intermediates)

    # 5. Readout: Extract embedding at asof time (last timestep)
    readout_embedding = x[:, -1, :]  # [B, D]

    # 6. Heads: Multi-horizon predictions
    log_return_quantiles = {}
    for horizon in self._config.horizons:
      quantiles = QuantileHead(
          num_quantiles=len(self._config.quantiles),
          name=f'return_head_h{horizon}',
      )(readout_embedding)
      log_return_quantiles[horizon] = quantiles

    # Optional heads
    log_range_quantiles = None
    if self._config.predict_range:
      log_range_quantiles = {}
      for horizon in self._config.horizons:
        quantiles = QuantileHead(
            num_quantiles=len(self._config.quantiles),
            name=f'range_head_h{horizon}',
        )(readout_embedding)
        log_range_quantiles[horizon] = quantiles

    log_volume_quantiles = None
    if self._config.predict_volume:
      log_volume_quantiles = {}
      for horizon in self._config.horizons:
        quantiles = QuantileHead(
            num_quantiles=len(self._config.quantiles),
            name=f'volume_head_h{horizon}',
        )(readout_embedding)
        log_volume_quantiles[horizon] = quantiles

    regime_probs = None
    if self._config.predict_regime:
      regime_probs = RegimeHead(
          num_classes=self._config.num_regime_classes,
      )(readout_embedding)

    return schemas.AlphaTradeOutput(
        log_return_quantiles=log_return_quantiles,
        log_range_quantiles=log_range_quantiles,
        log_volume_quantiles=log_volume_quantiles,
        regime_probs=regime_probs,
        trunk_embedding=readout_embedding,
    )

  @typing.jaxtyped
  def loss(
      self,
      batch: schemas.TrainingBatch,
  ) -> tuple[Float[Array, ''], PyTree[Float[Array, '']]]:
    """Computes loss for a training batch.

    Args:
      batch: Training batch with inputs and targets

    Returns:
      Tuple of (total_loss, loss_dict).
    """
    from alphatrade.core import losses

    # Forward pass
    predictions = self(batch.inputs.features)

    # Prepare targets and predictions for loss computation
    targets_dict = {'log_returns': batch.targets.log_returns}
    predictions_dict = {'log_return_quantiles': predictions.log_return_quantiles}

    if batch.targets.log_ranges is not None:
      targets_dict['log_ranges'] = batch.targets.log_ranges
      predictions_dict['log_range_quantiles'] = predictions.log_range_quantiles

    if batch.targets.log_volumes is not None:
      targets_dict['log_volumes'] = batch.targets.log_volumes
      predictions_dict['log_volume_quantiles'] = predictions.log_volume_quantiles

    if batch.targets.regime_labels is not None:
      targets_dict['regime_labels'] = batch.targets.regime_labels
      predictions_dict['regime_probs'] = predictions.regime_probs

    # Compute combined loss
    quantiles_array = jnp.array(self._config.quantiles)
    return losses.combined_loss(
        targets=targets_dict,
        predictions=predictions_dict,
        quantiles=quantiles_array,
    )
