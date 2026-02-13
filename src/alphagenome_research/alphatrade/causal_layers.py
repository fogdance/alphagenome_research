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

"""Causal layers for time-series modeling."""

from alphagenome import typing
from alphagenome_research.model import layers
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class CausalStandardizedConv1D(hk.Module):
  """Causal 1D Convolution with weight standardization.

  Uses left-padding to ensure causality: output at time t only depends on
  inputs at times <= t.
  """

  def __init__(
      self,
      num_channels: int,
      width: int,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._num_channels = num_channels
    self._width = width

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D']
  ) -> Float[Array, 'B S {self._num_channels}']:
    input_channels = x.shape[-1]
    fan_in = self._width * input_channels
    kernel_shape = (self._width, input_channels, self._num_channels)
    # Use VarianceScaling initialization instead of zeros for better training
    w_init = hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")
    w = hk.get_parameter('w', shape=kernel_shape, dtype=x.dtype, init=w_init)

    # Weight standardization (same as AlphaGenome)
    w -= jnp.mean(w, axis=(0, 1), keepdims=True)
    var_w = jnp.var(w, axis=(0, 1), keepdims=True)
    scale = hk.get_parameter(
        'scale',
        shape=[1, 1, self._num_channels],
        init=jnp.ones,
        dtype=w.dtype,
    )
    scale = scale * jax.lax.rsqrt(jnp.maximum(fan_in * var_w, 1e-4))
    w_standardized = w * scale

    # Causal padding: pad (width-1) on the left, 0 on the right
    padding = [(0, 0), (self._width - 1, 0), (0, 0)]
    x_padded = jnp.pad(x, padding, mode='constant', constant_values=0)

    out = jax.lax.conv_general_dilated(
        lhs=x_padded,
        rhs=w_standardized,
        window_strides=[1],
        padding='VALID',
        dimension_numbers=jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 2, 1), rhs_spec=(2, 1, 0), out_spec=(0, 2, 1)
        ),
    )
    bias = hk.get_parameter(
        'bias', shape=(self._num_channels,), dtype=x.dtype, init=jnp.zeros
    )
    bias = jnp.broadcast_to(bias, out.shape)
    return out + bias


class CausalConvBlock(hk.Module):
  """Causal convolutional block with GELU and RMSNorm."""

  def __init__(self, num_channels: int, width: int, name: str | None = None):
    super().__init__(name=name)
    self._num_channels = num_channels
    self._width = width

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D']
  ) -> Float[Array, 'B S {self._num_channels}']:
    x = layers.gelu(layers.LayerNorm(rms_norm=True)(x))
    if self._width == 1:
      return hk.Linear(self._num_channels)(x)
    else:
      return CausalStandardizedConv1D(
          num_channels=self._num_channels, width=self._width
      )(x)


def causal_pool(
    x: Float[Array, 'B S D'], by: int = 2
) -> Float[Array, 'B S/{by} D']:
  """Causal max pooling: only looks at past values.

  For each output position i, pools from input positions [i*by, (i+1)*by).

  If seq_len is not divisible by `by`, we **left-pad** with -inf to make it
  divisible. Left-padding keeps the most recent timesteps aligned (i.e., the
  tail of the sequence is preserved without shifting relative to output bins).
  """
  batch_size, seq_len, channels = x.shape

  # Ensure sequence length is divisible by pooling factor
  if seq_len % by != 0:
    pad_len = by - (seq_len % by)
    neg_inf = jnp.array(-jnp.inf, dtype=x.dtype)
    x = jnp.pad(x, [(0, 0), (pad_len, 0), (0, 0)], constant_values=neg_inf)
    seq_len = x.shape[1]

  # Reshape to [B, S//by, by, D] and take max over the pooling dimension
  x_reshaped = x.reshape(batch_size, seq_len // by, by, channels)
  return jnp.max(x_reshaped, axis=2)


class FeatureEmbedder(hk.Module):
  """Embeds input features into a higher-dimensional space."""

  def __init__(self, output_channels: int = 128, name: str | None = None):
    super().__init__(name=name)
    self._output_channels = output_channels

  @typing.jaxtyped
  def __call__(
      self, features: Float[Array, 'B S F']
  ) -> Float[Array, 'B S {self._output_channels}']:
    # Initial projection with causal conv
    x = CausalStandardizedConv1D(
        num_channels=self._output_channels, width=7
    )(features)
    # Add residual block
    return x + CausalConvBlock(
        num_channels=self._output_channels, width=5
    )(x)


class CausalDownResBlock(hk.Module):
  """Causal downsampling residual block."""

  def __init__(self, channel_increment: int = 64, name: str | None = None):
    super().__init__(name=name)
    self._channel_increment = channel_increment

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D']
  ) -> Float[Array, 'B S D+increment']:
    num_out_channels = x.shape[-1] + self._channel_increment
    out = CausalConvBlock(num_channels=num_out_channels, width=5)(x)
    # Pad channels for residual connection
    out = out + jnp.pad(x, [(0, 0), (0, 0), (0, self._channel_increment)])
    return out + CausalConvBlock(num_channels=out.shape[-1], width=5)(out)


class CausalUpResBlock(hk.Module):
  """Causal upsampling residual block with skip connections."""

  @typing.jaxtyped
  def __call__(
      self, x: Float[Array, 'B S D'], skip: Float[Array, 'B S_skip D_skip']
  ) -> Float[Array, 'B S_up D_skip']:
    num_channels = skip.shape[-1]

    # Project and add residual
    out = (
        CausalConvBlock(num_channels=num_channels, width=5, name='conv_in')(x)
        + x[:, :, :num_channels]
    )

    # Upsample by repeating (AlphaGenome style)
    out = jnp.repeat(out, 2, axis=1)

    # Trim to match skip length if needed
    if out.shape[1] > skip.shape[1]:
      out = out[:, :skip.shape[1], :]

    # Apply residual scale
    residual_scale = hk.get_parameter(
        'residual_scale', (), init=jnp.ones
    ).astype(out.dtype)
    out *= residual_scale

    # Add skip connection
    out += CausalConvBlock(
        num_channels=num_channels, width=1, name='pointwise_conv_skip'
    )(skip)

    return out + CausalConvBlock(
        num_channels=num_channels, width=5, name='conv_out'
    )(out)
