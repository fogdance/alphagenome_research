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

"""Causal attention layers for time-series modeling."""

import math
from alphagenome import typing
from alphagenome_research.model import layers
import haiku as hk
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def apply_rope(
    x: Float[Array, 'B S H C'],
    positions: Int[Array, 'B S'] | None,
    max_position: int,
    base: float = 10000.0,
) -> Float[Array, 'B S H C']:
  """Applies Rotary Position Embeddings (RoPE) to the input tensor.

  Uses standard RoPE formulation from "RoFormer: Enhanced Transformer with
  Rotary Position Embedding" (Su et al., 2021).

  Args:
    x: Input tensor [batch, seq, heads, channels]
    positions: Optional position indices [batch, seq]. If None, uses 0..seq-1
    max_position: Maximum position (used for base scaling if needed)
    base: Base for frequency computation (default 10000)

  Returns:
    Tensor with RoPE applied, same shape as input.
  """
  if positions is None:
    positions = jnp.arange(x.shape[1]).astype(x.dtype).reshape(1, x.shape[1])

  # Standard RoPE frequency computation
  d = x.shape[-1]
  if d % 2 != 0:
    raise ValueError(f"RoPE requires even head_dim, got {d}")

  inv_freq = 1.0 / (base ** (jnp.arange(0, d, 2).astype(x.dtype) / d))

  # Compute angles: [batch, seq, d/2]
  theta = jnp.einsum('bs,f->bsf', positions, inv_freq)

  # Repeat to match full dimension: [batch, seq, 1, d]
  theta = jnp.repeat(theta, 2, axis=-1)[..., None, :]

  # Rotate: stack [-x_odd, x_even] for 90-degree rotation
  x_rotated = jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(
      x.shape
  )

  # Apply rotation: x * cos(theta) + x_rotated * sin(theta)
  return x * jnp.cos(theta) + x_rotated * jnp.sin(theta)


def create_causal_mask(seq_len: int) -> Float[Array, 'S S']:
  """Creates a causal mask for attention: position i can only attend to j <= i."""
  mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
  # Convert to attention bias: 0 for allowed, -inf for masked
  return jnp.where(mask, 0.0, -1e10)


class CausalMLPBlock(hk.Module):
  """MLP block for causal sequence modeling."""

  def __init__(self, expansion: int = 2, name: str | None = None):
    super().__init__(name=name)
    self._expansion = expansion

  @typing.jaxtyped
  def __call__(self, x: Float[Array, 'B S D']) -> Float[Array, 'B S D']:
    h = layers.LayerNorm(rms_norm=True)(x)
    h = hk.Linear(x.shape[-1] * self._expansion)(h)
    h = jax.nn.gelu(h)
    h = hk.Linear(x.shape[-1])(h)
    return layers.LayerNorm(rms_norm=True)(h)


class CausalMHABlock(hk.Module):
  """Causal Multi-Head Attention with RoPE and logits soft-cap."""

  def __init__(
      self,
      num_heads: int = 8,
      head_dim: int = 64,
      logits_soft_cap: float = 5.0,
      max_position: int = 8192,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._num_heads = num_heads
    self._head_dim = head_dim
    self._logits_soft_cap = logits_soft_cap
    self._max_position = max_position

  @typing.jaxtyped
  def __call__(self, x: Float[Array, 'B S D']) -> Float[Array, 'B S D']:
    batch_size, seq_len, d_model = x.shape

    # Pre-norm
    h = layers.LayerNorm(rms_norm=True)(x)

    # Project to Q, K, V
    qkv_dim = self._num_heads * self._head_dim
    q = layers.LayerNorm(name='norm_q', rms_norm=True)(
        hk.Linear(qkv_dim, with_bias=False, name='q_layer')(h).reshape(
            batch_size, seq_len, self._num_heads, self._head_dim
        )
    )
    k = layers.LayerNorm(name='norm_k', rms_norm=True)(
        hk.Linear(qkv_dim, with_bias=False, name='k_layer')(h).reshape(
            batch_size, seq_len, self._num_heads, self._head_dim
        )
    )
    v = layers.LayerNorm(name='norm_v', rms_norm=True)(
        hk.Linear(qkv_dim, with_bias=False, name='v_layer')(h).reshape(
            batch_size, seq_len, self._num_heads, self._head_dim
        )
    )

    # Apply RoPE to Q and K
    q = apply_rope(q, None, max_position=self._max_position)
    k = apply_rope(k, None, max_position=self._max_position)

    # Compute attention logits
    logits_dtype = jnp.float32
    # Note: Uses BF16 for attention computation (lhs/rhs bf16, accum f32)
    # for better speed and stability. This is standard practice in modern
    # transformers (GPT-3/4, PaLM, Gemini). For strict fp32 debugging,
    # change precision to jax.lax.Precision.DEFAULT.
    attention_logits = jnp.einsum(
        'bshc,bShc->bhsS',
        q,
        k,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        preferred_element_type=logits_dtype,
    )
    attention_logits = attention_logits / math.sqrt(self._head_dim)

    # Apply logits soft-cap (AlphaGenome style) BEFORE causal mask
    # This prevents the mask from being affected by the soft-cap
    attention_logits = (
        jnp.tanh(attention_logits / self._logits_soft_cap)
        * self._logits_soft_cap
    )

    # Apply causal mask AFTER soft-cap to ensure strict causality
    # Using -inf ensures future positions get exactly zero attention weight
    causal_mask = create_causal_mask(seq_len)
    attention_logits = jnp.where(
        causal_mask[None, None, :, :] == 0.0,
        attention_logits,
        -jnp.inf,
    )

    # Softmax and apply to values
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    y = jnp.einsum(
        'bhsS,bShc->bshc',
        attention_weights,
        v,
        precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
    ).astype(q.dtype)

    # Project back to d_model
    y = hk.Linear(d_model, name='output_projection')(
        y.reshape(batch_size, seq_len, -1)
    )
    return layers.LayerNorm(rms_norm=True)(y)


class CausalTransformerLayer(hk.Module):
  """Single causal transformer layer with attention and MLP."""

  def __init__(
      self,
      num_heads: int = 8,
      head_dim: int = 64,
      mlp_expansion: int = 4,
      logits_soft_cap: float = 5.0,
      max_position: int = 8192,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._num_heads = num_heads
    self._head_dim = head_dim
    self._mlp_expansion = mlp_expansion
    self._logits_soft_cap = logits_soft_cap
    self._max_position = max_position

  @typing.jaxtyped
  def __call__(self, x: Float[Array, 'B S D']) -> Float[Array, 'B S D']:
    # Attention with residual
    x = x + CausalMHABlock(
        num_heads=self._num_heads,
        head_dim=self._head_dim,
        logits_soft_cap=self._logits_soft_cap,
        max_position=self._max_position,
    )(x)

    # MLP with residual
    x = x + CausalMLPBlock(expansion=self._mlp_expansion)(x)

    return x


class CausalTransformerTower(hk.Module):
  """Stack of causal transformer layers."""

  def __init__(
      self,
      num_layers: int = 6,
      num_heads: int = 8,
      head_dim: int = 64,
      mlp_expansion: int = 4,
      logits_soft_cap: float = 5.0,
      max_position: int = 8192,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._head_dim = head_dim
    self._mlp_expansion = mlp_expansion
    self._logits_soft_cap = logits_soft_cap
    self._max_position = max_position

  @typing.jaxtyped
  def __call__(self, x: Float[Array, 'B S D']) -> Float[Array, 'B S D']:
    for i in range(self._num_layers):
      x = CausalTransformerLayer(
          num_heads=self._num_heads,
          head_dim=self._head_dim,
          mlp_expansion=self._mlp_expansion,
          logits_soft_cap=self._logits_soft_cap,
          max_position=self._max_position,
          name=f'layer_{i}',
      )(x)
    return x
