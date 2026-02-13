"""Tests for AlphaTrade causal attention layers."""

import jax
import jax.numpy as jnp
import pytest

from alphagenome_research.alphatrade import causal_attention


class TestApplyRoPE:
  """Tests for RoPE (Rotary Position Embeddings)."""

  def test_output_shape(self):
    """Test that RoPE preserves shape."""
    x = jnp.ones((2, 100, 8, 64))
    output = causal_attention.apply_rope(x, None, max_position=8192)
    assert output.shape == x.shape

  def test_position_encoding(self):
    """Test that different positions get different encodings."""
    x = jnp.ones((1, 10, 1, 64))
    output = causal_attention.apply_rope(x, None, max_position=8192)

    # Different positions should have different encodings
    assert not jnp.allclose(output[0, 0], output[0, 5])

  def test_custom_positions(self):
    """Test with custom position indices."""
    x = jnp.ones((1, 5, 1, 64))
    positions = jnp.array([[0, 10, 20, 30, 40]])
    output = causal_attention.apply_rope(x, positions, max_position=8192)

    assert output.shape == x.shape


class TestCreateCausalMask:
  """Tests for causal mask creation."""

  def test_mask_shape(self):
    """Test mask has correct shape."""
    mask = causal_attention.create_causal_mask(10)
    assert mask.shape == (10, 10)

  def test_mask_values(self):
    """Test mask allows past and blocks future."""
    mask = causal_attention.create_causal_mask(5)

    # Diagonal and below should be 0 (allowed)
    assert mask[0, 0] == 0.0
    assert mask[1, 0] == 0.0
    assert mask[1, 1] == 0.0
    assert mask[4, 2] == 0.0

    # Above diagonal should be -inf (blocked)
    assert mask[0, 1] < -1e9
    assert mask[1, 2] < -1e9
    assert mask[2, 4] < -1e9


class TestCausalMLPBlock:
  """Tests for causal MLP block."""

  def test_output_shape(self):
    """Test output shape matches input."""
    import haiku as hk

    def forward(x):
      return causal_attention.CausalMLPBlock(expansion=4)(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 100, 256))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    assert output.shape == x.shape


class TestCausalMHABlock:
  """Tests for causal multi-head attention."""

  def test_output_shape(self):
    """Test output shape matches input."""
    import haiku as hk

    def forward(x):
      return causal_attention.CausalMHABlock(
          num_heads=8, head_dim=64
      )(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 100, 512))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    assert output.shape == x.shape

  def test_causality(self):
    """Test that attention is causal."""
    import haiku as hk

    def forward(x):
      return causal_attention.CausalMHABlock(
          num_heads=4, head_dim=32
      )(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    # Create input with spike at position t
    x = jnp.zeros((1, 100, 128))
    t = 50
    x = x.at[:, t, :].set(1.0)

    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    # Output at positions before spike should be close to zero
    # (no information from future)
    assert jnp.allclose(output[:, :t-5, :], 0.0, atol=1e-4)

  def test_logits_soft_cap(self):
    """Test that logits soft-cap is applied."""
    import haiku as hk

    def forward(x):
      return causal_attention.CausalMHABlock(
          num_heads=2, head_dim=32, logits_soft_cap=5.0
      )(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    # Create input with large values
    x = jnp.ones((1, 50, 64)) * 10.0

    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    # Output should be finite (not explode)
    assert jnp.all(jnp.isfinite(output))


class TestCausalTransformerLayer:
  """Tests for causal transformer layer."""

  def test_output_shape(self):
    """Test output shape matches input."""
    import haiku as hk

    def forward(x):
      return causal_attention.CausalTransformerLayer(
          num_heads=8, head_dim=64, mlp_expansion=4
      )(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 100, 512))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    assert output.shape == x.shape

  def test_residual_connections(self):
    """Test that residual connections work."""
    import haiku as hk

    def forward(x):
      return causal_attention.CausalTransformerLayer(
          num_heads=4, head_dim=32
      )(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((1, 50, 128))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    # Output should be different from input (not identity)
    assert not jnp.allclose(output, x)


class TestCausalTransformerTower:
  """Tests for causal transformer tower."""

  def test_output_shape(self):
    """Test output shape matches input."""
    import haiku as hk

    def forward(x):
      return causal_attention.CausalTransformerTower(
          num_layers=4,
          num_heads=8,
          head_dim=64,
      )(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 100, 512))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    assert output.shape == x.shape

  def test_multiple_layers(self):
    """Test that multiple layers are applied."""
    import haiku as hk

    def forward_1layer(x):
      return causal_attention.CausalTransformerTower(
          num_layers=1, num_heads=4, head_dim=32
      )(x)

    def forward_4layers(x):
      return causal_attention.CausalTransformerTower(
          num_layers=4, num_heads=4, head_dim=32
      )(x)

    forward_fn_1 = hk.transform(forward_1layer)
    forward_fn_4 = hk.transform(forward_4layers)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((1, 50, 128))

    params_1 = forward_fn_1.init(rng, x)
    params_4 = forward_fn_4.init(rng, x)

    output_1 = forward_fn_1.apply(params_1, rng, x)
    output_4 = forward_fn_4.apply(params_4, rng, x)

    # Different number of layers should give different outputs
    assert not jnp.allclose(output_1, output_4)

  def test_causality_preserved(self):
    """Test that causality is preserved through tower."""
    import haiku as hk

    def forward(x):
      return causal_attention.CausalTransformerTower(
          num_layers=3, num_heads=4, head_dim=32
      )(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    # Create input with spike
    x = jnp.zeros((1, 100, 128))
    t = 50
    x = x.at[:, t, :].set(1.0)

    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    # Output before spike should be close to zero
    assert jnp.allclose(output[:, :t-10, :], 0.0, atol=1e-3)


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
