"""Tests for AlphaTrade causal layers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphatrade.core import causal_layers


class TestCausalStandardizedConv1D:
  """Tests for causal standardized convolution."""

  def test_causality(self):
    """Test that convolution is strictly causal using prefix-invariance."""
    import haiku as hk

    def forward(x):
      return causal_layers.CausalStandardizedConv1D(
          num_channels=16, width=5
      )(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    # Create two inputs with same prefix but different futures
    x1 = jax.random.normal(jax.random.PRNGKey(1), (1, 100, 8))
    x2 = jax.random.normal(jax.random.PRNGKey(2), (1, 100, 8))

    # Make prefix identical up to position t
    t = 50
    x2 = x2.at[:, :t+1, :].set(x1[:, :t+1, :])

    params = forward_fn.init(rng, x1)
    output1 = forward_fn.apply(params, rng, x1)
    output2 = forward_fn.apply(params, rng, x2)

    # Outputs at position t should be identical (prefix-invariance)
    assert jnp.allclose(output1[:, t, :], output2[:, t, :], atol=1e-5)
    # Outputs after t can differ (future information allowed to differ)
    # This just checks the test is meaningful
    assert output1.shape == output2.shape

  def test_output_shape(self):
    """Test output shape matches input sequence length."""
    import haiku as hk

    def forward(x):
      return causal_layers.CausalStandardizedConv1D(
          num_channels=32, width=7
      )(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 100, 16))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    assert output.shape == (2, 100, 32)


class TestCausalPool:
  """Tests for causal pooling."""

  def test_causality(self):
    """Test that pooling only looks at past values."""
    x = jnp.arange(100).reshape(1, 100, 1).astype(jnp.float32)
    pooled = causal_layers.causal_pool(x, by=2)

    # Each output position should be max of [2i, 2i+1]
    for i in range(pooled.shape[1]):
      expected = max(2*i, 2*i+1)
      assert pooled[0, i, 0] == expected

  def test_output_shape(self):
    """Test output shape is correctly downsampled."""
    x = jnp.ones((2, 100, 16))
    pooled = causal_layers.causal_pool(x, by=2)
    assert pooled.shape == (2, 50, 16)

    pooled = causal_layers.causal_pool(x, by=4)
    assert pooled.shape == (2, 25, 16)


class TestFeatureEmbedder:
  """Tests for feature embedder."""

  def test_output_shape(self):
    """Test output shape is correct."""
    import haiku as hk

    def forward(x):
      return causal_layers.FeatureEmbedder(output_channels=128)(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 100, 8))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    assert output.shape == (2, 100, 128)

  def test_causality(self):
    """Test that embedding is causal using prefix-invariance."""
    import haiku as hk

    def forward(x):
      return causal_layers.FeatureEmbedder(output_channels=64)(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    # Create two inputs with same prefix but different futures
    x1 = jax.random.normal(jax.random.PRNGKey(1), (1, 100, 8))
    x2 = jax.random.normal(jax.random.PRNGKey(2), (1, 100, 8))

    # Make prefix identical up to position t
    t = 50
    x2 = x2.at[:, :t+1, :].set(x1[:, :t+1, :])

    params = forward_fn.init(rng, x1)
    output1 = forward_fn.apply(params, rng, x1)
    output2 = forward_fn.apply(params, rng, x2)

    # Outputs at position t should be identical (prefix-invariance)
    assert jnp.allclose(output1[:, t, :], output2[:, t, :], atol=1e-5)


class TestCausalDownResBlock:
  """Tests for causal downsampling residual block."""

  def test_channel_increment(self):
    """Test that channels are incremented correctly."""
    import haiku as hk

    def forward(x):
      return causal_layers.CausalDownResBlock(channel_increment=64)(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 100, 128))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    assert output.shape == (2, 100, 192)  # 128 + 64


class TestCausalUpResBlock:
  """Tests for causal upsampling residual block."""

  def test_upsampling(self):
    """Test that upsampling doubles sequence length."""
    import haiku as hk

    def forward(x, skip):
      return causal_layers.CausalUpResBlock()(x, skip)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 50, 256))
    skip = jnp.ones((2, 100, 128))

    params = forward_fn.init(rng, x, skip)
    output = forward_fn.apply(params, rng, x, skip)

    assert output.shape == (2, 100, 128)  # Matches skip shape

  def test_skip_connection(self):
    """Test that skip connection is incorporated."""
    import haiku as hk

    def forward(x, skip):
      return causal_layers.CausalUpResBlock()(x, skip)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.zeros((1, 50, 256))
    skip = jnp.ones((1, 100, 128))

    params = forward_fn.init(rng, x, skip)
    output = forward_fn.apply(params, rng, x, skip)

    # Output should be influenced by skip (non-zero)
    assert not jnp.allclose(output, 0.0)


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
