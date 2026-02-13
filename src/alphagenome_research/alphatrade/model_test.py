"""Tests for AlphaTrade model."""

import jax
import jax.numpy as jnp
import pytest

from alphagenome_research.alphatrade import model as model_lib
from alphagenome_research.alphatrade import schemas


class TestAlphaTrade:
  """Tests for AlphaTrade model."""

  def test_forward_pass(self):
    """Test basic forward pass."""
    import haiku as hk

    config = schemas.AlphaTradeConfig(
        lookback_length=128,
        stem_channels=32,
        num_encoder_stages=3,  # Downsample to /8
        d_model=64,
        num_transformer_layers=2,
        num_heads=4,
        horizons=[1, 5],
        quantiles=[0.25, 0.5, 0.75],
    )

    def forward(features):
      model = model_lib.AlphaTrade(config)
      return model(features)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    features = jnp.ones((2, config.lookback_length, 8))
    params = forward_fn.init(rng, features)
    output = forward_fn.apply(params, rng, features)

    # Check output structure
    assert isinstance(output, schemas.AlphaTradeOutput)
    assert len(output.log_return_quantiles) == 2  # 2 horizons
    assert 1 in output.log_return_quantiles
    assert 5 in output.log_return_quantiles

    # Check quantile shapes
    for horizon, quantiles in output.log_return_quantiles.items():
      assert quantiles.shape == (2, 3)  # batch_size=2, num_quantiles=3

  def test_quantile_ordering(self):
    """Test that predicted quantiles are ordered."""
    import haiku as hk

    config = schemas.AlphaTradeConfig(
        lookback_length=64,
        stem_channels=16,
        num_encoder_stages=2,
        d_model=32,
        num_transformer_layers=1,
        num_heads=2,
        horizons=[1],
        quantiles=[0.1, 0.5, 0.9],
    )

    def forward(features):
      model = model_lib.AlphaTrade(config)
      return model(features)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(42)

    features = jnp.ones((1, config.lookback_length, 8))
    params = forward_fn.init(rng, features)
    output = forward_fn.apply(params, rng, features)

    # Check quantiles are ordered: q10 <= q50 <= q90
    quantiles = output.log_return_quantiles[1][0]
    assert quantiles[0] <= quantiles[1]  # q10 <= q50
    assert quantiles[1] <= quantiles[2]  # q50 <= q90

  def test_loss_computation(self):
    """Test loss computation."""
    import haiku as hk

    config = schemas.AlphaTradeConfig(
        lookback_length=64,
        stem_channels=16,
        num_encoder_stages=2,
        d_model=32,
        num_transformer_layers=1,
        horizons=[1, 5],
        quantiles=[0.25, 0.5, 0.75],
    )

    def forward(batch):
      model = model_lib.AlphaTrade(config)
      return model.loss(batch)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    # Create dummy batch
    features = jnp.ones((2, config.lookback_length, 8))
    targets = schemas.AlphaTradeTargets(
        log_returns={
            1: jnp.array([0.001, -0.002]),
            5: jnp.array([0.003, -0.001]),
        }
    )
    batch = schemas.TrainingBatch(
        inputs=schemas.AlphaTradeInput(features=features),
        targets=targets,
    )

    params = forward_fn.init(rng, batch)
    loss, metrics = forward_fn.apply(params, rng, batch)

    # Check loss is scalar and finite
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert loss > 0

    # Check metrics
    assert 'combined_total_loss' in metrics
    assert 'return_total_loss' in metrics

  def test_optional_heads(self):
    """Test optional prediction heads."""
    import haiku as hk

    config = schemas.AlphaTradeConfig(
        lookback_length=64,
        stem_channels=16,
        num_encoder_stages=2,
        d_model=32,
        num_transformer_layers=1,
        horizons=[1],
        quantiles=[0.5],
        predict_range=True,
        predict_volume=True,
        predict_regime=True,
        num_regime_classes=3,
    )

    def forward(features):
      model = model_lib.AlphaTrade(config)
      return model(features)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    features = jnp.ones((1, config.lookback_length, 8))
    params = forward_fn.init(rng, features)
    output = forward_fn.apply(params, rng, features)

    # Check optional outputs exist
    assert output.log_range_quantiles is not None
    assert output.log_volume_quantiles is not None
    assert output.regime_probs is not None

    # Check shapes
    assert output.log_range_quantiles[1].shape == (1, 1)
    assert output.log_volume_quantiles[1].shape == (1, 1)
    assert output.regime_probs.shape == (1, 3)


class TestTemporalEncoder:
  """Tests for temporal encoder."""

  def test_downsampling(self):
    """Test that encoder downsamples correctly."""
    import haiku as hk

    def forward(x):
      encoder = model_lib.TemporalEncoder(
          num_stages=3, channel_increment=32
      )
      return encoder(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((2, 128, 64))
    params = forward_fn.init(rng, x)
    encoded, intermediates = forward_fn.apply(params, rng, x)

    # After 3 stages: /2^3 = /8
    assert encoded.shape[1] == 16  # 128 / 8

    # Check intermediates
    assert 'scale_1' in intermediates
    assert 'scale_2' in intermediates
    assert 'scale_4' in intermediates
    assert 'scale_8' in intermediates


class TestTemporalDecoder:
  """Tests for temporal decoder."""

  def test_upsampling(self):
    """Test that decoder upsamples correctly."""
    import haiku as hk

    def forward(x, intermediates):
      decoder = model_lib.TemporalDecoder(num_stages=3)
      return decoder(x, intermediates)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    # Create dummy intermediates
    intermediates = {
        'scale_1': jnp.ones((2, 128, 64)),
        'scale_2': jnp.ones((2, 64, 96)),
        'scale_4': jnp.ones((2, 32, 128)),
        'scale_8': jnp.ones((2, 16, 160)),
    }
    x = jnp.ones((2, 16, 160))

    params = forward_fn.init(rng, x, intermediates)
    decoded = forward_fn.apply(params, rng, x, intermediates)

    # Should upsample back to original resolution
    assert decoded.shape[1] == 128


class TestQuantileHead:
  """Tests for quantile prediction head."""

  def test_output_shape(self):
    """Test output shape is correct."""
    import haiku as hk

    def forward(x):
      head = model_lib.QuantileHead(num_quantiles=5)
      return head(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    x = jnp.ones((4, 256))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    assert output.shape == (4, 5)

  def test_quantile_ordering(self):
    """Test that quantiles are ordered."""
    import haiku as hk

    def forward(x):
      head = model_lib.QuantileHead(num_quantiles=5)
      return head(x)

    forward_fn = hk.transform(forward)
    rng = jax.random.PRNGKey(42)

    x = jnp.ones((1, 256))
    params = forward_fn.init(rng, x)
    output = forward_fn.apply(params, rng, x)

    # Check monotonicity
    quantiles = output[0]
    for i in range(len(quantiles) - 1):
      assert quantiles[i] <= quantiles[i + 1]


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
