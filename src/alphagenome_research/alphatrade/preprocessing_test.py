"""Tests for AlphaTrade preprocessing."""

import jax.numpy as jnp
import numpy as np
import pytest

from alphagenome_research.alphatrade import preprocessing


class TestFeaturePreprocessor:
  """Tests for feature preprocessor."""

  def test_compute_features_shape(self):
    """Test that computed features have correct shape."""
    L = 100
    preprocessor = preprocessing.FeaturePreprocessor()

    features = preprocessor.compute_features(
        open_prices=jnp.ones(L) * 100,
        high_prices=jnp.ones(L) * 101,
        low_prices=jnp.ones(L) * 99,
        close_prices=jnp.ones(L) * 100.5,
        volumes=jnp.ones(L) * 1000,
    )

    assert features.shape == (L, 8)

  def test_lr_close_computation(self):
    """Test log return computation."""
    preprocessor = preprocessing.FeaturePreprocessor()
    close_prices = jnp.array([100.0, 101.0, 100.5, 102.0])

    features = preprocessor.compute_features(
        open_prices=close_prices,
        high_prices=close_prices * 1.01,
        low_prices=close_prices * 0.99,
        close_prices=close_prices,
        volumes=jnp.ones(4) * 1000,
    )

    lr_close = features[:, 0]
    # First value should be 0 (no previous bar)
    assert lr_close[0] == 0.0
    # Second value should be log(101/100)
    expected = jnp.log(101.0 / 100.0)
    assert jnp.allclose(lr_close[1], expected, atol=1e-6)

  def test_pos_in_range(self):
    """Test position in range computation."""
    preprocessor = preprocessing.FeaturePreprocessor()

    # Close at high
    features_high = preprocessor.compute_features(
        open_prices=jnp.array([100.0]),
        high_prices=jnp.array([102.0]),
        low_prices=jnp.array([98.0]),
        close_prices=jnp.array([102.0]),
        volumes=jnp.array([1000.0]),
    )
    assert jnp.allclose(features_high[0, 4], 1.0, atol=1e-6)

    # Close at low
    features_low = preprocessor.compute_features(
        open_prices=jnp.array([100.0]),
        high_prices=jnp.array([102.0]),
        low_prices=jnp.array([98.0]),
        close_prices=jnp.array([98.0]),
        volumes=jnp.array([1000.0]),
    )
    assert jnp.allclose(features_low[0, 4], 0.0, atol=1e-6)

    # Close at midpoint
    features_mid = preprocessor.compute_features(
        open_prices=jnp.array([100.0]),
        high_prices=jnp.array([102.0]),
        low_prices=jnp.array([98.0]),
        close_prices=jnp.array([100.0]),
        volumes=jnp.array([1000.0]),
    )
    assert jnp.allclose(features_mid[0, 4], 0.5, atol=1e-6)

  def test_time_encoding(self):
    """Test time encoding with sin/cos."""
    preprocessor = preprocessing.FeaturePreprocessor(
        session_period_minutes=1440
    )

    minute_indices = jnp.array([0.0, 360.0, 720.0, 1080.0])
    features = preprocessor.compute_features(
        open_prices=jnp.ones(4) * 100,
        high_prices=jnp.ones(4) * 101,
        low_prices=jnp.ones(4) * 99,
        close_prices=jnp.ones(4) * 100,
        volumes=jnp.ones(4) * 1000,
        minute_indices=minute_indices,
    )

    time_sin = features[:, 5]
    time_cos = features[:, 6]

    # Check that sin^2 + cos^2 = 1
    for i in range(4):
      assert jnp.allclose(
          time_sin[i]**2 + time_cos[i]**2, 1.0, atol=1e-6
      )


class TestRobustScaler:
  """Tests for robust scaler."""

  def test_fit_transform(self):
    """Test fitting and transforming."""
    scaler = preprocessing.RobustScaler()
    features = np.random.randn(100, 8)

    normalized = scaler.fit_transform(features)

    assert normalized.shape == features.shape
    assert scaler.fitted_

    # Check that median is approximately 0
    medians = np.median(normalized, axis=0)
    assert np.allclose(medians, 0.0, atol=0.1)

  def test_inverse_transform(self):
    """Test inverse transformation."""
    scaler = preprocessing.RobustScaler()
    features = np.random.randn(100, 8)

    normalized = scaler.fit_transform(features)
    reconstructed = scaler.inverse_transform(normalized)

    assert np.allclose(reconstructed, features, atol=1e-6)

  def test_transform_without_fit(self):
    """Test that transform fails without fitting."""
    scaler = preprocessing.RobustScaler()
    features = np.random.randn(10, 8)

    with pytest.raises(ValueError, match='must be fitted'):
      scaler.transform(features)

  def test_get_set_params(self):
    """Test getting and setting parameters."""
    scaler1 = preprocessing.RobustScaler()
    features = np.random.randn(100, 8)
    scaler1.fit(features)

    params = scaler1.get_params()
    assert len(params) == 8

    scaler2 = preprocessing.RobustScaler()
    scaler2.set_params(params)

    # Both scalers should produce same output
    test_features = np.random.randn(10, 8)
    output1 = scaler1.transform(test_features)
    output2 = scaler2.transform(test_features)

    assert np.allclose(output1, output2)

  def test_batch_input(self):
    """Test handling of batch input [N, L, 8]."""
    scaler = preprocessing.RobustScaler()
    features = np.random.randn(10, 100, 8)  # 10 samples, 100 timesteps

    normalized = scaler.fit_transform(features)

    assert normalized.shape == features.shape
    assert scaler.fitted_


class TestValidateFeatures:
  """Tests for feature validation."""

  def test_valid_features(self):
    """Test validation passes for valid features."""
    features = jnp.array([
        [0.001, -1.5, 0.0005, 5.0, 0.5, 0.7, 0.7, 1.0],
        [-0.002, -1.3, -0.001, 5.2, 0.3, 0.8, 0.6, 1.0],
    ])

    assert preprocessing.validate_features(features)

  def test_wrong_feature_count(self):
    """Test validation fails for wrong feature count."""
    features = jnp.ones((10, 7))  # Only 7 features

    with pytest.raises(ValueError, match='Expected 8 features'):
      preprocessing.validate_features(features)

  def test_nan_values(self):
    """Test validation fails for NaN values."""
    features = jnp.ones((10, 8))
    features = features.at[5, 3].set(jnp.nan)

    with pytest.raises(ValueError, match='NaN'):
      preprocessing.validate_features(features)

  def test_inf_values(self):
    """Test validation fails for Inf values."""
    features = jnp.ones((10, 8))
    features = features.at[5, 3].set(jnp.inf)

    with pytest.raises(ValueError, match='Inf'):
      preprocessing.validate_features(features)

  def test_pos_in_range_bounds(self):
    """Test validation fails for out-of-range pos_in_range."""
    features = jnp.ones((10, 8))
    features = features.at[:, 4].set(0.5)  # Valid
    assert preprocessing.validate_features(features)

    # Invalid: > 1
    features = features.at[5, 4].set(1.5)
    with pytest.raises(ValueError, match='pos_in_range'):
      preprocessing.validate_features(features)

  def test_session_open_binary(self):
    """Test validation fails for non-binary is_session_open."""
    features = jnp.ones((10, 8))
    features = features.at[:, 7].set(1.0)  # Valid
    assert preprocessing.validate_features(features)

    # Invalid: not 0 or 1
    features = features.at[5, 7].set(0.5)
    with pytest.raises(ValueError, match='is_session_open'):
      preprocessing.validate_features(features)


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
