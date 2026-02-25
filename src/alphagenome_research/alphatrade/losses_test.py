"""Tests for AlphaTrade loss functions."""

import jax.numpy as jnp
import pytest

from alphagenome_research.alphatrade import losses


class TestQuantilePinballLoss:
  """Tests for quantile pinball loss."""

  def test_perfect_prediction(self):
    """Test loss is zero for perfect predictions."""
    y_true = jnp.array([0.0, 1.0, 2.0])
    y_pred = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    quantiles = jnp.array([0.25, 0.5, 0.75])

    loss = losses.quantile_pinball_loss(y_true, y_pred, quantiles)
    assert jnp.allclose(loss, 0.0, atol=1e-6)

  def test_positive_loss(self):
    """Test loss is positive for imperfect predictions."""
    y_true = jnp.array([0.0, 1.0])
    y_pred = jnp.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
    quantiles = jnp.array([0.25, 0.5, 0.75])

    loss = losses.quantile_pinball_loss(y_true, y_pred, quantiles)
    assert loss > 0

  def test_asymmetric_penalty(self):
    """Test that loss is asymmetric for different quantiles."""
    y_true = jnp.array([0.0])
    y_pred_low = jnp.array([[-1.0]])  # Predict -1.0 when true is 0.0 (underestimate)
    y_pred_high = jnp.array([[1.0]])  # Predict 1.0 when true is 0.0 (overestimate)

    # For q=0.9, underestimating (predicting too low) should be more costly
    # error = y_true - y_pred = 0 - (-1) = 1 (positive error = underestimate)
    quantiles_high = jnp.array([0.9])
    loss_under = losses.quantile_pinball_loss(y_true, y_pred_low, quantiles_high)
    loss_over = losses.quantile_pinball_loss(y_true, y_pred_high, quantiles_high)
    assert loss_under > loss_over

    # For q=0.1, overestimating (predicting too high) should be more costly
    # error = y_true - y_pred = 0 - 1 = -1 (negative error = overestimate)
    quantiles_low = jnp.array([0.1])
    loss_under = losses.quantile_pinball_loss(y_true, y_pred_low, quantiles_low)
    loss_over = losses.quantile_pinball_loss(y_true, y_pred_high, quantiles_low)
    assert loss_over > loss_under


class TestQuantileCrossingPenalty:
  """Tests for quantile crossing penalty."""

  def test_no_crossing(self):
    """Test penalty is zero when quantiles are ordered."""
    y_pred = jnp.array([
        [-1.0, 0.0, 1.0],
        [-2.0, -1.0, 0.0],
    ])
    penalty = losses.quantile_crossing_penalty(y_pred)
    assert jnp.allclose(penalty, 0.0, atol=1e-6)

  def test_with_crossing(self):
    """Test penalty is positive when quantiles cross."""
    y_pred = jnp.array([
        [1.0, 0.0, -1.0],  # Reversed order
    ])
    penalty = losses.quantile_crossing_penalty(y_pred)
    assert penalty > 0

  def test_partial_crossing(self):
    """Test penalty for partial crossing."""
    y_pred = jnp.array([
        [-1.0, 0.5, 0.0, 1.0],  # q2 > q3 (crossing)
    ])
    penalty = losses.quantile_crossing_penalty(y_pred)
    assert penalty > 0


class TestMultiHorizonQuantileLoss:
  """Tests for multi-horizon quantile loss."""

  def test_single_horizon(self):
    """Test loss computation for single horizon."""
    targets = {1: jnp.array([0.0, 1.0])}
    predictions = {1: jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])}
    quantiles = jnp.array([0.25, 0.5, 0.75])

    total_loss, loss_dict = losses.multi_horizon_quantile_loss(
        targets, predictions, quantiles
    )

    assert total_loss >= 0
    assert 'pinball_h1' in loss_dict
    assert 'crossing_h1' in loss_dict
    assert 'total_h1' in loss_dict

  def test_multiple_horizons(self):
    """Test loss computation for multiple horizons."""
    targets = {
        1: jnp.array([0.0]),
        5: jnp.array([0.0]),
    }
    predictions = {
        1: jnp.array([[0.0, 0.0, 0.0]]),
        5: jnp.array([[0.0, 0.0, 0.0]]),
    }
    quantiles = jnp.array([0.25, 0.5, 0.75])

    total_loss, loss_dict = losses.multi_horizon_quantile_loss(
        targets, predictions, quantiles
    )

    assert 'pinball_h1' in loss_dict
    assert 'pinball_h5' in loss_dict
    assert 'total_loss' in loss_dict

  def test_horizon_weights(self):
    """Test that horizon weights affect total loss."""
    targets = {
        1: jnp.array([1.0]),
        5: jnp.array([1.0]),
    }
    predictions = {
        1: jnp.array([[0.0, 0.0, 0.0]]),
        5: jnp.array([[0.0, 0.0, 0.0]]),
    }
    quantiles = jnp.array([0.5])

    # Equal weights
    loss_equal, _ = losses.multi_horizon_quantile_loss(
        targets, predictions, quantiles,
        horizon_weights={1: 1.0, 5: 1.0}
    )

    # Different weights
    loss_weighted, _ = losses.multi_horizon_quantile_loss(
        targets, predictions, quantiles,
        horizon_weights={1: 2.0, 5: 0.5}
    )

    assert loss_equal != loss_weighted


class TestRegimeClassificationLoss:
  """Tests for regime classification loss."""

  def test_perfect_prediction(self):
    """Test loss for perfect predictions."""
    y_true = jnp.array([0, 1, 2])
    y_pred = jnp.array([
        [10.0, -10.0, -10.0],  # Class 0
        [-10.0, 10.0, -10.0],  # Class 1
        [-10.0, -10.0, 10.0],  # Class 2
    ])

    loss = losses.regime_classification_loss(y_true, y_pred)
    assert loss < 0.1  # Should be close to zero

  def test_random_prediction(self):
    """Test loss for random predictions."""
    y_true = jnp.array([0, 1, 2])
    y_pred = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    loss = losses.regime_classification_loss(y_true, y_pred)
    # For uniform predictions, loss should be -log(1/3) ≈ 1.099
    expected = -jnp.log(1.0 / 3.0)
    assert jnp.allclose(loss, expected, atol=0.01)


class TestCombinedLoss:
  """Tests for combined loss function."""

  def test_return_only(self):
    """Test combined loss with only return predictions."""
    targets = {
        'log_returns': {1: jnp.array([0.0])},
    }
    predictions = {
        'log_return_quantiles': {1: jnp.array([[0.0, 0.0, 0.0]])},
    }
    quantiles = jnp.array([0.25, 0.5, 0.75])

    total_loss, loss_dict = losses.combined_loss(
        targets, predictions, quantiles
    )

    assert total_loss >= 0
    assert 'return_total_loss' in loss_dict
    assert 'combined_total_loss' in loss_dict

  def test_all_outputs(self):
    """Test combined loss with all optional outputs."""
    targets = {
        'log_returns': {1: jnp.array([0.0])},
        'log_ranges': {1: jnp.array([0.0])},
        'log_volumes': {1: jnp.array([0.0])},
        'regime_labels': jnp.array([0]),
    }
    predictions = {
        'log_return_quantiles': {1: jnp.array([[0.0, 0.0, 0.0]])},
        'log_range_quantiles': {1: jnp.array([[0.0, 0.0, 0.0]])},
        'log_volume_quantiles': {1: jnp.array([[0.0, 0.0, 0.0]])},
        'regime_probs': jnp.array([[1.0, 0.0, 0.0]]),
    }
    quantiles = jnp.array([0.25, 0.5, 0.75])

    total_loss, loss_dict = losses.combined_loss(
        targets, predictions, quantiles
    )

    assert total_loss >= 0
    assert 'return_total_loss' in loss_dict
    assert 'range_total_loss' in loss_dict
    assert 'volume_total_loss' in loss_dict
    assert 'regime_loss' in loss_dict

  def test_custom_weights(self):
    """Test that custom weights affect total loss."""
    targets = {
        'log_returns': {1: jnp.array([1.0])},
        'log_ranges': {1: jnp.array([1.0])},
    }
    predictions = {
        'log_return_quantiles': {1: jnp.array([[0.0, 0.0, 0.0]])},
        'log_range_quantiles': {1: jnp.array([[0.0, 0.0, 0.0]])},
    }
    quantiles = jnp.array([0.5])

    # Default weights
    loss_default, _ = losses.combined_loss(
        targets, predictions, quantiles
    )

    # Custom weights
    config_custom = {
        'horizon_weights': {1: 1.0},
        'crossing_penalty_weight': 1.0,
        'range_weight': 2.0,  # Increased
        'volume_weight': 0.3,
        'regime_weight': 0.2,
    }
    loss_custom, _ = losses.combined_loss(
        targets, predictions, quantiles, config=config_custom
    )

    assert loss_default != loss_custom


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
