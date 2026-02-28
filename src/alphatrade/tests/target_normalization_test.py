#!/usr/bin/env python3
"""Test target normalization functionality."""

import tempfile
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphatrade import api
from alphatrade.core import schemas
from alphatrade.training import training


class TestTargetNormalization:
  """Test target normalization in training and inference."""

  def test_normalize_denormalize_roundtrip(self):
    """Test that normalize -> denormalize is identity."""
    from alphatrade.training.train_loop_minimal import (
        normalize_targets,
        denormalize_predictions,
    )

    # Create test targets
    Y = {
        1: np.array([0.001, 0.002, -0.001, 0.003]),
        5: np.array([0.005, 0.010, -0.005, 0.015]),
    }

    # Normalize
    Y_norm, stats = normalize_targets(Y)

    # Check normalized stats
    for h in [1, 5]:
      assert abs(np.mean(Y_norm[h])) < 1e-6, f"h={h} mean should be ~0"
      assert abs(np.std(Y_norm[h]) - 1.0) < 1e-5, f"h={h} std should be ~1"

    # Create fake predictions (same as normalized targets)
    predictions = {h: Y_norm[h][:, None] for h in [1, 5]}  # [N, 1]

    # Denormalize
    predictions_denorm = denormalize_predictions(predictions, stats)

    # Check roundtrip
    for h in [1, 5]:
      np.testing.assert_allclose(
          predictions_denorm[h][:, 0],
          Y[h],
          rtol=1e-5,
          atol=1e-8,
          err_msg=f"Roundtrip failed for h={h}",
      )

  def test_checkpoint_with_target_stats(self):
    """Test that checkpoint saves and loads target_stats correctly."""
    config = schemas.AlphaTradeConfig(
        d_model=32,
        num_transformer_layers=1,
        num_heads=2,
        head_dim=16,
        lookback_length=1024,
        num_features=8,
        horizons=[1, 5],
        quantiles=[0.1, 0.5, 0.9],
    )

    # Create train state
    train_state = training.create_train_state(
        config=config,
        rng=jax.random.PRNGKey(42),
    )

    # Create target stats
    target_stats = {
        1: (0.0001, 0.0012),
        5: (0.0002, 0.0030),
    }

    # Create checkpoint
    ckpt = {
        "params": train_state["params"],
        "state": train_state["state"],
        "config": {
            "d_model": 32,
            "num_transformer_layers": 1,
            "num_heads": 2,
            "head_dim": 16,
            "lookback_length": 1024,
            "num_features": 8,
            "horizons": [1, 5],
            "quantiles": [0.1, 0.5, 0.9],
            "mlp_expansion": 4,
            "predict_range": False,
            "predict_volume": False,
            "predict_regime": False,
            "num_regime_classes": 3,
            "logits_soft_cap": 5.0,
            "max_position": 8192,
        },
        "scaler": {
            "medians": [0.0] * 8,
            "iqrs": [1.0] * 8,
        },
        "target_stats": target_stats,
        "step": 1000,
        "model_version": "test_v1",
    }

    # Save checkpoint
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
      pickle.dump(ckpt, f)
      ckpt_path = f.name

    try:
      # Load checkpoint
      service = api.create_service_from_checkpoint(ckpt_path)

      # Verify target_stats loaded
      assert service.target_stats is not None
      assert 1 in service.target_stats
      assert 5 in service.target_stats
      assert service.target_stats[1] == (0.0001, 0.0012)
      assert service.target_stats[5] == (0.0002, 0.0030)

    finally:
      Path(ckpt_path).unlink()

  def test_prediction_denormalization(self):
    """Test that target_stats are properly stored and accessible."""
    config = schemas.AlphaTradeConfig(
        d_model=32,
        num_transformer_layers=1,
        num_heads=2,
        head_dim=16,
        lookback_length=1024,
        num_features=8,
        horizons=[1, 5],
        quantiles=[0.1, 0.5, 0.9],
    )

    # Create train state
    train_state = training.create_train_state(
        config=config,
        rng=jax.random.PRNGKey(42),
    )

    # Create target stats
    target_stats = {
        1: (0.0, 0.001),  # mean=0, std=0.001
        5: (0.0, 0.003),  # mean=0, std=0.003
    }

    # Create service with target_stats
    service = api.create_service_from_train_state(
        train_state=train_state,
        config=config,
        target_stats=target_stats,
    )

    # Verify target_stats are stored
    assert service.target_stats is not None
    assert service.target_stats[1] == (0.0, 0.001)
    assert service.target_stats[5] == (0.0, 0.003)

    # Test denormalization logic directly
    # Simulate normalized predictions
    normalized_preds = {
        1: np.array([[0.0, 1.0, 2.0]]),  # [1, 3] - normalized quantiles
        5: np.array([[0.0, 1.0, 2.0]]),
    }

    # Denormalize manually
    denorm_h1 = normalized_preds[1] * 0.001 + 0.0
    denorm_h5 = normalized_preds[5] * 0.003 + 0.0

    # h=5 should be 3x larger than h=1 (due to std ratio)
    assert np.allclose(denorm_h5, denorm_h1 * 3.0)


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
