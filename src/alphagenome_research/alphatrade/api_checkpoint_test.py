"""Tests for API checkpoint loading."""

import pickle
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphagenome_research.alphatrade import api
from alphagenome_research.alphatrade import schemas


class TestCheckpointLoading:
  """Tests for create_service_from_checkpoint."""

  def test_load_minimal_checkpoint(self):
    """Test loading a minimal checkpoint with all required fields."""
    # Create a minimal config (use 1024 to avoid downsampling issues)
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

    # Create a service to get valid params/state
    from alphagenome_research.alphatrade import training
    train_state = training.create_train_state(
        config=config,
        rng=jax.random.PRNGKey(42),
    )

    # Create checkpoint dict
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
        "step": 1000,
        "model_version": "test_v1",
    }

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
      pickle.dump(ckpt, f)
      ckpt_path = f.name

    try:
      # Load checkpoint
      service = api.create_service_from_checkpoint(ckpt_path)

      # Verify service is created correctly
      assert service.config.d_model == 32
      assert service.config.horizons == [1, 5]
      assert service.model_version == "test_v1"
      assert service.scaler is not None
      assert len(service.scaler.medians_) == 8

      # Verify train_state structure
      assert "params" in service.train_state
      assert "state" in service.train_state
      assert "forward" in service.train_state

    finally:
      # Cleanup
      Path(ckpt_path).unlink()

  def test_load_with_feature_dict_scaler(self):
    """Test loading checkpoint with feature_i scaler format."""
    config = schemas.AlphaTradeConfig(
        d_model=32,
        num_transformer_layers=1,
        num_heads=2,
        head_dim=16,
        lookback_length=1024,
        num_features=8,
        horizons=[1],
        quantiles=[0.1, 0.5, 0.9],
    )

    from alphagenome_research.alphatrade import training
    train_state = training.create_train_state(
        config=config,
        rng=jax.random.PRNGKey(42),
    )

    # Use feature_i format for scaler
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
            "horizons": [1],
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
            "feature_0": [0.0, 1.0],
            "feature_1": [0.5, 1.5],
            "feature_2": [0.0, 1.0],
            "feature_3": [5.0, 2.0],
            "feature_4": [0.5, 0.5],
            "feature_5": [0.0, 1.0],
            "feature_6": [0.0, 1.0],
            "feature_7": [1.0, 1.0],
        },
    }

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
      pickle.dump(ckpt, f)
      ckpt_path = f.name

    try:
      service = api.create_service_from_checkpoint(ckpt_path)
      assert service.scaler is not None
      assert len(service.scaler.medians_) == 8
      assert service.scaler.medians_[1] == 0.5
      assert service.scaler.iqrs_[3] == 2.0

    finally:
      Path(ckpt_path).unlink()

  def test_missing_required_keys(self):
    """Test error handling for missing required keys."""
    ckpt = {"config": {}}  # Missing params and state

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
      pickle.dump(ckpt, f)
      ckpt_path = f.name

    try:
      with pytest.raises(ValueError, match="missing required keys"):
        api.create_service_from_checkpoint(ckpt_path)
    finally:
      Path(ckpt_path).unlink()

  def test_invalid_scaler_format(self):
    """Test error handling for invalid scaler format."""
    config = schemas.AlphaTradeConfig(
        d_model=32,
        num_transformer_layers=1,
        num_heads=2,
        head_dim=16,
        lookback_length=1024,
        num_features=8,
        horizons=[1],
        quantiles=[0.1, 0.5, 0.9],
    )

    from alphagenome_research.alphatrade import training
    train_state = training.create_train_state(
        config=config,
        rng=jax.random.PRNGKey(42),
    )

    # Invalid scaler: feature_0 missing (gap in indices)
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
            "horizons": [1],
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
            "feature_1": [0.0, 1.0],  # Missing feature_0
            "feature_2": [0.0, 1.0],
        },
    }

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
      pickle.dump(ckpt, f)
      ckpt_path = f.name

    try:
      with pytest.raises(ValueError, match="contiguous"):
        api.create_service_from_checkpoint(ckpt_path)
    finally:
      Path(ckpt_path).unlink()

  def test_numerical_consistency(self):
    """Test that loaded checkpoint has same params/state as original."""
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

    from alphagenome_research.alphatrade import training
    train_state = training.create_train_state(
        config=config,
        rng=jax.random.PRNGKey(42),
    )

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
    }

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
      pickle.dump(ckpt, f)
      ckpt_path = f.name

    try:
      # Create service from train_state
      service1 = api.create_service_from_train_state(
          train_state=train_state,
          config=config,
      )

      # Create service from checkpoint
      service2 = api.create_service_from_checkpoint(ckpt_path)

      # Verify params are identical
      def compare_trees(tree1, tree2, path=""):
        leaves1, struct1 = jax.tree_util.tree_flatten(tree1)
        leaves2, struct2 = jax.tree_util.tree_flatten(tree2)
        assert struct1 == struct2, f"Tree structures differ at {path}"
        assert len(leaves1) == len(leaves2), f"Leaf counts differ at {path}"
        for i, (l1, l2) in enumerate(zip(leaves1, leaves2)):
          np.testing.assert_allclose(l1, l2, rtol=1e-7, atol=1e-7,
                                     err_msg=f"Leaf {i} differs at {path}")

      compare_trees(service1.train_state["params"], service2.train_state["params"], "params")
      compare_trees(service1.train_state["state"], service2.train_state["state"], "state")

    finally:
      Path(ckpt_path).unlink()
