#!/usr/bin/env python3
"""
M3 AlphaTrade v0.2 Forward Smoke Test

Tests that AlphaTrade v0.2 (JAX) can:
1. Initialize with M2 config (lookback=60, features=8)
2. Forward pass with correct input shape [B, L, 8]
3. Output correct quantile predictions shape [B, Q] per horizon
4. No NaN/Inf in outputs
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphatrade.core import model as model_lib
from alphatrade.core import schemas
from alphatrade.training import training


def test_forward_smoke():
    """Smoke test for AlphaTrade v0.2 forward pass."""

    print("=" * 60)
    print("M3 AlphaTrade v0.2 Forward Smoke Test")
    print("=" * 60)

    # 1. Create config matching M2 dataset
    print("\n[1/5] Creating AlphaTrade config...")
    config = schemas.AlphaTradeConfig(
        lookback_length=60,  # M2 lookback
        num_features=8,      # M2 feature_dim
        horizons=[1, 5, 20, 60],  # M2 horizons
        quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],  # M2 quantiles (must be odd, contain 0.5)
        stem_channels=128,
        num_encoder_stages=6,
        d_model=512,
        num_transformer_layers=6,
    )

    print(f"  ✓ Config created")
    print(f"    - lookback_length: {config.lookback_length}")
    print(f"    - num_features: {config.num_features}")
    print(f"    - horizons: {config.horizons}")
    print(f"    - quantiles: {config.quantiles}")

    # 2. Create dummy batch
    print("\n[2/5] Creating dummy batch...")
    batch_size = 4
    features = jnp.zeros((batch_size, config.lookback_length, config.num_features), dtype=jnp.float32)
    print(f"  ✓ Batch shape: {features.shape} (dtype: {features.dtype})")

    # 3. Initialize model
    print("\n[3/5] Initializing model...")
    rng = jax.random.PRNGKey(42)

    # Create train state (initializes params)
    train_state = training.create_train_state(config, rng, learning_rate=1e-4)

    # Count parameters
    def count_params(pytree):
        return sum(x.size for x in jax.tree_util.tree_leaves(pytree))

    total_params = count_params(train_state['params'])
    print(f"  ✓ Model initialized")
    print(f"    - Total params: {total_params:,}")
    print(f"    - Backend: JAX")

    # 4. Forward pass
    print("\n[4/5] Running forward pass...")
    forward_fn = train_state['forward']

    output, state = forward_fn.apply(
        train_state['params'],
        train_state['state'],
        rng,
        features
    )

    print(f"  ✓ Forward pass successful")
    print(f"    - Output type: {type(output).__name__}")

    # 5. Validate outputs
    print("\n[5/5] Validating outputs...")

    # Check log_return_quantiles
    assert hasattr(output, 'log_return_quantiles'), "Missing log_return_quantiles"

    all_valid = True
    for horizon in config.horizons:
        if horizon not in output.log_return_quantiles:
            print(f"  ✗ Missing horizon {horizon}")
            all_valid = False
            continue

        pred = output.log_return_quantiles[horizon]
        expected_shape = (batch_size, len(config.quantiles))

        if pred.shape != expected_shape:
            print(f"  ✗ h{horizon}: shape {pred.shape}, expected {expected_shape}")
            all_valid = False
            continue

        # Check for NaN/Inf
        has_nan = jnp.any(jnp.isnan(pred))
        has_inf = jnp.any(jnp.isinf(pred))

        if has_nan or has_inf:
            print(f"  ✗ h{horizon}: NaN={has_nan}, Inf={has_inf}")
            all_valid = False
            continue

        # Check quantile ordering (should be monotonic)
        diffs = pred[:, 1:] - pred[:, :-1]
        min_diff = jnp.min(diffs)

        print(f"  ✓ h{horizon}: shape={pred.shape}, dtype={pred.dtype}, "
              f"range=[{jnp.min(pred):.6f}, {jnp.max(pred):.6f}], "
              f"min_diff={min_diff:.6f}")

    # Summary
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ SMOKE TEST PASSED")
        print("=" * 60)
        print("\nAlphaTrade v0.2 is ready for M3 training:")
        print("  - Input: [B, 60, 8] float32")
        print("  - Output: quantiles [B, 5] per horizon")
        print("  - No NaN/Inf detected")
        print(f"  - Total params: {total_params:,}")
        return 0
    else:
        print("❌ SMOKE TEST FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = test_forward_smoke()
    sys.exit(exit_code)
