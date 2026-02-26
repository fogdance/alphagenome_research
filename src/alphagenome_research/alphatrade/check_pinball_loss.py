#!/usr/bin/env python3
"""检查 pinball loss 的数值范围和梯度"""

import numpy as np
import jax
import jax.numpy as jnp

# Load data
npz = np.load("src/alphagenome_research/alphatrade/alphatrade_ds_jm/val.npz")

print("="*72)
print("Pinball Loss Numerical Analysis")
print("="*72)
print()

quantiles = jnp.array([0.1, 0.5, 0.9], dtype=jnp.float32)

for h in [1, 5, 20, 60]:
    Y = npz[f"y_h{h}"].astype(np.float32)

    print(f"h={h}:")
    print(f"  Y: mean={np.mean(Y):.6f}, std={np.std(Y):.6f}")
    print(f"  Y range: [{np.min(Y):.6f}, {np.max(Y):.6f}]")

    # Simulate predictions at different scales
    for pred_scale in [0.001, 0.01, 0.1, 1.0]:
        # Random predictions
        y_pred = jnp.array(np.random.randn(len(Y), 3).astype(np.float32) * pred_scale)
        y_true = jnp.array(Y)

        # Compute pinball loss
        y_true_expanded = y_true[:, None]
        q_expanded = quantiles[None, :]
        residuals = y_true_expanded - y_pred
        loss = jnp.where(residuals >= 0, q_expanded * residuals, (q_expanded - 1) * residuals)
        loss_mean = jnp.mean(loss)

        print(f"  pred_scale={pred_scale:.3f}: loss={float(loss_mean):.6f}")

    print()

print("="*72)
print("OBSERVATION")
print("="*72)
print()
print("Notice that:")
print("  - h=1 has std=0.0012 (very small)")
print("  - h=60 has std=0.0100 (8x larger)")
print()
print("If we use the same learning rate for all horizons,")
print("h=1 will have much smaller gradients and learn slower.")
print()
print("But we're seeing the OPPOSITE - h=1 has huge std_ratio!")
print("This suggests the model is learning the WRONG thing.")
print()
print("Let's check if there's a bug in the pinball loss implementation...")
