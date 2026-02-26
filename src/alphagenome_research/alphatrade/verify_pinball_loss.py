#!/usr/bin/env python3
"""验证 pinball loss 实现的正确性"""

import numpy as np
import jax.numpy as jnp

print("="*72)
print("Pinball Loss Implementation Test")
print("="*72)
print()

# Test case: simple example
y_true = jnp.array([0.0, 1.0, -1.0])
y_pred = jnp.array([
    [0.0, 0.0, 0.0],  # predict 0 for all quantiles
    [0.5, 1.0, 1.5],  # predict around true value
    [-0.5, -1.0, -1.5],  # predict around true value
])
quantiles = jnp.array([0.1, 0.5, 0.9])

print("Test case:")
print(f"  y_true: {y_true}")
print(f"  y_pred:\n{y_pred}")
print(f"  quantiles: {quantiles}")
print()

# My implementation
y_true_expanded = y_true[:, None]  # [3, 1]
q_expanded = quantiles[None, :]  # [1, 3]
residuals = y_true_expanded - y_pred  # [3, 3]

print("Residuals (y_true - y_pred):")
print(residuals)
print()

loss = jnp.where(residuals >= 0, q_expanded * residuals, (q_expanded - 1) * residuals)

print("Loss per sample per quantile:")
print(loss)
print()

loss_mean = jnp.mean(loss)
print(f"Mean loss: {float(loss_mean):.6f}")
print()

# Manual calculation for verification
print("="*72)
print("Manual Verification")
print("="*72)
print()

for i in range(3):
    print(f"Sample {i}: y_true={float(y_true[i]):.1f}")
    for j, q in enumerate([0.1, 0.5, 0.9]):
        y_p = float(y_pred[i, j])
        residual = float(y_true[i]) - y_p
        if residual >= 0:
            l = q * residual
        else:
            l = (q - 1) * residual
        print(f"  q={q:.1f}, y_pred={y_p:.1f}, residual={residual:.1f}, loss={l:.4f}")
    print()

print("="*72)
print("Test with OLS-like predictions")
print("="*72)
print()

# Load real data
npz = np.load("src/alphagenome_research/alphatrade/alphatrade_ds_jm/val.npz")
X = npz["features"].astype(np.float32)
Y_h1 = npz["y_h1"].astype(np.float32)

# OLS prediction
X_last = X[:, -1, :]
from numpy.linalg import lstsq
Xb = np.column_stack([X_last, np.ones((X_last.shape[0],))])
w = lstsq(Xb, Y_h1, rcond=None)[0]
y_ols = Xb @ w

# Use OLS prediction for all quantiles (not optimal, but for testing)
y_pred_ols = jnp.array(np.stack([y_ols, y_ols, y_ols], axis=1))
y_true_h1 = jnp.array(Y_h1)

# Compute loss
y_true_expanded = y_true_h1[:, None]
q_expanded = quantiles[None, :]
residuals = y_true_expanded - y_pred_ols
loss = jnp.where(residuals >= 0, q_expanded * residuals, (q_expanded - 1) * residuals)
loss_mean = jnp.mean(loss)

corr_ols = float(np.corrcoef(Y_h1, y_ols)[0, 1])

print(f"h=1 with OLS predictions:")
print(f"  OLS corr: {corr_ols:+.4f}")
print(f"  Pinball loss: {float(loss_mean):.6f}")
print(f"  Y std: {float(np.std(Y_h1)):.6f}")
print(f"  Pred std: {float(np.std(y_ols)):.6f}")
print()

print("="*72)
print("DIAGNOSIS")
print("="*72)
print()
print("If pinball loss implementation is correct, the issue is elsewhere.")
print("Possible causes:")
print("  1. Optimizer/learning rate not suitable for this scale")
print("  2. Need to normalize targets before training")
print("  3. Need different loss scaling per horizon")
print()
