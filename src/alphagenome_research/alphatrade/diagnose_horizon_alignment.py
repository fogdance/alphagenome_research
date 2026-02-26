#!/usr/bin/env python3
"""诊断线性模型的 horizon 对齐问题"""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
from pathlib import Path

# Load checkpoint
ckpt_path = Path("src/alphagenome_research/alphatrade/runs/linear_overfit_v2/best.pkl")
with open(ckpt_path, "rb") as f:
    ckpt = pickle.load(f)

params = ckpt["params"]

# Load data
npz = np.load("src/alphagenome_research/alphatrade/alphatrade_ds_jm/val.npz")
X = npz["features"].astype(np.float32)
Y = {h: npz[f"y_h{h}"].astype(np.float32) for h in [1, 5, 20, 60]}

# Rebuild model
horizons = [1, 5, 20, 60]
quantiles = [0.1, 0.5, 0.9]

def forward(features):
    x = features[:, -1, :]
    predictions = {}
    for h in horizons:
        pred = hk.Linear(len(quantiles), name=f"linear_h{h}")(x)
        predictions[h] = pred
    return predictions

forward_fn = hk.transform(forward)

# Forward pass
rng = jax.random.PRNGKey(42)
predictions = forward_fn.apply(params, rng, X)

print("="*72)
print("Horizon Alignment Diagnostic")
print("="*72)
print()

# Check each horizon
for h in horizons:
    y_true = Y[h]
    y_pred = predictions[h][:, 1]  # median (q50)

    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))

    print(f"h={h}:")
    print(f"  y_true: mean={np.mean(y_true):+.6f}, std={std_true:.6f}")
    print(f"  y_pred: mean={np.mean(y_pred):+.6f}, std={std_pred:.6f}")
    print(f"  corr={corr:+.4f}, std_ratio={std_pred/std_true:.2f}")

    # Cross-correlation with other horizons
    print(f"  Cross-correlations:")
    for h2 in horizons:
        if h2 != h:
            y_true2 = Y[h2]
            cross_corr = float(np.corrcoef(y_pred, y_true2)[0, 1])
            print(f"    pred_h{h} vs true_h{h2}: {cross_corr:+.4f}")
    print()

print("="*72)
print("INTERPRETATION")
print("="*72)
print("If pred_h1 has higher corr with true_h5 than true_h1:")
print("  -> Horizon mapping is wrong!")
print()
print("If pred_h1 has correct alignment but very high std_ratio:")
print("  -> Loss weighting issue (h=1 getting too much gradient)")
print()