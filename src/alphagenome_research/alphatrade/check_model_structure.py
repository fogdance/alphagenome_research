#!/usr/bin/env python3
"""检查模型输出和 loss 计算中的 horizon 顺序"""

import jax
import jax.numpy as jnp
import haiku as hk

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

# Initialize
rng = jax.random.PRNGKey(42)
dummy_input = jnp.zeros((2, 100, 8))
params = forward_fn.init(rng, dummy_input)

# Check parameter structure
print("="*72)
print("Model Parameter Structure")
print("="*72)
print()
print("Params keys:")
for key in params.keys():
    print(f"  {key}")
    if isinstance(params[key], dict):
        for subkey in params[key].keys():
            print(f"    {subkey}")
print()

# Forward pass
predictions = forward_fn.apply(params, rng, dummy_input)

print("="*72)
print("Model Output Structure")
print("="*72)
print()
print("Predictions dict:")
for h in predictions.keys():
    print(f"  h={h}: shape={predictions[h].shape}")
print()

# Simulate loss computation
print("="*72)
print("Loss Computation Order")
print("="*72)
print()

yb_dict = {h: jnp.ones((2,)) * h for h in horizons}  # Use horizon value as dummy target

print("Target dict (yb_dict):")
for h in yb_dict.keys():
    print(f"  h={h}: value={yb_dict[h][0]}")
print()

print("Loss computation loop:")
for h in horizons:
    if h in yb_dict and h in predictions:
        print(f"  Computing loss for h={h}")
        print(f"    y_true from yb_dict[{h}] = {yb_dict[h][0]}")
        print(f"    y_pred from predictions[{h}]")
print()

print("="*72)
print("DIAGNOSIS")
print("="*72)
print()
print("The model creates separate Linear layers for each horizon:")
for h in horizons:
    print(f"  linear_h{h} -> predictions[{h}]")
print()
print("The loss loop matches predictions[h] with yb_dict[h].")
print()
print("This should be correct IF:")
print("  1. yb_dict is constructed correctly from npz")
print("  2. Dict iteration order is consistent")
print()
print("Let's check the data loading...")
