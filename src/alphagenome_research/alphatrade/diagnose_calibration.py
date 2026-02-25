#!/usr/bin/env python3
"""Diagnose calibration issues in AlphaTrade predictions."""

import numpy as np
import pickle
import jax
import jax.numpy as jnp
import haiku as hk
from pathlib import Path

from alphagenome_research.alphatrade import model as model_lib
from alphagenome_research.alphatrade import schemas

def main():
    # Load checkpoint
    ckpt_path = Path("src/alphagenome_research/alphatrade/runs/jm_v02/best.pkl")
    print(f"Loading checkpoint from {ckpt_path}")

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    config = schemas.AlphaTradeConfig(**ckpt["config"])
    params = ckpt["params"]
    state = ckpt.get("state", {})

    med = np.array(ckpt["scaler"]["medians"], np.float32)
    iqr = np.array(ckpt["scaler"]["iqrs"], np.float32)

    print(f"\nConfig:")
    print(f"  Horizons: {config.horizons}")
    print(f"  Quantiles: {config.quantiles}")

    # Load validation data
    val_path = Path("src/alphagenome_research/alphatrade/alphatrade_ds_jm/val.npz")
    print(f"\nLoading validation data from {val_path}")

    npz = np.load(val_path)
    X = npz["features"].astype(np.float32)

    # Extract y for each horizon
    Y = {}
    for key in npz.files:
        if key.startswith("y_h"):
            h = int(key.split("h")[-1])
            Y[h] = npz[key].astype(np.float32)

    print(f"\nData shapes:")
    print(f"  X: {X.shape}")
    for h in sorted(Y.keys()):
        print(f"  y_h{h}: {Y[h].shape}")

    # Normalize X
    Xn = (X - med[None, None, :]) / iqr[None, None, :]

    # Create prediction function
    def forward(features):
        m = model_lib.AlphaTrade(config)
        return m(features)

    pred_t = hk.transform_with_state(forward)

    def _pred_apply(params, state, rng, features):
        out, _ = pred_t.apply(params, state, rng, features)
        return out.log_return_quantiles

    pred_apply_jit = jax.jit(_pred_apply)

    # Run predictions in batches
    print(f"\nRunning predictions...")
    batch_size = 64
    N = len(Xn)

    pred_q = {h: [] for h in config.horizons}
    rng = jax.random.PRNGKey(0)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        xb = jnp.asarray(Xn[i:end])

        rng, rk = jax.random.split(rng)
        out = pred_apply_jit(params, state, rk, xb)

        for h in config.horizons:
            pred_q[h].append(np.array(out[h]))

    for h in config.horizons:
        pred_q[h] = np.concatenate(pred_q[h], axis=0)  # [N, Q]

    print(f"\nPrediction shapes:")
    for h in config.horizons:
        print(f"  pred_h{h}: {pred_q[h].shape}")

    # Diagnostic 1: Check quantile crossing
    print(f"\n{'='*60}")
    print("DIAGNOSTIC 1: Quantile Crossing Rate")
    print(f"{'='*60}")

    for h in config.horizons:
        q = pred_q[h]
        # Check if quantiles are monotonically increasing
        diffs = np.diff(q, axis=1)  # [N, Q-1]
        crossing = np.mean(np.any(diffs < 0, axis=1))
        print(f"h={h:>3} crossing_rate={crossing:.4f} (should be ~0)")

    # Diagnostic 2: Bias and scale
    print(f"\n{'='*60}")
    print("DIAGNOSTIC 2: Bias and Scale")
    print(f"{'='*60}")

    quantiles_np = np.array(config.quantiles, dtype=np.float32)

    for h in config.horizons:
        y = Y[h]
        q = pred_q[h]

        # Compute coverage (empirical CDF at predicted quantiles)
        obs = [(y <= q[:, j]).mean() for j in range(q.shape[1])]

        # Bias: y - median_pred
        median_idx = len(config.quantiles) // 2
        bias = float(np.mean(y - q[:, median_idx]))

        # Scale: IQR
        iqr_y = float(np.quantile(y, 0.75) - np.quantile(y, 0.25))

        # Find q25 and q75 indices
        q25_idx = None
        q75_idx = None
        for i, qv in enumerate(config.quantiles):
            if abs(qv - 0.25) < 0.01:
                q25_idx = i
            if abs(qv - 0.75) < 0.01:
                q75_idx = i

        if q25_idx is not None and q75_idx is not None:
            iqr_pred = float(np.mean(q[:, q75_idx] - q[:, q25_idx]))
        else:
            iqr_pred = float('nan')

        print(f"\nh={h:>3}")
        print(f"  Coverage: {[f'{v:.3f}' for v in obs]}")
        print(f"  Expected: {[f'{v:.3f}' for v in config.quantiles]}")
        print(f"  Bias (y - q50): {bias:+.6f}")
        print(f"  IQR_y:          {iqr_y:.6f}")
        print(f"  IQR_pred:       {iqr_pred:.6f}")
        print(f"  Scale ratio:    {iqr_pred / iqr_y:.3f}")

    # Diagnostic 3: Distribution statistics
    print(f"\n{'='*60}")
    print("DIAGNOSTIC 3: Distribution Statistics")
    print(f"{'='*60}")

    for h in config.horizons:
        y = Y[h]
        q = pred_q[h]

        print(f"\nh={h:>3}")
        print(f"  y_true:  mean={np.mean(y):+.6f}, std={np.std(y):.6f}, "
              f"min={np.min(y):+.6f}, max={np.max(y):+.6f}")

        median_idx = len(config.quantiles) // 2
        q_median = q[:, median_idx]
        print(f"  q50:     mean={np.mean(q_median):+.6f}, std={np.std(q_median):.6f}, "
              f"min={np.min(q_median):+.6f}, max={np.max(q_median):+.6f}")

    # Diagnostic 4: Check if horizons are swapped
    print(f"\n{'='*60}")
    print("DIAGNOSTIC 4: Horizon Alignment Check")
    print(f"{'='*60}")
    print("\nIf horizons are swapped, you'll see:")
    print("  - Short horizons (h=1,5) have predictions matching long horizon targets")
    print("  - Long horizons (h=20,60) have predictions matching short horizon targets")

    for h_pred in config.horizons:
        q = pred_q[h_pred]
        median_idx = len(config.quantiles) // 2
        q_median = q[:, median_idx]

        print(f"\nPredictions for h={h_pred}:")
        for h_true in config.horizons:
            y = Y[h_true]
            corr = np.corrcoef(y, q_median)[0, 1]
            mae = np.mean(np.abs(y - q_median))
            print(f"  vs y_h{h_true}: corr={corr:+.4f}, MAE={mae:.6f}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("\nLook for:")
    print("1. High crossing_rate → QuantileHead implementation issue")
    print("2. Large bias → predictions systematically too high/low")
    print("3. Scale ratio far from 1.0 → predictions too narrow/wide")
    print("4. Coverage far from expected → calibration failure")
    print("5. Best correlation on wrong horizon → horizon mapping bug")

if __name__ == "__main__":
    main()
