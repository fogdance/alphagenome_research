#!/usr/bin/env python3
"""JAX Linear Baseline for AlphaTrade - 验证训练环路是否正确

这个脚本用最简单的线性模型复现 OLS baseline 的效果（corr≈0.15-0.21）。
如果这个脚本能达到 OLS 水平，说明训练环路（数据加载、loss、优化器）是正确的。
如果达不到，说明有 label 对齐、归一化、或 loss 聚合的问题。

用法:
  python train_linear_baseline.py --train train.npz --val val.npz --workdir runs/linear_baseline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


def load_data(path: Path) -> tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Load features and targets from npz."""
    npz = np.load(path)

    # Find features
    if "features" in npz.files:
        X = npz["features"]
    elif "X" in npz.files:
        X = npz["X"]
    else:
        raise ValueError(f"No features found in {path}")

    # Find targets (y_h1, y_h5, etc.)
    Y = {}
    for key in npz.files:
        if key.startswith("y_h"):
            try:
                h = int(key.split("h")[-1])
                Y[h] = npz[key].astype(np.float32).reshape(-1)
            except:
                continue

    if not Y:
        raise ValueError(f"No y_h* targets found in {path}")

    return X.astype(np.float32), Y


def build_linear_model(horizons: list[int], quantiles: list[float], num_features: int):
    """Build a simple linear model: X_last -> quantile predictions per horizon.

    Architecture:
      - Input: [B, L, F] -> take last timestep -> [B, F]
      - For each horizon h:
          - Linear layer: [B, F] -> [B, Q]

    This is the simplest possible model that should match OLS performance.
    """

    def forward(features: jax.Array) -> Dict[int, jax.Array]:
        # features: [B, L, F]
        x = features[:, -1, :]  # [B, F] - last timestep only

        predictions = {}
        for h in horizons:
            # Simple linear layer for each horizon
            pred = hk.Linear(len(quantiles), name=f"linear_h{h}")(x)  # [B, Q]
            predictions[h] = pred

        return predictions

    return hk.transform(forward)


def quantile_pinball_loss(y_true: jax.Array, y_pred: jax.Array, quantiles: jax.Array) -> jax.Array:
    """Pinball loss for quantile regression.

    Args:
        y_true: [B]
        y_pred: [B, Q]
        quantiles: [Q]

    Returns:
        Scalar loss
    """
    y_true = y_true[:, None]  # [B, 1]
    quantiles = quantiles[None, :]  # [1, Q]

    residuals = y_true - y_pred  # [B, Q]
    loss = jnp.where(
        residuals >= 0,
        quantiles * residuals,
        (quantiles - 1) * residuals,
    )

    return jnp.mean(loss)


def make_train_step(forward_fn, optimizer, horizons, quantiles_array):
    """Create jitted training step."""

    @jax.jit
    def train_step(params, opt_state, rng, xb, yb_dict):
        """Single training step.

        Args:
            params: Model parameters
            opt_state: Optimizer state
            rng: Random key
            xb: Features [B, L, F]
            yb_dict: Targets {h: [B]}

        Returns:
            (params, opt_state, loss, metrics)
        """

        def loss_fn(p):
            predictions = forward_fn.apply(p, rng, xb)

            # Compute loss for each horizon
            total_loss = jnp.zeros((), dtype=jnp.float32)
            horizon_losses = {}

            for h in horizons:
                if h in yb_dict and h in predictions:
                    y_true = yb_dict[h]
                    y_pred = predictions[h]

                    h_loss = quantile_pinball_loss(y_true, y_pred, quantiles_array)
                    total_loss += h_loss
                    horizon_losses[f"h{h}"] = h_loss

            return total_loss, horizon_losses

        (loss, horizon_losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        metrics = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            **horizon_losses,
        }

        return params, opt_state, loss, metrics

    return train_step


def evaluate(forward_fn, params, rng, X, Y, horizons, quantiles_array):
    """Evaluate model on dataset.

    Returns:
        metrics: Dict with loss and correlation per horizon
    """
    # Forward pass
    predictions = forward_fn.apply(params, rng, X)

    metrics = {}

    for h in horizons:
        if h not in Y or h not in predictions:
            continue

        y_true = Y[h]
        y_pred = predictions[h]  # [N, Q]

        # Loss
        loss = quantile_pinball_loss(y_true, y_pred, quantiles_array)
        metrics[f"loss_h{h}"] = float(loss)

        # Correlation with median prediction
        q_median_idx = len(quantiles_array) // 2
        y_pred_median = y_pred[:, q_median_idx]

        corr = float(np.corrcoef(y_true, y_pred_median)[0, 1])
        metrics[f"corr_h{h}"] = corr

        # Std ratio
        std_true = float(np.std(y_true))
        std_pred = float(np.std(y_pred_median))
        metrics[f"std_ratio_h{h}"] = std_pred / (std_true + 1e-9)

    # Average metrics
    avg_loss = np.mean([v for k, v in metrics.items() if k.startswith("loss_")])
    avg_corr = np.mean([v for k, v in metrics.items() if k.startswith("corr_")])

    metrics["avg_loss"] = float(avg_loss)
    metrics["avg_corr"] = float(avg_corr)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to train.npz")
    parser.add_argument("--val", type=str, required=True, help="Path to val.npz")
    parser.add_argument("--workdir", type=str, default="runs/linear_baseline")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=500)
    args = parser.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("JAX Linear Baseline - Training Loop Validation")
    print("="*72)
    print(f"Train: {args.train}")
    print(f"Val: {args.val}")
    print(f"Workdir: {workdir}")
    print()

    # Load data
    print("Loading data...")
    X_train, Y_train = load_data(Path(args.train))
    X_val, Y_val = load_data(Path(args.val))

    N_train, L, F = X_train.shape
    N_val = X_val.shape[0]
    horizons = sorted(Y_train.keys())
    quantiles = [0.1, 0.5, 0.9]  # Simple 3-quantile setup

    print(f"Train: {N_train} samples, {L} timesteps, {F} features")
    print(f"Val: {N_val} samples")
    print(f"Horizons: {horizons}")
    print(f"Quantiles: {quantiles}")
    print()

    # Build model
    print("Building linear model...")
    forward_fn = build_linear_model(horizons, quantiles, F)

    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    dummy_input = jnp.zeros((1, L, F))
    params = forward_fn.init(init_rng, dummy_input)

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {param_count:,}")
    print()

    # Optimizer
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    # Training step
    quantiles_array = jnp.array(quantiles, dtype=jnp.float32)
    train_step = make_train_step(forward_fn, optimizer, horizons, quantiles_array)

    # Training loop
    print("Training...")
    print(f"{'Step':<8} {'Loss':<10} {'GradNorm':<10} {'ValCorr':<10} {'Time':<8}")
    print("-"*72)

    best_corr = -999.0

    for step in range(1, args.steps + 1):
        t0 = time.time()

        # Sample batch
        rng, batch_rng = jax.random.split(rng)
        indices = jax.random.choice(batch_rng, N_train, shape=(args.batch_size,), replace=False)

        xb = X_train[indices]
        yb_dict = {h: Y_train[h][indices] for h in horizons}

        # Train step
        rng, step_rng = jax.random.split(rng)
        params, opt_state, loss, metrics = train_step(params, opt_state, step_rng, xb, yb_dict)

        t1 = time.time()

        # Evaluate
        if step % args.eval_every == 0 or step == 1:
            rng, eval_rng = jax.random.split(rng)
            val_metrics = evaluate(forward_fn, params, eval_rng, X_val, Y_val, horizons, quantiles_array)

            val_corr = val_metrics["avg_corr"]

            print(f"{step:<8} {float(loss):<10.6f} {float(metrics['grad_norm']):<10.4f} {val_corr:<10.4f} {(t1-t0)*1000:<8.1f}ms")

            # Save best
            if val_corr > best_corr:
                best_corr = val_corr

                # Save checkpoint
                ckpt = {
                    "params": params,
                    "step": step,
                    "val_metrics": val_metrics,
                }

                import pickle
                with open(workdir / "best.pkl", "wb") as f:
                    pickle.dump(ckpt, f)

                # Save metrics
                with open(workdir / "best_metrics.json", "w") as f:
                    json.dump(val_metrics, f, indent=2)

    print()
    print("="*72)
    print("Final Evaluation")
    print("="*72)

    rng, eval_rng = jax.random.split(rng)
    final_metrics = evaluate(forward_fn, params, eval_rng, X_val, Y_val, horizons, quantiles_array)

    print("\nValidation Metrics:")
    for h in horizons:
        corr = final_metrics.get(f"corr_h{h}", 0.0)
        std_ratio = final_metrics.get(f"std_ratio_h{h}", 0.0)
        print(f"  h={h:>2}: corr={corr:+.4f}, std_ratio={std_ratio:.4f}")

    print(f"\nAverage correlation: {final_metrics['avg_corr']:.4f}")
    print(f"Best correlation: {best_corr:.4f}")

    print()
    print("="*72)
    print("INTERPRETATION")
    print("="*72)
    print("Expected: corr ≈ 0.15-0.21 (matching OLS baseline)")
    print()
    print("If corr < 0.10:")
    print("  ❌ Training loop has issues (label alignment, loss, optimizer)")
    print()
    print("If corr ≈ 0.15-0.21:")
    print("  ✅ Training loop is correct!")
    print("  ✅ Deep model failure is due to architecture/capacity/overfitting")
    print()


if __name__ == "__main__":
    main()
