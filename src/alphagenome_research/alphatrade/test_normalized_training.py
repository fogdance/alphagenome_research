#!/usr/bin/env python3
"""带目标归一化的线性 baseline"""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

# Load data
npz = np.load("src/alphagenome_research/alphatrade/alphatrade_ds_jm/val.npz")
X = npz["features"].astype(np.float32)

print("="*72)
print("Normalized Target Training Test")
print("="*72)
print()

results = {}

for test_horizon in [1, 5, 20, 60]:
    print(f"\n{'='*72}")
    print(f"Training h={test_horizon} with target normalization")
    print(f"{'='*72}")

    Y = npz[f"y_h{test_horizon}"].astype(np.float32)

    # Normalize targets
    y_mean = np.mean(Y)
    y_std = np.std(Y)
    Y_norm = (Y - y_mean) / (y_std + 1e-8)

    print(f"Original Y: mean={y_mean:.6f}, std={y_std:.6f}")
    print(f"Normalized Y: mean={np.mean(Y_norm):.6f}, std={np.std(Y_norm):.6f}")

    # Simple model
    def forward(features):
        x = features[:, -1, :]
        return hk.Linear(3, name="linear")(x)

    forward_fn = hk.transform(forward)

    # Init
    rng = jax.random.PRNGKey(42 + test_horizon)
    params = forward_fn.init(rng, X[:1])

    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    quantiles = jnp.array([0.1, 0.5, 0.9], dtype=jnp.float32)

    @jax.jit
    def train_step(params, opt_state, rng, xb, yb):
        def loss_fn(p):
            y_pred = forward_fn.apply(p, rng, xb)

            # Pinball loss on normalized targets
            y_true = yb[:, None]
            q = quantiles[None, :]
            residuals = y_true - y_pred
            loss = jnp.where(residuals >= 0, q * residuals, (q - 1) * residuals)
            return jnp.mean(loss)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Train
    N = X.shape[0]
    batch_size = 64

    for step in range(3000):
        rng, batch_rng = jax.random.split(rng)
        indices = jax.random.choice(batch_rng, N, shape=(batch_size,), replace=False)

        xb = X[indices]
        yb = Y_norm[indices]

        params, opt_state, loss = train_step(params, opt_state, rng, xb, yb)

    # Evaluate on normalized scale
    rng, eval_rng = jax.random.split(rng)
    y_pred_norm = forward_fn.apply(params, eval_rng, X)
    y_pred_norm_median = y_pred_norm[:, 1]

    # Denormalize predictions
    y_pred_median = y_pred_norm_median * y_std + y_mean

    corr = float(np.corrcoef(Y, y_pred_median)[0, 1])
    std_ratio = float(np.std(y_pred_median) / y_std)

    print(f"JAX (normalized): corr={corr:+.4f}, std_ratio={std_ratio:.2f}, loss={float(loss):.6f}")

    # Compare with OLS
    X_last = X[:, -1, :]
    from numpy.linalg import lstsq
    Xb = np.column_stack([X_last, np.ones((X_last.shape[0],))])
    w = lstsq(Xb, Y, rcond=None)[0]
    y_ols = Xb @ w
    corr_ols = float(np.corrcoef(Y, y_ols)[0, 1])

    print(f"OLS:              corr={corr_ols:+.4f}")
    print(f"Gap:              {corr - corr_ols:+.4f}")

    results[test_horizon] = {
        'jax_corr': corr,
        'ols_corr': corr_ols,
        'std_ratio': std_ratio,
    }

print()
print("="*72)
print("Summary")
print("="*72)
print()
print(f"{'Horizon':<10} {'JAX corr':<12} {'OLS corr':<12} {'Gap':<12} {'std_ratio':<12}")
print("-"*72)
for h in [1, 5, 20, 60]:
    r = results[h]
    gap = r['jax_corr'] - r['ols_corr']
    print(f"h={h:<8} {r['jax_corr']:+.4f}       {r['ols_corr']:+.4f}       {gap:+.4f}       {r['std_ratio']:.2f}")

print()
print("="*72)
print("INTERPRETATION")
print("="*72)
print()
print("If normalization fixes the problem:")
print("  ✅ The deep model needs target normalization!")
print()
print("If still not matching OLS:")
print("  ❌ There's a deeper issue with the training loop")
print()
