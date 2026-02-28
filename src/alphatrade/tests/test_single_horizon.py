#!/usr/bin/env python3
"""单 horizon 训练测试 - 隔离问题"""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

# Load data
npz = np.load("src/alphatrade/alphatrade_ds_jm/val.npz")
X = npz["features"].astype(np.float32)

print("="*72)
print("Single-Horizon Training Test")
print("="*72)
print()

for test_horizon in [1, 5, 20, 60]:
    print(f"\n{'='*72}")
    print(f"Training ONLY h={test_horizon}")
    print(f"{'='*72}")

    Y = npz[f"y_h{test_horizon}"].astype(np.float32)

    print(f"Data: X={X.shape}, Y={Y.shape}")
    print(f"Y stats: mean={np.mean(Y):.6f}, std={np.std(Y):.6f}")

    # Simple model
    def forward(features):
        x = features[:, -1, :]  # [B, F]
        return hk.Linear(3, name="linear")(x)  # [B, 3] for 3 quantiles

    forward_fn = hk.transform(forward)

    # Init
    rng = jax.random.PRNGKey(42 + test_horizon)
    params = forward_fn.init(rng, X[:1])

    # Optimizer
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)

    quantiles = jnp.array([0.1, 0.5, 0.9], dtype=jnp.float32)

    @jax.jit
    def train_step(params, opt_state, rng, xb, yb):
        def loss_fn(p):
            y_pred = forward_fn.apply(p, rng, xb)

            # Pinball loss
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
        yb = Y[indices]

        params, opt_state, loss = train_step(params, opt_state, rng, xb, yb)

    # Evaluate
    rng, eval_rng = jax.random.split(rng)
    y_pred = forward_fn.apply(params, eval_rng, X)
    y_pred_median = y_pred[:, 1]  # q50

    corr = float(np.corrcoef(Y, y_pred_median)[0, 1])
    std_ratio = float(np.std(y_pred_median) / np.std(Y))

    print(f"Final: corr={corr:+.4f}, std_ratio={std_ratio:.2f}, loss={float(loss):.6f}")

    # Compare with OLS
    X_last = X[:, -1, :]
    from numpy.linalg import lstsq
    Xb = np.column_stack([X_last, np.ones((X_last.shape[0],))])
    w = lstsq(Xb, Y, rcond=None)[0]
    y_ols = Xb @ w
    corr_ols = float(np.corrcoef(Y, y_ols)[0, 1])

    print(f"OLS:   corr={corr_ols:+.4f}")
    print(f"Gap:   {corr - corr_ols:+.4f}")

print()
print("="*72)
print("INTERPRETATION")
print("="*72)
print("If single-horizon training matches OLS for all horizons:")
print("  -> Multi-horizon loss weighting is the problem")
print()
print("If single-horizon training still fails for h=1 or h=20:")
print("  -> Data/label issue for those specific horizons")
print()
