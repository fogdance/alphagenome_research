#!/usr/bin/env python3
"""Check if the data has any predictive signal."""

import numpy as np
from pathlib import Path

def main():
    # Load validation data
    val_path = Path("src/alphagenome_research/alphatrade/alphatrade_ds_jm/val.npz")
    print(f"Loading validation data from {val_path}")

    npz = np.load(val_path)
    X = npz["features"].astype(np.float32)  # [N, L, 8]

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

    # Use last timestep features as predictors
    X_last = X[:, -1, :]  # [N, 8]

    print(f"\n{'='*60}")
    print("SIGNAL CHECK: Feature-Target Correlations")
    print(f"{'='*60}")

    for h in sorted(Y.keys()):
        y = Y[h]
        print(f"\nh={h}:")

        # Compute correlation for each feature
        max_corr = 0.0
        for i in range(8):
            corr = np.corrcoef(X_last[:, i], y)[0, 1]
            print(f"  feature_{i}: {corr:+.4f}")
            max_corr = max(max_corr, abs(corr))

        # Simple linear regression: y = X @ w + b
        # Add bias term
        X_with_bias = np.column_stack([X_last, np.ones(len(X_last))])

        # Solve: w = (X^T X)^{-1} X^T y
        w = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
        y_pred = X_with_bias @ w

        # Compute R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        corr_pred = np.corrcoef(y, y_pred)[0, 1]

        print(f"  Linear model: R²={r2:.6f}, corr={corr_pred:+.4f}, max_|corr|={max_corr:.4f}")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print("\nIf all correlations are near 0 and R² < 0.01:")
    print("  → Data has NO predictive signal")
    print("  → Model cannot learn anything useful")
    print("  → This explains the calibration failure")
    print("\nIf correlations are significant (|corr| > 0.1) and R² > 0.01:")
    print("  → Data has signal, but model is not learning it")
    print("  → Need to debug model architecture or training")

if __name__ == "__main__":
    main()
