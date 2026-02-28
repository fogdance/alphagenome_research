#!/usr/bin/env python3
"""Check if the data has any predictive signal (quick baselines).

This script:
- Loads val.npz (features + y_h*)
- Builds simple predictors from X:
  * last timestep features
  * optional summary features (mean/std over lookback)
- Computes per-feature correlation with y
- Fits OLS linear regression as a sanity baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _load_val_npz(val_path: Path) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    npz = np.load(val_path)
    if "features" not in npz.files:
        raise ValueError(f"{val_path} missing 'features' array")
    X = npz["features"].astype(np.float32)  # [N, L, F]

    Y: Dict[int, np.ndarray] = {}
    for key in npz.files:
        if key.startswith("y_h"):
            try:
                h = int(key.split("h")[-1])
            except Exception:
                continue
            Y[h] = npz[key].astype(np.float32).reshape(-1)

    if not Y:
        raise ValueError(f"{val_path} contains no y_h* arrays")

    return X, Y


def _ols_fit_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS with bias term via lstsq."""
    Xb = np.column_stack([X, np.ones((X.shape[0],), dtype=X.dtype)])
    w = np.linalg.lstsq(Xb, y, rcond=None)[0]
    return Xb @ w


def _r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=str, default="src/alphatrade/alphatrade_ds_jm/val.npz")
    ap.add_argument("--use_summary", action="store_true", help="Use (last, mean, std) features instead of only last.")
    args = ap.parse_args()

    val_path = Path(args.val)
    print(f"Loading validation data from {val_path}")

    X, Y = _load_val_npz(val_path)
    N, L, F = X.shape

    print("\nData shapes:")
    print(f"  X: {X.shape}")
    for h in sorted(Y.keys()):
        print(f"  y_h{h}: {Y[h].shape}")

    # Feature views
    X_last = X[:, -1, :]  # [N, F]
    if args.use_summary:
        X_mean = np.mean(X, axis=1)
        X_std = np.std(X, axis=1)
        X_feat = np.concatenate([X_last, X_mean, X_std], axis=1)
        feat_names = [f"last_{i}" for i in range(F)] + [f"mean_{i}" for i in range(F)] + [f"std_{i}" for i in range(F)]
        print("\nUsing feature set: concat(last, mean, std) over lookback")
    else:
        X_feat = X_last
        feat_names = [f"last_{i}" for i in range(F)]
        print("\nUsing feature set: last timestep only")

    print(f"Feature matrix: {X_feat.shape}")

    # constant-feature check
    const_feats = [feat_names[i] for i in range(X_feat.shape[1]) if float(np.std(X_feat[:, i])) == 0.0]
    if const_feats:
        print("\nWARNING: constant features detected:", const_feats)

    print(f"\n{'='*72}")
    print("SIGNAL CHECK: Feature-Target Correlations + Linear Baseline (OLS)")
    print(f"{'='*72}")

    for h in sorted(Y.keys()):
        y = Y[h]
        print(f"\nh={h}:")

        # per-feature correlation
        max_abs_corr = 0.0
        best_feat = None
        for i in range(X_feat.shape[1]):
            xi = X_feat[:, i]
            if np.std(xi) == 0:
                corr = np.nan
            else:
                corr = float(np.corrcoef(xi, y)[0, 1])
            if np.isfinite(corr) and abs(corr) > max_abs_corr:
                max_abs_corr = abs(corr)
                best_feat = feat_names[i]
            if i < 24:  # avoid printing too many when --use_summary
                print(f"  {feat_names[i]:>10}: {corr:+.4f}")
        if X_feat.shape[1] > 24:
            print(f"  ... ({X_feat.shape[1]-24} more features not shown)")

        # OLS baseline
        y_pred = _ols_fit_predict(X_feat, y)
        r2 = _r2(y, y_pred)
        corr_pred = float(np.corrcoef(y, y_pred)[0, 1]) if np.std(y_pred) > 0 else float("nan")

        print(f"  Best |corr| feature: {best_feat}  |corr|={max_abs_corr:.4f}")
        print(f"  OLS baseline: R²={r2:.6f}, corr(y, y_pred)={corr_pred:+.4f}")

    print(f"\n{'='*72}")
    print("INTERPRETATION")
    print(f"{'='*72}")
    print("If max |corr| is near 0 and OLS R² < 0.01 across horizons:")
    print("  -> data may have very weak signal (hard to learn).")
    print("If max |corr| ~ 0.1+ and OLS corr ~ 0.15+ but deep model corr ~ 0:")
    print("  -> training/pipeline/model issue likely (e.g., label alignment, loss weighting, over-regularization).")
    print()


if __name__ == "__main__":
    main()