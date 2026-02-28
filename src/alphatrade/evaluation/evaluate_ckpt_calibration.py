#!/usr/bin/env python3
"""
Evaluate AlphaTrade checkpoint on val.npz with calibration diagnostics.

What it does:
- Load ckpt (params/state/config/scaler/target_stats)
- Load val.npz (features + y_h*)
- Apply feature normalization (robust scaler from ckpt)
- Run model to get quantile predictions
- If target_stats present, de-normalize predictions back to original y space
- Report:
  * q50 corr
  * std ratio (pred q50 std / y std)
  * quantile crossing rate
  * coverage per quantile
  * bias, scale ratio (IQR_pred / IQR_y)
  * horizon alignment matrix (q50 vs each y_h*)

Usage:
  python src/alphatrade/evaluate_ckpt_calibration.py \
    --ckpt src/alphatrade/runs/jm_v03/best.pkl \
    --val_npz src/alphatrade/alphatrade_ds_jm/val.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphatrade.core import model as model_lib
from alphatrade.core import schemas


def _as_config(cfg_obj: Any) -> schemas.AlphaTradeConfig:
    if isinstance(cfg_obj, schemas.AlphaTradeConfig):
        return cfg_obj
    if isinstance(cfg_obj, dict):
        return schemas.AlphaTradeConfig(**cfg_obj)
    raise ValueError(f"Unsupported config type in ckpt: {type(cfg_obj)}")


def _load_val_npz(val_path: Path) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    npz = np.load(val_path)
    if "features" not in npz.files:
        raise ValueError(f"{val_path} missing 'features' array")
    X = np.asarray(npz["features"], dtype=np.float32)  # [N, L, F]

    Y: Dict[int, np.ndarray] = {}
    for k in npz.files:
        if k.startswith("y_h"):
            try:
                h = int(k.split("h")[-1])
            except Exception:
                continue
            Y[h] = np.asarray(npz[k], dtype=np.float32).reshape(-1)

    if not Y:
        raise ValueError(f"{val_path} contains no y_h* arrays")

    return X, Y


def _parse_scaler(scaler_obj: Any, num_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supports:
      scaler = {"medians":[...], "iqrs":[...]}
      scaler = {"feature_0":[m,iqr], ...}
    Returns (medians, iqrs) float32 arrays of length num_features.
    """
    if scaler_obj is None:
        raise ValueError("ckpt missing 'scaler' (needed to normalize features)")
    if not isinstance(scaler_obj, dict):
        raise ValueError(f"Unsupported scaler type: {type(scaler_obj)}")

    if "medians" in scaler_obj and "iqrs" in scaler_obj:
        med = np.asarray(scaler_obj["medians"], dtype=np.float32)
        iqr = np.asarray(scaler_obj["iqrs"], dtype=np.float32)
    else:
        # feature_i format
        feat_keys = [k for k in scaler_obj.keys() if k.startswith("feature_")]
        if not feat_keys:
            raise ValueError("Unrecognized scaler dict format (no medians/iqrs and no feature_*)")
        # require contiguous 0..n-1
        idx = sorted(int(k.split("_")[1]) for k in feat_keys)
        if idx != list(range(len(idx))):
            raise ValueError(f"Scaler feature indices not contiguous: {idx[:20]} ...")
        med = np.asarray([scaler_obj[f"feature_{i}"][0] for i in idx], dtype=np.float32)
        iqr = np.asarray([scaler_obj[f"feature_{i}"][1] for i in idx], dtype=np.float32)

    # adjust length
    if med.shape[0] < num_features or iqr.shape[0] < num_features:
        raise ValueError(f"Scaler dims too small: med={med.shape}, iqr={iqr.shape}, need {num_features}")
    if med.shape[0] != num_features or iqr.shape[0] != num_features:
        med = med[:num_features]
        iqr = iqr[:num_features]

    iqr = np.where(iqr < 1e-6, 1.0, iqr).astype(np.float32)
    return med, iqr


def _parse_target_stats(ts: Any, horizons: list[int]) -> Dict[int, Tuple[float, float]]:
    """
    Supports common formats:
      {1: {"mean":..., "std":...}, 5: {...}, ...}
      {"1": {"mean":..., "std":...}, ...}
      {1: (mean, std), ...}
      {"means": {"1":..., ...}, "stds": {...}}
      {"means":[...], "stds":[...]} aligned with horizons order
    Returns: {h: (mean, std)}
    """
    if ts is None:
        return {}

    out: Dict[int, Tuple[float, float]] = {}

    # case: dict keyed by horizon
    if isinstance(ts, dict):
        # case: {"means":..., "stds":...}
        if "means" in ts and "stds" in ts:
            means = ts["means"]
            stds = ts["stds"]
            # dict keyed by horizon
            if isinstance(means, dict) and isinstance(stds, dict):
                for h in horizons:
                    mh = means.get(str(h), means.get(h))
                    sh = stds.get(str(h), stds.get(h))
                    if mh is None or sh is None:
                        continue
                    out[int(h)] = (float(mh), float(sh))
                return out
            # list aligned with horizons
            if isinstance(means, (list, tuple)) and isinstance(stds, (list, tuple)):
                if len(means) < len(horizons) or len(stds) < len(horizons):
                    raise ValueError("target_stats.means/stds lists shorter than horizons")
                for i, h in enumerate(horizons):
                    out[int(h)] = (float(means[i]), float(stds[i]))
                return out

        # direct per-horizon mapping
        for h in horizons:
            v = ts.get(h, ts.get(str(h)))
            if v is None:
                continue
            if isinstance(v, dict) and "mean" in v and "std" in v:
                out[int(h)] = (float(v["mean"]), float(v["std"]))
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                out[int(h)] = (float(v[0]), float(v[1]))

    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--val_npz", type=str, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no_jit", action="store_true")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    val_path = Path(args.val_npz)

    ckpt = pickle.load(open(ckpt_path, "rb"))
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be dict, got {type(ckpt)}")

    config = _as_config(ckpt.get("config"))
    params = jax.tree_util.tree_map(jnp.asarray, ckpt.get("params"))
    state = jax.tree_util.tree_map(jnp.asarray, ckpt.get("state", {}))

    print(f"CKPT: {ckpt_path}")
    print(f"  step={ckpt.get('step')}, model_version={ckpt.get('model_version', 'N/A')}")
    print(f"  horizons={config.horizons}")
    print(f"  quantiles={config.quantiles}")
    print(f"  lookback_length={config.lookback_length}, num_features={config.num_features}")

    X, Y_all = _load_val_npz(val_path)
    N, L, F = X.shape
    print(f"\nVAL: {val_path}")
    print(f"  X shape={X.shape}, y keys={sorted(Y_all.keys())}")

    # align features length
    if L < config.lookback_length:
        raise ValueError(f"val lookback too short: L={L} < config.lookback_length={config.lookback_length}")
    if L != config.lookback_length:
        # match common training behavior: take the most recent window
        X = X[:, -config.lookback_length :, :]
        print(f"  NOTE: sliced X to last lookback window -> {X.shape}")

    # align feature dims
    if F < config.num_features:
        raise ValueError(f"val feature dim too small: F={F} < config.num_features={config.num_features}")
    if F != config.num_features:
        X = X[:, :, : config.num_features]
        print(f"  NOTE: sliced X features to first num_features -> {X.shape}")

    # choose horizons present in both config and val
    horizons = [h for h in config.horizons if h in Y_all]
    if not horizons:
        raise ValueError(f"No overlapping horizons between config.horizons={config.horizons} and val y keys={sorted(Y_all.keys())}")
    if horizons != config.horizons:
        print(f"  NOTE: evaluating subset horizons={horizons}")

    # scaler
    med, iqr = _parse_scaler(ckpt.get("scaler"), num_features=config.num_features)
    Xn = (X - med[None, None, :]) / iqr[None, None, :]

    # forward
    def forward_fn(features: jax.Array) -> schemas.AlphaTradeOutput:
        m = model_lib.AlphaTrade(config)
        return m(features)

    net = hk.transform_with_state(forward_fn)

    def _apply(params, state, rng, xb):
        out, _st = net.apply(params, state, rng, xb)
        return out.log_return_quantiles  # dict[h] -> [B, Q]

    apply_fn = _apply if args.no_jit else jax.jit(_apply)

    # predict
    pred_q: Dict[int, np.ndarray] = {}
    for h in horizons:
        pred_q[h] = []

    rng = jax.random.PRNGKey(0)
    bs = int(args.batch)

    print(f"\nRunning inference: N={N}, batch={bs}, jit={not args.no_jit}")
    for i in range(0, N, bs):
        j = min(i + bs, N)
        xb = jnp.asarray(Xn[i:j], dtype=jnp.float32)
        rng, rk = jax.random.split(rng)
        out = apply_fn(params, state, rk, xb)
        for h in horizons:
            q = np.asarray(out[h], dtype=np.float32)  # [B, Q]
            pred_q[h].append(q)

    for h in horizons:
        pred_q[h] = np.concatenate(pred_q[h], axis=0)  # [N, Q]
        assert pred_q[h].shape[0] == N

    # de-normalize targets (if present)
    ts_map = _parse_target_stats(ckpt.get("target_stats"), horizons=horizons)
    if ts_map:
        print("\nTarget de-normalization: ON (using ckpt['target_stats'])")
    else:
        print("\nTarget de-normalization: OFF (ckpt has no parsable target_stats)")

    q_denorm: Dict[int, np.ndarray] = {}
    for h in horizons:
        q = pred_q[h]
        if h in ts_map:
            mean, std = ts_map[h]
            std = float(std) if float(std) > 0 else 1.0
            q = q * std + mean
        q_denorm[h] = q

    # ---- Diagnostics ----
    qs = np.asarray([float(q) for q in config.quantiles], dtype=np.float32)
    median_idx = len(qs) // 2
    q25_idx = int(np.argmin(np.abs(qs - 0.25)))
    q75_idx = int(np.argmin(np.abs(qs - 0.75)))

    print(f"\n{'='*72}")
    print("DIAGNOSTICS (per horizon)")
    print(f"{'='*72}")

    summary = []
    for h in horizons:
        y = np.asarray(Y_all[h], dtype=np.float32)
        q = q_denorm[h]  # [N, Q]

        # crossing
        diffs = np.diff(q, axis=1)
        crossing_rate = float(np.mean(np.any(diffs < 0, axis=1)))

        # coverage
        coverage = [float(np.mean(y <= q[:, j])) for j in range(q.shape[1])]
        cov_abs_err = float(np.mean([abs(coverage[j] - float(qs[j])) for j in range(len(qs))]))

        # bias & scale
        bias = float(np.mean(y - q[:, median_idx]))
        iqr_y = float(np.quantile(y, 0.75) - np.quantile(y, 0.25))
        iqr_pred = float(np.mean(q[:, q75_idx] - q[:, q25_idx])) if len(qs) >= 3 else float("nan")
        scale_ratio = float(iqr_pred / (iqr_y + 1e-12)) if np.isfinite(iqr_pred) else float("nan")

        # corr/std ratio on q50
        q50 = q[:, median_idx]
        corr50 = _corr(y, q50)
        std_ratio = float(np.std(q50) / (np.std(y) + 1e-12))

        print(f"\nh={h}")
        print(f"  q50 corr        : {corr50:+.4f}")
        print(f"  std_ratio(q50/y): {std_ratio:.3f}")
        print(f"  crossing_rate   : {crossing_rate:.4f} (want ~0)")
        print(f"  coverage        : {[f'{c:.3f}' for c in coverage]}")
        print(f"  expected        : {[f'{float(x):.3f}' for x in qs]}")
        print(f"  cov_abs_err     : {cov_abs_err:.4f} (lower better)")
        print(f"  bias(y-q50)     : {bias:+.6f}")
        print(f"  IQR_y           : {iqr_y:.6f}")
        print(f"  IQR_pred        : {iqr_pred:.6f}")
        print(f"  scale_ratio     : {scale_ratio:.3f}")

        summary.append((h, corr50, std_ratio, cov_abs_err, crossing_rate, scale_ratio))

    # horizon alignment matrix
    print(f"\n{'='*72}")
    print("HORIZON ALIGNMENT (corr of q50_pred(h_pred) vs y_h_true)")
    print(f"{'='*72}")
    for h_pred in horizons:
        q50 = q_denorm[h_pred][:, median_idx]
        row = []
        for h_true in horizons:
            y = np.asarray(Y_all[h_true], dtype=np.float32)
            row.append(_corr(y, q50))
        print(f"pred_h{h_pred}: " + "  ".join([f"vs y_h{h_true}:{c:+.3f}" for h_true, c in zip(horizons, row)]))

    # global summary
    corr_mean = float(np.mean([x[1] for x in summary if np.isfinite(x[1])])) if summary else float("nan")
    cov_mean = float(np.mean([x[3] for x in summary if np.isfinite(x[3])])) if summary else float("nan")
    cross_mean = float(np.mean([x[4] for x in summary if np.isfinite(x[4])])) if summary else float("nan")
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"corr_mean(q50): {corr_mean:+.4f}  |  cov_abs_err_mean: {cov_mean:.4f}  |  crossing_mean: {cross_mean:.4f}")
    print("Per-horizon: (h, corr50, std_ratio, cov_abs_err, crossing_rate, scale_ratio)")
    for t in summary:
        print(f"  {t[0]:>3}: corr={t[1]:+0.4f}, std_ratio={t[2]:.3f}, cov_err={t[3]:.4f}, cross={t[4]:.4f}, scale={t[5]:.3f}")


if __name__ == "__main__":
    main()