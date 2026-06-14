"""Model-quality diagnostics for AlphaTrade reports."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def horizon_key(horizon: int) -> str:
    return f"h{int(horizon)}"


def quantile_key(quantile: float) -> str:
    return f"q{int(round(float(quantile) * 100))}"


def resolve_horizon_weights(raw_weights, horizons: Sequence[int]) -> dict[int, float]:
    """Resolve list/dict horizon weights into an int-keyed mapping."""
    if raw_weights is None:
        return {int(h): 1.0 for h in horizons}

    if isinstance(raw_weights, Mapping):
        resolved = {}
        for h in horizons:
            candidates = (h, int(h), str(h), horizon_key(h))
            value = None
            for key in candidates:
                if key in raw_weights:
                    value = raw_weights[key]
                    break
            resolved[int(h)] = float(1.0 if value is None else value)
        if any(w < 0 for w in resolved.values()):
            raise ValueError("horizon_weights must be non-negative")
        if sum(resolved.values()) <= 0:
            raise ValueError("at least one horizon_weights value must be positive")
        return resolved

    values = list(raw_weights)
    if len(values) != len(horizons):
        raise ValueError(
            "horizon_weights length must match horizons: "
            f"{len(values)} != {len(horizons)}"
        )
    resolved = {int(h): float(w) for h, w in zip(horizons, values)}
    if any(w < 0 for w in resolved.values()):
        raise ValueError("horizon_weights must be non-negative")
    if sum(resolved.values()) <= 0:
        raise ValueError("at least one horizon_weights value must be positive")
    return resolved


def resolve_quantile_weights(raw_weights, quantiles: Sequence[float]) -> list[float]:
    """Resolve optional quantile weights; defaults to uniform explicit weights."""
    if raw_weights is None:
        return [1.0 for _ in quantiles]
    if isinstance(raw_weights, Mapping):
        values = []
        for q in quantiles:
            candidates = (q, float(q), str(q), quantile_key(q))
            value = None
            for key in candidates:
                if key in raw_weights:
                    value = raw_weights[key]
                    break
            values.append(float(1.0 if value is None else value))
        if any(w < 0 for w in values):
            raise ValueError("quantile_weights must be non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one quantile_weights value must be positive")
        return values

    values = [float(w) for w in raw_weights]
    if len(values) != len(quantiles):
        raise ValueError(
            "quantile_weights length must match quantiles: "
            f"{len(values)} != {len(quantiles)}"
        )
    if any(w < 0 for w in values):
        raise ValueError("quantile_weights must be non-negative")
    if sum(values) <= 0:
        raise ValueError("at least one quantile_weights value must be positive")
    return values


def loss_weight_report(
    *,
    horizons: Sequence[int],
    quantiles: Sequence[float],
    horizon_weights: Mapping[int, float] | None = None,
    quantile_weights: Sequence[float] | None = None,
    crossing_penalty_weight: float = 0.1,
    normalization: str = "weighted_sum",
) -> dict:
    """Return the effective loss weights used by training/eval."""
    h_weights = resolve_horizon_weights(horizon_weights, horizons)
    q_weights = resolve_quantile_weights(quantile_weights, quantiles)

    return {
        "normalization": normalization,
        "horizon_weights": {
            horizon_key(h): float(h_weights[int(h)]) for h in horizons
        },
        "quantile_weights": {
            quantile_key(q): float(w) for q, w in zip(quantiles, q_weights)
        },
        "pinball_residual_weights": {
            quantile_key(q): {
                "under_prediction": float(q),
                "over_prediction": float(1.0 - float(q)),
            }
            for q in quantiles
        },
        "crossing_penalty_weight": float(crossing_penalty_weight),
    }


def target_scale_summary(y: np.ndarray, horizons: Sequence[int]) -> dict:
    """Summarize target scale per horizon for raw log-return targets."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"targets must be 2D [N, H], got shape {arr.shape}")
    if arr.shape[1] != len(horizons):
        raise ValueError(
            "target horizon dimension does not match horizons: "
            f"{arr.shape[1]} != {len(horizons)}"
        )

    by_horizon = {}
    abs_p90_values = []
    for idx, horizon in enumerate(horizons):
        values = arr[:, idx]
        finite = values[np.isfinite(values)]
        key = horizon_key(horizon)
        if finite.size == 0:
            by_horizon[key] = {
                "samples": 0,
                "mean": 0.0,
                "std": 0.0,
                "iqr": 0.0,
                "abs_mean": 0.0,
                "abs_p50": 0.0,
                "abs_p90": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            continue

        abs_values = np.abs(finite)
        q25, q75 = np.percentile(finite, [25, 75])
        abs_p90 = float(np.percentile(abs_values, 90))
        abs_p90_values.append(abs_p90)
        by_horizon[key] = {
            "samples": int(finite.size),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "iqr": float(q75 - q25),
            "abs_mean": float(np.mean(abs_values)),
            "abs_p50": float(np.percentile(abs_values, 50)),
            "abs_p90": abs_p90,
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }

    positive_scales = [v for v in abs_p90_values if v > 0]
    scale_ratio = (
        float(max(positive_scales) / min(positive_scales))
        if positive_scales else 0.0
    )
    warnings = []
    if scale_ratio > 5.0:
        warnings.append(
            f"target abs_p90 scale ratio across horizons is {scale_ratio:.2f}"
        )

    return {
        "samples": int(arr.shape[0]),
        "by_horizon": by_horizon,
        "scale_ratio_abs_p90": scale_ratio,
        "warnings": warnings,
    }


def coverage_calibration_summary(
    coverage_by_horizon: Mapping[str, Mapping[str, float]],
    quantiles: Sequence[float],
) -> dict:
    """Summarize empirical quantile coverage error."""
    by_horizon = {}
    by_quantile_values = {quantile_key(q): [] for q in quantiles}
    errors = []
    worst = {"horizon": "", "quantile": "", "expected": 0.0, "observed": 0.0, "abs_error": 0.0}

    for h_key, coverage in coverage_by_horizon.items():
        q_errors = {}
        horizon_errors = []
        for q in quantiles:
            q_key = quantile_key(q)
            observed = float(coverage.get(q_key, 0.0))
            expected = float(q)
            abs_error = abs(observed - expected)
            q_errors[q_key] = {
                "expected": expected,
                "observed": observed,
                "abs_error": float(abs_error),
            }
            by_quantile_values[q_key].append(observed)
            horizon_errors.append(abs_error)
            errors.append(abs_error)
            if not worst["horizon"] or abs_error > worst["abs_error"]:
                worst = {
                    "horizon": h_key,
                    "quantile": q_key,
                    "expected": expected,
                    "observed": observed,
                    "abs_error": float(abs_error),
                }
        by_horizon[h_key] = {
            "mae": float(np.mean(horizon_errors)) if horizon_errors else 0.0,
            "max_abs_error": float(max(horizon_errors)) if horizon_errors else 0.0,
            "by_quantile": q_errors,
        }

    by_quantile = {}
    for q in quantiles:
        q_key = quantile_key(q)
        observed_values = by_quantile_values[q_key]
        expected = float(q)
        mean_observed = float(np.mean(observed_values)) if observed_values else 0.0
        by_quantile[q_key] = {
            "expected": expected,
            "observed": mean_observed,
            "abs_error": float(abs(mean_observed - expected)),
        }

    return {
        "overall_mae": float(np.mean(errors)) if errors else 0.0,
        "max_abs_error": float(max(errors)) if errors else 0.0,
        "worst": worst,
        "by_horizon": by_horizon,
        "by_quantile": by_quantile,
    }
