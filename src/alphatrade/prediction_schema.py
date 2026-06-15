"""Stable prediction parquet schema helpers for AlphaTrade product outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


PREDICTION_SCHEMA_VERSION = "m9_predictions_v1"
DEFAULT_HORIZONS = (1, 5, 20, 60)
DEFAULT_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
BASE_COLUMNS = ("symbol", "eob", "model_version")


@dataclass(frozen=True)
class PredictionSchemaIssue:
    """A single prediction schema validation issue."""

    field: str
    message: str


def quantile_label(quantile: float) -> str:
    """Return the canonical q-label for a quantile, e.g. 0.5 -> q50."""
    q_pct = round(float(quantile) * 100)
    if not np.isclose(float(quantile) * 100, q_pct, atol=1e-8):
        raise ValueError(f"quantile cannot be represented as integer percent: {quantile}")
    return f"q{int(q_pct)}"


def prediction_column(horizon: int, quantile: float) -> str:
    """Return the canonical prediction column name."""
    return f"h{int(horizon)}_{quantile_label(quantile)}"


def prediction_columns(
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> list[str]:
    """Return prediction columns in stable contract order."""
    return [prediction_column(h, q) for h in horizons for q in quantiles]


def required_columns(
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> list[str]:
    """Return all required parquet columns in stable contract order."""
    return [*BASE_COLUMNS, *prediction_columns(horizons, quantiles)]


def build_prediction_frame(
    *,
    symbols: Sequence[str],
    eobs: Sequence,
    model_version: str,
    predictions_by_horizon: Mapping[int, np.ndarray],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Build the stable wide prediction DataFrame."""
    n_rows = len(symbols)
    if len(eobs) != n_rows:
        raise ValueError(f"symbols/eobs length mismatch: {n_rows} vs {len(eobs)}")

    rows = {
        "symbol": list(symbols),
        "eob": pd.to_datetime(list(eobs)),
        "model_version": [model_version] * n_rows,
    }
    for horizon in horizons:
        if horizon not in predictions_by_horizon:
            raise ValueError(f"missing predictions for horizon {horizon}")
        preds = np.asarray(predictions_by_horizon[horizon])
        expected_shape = (n_rows, len(quantiles))
        if preds.shape != expected_shape:
            raise ValueError(
                f"horizon {horizon} predictions shape mismatch: "
                f"expected {expected_shape}, got {preds.shape}"
            )
        for qi, quantile in enumerate(quantiles):
            rows[prediction_column(horizon, quantile)] = preds[:, qi].astype(np.float64)

    return pd.DataFrame(rows, columns=required_columns(horizons, quantiles))


def validate_prediction_frame(
    df: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    expected_model_version: str | None = None,
    require_non_empty: bool = True,
) -> list[PredictionSchemaIssue]:
    """Validate prediction parquet columns, dtypes, and core invariants."""
    issues: list[PredictionSchemaIssue] = []

    expected = required_columns(horizons, quantiles)
    actual = set(df.columns)
    missing = [col for col in expected if col not in actual]
    if missing:
        issues.append(PredictionSchemaIssue("columns", f"missing columns: {missing}"))
        return issues

    if require_non_empty and len(df) == 0:
        issues.append(PredictionSchemaIssue("rows", "prediction frame is empty"))

    if not pd.api.types.is_string_dtype(df["symbol"]):
        issues.append(PredictionSchemaIssue("symbol", f"expected string dtype, got {df['symbol'].dtype}"))

    if not pd.api.types.is_datetime64_any_dtype(df["eob"]):
        issues.append(PredictionSchemaIssue("eob", f"expected datetime dtype, got {df['eob'].dtype}"))

    if not pd.api.types.is_string_dtype(df["model_version"]):
        issues.append(
            PredictionSchemaIssue(
                "model_version",
                f"expected string dtype, got {df['model_version'].dtype}",
            )
        )
    elif expected_model_version is not None:
        versions = set(df["model_version"].dropna().astype(str).unique().tolist())
        if versions != {expected_model_version}:
            issues.append(
                PredictionSchemaIssue(
                    "model_version",
                    f"expected only {expected_model_version!r}, got {sorted(versions)}",
                )
            )

    duplicate_count = int(df.duplicated(list(BASE_COLUMNS)).sum())
    if duplicate_count:
        issues.append(
            PredictionSchemaIssue(
                "duplicates",
                f"duplicate (symbol, eob, model_version) rows: {duplicate_count}",
            )
        )

    type_errors = []
    nonfinite_errors = []
    for col in prediction_columns(horizons, quantiles):
        if not pd.api.types.is_numeric_dtype(df[col]):
            type_errors.append(f"{col}={df[col].dtype}")
            continue
        values = df[col].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            nonfinite_errors.append(col)
    if type_errors:
        issues.append(PredictionSchemaIssue("prediction_types", f"non-numeric columns: {type_errors}"))
    if nonfinite_errors:
        issues.append(PredictionSchemaIssue("prediction_nonfinite", f"columns contain NaN/Inf: {nonfinite_errors}"))

    crossing_errors = []
    for horizon in horizons:
        cols = [prediction_column(horizon, q) for q in quantiles]
        if any(col not in df.columns for col in cols):
            continue
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in cols):
            continue
        values = df[cols].to_numpy(dtype=np.float64, copy=False)
        crossings = int((values[:, 1:] < values[:, :-1]).sum())
        if crossings:
            crossing_errors.append(f"h{int(horizon)}={crossings}")
    if crossing_errors:
        issues.append(
            PredictionSchemaIssue(
                "quantile_crossing",
                f"non-monotonic quantiles: {crossing_errors}",
            )
        )

    return issues
