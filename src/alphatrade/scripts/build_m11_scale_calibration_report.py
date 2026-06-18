#!/usr/bin/env python3
"""M11: Target scale audit, model quality checks, and calibration diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from alphatrade import prediction_schema
from alphatrade import runtime_paths
from alphatrade.scripts import build_m10_prediction_eval as m10


_STAGE_NOTE = (
    "AlphaTrade is in active development. These reports are evaluation and "
    "calibration diagnostics only, not trading-readiness claims."
)
_BACKTEST_NOTE = (
    "M9/M10/M11 backtest outputs are lightweight evaluation handoff checks, "
    "not full execution simulators."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="M11 target scale audit and calibration diagnostic"
    )
    parser.add_argument("--m10-reports-dir", type=str, default=None,
                        help="Directory containing the completed M10 reports")
    parser.add_argument("--test-predictions", type=str, default=None,
                        help="M9 test predictions parquet (default: <m10-reports-dir>/m9_predictions.parquet)")
    parser.add_argument("--test-eval-rows", type=str, default=None,
                        help="M10 aligned test eval rows parquet (default: <m10-reports-dir>/m10_eval_rows.parquet)")
    parser.add_argument("--val-predictions", type=str, default=None,
                        help="Validation predictions parquet used to fit calibration")
    parser.add_argument("--val-eval-rows", type=str, default=None,
                        help="Pre-aligned validation eval rows parquet; skips validation alignment")
    parser.add_argument("--m10-metrics", type=str, default=None,
                        help="M10 prediction metrics JSON (default: <m10-reports-dir>/m10_prediction_eval_metrics.json)")
    parser.add_argument("--m10-baseline-comparison", type=str, default=None,
                        help="M10 baseline comparison JSON (default: <m10-reports-dir>/m10_baseline_comparison.json)")
    parser.add_argument("--data-dir", type=str, default="data/processed/m1_f8",
                        help="Processed bars/index root")
    parser.add_argument("--horizons", type=str, default="1,5,20,60",
                        help="Comma-separated horizons")
    parser.add_argument("--quantiles", type=str, default="0.1,0.3,0.5,0.7,0.9",
                        help="Comma-separated quantiles")
    parser.add_argument("--cost-bps-list", type=str, default="0,1,2,5",
                        help="Comma-separated cost bps list")
    parser.add_argument("--thresholds", type=str, default="0,0.5sigma,1sigma",
                        help="Comma-separated thresholds: numeric or Ns sigma")
    parser.add_argument("--rolling-window-rows", type=int, default=7200,
                        help="Past-only rolling baseline window rows")
    parser.add_argument("--min-rolling-samples", type=int, default=10,
                        help="Minimum past samples for rolling baseline quantiles")
    parser.add_argument("--calibration-fit-split", type=str, default="val",
                        help="Index split used to filter calibration fit rows")
    parser.add_argument("--out-prefix", type=str, default="m11",
                        help="Output filename prefix")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory")
    parser.add_argument("--progress-log", type=str, default=None,
                        help="JSONL progress log path")
    return parser.parse_args()


def _parse_int_csv(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _q_key(q: float) -> str:
    return prediction_schema.quantile_label(q)


def _h_key(h: int) -> str:
    return f"h{int(h)}"


def _prediction_col(h: int, q: float) -> str:
    return prediction_schema.prediction_column(h, q)


def _realized_col(h: int) -> str:
    return f"realized_h{int(h)}"


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _process_rss_mb() -> float | None:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            pages = int(f.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)
    except Exception:
        return None


def _write_progress_event(progress_log: Path | None, t0: float, event: dict) -> None:
    if progress_log is None:
        return
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "m11_progress_v1",
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "rss_mb": _process_rss_mb(),
        **event,
    }
    with progress_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(m10._json_ready(payload), sort_keys=True) + "\n")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_input(
    explicit: str | None,
    *,
    m10_reports_dir: Path | None,
    filename: str,
) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if m10_reports_dir is None:
        raise ValueError(f"{filename} requires --m10-reports-dir or an explicit path")
    return (m10_reports_dir / filename).resolve()


def _stats_from_arrays(arrays: list[np.ndarray]) -> dict:
    if not arrays:
        return m10._summary([])
    finite_arrays = []
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float64)
        if values.size:
            finite_arrays.append(values[np.isfinite(values)])
    if not finite_arrays:
        return m10._summary([])
    return m10._summary(np.concatenate(finite_arrays))


def collect_index_target_stats(
    *,
    data_dir: Path,
    symbols: list[str],
    horizons: list[int],
    splits: list[str],
) -> dict:
    """Collect raw y_h* statistics from sample index parquet files."""
    result = {}
    target_cols = [f"y_h{h}" for h in horizons]
    for split in splits:
        by_horizon_arrays = {h: [] for h in horizons}
        by_symbol_horizon = {}
        missing_files = []
        row_count = 0
        for symbol in symbols:
            path = data_dir / symbol / f"index_{split}.parquet"
            if not path.exists():
                missing_files.append(str(path))
                continue
            df = pd.read_parquet(path, columns=target_cols)
            row_count += int(len(df))
            by_symbol_horizon[symbol] = {}
            for h in horizons:
                col = f"y_h{h}"
                values = df[col].to_numpy(dtype=np.float64, copy=False)
                by_horizon_arrays[h].append(values)
                by_symbol_horizon[symbol][_h_key(h)] = m10._summary(values)

        result[split] = {
            "rows": int(row_count),
            "missing_files": missing_files,
            "by_horizon": {
                _h_key(h): _stats_from_arrays(by_horizon_arrays[h])
                for h in horizons
            },
            "by_symbol_horizon": by_symbol_horizon,
        }
    return result


def collect_prediction_stats_from_parquet(
    path: Path,
    *,
    horizons: list[int],
    quantiles: list[float],
) -> dict:
    cols = prediction_schema.prediction_columns(horizons, quantiles)
    predictions = pd.read_parquet(path, columns=cols)
    by_horizon = {}
    for h in horizons:
        by_horizon[_h_key(h)] = {
            _q_key(q): m10._summary(predictions[_prediction_col(h, q)])
            for q in quantiles
        }
    return {
        "path": str(path),
        "rows": int(len(predictions)),
        "by_horizon": by_horizon,
    }


def collect_realized_stats(rows: pd.DataFrame, horizons: list[int]) -> dict:
    return {
        "rows": int(len(rows)),
        "by_horizon": {
            _h_key(h): m10._summary(rows[_realized_col(h)])
            for h in horizons
        },
    }


def compute_scale_ratios(
    rows: pd.DataFrame,
    *,
    horizons: list[int],
) -> dict:
    result = {}
    for h in horizons:
        realized = m10._summary(rows[_realized_col(h)])
        q10 = rows[_prediction_col(h, 0.1)].to_numpy(dtype=np.float64)
        q90 = rows[_prediction_col(h, 0.9)].to_numpy(dtype=np.float64)
        pred_width = float(np.nanmean(q90 - q10))
        realized_width = None
        if realized["p99"] is not None and realized["p01"] is not None:
            realized_width = float(realized["p99"] - realized["p01"])
        ratio = None
        if realized_width is not None and abs(realized_width) > 0.0:
            ratio = float(abs(pred_width) / abs(realized_width))
        result[_h_key(h)] = {
            "prediction_mean_q10": float(np.nanmean(q10)),
            "prediction_mean_q90": float(np.nanmean(q90)),
            "prediction_mean_q90_minus_q10": pred_width,
            "realized_p99_minus_p01": realized_width,
            "prediction_width_to_realized_p99_p01_ratio": ratio,
            "realized_std": realized["std"],
        }
    return result


def build_target_scale_audit(
    *,
    test_predictions_path: Path,
    test_eval_rows: pd.DataFrame,
    data_dir: Path,
    symbols: list[str],
    horizons: list[int],
    quantiles: list[float],
    inputs: dict,
) -> dict:
    index_stats = collect_index_target_stats(
        data_dir=data_dir,
        symbols=symbols,
        horizons=horizons,
        splits=["train", "val", "test"],
    )
    final_prediction_stats = collect_prediction_stats_from_parquet(
        test_predictions_path,
        horizons=horizons,
        quantiles=quantiles,
    )
    scale_ratios = compute_scale_ratios(test_eval_rows, horizons=horizons)
    mismatch_horizons = [
        h_key for h_key, stats in scale_ratios.items()
        if stats["prediction_width_to_realized_p99_p01_ratio"] is not None
        and stats["prediction_width_to_realized_p99_p01_ratio"] > 10.0
    ]
    return {
        "schema_version": "m11_target_scale_audit_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "stage": "development_evaluation",
        "notes": [_STAGE_NOTE, _BACKTEST_NOTE],
        "inputs": inputs,
        "target_unit_conclusion": {
            "target_unit": "raw_log_return",
            "target_formula": "log(close[t+h] / close[t])",
            "training_target_normalization": "none_found",
            "inference_inverse_transform_required": False,
            "inference_inverse_transform_present": False,
            "engineering_scale_mismatch_found": False,
            "model_output_scale_mismatch_found": bool(mismatch_horizons),
            "mismatch_horizons": mismatch_horizons,
            "summary": (
                "Targets, training loss, M9 inference, and M10 evaluation all use raw "
                "log returns. The M9 parquet values are direct model outputs. The "
                "observed mismatch is therefore an output/calibration problem in the "
                "current champion, not an identified missing inverse transform."
            ),
        },
        "code_path_audit": {
            "target_generation": {
                "file": "src/alphatrade/scripts/build_m1_sample_index_v2.py",
                "function": "compute_labels",
                "behavior": "df[y_h] = log((future_close + eps) / (close + eps))",
            },
            "training_dataset": {
                "file": "src/alphatrade/window_cache.py",
                "function": "materialize_windows",
                "behavior": "y_h1/y_h5/y_h20/y_h60 are copied directly into the y array",
            },
            "training_loss": {
                "file": "src/alphatrade/scripts/train_m4_alphatrade.py",
                "function": "make_train_step",
                "behavior": "pinball loss compares yb[:, i] directly to output.log_return_quantiles[horizon]",
            },
            "m9_inference": {
                "file": "src/alphatrade/inference.py",
                "function": "AlphaTradeBundlePredictor.predict_arrays",
                "behavior": "returns output.log_return_quantiles with no inverse transform",
            },
            "m10_evaluation": {
                "file": "src/alphatrade/scripts/build_m10_prediction_eval.py",
                "function": "align_predictions_with_realized",
                "behavior": "realized_h = log(close[t+h] / close[t])",
            },
        },
        "target_statistics": {
            "index_targets": index_stats,
            "m10_realized_test_targets": collect_realized_stats(test_eval_rows, horizons),
            "raw_model_output": {
                "source": "M9 prediction parquet columns; inference is a direct output passthrough",
                "same_as_final_m9_predictions": True,
                **final_prediction_stats,
            },
            "final_m9_predictions": final_prediction_stats,
        },
        "scale_diagnostics": {
            "definition": "mean(q90 - q10) divided by realized target (p99 - p01)",
            "fail_threshold": 10.0,
            "by_horizon": scale_ratios,
        },
    }


def _status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL_MODEL_QUALITY": 2}.get(status, 1)


def _overall_status(checks: list[dict]) -> str:
    if not checks:
        return "WARN"
    return max((c["status"] for c in checks), key=_status_rank)


def _check(
    checks: list[dict],
    *,
    name: str,
    status: str,
    observed,
    threshold,
    detail: str,
) -> None:
    checks.append({
        "name": name,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    })


def _get_nested(obj: dict, path: list[str], default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_quality_validation(
    *,
    m10_metrics: dict,
    m10_baseline: dict,
    scale_ratios: dict,
) -> dict:
    checks = []
    model_pinball = _get_nested(m10_metrics, ["pinball_loss", "overall"])
    zero_pinball = _get_nested(
        m10_baseline,
        ["baselines", "zero_return_quantile", "pinball_loss", "overall"],
    )
    rolling_pinball = _get_nested(
        m10_baseline,
        ["baselines", "rolling_historical_quantile", "pinball_loss", "overall"],
    )
    coverage_mae = _get_nested(m10_metrics, ["quantile_coverage", "overall_mae"])
    max_abs_pearson_ic = _get_nested(
        m10_metrics,
        ["ic_metrics", "materiality_proxy", "max_abs_pearson_ic"],
    )
    max_abs_rank_ic = _get_nested(
        m10_metrics,
        ["ic_metrics", "materiality_proxy", "max_abs_rank_ic"],
    )
    max_abs_ic = None
    ic_values = [v for v in [max_abs_pearson_ic, max_abs_rank_ic] if v is not None]
    if ic_values:
        max_abs_ic = float(max(ic_values))

    if model_pinball is not None and zero_pinball is not None:
        _check(
            checks,
            name="model_pinball_vs_zero_return_quantile",
            status="FAIL_MODEL_QUALITY" if model_pinball > zero_pinball else "PASS",
            observed={"model": model_pinball, "baseline": zero_pinball},
            threshold="model <= zero_return_quantile",
            detail="Model pinball must not be worse than the zero-return quantile baseline.",
        )
    else:
        _check(
            checks,
            name="model_pinball_vs_zero_return_quantile",
            status="WARN",
            observed={"model": model_pinball, "baseline": zero_pinball},
            threshold="model <= zero_return_quantile",
            detail="Missing model or zero baseline pinball.",
        )

    if model_pinball is not None and rolling_pinball is not None:
        _check(
            checks,
            name="model_pinball_vs_rolling_historical_quantile",
            status="FAIL_MODEL_QUALITY" if model_pinball > rolling_pinball else "PASS",
            observed={"model": model_pinball, "baseline": rolling_pinball},
            threshold="model <= rolling_historical_quantile",
            detail="Model pinball must not be worse than the rolling historical baseline.",
        )
    else:
        _check(
            checks,
            name="model_pinball_vs_rolling_historical_quantile",
            status="WARN",
            observed={"model": model_pinball, "baseline": rolling_pinball},
            threshold="model <= rolling_historical_quantile",
            detail="Missing model or rolling baseline pinball.",
        )

    _check(
        checks,
        name="coverage_calibration_mae",
        status=(
            "WARN" if coverage_mae is None
            else "FAIL_MODEL_QUALITY" if coverage_mae > 0.05
            else "PASS"
        ),
        observed=coverage_mae,
        threshold="<= 0.05",
        detail="Overall quantile coverage MAE should be low before a champion is considered healthy.",
    )

    _check(
        checks,
        name="max_abs_ic",
        status=(
            "WARN" if max_abs_ic is None
            else "FAIL_MODEL_QUALITY" if max_abs_ic < m10.IC_MATERIALITY_THRESHOLD
            else "PASS"
        ),
        observed=max_abs_ic,
        threshold=f">= {m10.IC_MATERIALITY_THRESHOLD}",
        detail="At least one horizon should show non-trivial absolute Pearson or rank IC.",
    )

    scale_failures = {
        h_key: stats["prediction_width_to_realized_p99_p01_ratio"]
        for h_key, stats in scale_ratios.items()
        if stats["prediction_width_to_realized_p99_p01_ratio"] is not None
        and stats["prediction_width_to_realized_p99_p01_ratio"] > 10.0
    }
    _check(
        checks,
        name="prediction_q90_q10_scale",
        status="FAIL_MODEL_QUALITY" if scale_failures else "PASS",
        observed=scale_failures or {
            h_key: stats["prediction_width_to_realized_p99_p01_ratio"]
            for h_key, stats in scale_ratios.items()
        },
        threshold="<= 10x realized p99-p01 scale",
        detail="Mean prediction interval width should not dwarf realized target scale.",
    )

    q50_failures = {}
    by_horizon = _get_nested(m10_metrics, ["quantile_coverage", "by_horizon"], {})
    for h_key, h_metrics in by_horizon.items():
        q50 = _get_nested(h_metrics, ["coverage", "q50", "abs_error"])
        if q50 is not None and q50 > 0.20:
            q50_failures[h_key] = q50
    _check(
        checks,
        name="q50_coverage_abs_error",
        status="FAIL_MODEL_QUALITY" if q50_failures else "PASS",
        observed=q50_failures or {
            h_key: _get_nested(h_metrics, ["coverage", "q50", "abs_error"])
            for h_key, h_metrics in by_horizon.items()
        },
        threshold="<= 0.20 by horizon",
        detail="Median quantile coverage should be close enough to 50%.",
    )

    return {
        "schema_version": "m11_model_quality_validation_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "stage": "development_evaluation",
        "notes": [_STAGE_NOTE, _BACKTEST_NOTE],
        "severity_levels": ["PASS", "WARN", "FAIL_MODEL_QUALITY"],
        "overall_status": _overall_status(checks),
        "inputs": {
            "m10_model_pinball": model_pinball,
            "zero_return_quantile_pinball": zero_pinball,
            "rolling_historical_quantile_pinball": rolling_pinball,
            "coverage_calibration_mae": coverage_mae,
            "max_abs_pearson_ic": max_abs_pearson_ic,
            "max_abs_rank_ic": max_abs_rank_ic,
        },
        "checks": checks,
    }


def filter_rows_to_split_index(
    rows: pd.DataFrame,
    *,
    data_dir: Path,
    split: str,
) -> tuple[pd.DataFrame, dict]:
    """Keep rows whose (symbol, eob) is present in index_<split>.parquet."""
    pieces = []
    missing_files = []
    kept_by_symbol = {}
    rows = rows.copy()
    rows["eob"] = pd.to_datetime(rows["eob"])
    for symbol, sdf in rows.groupby("symbol", sort=True):
        path = data_dir / str(symbol) / f"index_{split}.parquet"
        if not path.exists():
            missing_files.append(str(path))
            continue
        index_df = pd.read_parquet(path, columns=["eob"])
        allowed = pd.to_datetime(index_df["eob"])
        mask = sdf["eob"].isin(allowed)
        kept = sdf.loc[mask].copy()
        kept_by_symbol[str(symbol)] = int(len(kept))
        if not kept.empty:
            pieces.append(kept)
    filtered = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else rows.iloc[:0].copy()
    )
    return filtered, {
        "split": split,
        "rows_before": int(len(rows)),
        "rows_after": int(len(filtered)),
        "missing_files": missing_files,
        "kept_by_symbol": kept_by_symbol,
    }


def fit_quantile_shift_calibrator(
    rows: pd.DataFrame,
    *,
    horizons: list[int],
    quantiles: list[float],
) -> dict:
    by_horizon = {}
    for h in horizons:
        h_key = _h_key(h)
        by_horizon[h_key] = {}
        y = rows[_realized_col(h)].to_numpy(dtype=np.float64)
        for q in quantiles:
            p = rows[_prediction_col(h, q)].to_numpy(dtype=np.float64)
            residual = y - p
            residual = residual[np.isfinite(residual)]
            shift = float(np.quantile(residual, float(q))) if residual.size else 0.0
            by_horizon[h_key][_q_key(q)] = {
                "method": "additive_residual_quantile_shift",
                "expected_coverage": float(q),
                "shift": shift,
                "fit_samples": int(residual.size),
            }
    return {
        "schema_version": "m11_quantile_shift_calibrator_v1",
        "method": (
            "Per-horizon additive residual quantile shifts fitted on validation "
            "rows only, followed by non-crossing cumulative maximum enforcement."
        ),
        "fit_rows": int(len(rows)),
        "by_horizon": by_horizon,
    }


def apply_quantile_shift_calibrator(
    rows: pd.DataFrame,
    calibrator: dict,
    *,
    horizons: list[int],
    quantiles: list[float],
) -> pd.DataFrame:
    calibrated = rows.copy()
    for h in horizons:
        cols = [_prediction_col(h, q) for q in quantiles]
        h_params = calibrator["by_horizon"][_h_key(h)]
        for q in quantiles:
            col = _prediction_col(h, q)
            calibrated[col] = (
                calibrated[col].astype(np.float64)
                + float(h_params[_q_key(q)]["shift"])
            )
        values = calibrated[cols].to_numpy(dtype=np.float64, copy=True)
        calibrated[cols] = np.maximum.accumulate(values, axis=1)
    return calibrated


def _baseline_available_mask(
    rows: pd.DataFrame,
    *,
    horizons: list[int],
    quantiles: list[float],
) -> pd.Series:
    return m10._baseline_available_mask(
        rows,
        source="rolling_historical_quantile",
        horizons=horizons,
        quantiles=quantiles,
    )


def compute_source_report(
    rows: pd.DataFrame,
    *,
    horizons: list[int],
    quantiles: list[float],
    cost_bps_list: list[float],
    threshold_specs: list[dict],
    source: str,
    progress_log: Path | None,
    t0: float,
) -> dict:
    print(f"  [m11:{source}] metrics start on {len(rows):,} rows", flush=True)
    coverage = m10.compute_quantile_coverage(rows, horizons, quantiles)
    report = {
        "rows": int(len(rows)),
        "pinball_loss": m10.compute_pinball_loss(rows, horizons, quantiles),
        "quantile_coverage": coverage,
        "quantile_crossing": m10.compute_quantile_crossing(rows, horizons, quantiles),
        "ic_metrics": m10.compute_ic_metrics(rows, horizons),
        "direction_metrics": m10.compute_direction_metrics(rows, horizons),
        "backtest_matrix": m10.compute_backtest_matrix(
            rows,
            horizons=horizons,
            cost_bps_list=cost_bps_list,
            threshold_specs=threshold_specs,
            source=source,
            progress_log=progress_log,
            t0=t0,
        ),
    }
    print(f"  [m11:{source}] metrics done", flush=True)
    return report


def _selected_backtest_rows(report: dict) -> list[dict]:
    matrix = report.get("backtest_matrix", {}).get("matrix", [])
    return [
        row for row in matrix
        if row.get("signal_variant") == "q50_sign"
        and row.get("threshold_label") == "0"
    ]


def _best_net(report: dict) -> dict | None:
    matrix = report.get("backtest_matrix", {}).get("matrix", [])
    if not matrix:
        return None
    return max(matrix, key=lambda row: row.get("net_return", float("-inf")))


def build_calibration_comparison(
    *,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    horizons: list[int],
    quantiles: list[float],
    cost_bps_list: list[float],
    threshold_specs: list[dict],
    progress_log: Path | None,
    t0: float,
) -> dict:
    calibrator = fit_quantile_shift_calibrator(
        val_rows,
        horizons=horizons,
        quantiles=quantiles,
    )
    val_calibrated = apply_quantile_shift_calibrator(
        val_rows,
        calibrator,
        horizons=horizons,
        quantiles=quantiles,
    )
    calibrated_test_rows = apply_quantile_shift_calibrator(
        test_rows,
        calibrator,
        horizons=horizons,
        quantiles=quantiles,
    )

    raw_report = compute_source_report(
        test_rows,
        horizons=horizons,
        quantiles=quantiles,
        cost_bps_list=cost_bps_list,
        threshold_specs=threshold_specs,
        source="raw_model",
        progress_log=progress_log,
        t0=t0,
    )
    calibrated_report = compute_source_report(
        calibrated_test_rows,
        horizons=horizons,
        quantiles=quantiles,
        cost_bps_list=cost_bps_list,
        threshold_specs=threshold_specs,
        source="calibrated_model",
        progress_log=progress_log,
        t0=t0,
    )

    rolling_mask = _baseline_available_mask(
        test_rows,
        horizons=horizons,
        quantiles=quantiles,
    )
    rolling_common = test_rows.loc[rolling_mask].copy()
    raw_common = test_rows.loc[rolling_mask].copy()
    calibrated_common = calibrated_test_rows.loc[rolling_mask].copy()
    rolling_view = m10.make_prediction_view(
        rolling_common,
        source="rolling_historical_quantile",
        horizons=horizons,
        quantiles=quantiles,
    )
    rolling_report = compute_source_report(
        rolling_view,
        horizons=horizons,
        quantiles=quantiles,
        cost_bps_list=cost_bps_list,
        threshold_specs=threshold_specs,
        source="rolling_historical_quantile",
        progress_log=progress_log,
        t0=t0,
    )
    raw_common_report = {
        "rows": int(len(raw_common)),
        "pinball_loss": m10.compute_pinball_loss(raw_common, horizons, quantiles),
        "quantile_coverage": m10.compute_quantile_coverage(raw_common, horizons, quantiles),
        "ic_metrics": m10.compute_ic_metrics(raw_common, horizons),
    }
    calibrated_common_report = {
        "rows": int(len(calibrated_common)),
        "pinball_loss": m10.compute_pinball_loss(calibrated_common, horizons, quantiles),
        "quantile_coverage": m10.compute_quantile_coverage(calibrated_common, horizons, quantiles),
        "ic_metrics": m10.compute_ic_metrics(calibrated_common, horizons),
    }

    raw_mae = raw_report["quantile_coverage"]["overall_mae"]
    calibrated_mae = calibrated_report["quantile_coverage"]["overall_mae"]
    rolling_pinball = rolling_report["pinball_loss"]["overall"]
    calibrated_common_pinball = calibrated_common_report["pinball_loss"]["overall"]
    calibrated_max_abs_ic = max(
        [
            v for v in [
                calibrated_report["ic_metrics"]["materiality_proxy"].get("max_abs_pearson_ic"),
                calibrated_report["ic_metrics"]["materiality_proxy"].get("max_abs_rank_ic"),
            ]
            if v is not None
        ],
        default=None,
    )
    coverage_reduction = (
        None if raw_mae is None or calibrated_mae is None
        else float(raw_mae - calibrated_mae)
    )
    can_retest = bool(
        calibrated_common_pinball is not None
        and rolling_pinball is not None
        and calibrated_common_pinball <= rolling_pinball
        and calibrated_mae is not None
        and calibrated_mae <= 0.05
        and calibrated_max_abs_ic is not None
        and calibrated_max_abs_ic >= m10.IC_MATERIALITY_THRESHOLD
    )

    return {
        "schema_version": "m11_calibration_comparison_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "stage": "development_evaluation",
        "notes": [_STAGE_NOTE, _BACKTEST_NOTE],
        "calibrator": calibrator,
        "fit_diagnostics": {
            "validation_rows_raw": int(len(val_rows)),
            "validation_fit_raw_coverage": m10.compute_quantile_coverage(val_rows, horizons, quantiles),
            "validation_fit_calibrated_coverage": m10.compute_quantile_coverage(val_calibrated, horizons, quantiles),
        },
        "raw": raw_report,
        "calibrated": calibrated_report,
        "raw_vs_calibrated": {
            "pinball_delta_raw_minus_calibrated": (
                None if raw_report["pinball_loss"]["overall"] is None
                or calibrated_report["pinball_loss"]["overall"] is None
                else float(raw_report["pinball_loss"]["overall"] - calibrated_report["pinball_loss"]["overall"])
            ),
            "coverage_mae_delta_raw_minus_calibrated": coverage_reduction,
            "raw_selected_backtest": _selected_backtest_rows(raw_report),
            "calibrated_selected_backtest": _selected_backtest_rows(calibrated_report),
            "raw_best_net": _best_net(raw_report),
            "calibrated_best_net": _best_net(calibrated_report),
        },
        "rolling_historical_recomparison": {
            "common_rows": int(len(rolling_common)),
            "raw_model_on_common": raw_common_report,
            "calibrated_model_on_common": calibrated_common_report,
            "rolling_historical_quantile": rolling_report,
            "pinball_delta_rolling_minus_calibrated": (
                None if rolling_pinball is None or calibrated_common_pinball is None
                else float(rolling_pinball - calibrated_common_pinball)
            ),
        },
        "conclusion": {
            "coverage_mae_reduced_materially": bool(
                coverage_reduction is not None and coverage_reduction > 0.01
            ),
            "calibrated_model_can_be_retested_as_champion": can_retest,
            "champion_status": (
                "can_be_retested_with_productization_gates"
                if can_retest
                else "rejected_pending_retrain_or_stronger_fix"
            ),
            "decision_rule": (
                "Retest only if calibrated pinball beats rolling historical on common rows, "
                f"coverage MAE <= 0.05, and max abs IC >= {m10.IC_MATERIALITY_THRESHOLD}."
            ),
        },
    }


def write_target_scale_audit_md(report: dict, path: Path) -> None:
    c = report["target_unit_conclusion"]
    lines = [
        "# M11 Target Scale Audit",
        "",
        _STAGE_NOTE,
        "",
        _BACKTEST_NOTE,
        "",
        "## Conclusion",
        "",
        f"- Target unit: `{c['target_unit']}`",
        f"- Target formula: `{c['target_formula']}`",
        f"- Training target normalization: `{c['training_target_normalization']}`",
        f"- Inference inverse transform required: `{c['inference_inverse_transform_required']}`",
        f"- Engineering scale mismatch found: `{c['engineering_scale_mismatch_found']}`",
        f"- Model output scale mismatch found: `{c['model_output_scale_mismatch_found']}`",
        f"- Mismatch horizons: `{c['mismatch_horizons']}`",
        "",
        c["summary"],
        "",
        "## Scale Ratios",
        "",
        "| Horizon | Realized std | Realized p99-p01 | Mean q90-q10 | Ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for h_key, stats in report["scale_diagnostics"]["by_horizon"].items():
        lines.append(
            f"| {h_key} | {stats['realized_std']} | "
            f"{stats['realized_p99_minus_p01']} | "
            f"{stats['prediction_mean_q90_minus_q10']} | "
            f"{stats['prediction_width_to_realized_p99_p01_ratio']} |"
        )
    lines.extend([
        "",
        "## Code Path",
        "",
        "| Stage | File | Behavior |",
        "|---|---|---|",
    ])
    for stage, item in report["code_path_audit"].items():
        lines.append(f"| {stage} | `{item['file']}` | {item['behavior']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_validation_md(report: dict, path: Path) -> None:
    lines = [
        "# M11 Model Quality Validation",
        "",
        _STAGE_NOTE,
        "",
        _BACKTEST_NOTE,
        "",
        f"- Overall status: `{report['overall_status']}`",
        "",
        "| Check | Status | Observed | Threshold |",
        "|---|---|---|---|",
    ]
    for check in report["checks"]:
        observed = json.dumps(m10._json_ready(check["observed"]), sort_keys=True)
        lines.append(
            f"| {check['name']} | `{check['status']}` | `{observed}` | "
            f"{check['threshold']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_calibration_comparison_md(report: dict, path: Path) -> None:
    raw = report["raw"]
    calibrated = report["calibrated"]
    rolling = report["rolling_historical_recomparison"]
    conclusion = report["conclusion"]
    lines = [
        "# M11 Calibration Comparison",
        "",
        _STAGE_NOTE,
        "",
        _BACKTEST_NOTE,
        "",
        "## Summary",
        "",
        f"- Raw pinball: {raw['pinball_loss']['overall']}",
        f"- Calibrated pinball: {calibrated['pinball_loss']['overall']}",
        f"- Raw coverage MAE: {raw['quantile_coverage']['overall_mae']}",
        f"- Calibrated coverage MAE: {calibrated['quantile_coverage']['overall_mae']}",
        f"- Coverage MAE reduction: {report['raw_vs_calibrated']['coverage_mae_delta_raw_minus_calibrated']}",
        f"- Champion status: `{conclusion['champion_status']}`",
        "",
        "## IC",
        "",
        "| Source | Max abs Pearson IC | Max abs Rank IC |",
        "|---|---:|---:|",
    ]
    for name, item in [("raw", raw), ("calibrated", calibrated)]:
        proxy = item["ic_metrics"]["materiality_proxy"]
        lines.append(
            f"| {name} | {proxy.get('max_abs_pearson_ic')} | "
            f"{proxy.get('max_abs_rank_ic')} |"
        )
    lines.extend([
        "",
        "## Rolling Historical Recomparison",
        "",
        f"- Common rows: {rolling['common_rows']}",
        f"- Raw model common pinball: {rolling['raw_model_on_common']['pinball_loss']['overall']}",
        f"- Calibrated model common pinball: {rolling['calibrated_model_on_common']['pinball_loss']['overall']}",
        f"- Rolling historical pinball: {rolling['rolling_historical_quantile']['pinball_loss']['overall']}",
        f"- Rolling minus calibrated pinball delta: {rolling['pinball_delta_rolling_minus_calibrated']}",
        "",
        "## Coverage By Horizon",
        "",
        "| Horizon | Raw MAE | Calibrated MAE |",
        "|---|---:|---:|",
    ])
    raw_h = raw["quantile_coverage"]["by_horizon"]
    cal_h = calibrated["quantile_coverage"]["by_horizon"]
    for h_key in raw_h:
        lines.append(
            f"| {h_key} | {raw_h[h_key]['coverage_calibration_mae']} | "
            f"{cal_h.get(h_key, {}).get('coverage_calibration_mae')} |"
        )
    lines.extend([
        "",
        "## Backtest Note",
        "",
        "The backtest matrix is included for evaluation handoff only and is not a full execution simulator.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    t0 = time.time()
    horizons = _parse_int_csv(args.horizons)
    quantiles = _parse_float_csv(args.quantiles)
    cost_bps_list = _parse_float_csv(args.cost_bps_list)
    threshold_specs = m10.parse_threshold_specs(args.thresholds)

    output_root = runtime_paths.resolve_output_root(args.output_root)
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = (
        Path(args.progress_log).expanduser().resolve()
        if args.progress_log
        else reports_dir / f"{args.out_prefix}_progress.jsonl"
    )
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text("", encoding="utf-8")

    m10_reports_dir = (
        Path(args.m10_reports_dir).expanduser().resolve()
        if args.m10_reports_dir
        else None
    )
    test_predictions_path = _resolve_input(
        args.test_predictions,
        m10_reports_dir=m10_reports_dir,
        filename="m9_predictions.parquet",
    )
    test_eval_rows_path = _resolve_input(
        args.test_eval_rows,
        m10_reports_dir=m10_reports_dir,
        filename="m10_eval_rows.parquet",
    )
    m10_metrics_path = _resolve_input(
        args.m10_metrics,
        m10_reports_dir=m10_reports_dir,
        filename="m10_prediction_eval_metrics.json",
    )
    m10_baseline_path = _resolve_input(
        args.m10_baseline_comparison,
        m10_reports_dir=m10_reports_dir,
        filename="m10_baseline_comparison.json",
    )

    print(f"\n{'='*60}")
    print("M11: Calibration + Target Scaling Fix")
    print(f"{'='*60}\n")
    print(f"Output root: {output_root}")
    print(f"Reports:     {reports_dir}")
    print(f"M10 reports: {m10_reports_dir}")
    print(f"Test rows:   {test_eval_rows_path}")
    print(f"Progress:    {progress_log_path}\n")

    _write_progress_event(progress_log_path, t0, {"event": "start"})
    print("Loading M10 test eval rows...", flush=True)
    test_eval_rows = pd.read_parquet(test_eval_rows_path)
    test_eval_rows["eob"] = pd.to_datetime(test_eval_rows["eob"])
    symbols = sorted(str(s) for s in test_eval_rows["symbol"].unique())
    inputs = {
        "test_predictions": str(test_predictions_path),
        "test_eval_rows": str(test_eval_rows_path),
        "m10_metrics": str(m10_metrics_path),
        "m10_baseline_comparison": str(m10_baseline_path),
        "val_predictions": args.val_predictions,
        "val_eval_rows": args.val_eval_rows,
        "data_dir": args.data_dir,
        "horizons": horizons,
        "quantiles": quantiles,
        "cost_bps_list": cost_bps_list,
        "thresholds": threshold_specs,
        "rolling_window_rows": int(args.rolling_window_rows),
        "min_rolling_samples": int(args.min_rolling_samples),
    }

    print("Building target scale audit...", flush=True)
    _write_progress_event(progress_log_path, t0, {"event": "target_scale_audit_start"})
    audit = build_target_scale_audit(
        test_predictions_path=test_predictions_path,
        test_eval_rows=test_eval_rows,
        data_dir=Path(args.data_dir),
        symbols=symbols,
        horizons=horizons,
        quantiles=quantiles,
        inputs=inputs,
    )
    _write_progress_event(progress_log_path, t0, {"event": "target_scale_audit_done"})

    print("Building model quality validation...", flush=True)
    m10_metrics = _load_json(m10_metrics_path)
    m10_baseline = _load_json(m10_baseline_path)
    quality = build_quality_validation(
        m10_metrics=m10_metrics,
        m10_baseline=m10_baseline,
        scale_ratios=audit["scale_diagnostics"]["by_horizon"],
    )

    if args.val_eval_rows:
        print("Loading validation eval rows...", flush=True)
        val_eval_rows = pd.read_parquet(Path(args.val_eval_rows).expanduser())
        val_alignment = {
            "source": "prealigned",
            "path": str(Path(args.val_eval_rows).expanduser()),
            "rows": int(len(val_eval_rows)),
        }
    else:
        if not args.val_predictions:
            raise SystemExit(
                "ERROR: --val-predictions or --val-eval-rows is required for M11 calibration"
            )
        print("Loading and aligning validation predictions...", flush=True)
        _write_progress_event(progress_log_path, t0, {"event": "val_alignment_start"})
        val_predictions = m10.load_predictions(
            Path(args.val_predictions).expanduser(),
            horizons,
            quantiles,
        )
        val_eval_rows, val_alignment = m10.align_predictions_with_realized(
            val_predictions,
            data_dir=args.data_dir,
            horizons=horizons,
            quantiles=quantiles,
            rolling_window_rows=args.rolling_window_rows,
            min_rolling_samples=args.min_rolling_samples,
            progress_log=progress_log_path,
            t0=t0,
        )
        val_eval_rows_path = reports_dir / f"{args.out_prefix}_val_eval_rows.parquet"
        val_eval_rows.to_parquet(val_eval_rows_path, index=False)
        val_alignment["val_eval_rows_path"] = str(val_eval_rows_path)
        _write_progress_event(progress_log_path, t0, {"event": "val_alignment_done", **val_alignment})

    if args.calibration_fit_split:
        print(
            f"Filtering validation fit rows to index_{args.calibration_fit_split}.parquet...",
            flush=True,
        )
        val_eval_rows, split_filter = filter_rows_to_split_index(
            val_eval_rows,
            data_dir=Path(args.data_dir),
            split=args.calibration_fit_split,
        )
    else:
        split_filter = {"split": None, "rows_before": int(len(val_eval_rows)), "rows_after": int(len(val_eval_rows))}
    if val_eval_rows.empty:
        raise SystemExit("ERROR: no validation rows remain for calibration fitting")

    print("Building calibration comparison...", flush=True)
    _write_progress_event(
        progress_log_path,
        t0,
        {
            "event": "calibration_comparison_start",
            "validation_fit_rows": int(len(val_eval_rows)),
            "test_rows": int(len(test_eval_rows)),
        },
    )
    calibration = build_calibration_comparison(
        val_rows=val_eval_rows,
        test_rows=test_eval_rows,
        horizons=horizons,
        quantiles=quantiles,
        cost_bps_list=cost_bps_list,
        threshold_specs=threshold_specs,
        progress_log=progress_log_path,
        t0=t0,
    )
    calibration["inputs"] = inputs
    calibration["validation_alignment"] = val_alignment
    calibration["validation_split_filter"] = split_filter
    _write_progress_event(progress_log_path, t0, {"event": "calibration_comparison_done"})

    audit_json = reports_dir / f"{args.out_prefix}_target_scale_audit.json"
    audit_md = reports_dir / f"{args.out_prefix}_target_scale_audit.md"
    quality_json = reports_dir / f"{args.out_prefix}_model_quality_validation.json"
    quality_md = reports_dir / f"{args.out_prefix}_model_quality_validation.md"
    calibration_json = reports_dir / f"{args.out_prefix}_calibration_comparison.json"
    calibration_md = reports_dir / f"{args.out_prefix}_calibration_comparison.md"

    print("Writing M11 reports...", flush=True)
    audit_json.write_text(json.dumps(m10._json_ready(audit), indent=2), encoding="utf-8")
    quality_json.write_text(json.dumps(m10._json_ready(quality), indent=2), encoding="utf-8")
    calibration_json.write_text(json.dumps(m10._json_ready(calibration), indent=2), encoding="utf-8")
    write_target_scale_audit_md(audit, audit_md)
    write_quality_validation_md(quality, quality_md)
    write_calibration_comparison_md(calibration, calibration_md)
    _write_progress_event(
        progress_log_path,
        t0,
        {
            "event": "done",
            "elapsed_seconds": round(time.time() - t0, 2),
            "reports": [
                str(audit_json),
                str(audit_md),
                str(quality_json),
                str(quality_md),
                str(calibration_json),
                str(calibration_md),
            ],
        },
    )

    print(f"  Wrote {audit_json}")
    print(f"  Wrote {audit_md}")
    print(f"  Wrote {quality_json}")
    print(f"  Wrote {quality_md}")
    print(f"  Wrote {calibration_json}")
    print(f"  Wrote {calibration_md}")
    print(f"\n{'='*60}")
    print("M11 Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
