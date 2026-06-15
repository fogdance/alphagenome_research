#!/usr/bin/env python3
"""M10: Evaluate M9 predictions against realized targets and baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from alphatrade import prediction_schema
from alphatrade import runtime_paths


_STAGE_NOTE = (
    "AlphaTrade is in active development. These reports are evaluation and "
    "productization gates only, not trading-readiness claims."
)
_M9_BACKTEST_NOTE = (
    "M9/M10 backtest outputs are lightweight handoff checks, not full "
    "execution simulators."
)


def parse_args():
    parser = argparse.ArgumentParser(description="M10 prediction evaluation and calibration")
    parser.add_argument("--predictions", type=str, default=None,
                        help="M9 predictions parquet (default: <reports-dir>/m9_predictions.parquet)")
    parser.add_argument("--data-dir", type=str, default="data/processed/m1_f8",
                        help="Processed bars root")
    parser.add_argument("--horizons", type=str, default="1,5,20,60",
                        help="Comma-separated horizons")
    parser.add_argument("--quantiles", type=str, default="0.1,0.3,0.5,0.7,0.9",
                        help="Comma-separated quantiles")
    parser.add_argument("--out-prefix", type=str, default="m10",
                        help="Output filename prefix")
    parser.add_argument("--cost-bps-list", type=str, default="0,1,2,5",
                        help="Comma-separated cost bps list")
    parser.add_argument("--thresholds", type=str, default="0,0.5sigma,1sigma",
                        help="Comma-separated thresholds: numeric or Ns sigma")
    parser.add_argument("--rolling-window-rows", type=int, default=7200,
                        help="Past-only rolling baseline window rows, default approx 20 trading days")
    parser.add_argument("--min-rolling-samples", type=int, default=10,
                        help="Minimum past samples for rolling baseline quantiles")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory")
    parser.add_argument("--progress-log", type=str, default=None,
                        help="JSONL progress log path (default: <reports-dir>/<out-prefix>_progress.jsonl)")
    parser.add_argument("--no-eval-rows", action="store_true",
                        help="Do not write aligned eval rows parquet")
    return parser.parse_args()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _parse_int_csv(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _q_key(q: float) -> str:
    return prediction_schema.quantile_label(q)


def _h_key(h: int) -> str:
    return f"h{int(h)}"


def _realized_col(h: int) -> str:
    return f"realized_h{int(h)}"


def _rolling_col(h: int, q: float) -> str:
    return f"rolling_h{int(h)}_{_q_key(q)}"


def _prediction_col(h: int, q: float) -> str:
    return prediction_schema.prediction_column(h, q)


def _finite(values) -> np.ndarray:
    return np.isfinite(np.asarray(values, dtype=np.float64))


def _safe_number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _summary(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "samples": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p05": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    p01, p05, p10, p50, p90, p95, p99 = np.percentile(
        arr, [1, 5, 10, 50, 90, 95, 99]
    )
    return {
        "samples": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p01": float(p01),
        "p05": float(p05),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
    }


def _safe_corr(x, y, *, rank: bool = False) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return None
    if rank:
        x_arr = pd.Series(x_arr).rank(method="average").to_numpy(dtype=np.float64)
        y_arr = pd.Series(y_arr).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(x_arr)) <= 0.0 or float(np.std(y_arr)) <= 0.0:
        return 0.0
    corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _pinball_losses(y, pred, q: float) -> np.ndarray:
    error = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return np.where(error >= 0.0, q * error, (q - 1.0) * error)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _safe_number(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


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
        "schema_version": "m10_progress_v1",
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "rss_mb": _process_rss_mb(),
        **event,
    }
    with progress_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_ready(payload), sort_keys=True) + "\n")


def parse_threshold_specs(raw: str) -> list[dict]:
    specs = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if item.endswith("sigma"):
            multiplier_raw = item[:-5]
            multiplier = 1.0 if multiplier_raw == "" else float(multiplier_raw)
            specs.append({"label": item, "kind": "sigma", "value": multiplier})
        else:
            value = float(item)
            label = str(int(value)) if value.is_integer() else str(value)
            specs.append({"label": label, "kind": "absolute", "value": value})
    return specs


def load_predictions(path: Path, horizons: list[int], quantiles: list[float]) -> pd.DataFrame:
    predictions = pd.read_parquet(path)
    issues = prediction_schema.validate_prediction_frame(
        predictions,
        horizons=horizons,
        quantiles=quantiles,
    )
    if issues:
        detail = "; ".join(f"{issue.field}: {issue.message}" for issue in issues)
        raise ValueError(f"M9 prediction schema validation failed: {detail}")
    predictions = predictions.copy()
    predictions["eob"] = pd.to_datetime(predictions["eob"])
    return predictions


def align_predictions_with_realized(
    predictions: pd.DataFrame,
    *,
    data_dir: str | os.PathLike[str],
    horizons: list[int],
    quantiles: list[float],
    rolling_window_rows: int,
    min_rolling_samples: int,
    progress_log: Path | None = None,
    t0: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    data_root = Path(data_dir)
    aligned = []
    missing_symbols = []
    missing_eob_rows = 0
    unavailable_targets = {_h_key(h): 0 for h in horizons}
    total_symbols = int(predictions["symbol"].nunique()) if "symbol" in predictions.columns else 0

    _write_progress_event(
        progress_log,
        t0,
        {
            "event": "alignment_start",
            "prediction_rows": int(len(predictions)),
            "symbols": total_symbols,
            "horizons": horizons,
            "rolling_window_rows": int(rolling_window_rows),
            "min_rolling_samples": int(min_rolling_samples),
        },
    )

    for symbol_idx, (symbol, symbol_preds) in enumerate(predictions.groupby("symbol", sort=True), start=1):
        symbol_t0 = time.time()
        print(
            f"  [alignment {symbol_idx}/{total_symbols}] {symbol}: "
            f"{len(symbol_preds):,} prediction rows",
            flush=True,
        )
        _write_progress_event(
            progress_log,
            t0,
            {
                "event": "alignment_symbol_start",
                "symbol": str(symbol),
                "symbol_index": symbol_idx,
                "total_symbols": total_symbols,
                "prediction_rows": int(len(symbol_preds)),
            },
        )
        bars_path = data_root / str(symbol) / "bars.parquet"
        if not bars_path.exists():
            missing_symbols.append(str(symbol))
            _write_progress_event(
                progress_log,
                t0,
                {
                    "event": "alignment_symbol_missing",
                    "symbol": str(symbol),
                    "symbol_index": symbol_idx,
                    "total_symbols": total_symbols,
                    "path": str(bars_path),
                },
            )
            continue

        bars = pd.read_parquet(bars_path)
        if "eob" not in bars.columns or "close" not in bars.columns:
            raise ValueError(f"{bars_path} must contain eob and close columns")
        bars = bars.copy()
        bars["eob"] = pd.to_datetime(bars["eob"])
        bars = bars.sort_values("eob").drop_duplicates("eob", keep="last").reset_index(drop=True)
        closes = bars["close"].astype(float)

        eob_to_idx = pd.Series(np.arange(len(bars), dtype=np.int64), index=bars["eob"])
        sdf = symbol_preds.sort_values("eob").copy()
        mapped_idx = sdf["eob"].map(eob_to_idx)
        missing_eob_rows += int(mapped_idx.isna().sum())
        sdf = sdf.loc[mapped_idx.notna()].copy()
        if sdf.empty:
            _write_progress_event(
                progress_log,
                t0,
                {
                    "event": "alignment_symbol_empty_after_eob_match",
                    "symbol": str(symbol),
                    "symbol_index": symbol_idx,
                    "total_symbols": total_symbols,
                    "bars_rows": int(len(bars)),
                    "missing_eob_rows_for_symbol": int(mapped_idx.isna().sum()),
                },
            )
            continue
        sdf["_bar_idx"] = mapped_idx.loc[mapped_idx.notna()].astype(int).to_numpy()

        for horizon in horizons:
            h = int(horizon)
            h_t0 = time.time()
            _write_progress_event(
                progress_log,
                t0,
                {
                    "event": "alignment_symbol_horizon_start",
                    "symbol": str(symbol),
                    "symbol_index": symbol_idx,
                    "total_symbols": total_symbols,
                    "horizon": h,
                    "bars_rows": int(len(bars)),
                    "matched_rows": int(len(sdf)),
                },
            )
            realized = np.log(closes.shift(-h) / closes)
            idx = sdf["_bar_idx"].to_numpy(dtype=np.int64)
            target_values = realized.iloc[idx].to_numpy(dtype=np.float64)
            has_future = idx + h < len(bars)
            target_values = np.where(has_future, target_values, np.nan)
            unavailable_targets[_h_key(h)] += int((~np.isfinite(target_values)).sum())
            sdf[_realized_col(h)] = target_values

            past_targets = realized.shift(h)
            rolling = past_targets.rolling(
                window=int(rolling_window_rows),
                min_periods=int(min_rolling_samples),
            )
            for quantile in quantiles:
                q_series = rolling.quantile(float(quantile))
                sdf[_rolling_col(h, quantile)] = q_series.iloc[idx].to_numpy(dtype=np.float64)
            _write_progress_event(
                progress_log,
                t0,
                {
                    "event": "alignment_symbol_horizon_done",
                    "symbol": str(symbol),
                    "symbol_index": symbol_idx,
                    "total_symbols": total_symbols,
                    "horizon": h,
                    "matched_rows": int(len(sdf)),
                    "unavailable_targets": int((~np.isfinite(target_values)).sum()),
                    "horizon_elapsed_seconds": round(time.time() - h_t0, 2),
                },
            )

        aligned.append(sdf.drop(columns=["_bar_idx"]))
        symbol_elapsed = time.time() - symbol_t0
        print(
            f"  [alignment {symbol_idx}/{total_symbols}] {symbol} done in "
            f"{symbol_elapsed:.1f}s",
            flush=True,
        )
        _write_progress_event(
            progress_log,
            t0,
            {
                "event": "alignment_symbol_done",
                "symbol": str(symbol),
                "symbol_index": symbol_idx,
                "total_symbols": total_symbols,
                "bars_rows": int(len(bars)),
                "matched_rows": int(len(sdf)),
                "symbol_elapsed_seconds": round(symbol_elapsed, 2),
            },
        )

    if aligned:
        eval_rows_raw = pd.concat(aligned, ignore_index=True)
        realized_cols = [_realized_col(h) for h in horizons]
        available_mask = pd.Series(True, index=eval_rows_raw.index)
        for col in realized_cols:
            available_mask &= _finite(eval_rows_raw[col])
        dropped_unavailable_target_rows = int((~available_mask).sum())
        eval_rows = eval_rows_raw.loc[available_mask].copy()
        eval_rows = eval_rows.sort_values(["symbol", "eob", "model_version"]).reset_index(drop=True)
    else:
        dropped_unavailable_target_rows = 0
        eval_rows = pd.DataFrame(columns=list(predictions.columns))

    metadata = {
        "prediction_rows": int(len(predictions)),
        "aligned_rows": int(len(eval_rows)),
        "dropped_unavailable_target_rows": int(dropped_unavailable_target_rows),
        "symbols": sorted(str(s) for s in eval_rows["symbol"].unique()) if not eval_rows.empty else [],
        "missing_symbols": missing_symbols,
        "missing_eob_rows": int(missing_eob_rows),
        "unavailable_targets_by_horizon": unavailable_targets,
        "rolling_window_rows": int(rolling_window_rows),
        "min_rolling_samples": int(min_rolling_samples),
    }
    _write_progress_event(
        progress_log,
        t0,
        {"event": "alignment_done", **metadata},
    )
    return eval_rows, metadata


def compute_pinball_loss(rows: pd.DataFrame, horizons: list[int], quantiles: list[float]) -> dict:
    def calc(frame: pd.DataFrame, horizon_subset: list[int]) -> tuple[float | None, int]:
        total = 0.0
        count = 0
        for h in horizon_subset:
            y_col = _realized_col(h)
            if y_col not in frame.columns:
                continue
            for q in quantiles:
                p_col = _prediction_col(h, q)
                mask = _finite(frame[y_col]) & _finite(frame[p_col])
                if not mask.any():
                    continue
                total += float(_pinball_losses(frame.loc[mask, y_col], frame.loc[mask, p_col], q).sum())
                count += int(mask.sum())
        return (float(total / count), count) if count else (None, 0)

    overall, overall_count = calc(rows, horizons)
    by_horizon = {}
    for h in horizons:
        value, count = calc(rows, [h])
        by_horizon[_h_key(h)] = {"loss": value, "loss_count": count}

    by_symbol = {}
    by_symbol_horizon = {}
    for symbol, sdf in rows.groupby("symbol", sort=True):
        value, count = calc(sdf, horizons)
        by_symbol[str(symbol)] = {"loss": value, "loss_count": count}
        by_symbol_horizon[str(symbol)] = {}
        for h in horizons:
            h_value, h_count = calc(sdf, [h])
            by_symbol_horizon[str(symbol)][_h_key(h)] = {
                "loss": h_value,
                "loss_count": h_count,
            }

    return {
        "overall": overall,
        "loss_count": overall_count,
        "by_horizon": by_horizon,
        "by_symbol": by_symbol,
        "by_symbol_horizon": by_symbol_horizon,
    }


def compute_quantile_coverage(rows: pd.DataFrame, horizons: list[int], quantiles: list[float]) -> dict:
    by_horizon = {}
    by_symbol_horizon = {}
    errors = []
    worst = {
        "horizon": None,
        "symbol": None,
        "quantile": None,
        "expected": None,
        "observed": None,
        "abs_error": -1.0,
    }

    def coverage_for(frame: pd.DataFrame, h: int, q: float) -> tuple[float | None, int]:
        y_col = _realized_col(h)
        p_col = _prediction_col(h, q)
        mask = _finite(frame[y_col]) & _finite(frame[p_col])
        if not mask.any():
            return None, 0
        observed = float((frame.loc[mask, y_col] <= frame.loc[mask, p_col]).mean())
        return observed, int(mask.sum())

    for h in horizons:
        h_errors = []
        q_metrics = {}
        for q in quantiles:
            observed, count = coverage_for(rows, h, q)
            abs_error = None if observed is None else abs(observed - float(q))
            if abs_error is not None:
                h_errors.append(abs_error)
                errors.append(abs_error)
                if abs_error > worst["abs_error"]:
                    worst = {
                        "horizon": _h_key(h),
                        "symbol": None,
                        "quantile": _q_key(q),
                        "expected": float(q),
                        "observed": observed,
                        "abs_error": float(abs_error),
                    }
            q_metrics[_q_key(q)] = {
                "expected": float(q),
                "observed": observed,
                "abs_error": None if abs_error is None else float(abs_error),
                "samples": count,
            }
        by_horizon[_h_key(h)] = {
            "coverage": q_metrics,
            "coverage_calibration_mae": float(np.mean(h_errors)) if h_errors else None,
        }

    worst_symbol = {
        "horizon": None,
        "symbol": None,
        "quantile": None,
        "expected": None,
        "observed": None,
        "abs_error": -1.0,
    }
    for symbol, sdf in rows.groupby("symbol", sort=True):
        by_symbol_horizon[str(symbol)] = {}
        for h in horizons:
            h_errors = []
            q_metrics = {}
            for q in quantiles:
                observed, count = coverage_for(sdf, h, q)
                abs_error = None if observed is None else abs(observed - float(q))
                if abs_error is not None:
                    h_errors.append(abs_error)
                    if abs_error > worst_symbol["abs_error"]:
                        worst_symbol = {
                            "horizon": _h_key(h),
                            "symbol": str(symbol),
                            "quantile": _q_key(q),
                            "expected": float(q),
                            "observed": observed,
                            "abs_error": float(abs_error),
                        }
                q_metrics[_q_key(q)] = {
                    "expected": float(q),
                    "observed": observed,
                    "abs_error": None if abs_error is None else float(abs_error),
                    "samples": count,
                }
            by_symbol_horizon[str(symbol)][_h_key(h)] = {
                "coverage": q_metrics,
                "coverage_calibration_mae": float(np.mean(h_errors)) if h_errors else None,
            }

    if worst["abs_error"] < 0:
        worst["abs_error"] = None
    if worst_symbol["abs_error"] < 0:
        worst_symbol["abs_error"] = None

    return {
        "overall_mae": float(np.mean(errors)) if errors else None,
        "by_horizon": by_horizon,
        "by_symbol_horizon": by_symbol_horizon,
        "worst": worst,
        "worst_symbol_horizon": worst_symbol,
    }


def compute_quantile_crossing(rows: pd.DataFrame, horizons: list[int], quantiles: list[float]) -> dict:
    total_crossings = 0
    total_pairs = 0
    by_horizon = {}
    for h in horizons:
        cols = [_prediction_col(h, q) for q in quantiles]
        frame = rows[cols].replace([np.inf, -np.inf], np.nan).dropna()
        pairs = int(len(frame) * (len(cols) - 1))
        crossings = int((frame.to_numpy(dtype=np.float64)[:, 1:] < frame.to_numpy(dtype=np.float64)[:, :-1]).sum()) if pairs else 0
        total_crossings += crossings
        total_pairs += pairs
        by_horizon[_h_key(h)] = {
            "rate": float(crossings / pairs) if pairs else None,
            "count": crossings,
            "pair_count": pairs,
        }
    return {
        "rate": float(total_crossings / total_pairs) if total_pairs else None,
        "count": total_crossings,
        "pair_count": total_pairs,
        "by_horizon": by_horizon,
    }


def compute_distribution_diagnostics(rows: pd.DataFrame, horizons: list[int], quantiles: list[float]) -> dict:
    realized_by_horizon = {}
    prediction_by_horizon = {}
    realized_by_symbol = {}
    prediction_by_symbol = {}

    for h in horizons:
        realized_by_horizon[_h_key(h)] = _summary(rows[_realized_col(h)])
        prediction_by_horizon[_h_key(h)] = {
            _q_key(q): _summary(rows[_prediction_col(h, q)]) for q in quantiles
        }

    for symbol, sdf in rows.groupby("symbol", sort=True):
        realized_by_symbol[str(symbol)] = {
            _h_key(h): _summary(sdf[_realized_col(h)]) for h in horizons
        }
        prediction_by_symbol[str(symbol)] = {
            _h_key(h): {
                _q_key(q): _summary(sdf[_prediction_col(h, q)]) for q in quantiles
            }
            for h in horizons
        }

    return {
        "realized_target": {
            "by_horizon": realized_by_horizon,
            "by_symbol": realized_by_symbol,
        },
        "prediction": {
            "by_horizon": prediction_by_horizon,
            "by_symbol": prediction_by_symbol,
        },
    }


def compute_ic_metrics(rows: pd.DataFrame, horizons: list[int]) -> dict:
    by_horizon = {}
    by_symbol = {}
    for h in horizons:
        y_col = _realized_col(h)
        p_col = _prediction_col(h, 0.5)
        mask = _finite(rows[y_col]) & _finite(rows[p_col])
        by_horizon[_h_key(h)] = {
            "pearson_ic": _safe_corr(rows.loc[mask, p_col], rows.loc[mask, y_col]),
            "rank_ic": _safe_corr(rows.loc[mask, p_col], rows.loc[mask, y_col], rank=True),
            "samples": int(mask.sum()),
        }

    for symbol, sdf in rows.groupby("symbol", sort=True):
        by_symbol[str(symbol)] = {}
        for h in horizons:
            y_col = _realized_col(h)
            p_col = _prediction_col(h, 0.5)
            mask = _finite(sdf[y_col]) & _finite(sdf[p_col])
            by_symbol[str(symbol)][_h_key(h)] = {
                "pearson_ic": _safe_corr(sdf.loc[mask, p_col], sdf.loc[mask, y_col]),
                "rank_ic": _safe_corr(sdf.loc[mask, p_col], sdf.loc[mask, y_col], rank=True),
                "samples": int(mask.sum()),
            }

    ic_values = [
        abs(v["pearson_ic"])
        for v in by_horizon.values()
        if v["pearson_ic"] is not None
    ]
    rank_values = [
        abs(v["rank_ic"])
        for v in by_horizon.values()
        if v["rank_ic"] is not None
    ]
    return {
        "by_horizon": by_horizon,
        "by_symbol": by_symbol,
        "materiality_proxy": {
            "threshold_abs_ic": 0.02,
            "max_abs_pearson_ic": max(ic_values) if ic_values else None,
            "max_abs_rank_ic": max(rank_values) if rank_values else None,
            "materially_different_from_zero": bool(
                (max(ic_values) if ic_values else 0.0) >= 0.02
                or (max(rank_values) if rank_values else 0.0) >= 0.02
            ),
        },
    }


def compute_direction_metrics(rows: pd.DataFrame, horizons: list[int]) -> dict:
    by_horizon = {}
    by_symbol = {}

    def calc(frame: pd.DataFrame, h: int) -> dict:
        y_col = _realized_col(h)
        p_col = _prediction_col(h, 0.5)
        mask = _finite(frame[y_col]) & _finite(frame[p_col])
        if not mask.any():
            return {
                "samples": 0,
                "hit_rate": None,
                "active_hit_rate": None,
                "long_count": 0,
                "short_count": 0,
                "near_zero_count": 0,
                "near_zero_fraction": None,
            }
        pred = frame.loc[mask, p_col].to_numpy(dtype=np.float64)
        realized = frame.loc[mask, y_col].to_numpy(dtype=np.float64)
        signal = np.sign(pred)
        target = np.sign(realized)
        active = signal != 0
        near_zero = np.abs(pred) <= 1e-12
        return {
            "samples": int(mask.sum()),
            "hit_rate": float((signal == target).mean()),
            "active_hit_rate": float((signal[active] == target[active]).mean()) if active.any() else None,
            "long_count": int((signal > 0).sum()),
            "short_count": int((signal < 0).sum()),
            "near_zero_count": int(near_zero.sum()),
            "near_zero_fraction": float(near_zero.mean()),
        }

    for h in horizons:
        by_horizon[_h_key(h)] = calc(rows, h)
    for symbol, sdf in rows.groupby("symbol", sort=True):
        by_symbol[str(symbol)] = {_h_key(h): calc(sdf, h) for h in horizons}
    return {"by_horizon": by_horizon, "by_symbol": by_symbol}


def _threshold_value(rows: pd.DataFrame, h: int, spec: dict) -> float:
    if spec["kind"] == "absolute":
        return float(spec["value"])
    q50 = rows[_prediction_col(h, 0.5)].to_numpy(dtype=np.float64)
    sigma = float(np.std(q50[np.isfinite(q50)], ddof=0)) if np.isfinite(q50).any() else 0.0
    return float(spec["value"]) * sigma


def _positions_for_variant(frame: pd.DataFrame, h: int, variant: str, threshold: float) -> np.ndarray:
    pos = np.zeros(len(frame), dtype=np.float64)
    q50 = frame[_prediction_col(h, 0.5)].to_numpy(dtype=np.float64)
    if variant == "q50_sign":
        pos[q50 > threshold] = 1.0
        pos[q50 < -threshold] = -1.0
        return pos

    if variant == "q30_q70_confirmed":
        q30 = frame[_prediction_col(h, 0.3)].to_numpy(dtype=np.float64)
        q70 = frame[_prediction_col(h, 0.7)].to_numpy(dtype=np.float64)
        pos[q30 > threshold] = 1.0
        pos[q70 < -threshold] = -1.0
        return pos

    if variant == "uncertainty_filtered_q50":
        width = (
            frame[_prediction_col(h, 0.9)].to_numpy(dtype=np.float64)
            - frame[_prediction_col(h, 0.1)].to_numpy(dtype=np.float64)
        )
        finite_width = width[np.isfinite(width)]
        max_width = float(np.median(finite_width)) if finite_width.size else -np.inf
        allowed = width <= max_width
        pos[(q50 > threshold) & allowed] = 1.0
        pos[(q50 < -threshold) & allowed] = -1.0
        return pos

    raise ValueError(f"unknown signal variant: {variant}")


def _max_drawdown(returns: np.ndarray) -> float | None:
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return None
    equity = np.cumsum(returns)
    running_max = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    drawdown = equity - running_max
    return float(drawdown.min())


def compute_backtest_matrix(
    rows: pd.DataFrame,
    *,
    horizons: list[int],
    cost_bps_list: list[float],
    threshold_specs: list[dict],
    source: str = "model",
    progress_log: Path | None = None,
    t0: float = 0.0,
) -> dict:
    variants = ["q50_sign", "q30_q70_confirmed", "uncertainty_filtered_q50"]
    matrix = []
    _write_progress_event(
        progress_log,
        t0,
        {
            "event": "backtest_matrix_start",
            "source": source,
            "rows": int(len(rows)),
            "horizons": horizons,
            "cost_bps_list": cost_bps_list,
            "thresholds": [spec["label"] for spec in threshold_specs],
        },
    )
    for h in horizons:
        h_t0 = time.time()
        print(f"  [backtest:{source}] horizon h{h} start", flush=True)
        _write_progress_event(
            progress_log,
            t0,
            {
                "event": "backtest_horizon_start",
                "source": source,
                "horizon": int(h),
                "rows": int(len(rows)),
            },
        )
        y_col = _realized_col(h)
        pred_cols = [_prediction_col(h, q) for q in [0.1, 0.3, 0.5, 0.7, 0.9]]
        base_mask = _finite(rows[y_col])
        for col in pred_cols:
            base_mask &= _finite(rows[col])
        base = rows.loc[base_mask].sort_values(["symbol", "eob"]).copy()
        if base.empty:
            _write_progress_event(
                progress_log,
                t0,
                {
                    "event": "backtest_horizon_empty",
                    "source": source,
                    "horizon": int(h),
                },
            )
            continue
        _write_progress_event(
            progress_log,
            t0,
            {
                "event": "backtest_horizon_base_ready",
                "source": source,
                "horizon": int(h),
                "base_rows": int(len(base)),
            },
        )

        for spec in threshold_specs:
            threshold = _threshold_value(base, h, spec)
            for variant in variants:
                variant_t0 = time.time()
                work = base[["symbol", "eob", y_col]].copy()
                work["position"] = _positions_for_variant(base, h, variant, threshold)
                turnovers = []
                for _, g in work.groupby("symbol", sort=True):
                    prev = 0.0
                    for pos in g["position"].to_numpy(dtype=np.float64):
                        turnovers.append(abs(pos - prev))
                        prev = pos
                work["turnover"] = turnovers
                realized = work[y_col].to_numpy(dtype=np.float64)
                position = work["position"].to_numpy(dtype=np.float64)
                turnover = work["turnover"].to_numpy(dtype=np.float64)
                active = position != 0
                gross = position * realized

                for cost_bps in cost_bps_list:
                    cost_rate = float(cost_bps) / 10000.0
                    cost = turnover * cost_rate
                    net = gross - cost
                    matrix.append(
                        {
                            "source": source,
                            "horizon": int(h),
                            "signal_variant": variant,
                            "cost_bps": float(cost_bps),
                            "threshold_label": spec["label"],
                            "threshold_value": float(threshold),
                            "samples": int(len(work)),
                            "gross_return": float(gross.sum()),
                            "net_return": float(net.sum()),
                            "annualized_return_if_available": None,
                            "max_drawdown": _max_drawdown(net),
                            "turnover": float(turnover.sum()),
                            "hit_rate": float((net[active] > 0).mean()) if active.any() else None,
                            "avg_trade_return": float(net[active].mean()) if active.any() else None,
                            "trade_count": int(active.sum()),
                        }
                    )
                _write_progress_event(
                    progress_log,
                    t0,
                    {
                        "event": "backtest_variant_done",
                        "source": source,
                        "horizon": int(h),
                        "signal_variant": variant,
                        "threshold_label": spec["label"],
                        "threshold_value": float(threshold),
                        "rows": int(len(work)),
                        "variant_elapsed_seconds": round(time.time() - variant_t0, 2),
                    },
                )
        print(
            f"  [backtest:{source}] horizon h{h} done in {time.time() - h_t0:.1f}s",
            flush=True,
        )
        _write_progress_event(
            progress_log,
            t0,
            {
                "event": "backtest_horizon_done",
                "source": source,
                "horizon": int(h),
                "horizon_elapsed_seconds": round(time.time() - h_t0, 2),
            },
        )

    best = max(matrix, key=lambda x: x["net_return"]) if matrix else None
    worst_drawdown = min(
        (m for m in matrix if m["max_drawdown"] is not None),
        key=lambda x: x["max_drawdown"],
        default=None,
    )
    result = {
        "matrix": matrix,
        "summary": {
            "rows": int(len(matrix)),
            "best_net_return": best,
            "worst_drawdown": worst_drawdown,
            "annualization_note": "annualized_return_if_available is null unless a full execution calendar is implemented",
            "backtest_note": _M9_BACKTEST_NOTE,
        },
    }
    _write_progress_event(
        progress_log,
        t0,
        {
            "event": "backtest_matrix_done",
            "source": source,
            "matrix_rows": int(len(matrix)),
        },
    )
    return result


def make_prediction_view(
    eval_rows: pd.DataFrame,
    *,
    source: str,
    horizons: list[int],
    quantiles: list[float],
) -> pd.DataFrame:
    view = eval_rows.copy()
    for h in horizons:
        for q in quantiles:
            target_col = _prediction_col(h, q)
            if source == "model":
                continue
            if source == "zero_return_quantile":
                view[target_col] = 0.0
            elif source == "rolling_historical_quantile":
                view[target_col] = view[_rolling_col(h, q)]
            elif source == "rolling_median_direction":
                view[target_col] = view[_rolling_col(h, 0.5)]
            else:
                raise ValueError(f"unknown prediction source: {source}")
    return view


def _baseline_available_mask(
    eval_rows: pd.DataFrame,
    *,
    source: str,
    horizons: list[int],
    quantiles: list[float],
) -> pd.Series:
    mask = pd.Series(True, index=eval_rows.index)
    if source == "zero_return_quantile":
        return mask
    if source == "rolling_median_direction":
        for h in horizons:
            mask &= _finite(eval_rows[_rolling_col(h, 0.5)])
        return mask
    if source == "rolling_historical_quantile":
        for h in horizons:
            for q in quantiles:
                mask &= _finite(eval_rows[_rolling_col(h, q)])
        return mask
    raise ValueError(f"unknown baseline: {source}")


def summarize_source(
    rows: pd.DataFrame,
    *,
    horizons: list[int],
    quantiles: list[float],
    cost_bps_list: list[float],
    threshold_specs: list[dict],
    source: str,
    progress_log: Path | None = None,
    t0: float = 0.0,
) -> dict:
    print(f"  [summary:{source}] start on {len(rows):,} rows", flush=True)
    _write_progress_event(
        progress_log,
        t0,
        {
            "event": "source_summary_start",
            "source": source,
            "rows": int(len(rows)),
        },
    )
    backtest = compute_backtest_matrix(
        rows,
        horizons=horizons,
        cost_bps_list=cost_bps_list,
        threshold_specs=threshold_specs,
        source=source,
        progress_log=progress_log,
        t0=t0,
    )
    selected_backtest = [
        item for item in backtest["matrix"]
        if item["signal_variant"] == "q50_sign" and item["threshold_label"] == "0"
    ]
    result = {
        "pinball_loss": compute_pinball_loss(rows, horizons, quantiles),
        "coverage_calibration_mae": compute_quantile_coverage(rows, horizons, quantiles)["overall_mae"],
        "ic_metrics": compute_ic_metrics(rows, horizons),
        "direction_metrics": compute_direction_metrics(rows, horizons),
        "selected_backtest": selected_backtest,
    }
    _write_progress_event(
        progress_log,
        t0,
        {
            "event": "source_summary_done",
            "source": source,
            "rows": int(len(rows)),
            "pinball_overall": result["pinball_loss"]["overall"],
            "coverage_calibration_mae": result["coverage_calibration_mae"],
        },
    )
    print(f"  [summary:{source}] done", flush=True)
    return result


def build_baseline_comparison(
    eval_rows: pd.DataFrame,
    *,
    horizons: list[int],
    quantiles: list[float],
    cost_bps_list: list[float],
    threshold_specs: list[dict],
    progress_log: Path | None = None,
    t0: float = 0.0,
) -> dict:
    baselines = {}
    deltas = {}
    _write_progress_event(
        progress_log,
        t0,
        {
            "event": "baseline_comparison_start",
            "rows": int(len(eval_rows)),
        },
    )
    model_view_all = make_prediction_view(
        eval_rows, source="model", horizons=horizons, quantiles=quantiles
    )
    model_summary = summarize_source(
        model_view_all,
        horizons=horizons,
        quantiles=quantiles,
        cost_bps_list=cost_bps_list,
        threshold_specs=threshold_specs,
        source="model",
        progress_log=progress_log,
        t0=t0,
    )

    for baseline in [
        "zero_return_quantile",
        "rolling_historical_quantile",
        "rolling_median_direction",
    ]:
        baseline_t0 = time.time()
        print(f"  [baseline:{baseline}] start", flush=True)
        _write_progress_event(
            progress_log,
            t0,
            {
                "event": "baseline_start",
                "baseline": baseline,
                "rows_before_mask": int(len(eval_rows)),
            },
        )
        mask = _baseline_available_mask(
            eval_rows, source=baseline, horizons=horizons, quantiles=quantiles
        )
        common = eval_rows.loc[mask].copy()
        _write_progress_event(
            progress_log,
            t0,
            {
                "event": "baseline_common_rows_ready",
                "baseline": baseline,
                "common_rows": int(len(common)),
            },
        )
        baseline_view = make_prediction_view(
            common, source=baseline, horizons=horizons, quantiles=quantiles
        )
        model_common = make_prediction_view(
            common, source="model", horizons=horizons, quantiles=quantiles
        )
        baseline_summary = summarize_source(
            baseline_view,
            horizons=horizons,
            quantiles=quantiles,
            cost_bps_list=cost_bps_list,
            threshold_specs=threshold_specs,
            source=baseline,
            progress_log=progress_log,
            t0=t0,
        )
        model_common_summary = summarize_source(
            model_common,
            horizons=horizons,
            quantiles=quantiles,
            cost_bps_list=cost_bps_list,
            threshold_specs=threshold_specs,
            source="model_common",
            progress_log=progress_log,
            t0=t0,
        )
        baselines[baseline] = {
            "common_rows": int(len(common)),
            **baseline_summary,
        }
        base_pinball = baseline_summary["pinball_loss"]["overall"]
        model_pinball = model_common_summary["pinball_loss"]["overall"]
        base_calib = baseline_summary["coverage_calibration_mae"]
        model_calib = model_common_summary["coverage_calibration_mae"]
        deltas[baseline] = {
            "common_rows": int(len(common)),
            "pinball_improvement_baseline_minus_model": (
                None if base_pinball is None or model_pinball is None else float(base_pinball - model_pinball)
            ),
            "coverage_mae_delta_model_minus_baseline": (
                None if base_calib is None or model_calib is None else float(model_calib - base_calib)
            ),
            "model_on_common": model_common_summary,
        }
        _write_progress_event(
            progress_log,
            t0,
            {
                "event": "baseline_done",
                "baseline": baseline,
                "common_rows": int(len(common)),
                "baseline_elapsed_seconds": round(time.time() - baseline_t0, 2),
            },
        )
        print(f"  [baseline:{baseline}] done in {time.time() - baseline_t0:.1f}s", flush=True)

    result = {
        "model": model_summary,
        "baselines": baselines,
        "deltas": deltas,
    }
    _write_progress_event(
        progress_log,
        t0,
        {
            "event": "baseline_comparison_done",
            "baselines": list(baselines.keys()),
        },
    )
    return result


def _top_backtest_rows(matrix: list[dict], limit: int = 12) -> list[dict]:
    return sorted(matrix, key=lambda x: x["net_return"], reverse=True)[:limit]


def write_prediction_eval_md(metrics: dict, path: Path) -> None:
    coverage = metrics["quantile_coverage"]
    ic = metrics["ic_metrics"]["materiality_proxy"]
    lines = [
        "# M10 Prediction Evaluation Metrics",
        "",
        _STAGE_NOTE,
        "",
        _M9_BACKTEST_NOTE,
        "",
        "## Summary",
        "",
        f"- Rows aligned: {metrics['data']['aligned_rows']:,}",
        f"- Pinball overall: {metrics['pinball_loss']['overall']}",
        f"- Coverage MAE: {coverage['overall_mae']}",
        f"- Worst calibration: {coverage['worst']}",
        f"- Worst symbol/horizon calibration: {coverage['worst_symbol_horizon']}",
        f"- Quantile crossing rate: {metrics['quantile_crossing']['rate']}",
        f"- IC materiality proxy: {ic}",
        "",
        "## By Horizon",
        "",
        "| Horizon | Pinball | Coverage MAE | Pearson IC | Rank IC | Direction Hit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for h_key, pinball in metrics["pinball_loss"]["by_horizon"].items():
        cov = coverage["by_horizon"].get(h_key, {})
        ic_h = metrics["ic_metrics"]["by_horizon"].get(h_key, {})
        direction = metrics["direction_metrics"]["by_horizon"].get(h_key, {})
        lines.append(
            f"| {h_key} | {pinball['loss']} | {cov.get('coverage_calibration_mae')} | "
            f"{ic_h.get('pearson_ic')} | {ic_h.get('rank_ic')} | {direction.get('hit_rate')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_backtest_md(report: dict, path: Path) -> None:
    lines = [
        "# M10 Backtest Matrix",
        "",
        _STAGE_NOTE,
        "",
        _M9_BACKTEST_NOTE,
        "",
        "## Top Net Return Rows",
        "",
        "| Horizon | Variant | Cost bps | Threshold | Net | Drawdown | Turnover | Trades |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in _top_backtest_rows(report["matrix"]):
        lines.append(
            f"| {row['horizon']} | {row['signal_variant']} | {row['cost_bps']} | "
            f"{row['threshold_label']} | {row['net_return']} | {row['max_drawdown']} | "
            f"{row['turnover']} | {row['trade_count']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _selected_net(summary: dict, cost_bps: float = 0.0) -> float | None:
    for row in summary.get("selected_backtest", []):
        if float(row.get("cost_bps", -1)) == float(cost_bps):
            return row.get("net_return")
    return None


def write_baseline_md(report: dict, path: Path) -> None:
    lines = [
        "# M10 Baseline Comparison",
        "",
        _STAGE_NOTE,
        "",
        _M9_BACKTEST_NOTE,
        "",
        "## Summary",
        "",
        "| Baseline | Common Rows | Beats Pinball | Pinball Improvement | Better Coverage MAE | Coverage MAE Delta | Model Net@0bps | Baseline Net@0bps |",
        "|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for name, baseline in report["baselines"].items():
        delta = report["deltas"][name]
        model_net = _selected_net(delta["model_on_common"], cost_bps=0.0)
        baseline_net = _selected_net(baseline, cost_bps=0.0)
        pinball_delta = delta["pinball_improvement_baseline_minus_model"]
        coverage_delta = delta["coverage_mae_delta_model_minus_baseline"]
        beats_pinball = "yes" if pinball_delta is not None and pinball_delta > 0 else "no"
        better_coverage = "yes" if coverage_delta is not None and coverage_delta < 0 else "no"
        lines.append(
            f"| {name} | {baseline['common_rows']} | "
            f"{beats_pinball} | "
            f"{pinball_delta} | "
            f"{better_coverage} | "
            f"{coverage_delta} | "
            f"{model_net} | {baseline_net} |"
        )
    lines.extend([
        "",
        "Positive pinball improvement means the model has lower pinball loss than the baseline on the same rows.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    t0 = time.time()
    horizons = _parse_int_csv(args.horizons)
    quantiles = _parse_float_csv(args.quantiles)
    cost_bps_list = _parse_float_csv(args.cost_bps_list)
    threshold_specs = parse_threshold_specs(args.thresholds)

    output_root = runtime_paths.resolve_output_root(args.output_root)
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    predictions_path = (
        Path(args.predictions).expanduser()
        if args.predictions
        else reports_dir / "m9_predictions.parquet"
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = (
        Path(args.progress_log).expanduser()
        if args.progress_log
        else reports_dir / f"{args.out_prefix}_progress.jsonl"
    )
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text("", encoding="utf-8")

    print(f"\n{'='*60}")
    print("M10: Prediction Evaluation + Calibration")
    print(f"{'='*60}\n")
    print(f"Output root: {output_root}")
    print(f"Reports:     {reports_dir}")
    print(f"Predictions: {predictions_path}")
    print(f"Data dir:    {args.data_dir}")
    print(f"Progress:    {progress_log_path}\n")

    _write_progress_event(
        progress_log_path,
        t0,
        {
            "event": "start",
            "predictions": str(predictions_path),
            "data_dir": args.data_dir,
            "horizons": horizons,
            "quantiles": quantiles,
            "cost_bps_list": cost_bps_list,
            "thresholds": threshold_specs,
            "rolling_window_rows": int(args.rolling_window_rows),
            "min_rolling_samples": int(args.min_rolling_samples),
        },
    )

    print("Loading predictions...", flush=True)
    _write_progress_event(progress_log_path, t0, {"event": "predictions_load_start"})
    predictions = load_predictions(predictions_path, horizons, quantiles)
    _write_progress_event(
        progress_log_path,
        t0,
        {
            "event": "predictions_load_done",
            "prediction_rows": int(len(predictions)),
            "symbols": int(predictions["symbol"].nunique()),
        },
    )
    print(
        f"Loaded {len(predictions):,} prediction rows across "
        f"{predictions['symbol'].nunique()} symbols.",
        flush=True,
    )

    eval_rows, alignment = align_predictions_with_realized(
        predictions,
        data_dir=args.data_dir,
        horizons=horizons,
        quantiles=quantiles,
        rolling_window_rows=args.rolling_window_rows,
        min_rolling_samples=args.min_rolling_samples,
        progress_log=progress_log_path,
        t0=t0,
    )
    if eval_rows.empty:
        sys.exit("ERROR: M10 evaluation produced zero aligned rows")

    _write_progress_event(
        progress_log_path,
        t0,
        {
            "event": "model_view_start",
            "aligned_rows": int(len(eval_rows)),
        },
    )
    model_rows = make_prediction_view(
        eval_rows, source="model", horizons=horizons, quantiles=quantiles
    )
    _write_progress_event(progress_log_path, t0, {"event": "model_view_done"})
    eval_rows_path = reports_dir / f"{args.out_prefix}_eval_rows.parquet"
    if not args.no_eval_rows:
        print(f"Writing eval rows parquet: {eval_rows_path}", flush=True)
        _write_progress_event(
            progress_log_path,
            t0,
            {
                "event": "eval_rows_write_start",
                "path": str(eval_rows_path),
                "rows": int(len(eval_rows)),
            },
        )
        eval_rows.to_parquet(eval_rows_path, index=False)
        _write_progress_event(
            progress_log_path,
            t0,
            {
                "event": "eval_rows_write_done",
                "path": str(eval_rows_path),
                "rows": int(len(eval_rows)),
            },
        )

    inputs = {
        "predictions": str(predictions_path),
        "data_dir": args.data_dir,
        "horizons": horizons,
        "quantiles": quantiles,
        "cost_bps_list": cost_bps_list,
        "thresholds": threshold_specs,
        "rolling_window_rows": int(args.rolling_window_rows),
        "min_rolling_samples": int(args.min_rolling_samples),
    }
    git_sha = get_git_sha()

    print("Computing M10 metrics...", flush=True)
    metric_stages: dict[str, Callable[[], dict]] = {
        "pinball_loss": lambda: compute_pinball_loss(model_rows, horizons, quantiles),
        "quantile_coverage": lambda: compute_quantile_coverage(model_rows, horizons, quantiles),
        "quantile_crossing": lambda: compute_quantile_crossing(model_rows, horizons, quantiles),
        "distribution_diagnostics": lambda: compute_distribution_diagnostics(model_rows, horizons, quantiles),
        "ic_metrics": lambda: compute_ic_metrics(model_rows, horizons),
        "direction_metrics": lambda: compute_direction_metrics(model_rows, horizons),
    }
    metric_results = {}
    for stage_name, stage_fn in metric_stages.items():
        stage_t0 = time.time()
        print(f"  [metrics] {stage_name} start", flush=True)
        _write_progress_event(
            progress_log_path,
            t0,
            {"event": "metric_stage_start", "stage": stage_name},
        )
        metric_results[stage_name] = stage_fn()
        _write_progress_event(
            progress_log_path,
            t0,
            {
                "event": "metric_stage_done",
                "stage": stage_name,
                "stage_elapsed_seconds": round(time.time() - stage_t0, 2),
            },
        )
        print(f"  [metrics] {stage_name} done in {time.time() - stage_t0:.1f}s", flush=True)

    eval_metrics = {
        "schema_version": "m10_prediction_eval_metrics_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "stage": "development_evaluation",
        "notes": [_STAGE_NOTE, _M9_BACKTEST_NOTE],
        "inputs": inputs,
        "data": alignment,
        "pinball_loss": metric_results["pinball_loss"],
        "quantile_coverage": metric_results["quantile_coverage"],
        "quantile_crossing": metric_results["quantile_crossing"],
        "distribution_diagnostics": metric_results["distribution_diagnostics"],
        "ic_metrics": metric_results["ic_metrics"],
        "direction_metrics": metric_results["direction_metrics"],
        "output": {
            "eval_rows_path": str(eval_rows_path) if not args.no_eval_rows else "",
            "elapsed_seconds": round(time.time() - t0, 2),
        },
    }

    print("Computing model backtest matrix...", flush=True)
    backtest = compute_backtest_matrix(
        model_rows,
        horizons=horizons,
        cost_bps_list=cost_bps_list,
        threshold_specs=threshold_specs,
        source="model",
        progress_log=progress_log_path,
        t0=t0,
    )
    backtest_report = {
        "schema_version": "m10_backtest_matrix_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "stage": "development_evaluation",
        "notes": [_STAGE_NOTE, _M9_BACKTEST_NOTE],
        "inputs": inputs,
        **backtest,
    }

    print("Computing baseline comparison...", flush=True)
    baseline_parts = build_baseline_comparison(
        eval_rows,
        horizons=horizons,
        quantiles=quantiles,
        cost_bps_list=cost_bps_list,
        threshold_specs=threshold_specs,
        progress_log=progress_log_path,
        t0=t0,
    )
    baseline_report = {
        "schema_version": "m10_baseline_comparison_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "stage": "development_evaluation",
        "notes": [_STAGE_NOTE, _M9_BACKTEST_NOTE],
        "inputs": inputs,
        **baseline_parts,
    }

    eval_json = reports_dir / f"{args.out_prefix}_prediction_eval_metrics.json"
    eval_md = reports_dir / f"{args.out_prefix}_prediction_eval_metrics.md"
    backtest_json = reports_dir / f"{args.out_prefix}_backtest_matrix.json"
    backtest_md = reports_dir / f"{args.out_prefix}_backtest_matrix.md"
    baseline_json = reports_dir / f"{args.out_prefix}_baseline_comparison.json"
    baseline_md = reports_dir / f"{args.out_prefix}_baseline_comparison.md"

    _write_progress_event(progress_log_path, t0, {"event": "reports_write_start"})
    eval_json.write_text(json.dumps(_json_ready(eval_metrics), indent=2), encoding="utf-8")
    backtest_json.write_text(json.dumps(_json_ready(backtest_report), indent=2), encoding="utf-8")
    baseline_json.write_text(json.dumps(_json_ready(baseline_report), indent=2), encoding="utf-8")
    write_prediction_eval_md(eval_metrics, eval_md)
    write_backtest_md(backtest_report, backtest_md)
    write_baseline_md(baseline_report, baseline_md)
    _write_progress_event(
        progress_log_path,
        t0,
        {
            "event": "reports_write_done",
            "reports": [
                str(eval_json),
                str(eval_md),
                str(backtest_json),
                str(backtest_md),
                str(baseline_json),
                str(baseline_md),
            ],
        },
    )

    print(f"  Wrote {eval_json}")
    print(f"  Wrote {eval_md}")
    print(f"  Wrote {backtest_json}")
    print(f"  Wrote {backtest_md}")
    print(f"  Wrote {baseline_json}")
    print(f"  Wrote {baseline_md}")
    if not args.no_eval_rows:
        print(f"  Wrote {eval_rows_path}")
    _write_progress_event(
        progress_log_path,
        t0,
        {
            "event": "done",
            "elapsed_seconds": round(time.time() - t0, 2),
            "aligned_rows": int(len(eval_rows)),
        },
    )
    print(f"\n{'='*60}")
    print("M10 Evaluation Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
