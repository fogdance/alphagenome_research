#!/usr/bin/env python3
"""M9: Lightweight backtest integration for stable prediction parquet."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alphatrade import prediction_schema
from alphatrade import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="M9: Backtest predictions parquet")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Predictions parquet (default: <reports-dir>/m9_predictions.parquet)")
    parser.add_argument("--data-dir", type=str, default="data/processed/m1_f8",
                        help="Path to processed data directory")
    parser.add_argument("--horizon", type=int, default=20,
                        help="Prediction horizon to backtest")
    parser.add_argument("--quantile", type=float, default=0.5,
                        help="Prediction quantile to use as signal")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="No-trade band around zero forecast log return")
    parser.add_argument("--cost-bps", type=float, default=0.0,
                        help="One-way turnover cost in basis points")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory")
    parser.add_argument("--output-metrics", type=str, default=None,
                        help="Metrics JSON path (default: <reports-dir>/m9_backtest_metrics.json)")
    parser.add_argument("--output-md", type=str, default=None,
                        help="Markdown path (default: <reports-dir>/m9_backtest_metrics.md)")
    parser.add_argument("--output-trades", type=str, default=None,
                        help="Per-sample trades parquet path (default: <reports-dir>/m9_backtest_trades.parquet)")
    return parser.parse_args()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _position_from_forecast(forecast: float, threshold: float) -> float:
    if forecast > threshold:
        return 1.0
    if forecast < -threshold:
        return -1.0
    return 0.0


def _sharpe_like(values: pd.Series) -> float:
    std = float(values.std(ddof=0))
    if std <= 0:
        return 0.0
    return float(values.mean() / std * np.sqrt(len(values)))


def _performance_summary(df: pd.DataFrame) -> dict:
    return {
        "sum_gross_log_return": float(df["gross_log_return"].sum()),
        "sum_net_log_return": float(df["net_log_return"].sum()),
        "mean_gross_log_return": float(df["gross_log_return"].mean()),
        "mean_net_log_return": float(df["net_log_return"].mean()),
        "win_rate": float((df["net_log_return"] > 0).mean()),
        "direction_hit_rate": float((np.sign(df["forecast"]) == np.sign(df["realized_log_return"])).mean()),
        "mean_turnover": float(df["turnover"].mean()),
        "sharpe_like": _sharpe_like(df["net_log_return"]),
    }


def compute_backtest(
    predictions: pd.DataFrame,
    *,
    data_dir: str | os.PathLike[str],
    horizon: int,
    quantile: float,
    threshold: float,
    cost_bps: float,
) -> tuple[dict, pd.DataFrame]:
    """Compute a deterministic one-bar-position backtest from predictions."""
    signal_col = prediction_schema.prediction_column(horizon, quantile)
    if signal_col not in predictions.columns:
        raise ValueError(f"missing signal column: {signal_col}")
    if "symbol" not in predictions.columns or "eob" not in predictions.columns:
        raise ValueError("predictions must contain symbol and eob columns")

    pred_df = predictions.copy()
    pred_df["eob"] = pd.to_datetime(pred_df["eob"])
    data_root = Path(data_dir)
    cost_rate = float(cost_bps) / 10000.0

    rows = []
    missing_symbols = []
    skipped = {
        "missing_eob": 0,
        "insufficient_future": 0,
        "invalid_close": 0,
    }

    for symbol, symbol_preds in pred_df.groupby("symbol", sort=True):
        bars_path = data_root / str(symbol) / "bars.parquet"
        if not bars_path.exists():
            missing_symbols.append(str(symbol))
            continue

        bars = pd.read_parquet(bars_path)
        for required_col in ("eob", "close"):
            if required_col not in bars.columns:
                raise ValueError(f"{bars_path} missing required column: {required_col}")
        bars = bars.copy()
        bars["eob"] = pd.to_datetime(bars["eob"])
        bars = bars.sort_values("eob").drop_duplicates("eob", keep="last").reset_index(drop=True)
        eob_to_idx = {eob: idx for idx, eob in enumerate(bars["eob"])}

        prev_position = 0.0
        for _, pred in symbol_preds.sort_values("eob").iterrows():
            idx = eob_to_idx.get(pred["eob"])
            if idx is None:
                skipped["missing_eob"] += 1
                continue
            future_idx = idx + int(horizon)
            if future_idx >= len(bars):
                skipped["insufficient_future"] += 1
                continue

            close_now = float(bars.at[idx, "close"])
            close_future = float(bars.at[future_idx, "close"])
            if close_now <= 0 or close_future <= 0:
                skipped["invalid_close"] += 1
                continue

            forecast = float(pred[signal_col])
            position = _position_from_forecast(forecast, threshold)
            realized = float(np.log(close_future / close_now))
            turnover = abs(position - prev_position)
            prev_position = position

            gross = position * realized
            cost = turnover * cost_rate
            rows.append(
                {
                    "symbol": str(symbol),
                    "eob": pred["eob"],
                    "model_version": str(pred.get("model_version", "")),
                    "forecast": forecast,
                    "realized_log_return": realized,
                    "position": position,
                    "turnover": turnover,
                    "gross_log_return": gross,
                    "cost_log_return": cost,
                    "net_log_return": gross - cost,
                }
            )

    trades = pd.DataFrame(rows)
    if trades.empty:
        raise ValueError("backtest produced zero matched samples")

    by_symbol = []
    for symbol, sdf in trades.groupby("symbol", sort=True):
        summary = _performance_summary(sdf)
        by_symbol.append({"symbol": symbol, "n_samples": int(len(sdf)), **summary})

    model_versions = sorted(str(v) for v in pred_df["model_version"].dropna().unique())
    metrics = {
        "schema_version": "m9_backtest_metrics_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "predictions_path": "",
        "data_dir": str(data_root),
        "model_versions": model_versions,
        "config": {
            "horizon": int(horizon),
            "quantile": float(quantile),
            "signal_column": signal_col,
            "threshold": float(threshold),
            "cost_bps": float(cost_bps),
        },
        "data": {
            "n_predictions": int(len(pred_df)),
            "n_matched": int(len(trades)),
            "n_symbols": int(trades["symbol"].nunique()),
            "missing_symbols": missing_symbols,
            "skipped": skipped,
        },
        "performance": _performance_summary(trades),
        "by_symbol": by_symbol,
        "output": {
            "trades_path": "",
            "trades_rows": int(len(trades)),
        },
    }
    return metrics, trades


def write_markdown(metrics: dict, path: Path) -> None:
    perf = metrics["performance"]
    lines = [
        "# M9 Backtest Metrics",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{metrics['predictions_path']}`",
        f"- Data dir: `{metrics['data_dir']}`",
        f"- Model versions: `{', '.join(metrics['model_versions'])}`",
        "",
        "## Config",
        "",
        f"- Signal: `{metrics['config']['signal_column']}`",
        f"- Threshold: {metrics['config']['threshold']}",
        f"- Cost bps: {metrics['config']['cost_bps']}",
        "",
        "## Data",
        "",
        f"- Predictions: {metrics['data']['n_predictions']:,}",
        f"- Matched samples: {metrics['data']['n_matched']:,}",
        f"- Symbols: {metrics['data']['n_symbols']}",
        f"- Missing symbols: {metrics['data']['missing_symbols']}",
        f"- Skipped: {metrics['data']['skipped']}",
        "",
        "## Performance",
        "",
        f"- Sum net log return: {perf['sum_net_log_return']:.8f}",
        f"- Mean net log return: {perf['mean_net_log_return']:.8f}",
        f"- Win rate: {perf['win_rate']:.4f}",
        f"- Direction hit rate: {perf['direction_hit_rate']:.4f}",
        f"- Mean turnover: {perf['mean_turnover']:.4f}",
        f"- Sharpe-like: {perf['sharpe_like']:.4f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_root = runtime_paths.resolve_output_root(args.output_root)
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    predictions_path = (
        Path(args.predictions).expanduser()
        if args.predictions
        else reports_dir / "m9_predictions.parquet"
    )
    metrics_path = (
        Path(args.output_metrics).expanduser()
        if args.output_metrics
        else reports_dir / "m9_backtest_metrics.json"
    )
    md_path = (
        Path(args.output_md).expanduser()
        if args.output_md
        else reports_dir / "m9_backtest_metrics.md"
    )
    trades_path = (
        Path(args.output_trades).expanduser()
        if args.output_trades
        else reports_dir / "m9_backtest_trades.parquet"
    )

    print(f"\n{'='*60}")
    print("M9: Backtest Predictions")
    print(f"{'='*60}\n")
    print(f"Output root: {output_root}")
    print(f"Reports:     {reports_dir}")
    print(f"Predictions: {predictions_path}\n")

    if not predictions_path.exists():
        sys.exit(f"ERROR: predictions not found: {predictions_path}")

    predictions = pd.read_parquet(predictions_path)
    issues = prediction_schema.validate_prediction_frame(predictions)
    if issues:
        detail = "; ".join(f"{issue.field}: {issue.message}" for issue in issues)
        sys.exit(f"ERROR: predictions schema validation failed: {detail}")

    try:
        metrics, trades = compute_backtest(
            predictions,
            data_dir=args.data_dir,
            horizon=args.horizon,
            quantile=args.quantile,
            threshold=args.threshold,
            cost_bps=args.cost_bps,
        )
    except Exception as e:
        sys.exit(f"ERROR: backtest failed: {e}")

    metrics["predictions_path"] = str(predictions_path)
    metrics["output"]["trades_path"] = str(trades_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    trades.to_parquet(trades_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_markdown(metrics, md_path)

    print(f"  Wrote {metrics_path}")
    print(f"  Wrote {md_path}")
    print(f"  Wrote {trades_path} ({len(trades)} rows)")
    print(f"\n{'='*60}")
    print("M9 Backtest Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
