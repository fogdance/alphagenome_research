#!/usr/bin/env python3
"""
M9: Offline batch inference.

Loads a champion model bundle, runs inference on bars.parquet data sliced by
timestamp range, and outputs predictions.parquet in wide format.

Usage:
    python src/alphatrade/scripts/batch_infer_offline.py \
      --bundle <runs>/artifacts/model_bundle/<model_version> \
      --data-dir data/processed/m1_f8 \
      --symbols DCE.JM,SHFE.AG \
      --start 2024-01-02 --end 2024-01-04 \
      --output <runs>/reports/m9_predictions.parquet \
      [--batch-size 256] [--smoke]
"""

import os as _os
import sys as _sys

_DEFAULT_XLA_FLAGS = "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="
_DEFAULT_INFER_BATCH_SIZE = int(_os.environ.get("ALPHATRADE_M9_INFER_BATCH_SIZE", "2048"))
_DEFAULT_PROGRESS_EVERY_BATCHES = int(_os.environ.get("ALPHATRADE_M9_PROGRESS_EVERY_BATCHES", "10"))

if "ALPHATRADE_KEEP_LD_LIBRARY_PATH" not in _os.environ:
    _os.environ.pop("LD_LIBRARY_PATH", None)


def _requested_device_before_jax_import(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            return _normalize_requested_device(argv[i + 1])
        if arg.startswith("--device="):
            return _normalize_requested_device(arg.split("=", 1)[1])
    return _normalize_requested_device(_os.environ.get("ALPHATRADE_DEVICE", "gpu"))


def _normalize_requested_device(value: str) -> str:
    value = value.lower()
    if value in ("gpu", "cuda", "auto"):
        return "gpu"
    if value == "cpu":
        return "cpu"
    return "gpu"


_requested_device = _requested_device_before_jax_import(_sys.argv)
if "JAX_PLATFORMS" not in _os.environ:
    if _requested_device in ("gpu", "cuda"):
        _os.environ["JAX_PLATFORMS"] = "cuda"
        _os.environ.setdefault("XLA_FLAGS", _DEFAULT_XLA_FLAGS)
        _os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    elif _requested_device == "cpu":
        _os.environ["JAX_PLATFORMS"] = "cpu"
elif _requested_device in ("gpu", "cuda"):
    _os.environ.setdefault("XLA_FLAGS", _DEFAULT_XLA_FLAGS)
    _os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphatrade import prediction_schema
from alphatrade import runtime_paths
from alphatrade.inference import AlphaTradeBundlePredictor, load_bundle_metadata
from data_pipeline.feature_schema import FEATURE_COLS, FEATURE_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="M9: Offline batch inference")
    parser.add_argument("--bundle", type=str, required=True,
                        help="Path to model bundle directory")
    parser.add_argument("--data-dir", type=str, default="data/processed/m1_f8",
                        help="Path to processed data directory")
    parser.add_argument("--symbols", type=str, required=True,
                        help="Comma-separated list of symbols")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (inclusive), e.g. 2024-01-02")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (inclusive), e.g. 2024-01-04")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs (default: ALPHATRADE_RUNS_ROOT or ../alphatrade_runs/default)")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory (default: <output-root>/reports)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet path (default: <reports-dir>/m9_predictions.parquet)")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_INFER_BATCH_SIZE,
                        help="Batch size for inference")
    parser.add_argument("--progress-every-batches", type=int, default=_DEFAULT_PROGRESS_EVERY_BATCHES,
                        help="Log progress every N batches (default: 10)")
    parser.add_argument("--progress-log", type=str, default=None,
                        help="JSONL progress log path (default: <reports-dir>/m9_infer_progress.jsonl)")
    parser.add_argument("--device", choices=["gpu", "cpu"], default=None,
                        help="JAX device policy (default: gpu; cpu is for tests/debug only)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: limit to first 2 symbols, max 100 samples")
    parser.add_argument("--output-metrics", type=str, default=None,
                        help="Inference metrics JSON path (default: <reports-dir>/m9_infer_metrics.json)")
    parser.add_argument("--output-metrics-md", type=str, default=None,
                        help="Inference metrics markdown path (default: <reports-dir>/m9_infer_metrics.md)")
    return parser.parse_args()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _is_date_only(value: str) -> bool:
    return len(value) == 10 and value[4] == "-" and value[7] == "-"


def _eob_range_mask(eob: pd.Series, start: str, end: str):
    eob_ts = pd.to_datetime(eob)
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if _is_date_only(end):
        return (eob_ts >= start_ts) & (eob_ts < end_ts + pd.Timedelta(days=1))
    return (eob_ts >= start_ts) & (eob_ts <= end_ts)


def _eligible_window_indices(
    bars_df: pd.DataFrame,
    lookback: int,
    start: str,
    end: str,
) -> np.ndarray:
    """Return row positions whose eob is in range and has enough lookback."""
    mask = _eob_range_mask(bars_df["eob"], start, end)
    indices = np.flatnonzero(mask.to_numpy())
    return indices[indices >= lookback - 1].astype(np.int64, copy=False)


def _windows_for_indices(features: np.ndarray, indices: np.ndarray, lookback: int) -> np.ndarray:
    """Materialize a batch of sliding windows for integer row positions."""
    if len(indices) == 0:
        return np.empty((0, lookback, FEATURE_DIM), dtype=np.float32)
    starts = indices.astype(np.int64) - int(lookback) + 1
    offsets = np.arange(int(lookback), dtype=np.int64)
    return features[starts[:, None] + offsets[None, :]].astype(np.float32, copy=False)


def _process_rss_mb() -> float | None:
    """Return current process RSS in MiB on Linux, if available."""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            pages = int(f.read().split()[1])
        return round(pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)
    except Exception:
        return None


def _write_progress_event(progress_log: Path | None, event: dict) -> None:
    if progress_log is None:
        return
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "m9_infer_progress_v1",
        "generated_at": datetime.now().isoformat(),
        **event,
    }
    with progress_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def build_sliding_windows(bars_df: pd.DataFrame, lookback: int, start: str, end: str):
    """Build sliding windows from bars.parquet filtered by eob range.

    Returns:
        windows: np.ndarray of shape [N, lookback, features]
        eob_timestamps: list of eob timestamps for each window
    """
    bars_df = bars_df.reset_index(drop=True)
    indices = _eligible_window_indices(bars_df, lookback, start, end)
    if len(indices) == 0:
        return np.array([]).reshape(0, lookback, FEATURE_DIM), []

    features = bars_df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    windows = _windows_for_indices(features, indices, lookback)
    eob_timestamps = bars_df.iloc[indices]["eob"].tolist()
    return windows, eob_timestamps


def iter_sliding_window_batches(
    bars_df: pd.DataFrame,
    lookback: int,
    start: str,
    end: str,
    batch_size: int,
    max_samples: int | None = None,
):
    """Yield sliding-window batches without materializing the full symbol matrix."""
    bars_df = bars_df.reset_index(drop=True)
    indices = _eligible_window_indices(bars_df, lookback, start, end)
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    if len(indices) == 0:
        return

    features = bars_df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    for start_idx in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start_idx:start_idx + int(batch_size)]
        windows = _windows_for_indices(features, batch_indices, lookback)
        eobs = bars_df.iloc[batch_indices]["eob"].tolist()
        yield windows, eobs


def smoke_limit_per_symbol(symbol_count: int, total_limit: int = 100) -> int:
    """Return a per-symbol smoke cap that keeps each requested symbol represented."""
    if symbol_count <= 0:
        return 0
    return max(1, int(total_limit) // int(symbol_count))


def main():
    args = parse_args()
    output_root = runtime_paths.resolve_output_root(args.output_root)
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    output_path = Path(args.output).expanduser() if args.output else reports_dir / "m9_predictions.parquet"
    metrics_json_path = (
        Path(args.output_metrics).expanduser()
        if args.output_metrics
        else reports_dir / "m9_infer_metrics.json"
    )
    metrics_md_path = (
        Path(args.output_metrics_md).expanduser()
        if args.output_metrics_md
        else reports_dir / "m9_infer_metrics.md"
    )
    progress_log_path = (
        Path(args.progress_log).expanduser()
        if args.progress_log
        else reports_dir / "m9_infer_progress.jsonl"
    )
    if args.batch_size <= 0:
        sys.exit(f"ERROR: --batch-size must be positive, got {args.batch_size}")
    if args.progress_every_batches < 0:
        sys.exit(
            "ERROR: --progress-every-batches must be non-negative, "
            f"got {args.progress_every_batches}"
        )
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    progress_log_path.write_text("", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"M9: Offline Batch Inference")
    print(f"{'='*60}\n")
    print(f"Output root: {output_root}")
    print(f"Reports:     {reports_dir}\n")
    print(f"Batch size:  {args.batch_size}")
    print(f"Progress log: {progress_log_path}\n")

    # Load bundle
    try:
        metadata = load_bundle_metadata(args.bundle)
    except Exception as e:
        sys.exit(f"ERROR: failed to load bundle metadata: {e}")
    manifest = metadata["manifest"]
    model_config_dict = metadata["model_config"]
    model_version = manifest["model_version"]

    print(f"Model version: {model_version}")
    print(f"Bundle: {args.bundle}")
    print(f"JAX backend: {jax.default_backend()}")
    device_policy = args.device or _requested_device
    if device_policy in ("gpu", "cuda") and jax.default_backend() != "gpu":
        sys.exit(
            "ERROR: GPU inference requested but JAX did not initialize the GPU "
            f"backend (got {jax.default_backend()!r})."
        )

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    if args.smoke:
        symbols = symbols[:2]
        smoke_symbol_limit = smoke_limit_per_symbol(len(symbols), total_limit=100)
        print(f"SMOKE mode: limited to {symbols}, max {smoke_symbol_limit} samples/symbol")
    else:
        smoke_symbol_limit = None

    lookback = int(model_config_dict["lookback_length"])
    horizons = [int(h) for h in model_config_dict["horizons"]]
    quantiles = [float(q) for q in model_config_dict["quantiles"]]

    print(f"Lookback: {lookback}, Horizons: {horizons}, Quantiles: {quantiles}")

    print("\nLoading bundle predictor...")
    try:
        predictor = AlphaTradeBundlePredictor(args.bundle)
    except Exception as e:
        sys.exit(f"ERROR: failed to initialize predictor: {e}")
    print(f"  Loaded checkpoint from step {predictor.checkpoint_step}\n")

    print("Loading data and running streaming inference...")
    _write_progress_event(
        progress_log_path,
        {
            "event": "start",
            "batch_size": args.batch_size,
            "progress_every_batches": args.progress_every_batches,
            "symbols_requested": len(symbols),
            "start": args.start,
            "end": args.end,
            "jax_backend": jax.default_backend(),
            "rss_mb": _process_rss_mb(),
        },
    )
    all_eobs = []
    all_symbols_list = []
    missing_symbols = []
    processed_symbols = []
    all_predictions = {h: [] for h in horizons}
    N = 0
    t_start = time.time()

    for symbol_idx, symbol in enumerate(symbols, start=1):
        bars_path = os.path.join(args.data_dir, symbol, "bars.parquet")
        if not os.path.exists(bars_path):
            print(f"  WARNING: {bars_path} not found, skipping {symbol}")
            missing_symbols.append(symbol)
            _write_progress_event(
                progress_log_path,
                {
                    "event": "symbol_missing",
                    "symbol": symbol,
                    "symbol_index": symbol_idx,
                    "total_symbols": len(symbols),
                    "path": bars_path,
                    "total_rows": N,
                    "elapsed_seconds": round(time.time() - t_start, 2),
                    "rss_mb": _process_rss_mb(),
                },
            )
            continue

        symbol_start_time = time.time()
        bars_df = pd.read_parquet(bars_path)
        bars_df = bars_df.reset_index(drop=True)
        symbol_indices = _eligible_window_indices(bars_df, lookback, args.start, args.end)
        if args.smoke:
            symbol_indices = symbol_indices[:smoke_symbol_limit]
        symbol_windows = int(len(symbol_indices))

        if symbol_windows == 0:
            print(f"  {symbol}: 0 windows in range [{args.start}, {args.end}]")
            _write_progress_event(
                progress_log_path,
                {
                    "event": "symbol_empty",
                    "symbol": symbol,
                    "symbol_index": symbol_idx,
                    "total_symbols": len(symbols),
                    "total_rows": N,
                    "elapsed_seconds": round(time.time() - t_start, 2),
                    "rss_mb": _process_rss_mb(),
                },
            )
            continue

        processed_symbols.append(symbol)
        symbol_batches = (symbol_windows + args.batch_size - 1) // args.batch_size
        print(
            f"  [{symbol_idx}/{len(symbols)}] {symbol}: "
            f"{symbol_windows} windows, {symbol_batches} batches",
            flush=True,
        )
        _write_progress_event(
            progress_log_path,
            {
                "event": "symbol_start",
                "symbol": symbol,
                "symbol_index": symbol_idx,
                "total_symbols": len(symbols),
                "symbol_windows": symbol_windows,
                "symbol_batches": symbol_batches,
                "batch_size": args.batch_size,
                "total_rows": N,
                "elapsed_seconds": round(time.time() - t_start, 2),
                "rss_mb": _process_rss_mb(),
            },
        )

        features = bars_df[FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
        for batch_start in range(0, symbol_windows, args.batch_size):
            batch_number = batch_start // args.batch_size + 1
            batch_indices = symbol_indices[batch_start:batch_start + args.batch_size]
            X_batch = _windows_for_indices(features, batch_indices, lookback)
            eobs = bars_df.iloc[batch_indices]["eob"].tolist()
            batch_predictions = predictor.predict_arrays(X_batch)

            for h in horizons:
                if h in batch_predictions:
                    all_predictions[h].append(batch_predictions[h])

            batch_n = int(len(eobs))
            all_eobs.extend(eobs)
            all_symbols_list.extend([symbol] * batch_n)
            N += batch_n

            should_log_progress = (
                args.progress_every_batches > 0
                and (
                    batch_number % args.progress_every_batches == 0
                    or batch_number == symbol_batches
                )
            )
            if should_log_progress:
                elapsed = time.time() - t_start
                rate = N / elapsed if elapsed > 0 else 0.0
                symbol_elapsed = time.time() - symbol_start_time
                symbol_done = min(batch_start + args.batch_size, symbol_windows)
                symbol_rate = symbol_done / symbol_elapsed if symbol_elapsed > 0 else 0.0
                print(
                    f"    {symbol} batch {batch_number}/{symbol_batches}: "
                    f"{symbol_done}/{symbol_windows} symbol rows, "
                    f"{N} total rows, {rate:.0f} rows/sec total, "
                    f"{symbol_rate:.0f} rows/sec symbol",
                    flush=True,
                )
                _write_progress_event(
                    progress_log_path,
                    {
                        "event": "batch_progress",
                        "symbol": symbol,
                        "symbol_index": symbol_idx,
                        "total_symbols": len(symbols),
                        "batch_number": batch_number,
                        "symbol_batches": symbol_batches,
                        "batch_rows": batch_n,
                        "symbol_rows_done": symbol_done,
                        "symbol_windows": symbol_windows,
                        "total_rows": N,
                        "elapsed_seconds": round(elapsed, 2),
                        "samples_per_second": round(rate, 1),
                        "symbol_samples_per_second": round(symbol_rate, 1),
                        "rss_mb": _process_rss_mb(),
                    },
                )

        symbol_elapsed = time.time() - symbol_start_time
        symbol_rate = symbol_windows / symbol_elapsed if symbol_elapsed > 0 else 0.0
        print(
            f"  [{symbol_idx}/{len(symbols)}] {symbol} done in "
            f"{symbol_elapsed:.1f}s ({symbol_rate:.0f} rows/sec)",
            flush=True,
        )
        _write_progress_event(
            progress_log_path,
            {
                "event": "symbol_done",
                "symbol": symbol,
                "symbol_index": symbol_idx,
                "total_symbols": len(symbols),
                "symbol_windows": symbol_windows,
                "symbol_batches": symbol_batches,
                "symbol_elapsed_seconds": round(symbol_elapsed, 2),
                "symbol_samples_per_second": round(symbol_rate, 1),
                "total_rows": N,
                "elapsed_seconds": round(time.time() - t_start, 2),
                "rss_mb": _process_rss_mb(),
            },
        )

    if N == 0:
        sys.exit("ERROR: No data windows created. Check symbols and date range.")

    print(f"\nTotal samples: {N}")

    t_end = time.time()
    duration = t_end - t_start
    print(f"  Inference complete in {duration:.1f}s ({N/duration:.0f} samples/sec)\n")
    _write_progress_event(
        progress_log_path,
        {
            "event": "inference_done",
            "total_rows": N,
            "processed_symbols": processed_symbols,
            "missing_symbols": missing_symbols,
            "elapsed_seconds": round(duration, 2),
            "samples_per_second": round(N / duration, 1) if duration > 0 else 0,
            "rss_mb": _process_rss_mb(),
        },
    )

    # Concatenate predictions
    for h in horizons:
        if all_predictions[h]:
            all_predictions[h] = np.concatenate(all_predictions[h], axis=0)

    # Build wide-format DataFrame
    print("Building predictions DataFrame...")
    df = prediction_schema.build_prediction_frame(
        symbols=all_symbols_list,
        eobs=all_eobs,
        model_version=model_version,
        predictions_by_horizon=all_predictions,
        horizons=horizons,
        quantiles=quantiles,
    )
    schema_issues = prediction_schema.validate_prediction_frame(
        df,
        horizons=horizons,
        quantiles=quantiles,
        expected_model_version=model_version,
    )
    if schema_issues:
        details = "; ".join(f"{issue.field}: {issue.message}" for issue in schema_issues)
        sys.exit(f"ERROR: prediction schema validation failed: {details}")

    # Write parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Wrote {output_path} ({len(df)} rows, {len(df.columns)} columns)")

    # Write inference metrics
    git_sha = get_git_sha()
    infer_metrics = {
        "schema_version": "m9_infer_metrics_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "model_version": model_version,
        "bundle_path": args.bundle,
        "prediction_path": str(output_path),
        "symbols": processed_symbols,
        "start": args.start,
        "end": args.end,
        "num_rows": len(df),
        "num_symbols": len(processed_symbols),
        "batch_size": args.batch_size,
        "jax_backend": jax.default_backend(),
        "progress_log_path": str(progress_log_path),
        "elapsed_seconds": round(duration, 2),
        "samples_per_second": round(N / duration, 1) if duration > 0 else 0,
        "inference": {
            "symbols": processed_symbols,
            "n_symbols": len(processed_symbols),
            "time_range": {"start": args.start, "end": args.end},
            "n_samples": N,
            "n_predictions": N * len(horizons) * len(quantiles),
            "batch_size": args.batch_size,
            "jax_backend": jax.default_backend(),
            "progress_log_path": str(progress_log_path),
            "missing_symbols": missing_symbols,
            "duration_seconds": round(duration, 2),
            "samples_per_second": round(N / duration, 1) if duration > 0 else 0,
        },
        "output": {
            "predictions_path": str(output_path),
            "predictions_rows": len(df),
            "predictions_columns": len(df.columns),
        },
    }

    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_path, 'w') as f:
        json.dump(infer_metrics, f, indent=2)
    print(f"  Wrote {metrics_json_path}")

    # Write markdown report
    metrics_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_md_path, 'w') as f:
        f.write("# M9 Inference Metrics\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Model\n\n")
        f.write(f"- Model version: `{model_version}`\n")
        f.write(f"- Bundle: `{args.bundle}`\n")
        f.write(f"- JAX backend: `{jax.default_backend()}`\n")
        f.write(f"- Git SHA: {git_sha}\n\n")
        f.write("## Inference\n\n")
        f.write(f"- Symbols: {len(processed_symbols)}\n")
        f.write(f"- Time range: {args.start} to {args.end}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Progress log: `{progress_log_path}`\n")
        f.write(f"- Samples: {N:,}\n")
        f.write(f"- Predictions: {N * len(horizons) * len(quantiles):,}\n")
        f.write(f"- Duration: {duration:.1f}s\n")
        f.write(f"- Throughput: {N/duration:.0f} samples/sec\n")
        if missing_symbols:
            f.write(f"- Missing symbols: {missing_symbols}\n")
        f.write(f"\n## Output\n\n")
        f.write(f"- Path: `{output_path}`\n")
        f.write(f"- Rows: {len(df):,}\n")
        f.write(f"- Columns: {len(df.columns)}\n")
        if args.smoke:
            f.write(f"\n**Note**: smoke mode was enabled.\n")
    print(f"  Wrote {metrics_md_path}")

    print(f"\n{'='*60}")
    print(f"M9 Inference Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
