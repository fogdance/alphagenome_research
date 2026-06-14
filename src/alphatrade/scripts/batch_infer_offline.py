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
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for inference")
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


def build_sliding_windows(bars_df: pd.DataFrame, lookback: int, start: str, end: str):
    """Build sliding windows from bars.parquet filtered by eob range.

    Returns:
        windows: np.ndarray of shape [N, lookback, features]
        eob_timestamps: list of eob timestamps for each window
    """
    # Date-only end values are treated as inclusive calendar days, matching CLI
    # examples such as --start 2024-01-02 --end 2024-01-04.
    mask = _eob_range_mask(bars_df["eob"], start, end)

    # We need lookback bars before the start of the range, so find earliest
    # valid window start index
    filtered_indices = bars_df.index[mask].tolist()
    if not filtered_indices:
        return np.array([]).reshape(0, lookback, FEATURE_DIM), []

    windows = []
    eob_timestamps = []

    for idx in filtered_indices:
        # Window: [idx - lookback + 1, idx + 1)
        w_start = idx - lookback + 1
        if w_start < 0:
            continue
        window = bars_df.iloc[w_start:idx + 1][FEATURE_COLS].values.astype(np.float32)
        if window.shape[0] != lookback:
            continue
        windows.append(window)
        eob_timestamps.append(bars_df.iloc[idx]["eob"])

    if not windows:
        return np.array([]).reshape(0, lookback, FEATURE_DIM), []

    return np.stack(windows), eob_timestamps


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

    print(f"\n{'='*60}")
    print(f"M9: Offline Batch Inference")
    print(f"{'='*60}\n")
    print(f"Output root: {output_root}")
    print(f"Reports:     {reports_dir}\n")

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
        print(f"SMOKE mode: limited to {symbols}")

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

    # Build sliding windows for each symbol
    print("Loading data and building windows...")
    all_windows = []
    all_eobs = []
    all_symbols_list = []
    missing_symbols = []
    processed_symbols = []

    for symbol in symbols:
        bars_path = os.path.join(args.data_dir, symbol, "bars.parquet")
        if not os.path.exists(bars_path):
            print(f"  WARNING: {bars_path} not found, skipping {symbol}")
            missing_symbols.append(symbol)
            continue

        bars_df = pd.read_parquet(bars_path)
        windows, eobs = build_sliding_windows(bars_df, lookback, args.start, args.end)

        if len(windows) == 0:
            print(f"  {symbol}: 0 windows in range [{args.start}, {args.end}]")
            continue

        if args.smoke:
            # In smoke mode, limit total samples to 100 across all symbols.
            remaining = 100 - sum(w.shape[0] for w in all_windows)
            if remaining <= 0:
                break
            windows = windows[:remaining]
            eobs = eobs[:remaining]
            if len(windows) == 0:
                break

        all_windows.append(windows)
        all_eobs.extend(eobs)
        all_symbols_list.extend([symbol] * len(eobs))
        processed_symbols.append(symbol)
        print(f"  {symbol}: {len(eobs)} windows")

    if not all_windows:
        sys.exit("ERROR: No data windows created. Check symbols and date range.")

    X_all = np.concatenate(all_windows, axis=0)
    N = X_all.shape[0]
    print(f"\nTotal samples: {N}")

    # Run batched inference
    print(f"Running inference (batch_size={args.batch_size})...")
    t_start = time.time()

    all_predictions = {h: [] for h in horizons}

    for i in range(0, N, args.batch_size):
        if i % (args.batch_size * 10) == 0:
            print(f"  Progress: {i}/{N} ({100*i//N}%)")

        X_batch = X_all[i:i + args.batch_size]
        batch_predictions = predictor.predict_arrays(X_batch)

        for h in horizons:
            if h in batch_predictions:
                all_predictions[h].append(batch_predictions[h])

    print(f"  Progress: {N}/{N} (100%)")
    t_end = time.time()
    duration = t_end - t_start
    print(f"  Inference complete in {duration:.1f}s ({N/duration:.0f} samples/sec)\n")

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
        "inference": {
            "symbols": processed_symbols,
            "n_symbols": len(processed_symbols),
            "time_range": {"start": args.start, "end": args.end},
            "n_samples": N,
            "n_predictions": N * len(horizons) * len(quantiles),
            "batch_size": args.batch_size,
            "jax_backend": jax.default_backend(),
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
