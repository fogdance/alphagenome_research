#!/usr/bin/env python3
"""
M4 Evaluation Script (Fast Version with Batched Inference)

Optimized version with:
- Batched inference (batch_size=128 or 256)
- JIT compilation
- 15-25x speedup compared to eval_m4.py
"""
import os as _os

if "JAX_PLATFORMS" not in _os.environ:
    _os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.training import checkpoints as flax_ckpt
from scipy.stats import spearmanr

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphatrade.core import model as model_lib
from alphatrade.core import schemas
from alphatrade import quality_metrics
from alphatrade import runtime_paths
from alphatrade import window_cache
from data_pipeline.feature_profiles import (
    FeatureProfile,
    assert_feature_profiles_match,
    feature_profile_from_mapping,
    feature_profile_to_dict,
    resolve_feature_profile,
)


def parse_args():
    parser = argparse.ArgumentParser(description="M4 AlphaTrade v0.2 Evaluation (Fast)")
    parser.add_argument("--train-metrics", type=str, default=None,
                        help="Path to training metrics JSON (default: <reports-dir>/m4_train_metrics.json)")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset/m2.yaml",
                        help="Dataset config file")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for inference (default: 128)")
    parser.add_argument("--ckpt-step", type=str, default="best",
                        help="Checkpoint step: 'best' (default), 'last', or integer step number")
    parser.add_argument("--smoke", action="store_true",
                        help="Use smoke test symbols (overrides auto-detection)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs (default: ALPHATRADE_RUNS_ROOT or ../alphatrade_runs/default)")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory (default: <output-root>/reports)")
    parser.add_argument("--window-cache", type=str, default="auto", choices=["auto", "refresh", "off"],
                        help="Window cache mode (default: auto)")
    parser.add_argument("--window-cache-dir", type=str, default=None,
                        help="Window cache directory (default: <output-root>/cache/windows)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_train_metrics(metrics_path: str) -> dict:
    """Load training metrics to get run_id and model config."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


class M4EvalDataset:
    """Dataset for M4 evaluation backed by shared cached windows."""

    def __init__(
        self,
        symbols: List[str],
        processed_dir: str,
        feature_profile: FeatureProfile,
        split: str,
        cache_dir: str | os.PathLike[str] | None = None,
        cache_mode: str = "auto",
    ):
        self.symbols = symbols
        self.feature_profile = feature_profile
        self.processed_dir = feature_profile.processed_root or processed_dir
        self.split = split
        self.cached = window_cache.load_or_build(
            symbols=symbols,
            processed_root=self.processed_dir,
            split=split,
            features=list(feature_profile.feature_cols),
            feature_profile_id=feature_profile.profile_id,
            cache_dir=cache_dir,
            scaler_hash=feature_profile.scaler_hash,
            mode=cache_mode,
            mmap=True,
        )
        self.x = self.cached.x
        self.y = self.cached.y
        self.sample_symbols = self.cached.symbols
        self.cache_metadata = self.cached.metadata

    def __len__(self):
        return len(self.x)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantiles: List[float]) -> float:
    """Calculate pinball loss for quantile predictions."""
    losses = []
    for i, q in enumerate(quantiles):
        error = y_true - y_pred[:, i]
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        losses.append(loss.mean())
    return np.mean(losses)


def pinball_loss_sum_count(y_true: np.ndarray, y_pred: np.ndarray,
                           quantiles: List[float]) -> tuple[float, int]:
    """Return summed pinball loss and element count for streaming metrics."""
    loss_by_quantile = pinball_loss_sums_by_quantile(y_true, y_pred, quantiles)
    return float(loss_by_quantile.sum()), int(y_true.size * len(quantiles))


def pinball_loss_sums_by_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: List[float],
) -> np.ndarray:
    """Return summed pinball loss for each quantile."""
    losses = []
    for i, q in enumerate(quantiles):
        error = y_true - y_pred[:, i]
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        losses.append(float(loss.sum()))
    return np.asarray(losses, dtype=np.float64)


def calculate_quantile_coverage(y_true: np.ndarray, y_pred: np.ndarray, quantiles: List[float]) -> Dict[str, float]:
    """Calculate actual coverage rate for each quantile."""
    coverage = {}
    for i, q in enumerate(quantiles):
        actual_coverage = (y_true < y_pred[:, i]).mean()
        coverage[f"q{int(q*100)}"] = float(actual_coverage)
    return coverage


def calculate_quantile_crossing(y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate quantile crossing rate (non-monotonic predictions)."""
    crossings = (y_pred[:, 1:] < y_pred[:, :-1]).sum()
    total = y_pred.shape[0] * (y_pred.shape[1] - 1)

    return {
        "rate": float(crossings / total) if total else 0.0,
        "count": int(crossings)
    }


def calculate_ic_metrics(y_true: np.ndarray, y_pred_median: np.ndarray) -> Dict[str, float]:
    """Calculate IC metrics using median prediction."""
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred_median) == 0:
        return {"ic": 0.0, "rank_ic": 0.0}

    ic = np.corrcoef(y_true, y_pred_median)[0, 1]
    rank_ic = spearmanr(y_true, y_pred_median).correlation

    return {
        "ic": float(ic) if not np.isnan(ic) else 0.0,
        "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0
    }


def evaluate_model_batched(model_apply_fn, params, state, dataset: M4EvalDataset,
                           horizons: List[int], quantiles: List[float], batch_size: int = 128,
                           quantile_loss_report: dict | None = None) -> Dict:
    """Evaluate model with streamed host→device batches."""

    print(f"Running streamed batched evaluation (batch_size={batch_size})...")

    N = len(dataset)
    if N == 0:
        raise ValueError(
            f"No samples available for eval split '{dataset.split}' "
            f"across {len(dataset.symbols)} symbols"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    y_all = np.asarray(dataset.y)
    all_symbols = np.asarray(dataset.sample_symbols)
    horizon_to_idx = {h: i for i, h in enumerate(window_cache.HORIZONS)}
    missing_horizons = [h for h in horizons if h not in horizon_to_idx]
    if missing_horizons:
        raise ValueError(f"Unsupported horizons requested: {missing_horizons}")

    print(f"  Samples: {N:,}")
    print(f"  Data shape: {dataset.x.shape}")
    print(f"  Batches: {(N + batch_size - 1) // batch_size}")

    # Batched inference (no JIT: AlphaTradeOutput is not a JAX pytree).
    # Metrics are accumulated per batch; only median predictions are retained
    # for IC/rank-IC, which require full-series correlation.
    median_idx = len(quantiles) // 2
    pinball_sum = {h: 0.0 for h in horizons}
    pinball_count = {h: 0 for h in horizons}
    pinball_sum_by_quantile = {h: np.zeros(len(quantiles), dtype=np.float64) for h in horizons}
    pinball_sample_count = {h: 0 for h in horizons}
    predicted_count = {h: 0 for h in horizons}
    y_chunks = {h: [] for h in horizons}
    median_pred_chunks = {h: [] for h in horizons}
    horizon_weights = quality_metrics.resolve_horizon_weights(
        (quantile_loss_report or {}).get("horizon_weights"),
        horizons,
    )
    quantile_weights = np.asarray(
        quality_metrics.resolve_quantile_weights(
            (quantile_loss_report or {}).get("quantile_weights"),
            quantiles,
        ),
        dtype=np.float64,
    )
    quantile_weight_norm = quantile_weights / quantile_weights.sum()

    h_first = horizons[0]
    coverage_counts = {h: np.zeros(len(quantiles), dtype=np.int64) for h in horizons}
    coverage_total = {h: 0 for h in horizons}
    crossing_count = {h: 0 for h in horizons}
    crossing_total = {h: 0 for h in horizons}

    symbol_order = []
    symbol_stats = {}

    def _symbol_stats(symbol):
        key = str(symbol)
        if key not in symbol_stats:
            symbol_order.append(key)
            symbol_stats[key] = {
                "samples": 0,
                "pinball_sum": {h: 0.0 for h in horizons},
                "pinball_count": {h: 0 for h in horizons},
                "coverage_counts": {
                    h: np.zeros(len(quantiles), dtype=np.int64) for h in horizons
                },
                "coverage_total": {h: 0 for h in horizons},
                "crossing_count": {h: 0 for h in horizons},
                "crossing_total": {h: 0 for h in horizons},
                "median_y": {h: [] for h in horizons},
                "median_pred": {h: [] for h in horizons},
            }
        return symbol_stats[key]

    for symbol in all_symbols:
        _symbol_stats(symbol)["samples"] += 1

    print(f"  Running inference...")
    for i in range(0, N, batch_size):
        if i % (batch_size * 10) == 0:
            pct = 0 if N == 0 else 100 * i // N
            print(f"    Progress: {i}/{N} ({pct}%)")

        end = min(i + batch_size, N)
        X_batch = jax.device_put(dataset.x[i:end])
        batch_symbols = all_symbols[i:end]
        rng = jax.random.PRNGKey(0)
        output, _ = model_apply_fn(params, state, rng, X_batch)

        # Extract predictions for each horizon
        for h in horizons:
            if h in output.log_return_quantiles:
                # [B, Q]
                preds = np.asarray(output.log_return_quantiles[h])
                y_true = y_all[i:end, horizon_to_idx[h]]
                loss_by_quantile = pinball_loss_sums_by_quantile(y_true, preds, quantiles)
                loss_sum = float(loss_by_quantile.sum())
                loss_count = int(y_true.size * len(quantiles))
                pinball_sum[h] += loss_sum
                pinball_count[h] += loss_count
                pinball_sum_by_quantile[h] += loss_by_quantile
                pinball_sample_count[h] += int(y_true.shape[0])
                predicted_count[h] += int(preds.shape[0])
                y_chunks[h].append(y_true)
                median_pred_chunks[h].append(preds[:, median_idx])

                batch_coverage = (y_true[:, None] < preds).sum(axis=0)
                coverage_counts[h] += batch_coverage
                coverage_total[h] += int(y_true.shape[0])
                if preds.shape[1] > 1:
                    crossing_count[h] += int((preds[:, 1:] < preds[:, :-1]).sum())
                    crossing_total[h] += int(preds.shape[0] * (preds.shape[1] - 1))

                for symbol in np.unique(batch_symbols):
                    symbol_mask = batch_symbols == symbol
                    stats = _symbol_stats(symbol)
                    sym_loss_sum, sym_loss_count = pinball_loss_sum_count(
                        y_true[symbol_mask],
                        preds[symbol_mask],
                        quantiles,
                    )
                    stats["pinball_sum"][h] += sym_loss_sum
                    stats["pinball_count"][h] += sym_loss_count
                    stats["coverage_counts"][h] += (
                        y_true[symbol_mask, None] < preds[symbol_mask]
                    ).sum(axis=0)
                    stats["coverage_total"][h] += int(symbol_mask.sum())
                    if preds.shape[1] > 1:
                        stats["crossing_count"][h] += int(
                            (preds[symbol_mask, 1:] < preds[symbol_mask, :-1]).sum()
                        )
                        stats["crossing_total"][h] += int(
                            symbol_mask.sum() * (preds.shape[1] - 1)
                        )
                    stats["median_y"][h].append(y_true[symbol_mask])
                    stats["median_pred"][h].append(preds[symbol_mask, median_idx])

    print(f"    Progress: {N}/{N} (100%)")
    print(f"  ✓ Inference complete\n")

    missing_predictions = [h for h in horizons if predicted_count[h] != N]
    if missing_predictions:
        raise ValueError(
            "Model did not return predictions for all samples at horizons: "
            f"{missing_predictions}"
        )

    # Calculate metrics
    print("Calculating metrics...")

    overall_pinball = []
    by_horizon_pinball = {}
    ic_by_horizon = {}
    rank_ic_by_horizon = {}
    coverage_by_horizon = {}
    crossing_by_horizon = {}
    by_horizon = {}

    for h in horizons:
        pb_loss = pinball_sum[h] / pinball_count[h] if pinball_count[h] else 0.0
        h_key = quality_metrics.horizon_key(h)
        by_horizon_pinball[h_key] = float(pb_loss)
        overall_pinball.append(pb_loss)

        y_true = np.concatenate(y_chunks[h], axis=0)
        y_pred_median = np.concatenate(median_pred_chunks[h], axis=0)
        ic_metrics = calculate_ic_metrics(y_true, y_pred_median)
        ic_by_horizon[h_key] = ic_metrics["ic"]
        rank_ic_by_horizon[h_key] = ic_metrics["rank_ic"]
        coverage_by_horizon[h_key] = {
            quality_metrics.quantile_key(q): (
                float(coverage_counts[h][qi] / coverage_total[h])
                if coverage_total[h] else 0.0
            )
            for qi, q in enumerate(quantiles)
        }
        crossing_by_horizon[h_key] = {
            "rate": (
                float(crossing_count[h] / crossing_total[h])
                if crossing_total[h] else 0.0
            ),
            "count": int(crossing_count[h]),
        }

    # Overall pinball loss
    overall_pb = float(np.mean(overall_pinball)) if overall_pinball else 0.0
    weighted_pinball_by_horizon = {}
    weighted_terms = []
    horizon_weight_total = 0.0
    for h in horizons:
        h_key = quality_metrics.horizon_key(h)
        if pinball_sample_count[h]:
            per_quantile_loss = pinball_sum_by_quantile[h] / pinball_sample_count[h]
            weighted_loss = float(np.sum(per_quantile_loss * quantile_weight_norm))
        else:
            weighted_loss = 0.0
        h_weight = float(horizon_weights.get(h, 1.0))
        weighted_pinball_by_horizon[h_key] = weighted_loss
        weighted_terms.append(h_weight * weighted_loss)
        horizon_weight_total += h_weight
    weighted_pinball_loss = {
        "overall": (
            float(sum(weighted_terms) / horizon_weight_total)
            if horizon_weight_total else 0.0
        ),
        "by_horizon": weighted_pinball_by_horizon,
        "horizon_weights": {
            quality_metrics.horizon_key(h): float(horizon_weights.get(h, 1.0))
            for h in horizons
        },
        "quantile_weights": {
            quality_metrics.quantile_key(q): float(w)
            for q, w in zip(quantiles, quantile_weights)
        },
        "normalization": (
            "quantile weights normalized within each horizon; "
            "horizon weighted average normalized by sum of horizon weights"
        ),
    }

    # Quantile coverage across all horizons
    coverage = {
        quality_metrics.quantile_key(q): (
            float(
                sum(coverage_counts[h][qi] for h in horizons)
                / sum(coverage_total[h] for h in horizons)
            )
            if sum(coverage_total[h] for h in horizons) else 0.0
        )
        for qi, q in enumerate(quantiles)
    }

    # Quantile crossing across all horizons
    total_crossing_count = sum(crossing_count[h] for h in horizons)
    total_crossing_slots = sum(crossing_total[h] for h in horizons)
    crossing = {
        "rate": float(total_crossing_count / total_crossing_slots) if total_crossing_slots else 0.0,
        "count": int(total_crossing_count),
        "by_horizon": crossing_by_horizon,
    }

    # IC metrics (overall)
    y_true_first = np.concatenate(y_chunks[h_first], axis=0)
    y_pred_median = np.concatenate(median_pred_chunks[h_first], axis=0)
    ic_metrics = calculate_ic_metrics(y_true_first, y_pred_median)
    coverage_calibration = quality_metrics.coverage_calibration_summary(
        coverage_by_horizon,
        quantiles,
    )

    for h in horizons:
        h_key = quality_metrics.horizon_key(h)
        by_horizon[h_key] = {
            "samples": int(predicted_count[h]),
            "pinball_loss": by_horizon_pinball[h_key],
            "ic": ic_by_horizon[h_key],
            "rank_ic": rank_ic_by_horizon[h_key],
            "quantile_coverage": coverage_by_horizon[h_key],
            "coverage_calibration": coverage_calibration["by_horizon"][h_key],
            "quantile_crossing": crossing_by_horizon[h_key],
        }

    # By-symbol metrics
    print("Calculating by-symbol metrics...")
    by_symbol = []

    for symbol in symbol_order:
        stats = symbol_stats[symbol]
        symbol_pinball = []
        symbol_samples = stats["samples"]
        symbol_by_horizon = {}

        for h in horizons:
            if stats["pinball_count"][h]:
                h_key = quality_metrics.horizon_key(h)
                pb_loss = stats["pinball_sum"][h] / stats["pinball_count"][h]
                symbol_pinball.append(pb_loss)
                symbol_coverage = {
                    quality_metrics.quantile_key(q): (
                        float(stats["coverage_counts"][h][qi] / stats["coverage_total"][h])
                        if stats["coverage_total"][h] else 0.0
                    )
                    for qi, q in enumerate(quantiles)
                }
                if stats["median_y"][h]:
                    y_true_symbol_h = np.concatenate(stats["median_y"][h], axis=0)
                    y_pred_symbol_h = np.concatenate(stats["median_pred"][h], axis=0)
                    symbol_ic_h = calculate_ic_metrics(y_true_symbol_h, y_pred_symbol_h)
                else:
                    symbol_ic_h = {"ic": 0.0, "rank_ic": 0.0}
                symbol_by_horizon[h_key] = {
                    "samples": int(stats["coverage_total"][h]),
                    "pinball_loss": float(pb_loss),
                    "ic": float(symbol_ic_h["ic"]),
                    "rank_ic": float(symbol_ic_h["rank_ic"]),
                    "quantile_coverage": symbol_coverage,
                    "coverage_calibration": quality_metrics.coverage_calibration_summary(
                        {h_key: symbol_coverage},
                        quantiles,
                    )["by_horizon"][h_key],
                    "quantile_crossing": {
                        "rate": (
                            float(stats["crossing_count"][h] / stats["crossing_total"][h])
                            if stats["crossing_total"][h] else 0.0
                        ),
                        "count": int(stats["crossing_count"][h]),
                    },
                }

        if symbol_pinball:
            if stats["median_y"][h_first]:
                y_true_symbol = np.concatenate(stats["median_y"][h_first], axis=0)
                y_pred_median = np.concatenate(stats["median_pred"][h_first], axis=0)
                symbol_ic = calculate_ic_metrics(y_true_symbol, y_pred_median)["ic"]
            else:
                symbol_ic = 0.0

            by_symbol.append({
                "symbol": symbol,
                "samples": symbol_samples,
                "pinball_loss": float(np.mean(symbol_pinball)),
                "ic": float(symbol_ic),
                "by_horizon": symbol_by_horizon,
            })

    print("  ✓ Metrics complete\n")

    return {
        "pinball_loss": {
            "overall": overall_pb,
            "by_horizon": by_horizon_pinball
        },
        "weighted_pinball_loss": weighted_pinball_loss,
        "quantile_coverage": coverage,
        "coverage_calibration": coverage_calibration,
        "quantile_crossing": crossing,
        "ic_metrics": {
            "ic": ic_metrics["ic"],
            "rank_ic": ic_metrics["rank_ic"],
            "ic_by_horizon": ic_by_horizon,
            "rank_ic_by_horizon": rank_ic_by_horizon,
        },
        "by_horizon": by_horizon,
        "by_symbol": by_symbol
    }


def main():
    args = parse_args()
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    window_cache_dir = (
        Path(args.window_cache_dir).expanduser().resolve()
        if args.window_cache_dir
        else runtime_paths.cache_dir(args.output_root) / "windows"
    )
    train_metrics_path = args.train_metrics or str(reports_dir / "m4_train_metrics.json")

    # Load training metrics
    train_metrics = load_train_metrics(train_metrics_path)
    run_id = train_metrics['run']['run_id']

    print(f"\n{'='*60}")
    print(f"M4: AlphaTrade v0.2 Evaluation (FAST)")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Reports dir: {reports_dir}")
    print(f"Window cache: {args.window_cache} ({window_cache_dir})")
    print(f"Train metrics: {train_metrics_path}")
    print(f"{'='*60}\n")

    # Load config
    config_dict = load_config(args.dataset_config)
    config_profile = resolve_feature_profile(config_dict)
    train_profile_data = (
        train_metrics.get("dataset", {}).get("feature_profile")
        or train_metrics.get("model", {}).get("config", {}).get("feature_profile")
    )
    if not train_profile_data:
        sys.exit("ERROR: train metrics missing dataset/model feature_profile")
    try:
        train_profile = feature_profile_from_mapping(train_profile_data)
        assert_feature_profiles_match(config_profile, train_profile, context="m4_eval")
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    feature_profile_dict = feature_profile_to_dict(config_profile)
    processed_dir = config_profile.processed_root or config_dict['paths']['processed_dir']
    print(
        f"Feature profile: {config_profile.profile_id} "
        f"({config_profile.feature_dim} cols)"
    )
    print(f"Processed dir: {processed_dir}")

    # Select symbols
    if args.smoke:
        symbols = config_dict['universe']['smoke_symbols']
    else:
        symbols_file = config_dict['universe']['candidates_file']
        with open(symbols_file, 'r') as f:
            symbols = yaml.safe_load(f)['candidates']

    # Load dataset
    dataset = M4EvalDataset(
        symbols,
        processed_dir,
        config_profile,
        args.split,
        cache_dir=window_cache_dir,
        cache_mode=args.window_cache,
    )

    # Create model
    model_config = train_metrics['model']['config']
    if int(model_config["num_features"]) != config_profile.feature_dim:
        sys.exit(
            "ERROR: model_config.num_features does not match feature profile: "
            f"{model_config['num_features']} vs {config_profile.feature_dim}"
        )
    alphatrade_config = schemas.AlphaTradeConfig(
        lookback_length=model_config['lookback_length'],
        num_features=model_config['num_features'],
        stem_channels=model_config['stem_channels'],
        num_encoder_stages=model_config['num_encoder_stages'],
        d_model=model_config['d_model'],
        num_transformer_layers=model_config['num_transformer_layers'],
        horizons=model_config['horizons'],
        quantiles=model_config['quantiles']
    )

    print("Initializing model...")

    # Create model using Haiku
    def forward(x):
        model = model_lib.AlphaTrade(alphatrade_config)
        return model(x)

    import haiku as hk
    forward_t = hk.transform_with_state(lambda x: forward(x))

    # Initialize params
    rng = jax.random.PRNGKey(42)
    dummy_x = jnp.zeros((1, alphatrade_config.lookback_length, alphatrade_config.num_features), dtype=jnp.float32)
    params, state = forward_t.init(rng, dummy_x)

    # Create dummy optimizer state for checkpoint restoration
    import optax
    dummy_optimizer = optax.adam(1e-3)
    opt_state = dummy_optimizer.init(params)

    print("  ✓ Model initialized\n")

    # --- Hard-fail checkpoint loading ---
    ckpt_dir = train_metrics['run'].get('checkpoint_dir')
    if not ckpt_dir:
        sys.exit("ERROR: train_metrics missing 'run.checkpoint_dir'")

    # Determine checkpoint path and restore step based on --ckpt-step
    chosen_step = None
    if args.ckpt_step == "best":
        ckpt_path = str(Path(ckpt_dir) / "best")
    elif args.ckpt_step == "last":
        ckpt_path = ckpt_dir
    else:
        # Integer step number
        try:
            chosen_step = int(args.ckpt_step)
        except ValueError:
            sys.exit(f"ERROR: --ckpt-step must be 'best', 'last', or an integer, got '{args.ckpt_step}'")
        ckpt_path = ckpt_dir

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt_state = {"params": params, "state": state, "opt_state": opt_state, "step": 0}
    try:
        restored = flax_ckpt.restore_checkpoint(ckpt_path, ckpt_state, step=chosen_step)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    if restored["step"] == 0:
        sys.exit(f"ERROR: No checkpoint found at {ckpt_path}")

    params = restored["params"]
    state = restored["state"]
    checkpoint_step = int(restored["step"])
    print(f"  Loaded checkpoint from step {checkpoint_step}\n")

    # Get git sha
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    # Build model provenance info
    model_info = {
        "source": "checkpoint",
        "checkpoint_dir": str(Path(ckpt_path).resolve()),
        "checkpoint_step": checkpoint_step,
        "train_run_id": run_id,
        "git_sha": git_sha,
    }
    quantile_loss_report = train_metrics.get("loss", {}).get("quantile_loss_weights")
    if quantile_loss_report is None:
        quantile_loss_report = quality_metrics.loss_weight_report(
            horizons=alphatrade_config.horizons,
            quantiles=alphatrade_config.quantiles,
        )

    # Evaluate (BATCHED)
    # Note: forward_t.apply is NOT jit-wrapped here because AlphaTradeOutput
    # is a custom Python type (not a JAX pytree), which jax.jit cannot trace.
    eval_results = evaluate_model_batched(
        forward_t.apply,
        params,
        state,
        dataset,
        alphatrade_config.horizons,
        alphatrade_config.quantiles,
        args.batch_size,
        quantile_loss_report=quantile_loss_report,
    )
    target_scaling = quality_metrics.target_scale_summary(
        dataset.y,
        alphatrade_config.horizons,
    )

    # --- Sanity check on quantile coverage ---
    sanity_warnings = []
    for q_label, cov_val in eval_results['quantile_coverage'].items():
        if cov_val < 0.02:
            warn = f"quantile_coverage[{q_label}] = {cov_val:.4f} < 0.02"
            sanity_warnings.append(warn)
            print(f"  WARNING: {warn}")
        elif cov_val > 0.98:
            warn = f"quantile_coverage[{q_label}] = {cov_val:.4f} > 0.98"
            sanity_warnings.append(warn)
            print(f"  WARNING: {warn}")
    calibration = eval_results.get("coverage_calibration", {})
    if calibration.get("overall_mae", 0.0) > 0.15:
        warn = f"coverage_calibration.overall_mae = {calibration['overall_mae']:.4f} > 0.15"
        sanity_warnings.append(warn)
        print(f"  WARNING: {warn}")
    worst_calibration = calibration.get("worst", {})
    if worst_calibration.get("abs_error", 0.0) > 0.25:
        warn = (
            "coverage_calibration.worst = "
            f"{worst_calibration.get('horizon')} {worst_calibration.get('quantile')} "
            f"abs_error={worst_calibration.get('abs_error'):.4f}"
        )
        sanity_warnings.append(warn)
        print(f"  WARNING: {warn}")
    for warning in target_scaling.get("warnings", []):
        sanity_warnings.append(f"target_scaling: {warning}")
        print(f"  WARNING: target_scaling: {warning}")

    # Generate output
    eval_metrics = {
        "run_id": run_id,
        "eval_timestamp": datetime.now().isoformat(),
        "model": model_info,
        "dataset": {
            "split": args.split,
            "symbols": len(symbols),
            "samples": len(dataset),
            "eval_mode": "streamed_batches",
            "batch_size": args.batch_size,
            "processed_dir": processed_dir,
            "feature_profile": feature_profile_dict,
            "window_cache": {
                "mode": args.window_cache,
                "cache_dir": str(window_cache_dir),
                **dataset.cache_metadata,
            },
            "target_scaling": target_scaling,
        },
        "quantile_loss": quantile_loss_report,
        **eval_results
    }
    if sanity_warnings:
        eval_metrics["sanity_warnings"] = sanity_warnings

    # Save JSON
    json_path = reports_dir / "m4_eval_metrics_fast.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Metrics: {json_path}")

    # Save markdown report
    md_path = reports_dir / "m4_eval_run_fast.md"
    with open(md_path, 'w') as f:
        f.write("# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Checkpoint Info\n\n")
        f.write(f"- Source: {model_info['source']}\n")
        f.write(f"- Checkpoint step: {checkpoint_step}\n")
        f.write(f"- Checkpoint dir: `{model_info['checkpoint_dir']}`\n")
        f.write(f"- Train run ID: {run_id}\n")
        f.write(f"- Git SHA: {git_sha}\n")
        f.write(f"- Split: {args.split}\n\n")
        f.write("Reproduce:\n")
        f.write("```bash\n")
        f.write(f"python src/alphatrade/scripts/eval_m4_fast.py \\\n")
        f.write(f"  --train-metrics {train_metrics_path} \\\n")
        f.write(f"  --ckpt-step {args.ckpt_step} \\\n")
        f.write(f"  --reports-dir {reports_dir} \\\n")
        f.write(f"  --split {args.split}")
        if args.smoke:
            f.write(" \\\n  --smoke")
        f.write("\n```\n\n")

        f.write("## 配置\n\n")
        f.write(f"- Run ID: {run_id}\n")
        f.write(f"- Split: {args.split}\n")
        f.write(f"- Symbols: {len(symbols)}\n")
        f.write(f"- Samples: {len(dataset):,}\n")
        f.write(f"- Batch size: {args.batch_size}\n\n")

        f.write("## Pinball Loss\n\n")
        f.write(f"- Overall: {eval_results['pinball_loss']['overall']:.6f}\n\n")
        f.write(
            f"- Weighted overall: "
            f"{eval_results['weighted_pinball_loss']['overall']:.6f}\n\n"
        )

        f.write("### By-Horizon\n\n")
        f.write("| Horizon | Loss | Weighted Loss |\n")
        f.write("|---------|------|---------------|\n")
        for h, loss in eval_results['pinball_loss']['by_horizon'].items():
            weighted_loss = eval_results["weighted_pinball_loss"]["by_horizon"][h]
            f.write(f"| {h} | {loss:.6f} | {weighted_loss:.6f} |\n")

        f.write("\n## Quantile Coverage\n\n")
        for q, cov in eval_results['quantile_coverage'].items():
            expected = int(q[1:]) / 100
            f.write(f"- {q}: {cov:.4f} (expected: {expected:.2f})\n")

        f.write("\n## Coverage Calibration\n\n")
        calibration = eval_results["coverage_calibration"]
        f.write(f"- Overall MAE: {calibration['overall_mae']:.4f}\n")
        f.write(f"- Max abs error: {calibration['max_abs_error']:.4f}\n")
        worst = calibration["worst"]
        f.write(
            f"- Worst: {worst['horizon']} {worst['quantile']} "
            f"observed={worst['observed']:.4f}, expected={worst['expected']:.2f}\n"
        )

        f.write("\n## Quantile Crossing\n\n")
        f.write(f"- Rate: {eval_results['quantile_crossing']['rate']:.4f}\n")
        f.write(f"- Count: {eval_results['quantile_crossing']['count']}\n")

        f.write("\n## IC Metrics\n\n")
        f.write(f"- IC: {eval_results['ic_metrics']['ic']:.4f}\n")
        f.write(f"- Rank IC: {eval_results['ic_metrics']['rank_ic']:.4f}\n\n")

        f.write("### By-Horizon IC\n\n")
        f.write("| Horizon | IC |\n")
        f.write("|---------|----|\n")
        for h, ic in eval_results['ic_metrics']['ic_by_horizon'].items():
            f.write(f"| {h} | {ic:.4f} |\n")

        f.write("\n## By-Horizon Quality\n\n")
        f.write("| Horizon | Samples | Pinball | IC | Rank IC | Coverage MAE | Crossing |\n")
        f.write("|---------|---------|---------|----|---------|--------------|----------|\n")
        for h, metrics in eval_results["by_horizon"].items():
            f.write(
                f"| {h} | {metrics['samples']:,} | {metrics['pinball_loss']:.6f} | "
                f"{metrics['ic']:.4f} | {metrics['rank_ic']:.4f} | "
                f"{metrics['coverage_calibration']['mae']:.4f} | "
                f"{metrics['quantile_crossing']['rate']:.4f} |\n"
            )

        f.write("\n## Target Scale\n\n")
        f.write("| Horizon | Std | Abs p50 | Abs p90 |\n")
        f.write("|---------|-----|---------|---------|\n")
        for h, stats in target_scaling["by_horizon"].items():
            f.write(
                f"| {h} | {stats['std']:.8f} | "
                f"{stats['abs_p50']:.8f} | {stats['abs_p90']:.8f} |\n"
            )

        f.write("\n## By-Symbol Metrics\n\n")
        f.write("| Symbol | Samples | Pinball Loss | IC |\n")
        f.write("|--------|---------|--------------|----|\n")
        for s in eval_results['by_symbol']:
            f.write(f"| {s['symbol']} | {s['samples']:,} | {s['pinball_loss']:.6f} | {s['ic']:.4f} |\n")

        if sanity_warnings:
            f.write("\n## Sanity Warnings\n\n")
            for w in sanity_warnings:
                f.write(f"- {w}\n")

    print(f"Report: {md_path}")

    print(f"\n{'='*60}")
    print(f"M4 Evaluation (FAST) Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
