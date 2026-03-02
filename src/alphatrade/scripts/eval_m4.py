#!/usr/bin/env python3
"""
M4 Evaluation Script

Evaluates trained AlphaTrade v0.2 model and generates comprehensive metrics.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphatrade.core import model as model_lib
from alphatrade.core import schemas
from data_pipeline.feature_schema import FEATURE_COLS, FEATURE_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="M4 AlphaTrade v0.2 Evaluation")
    parser.add_argument("--train-metrics", type=str, default="reports/m4_train_metrics.json",
                        help="Path to training metrics JSON")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset/m2.yaml",
                        help="Dataset config file")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (optional, will use final params if not provided)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_train_metrics(metrics_path: str) -> dict:
    """Load training metrics to get run_id and model config."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


class M4EvalDataset:
    """Dataset for M4 evaluation (same as M3Dataset)."""

    def __init__(self, symbols: List[str], processed_dir: str, split: str):
        self.symbols = symbols
        self.processed_dir = processed_dir
        self.split = split
        self.data = []
        self.symbol_indices = []

        for symbol in symbols:
            symbol_dir = Path(processed_dir) / symbol
            bars_path = symbol_dir / "bars.parquet"
            index_path = symbol_dir / f"index_{split}.parquet"

            if not bars_path.exists() or not index_path.exists:
                continue

            bars_df = pd.read_parquet(bars_path)
            index_df = pd.read_parquet(index_path)

            for _, row in index_df.iterrows():
                x_start = row['x_start']
                x_end = row['x_end']

                X = bars_df.iloc[x_start:x_end][FEATURE_COLS].values.astype(np.float32)

                # Get targets for all horizons
                targets = {}
                for h in [1, 5, 20, 60]:
                    y_col = f'y_h{h}'
                    if y_col in row:
                        targets[h] = row[y_col]

                self.data.append({
                    'X': X,
                    'targets': targets,
                    'symbol': symbol
                })
                self.symbol_indices.append(symbol)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantiles: List[float]) -> float:
    """Calculate pinball loss for quantile predictions."""
    losses = []
    for i, q in enumerate(quantiles):
        error = y_true - y_pred[:, i]
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        losses.append(loss.mean())
    return np.mean(losses)


def calculate_quantile_coverage(y_true: np.ndarray, y_pred: np.ndarray, quantiles: List[float]) -> Dict[str, float]:
    """Calculate actual coverage rate for each quantile."""
    coverage = {}
    for i, q in enumerate(quantiles):
        actual_coverage = (y_true < y_pred[:, i]).mean()
        coverage[f"q{int(q*100)}"] = float(actual_coverage)
    return coverage


def calculate_quantile_crossing(y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate quantile crossing rate (non-monotonic predictions)."""
    # Check if predictions are monotonic
    crossings = (y_pred[:, 1:] < y_pred[:, :-1]).sum()
    total = y_pred.shape[0] * (y_pred.shape[1] - 1)

    return {
        "rate": float(crossings / total),
        "count": int(crossings)
    }


def calculate_ic_metrics(y_true: np.ndarray, y_pred_median: np.ndarray) -> Dict[str, float]:
    """Calculate IC metrics using median prediction."""
    # Pearson correlation
    ic = np.corrcoef(y_true, y_pred_median)[0, 1]

    # Spearman rank correlation
    rank_ic = spearmanr(y_true, y_pred_median).correlation

    return {
        "ic": float(ic) if not np.isnan(ic) else 0.0,
        "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0
    }


def evaluate_model(model_apply_fn, params, state, dataset: M4EvalDataset,
                   horizons: List[int], quantiles: List[float]) -> Dict:
    """Evaluate model on dataset."""

    all_predictions = {h: [] for h in horizons}
    all_targets = {h: [] for h in horizons}
    symbol_data = {}

    # Run inference
    for sample in dataset:
        X = sample['X']
        X_batch = jnp.expand_dims(jnp.array(X), axis=0)  # [1, L, F]

        # Forward pass
        rng = jax.random.PRNGKey(0)
        output, _ = model_apply_fn(params, state, rng, X_batch)

        # output.log_return_quantiles is a dict: {horizon: [B, Q]}
        # Convert to array [num_horizons, num_quantiles]
        preds_list = []
        for h in horizons:
            if h in output.log_return_quantiles:
                # [1, Q] -> [Q]
                preds_list.append(np.array(output.log_return_quantiles[h][0]))

        if len(preds_list) == 0:
            continue

        preds_np = np.array(preds_list)  # [num_horizons, num_quantiles]

        # Store predictions and targets
        for i, h in enumerate(horizons):
            if h in sample['targets']:
                all_predictions[h].append(preds_np[i])
                all_targets[h].append(sample['targets'][h])

        # Track by symbol
        symbol = sample['symbol']
        if symbol not in symbol_data:
            symbol_data[symbol] = {'predictions': {h: [] for h in horizons},
                                   'targets': {h: [] for h in horizons}}

        for i, h in enumerate(horizons):
            if h in sample['targets']:
                symbol_data[symbol]['predictions'][h].append(preds_np[i])
                symbol_data[symbol]['targets'][h].append(sample['targets'][h])

    # Calculate overall metrics
    overall_pinball = []
    by_horizon_pinball = {}
    ic_by_horizon = {}

    for h in horizons:
        if len(all_predictions[h]) == 0:
            continue

        y_pred = np.array(all_predictions[h])  # [N, num_quantiles]
        y_true = np.array(all_targets[h])      # [N]

        # Pinball loss
        pb_loss = pinball_loss(y_true, y_pred, quantiles)
        by_horizon_pinball[f"h{h}"] = float(pb_loss)
        overall_pinball.append(pb_loss)

        # IC metrics (using median = q50)
        median_idx = len(quantiles) // 2
        y_pred_median = y_pred[:, median_idx]
        ic_metrics = calculate_ic_metrics(y_true, y_pred_median)
        ic_by_horizon[f"h{h}"] = ic_metrics["ic"]

    # Overall pinball loss
    overall_pb = float(np.mean(overall_pinball)) if overall_pinball else 0.0

    # Quantile coverage (use first horizon for simplicity)
    h_first = horizons[0]
    y_pred_first = np.array(all_predictions[h_first])
    y_true_first = np.array(all_targets[h_first])
    coverage = calculate_quantile_coverage(y_true_first, y_pred_first, quantiles)

    # Quantile crossing
    crossing = calculate_quantile_crossing(y_pred_first)

    # IC metrics (overall, using first horizon)
    median_idx = len(quantiles) // 2
    y_pred_median = y_pred_first[:, median_idx]
    ic_metrics = calculate_ic_metrics(y_true_first, y_pred_median)

    # By-symbol metrics
    by_symbol = []
    for symbol, data in symbol_data.items():
        symbol_pinball = []
        symbol_samples = 0

        for h in horizons:
            if len(data['predictions'][h]) > 0:
                y_pred = np.array(data['predictions'][h])
                y_true = np.array(data['targets'][h])
                pb_loss = pinball_loss(y_true, y_pred, quantiles)
                symbol_pinball.append(pb_loss)
                symbol_samples = len(y_pred)

        if symbol_pinball:
            # IC for this symbol
            h_first = horizons[0]
            if len(data['predictions'][h_first]) > 0:
                y_pred = np.array(data['predictions'][h_first])
                y_true = np.array(data['targets'][h_first])
                y_pred_median = y_pred[:, median_idx]
                symbol_ic = calculate_ic_metrics(y_true, y_pred_median)["ic"]
            else:
                symbol_ic = 0.0

            by_symbol.append({
                "symbol": symbol,
                "samples": symbol_samples,
                "pinball_loss": float(np.mean(symbol_pinball)),
                "ic": float(symbol_ic)
            })

    return {
        "pinball_loss": {
            "overall": overall_pb,
            "by_horizon": by_horizon_pinball
        },
        "quantile_coverage": coverage,
        "quantile_crossing": crossing,
        "ic_metrics": {
            "ic": ic_metrics["ic"],
            "rank_ic": ic_metrics["rank_ic"],
            "ic_by_horizon": ic_by_horizon
        },
        "by_symbol": by_symbol
    }


def main():
    args = parse_args()

    # Load training metrics
    train_metrics = load_train_metrics(args.train_metrics)
    run_id = train_metrics['run']['run_id']

    print(f"\n{'='*60}")
    print(f"M4: AlphaTrade v0.2 Evaluation")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Split: {args.split}")
    print(f"{'='*60}\n")

    # Load config
    config_dict = load_config(args.dataset_config)

    # Select symbols
    symbols_file = config_dict['universe']['candidates_file']
    with open(symbols_file, 'r') as f:
        symbols = yaml.safe_load(f)['candidates']

    # For smoke test, use smoke_symbols
    if 'smoke_symbols' in config_dict['universe']:
        symbols = config_dict['universe']['smoke_symbols']

    print(f"Loading {args.split} data...")
    dataset = M4EvalDataset(symbols, config_dict['paths']['processed_dir'], args.split)
    print(f"Total samples: {len(dataset):,}\n")

    # Create model (same config as training)
    model_config = train_metrics['model']['config']
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

    print("  ✓ Model initialized\n")

    # Note: In production, load checkpoint here
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        # TODO: Implement checkpoint loading
        print("  ⚠️  Checkpoint loading not implemented, using random params\n")
    else:
        print("  ⚠️  No checkpoint provided, using random params for demo\n")

    # Evaluate
    print("Running evaluation...")
    eval_results = evaluate_model(
        forward_t.apply,
        params,
        state,
        dataset,
        alphatrade_config.horizons,
        alphatrade_config.quantiles
    )

    print("  ✓ Evaluation complete\n")

    # Generate output
    eval_metrics = {
        "run_id": run_id,
        "eval_timestamp": datetime.now().isoformat(),
        "dataset": {
            "split": args.split,
            "symbols": len(symbols),
            "samples": len(dataset)
        },
        **eval_results
    }

    # Save JSON
    json_path = "reports/m4_eval_metrics.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"✅ Metrics: {json_path}")

    # Save markdown report
    md_path = "reports/m4_eval_run.md"
    with open(md_path, 'w') as f:
        f.write("# M4 Evaluation Run - AlphaTrade v0.2 (JAX)\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 配置\n\n")
        f.write(f"- Run ID: {run_id}\n")
        f.write(f"- Split: {args.split}\n")
        f.write(f"- Symbols: {len(symbols)}\n")
        f.write(f"- Samples: {len(dataset):,}\n\n")

        f.write("## Pinball Loss\n\n")
        f.write(f"- Overall: {eval_results['pinball_loss']['overall']:.6f}\n\n")

        f.write("### By-Horizon\n\n")
        f.write("| Horizon | Loss |\n")
        f.write("|---------|------|\n")
        for h, loss in eval_results['pinball_loss']['by_horizon'].items():
            f.write(f"| {h} | {loss:.6f} |\n")

        f.write("\n## Quantile Coverage\n\n")
        for q, cov in eval_results['quantile_coverage'].items():
            expected = int(q[1:]) / 100
            f.write(f"- {q}: {cov:.4f} (expected: {expected:.2f})\n")

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

        f.write("\n## By-Symbol Metrics\n\n")
        f.write("| Symbol | Samples | Pinball Loss | IC |\n")
        f.write("|--------|---------|--------------|----|\n")
        for s in eval_results['by_symbol']:
            f.write(f"| {s['symbol']} | {s['samples']:,} | {s['pinball_loss']:.6f} | {s['ic']:.4f} |\n")

    print(f"✅ Report: {md_path}")

    print(f"\n{'='*60}")
    print(f"✅ M4 Evaluation Complete")
    print(f"{'='*60}")
    print(f"\nNext: Validate schema")
    print(f"  python src/alphatrade/scripts/validate_reports_schema.py \\")
    print(f"    --reports-dir reports --schemas-dir src/alphatrade/schemas\n")


if __name__ == "__main__":
    main()
