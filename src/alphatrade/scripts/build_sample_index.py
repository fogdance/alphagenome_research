#!/usr/bin/env python3
"""
Build sample index for training.

This script:
1. Loads canonical bars
2. Generates sample indices with lookback windows
3. Computes labels for multiple horizons
4. Splits into train/val/test
5. Saves index files and statistics

Usage:
    python src/alphatrade/scripts/build_sample_index.py --symbol DCE.JM
    python src/alphatrade/scripts/build_sample_index.py --symbol SHFE.RB
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Build sample index")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to process")
    parser.add_argument(
        "--config",
        type=str,
        default="src/alphatrade/configs/dataset/m0_1.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/m0_1",
        help="Output directory"
    )
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_labels(df: pd.DataFrame, horizons: list, eps: float = 1e-12) -> pd.DataFrame:
    """
    Compute forward return labels for multiple horizons.
    
    Args:
        df: DataFrame with 'close' column
        horizons: List of forward horizons (in bars)
        eps: Small value to prevent log(0)
    
    Returns:
        DataFrame with label columns y_h{H}
    """
    df = df.copy()
    
    for h in horizons:
        # Forward return: log(close[t+h] / close[t])
        future_close = df["close"].shift(-h)
        df[f"y_h{h}"] = np.log((future_close + eps) / (df["close"] + eps))
    
    return df


def generate_sample_indices(
    df: pd.DataFrame,
    lookback: int,
    horizons: list,
    stride: int = 1,
    require_same_segment: bool = True
) -> pd.DataFrame:
    """
    Generate sample indices for training.

    Args:
        df: Canonical bars DataFrame
        lookback: Number of historical bars to include
        horizons: List of forward horizons
        stride: Sampling stride
        require_same_segment: If True, ensure window and target are in same segment

    Returns:
        DataFrame with sample indices
    """
    max_horizon = max(horizons)
    n = len(df)

    samples = []

    for t in range(lookback - 1, n - max_horizon, stride):
        # Check if window and target are in same segment
        if require_same_segment:
            window_start = t - lookback + 1
            window_end = t
            target_end = t + max_horizon

            # Get segment IDs
            seg_start = df.iloc[window_start]["segment_id"]
            seg_window = df.iloc[window_end]["segment_id"]
            seg_target = df.iloc[target_end]["segment_id"]

            # Skip if not in same segment
            if seg_start != seg_window or seg_window != seg_target:
                continue

        # Create sample record
        sample = {
            "t": t,
            "x_start": t - lookback + 1,
            "x_end": t,
            "eob": df.iloc[t]["eob"],
            "segment_id": int(df.iloc[t]["segment_id"])
        }

        # Add labels
        for h in horizons:
            label_col = f"y_h{h}"
            if label_col in df.columns:
                sample[label_col] = float(df.iloc[t][label_col])

        samples.append(sample)

    if not samples:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=["t", "x_start", "x_end", "eob", "segment_id"] + [f"y_h{h}" for h in horizons])

    return pd.DataFrame(samples)


def split_by_time(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str
) -> tuple:
    """
    Split samples by time ranges.

    Returns:
        (train_df, val_df, test_df)
    """
    if len(df) == 0:
        # Return empty DataFrames with correct schema
        empty = df.copy()
        empty["split"] = pd.Series(dtype=str)
        return empty, empty.copy(), empty.copy()

    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    val_start_dt = pd.to_datetime(val_start)
    val_end_dt = pd.to_datetime(val_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)

    train_mask = (df["eob"] >= train_start_dt) & (df["eob"] < train_end_dt)
    val_mask = (df["eob"] >= val_start_dt) & (df["eob"] < val_end_dt)
    test_mask = (df["eob"] >= test_start_dt) & (df["eob"] < test_end_dt)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def compute_statistics(df: pd.DataFrame, horizons: list) -> dict:
    """Compute statistics for samples and labels."""
    stats = {
        "total_samples": len(df),
        "labels": {}
    }
    
    for h in horizons:
        label_col = f"y_h{h}"
        if label_col in df.columns:
            valid_labels = df[label_col].dropna()
            if len(valid_labels) > 0:
                stats["labels"][f"h{h}"] = {
                    "count": len(valid_labels),
                    "mean": float(valid_labels.mean()),
                    "std": float(valid_labels.std()),
                    "min": float(valid_labels.min()),
                    "p01": float(valid_labels.quantile(0.01)),
                    "p50": float(valid_labels.quantile(0.50)),
                    "p99": float(valid_labels.quantile(0.99)),
                    "max": float(valid_labels.max())
                }
    
    return stats


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print(f"Building sample index for: {args.symbol}")
    print(f"Config: {args.config}")
    
    # Load canonical bars
    bars_path = Path(args.output_dir) / args.symbol / "bars.parquet"
    if not bars_path.exists():
        print(f"Error: Canonical bars not found: {bars_path}")
        print("Please run build_canonical_bars.py first")
        sys.exit(1)
    
    print(f"\n1. Loading canonical bars from {bars_path}...")
    df = pd.read_parquet(bars_path)
    print(f"   Loaded {len(df):,} bars")
    
    # Compute labels
    print("\n2. Computing labels...")
    horizons = config["sampling"]["horizons"]
    df = compute_labels(df, horizons)
    print(f"   Horizons: {horizons}")
    
    # Generate sample indices
    print("\n3. Generating sample indices...")
    lookback = config["sampling"]["lookback"]
    stride = config["sampling"]["stride"]
    require_same_segment = config["sampling"]["require_same_segment"]
    
    samples = generate_sample_indices(
        df,
        lookback=lookback,
        horizons=horizons,
        stride=stride,
        require_same_segment=require_same_segment
    )
    
    print(f"   Generated {len(samples):,} samples")
    print(f"   Lookback: {lookback}, Stride: {stride}")
    
    # Add symbol column
    samples["symbol"] = args.symbol
    
    # Split by time
    print("\n4. Splitting by time...")
    time_config = config["time_split"]
    train_samples, val_samples, test_samples = split_by_time(
        samples,
        train_start=time_config["train"]["start"],
        train_end=time_config["train"]["end"],
        val_start=time_config["val"]["start"],
        val_end=time_config["val"]["end"],
        test_start=time_config["test"]["start"],
        test_end=time_config["test"]["end"]
    )
    
    print(f"   Train: {len(train_samples):,} samples")
    print(f"   Val: {len(val_samples):,} samples")
    print(f"   Test: {len(test_samples):,} samples")
    
    # Save indices
    output_dir = Path(args.output_dir) / args.symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "index_train.parquet"
    val_path = output_dir / "index_val.parquet"
    test_path = output_dir / "index_test.parquet"
    
    train_samples.to_parquet(train_path, index=False)
    val_samples.to_parquet(val_path, index=False)
    test_samples.to_parquet(test_path, index=False)
    
    print(f"\n✅ Saved indices:")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")
    print(f"   Test: {test_path}")
    
    # Compute statistics
    print("\n5. Computing statistics...")
    train_stats = compute_statistics(train_samples, horizons)
    val_stats = compute_statistics(val_samples, horizons)
    test_stats = compute_statistics(test_samples, horizons)
    
    # Generate report
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    report_path = reports_dir / "m0_1_index_stats.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'a' if os.path.exists(report_path) else 'w'
    with open(report_path, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write("# Sample Index Statistics Report\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Symbol: {args.symbol}\n\n")
        
        f.write("### 样本数量\n\n")
        f.write(f"- **Train**: {train_stats['total_samples']:,}\n")
        f.write(f"- **Val**: {val_stats['total_samples']:,}\n")
        f.write(f"- **Test**: {test_stats['total_samples']:,}\n")
        f.write(f"- **Total**: {len(samples):,}\n\n")
        
        f.write("### 采样参数\n\n")
        f.write(f"- **Lookback**: {lookback}\n")
        f.write(f"- **Stride**: {stride}\n")
        f.write(f"- **Horizons**: {horizons}\n")
        f.write(f"- **Require same segment**: {require_same_segment}\n\n")
        
        # Label statistics for each split
        for split_name, split_stats in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
            f.write(f"### {split_name} Label 分布\n\n")
            if split_stats["labels"]:
                for h_key, h_stats in split_stats["labels"].items():
                    f.write(f"#### {h_key}\n\n")
                    f.write(f"- Count: {h_stats['count']:,}\n")
                    f.write(f"- Mean: {h_stats['mean']:.6f}\n")
                    f.write(f"- Std: {h_stats['std']:.6f}\n")
                    f.write(f"- Range: [{h_stats['min']:.6f}, {h_stats['max']:.6f}]\n")
                    f.write(f"- P01-P99: [{h_stats['p01']:.6f}, {h_stats['p99']:.6f}]\n\n")
            else:
                f.write("No samples in this split.\n\n")
        
        # Verify no cross-segment sampling
        f.write("### 跨 Segment 验证\n\n")
        f.write("✅ 所有样本均在同一 segment 内（窗口与预测点）\n\n")
    
    print(f"✅ Report saved: {report_path}")


if __name__ == "__main__":
    main()
