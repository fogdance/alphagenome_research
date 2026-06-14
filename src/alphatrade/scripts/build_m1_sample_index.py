#!/usr/bin/env python3
"""
Build sample index for M1 training.

This script:
1. Loads canonical bars
2. Generates sample indices with lookback windows
3. Computes labels for multiple horizons
4. Splits into train/val/test
5. Saves index files and statistics

Usage:
    # Single symbol
    python src/alphatrade/scripts/build_m1_sample_index.py --csymbol DCE.JM
    
    # Batch from universe
    python src/alphatrade/scripts/build_m1_sample_index.py \
      --universe configs/universe/m1_candidates.yaml
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
    parser = argparse.ArgumentParser(description="Build M1 sample index")
    parser.add_argument("--csymbol", type=str, help="Continuous symbol (e.g., DCE.JM)")
    parser.add_argument("--universe", type=str, help="Universe YAML file")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed/m1",
        help="Input directory with bars.parquet"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/m1",
        help="Output directory"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback window size (default: 60)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Sampling stride (default: 5)"
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,5,20,60",
        help="Comma-separated horizons (default: 1,5,20,60)"
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="2018-01-01",
        help="Train start date"
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2023-01-01",
        help="Train end date"
    )
    parser.add_argument(
        "--val-start",
        type=str,
        default="2023-01-01",
        help="Val start date"
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2024-01-01",
        help="Val end date"
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default="2024-01-01",
        help="Test start date"
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default="2026-01-01",
        help="Test end date"
    )
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def compute_labels(df: pd.DataFrame, horizons: list, eps: float = 1e-12) -> pd.DataFrame:
    """Compute forward return labels for multiple horizons."""
    df = df.copy()
    
    for h in horizons:
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
    """Generate sample indices for training."""
    max_horizon = max(horizons)
    n = len(df)
    
    samples = []
    
    for t in range(lookback - 1, n - max_horizon, stride):
        if require_same_segment:
            window_start = t - lookback + 1
            window_end = t
            target_end = t + max_horizon
            
            seg_start = df.iloc[window_start]["segment_id"]
            seg_window = df.iloc[window_end]["segment_id"]
            seg_target = df.iloc[target_end]["segment_id"]
            
            if seg_start != seg_window or seg_window != seg_target:
                continue
        
        sample = {
            "t": t,
            "x_start": t - lookback + 1,
            "x_end": t,
            "eob": df.iloc[t]["eob"],
            "segment_id": int(df.iloc[t]["segment_id"])
        }
        
        for h in horizons:
            label_col = f"y_h{h}"
            if label_col in df.columns:
                sample[label_col] = float(df.iloc[t][label_col])
        
        samples.append(sample)
    
    if not samples:
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
    """Split samples by time ranges."""
    if len(df) == 0:
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
                    "max": float(valid_labels.max()),
                }
    
    return stats


def process_symbol(csymbol: str, input_dir: str, output_dir: str,
                   lookback: int, stride: int, horizons: list,
                   train_start: str, train_end: str,
                   val_start: str, val_end: str,
                   test_start: str, test_end: str,
                   report_path: str) -> dict:
    """Process a single symbol."""
    print(f"\n{'='*60}")
    print(f"Building sample index for: {csymbol}")
    print(f"{'='*60}")
    
    # Load canonical bars
    input_path = Path(input_dir) / csymbol / "bars.parquet"
    
    if not input_path.exists():
        print(f"   ❌ Input file not found: {input_path}")
        return {
            'csymbol': csymbol,
            'status': 'FAIL',
            'error': 'Input file not found'
        }
    
    print(f"\n1. Loading canonical bars...")
    try:
        df = pd.read_parquet(input_path)
        print(f"   Loaded {len(df):,} rows")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return {
            'csymbol': csymbol,
            'status': 'FAIL',
            'error': f'Failed to load: {e}'
        }
    
    # Compute labels
    print(f"\n2. Computing labels (horizons={horizons})...")
    df = compute_labels(df, horizons)
    
    # Generate sample indices
    print(f"\n3. Generating sample indices (lookback={lookback}, stride={stride})...")
    index_df = generate_sample_indices(df, lookback, horizons, stride, require_same_segment=True)
    print(f"   Generated {len(index_df):,} samples")
    
    if len(index_df) == 0:
        print(f"   ⚠️  No samples generated")
        return {
            'csymbol': csymbol,
            'status': 'FAIL',
            'error': 'No samples generated'
        }
    
    # Split by time
    print(f"\n4. Splitting by time...")
    train_df, val_df, test_df = split_by_time(
        index_df, train_start, train_end, val_start, val_end, test_start, test_end
    )
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val: {len(val_df):,} samples")
    print(f"   Test: {len(test_df):,} samples")
    
    # Compute statistics
    print(f"\n5. Computing statistics...")
    train_stats = compute_statistics(train_df, horizons)
    val_stats = compute_statistics(val_df, horizons)
    test_stats = compute_statistics(test_df, horizons)
    
    # Save index files
    output_path = Path(output_dir) / csymbol
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / "index_train.parquet"
    val_path = output_path / "index_val.parquet"
    test_path = output_path / "index_test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\n✅ Saved index files:")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")
    print(f"   Test: {test_path}")
    
    # Append to report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    mode = 'a' if os.path.exists(report_path) else 'w'
    with open(report_path, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write("# M1 Sample Index Report\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Symbol: {csymbol}\n\n")
        f.write(f"- **Train samples**: {len(train_df):,}\n")
        f.write(f"- **Val samples**: {len(val_df):,}\n")
        f.write(f"- **Test samples**: {len(test_df):,}\n")
        f.write(f"- **Total samples**: {len(index_df):,}\n\n")
        
        f.write("### Label Statistics (Train)\n\n")
        for h_key, stats in train_stats["labels"].items():
            f.write(f"#### {h_key}\n\n")
            f.write(f"- Count: {stats['count']:,}\n")
            f.write(f"- Mean: {stats['mean']:.6f}\n")
            f.write(f"- Std: {stats['std']:.6f}\n")
            f.write(f"- Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
            f.write(f"- P01-P99: [{stats['p01']:.6f}, {stats['p99']:.6f}]\n\n")
    
    return {
        'csymbol': csymbol,
        'status': 'SUCCESS',
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'total_samples': len(index_df)
    }


def main():
    args = parse_args()
    
    # Validate arguments
    if not args.csymbol and not args.universe:
        print("Error: Either --csymbol or --universe must be specified")
        sys.exit(1)
    
    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(',')]
    
    # Get symbols to process
    if args.universe:
        with open(args.universe, 'r') as f:
            universe = yaml.safe_load(f)
        csymbols = universe.get('candidates', [])
        print(f"Loaded {len(csymbols)} symbols from universe")
    else:
        csymbols = [args.csymbol]
    
    print(f"\n{'='*60}")
    print(f"M1-T3: Build Sample Index")
    print(f"{'='*60}")
    print(f"Symbols: {len(csymbols)}")
    print(f"Lookback: {args.lookback}")
    print(f"Stride: {args.stride}")
    print(f"Horizons: {horizons}")
    print(f"Train: {args.train_start} to {args.train_end}")
    print(f"Val: {args.val_start} to {args.val_end}")
    print(f"Test: {args.test_start} to {args.test_end}")
    print(f"{'='*60}")
    
    # Process each symbol
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    report_path = reports_dir / "m1_t3_sample_index.md"
    reports = []
    
    for i, csymbol in enumerate(csymbols, 1):
        print(f"\n[{i}/{len(csymbols)}] Processing {csymbol}...")
        
        try:
            report = process_symbol(
                csymbol, args.input_dir, args.output_dir,
                args.lookback, args.stride, horizons,
                args.train_start, args.train_end,
                args.val_start, args.val_end,
                args.test_start, args.test_end,
                report_path
            )
            reports.append(report)
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            reports.append({
                'csymbol': csymbol,
                'status': 'FAIL',
                'error': f'Unexpected error: {e}'
            })
    
    # Summary
    success_count = sum(1 for r in reports if r['status'] == 'SUCCESS')
    total_train = sum(r.get('train_samples', 0) for r in reports if r['status'] == 'SUCCESS')
    total_val = sum(r.get('val_samples', 0) for r in reports if r['status'] == 'SUCCESS')
    total_test = sum(r.get('test_samples', 0) for r in reports if r['status'] == 'SUCCESS')
    
    print(f"\n{'='*60}")
    print(f"✅ M1-T3 Index Complete")
    print(f"{'='*60}")
    print(f"Success: {success_count}/{len(reports)}")
    print(f"Total train samples: {total_train:,}")
    print(f"Total val samples: {total_val:,}")
    print(f"Total test samples: {total_test:,}")
    print(f"Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
