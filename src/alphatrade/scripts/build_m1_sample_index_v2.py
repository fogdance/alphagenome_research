#!/usr/bin/env python3
"""
Build sample index for M1 training (v2 with multi-threading).

This script:
1. Loads canonical bars (8 features)
2. Generates sample indices with lookback windows
3. Computes labels for multiple horizons
4. Splits into train/val/test
5. Saves index files
6. Supports multi-threading for parallel processing

Usage:
    # Single symbol
    python src/alphatrade/scripts/build_m1_sample_index_v2.py --csymbol DCE.JM
    
    # Batch with multi-threading
    python src/alphatrade/scripts/build_m1_sample_index_v2.py \
      --universe configs/universe/m1_selected.yaml \
      --num-workers 8
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Build M1 sample index (v2)")
    parser.add_argument("--csymbol", type=str, help="Continuous symbol (e.g., DCE.JM)")
    parser.add_argument("--universe", type=str, help="Universe YAML file")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed/m1_f8",
        help="Input directory with bars.parquet"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/m1_f8",
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if output exists"
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


def process_symbol(csymbol: str, input_dir: str, output_dir: str,
                   lookback: int, stride: int, horizons: list,
                   train_start: str, train_end: str,
                   val_start: str, val_end: str,
                   test_start: str, test_end: str,
                   force: bool) -> Dict:
    """Process a single symbol (worker function)."""
    try:
        # Check if output exists
        output_path = Path(output_dir) / csymbol / "index_train.parquet"
        if output_path.exists() and not force:
            return {
                'csymbol': csymbol,
                'status': 'SKIP',
                'message': 'Output already exists (use --force to rebuild)'
            }
        
        # Load canonical bars
        input_path = Path(input_dir) / csymbol / "bars.parquet"
        
        if not input_path.exists():
            return {
                'csymbol': csymbol,
                'status': 'FAIL',
                'error': 'Input file not found'
            }
        
        df = pd.read_parquet(input_path)
        
        # Compute labels
        df = compute_labels(df, horizons)
        
        # Generate sample indices
        index_df = generate_sample_indices(df, lookback, horizons, stride, require_same_segment=True)
        
        if len(index_df) == 0:
            return {
                'csymbol': csymbol,
                'status': 'FAIL',
                'error': 'No samples generated'
            }
        
        # Split by time
        train_df, val_df, test_df = split_by_time(
            index_df, train_start, train_end, val_start, val_end, test_start, test_end
        )
        
        # Save index files
        output_path = Path(output_dir) / csymbol
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / "index_train.parquet"
        val_path = output_path / "index_val.parquet"
        test_path = output_path / "index_test.parquet"
        
        # Atomic write
        for df_split, path in [(train_df, train_path), (val_df, val_path), (test_df, test_path)]:
            temp_path = path.with_suffix('.parquet.tmp')
            df_split.to_parquet(temp_path, index=False)
            temp_path.replace(path)
        
        return {
            'csymbol': csymbol,
            'status': 'SUCCESS',
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'total_samples': len(index_df)
        }
        
    except Exception as e:
        import traceback
        return {
            'csymbol': csymbol,
            'status': 'FAIL',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def generate_reports(results: list, output_dir: str, reports_dir: Path):
    """Generate JSON and Markdown reports."""
    
    # JSON report
    json_report = {
        'timestamp': datetime.now().isoformat(),
        'output_processed_dir': output_dir,
        'symbols_total': len(results),
        'success': sum(1 for r in results if r['status'] == 'SUCCESS'),
        'skip': sum(1 for r in results if r['status'] == 'SKIP'),
        'failed': sum(1 for r in results if r['status'] == 'FAIL'),
        'total_train_samples': sum(r.get('train_samples', 0) for r in results if r['status'] == 'SUCCESS'),
        'total_val_samples': sum(r.get('val_samples', 0) for r in results if r['status'] == 'SUCCESS'),
        'total_test_samples': sum(r.get('test_samples', 0) for r in results if r['status'] == 'SUCCESS'),
        'results': results
    }
    
    json_path = reports_dir / "m1_t5_2_index_rebuild.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON report: {json_path}")
    
    # Markdown report
    md_path = reports_dir / "m1_t5_2_index_rebuild.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# M1-T5.2 Sample Index Rebuild Report\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 总览\n\n")
        f.write(f"- **总品种数**: {json_report['symbols_total']}\n")
        f.write(f"- **成功**: {json_report['success']}\n")
        f.write(f"- **跳过**: {json_report['skip']}\n")
        f.write(f"- **失败**: {json_report['failed']}\n")
        f.write(f"- **Total Train Samples**: {json_report['total_train_samples']:,}\n")
        f.write(f"- **Total Val Samples**: {json_report['total_val_samples']:,}\n")
        f.write(f"- **Total Test Samples**: {json_report['total_test_samples']:,}\n\n")
        
        # Success table
        success_results = [r for r in results if r['status'] == 'SUCCESS']
        if success_results:
            f.write("## 成功品种\n\n")
            f.write("| Symbol | Train | Val | Test | Total |\n")
            f.write("|--------|-------|-----|------|-------|\n")
            for r in success_results:
                f.write(f"| {r['csymbol']} | {r['train_samples']:,} | {r['val_samples']:,} | "
                       f"{r['test_samples']:,} | {r['total_samples']:,} |\n")
            f.write("\n")
        
        # Failures
        fail_results = [r for r in results if r['status'] == 'FAIL']
        if fail_results:
            f.write("## 失败品种\n\n")
            for r in fail_results:
                f.write(f"### {r['csymbol']}\n\n")
                f.write(f"- **错误**: {r['error']}\n\n")
    
    print(f"✅ Markdown report: {md_path}")


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
    print(f"M1-T5.2: Build Sample Index (v2)")
    print(f"{'='*60}")
    print(f"Symbols: {len(csymbols)}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Lookback: {args.lookback}, Stride: {args.stride}")
    print(f"Horizons: {horizons}")
    print(f"Workers: {args.num_workers}")
    print(f"{'='*60}\n")
    
    # Process symbols
    results = []
    
    if args.num_workers > 1:
        # Multi-threaded processing
        print(f"Processing with {args.num_workers} workers...\n")
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    process_symbol,
                    csymbol, args.input_dir, args.output_dir,
                    args.lookback, args.stride, horizons,
                    args.train_start, args.train_end,
                    args.val_start, args.val_end,
                    args.test_start, args.test_end,
                    args.force
                ): csymbol
                for csymbol in csymbols
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                csymbol = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    status_icon = {'SUCCESS': '✅', 'SKIP': '⚠️', 'FAIL': '❌'}[result['status']]
                    print(f"[{i}/{len(csymbols)}] {csymbol}: {status_icon} {result['status']}")
                    
                except Exception as e:
                    print(f"[{i}/{len(csymbols)}] {csymbol}: ❌ Exception: {e}")
                    results.append({
                        'csymbol': csymbol,
                        'status': 'FAIL',
                        'error': str(e)
                    })
    else:
        # Single-threaded processing
        for i, csymbol in enumerate(csymbols, 1):
            print(f"[{i}/{len(csymbols)}] Processing {csymbol}...")
            result = process_symbol(
                csymbol, args.input_dir, args.output_dir,
                args.lookback, args.stride, horizons,
                args.train_start, args.train_end,
                args.val_start, args.val_end,
                args.test_start, args.test_end,
                args.force
            )
            results.append(result)
            
            status_icon = {'SUCCESS': '✅', 'SKIP': '⚠️', 'FAIL': '❌'}[result['status']]
            print(f"  {status_icon} {result['status']}")
    
    # Generate reports
    print("\nGenerating reports...")
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    generate_reports(results, args.output_dir, reports_dir)
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'SUCCESS')
    total_train = sum(r.get('train_samples', 0) for r in results if r['status'] == 'SUCCESS')
    total_val = sum(r.get('val_samples', 0) for r in results if r['status'] == 'SUCCESS')
    total_test = sum(r.get('test_samples', 0) for r in results if r['status'] == 'SUCCESS')
    
    print(f"\n{'='*60}")
    print(f"✅ M1-T5.2 Complete")
    print(f"{'='*60}")
    print(f"Success: {success}/{len(results)}")
    print(f"Total train samples: {total_train:,}")
    print(f"Total val samples: {total_val:,}")
    print(f"Total test samples: {total_test:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
