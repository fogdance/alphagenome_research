#!/usr/bin/env python3
"""
Build canonical bars for M1 with 8 features (including is_session_open).

This script:
1. Loads raw_continuous_bars.parquet
2. Computes gap-based segmentation
3. Computes 7 basic features
4. Adds 8th feature (is_session_open)
5. Validates feature schema (F=8, dtype=float32)
6. Saves canonical bars to parquet
7. Supports multi-threading for parallel processing

Usage:
    # Single symbol
    python src/alphatrade/scripts/build_m1_canonical_bars_v2.py --csymbol DCE.JM
    
    # Batch with multi-threading
    python src/alphatrade/scripts/build_m1_canonical_bars_v2.py \
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import unified feature schema
from data_pipeline.feature_schema import FEATURE_COLS, FEATURE_DIM, FEATURE_DTYPE, validate_feature_schema
import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Build M1 canonical bars (8 features)")
    parser.add_argument("--csymbol", type=str, help="Continuous symbol (e.g., DCE.JM)")
    parser.add_argument("--universe", type=str, help="Universe YAML file")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed/m1",
        help="Input directory with raw_continuous_bars.parquet"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/m1_f8",
        help="Output directory (default: m1_f8 for new 8-feature version)"
    )
    parser.add_argument(
        "--gap-seconds",
        type=int,
        default=1800,
        help="Gap threshold for segmentation (default: 1800)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for division (default: 1e-8)"
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


def compute_segments(df: pd.DataFrame, gap_seconds: int = 1800) -> pd.DataFrame:
    """Compute segment_id based on time gaps."""
    df = df.copy()
    
    time_diff = df["eob"].diff().dt.total_seconds()
    is_new_segment = time_diff > gap_seconds
    is_new_segment.iloc[0] = True
    df["segment_id"] = is_new_segment.cumsum()
    
    return df


def compute_basic_features(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """Compute basic features (first 7)."""
    df = df.copy()
    
    # ret_1m: 1-minute return
    df["ret_1m"] = df["close"].pct_change()
    
    # hl_range: (high - low) / close
    df["hl_range"] = (df["high"] - df["low"]) / (df["close"] + eps)
    
    # co_change: (close - open) / open
    df["co_change"] = (df["close"] - df["open"]) / (df["open"] + eps)
    
    # vol_log1p: log(1 + volume)
    df["vol_log1p"] = np.log1p(df["volume"])
    
    # pos_log1p: log(1 + position)
    if "position" in df.columns:
        df["pos_log1p"] = np.log1p(df["position"])
    else:
        df["pos_log1p"] = 0.0
    
    return df


def compute_minute_features(df: pd.DataFrame, session_period_minutes: int = 1440) -> pd.DataFrame:
    """Compute minute phase features."""
    df = df.copy()
    
    df["minute_of_day"] = df["eob"].dt.hour * 60 + df["eob"].dt.minute
    phase = 2 * np.pi * df["minute_of_day"] / session_period_minutes
    
    df["minute_sin"] = np.sin(phase)
    df["minute_cos"] = np.cos(phase)
    
    df = df.drop(columns=["minute_of_day"])
    
    return df


def add_session_open_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add 8th feature: is_session_open."""
    df = df.copy()
    
    # Simple version: all 1.0
    # Future enhancement: mark first bar of each segment as 1.0, others as 0.0
    df["is_session_open"] = np.float32(1.0)
    
    return df


def cast_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all feature columns to float32."""
    df = df.copy()
    
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(FEATURE_DTYPE)
    
    # Cast other columns
    if "segment_id" in df.columns:
        df["segment_id"] = df["segment_id"].astype("int32")
    
    return df


def validate_and_assert_schema(df: pd.DataFrame, csymbol: str):
    """Validate feature schema and assert correctness."""
    # Check feature count
    feature_cols_present = [c for c in df.columns if c in FEATURE_COLS]
    assert len(feature_cols_present) == FEATURE_DIM, \
        f"{csymbol}: Expected {FEATURE_DIM} features, got {len(feature_cols_present)}"
    
    # Check is_session_open exists
    assert "is_session_open" in df.columns, \
        f"{csymbol}: Missing is_session_open (8th feature)"
    
    # Check dtype
    assert df["is_session_open"].dtype == FEATURE_DTYPE, \
        f"{csymbol}: is_session_open dtype is {df['is_session_open'].dtype}, expected {FEATURE_DTYPE}"
    
    # Use unified validation
    is_valid, error_msg = validate_feature_schema(df, strict=True)
    assert is_valid, f"{csymbol}: Schema validation failed: {error_msg}"


def process_symbol(csymbol: str, input_dir: str, output_dir: str, 
                   gap_seconds: int, eps: float, force: bool) -> Dict:
    """Process a single symbol (worker function for multiprocessing)."""
    try:
        # Check if output exists
        output_path = Path(output_dir) / csymbol / "bars.parquet"
        if output_path.exists() and not force:
            return {
                'csymbol': csymbol,
                'status': 'SKIP',
                'message': 'Output already exists (use --force to rebuild)'
            }
        
        # Load raw continuous bars
        input_path = Path(input_dir) / csymbol / "raw_continuous_bars.parquet"
        
        if not input_path.exists():
            return {
                'csymbol': csymbol,
                'status': 'FAIL',
                'error': 'Input file not found'
            }
        
        df = pd.read_parquet(input_path)
        rows_input = len(df)
        
        # Compute segments
        df = compute_segments(df, gap_seconds=gap_seconds)
        n_segments = df['segment_id'].nunique()
        
        # Compute features (first 7)
        df = compute_basic_features(df, eps=eps)
        df = compute_minute_features(df)
        
        # Add 8th feature
        df = add_session_open_feature(df)
        
        # Cast dtypes
        df = cast_feature_dtypes(df)
        
        # Validate schema
        validate_and_assert_schema(df, csymbol)
        
        # Save to temp file first (atomic write)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix('.parquet.tmp')
        df.to_parquet(temp_path, index=False)
        
        # Atomic replace
        temp_path.replace(output_path)
        
        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        return {
            'csymbol': csymbol,
            'status': 'SUCCESS',
            'rows': len(df),
            'segments': n_segments,
            'feature_cols': FEATURE_COLS,
            'feature_dim': FEATURE_DIM,
            'dtypes': {col: str(df[col].dtype) for col in FEATURE_COLS},
            'min_eob': str(df['eob'].min()),
            'max_eob': str(df['eob'].max()),
            'output_path': str(output_path),
            'output_size_mb': file_size_mb
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
        'git_sha': os.popen('git rev-parse --short HEAD 2>/dev/null').read().strip() or 'unknown',
        'input_processed_dir': results[0].get('input_dir', 'unknown') if results else 'unknown',
        'output_processed_dir': output_dir,
        'symbols_total': len(results),
        'success': sum(1 for r in results if r['status'] == 'SUCCESS'),
        'skip': sum(1 for r in results if r['status'] == 'SKIP'),
        'failed': sum(1 for r in results if r['status'] == 'FAIL'),
        'results': results
    }
    
    json_path = reports_dir / "m1_t5_1_feature_patch.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON report: {json_path}")
    
    # Markdown report
    md_path = reports_dir / "m1_t5_1_feature_patch.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# M1-T5.1 Feature Patch Report (8D Rebuild)\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 总览\n\n")
        f.write(f"- **总品种数**: {json_report['symbols_total']}\n")
        f.write(f"- **成功**: {json_report['success']}\n")
        f.write(f"- **跳过**: {json_report['skip']}\n")
        f.write(f"- **失败**: {json_report['failed']}\n")
        f.write(f"- **输出目录**: `{output_dir}`\n\n")
        
        # Success table
        success_results = [r for r in results if r['status'] == 'SUCCESS']
        if success_results:
            f.write("## 成功品种\n\n")
            f.write("| Symbol | Rows | Segments | Feature Dim | Size (MB) |\n")
            f.write("|--------|------|----------|-------------|------------|\n")
            for r in success_results:
                f.write(f"| {r['csymbol']} | {r['rows']:,} | {r['segments']} | "
                       f"{r['feature_dim']} | {r['output_size_mb']:.1f} |\n")
            f.write("\n")
            
            # Sample schema (first 3)
            f.write("### 抽样 Schema (前 3 个品种)\n\n")
            for r in success_results[:3]:
                f.write(f"#### {r['csymbol']}\n\n")
                f.write("**特征列** (8 个):\n\n")
                for i, col in enumerate(r['feature_cols'], 1):
                    dtype = r['dtypes'][col]
                    f.write(f"{i}. `{col}` ({dtype})\n")
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
    
    # Get symbols to process
    if args.universe:
        with open(args.universe, 'r') as f:
            universe = yaml.safe_load(f)
        csymbols = universe.get('candidates', [])
        print(f"Loaded {len(csymbols)} symbols from universe")
    else:
        csymbols = [args.csymbol]
    
    print(f"\n{'='*60}")
    print(f"M1-T5.1: Build Canonical Bars (8 Features)")
    print(f"{'='*60}")
    print(f"Symbols: {len(csymbols)}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Force rebuild: {args.force}")
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
                    args.gap_seconds, args.eps, args.force
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
                args.gap_seconds, args.eps, args.force
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
    print(f"\n{'='*60}")
    print(f"✅ M1-T5.1 Complete")
    print(f"{'='*60}")
    print(f"Success: {success}/{len(results)}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
