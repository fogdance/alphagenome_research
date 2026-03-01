#!/usr/bin/env python3
"""
Build canonical bars for M1 from raw_continuous_bars.

This script:
1. Loads raw_continuous_bars.parquet (already stitched from real contracts)
2. Computes gap-based segmentation
3. Computes basic features (ret_1m, hl_range, co_change, vol_log1p, pos_log1p)
4. Computes minute phase features (minute_sin, minute_cos)
5. Saves canonical bars to parquet
6. Generates profile report

Usage:
    # Single symbol
    python src/alphatrade/scripts/build_m1_canonical_bars.py --csymbol DCE.JM
    
    # Batch from universe
    python src/alphatrade/scripts/build_m1_canonical_bars.py \
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Build M1 canonical bars")
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
        default="data/processed/m1",
        help="Output directory"
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
    return parser.parse_args()


def compute_segments(df: pd.DataFrame, gap_seconds: int = 1800) -> pd.DataFrame:
    """Compute segment_id based on time gaps."""
    df = df.copy()
    
    # Compute time differences in seconds
    time_diff = df["eob"].diff().dt.total_seconds()
    
    # Mark segment boundaries
    is_new_segment = time_diff > gap_seconds
    is_new_segment.iloc[0] = True
    
    # Assign segment IDs
    df["segment_id"] = is_new_segment.cumsum()
    
    return df


def compute_basic_features(df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """Compute basic features."""
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


def compute_minute_features(df: pd.DataFrame, session_period_minutes: int = 1440, 
                            trading_day_start: str = "21:00") -> pd.DataFrame:
    """Compute minute phase features."""
    df = df.copy()
    
    # Extract minute of day
    df["minute_of_day"] = df["eob"].dt.hour * 60 + df["eob"].dt.minute
    
    # Compute phase
    phase = 2 * np.pi * df["minute_of_day"] / session_period_minutes
    
    df["minute_sin"] = np.sin(phase)
    df["minute_cos"] = np.cos(phase)
    
    # Drop helper column
    df = df.drop(columns=["minute_of_day"])
    
    return df


def cast_dtypes(df: pd.DataFrame, float_dtype: str = "float32") -> pd.DataFrame:
    """Cast dtypes for efficiency."""
    df = df.copy()
    
    # Float columns
    float_cols = ["open", "high", "low", "close", "volume", "position",
                  "ret_1m", "hl_range", "co_change", "vol_log1p", "pos_log1p",
                  "minute_sin", "minute_cos"]
    
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float_dtype)
    
    # Int columns
    if "segment_id" in df.columns:
        df["segment_id"] = df["segment_id"].astype("int32")
    
    return df


def generate_profile(df: pd.DataFrame, symbol: str) -> dict:
    """Generate profile statistics."""
    profile = {
        "symbol": symbol,
        "rows": len(df),
        "min_eob": str(df["eob"].min()),
        "max_eob": str(df["eob"].max()),
        "segments": {
            "count": df["segment_id"].nunique(),
            "lengths": {
                "min": int(df.groupby("segment_id").size().min()),
                "median": int(df.groupby("segment_id").size().median()),
                "p95": int(df.groupby("segment_id").size().quantile(0.95)),
                "max": int(df.groupby("segment_id").size().max()),
            }
        },
        "features": {}
    }
    
    # Feature statistics
    feature_cols = ["ret_1m", "hl_range", "co_change", "vol_log1p", "pos_log1p"]
    for col in feature_cols:
        if col in df.columns:
            profile["features"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "p01": float(df[col].quantile(0.01)),
                "p99": float(df[col].quantile(0.99)),
            }
    
    return profile


def process_symbol(csymbol: str, input_dir: str, output_dir: str, 
                   gap_seconds: int, eps: float, report_path: str) -> dict:
    """Process a single symbol."""
    print(f"\n{'='*60}")
    print(f"Building canonical bars for: {csymbol}")
    print(f"{'='*60}")
    
    # Load raw continuous bars
    input_path = Path(input_dir) / csymbol / "raw_continuous_bars.parquet"
    
    if not input_path.exists():
        print(f"   ❌ Input file not found: {input_path}")
        return {
            'csymbol': csymbol,
            'status': 'FAIL',
            'error': 'Input file not found'
        }
    
    print(f"\n1. Loading raw continuous bars...")
    print(f"   Path: {input_path}")
    
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
    
    # Compute segments
    print(f"\n2. Computing segments (gap_seconds={gap_seconds})...")
    df = compute_segments(df, gap_seconds=gap_seconds)
    n_segments = df['segment_id'].nunique()
    print(f"   Found {n_segments} segments")
    
    # Compute basic features
    print(f"\n3. Computing basic features...")
    df = compute_basic_features(df, eps=eps)
    
    # Compute minute features
    print(f"\n4. Computing minute phase features...")
    df = compute_minute_features(df)
    
    # Cast dtypes
    print(f"\n5. Casting dtypes...")
    df = cast_dtypes(df, float_dtype="float32")
    
    # Generate profile
    print(f"\n6. Generating profile...")
    profile = generate_profile(df, csymbol)
    
    # Save canonical bars
    output_path = Path(output_dir) / csymbol / "bars.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Saved canonical bars: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Segments: {n_segments}")
    print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Append to report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    mode = 'a' if os.path.exists(report_path) else 'w'
    with open(report_path, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write("# M1 Canonical Bars Profile Report\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Symbol: {profile['symbol']}\n\n")
        f.write(f"- **总行数**: {profile['rows']:,}\n")
        f.write(f"- **时间范围**: {profile['min_eob']} 至 {profile['max_eob']}\n\n")
        
        f.write("### Segment 统计\n\n")
        f.write(f"- **Segment 数量**: {profile['segments']['count']}\n")
        f.write(f"- **Segment 长度**:\n")
        f.write(f"  - Min: {profile['segments']['lengths']['min']}\n")
        f.write(f"  - Median: {profile['segments']['lengths']['median']}\n")
        f.write(f"  - P95: {profile['segments']['lengths']['p95']}\n")
        f.write(f"  - Max: {profile['segments']['lengths']['max']}\n\n")
        
        f.write("### 特征统计\n\n")
        for feat, stats in profile['features'].items():
            f.write(f"#### {feat}\n\n")
            f.write(f"- Mean: {stats['mean']:.6f}\n")
            f.write(f"- Std: {stats['std']:.6f}\n")
            f.write(f"- Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
            f.write(f"- P01-P99: [{stats['p01']:.6f}, {stats['p99']:.6f}]\n\n")
    
    return {
        'csymbol': csymbol,
        'status': 'SUCCESS',
        'rows': len(df),
        'segments': n_segments,
        'output_path': str(output_path),
        'output_size_mb': output_path.stat().st_size / (1024*1024)
    }


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
    print(f"M1-T3: Build Canonical Bars")
    print(f"{'='*60}")
    print(f"Symbols: {len(csymbols)}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Gap seconds: {args.gap_seconds}")
    print(f"{'='*60}")
    
    # Process each symbol
    report_path = "reports/m1_t3_canonical_profile.md"
    reports = []
    
    for i, csymbol in enumerate(csymbols, 1):
        print(f"\n[{i}/{len(csymbols)}] Processing {csymbol}...")
        
        try:
            report = process_symbol(
                csymbol,
                args.input_dir,
                args.output_dir,
                args.gap_seconds,
                args.eps,
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
    print(f"\n{'='*60}")
    print(f"✅ M1-T3 Canonical Complete")
    print(f"{'='*60}")
    print(f"Success: {success_count}/{len(reports)}")
    print(f"Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
