#!/usr/bin/env python3
"""
Build canonical bars from archive data.

This script:
1. Loads raw data from archive
2. Computes gap-based segmentation
3. Computes basic features (ret_1m, hl_range, co_change, vol_log1p, pos_log1p)
4. Computes minute phase features (minute_sin, minute_cos)
5. Saves canonical bars to parquet
6. Generates profile report

Usage:
    python -m alphatrade.scripts.build_canonical_bars --symbol DCE.JM
    python -m alphatrade.scripts.build_canonical_bars --symbol SHFE.RB
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import json

# Direct import of ArchiveReader module without package init
archive_reader_path = Path(__file__).parent.parent / "data_pipeline" / "archive_reader.py"
import importlib.util
spec = importlib.util.spec_from_file_location("archive_reader", archive_reader_path)
archive_reader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_reader_module)
ArchiveReader = archive_reader_module.ArchiveReader


def parse_args():
    parser = argparse.ArgumentParser(description="Build canonical bars")
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
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_segments(df: pd.DataFrame, gap_seconds: int = 1800) -> pd.DataFrame:
    """
    Compute segment_id based on time gaps.
    
    A new segment starts when eob[t] - eob[t-1] > gap_seconds.
    """
    df = df.copy()
    
    # Compute time differences in seconds
    time_diff = df["eob"].diff().dt.total_seconds()
    
    # Mark segment boundaries (gap > threshold)
    is_new_segment = time_diff > gap_seconds
    is_new_segment.iloc[0] = True  # First row is always a new segment
    
    # Assign segment IDs
    df["segment_id"] = is_new_segment.cumsum() - 1
    
    return df


def compute_basic_features(df: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Compute basic price and volume features."""
    df = df.copy()
    
    # Price features
    df["ret_1m"] = np.log((df["close"] + eps) / (df["close"].shift(1) + eps))
    df["hl_range"] = np.log((df["high"] + eps) / (df["low"] + eps))
    df["co_change"] = np.log((df["close"] + eps) / (df["open"] + eps))
    
    # Volume/position features
    df["vol_log1p"] = np.log1p(df["volume"])
    df["pos_log1p"] = np.log1p(df["position"])
    
    return df


def compute_minute_features(
    df: pd.DataFrame,
    session_period_minutes: int = 1440,
    trading_day_start: str = "21:00"
) -> pd.DataFrame:
    """
    Compute minute phase features (sin/cos encoding).
    
    Args:
        session_period_minutes: Period for phase calculation (default 1440 = 24h)
        trading_day_start: Start time of trading day (default "21:00" for night session)
    """
    df = df.copy()
    
    # Parse trading day start
    start_hour, start_minute = map(int, trading_day_start.split(":"))
    start_minute_of_day = start_hour * 60 + start_minute
    
    # Compute minute of day
    minute_of_day = df["eob"].dt.hour * 60 + df["eob"].dt.minute
    
    # Compute minute index relative to trading day start
    minute_idx = (minute_of_day - start_minute_of_day) % session_period_minutes
    
    # Compute sin/cos encoding
    phase = 2 * np.pi * minute_idx / session_period_minutes
    df["minute_idx"] = minute_idx
    df["minute_sin"] = np.sin(phase)
    df["minute_cos"] = np.cos(phase)
    
    # is_open: always 1.0 for real trading minutes (no gap filling)
    df["is_open"] = 1.0
    
    return df


def cast_dtypes(df: pd.DataFrame, float_dtype: str = "float32") -> pd.DataFrame:
    """Cast columns to appropriate dtypes for storage."""
    df = df.copy()
    
    # Float columns
    float_cols = [
        "open", "high", "low", "close", "volume", "position",
        "ret_1m", "hl_range", "co_change", "vol_log1p", "pos_log1p",
        "minute_sin", "minute_cos", "is_open"
    ]
    
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float_dtype)
    
    # Integer columns
    if "segment_id" in df.columns:
        df["segment_id"] = df["segment_id"].astype("int32")
    if "minute_idx" in df.columns:
        df["minute_idx"] = df["minute_idx"].astype("int16")
    
    return df


def generate_profile(df: pd.DataFrame, symbol: str) -> dict:
    """Generate profile statistics for canonical bars."""
    profile = {
        "symbol": symbol,
        "rows": len(df),
        "min_eob": df["eob"].min(),
        "max_eob": df["eob"].max(),
        "segments": {
            "count": df["segment_id"].nunique(),
            "lengths": {}
        },
        "features": {}
    }
    
    # Segment length statistics
    segment_lengths = df.groupby("segment_id").size()
    profile["segments"]["lengths"] = {
        "min": int(segment_lengths.min()),
        "median": int(segment_lengths.median()),
        "p95": int(segment_lengths.quantile(0.95)),
        "max": int(segment_lengths.max())
    }
    
    # Feature statistics
    feature_cols = ["ret_1m", "hl_range", "co_change", "vol_log1p", "pos_log1p"]
    for col in feature_cols:
        if col in df.columns:
            profile["features"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "p01": float(df[col].quantile(0.01)),
                "p99": float(df[col].quantile(0.99)),
                "max": float(df[col].max())
            }
    
    return profile


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print(f"Building canonical bars for: {args.symbol}")
    print(f"Config: {args.config}")
    
    # Initialize reader
    # Extract config (support both old and new structure)
    archive_dir = config.get("archive", {}).get("archive_dir") or config.get("paths", {}).get("archive_dir")
    manifest_path = config.get("archive", {}).get("manifest_path", "") or config.get("paths", {}).get("manifest_path", "")
    if not manifest_path:
        manifest_path = "/data/juejin/_manifest/3a1a389f-56e9-46ce-8a04-047cbd456a44.jsonl"
    
    reader = ArchiveReader(archive_dir, manifest_path)
    
    # Load data
    print("\n1. Loading data from archive...")
    df = reader.load_symbol(args.symbol, dedup_eob="last")
    print(f"   Loaded {len(df):,} rows")
    
    # Compute segments
    print("\n2. Computing segments...")
    gap_seconds = config["canonical"]["gap_seconds"]
    df = compute_segments(df, gap_seconds=gap_seconds)
    print(f"   Found {df['segment_id'].nunique()} segments")
    
    # Compute basic features
    print("\n3. Computing basic features...")
    eps = config["canonical"]["eps"]
    df = compute_basic_features(df, eps=eps)
    
    # Compute minute features
    print("\n4. Computing minute phase features...")
    df = compute_minute_features(df, session_period_minutes=1440, trading_day_start="21:00")
    
    # Cast dtypes
    print("\n5. Casting dtypes...")
    df = cast_dtypes(df, float_dtype=config["canonical"]["cast_float"])
    
    # Generate profile
    print("\n6. Generating profile...")
    profile = generate_profile(df, args.symbol)
    
    # Save canonical bars
    output_dir = Path(args.output_dir) / args.symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bars.parquet"
    
    df.to_parquet(output_path, index=False)
    print(f"\n✅ Saved canonical bars: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Segments: {profile['segments']['count']}")
    
    # Append to report
    report_path = "reports/m0_1_canonical_profile.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    mode = 'a' if os.path.exists(report_path) else 'w'
    with open(report_path, mode, encoding='utf-8') as f:
        if mode == 'w':
            f.write("# Canonical Bars Profile Report\n\n")
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
    
    print(f"✅ Report saved: {report_path}")


if __name__ == "__main__":
    main()
