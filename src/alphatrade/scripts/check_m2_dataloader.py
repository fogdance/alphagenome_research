#!/usr/bin/env python3
"""
M2 Dataloader Check Script

Validates dataset/dataloader quality:
- Sample shape verification
- NaN/Inf detection
- Segment constraint validation
- Multi-symbol sampling statistics
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class Dataset:
        pass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.feature_profiles import (
    FeatureProfile,
    feature_profile_to_dict,
    resolve_feature_profile,
)
import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="M2 Dataloader Check")
    parser.add_argument("--config", type=str, default="configs/dataset/m2.yaml")
    parser.add_argument("--smoke", action="store_true", help="Use smoke symbols")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to check")
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class M2Dataset(Dataset):
    """M2 dataset for validation."""
    
    def __init__(
        self,
        symbols: List[str],
        processed_root: str,
        feature_profile: FeatureProfile,
        lookback: int,
        split: str = "train"
    ):
        self.symbols = symbols
        self.feature_profile = feature_profile
        self.processed_root = feature_profile.processed_root or processed_root
        self.lookback = int(lookback)
        self.split = split
        self.features = list(feature_profile.feature_cols)
        self.feature_dim = feature_profile.feature_dim
        
        # Load bars and indices
        self.bars_dict = {}
        self.indices = []
        self.symbol_sample_counts = {}
        
        print(f"\nLoading {split} data...")
        for symbol in symbols:
            symbol_dir = Path(self.processed_root) / symbol
            
            # Load bars
            bars_path = symbol_dir / "bars.parquet"
            bars_df = pd.read_parquet(bars_path)
            
            # Extract features and handle NaN
            missing_features = [c for c in self.features if c not in bars_df.columns]
            if missing_features:
                raise ValueError(f"{symbol}: missing feature columns: {missing_features}")
            feature_data = bars_df[self.features].values.astype(np.float32)
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Store bars with segment_id for validation
            self.bars_dict[symbol] = {
                'features': feature_data,
                'segment_id': bars_df['segment_id'].values
            }
            
            # Load index
            index_path = symbol_dir / f"index_{split}.parquet"
            index_df = pd.read_parquet(index_path)
            
            # Store indices
            for _, row in index_df.iterrows():
                self.indices.append({
                    "symbol": symbol,
                    "x_start": int(row["x_start"]),
                    "x_end": int(row["x_end"]),
                    "segment_id": int(row["segment_id"]),
                    "y_h1": float(row["y_h1"]),
                    "y_h5": float(row["y_h5"]),
                    "y_h20": float(row["y_h20"]),
                    "y_h60": float(row["y_h60"]),
                })
            
            self.symbol_sample_counts[symbol] = len(index_df)
            print(f"  {symbol}: {len(bars_df):,} bars, {len(index_df):,} samples")
        
        print(f"Total {split} samples: {len(self.indices):,}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        entry = self.indices[idx]
        symbol = entry["symbol"]
        x_start = entry["x_start"]
        x_end = entry["x_end"]
        
        # Slice window
        x = self.bars_dict[symbol]['features'][x_start:x_end+1]
        
        # Labels
        y = np.array([
            entry["y_h1"],
            entry["y_h5"],
            entry["y_h20"],
            entry["y_h60"]
        ], dtype=np.float32)
        
        x_out = torch.from_numpy(x) if torch is not None else x
        y_out = torch.from_numpy(y) if torch is not None else y

        return {
            "x": x_out,
            "y": y_out,
            "symbol": symbol,
            "x_start": x_start,
            "x_end": x_end,
            "segment_id": entry["segment_id"]
        }
    
    def get_symbol_sample_counts(self):
        return self.symbol_sample_counts


def validate_sample(sample: dict, dataset: M2Dataset) -> dict:
    """Validate a single sample."""
    x_arr = np.asarray(sample["x"])
    y_arr = np.asarray(sample["y"])
    result = {
        "symbol": sample["symbol"],
        "x_start": sample["x_start"],
        "x_end": sample["x_end"],
        "segment_id": sample["segment_id"],
        "x_shape": tuple(x_arr.shape),
        "y_shape": tuple(y_arr.shape),
        "x_has_nan": bool(np.isnan(x_arr).any()),
        "x_has_inf": bool(np.isinf(x_arr).any()),
        "y_has_nan": bool(np.isnan(y_arr).any()),
        "y_has_inf": bool(np.isinf(y_arr).any()),
        "segment_consistent": False,
        "errors": []
    }
    
    # Check shape
    expected_x_shape = (dataset.lookback, dataset.feature_dim)
    expected_y_shape = (4,)
    
    if result["x_shape"] != expected_x_shape:
        result["errors"].append(f"X shape mismatch: expected {expected_x_shape}, got {result['x_shape']}")
    
    if result["y_shape"] != expected_y_shape:
        result["errors"].append(f"Y shape mismatch: expected {expected_y_shape}, got {result['y_shape']}")
    
    # Check NaN/Inf
    if result["x_has_nan"]:
        result["errors"].append("X contains NaN")
    
    if result["x_has_inf"]:
        result["errors"].append("X contains Inf")
    
    if result["y_has_nan"]:
        result["errors"].append("Y contains NaN")
    
    if result["y_has_inf"]:
        result["errors"].append("Y contains Inf")
    
    # Check segment consistency
    symbol = sample["symbol"]
    x_start = sample["x_start"]
    x_end = sample["x_end"]
    expected_segment_id = sample["segment_id"]
    
    segment_ids = dataset.bars_dict[symbol]['segment_id'][x_start:x_end+1]
    all_same_segment = np.all(segment_ids == expected_segment_id)
    
    result["segment_consistent"] = bool(all_same_segment)
    
    if not all_same_segment:
        unique_segments = np.unique(segment_ids)
        result["errors"].append(f"Segment crossing detected: {unique_segments.tolist()}")
    
    result["valid"] = len(result["errors"]) == 0
    
    return result


def check_dataloader(dataset: M2Dataset, num_samples: int = 10) -> List[dict]:
    """Check random samples from dataset."""
    print(f"\nChecking {num_samples} random samples...")
    
    # Sample indices
    total_samples = len(dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    results = []
    for idx in sample_indices:
        sample = dataset[idx]
        validation = validate_sample(sample, dataset)
        results.append(validation)
        
        status = "✅" if validation["valid"] else "❌"
        print(f"  Sample {idx}: {validation['symbol']} {status}")
        if not validation["valid"]:
            for error in validation["errors"]:
                print(f"    - {error}")
    
    return results


def check_symbol_samples(dataset: M2Dataset, symbols: List[str], samples_per_symbol: int = 3) -> List[dict]:
    """Check specific samples from each symbol."""
    print(f"\nChecking {samples_per_symbol} samples per symbol...")
    
    results = []
    
    for symbol in symbols:
        # Find indices for this symbol
        symbol_indices = [i for i, entry in enumerate(dataset.indices) if entry["symbol"] == symbol]
        
        if not symbol_indices:
            print(f"  {symbol}: No samples found")
            continue
        
        # Sample from this symbol
        sample_indices = random.sample(symbol_indices, min(samples_per_symbol, len(symbol_indices)))
        
        print(f"  {symbol}:")
        for idx in sample_indices:
            sample = dataset[idx]
            validation = validate_sample(sample, dataset)
            results.append(validation)
            
            status = "✅" if validation["valid"] else "❌"
            print(f"    Sample {idx}: {status} shape={validation['x_shape']}, segment={validation['segment_id']}")
            if not validation["valid"]:
                for error in validation["errors"]:
                    print(f"      - {error}")
    
    return results


def generate_reports(
    config: dict,
    feature_profile: FeatureProfile,
    dataset: M2Dataset,
    random_results: List[dict],
    symbol_results: List[dict],
    reports_dir: Path,
):
    """Generate JSON and Markdown reports."""
    
    # Aggregate statistics
    total_checks = len(random_results) + len(symbol_results)
    valid_checks = sum(1 for r in random_results + symbol_results if r["valid"])
    
    # JSON report
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "processed_dir": dataset.processed_root,
            "feature_profile": feature_profile_to_dict(feature_profile),
            "feature_dim": feature_profile.feature_dim,
            "feature_cols": list(feature_profile.feature_cols),
            "lookback": config['sample_index']['lookback'],
            "horizons": config['sample_index']['horizons']
        },
        "dataset": {
            "total_samples": len(dataset),
            "symbols": len(dataset.symbols),
            "symbol_sample_counts": dataset.get_symbol_sample_counts()
        },
        "validation": {
            "total_checks": total_checks,
            "valid_checks": valid_checks,
            "invalid_checks": total_checks - valid_checks,
            "success_rate": valid_checks / total_checks if total_checks > 0 else 0.0
        },
        "random_samples": random_results,
        "symbol_samples": symbol_results
    }
    
    json_path = reports_dir / "m2_t1_dataloader_check.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"\n✅ JSON report: {json_path}")
    
    # Markdown report
    md_path = reports_dir / "m2_t1_dataloader_check.md"
    with open(md_path, 'w') as f:
        f.write("# M2-T1 Dataloader Check Report\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 配置\n\n")
        f.write(f"- **Processed dir**: `{dataset.processed_root}`\n")
        f.write(f"- **Feature profile**: `{feature_profile.profile_id}`\n")
        f.write(f"- **Feature dim**: {feature_profile.feature_dim}\n")
        f.write(f"- **Lookback**: {config['sample_index']['lookback']}\n")
        f.write(f"- **Horizons**: {config['sample_index']['horizons']}\n\n")
        
        f.write("## Dataset 统计\n\n")
        f.write(f"- **Total samples**: {len(dataset):,}\n")
        f.write(f"- **Symbols**: {len(dataset.symbols)}\n\n")
        
        f.write("### Symbol Sample Counts\n\n")
        f.write("| Symbol | Samples |\n")
        f.write("|--------|----------|\n")
        for symbol, count in sorted(dataset.get_symbol_sample_counts().items()):
            f.write(f"| {symbol} | {count:,} |\n")
        f.write("\n")
        
        f.write("## 验证结果\n\n")
        f.write(f"- **Total checks**: {total_checks}\n")
        f.write(f"- **Valid**: {valid_checks} ({valid_checks/total_checks*100:.1f}%)\n")
        f.write(f"- **Invalid**: {total_checks - valid_checks}\n\n")
        
        # Random samples
        f.write("### Random Samples\n\n")
        f.write("| Symbol | X Shape | Y Shape | Segment OK | Valid |\n")
        f.write("|--------|---------|---------|------------|-------|\n")
        for r in random_results:
            status = "✅" if r["valid"] else "❌"
            segment_ok = "✅" if r["segment_consistent"] else "❌"
            f.write(f"| {r['symbol']} | {r['x_shape']} | {r['y_shape']} | {segment_ok} | {status} |\n")
        f.write("\n")
        
        # Symbol samples
        f.write("### Per-Symbol Samples\n\n")
        f.write("| Symbol | X Shape | Y Shape | Segment OK | Valid |\n")
        f.write("|--------|---------|---------|------------|-------|\n")
        for r in symbol_results:
            status = "✅" if r["valid"] else "❌"
            segment_ok = "✅" if r["segment_consistent"] else "❌"
            f.write(f"| {r['symbol']} | {r['x_shape']} | {r['y_shape']} | {segment_ok} | {status} |\n")
        f.write("\n")
        
        # Errors
        all_results = random_results + symbol_results
        errors = [r for r in all_results if not r["valid"]]
        
        if errors:
            f.write("### 错误详情\n\n")
            for r in errors:
                f.write(f"#### {r['symbol']} (segment {r['segment_id']})\n\n")
                for error in r["errors"]:
                    f.write(f"- {error}\n")
                f.write("\n")
        else:
            f.write("### 错误详情\n\n")
            f.write("✅ 无错误\n\n")
    
    print(f"✅ Markdown report: {md_path}")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    feature_profile = resolve_feature_profile(config)
    processed_dir = feature_profile.processed_root or config['paths']['processed_dir']
    
    # Get symbols
    if args.smoke:
        symbols = config['universe']['smoke_symbols']
    else:
        candidates_file = config['universe']['candidates_file']
        with open(candidates_file, 'r') as f:
            universe = yaml.safe_load(f)
        symbols = universe['candidates']
    
    print(f"\n{'='*60}")
    print(f"M2-T1: Dataloader Check")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Feature profile: {feature_profile.profile_id} ({feature_profile.feature_dim} cols)")
    print(f"Processed dir: {processed_dir}")
    print(f"Num samples to check: {args.num_samples}")
    print(f"{'='*60}")
    
    # Create dataset
    dataset = M2Dataset(
        symbols,
        processed_dir,
        feature_profile,
        lookback=config['sample_index']['lookback'],
        split="train",
    )
    
    # Check random samples
    random_results = check_dataloader(dataset, num_samples=args.num_samples)
    
    # Check per-symbol samples
    symbol_results = check_symbol_samples(dataset, symbols, samples_per_symbol=3)
    
    # Generate reports
    print(f"\nGenerating reports...")
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    generate_reports(config, feature_profile, dataset, random_results, symbol_results, reports_dir)
    
    # Summary
    total_checks = len(random_results) + len(symbol_results)
    valid_checks = sum(1 for r in random_results + symbol_results if r["valid"])
    
    print(f"\n{'='*60}")
    print(f"✅ M2-T1 Complete")
    print(f"{'='*60}")
    print(f"Valid: {valid_checks}/{total_checks} ({valid_checks/total_checks*100:.1f}%)")
    print(f"{'='*60}")
    if valid_checks != total_checks:
        sys.exit(1)


if __name__ == "__main__":
    main()
