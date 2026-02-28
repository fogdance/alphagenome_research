#!/usr/bin/env python3
"""
M1 Pipeline Runner - Batch process all symbols from config

Usage:
    python src/alphatrade/scripts/run_m1_pipeline.py --config src/alphatrade/configs/dataset/m1.yaml
    python src/alphatrade/scripts/run_m1_pipeline.py --config src/alphatrade/configs/dataset/m1.yaml --stage canonical
    python src/alphatrade/scripts/run_m1_pipeline.py --config src/alphatrade/configs/dataset/m1.yaml --stage index
    python src/alphatrade/scripts/run_m1_pipeline.py --config src/alphatrade/configs/dataset/m1.yaml --stage all
"""

import argparse
import sys
import yaml
import subprocess
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="M1 Pipeline Runner")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["canonical", "index", "all"],
        help="Which stage to run"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Specific symbols to process (default: all from config)"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other symbols if one fails"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_canonical_bars(config_path: str, symbol: str) -> bool:
    """Run build_canonical_bars for a single symbol."""
    cmd = [
        "python",
        "src/alphatrade/scripts/build_canonical_bars.py",
        "--config", config_path,
        "--symbol", symbol
    ]
    
    print(f"\n{'='*60}")
    print(f"Building canonical bars for: {symbol}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_sample_index(config_path: str, symbol: str) -> bool:
    """Run build_sample_index for a single symbol."""
    cmd = [
        "python",
        "src/alphatrade/scripts/build_sample_index.py",
        "--config", config_path,
        "--symbol", symbol
    ]
    
    print(f"\n{'='*60}")
    print(f"Building sample index for: {symbol}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get symbols list
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = config.get("symbol_universe", {}).get("symbols", [])
    
    if not symbols:
        print("Error: No symbols found in config or command line")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"M1 Pipeline Runner")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Stage: {args.stage}")
    print(f"Symbols: {len(symbols)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Track results
    results = {
        "canonical": {"success": [], "failed": []},
        "index": {"success": [], "failed": []}
    }
    
    # Process each symbol
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing: {symbol}")
        
        # Stage 1: Canonical bars
        if args.stage in ["canonical", "all"]:
            success = run_canonical_bars(args.config, symbol)
            if success:
                results["canonical"]["success"].append(symbol)
            else:
                results["canonical"]["failed"].append(symbol)
                if not args.continue_on_error:
                    print(f"\n❌ Failed on {symbol} (canonical). Stopping.")
                    break
                else:
                    print(f"\n⚠️  Failed on {symbol} (canonical). Continuing...")
                    continue
        
        # Stage 2: Sample index
        if args.stage in ["index", "all"]:
            success = run_sample_index(args.config, symbol)
            if success:
                results["index"]["success"].append(symbol)
            else:
                results["index"]["failed"].append(symbol)
                if not args.continue_on_error:
                    print(f"\n❌ Failed on {symbol} (index). Stopping.")
                    break
                else:
                    print(f"\n⚠️  Failed on {symbol} (index). Continuing...")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Pipeline Summary")
    print(f"{'='*60}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.stage in ["canonical", "all"]:
        print(f"Canonical Bars:")
        print(f"  ✅ Success: {len(results['canonical']['success'])}")
        print(f"  ❌ Failed:  {len(results['canonical']['failed'])}")
        if results['canonical']['failed']:
            print(f"  Failed symbols: {', '.join(results['canonical']['failed'])}")
        print()
    
    if args.stage in ["index", "all"]:
        print(f"Sample Index:")
        print(f"  ✅ Success: {len(results['index']['success'])}")
        print(f"  ❌ Failed:  {len(results['index']['failed'])}")
        if results['index']['failed']:
            print(f"  Failed symbols: {', '.join(results['index']['failed'])}")
        print()
    
    # Exit code
    total_failed = len(results['canonical']['failed']) + len(results['index']['failed'])
    if total_failed > 0:
        print(f"⚠️  Pipeline completed with {total_failed} failures")
        sys.exit(1)
    else:
        print(f"✅ Pipeline completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
