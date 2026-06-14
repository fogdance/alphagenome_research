#!/usr/bin/env python3
"""
M1-T2: Build continuous bars from real contracts using DB mapping.

This script:
1. Reads continuous->real mapping from fut_continuous_map_v2
2. Loads corresponding real contract bars from archive
3. Stitches them together to form continuous main contract series
4. Outputs raw_continuous_bars.parquet and audit report

Usage:
    # Single symbol
    python src/alphatrade/scripts/build_continuous_bars.py \
      --csymbol DCE.JM \
      --start 2018-01-01 \
      --end 2026-02-13

    # Batch from universe
    python src/alphatrade/scripts/build_continuous_bars.py \
      --universe configs/universe/m1_candidates.yaml \
      --start 2018-01-01 \
      --end 2026-02-13
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.db_connector import get_continuous_map
from data_pipeline.archive_reader import ArchiveReader
import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Build continuous bars from DB mapping")
    parser.add_argument("--csymbol", type=str, help="Continuous symbol (e.g., DCE.JM)")
    parser.add_argument("--universe", type=str, help="Universe YAML file with candidates")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-02-13", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--archive-dir",
        type=str,
        default="/data/juejin",
        help="Archive root directory"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/data/juejin/_manifest/3a1a389f-56e9-46ce-8a04-047cbd456a44.jsonl",
        help="Manifest file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/m1",
        help="Output directory"
    )
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def build_continuous_bars_for_symbol(
    csymbol: str,
    start_date: str,
    end_date: str,
    archive_reader: ArchiveReader,
    output_dir: str
) -> dict:
    """
    Build continuous bars for a single symbol.

    Returns:
        Audit report dict
    """
    print(f"\n{'='*60}")
    print(f"Building continuous bars for: {csymbol}")
    print(f"{'='*60}")

    # Get mapping from DB
    print(f"\n1. Loading continuous->real mapping from DB...")
    try:
        map_df = get_continuous_map(csymbol, start_date, end_date)
    except Exception as e:
        print(f"   ❌ Failed to load mapping: {e}")
        return {
            'csymbol': csymbol,
            'status': 'FAIL',
            'error': f"DB mapping failed: {e}"
        }

    if len(map_df) == 0:
        print(f"   ❌ No mapping found for {csymbol}")
        return {
            'csymbol': csymbol,
            'status': 'FAIL',
            'error': 'No mapping found in DB'
        }

    print(f"   Found {len(map_df):,} trading days")
    print(f"   Date range: {map_df['trading_date'].min()} to {map_df['trading_date'].max()}")

    # Get unique real symbols
    real_symbols = map_df['symbol'].unique()
    print(f"   Real contracts: {len(real_symbols)} ({', '.join(real_symbols[:5])}{'...' if len(real_symbols) > 5 else ''})")

    # Load bars for each real symbol
    print(f"\n2. Loading bars from archive...")
    all_bars = []
    missing_symbols = []

    for i, real_symbol in enumerate(real_symbols, 1):
        print(f"   [{i}/{len(real_symbols)}] Loading {real_symbol}...", end='')

        try:
            df = archive_reader.load_symbol(real_symbol, dedup_eob="last")
            if len(df) > 0:
                all_bars.append(df)
                print(f" OK ({len(df):,} rows)")
            else:
                missing_symbols.append(real_symbol)
                print(f" EMPTY")
        except Exception as e:
            missing_symbols.append(real_symbol)
            print(f" FAIL ({e})")

    if len(all_bars) == 0:
        print(f"   ❌ No bars loaded")
        return {
            'csymbol': csymbol,
            'status': 'FAIL',
            'error': 'No bars loaded from archive',
            'missing_symbols': missing_symbols
        }

    # Concatenate all bars
    print(f"\n3. Concatenating bars...")
    df_bars = pd.concat(all_bars, ignore_index=True)
    print(f"   Total bars: {len(df_bars):,}")

    # Add trading_date column (date part of eob)
    df_bars['trading_date'] = pd.to_datetime(df_bars['eob']).dt.date
    df_bars['trading_date'] = pd.to_datetime(df_bars['trading_date'])

    # Join with mapping to filter only mapped days
    print(f"\n4. Joining with mapping...")
    df_continuous = df_bars.merge(
        map_df[['trading_date', 'symbol']],
        on=['trading_date', 'symbol'],
        how='inner'
    )
    print(f"   Continuous bars: {len(df_continuous):,}")

    # Sort by eob
    df_continuous = df_continuous.sort_values('eob').reset_index(drop=True)

    # Calculate statistics
    print(f"\n5. Calculating statistics...")

    # Trading date coverage
    expected_dates = set(map_df['trading_date'].dt.date)
    actual_dates = set(df_continuous['trading_date'].dt.date)
    missing_dates = expected_dates - actual_dates
    coverage_rate = len(actual_dates) / len(expected_dates) if expected_dates else 0

    # Contract switches
    df_continuous['symbol_shift'] = df_continuous['symbol'].shift(1)
    switches = (df_continuous['symbol'] != df_continuous['symbol_shift']).sum() - 1  # -1 for first row

    # EOB range
    min_eob = df_continuous['eob'].min()
    max_eob = df_continuous['eob'].max()

    print(f"   Trading date coverage: {len(actual_dates)}/{len(expected_dates)} ({coverage_rate*100:.1f}%)")
    print(f"   Missing dates: {len(missing_dates)}")
    print(f"   Contract switches: {switches}")
    print(f"   EOB range: {min_eob} to {max_eob}")

    # Save output
    print(f"\n6. Saving output...")
    output_path = Path(output_dir) / csymbol / "raw_continuous_bars.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop helper columns
    df_output = df_continuous.drop(columns=['trading_date', 'symbol_shift'], errors='ignore')
    df_output.to_parquet(output_path, index=False)

    print(f"   ✅ Saved: {output_path}")
    print(f"   Rows: {len(df_output):,}")
    print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    # Return audit report
    return {
        'csymbol': csymbol,
        'status': 'SUCCESS',
        'mapping_days': len(map_df),
        'real_contracts': len(real_symbols),
        'missing_symbols': missing_symbols,
        'total_bars': len(df_bars),
        'continuous_bars': len(df_output),
        'trading_date_coverage': coverage_rate,
        'missing_dates_count': len(missing_dates),
        'contract_switches': switches,
        'min_eob': str(min_eob),
        'max_eob': str(max_eob),
        'output_path': str(output_path),
        'output_size_mb': output_path.stat().st_size / (1024*1024)
    }


def generate_report(reports: list, output_path: str):
    """Generate audit report."""
    print(f"\n{'='*60}")
    print(f"Generating audit report...")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# M1-T2 Continuous Bars Build Report\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        success_count = sum(1 for r in reports if r['status'] == 'SUCCESS')
        fail_count = len(reports) - success_count

        f.write("## 总览\n\n")
        f.write(f"- **总品种数**: {len(reports)}\n")
        f.write(f"- **成功**: {success_count}\n")
        f.write(f"- **失败**: {fail_count}\n\n")

        # Success details
        if success_count > 0:
            f.write("## 成功品种详情\n\n")
            f.write("| Symbol | Mapping Days | Real Contracts | Continuous Bars | Coverage | Missing Dates | Switches | Max EOB |\n")
            f.write("|--------|--------------|----------------|-----------------|----------|---------------|----------|----------|\n")

            for r in reports:
                if r['status'] == 'SUCCESS':
                    f.write(f"| {r['csymbol']} | {r['mapping_days']:,} | {r['real_contracts']} | "
                           f"{r['continuous_bars']:,} | {r['trading_date_coverage']*100:.1f}% | "
                           f"{r['missing_dates_count']} | {r['contract_switches']} | {r['max_eob']} |\n")

            f.write("\n")

        # Failure details
        if fail_count > 0:
            f.write("## 失败品种详情\n\n")
            for r in reports:
                if r['status'] == 'FAIL':
                    f.write(f"### {r['csymbol']}\n\n")
                    f.write(f"- **错误**: {r.get('error', 'Unknown')}\n")
                    if 'missing_symbols' in r and r['missing_symbols']:
                        f.write(f"- **缺失合约**: {', '.join(r['missing_symbols'][:10])}\n")
                    f.write("\n")

        # Stale analysis
        f.write("## Stale 分析\n\n")
        f.write("检查是否修复了 archive 中 continuous symbol 的 stale 问题：\n\n")

        global_max = "2026-02-13"
        for r in reports:
            if r['status'] == 'SUCCESS':
                max_eob = r['max_eob'][:10]  # YYYY-MM-DD
                days_behind = (pd.to_datetime(global_max) - pd.to_datetime(max_eob)).days

                if days_behind <= 30:
                    status = "✅ 已修复"
                else:
                    status = f"⚠️  仍落后 {days_behind} 天"

                f.write(f"- **{r['csymbol']}**: max_eob={max_eob}, {status}\n")

        f.write("\n")

        # Recommendations
        f.write("## 建议\n\n")
        f.write("1. 检查 coverage < 95% 的品种，确认缺失原因\n")
        f.write("2. 检查 max_eob 仍落后的品种，确认 DB mapping 是否最新\n")
        f.write("3. 对于失败品种，检查 archive 中是否缺少对应的 real 合约数据\n")

    print(f"   ✅ Report saved: {output_path}")


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
    print(f"M1-T2: Build Continuous Bars (Using DB)")
    print(f"{'='*60}")
    print(f"Symbols: {len(csymbols)}")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Archive: {args.archive_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")

    # Initialize archive reader
    reader = ArchiveReader(args.archive_dir, args.manifest)

    # Process each symbol
    reports = []
    for i, csymbol in enumerate(csymbols, 1):
        print(f"\n[{i}/{len(csymbols)}] Processing {csymbol}...")

        try:
            report = build_continuous_bars_for_symbol(
                csymbol,
                args.start,
                args.end,
                reader,
                args.output_dir
            )
            reports.append(report)
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            reports.append({
                'csymbol': csymbol,
                'status': 'FAIL',
                'error': f"Unexpected error: {e}"
            })

    # Generate report
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    report_path = reports_dir / "m1_t2_continuous_build.md"
    generate_report(reports, report_path)

    # Summary
    success_count = sum(1 for r in reports if r['status'] == 'SUCCESS')
    print(f"\n{'='*60}")
    print(f"✅ M1-T2 Complete")
    print(f"{'='*60}")
    print(f"Success: {success_count}/{len(reports)}")
    print(f"Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
