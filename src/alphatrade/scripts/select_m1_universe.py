#!/usr/bin/env python3
"""
Select final M1 universe based on data quality.

This script:
1. Reads M1-T2 continuous bars build report
2. Reads M1-T3 sample index report
3. Applies quality filters
4. Generates final universe YAML
5. Generates selection report

Usage:
    python src/alphatrade/scripts/select_m1_universe.py
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Select final M1 universe")
    parser.add_argument("--output-yaml", type=str, default="configs/universe/m1_selected.yaml")
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def parse_t2_report(report_path: str) -> dict:
    """Parse M1-T2 continuous bars report."""
    print(f"\n1. Parsing M1-T2 report: {report_path}")
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the success table
    lines = content.split('\n')
    table_started = False
    data = []
    
    for line in lines:
        if '| Symbol |' in line and 'Mapping Days' in line:
            table_started = True
            continue
        if table_started and line.startswith('|') and 'Symbol' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 8:
                symbol = parts[0]
                mapping_days = int(parts[1].replace(',', ''))
                continuous_bars = int(parts[3].replace(',', ''))
                coverage = float(parts[4].replace('%', ''))
                data.append({
                    'symbol': symbol,
                    'mapping_days': mapping_days,
                    'continuous_bars': continuous_bars,
                    'coverage': coverage
                })
    
    result = {row['symbol']: row for row in data}
    print(f"   Found {len(result)} symbols")
    return result


def parse_t3_report(report_path: str) -> dict:
    """Parse M1-T3 sample index report."""
    print(f"\n2. Parsing M1-T3 report: {report_path}")
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse symbol sections
    lines = content.split('\n')
    data = {}
    current_symbol = None
    
    for line in lines:
        if line.startswith('## Symbol:'):
            current_symbol = line.split('Symbol:')[1].strip()
            data[current_symbol] = {}
        elif current_symbol and '**Train samples**:' in line:
            train = int(line.split(':')[1].strip().replace(',', ''))
            data[current_symbol]['train_samples'] = train
        elif current_symbol and '**Val samples**:' in line:
            val = int(line.split(':')[1].strip().replace(',', ''))
            data[current_symbol]['val_samples'] = val
        elif current_symbol and '**Test samples**:' in line:
            test = int(line.split(':')[1].strip().replace(',', ''))
            data[current_symbol]['test_samples'] = test
        elif current_symbol and '**Total samples**:' in line:
            total = int(line.split(':')[1].strip().replace(',', ''))
            data[current_symbol]['total_samples'] = total
    
    print(f"   Found {len(data)} symbols")
    return data


def apply_filters(t2_data: dict, t3_data: dict, filters: dict) -> dict:
    """Apply quality filters to select final universe."""
    print(f"\n3. Applying filters:")
    print(f"   - Coverage >= {filters['min_coverage']}%")
    print(f"   - Train samples >= {filters['min_train_samples']:,}")
    print(f"   - Val samples >= {filters['min_val_samples']:,}")
    print(f"   - Test samples >= {filters['min_test_samples']:,}")
    
    selected = {}
    rejected = {}
    
    for symbol in t2_data.keys():
        t2_info = t2_data[symbol]
        t3_info = t3_data.get(symbol, {})
        
        reasons = []
        
        # Check coverage
        if t2_info['coverage'] < filters['min_coverage']:
            reasons.append(f"coverage={t2_info['coverage']:.1f}% < {filters['min_coverage']}%")
        
        # Check train samples
        train_samples = t3_info.get('train_samples', 0)
        if train_samples < filters['min_train_samples']:
            reasons.append(f"train_samples={train_samples:,} < {filters['min_train_samples']:,}")
        
        # Check val samples
        val_samples = t3_info.get('val_samples', 0)
        if val_samples < filters['min_val_samples']:
            reasons.append(f"val_samples={val_samples:,} < {filters['min_val_samples']:,}")
        
        # Check test samples
        test_samples = t3_info.get('test_samples', 0)
        if test_samples < filters['min_test_samples']:
            reasons.append(f"test_samples={test_samples:,} < {filters['min_test_samples']:,}")
        
        if reasons:
            rejected[symbol] = {
                **t2_info,
                **t3_info,
                'reject_reasons': reasons
            }
        else:
            selected[symbol] = {
                **t2_info,
                **t3_info
            }
    
    print(f"   Selected: {len(selected)}")
    print(f"   Rejected: {len(rejected)}")
    
    return selected, rejected


def generate_universe_yaml(selected: dict, output_path: str):
    """Generate final universe YAML file."""
    print(f"\n4. Generating universe YAML: {output_path}")
    
    universe = {
        'version': 'm1_selected',
        'description': 'M1 final selected universe based on data quality',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'selection_criteria': {
            'min_coverage': 95.0,
            'min_train_samples': 5000,
            'min_val_samples': 500,
            'min_test_samples': 1000
        },
        'candidates': sorted(selected.keys())
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(universe, f, default_flow_style=False, allow_unicode=True)
    
    print(f"   Saved {len(universe['candidates'])} symbols")


def generate_selection_report(selected: dict, rejected: dict, output_path: str):
    """Generate selection report."""
    print(f"\n5. Generating selection report: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# M1-T4 Universe Selection Report\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        f.write("## 总览\n\n")
        f.write(f"- **候选品种**: {len(selected) + len(rejected)}\n")
        f.write(f"- **选中品种**: {len(selected)}\n")
        f.write(f"- **拒绝品种**: {len(rejected)}\n")
        f.write(f"- **选中率**: {len(selected)/(len(selected)+len(rejected))*100:.1f}%\n\n")
        
        # Selection criteria
        f.write("## 筛选标准\n\n")
        f.write("- **Coverage**: >= 95%\n")
        f.write("- **Train samples**: >= 5,000\n")
        f.write("- **Val samples**: >= 500\n")
        f.write("- **Test samples**: >= 1,000\n\n")
        
        # Selected symbols table
        f.write("## 选中品种详情\n\n")
        f.write("| Symbol | Coverage | Mapping Days | Train | Val | Test | Total |\n")
        f.write("|--------|----------|--------------|-------|-----|------|-------|\n")
        
        for symbol in sorted(selected.keys()):
            info = selected[symbol]
            f.write(f"| {symbol} | {info['coverage']:.1f}% | {info['mapping_days']:,} | "
                   f"{info.get('train_samples', 0):,} | {info.get('val_samples', 0):,} | "
                   f"{info.get('test_samples', 0):,} | {info.get('total_samples', 0):,} |\n")
        
        f.write("\n")
        
        # Statistics
        if selected:
            f.write("### 统计摘要\n\n")
            
            total_train = sum(info.get('train_samples', 0) for info in selected.values())
            total_val = sum(info.get('val_samples', 0) for info in selected.values())
            total_test = sum(info.get('test_samples', 0) for info in selected.values())
            total_samples = sum(info.get('total_samples', 0) for info in selected.values())
            
            avg_coverage = sum(info['coverage'] for info in selected.values()) / len(selected)
            avg_train = total_train / len(selected)
            avg_val = total_val / len(selected)
            avg_test = total_test / len(selected)
            
            f.write(f"- **总 Train samples**: {total_train:,}\n")
            f.write(f"- **总 Val samples**: {total_val:,}\n")
            f.write(f"- **总 Test samples**: {total_test:,}\n")
            f.write(f"- **总 samples**: {total_samples:,}\n\n")
            
            f.write(f"- **平均 Coverage**: {avg_coverage:.1f}%\n")
            f.write(f"- **平均 Train samples**: {avg_train:,.0f}\n")
            f.write(f"- **平均 Val samples**: {avg_val:,.0f}\n")
            f.write(f"- **平均 Test samples**: {avg_test:,.0f}\n\n")
        
        # Rejected symbols
        if rejected:
            f.write("## 拒绝品种详情\n\n")
            f.write("| Symbol | Coverage | Train | Val | Test | Reject Reasons |\n")
            f.write("|--------|----------|-------|-----|------|----------------|\n")
            
            for symbol in sorted(rejected.keys()):
                info = rejected[symbol]
                reasons = '; '.join(info['reject_reasons'])
                f.write(f"| {symbol} | {info['coverage']:.1f}% | "
                       f"{info.get('train_samples', 0):,} | {info.get('val_samples', 0):,} | "
                       f"{info.get('test_samples', 0):,} | {reasons} |\n")
            
            f.write("\n")
        
        # Exchange distribution
        f.write("## 交易所分布\n\n")
        exchange_counts = {}
        for symbol in selected.keys():
            exchange = symbol.split('.')[0]
            exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1
        
        for exchange in sorted(exchange_counts.keys()):
            f.write(f"- **{exchange}**: {exchange_counts[exchange]} 个\n")
        
        f.write("\n")
    
    print(f"   Report saved")


def main():
    args = parse_args()

    print("="*60)
    print("M1-T4: Universe Selection")
    print("="*60)

    # Paths
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    t2_report = reports_dir / "m1_t2_continuous_build.md"
    t3_report = reports_dir / "m1_t3_sample_index.md"
    output_yaml = args.output_yaml
    output_report = reports_dir / "m1_t4_universe_selection.md"
    
    # Check inputs
    if not os.path.exists(t2_report):
        print(f"Error: M1-T2 report not found: {t2_report}")
        sys.exit(1)
    
    if not os.path.exists(t3_report):
        print(f"Error: M1-T3 report not found: {t3_report}")
        sys.exit(1)
    
    # Parse reports
    t2_data = parse_t2_report(t2_report)
    t3_data = parse_t3_report(t3_report)
    
    # Apply filters
    filters = {
        'min_coverage': 95.0,
        'min_train_samples': 5000,
        'min_val_samples': 500,
        'min_test_samples': 1000
    }
    
    selected, rejected = apply_filters(t2_data, t3_data, filters)
    
    # Generate outputs
    generate_universe_yaml(selected, output_yaml)
    generate_selection_report(selected, rejected, output_report)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ M1-T4 Complete")
    print(f"{'='*60}")
    print(f"Selected: {len(selected)}/{len(t2_data)} symbols")
    print(f"Universe: {output_yaml}")
    print(f"Report: {output_report}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
