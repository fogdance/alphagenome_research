#!/usr/bin/env python3
"""
Check M1 data contract for AlphaTrade v0.2 compatibility.

This script:
1. Samples 3 symbols from m1_selected.yaml
2. Checks bars.parquet schema and features
3. Checks index_train.parquet schema
4. Validates compatibility with AlphaTrade v0.2
5. Generates contract check report
"""

import os
import sys
import yaml
import random
from pathlib import Path
from datetime import datetime
import pandas as pd
import json


def load_universe(universe_path: str) -> list:
    """Load universe symbols."""
    with open(universe_path, 'r') as f:
        universe = yaml.safe_load(f)
    return universe['candidates']


def check_bars_schema(bars_path: str) -> dict:
    """Check bars.parquet schema."""
    df = pd.read_parquet(bars_path)
    
    return {
        'path': str(bars_path),
        'rows': len(df),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'sample_head': df.head(3).to_dict('records')
    }


def check_index_schema(index_path: str) -> dict:
    """Check index_train.parquet schema."""
    df = pd.read_parquet(index_path)
    
    return {
        'path': str(index_path),
        'rows': len(df),
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'sample_head': df.head(3).to_dict('records')
    }


def validate_contract(bars_info: dict, index_info: dict) -> dict:
    """Validate data contract against AlphaTrade v0.2 spec."""
    
    # Expected features for AlphaTrade v0.2 (F=8)
    expected_features = [
        'ret_1m',
        'hl_range',
        'co_change',
        'vol_log1p',
        'pos_log1p',
        'minute_sin',
        'minute_cos',
        # 8th feature: TBD
    ]
    
    bars_cols = bars_info['columns']
    index_cols = index_info['columns']
    
    # Check bars features
    missing_features = []
    present_features = []
    
    for feat in expected_features[:7]:  # First 7 features
        if feat in bars_cols:
            present_features.append(feat)
        else:
            missing_features.append(feat)
    
    # Check for 8th feature candidates
    candidate_8th = []
    for col in bars_cols:
        if col not in expected_features[:7] and col not in ['eob', 'symbol', 'segment_id', 'open', 'high', 'low', 'close', 'volume', 'position']:
            candidate_8th.append(col)
    
    # Check index window locator fields
    window_locator_candidates = []
    for col in index_cols:
        if any(keyword in col.lower() for keyword in ['start', 'end', 'idx', 'asof', 'bar']):
            window_locator_candidates.append(col)
    
    # Determine status
    has_7_features = len(missing_features) == 0
    has_8th_feature = len(candidate_8th) > 0
    has_window_locators = 'x_start' in index_cols and 'x_end' in index_cols
    
    return {
        'expected_features': expected_features,
        'present_features': present_features,
        'missing_features': missing_features,
        'candidate_8th_features': candidate_8th,
        'has_7_features': has_7_features,
        'has_8th_feature': has_8th_feature,
        'window_locator_fields': window_locator_candidates,
        'has_window_locators': has_window_locators,
        'status': 'PASS' if has_7_features and has_window_locators else 'FAIL'
    }


def generate_report(samples: list, output_path: str):
    """Generate contract check report."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# M1-T5 Data Contract Check Report\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 目的\n\n")
        f.write("确认 M1 canonical 数据能正确喂给 AlphaTrade v0.2，验证：\n")
        f.write("1. bars.parquet 包含所需的 8 个特征\n")
        f.write("2. index_train.parquet 包含窗口定位字段\n\n")
        
        f.write("## 抽样品种\n\n")
        for sample in samples:
            f.write(f"- {sample['symbol']}\n")
        f.write("\n")
        
        # Check each sample
        for i, sample in enumerate(samples, 1):
            f.write(f"## 品种 {i}: {sample['symbol']}\n\n")
            
            # Bars schema
            f.write("### bars.parquet Schema\n\n")
            bars_info = sample['bars_info']
            f.write(f"- **Path**: `{bars_info['path']}`\n")
            f.write(f"- **Rows**: {bars_info['rows']:,}\n")
            f.write(f"- **Columns** ({len(bars_info['columns'])}):\n")
            for col in bars_info['columns']:
                dtype = bars_info['dtypes'][col]
                f.write(f"  - `{col}`: {dtype}\n")
            f.write("\n")
            
            # Index schema
            f.write("### index_train.parquet Schema\n\n")
            index_info = sample['index_info']
            f.write(f"- **Path**: `{index_info['path']}`\n")
            f.write(f"- **Rows**: {index_info['rows']:,}\n")
            f.write(f"- **Columns** ({len(index_info['columns'])}):\n")
            for col in index_info['columns']:
                dtype = index_info['dtypes'][col]
                f.write(f"  - `{col}`: {dtype}\n")
            f.write("\n")
            
            # Validation
            f.write("### 契约验证\n\n")
            validation = sample['validation']
            
            f.write("#### 特征检查 (前 7 个)\n\n")
            if validation['has_7_features']:
                f.write("✅ **所有前 7 个特征都存在**\n\n")
                for feat in validation['present_features']:
                    f.write(f"- ✅ `{feat}`\n")
            else:
                f.write("❌ **缺少部分特征**\n\n")
                for feat in validation['missing_features']:
                    f.write(f"- ❌ `{feat}` (缺失)\n")
            f.write("\n")
            
            f.write("#### 第 8 个特征检查\n\n")
            if validation['has_8th_feature']:
                f.write(f"⚠️  **候选第 8 个特征**: {', '.join([f'`{c}`' for c in validation['candidate_8th_features']])}\n\n")
            else:
                f.write("❌ **缺少第 8 个特征**\n\n")
            
            f.write("#### 窗口定位字段检查\n\n")
            if validation['has_window_locators']:
                f.write("✅ **窗口定位字段存在**: `x_start`, `x_end`\n\n")
            else:
                f.write(f"⚠️  **候选窗口定位字段**: {', '.join([f'`{c}`' for c in validation['window_locator_fields']])}\n\n")
            
            f.write(f"#### 总体状态: **{validation['status']}**\n\n")
        
        # Summary
        f.write("## 总结\n\n")
        
        all_pass = all(s['validation']['status'] == 'PASS' for s in samples)
        all_have_7 = all(s['validation']['has_7_features'] for s in samples)
        all_have_8 = all(s['validation']['has_8th_feature'] for s in samples)
        all_have_locators = all(s['validation']['has_window_locators'] for s in samples)
        
        f.write(f"- **前 7 个特征**: {'✅ 全部存在' if all_have_7 else '❌ 部分缺失'}\n")
        f.write(f"- **第 8 个特征**: {'✅ 存在候选' if all_have_8 else '❌ 缺失'}\n")
        f.write(f"- **窗口定位字段**: {'✅ 存在' if all_have_locators else '❌ 缺失'}\n")
        f.write(f"- **总体状态**: {'✅ PASS' if all_pass else '❌ FAIL'}\n\n")
        
        # Final feature list
        f.write("## 最终特征列表 (喂给模型)\n\n")
        
        if all_have_7:
            f.write("### 方案 A: 补充 is_session_open (推荐)\n\n")
            f.write("```python\n")
            f.write("feature_cols = [\n")
            for feat in validation['expected_features'][:7]:
                f.write(f"    '{feat}',\n")
            f.write("    'is_session_open',  # 补充：全 1 或基于 segment_id\n")
            f.write("]\n")
            f.write("```\n\n")
            
            f.write("### 方案 B: Dataloader 拼接常数 1.0\n\n")
            f.write("```python\n")
            f.write("feature_cols = [\n")
            for feat in validation['expected_features'][:7]:
                f.write(f"    '{feat}',\n")
            f.write("]\n")
            f.write("# 在 dataloader 中拼接一维常数 1.0\n")
            f.write("```\n\n")
        
        # Window locator fields
        f.write("## 窗口定位字段\n\n")
        if all_have_locators:
            f.write("```python\n")
            f.write("# index_train.parquet 中的窗口定位字段\n")
            f.write("window_start_field = 'x_start'\n")
            f.write("window_end_field = 'x_end'\n")
            f.write("```\n\n")
        
        # Recommendations
        f.write("## 建议\n\n")
        if not all_have_8:
            f.write("### 补充第 8 个特征\n\n")
            f.write("**推荐方案 A**: 在 canonical bars 中补充 `is_session_open` 列\n\n")
            f.write("```python\n")
            f.write("# 在 build_m1_canonical_bars.py 中添加\n")
            f.write("df['is_session_open'] = 1.0  # 简单版本：全 1\n")
            f.write("# 或基于 segment_id 的首 bar 标记\n")
            f.write("```\n\n")
    
    print(f"✅ Report saved: {output_path}")


def generate_json_report(samples: list, output_path: str):
    """Generate JSON report for machine reading."""
    
    # Simplify for JSON
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'samples': []
    }
    
    for sample in samples:
        json_data['samples'].append({
            'symbol': sample['symbol'],
            'bars_columns': sample['bars_info']['columns'],
            'bars_rows': sample['bars_info']['rows'],
            'index_columns': sample['index_info']['columns'],
            'index_rows': sample['index_info']['rows'],
            'validation': sample['validation']
        })
    
    # Summary
    all_pass = all(s['validation']['status'] == 'PASS' for s in samples)
    json_data['summary'] = {
        'status': 'PASS' if all_pass else 'FAIL',
        'has_7_features': all(s['validation']['has_7_features'] for s in samples),
        'has_8th_feature': all(s['validation']['has_8th_feature'] for s in samples),
        'has_window_locators': all(s['validation']['has_window_locators'] for s in samples)
    }
    
    # Final feature list
    if samples:
        validation = samples[0]['validation']
        json_data['final_feature_cols'] = validation['expected_features'][:7] + ['is_session_open']
        json_data['window_locator_fields'] = {
            'start': 'x_start',
            'end': 'x_end'
        }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON report saved: {output_path}")


def main():
    print("="*60)
    print("M1-T5.0: Data Contract Check")
    print("="*60)
    
    # Load universe
    universe_path = "configs/universe/m1_selected.yaml"
    if not os.path.exists(universe_path):
        print(f"Error: Universe file not found: {universe_path}")
        sys.exit(1)
    
    symbols = load_universe(universe_path)
    print(f"\nLoaded {len(symbols)} symbols from universe")
    
    # Sample 3 symbols
    sample_symbols = random.sample(symbols, min(3, len(symbols)))
    print(f"Sampled symbols: {', '.join(sample_symbols)}")
    
    # Check each sample
    samples = []
    processed_dir = "data/processed/m1"
    
    for symbol in sample_symbols:
        print(f"\nChecking {symbol}...")
        
        symbol_dir = Path(processed_dir) / symbol
        bars_path = symbol_dir / "bars.parquet"
        index_path = symbol_dir / "index_train.parquet"
        
        if not bars_path.exists():
            print(f"  Warning: bars.parquet not found")
            continue
        
        if not index_path.exists():
            print(f"  Warning: index_train.parquet not found")
            continue
        
        # Check schemas
        bars_info = check_bars_schema(bars_path)
        index_info = check_index_schema(index_path)
        
        print(f"  bars.parquet: {bars_info['rows']:,} rows, {len(bars_info['columns'])} columns")
        print(f"  index_train.parquet: {index_info['rows']:,} rows, {len(index_info['columns'])} columns")
        
        # Validate
        validation = validate_contract(bars_info, index_info)
        print(f"  Validation: {validation['status']}")
        
        samples.append({
            'symbol': symbol,
            'bars_info': bars_info,
            'index_info': index_info,
            'validation': validation
        })
    
    if not samples:
        print("\nError: No valid samples found")
        sys.exit(1)
    
    # Generate reports
    print("\nGenerating reports...")
    generate_report(samples, "reports/m1_t5_contract_check.md")
    generate_json_report(samples, "reports/m1_t5_contract_check.json")
    
    # Summary
    all_pass = all(s['validation']['status'] == 'PASS' for s in samples)
    print(f"\n{'='*60}")
    print(f"Contract Check: {'✅ PASS' if all_pass else '❌ FAIL'}")
    print(f"{'='*60}")
    print(f"Reports:")
    print(f"  - reports/m1_t5_contract_check.md")
    print(f"  - reports/m1_t5_contract_check.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
