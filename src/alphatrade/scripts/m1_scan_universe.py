#!/usr/bin/env python3
"""
M1-T0: Universe Scanner - Scan archive and generate training candidates

This script:
1. Parses manifest to get all symbols and coverage stats
2. Optionally samples parquet files for quality/liquidity metrics
3. Scores and ranks symbols for training suitability
4. Outputs reports and candidate lists

Usage:
    # Fast mode (manifest only)
    python src/alphatrade/scripts/m1_scan_universe.py \
      --archive-dir /data/juejin \
      --manifest /data/juejin/_manifest/<run_id>.jsonl \
      --fast

    # Full mode (with parquet sampling)
    python src/alphatrade/scripts/m1_scan_universe.py \
      --archive-dir /data/juejin \
      --manifest /data/juejin/_manifest/<run_id>.jsonl \
      --sample-months-per-symbol 6 \
      --top-k 50
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="M1-T0 Universe Scanner")
    parser.add_argument("--archive-dir", type=str, required=True, help="Archive root directory")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest JSONL file")
    parser.add_argument(
        "--rules-config",
        type=str,
        default="src/alphatrade/configs/universe/m1_scan_rules.yaml",
        help="Scan rules config"
    )
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode (manifest only, no parquet)")
    parser.add_argument("--sample-months-per-symbol", type=int, default=6, help="Months to sample per symbol")
    parser.add_argument("--top-k", type=int, default=50, help="Top K candidates to output")
    parser.add_argument("--exchanges", type=str, default="SHFE,DCE,CZCE,CFFEX,INE,GFEX", help="Allowed exchanges")
    return parser.parse_args()


def load_rules(rules_path: str) -> dict:
    """Load scan rules from YAML."""
    with open(rules_path, 'r') as f:
        return yaml.safe_load(f)


def infer_symbol_type(symbol: str, rules: dict) -> str:
    """Infer symbol type based on rules."""
    if '.' not in symbol:
        return rules['symbol_type_rules']['default_type']
    
    exchange, sec_id = symbol.split('.', 1)
    
    for rule in rules['symbol_type_rules']['rules']:
        if not rule.get('enabled', True):
            continue
        
        regex_pattern = rule['regex']['sec_id']
        if re.match(regex_pattern, sec_id):
            return rule['type']
    
    return rules['symbol_type_rules']['default_type']


def parse_manifest(manifest_path: str, rules: dict) -> Tuple[pd.DataFrame, dict]:
    """Parse manifest and generate symbol statistics."""
    print(f"\n1. Parsing manifest: {manifest_path}")
    
    records = []
    with open(manifest_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    
    print(f"   Found {len(records)} manifest entries")
    
    # Group by symbol
    symbol_data = defaultdict(list)
    for rec in records:
        symbol = rec['symbol']
        symbol_data[symbol].append(rec)
    
    print(f"   Found {len(symbol_data)} unique symbols")
    
    # Compute statistics per symbol
    stats_list = []
    global_min_eob = None
    global_max_eob = None
    
    allowed_exchanges = set(rules['scope']['allowed_exchanges'])
    
    for symbol, recs in symbol_data.items():
        # Extract exchange
        if '.' in symbol:
            exchange = symbol.split('.')[0].upper()
        else:
            exchange = recs[0].get('exchange', 'UNKNOWN').upper()
        
        # Filter by exchange
        if rules['scope']['drop_unknown_exchange'] and exchange not in allowed_exchanges:
            continue
        
        # Infer symbol type
        symbol_type = infer_symbol_type(symbol, rules)
        
        # Coverage stats
        months = len(recs)
        rows_total = sum(r.get('rows', 0) for r in recs)
        
        # Time range
        eobs = []
        for r in recs:
            if 'min_eob' in r:
                eobs.append(pd.to_datetime(r['min_eob']))
            if 'max_eob' in r:
                eobs.append(pd.to_datetime(r['max_eob']))
        
        if eobs:
            min_eob = min(eobs)
            max_eob = max(eobs)
            
            if global_min_eob is None or min_eob < global_min_eob:
                global_min_eob = min_eob
            if global_max_eob is None or max_eob > global_max_eob:
                global_max_eob = max_eob
        else:
            min_eob = None
            max_eob = None
        
        # Year-month coverage
        year_months = sorted(set((r['year'], r['month']) for r in recs))
        first_ym = year_months[0] if year_months else (None, None)
        last_ym = year_months[-1] if year_months else (None, None)
        
        # Missing months
        missing_months = 0
        if len(year_months) > 1:
            start_y, start_m = year_months[0]
            end_y, end_m = year_months[-1]
            expected_months = (end_y - start_y) * 12 + (end_m - start_m) + 1
            missing_months = expected_months - len(year_months)
        
        # Path validation
        invalid_paths = 0
        for r in recs:
            path = r.get('path', '')
            if path:
                path = path.replace('\\', '/')
                full_path = os.path.join(args.archive_dir, path) if not os.path.isabs(path) else path
                if not os.path.exists(full_path):
                    invalid_paths += 1
        
        stats_list.append({
            'symbol': symbol,
            'exchange': exchange,
            'symbol_type': symbol_type,
            'months': months,
            'rows_total': rows_total,
            'min_eob': min_eob,
            'max_eob': max_eob,
            'first_year': first_ym[0],
            'first_month': first_ym[1],
            'last_year': last_ym[0],
            'last_month': last_ym[1],
            'missing_months_count': missing_months,
            'invalid_paths': invalid_paths,
        })
    
    df = pd.DataFrame(stats_list)
    
    global_stats = {
        'total_symbols': len(df),
        'global_min_eob': global_min_eob.isoformat() if global_min_eob else None,
        'global_max_eob': global_max_eob.isoformat() if global_max_eob else None,
        'by_exchange': df['exchange'].value_counts().to_dict(),
        'by_symbol_type': df['symbol_type'].value_counts().to_dict(),
    }
    
    print(f"   Filtered to {len(df)} symbols")
    print(f"   Global time range: {global_stats['global_min_eob']} to {global_stats['global_max_eob']}")
    
    return df, global_stats


if __name__ == "__main__":
    args = parse_args()
    
    print("="*60)
    print("M1-T0 Universe Scanner")
    print("="*60)
    print(f"Archive: {args.archive_dir}")
    print(f"Manifest: {args.manifest}")
    print(f"Mode: {'FAST (manifest only)' if args.fast else 'FULL (with parquet sampling)'}")
    print("="*60)
    
    # Load rules
    rules = load_rules(args.rules_config)
    print(f"Loaded rules: {rules['version']}")
    
    # Parse manifest
    df, global_stats = parse_manifest(args.manifest, rules)
    
    print(f"\n✅ Manifest parsing complete")
    print(f"   Total symbols: {len(df)}")
    print(f"   By exchange: {global_stats['by_exchange']}")
    print(f"   By type: {global_stats['by_symbol_type']}")

def sample_parquet_files(df: pd.DataFrame, manifest_path: str, archive_dir: str, rules: dict, sample_months: int) -> pd.DataFrame:
    """Sample parquet files for quality and liquidity metrics."""
    print(f"\n2. Sampling parquet files ({sample_months} months per symbol)...")
    
    # Load manifest records
    manifest_records = []
    with open(manifest_path, 'r') as f:
        for line in f:
            manifest_records.append(json.loads(line))
    
    # Group by symbol
    symbol_files = defaultdict(list)
    for rec in manifest_records:
        symbol_files[rec['symbol']].append(rec)
    
    # Sample and analyze
    results = []
    total_symbols = len(df)
    
    for idx, row in df.iterrows():
        symbol = row['symbol']
        print(f"   [{idx+1}/{total_symbols}] Sampling {symbol}...", end='')
        
        files = symbol_files.get(symbol, [])
        if not files:
            print(" No files")
            results.append({
                'symbol': symbol,
                'sampled_files': 0,
                'sample_read_fail_count': 0,
            })
            continue
        
        # Sort by year-month
        files = sorted(files, key=lambda x: (x['year'], x['month']))
        
        # Select sample positions
        n_files = len(files)
        sample_indices = set()
        
        # First, last, middle
        if n_files > 0:
            sample_indices.add(0)  # first
        if n_files > 1:
            sample_indices.add(n_files - 1)  # last
        if n_files > 2:
            sample_indices.add(n_files // 2)  # middle
        
        # Random fill
        if len(sample_indices) < sample_months and n_files > len(sample_indices):
            np.random.seed(rules['sampling']['random_seed'])
            remaining = set(range(n_files)) - sample_indices
            n_to_add = min(sample_months - len(sample_indices), len(remaining))
            sample_indices.update(np.random.choice(list(remaining), n_to_add, replace=False))
        
        sample_indices = sorted(sample_indices)
        
        # Read sampled files
        file_metrics = []
        read_fail_count = 0
        
        for i in sample_indices:
            rec = files[i]
            path = rec.get('path', '').replace('\\', '/')
            full_path = os.path.join(archive_dir, path) if not os.path.isabs(path) else path
            
            try:
                # Read only necessary columns
                cols = rules['sampling']['parquet_columns']
                df_sample = pd.read_parquet(full_path, columns=cols)
                
                # Compute metrics
                rows_read = len(df_sample)
                
                # EOB checks
                df_sample['eob'] = pd.to_datetime(df_sample['eob'])
                eob_monotonic = df_sample['eob'].is_monotonic_increasing
                duplicate_eob_count = df_sample['eob'].duplicated().sum()
                
                # NaN checks
                ohlc_cols = ['open', 'high', 'low', 'close']
                nan_count_ohlc = df_sample[ohlc_cols].isna().any(axis=1).sum()
                nan_rate_ohlc = nan_count_ohlc / rows_read if rows_read > 0 else 0
                
                # Liquidity checks
                volume_nonzero_ratio = (df_sample['volume'] > 0).mean() if 'volume' in df_sample else 0
                position_nonzero_ratio = (df_sample['position'] > 0).mean() if 'position' in df_sample else 0
                
                volume_p50 = df_sample['volume'].median() if 'volume' in df_sample else 0
                position_p50 = df_sample['position'].median() if 'position' in df_sample else 0
                
                file_metrics.append({
                    'rows_read': rows_read,
                    'eob_monotonic': eob_monotonic,
                    'duplicate_eob_count': duplicate_eob_count,
                    'dup_eob_ratio': duplicate_eob_count / rows_read if rows_read > 0 else 0,
                    'nan_rate_ohlc': nan_rate_ohlc,
                    'volume_nonzero_ratio': volume_nonzero_ratio,
                    'position_nonzero_ratio': position_nonzero_ratio,
                    'volume_p50': volume_p50,
                    'position_p50': position_p50,
                })
            except Exception as e:
                read_fail_count += 1
        
        # Aggregate metrics
        if file_metrics:
            volume_nonzero_ratios = [m['volume_nonzero_ratio'] for m in file_metrics]
            position_nonzero_ratios = [m['position_nonzero_ratio'] for m in file_metrics]
            volume_p50s = [m['volume_p50'] for m in file_metrics]
            position_p50s = [m['position_p50'] for m in file_metrics]
            dup_ratios = [m['dup_eob_ratio'] for m in file_metrics]
            nan_rates = [m['nan_rate_ohlc'] for m in file_metrics]
            
            results.append({
                'symbol': symbol,
                'sampled_files': len(sample_indices),
                'sample_read_fail_count': read_fail_count,
                'sample_read_fail_ratio': read_fail_count / len(sample_indices) if sample_indices else 0,
                'volume_nonzero_ratio_med': np.median(volume_nonzero_ratios),
                'position_nonzero_ratio_med': np.median(position_nonzero_ratios),
                'volume_p50_med': np.median(volume_p50s),
                'position_p50_med': np.median(position_p50s),
                'dup_eob_ratio_max': max(dup_ratios) if dup_ratios else 0,
                'nan_rate_ohlc_max': max(nan_rates) if nan_rates else 0,
                'quality_ok_count': sum(1 for m in file_metrics if m['eob_monotonic'] and m['dup_eob_ratio'] < 0.01),
                'quality_ok_rate': sum(1 for m in file_metrics if m['eob_monotonic'] and m['dup_eob_ratio'] < 0.01) / len(file_metrics) if file_metrics else 0,
            })
            print(f" OK ({len(file_metrics)} files)")
        else:
            results.append({
                'symbol': symbol,
                'sampled_files': len(sample_indices),
                'sample_read_fail_count': read_fail_count,
                'sample_read_fail_ratio': 1.0,
            })
            print(f" FAIL (all {read_fail_count} files failed)")
    
    # Merge with original df
    df_metrics = pd.DataFrame(results)
    df = df.merge(df_metrics, on='symbol', how='left')
    
    print(f"\n✅ Parquet sampling complete")
    return df


def score_symbols(df: pd.DataFrame, global_stats: dict, rules: dict, fast_mode: bool) -> pd.DataFrame:
    """Score and rank symbols for training suitability."""
    print(f"\n3. Scoring symbols...")
    
    scores = []
    tags_list = []
    
    global_max_eob = pd.to_datetime(global_stats['global_max_eob']) if global_stats['global_max_eob'] else None
    
    for idx, row in df.iterrows():
        symbol = row['symbol']
        tags = []
        
        # Hard filters (FAIL)
        fail = False
        
        # Coverage too short
        if row['months'] < rules['metrics']['coverage']['min_months']:
            tags.append('short_coverage')
            fail = True
        
        # Too many missing months
        if row['missing_months_count'] > rules['metrics']['coverage']['max_missing_months']:
            tags.append('missing_months')
        
        # Stale data
        if global_max_eob and row['max_eob']:
            staleness_days = (global_max_eob - row['max_eob']).days
            max_staleness = rules['metrics']['coverage'].get('max_staleness_days_from_global_max')
            if max_staleness and staleness_days > max_staleness:
                tags.append('stale')
        else:
            staleness_days = None
        
        # Parquet sampling failures (if not fast mode)
        if not fast_mode:
            if row.get('sample_read_fail_ratio', 0) > rules['filtering']['max_sample_read_fail_ratio']:
                tags.append('parquet_read_flaky')
                fail = True
            
            # Low liquidity
            vol_ratio = row.get('volume_nonzero_ratio_med', 0)
            pos_ratio = row.get('position_nonzero_ratio_med', 0)
            
            if vol_ratio < rules['metrics']['liquidity']['min_volume_nonzero_ratio_med'] and \
               pos_ratio < rules['metrics']['liquidity']['min_position_nonzero_ratio_med']:
                tags.append('low_liquidity')
                fail = True
            
            # Quality issues
            if row.get('dup_eob_ratio_max', 0) > rules['metrics']['file_health']['max_duplicate_eob_ratio']:
                tags.append('bad_eob_duplicates')
            
            if row.get('nan_rate_ohlc_max', 0) > rules['metrics']['file_health']['max_nan_rate_ohlc']:
                tags.append('nan_ohlc')
        
        if fail:
            scores.append(0)
            tags_list.append(tags)
            continue
        
        # Compute score (0-100)
        score = 0.0
        weights = rules['scoring']['weights']
        
        # Coverage score
        months = row['months']
        months_min = rules['scoring']['coverage_score']['months_cap_min']
        months_max = rules['scoring']['coverage_score']['months_cap_max']
        coverage_subscore = (months - months_min) / (months_max - months_min)
        coverage_subscore = max(0, min(1, coverage_subscore))
        
        # Penalize missing months
        missing_penalty = row['missing_months_count'] * rules['scoring']['coverage_score']['missing_month_penalty_per_month']
        coverage_subscore = max(0, coverage_subscore - missing_penalty)
        
        score += coverage_subscore * weights['coverage']
        
        # Freshness score
        if staleness_days is not None:
            staleness_cap = rules['scoring']['freshness_score']['staleness_days_cap']
            freshness_subscore = 1 - (staleness_days / staleness_cap)
            freshness_subscore = max(0, min(1, freshness_subscore))
        else:
            freshness_subscore = 0.5
        
        score += freshness_subscore * weights['freshness']
        
        # Liquidity score (if not fast mode)
        if not fast_mode:
            vol_ratio = row.get('volume_nonzero_ratio_med', 0)
            pos_ratio = row.get('position_nonzero_ratio_med', 0)
            
            vol_min = rules['scoring']['liquidity_score']['volume_nonzero_min']
            vol_target = rules['scoring']['liquidity_score']['volume_nonzero_target']
            vol_score = (vol_ratio - vol_min) / (vol_target - vol_min)
            vol_score = max(0, min(1, vol_score))
            
            pos_min = rules['scoring']['liquidity_score']['position_nonzero_min']
            pos_target = rules['scoring']['liquidity_score']['position_nonzero_target']
            pos_score = (pos_ratio - pos_min) / (pos_target - pos_min)
            pos_score = max(0, min(1, pos_score))
            
            liquidity_subscore = (vol_score + pos_score) / 2
        else:
            liquidity_subscore = 0.5
        
        score += liquidity_subscore * weights['liquidity']
        
        # Quality score (if not fast mode)
        if not fast_mode:
            quality_subscore = 1.0
            
            dup_ratio = row.get('dup_eob_ratio_max', 0)
            dup_penalty = min(1, dup_ratio * rules['scoring']['quality_score']['duplicate_eob_ratio_penalty_scale'])
            quality_subscore -= dup_penalty
            
            nan_rate = row.get('nan_rate_ohlc_max', 0)
            nan_penalty = min(1, nan_rate * rules['scoring']['quality_score']['nan_rate_penalty_scale'])
            quality_subscore -= nan_penalty
            
            fail_ratio = row.get('sample_read_fail_ratio', 0)
            fail_penalty = min(1, fail_ratio * rules['scoring']['quality_score']['read_fail_penalty_scale'])
            quality_subscore -= fail_penalty
            
            quality_subscore = max(0, quality_subscore)
        else:
            quality_subscore = 0.5
        
        score += quality_subscore * weights['quality']
        
        # Scale to 0-100
        score = score * 100
        
        # Type bonus
        symbol_type = row['symbol_type']
        type_bonus = rules['scoring']['type_bonus'].get(symbol_type, 0)
        score += type_bonus * 100
        
        # Clamp
        score = max(0, min(100, score))
        
        scores.append(score)
        tags_list.append(tags)
    
    df['score'] = scores
    df['tags'] = tags_list
    
    print(f"   Scored {len(df)} symbols")
    print(f"   Score range: {df['score'].min():.1f} - {df['score'].max():.1f}")
    print(f"   Mean score: {df['score'].mean():.1f}")
    
    return df


def generate_reports(df: pd.DataFrame, global_stats: dict, rules: dict, output_dir: str, top_k: int):
    """Generate all output reports."""
    print(f"\n4. Generating reports...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/metadata", exist_ok=True)
    os.makedirs("configs/universe", exist_ok=True)
    
    # Sort by score
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Select candidates
    min_score = rules['selection']['min_score_for_candidates']
    candidates = df[df['score'] >= min_score].head(top_k)
    
    # Markdown report
    md_path = f"{output_dir}/m1_t0_universe_inventory.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# M1-T0 Universe Inventory Report\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 全局概览\n\n")
        f.write(f"- **总 Symbols**: {global_stats['total_symbols']}\n")
        f.write(f"- **时间范围**: {global_stats['global_min_eob']} 至 {global_stats['global_max_eob']}\n")
        f.write(f"- **交易所分布**: {global_stats['by_exchange']}\n")
        f.write(f"- **类型分布**: {global_stats['by_symbol_type']}\n\n")
        
        f.write(f"## Top {len(candidates)} 候选品种\n\n")
        f.write("| Rank | Symbol | Type | Score | Months | Rows | Missing | Vol% | Pos% | Tags |\n")
        f.write("|------|--------|------|-------|--------|------|---------|------|------|------|\n")
        
        for idx, row in candidates.iterrows():
            rank = idx + 1
            vol_pct = f"{row.get('volume_nonzero_ratio_med', 0)*100:.0f}" if 'volume_nonzero_ratio_med' in row else "N/A"
            pos_pct = f"{row.get('position_nonzero_ratio_med', 0)*100:.0f}" if 'position_nonzero_ratio_med' in row else "N/A"
            tags_str = ", ".join(row['tags']) if row['tags'] else "-"
            
            f.write(f"| {rank} | {row['symbol']} | {row['symbol_type']} | {row['score']:.1f} | "
                   f"{row['months']} | {row['rows_total']:,} | {row['missing_months_count']} | "
                   f"{vol_pct}% | {pos_pct}% | {tags_str} |\n")
        
        f.write("\n## 统计摘要\n\n")
        f.write(f"- **候选品种数**: {len(candidates)}\n")
        f.write(f"- **平均 Score**: {candidates['score'].mean():.1f}\n")
        f.write(f"- **平均覆盖月数**: {candidates['months'].mean():.1f}\n")
        f.write(f"- **总数据行数**: {candidates['rows_total'].sum():,}\n\n")
        
        # WARN list
        warn_df = df[(df['score'] >= 45) & (df['score'] < min_score)]
        if len(warn_df) > 0:
            f.write(f"## ⚠️  WARN 清单 ({len(warn_df)} 个)\n\n")
            f.write("分数接近但未达标的品种：\n\n")
            for idx, row in warn_df.head(20).iterrows():
                tags_str = ", ".join(row['tags']) if row['tags'] else "无"
                f.write(f"- **{row['symbol']}** (Score: {row['score']:.1f}): {tags_str}\n")
            f.write("\n")
        
        # FAIL list
        fail_df = df[df['score'] < 45]
        if len(fail_df) > 0:
            f.write(f"## ❌ FAIL 清单 ({len(fail_df)} 个)\n\n")
            f.write("不符合训练要求的品种（按主要原因分类）：\n\n")
            
            fail_reasons = defaultdict(list)
            for idx, row in fail_df.iterrows():
                if 'short_coverage' in row['tags']:
                    fail_reasons['覆盖太短'].append(row['symbol'])
                elif 'low_liquidity' in row['tags']:
                    fail_reasons['流动性不足'].append(row['symbol'])
                elif 'parquet_read_flaky' in row['tags']:
                    fail_reasons['数据读取失败'].append(row['symbol'])
                else:
                    fail_reasons['其他'].append(row['symbol'])
            
            for reason, symbols in fail_reasons.items():
                f.write(f"### {reason} ({len(symbols)} 个)\n\n")
                f.write(", ".join(symbols[:50]))
                if len(symbols) > 50:
                    f.write(f" ... (共 {len(symbols)} 个)")
                f.write("\n\n")
        
        f.write("## 下一步建议\n\n")
        f.write("1. 优先选择 **CONTINUOUS_LIKELY** 且 **Score >= 75** 的品种进行训练\n")
        f.write("2. 检查 WARN 清单中的品种，评估是否可以通过调整阈值纳入\n")
        f.write("3. 对于 REAL_LIKELY 品种，需要额外验证其是否为主力合约\n")
    
    print(f"   ✅ Markdown report: {md_path}")
    
    # JSON report
    json_path = f"{output_dir}/m1_t0_universe_inventory.json"
    json_data = {
        'global': global_stats,
        'symbols': df.to_dict('records'),
        'selected_candidates': candidates[['symbol', 'score', 'symbol_type', 'months', 'rows_total']].to_dict('records'),
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"   ✅ JSON report: {json_path}")
    
    # Parquet table
    parquet_path = "data/metadata/m1_t0_universe_inventory.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"   ✅ Parquet table: {parquet_path}")
    
    # Candidates YAML
    yaml_path = "configs/universe/m1_candidates.yaml"
    candidates_list = candidates['symbol'].tolist()
    
    yaml_data = {
        'version': 'm1_t0_candidates',
        'generated_at': datetime.now().isoformat(),
        'selection_criteria': {
            'min_score': min_score,
            'top_k': top_k,
        },
        'candidates': candidates_list,
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"   ✅ Candidates YAML: {yaml_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ All reports generated successfully")
    print(f"{'='*60}")
    print(f"Top candidates: {len(candidates)}")
    print(f"Candidates list: {', '.join(candidates_list[:10])}")
    if len(candidates_list) > 10:
        print(f"                 ... and {len(candidates_list) - 10} more")


if __name__ == "__main__":
    args = parse_args()
    
    print("="*60)
    print("M1-T0 Universe Scanner")
    print("="*60)
    print(f"Archive: {args.archive_dir}")
    print(f"Manifest: {args.manifest}")
    print(f"Mode: {'FAST (manifest only)' if args.fast else 'FULL (with parquet sampling)'}")
    print("="*60)
    
    # Load rules
    rules = load_rules(args.rules_config)
    print(f"Loaded rules: {rules['version']}")
    
    # Parse manifest
    df, global_stats = parse_manifest(args.manifest, rules)
    
    # Sample parquet (if not fast mode)
    if not args.fast:
        df = sample_parquet_files(df, args.manifest, args.archive_dir, rules, args.sample_months_per_symbol)
    
    # Score symbols
    df = score_symbols(df, global_stats, rules, args.fast)
    
    # Generate reports
    generate_reports(df, global_stats, rules, args.output_dir, args.top_k)
    
    print(f"\n✅ M1-T0 Universe scan complete!")

