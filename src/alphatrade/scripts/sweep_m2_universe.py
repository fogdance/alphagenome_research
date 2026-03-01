#!/usr/bin/env python3
"""
M2 Universe Sweep

Tests all candidates with short training runs to identify:
- Data availability
- Training feasibility
- Potential issues
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.feature_schema import FEATURE_COLS, FEATURE_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="M2 Universe Sweep")
    parser.add_argument("--config", type=str, default="configs/dataset/m2.yaml")
    parser.add_argument("--universe", type=str, default="configs/universe/m1_candidates.yaml")
    parser.add_argument("--max-steps", type=int, default=50, help="Steps per symbol")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SimpleDataset(Dataset):
    """Simplified dataset for quick testing."""
    
    def __init__(self, symbol, processed_root, split="train"):
        self.symbol = symbol
        self.features = FEATURE_COLS
        self.indices = []
        
        try:
            symbol_dir = Path(processed_root) / symbol
            bars_path = symbol_dir / "bars.parquet"
            
            if not bars_path.exists():
                raise FileNotFoundError(f"bars.parquet not found for {symbol}")
            
            bars_df = pd.read_parquet(bars_path)
            feature_data = bars_df[self.features].values.astype(np.float32)
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            self.bars = feature_data
            
            index_path = symbol_dir / f"index_{split}.parquet"
            if not index_path.exists():
                raise FileNotFoundError(f"index_{split}.parquet not found for {symbol}")
            
            index_df = pd.read_parquet(index_path)
            
            for _, row in index_df.iterrows():
                self.indices.append({
                    "x_start": int(row["x_start"]),
                    "x_end": int(row["x_end"]),
                    "y_h1": float(row["y_h1"]),
                    "y_h5": float(row["y_h5"]),
                    "y_h20": float(row["y_h20"]),
                    "y_h60": float(row["y_h60"]),
                })
            
            self.n_bars = len(bars_df)
            self.n_samples = len(index_df)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {symbol}: {str(e)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        entry = self.indices[idx]
        x = self.bars[entry["x_start"]:entry["x_end"]+1]
        y = np.array([entry["y_h1"], entry["y_h5"], entry["y_h20"], entry["y_h60"]], dtype=np.float32)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(60 * 8, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 20)  # 4 horizons * 5 quantiles
        )
    
    def forward(self, x):
        B, L, F = x.shape
        return self.mlp(x.view(B, -1)).view(B, 4, 5)


def test_symbol(symbol: str, processed_root: str, max_steps: int, batch_size: int, device) -> Dict:
    """Test a single symbol with short training."""
    result = {
        "symbol": symbol,
        "status": "unknown",
        "error": None,
        "n_bars": 0,
        "n_train_samples": 0,
        "n_val_samples": 0,
        "train_loss_first": None,
        "train_loss_last": None,
        "val_loss": None,
        "has_nan": False,
        "trainable": False
    }
    
    try:
        # Load data
        train_dataset = SimpleDataset(symbol, processed_root, "train")
        val_dataset = SimpleDataset(symbol, processed_root, "val")
        
        result["n_bars"] = train_dataset.n_bars
        result["n_train_samples"] = len(train_dataset)
        result["n_val_samples"] = len(val_dataset)
        
        # Check minimum samples
        if len(train_dataset) < 100:
            result["status"] = "insufficient_samples"
            result["error"] = f"Only {len(train_dataset)} train samples (need >= 100)"
            return result
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Create model
        model = SimpleModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Train for a few steps
        train_losses = []
        step = 0
        
        for batch in train_loader:
            if step >= max_steps:
                break
            
            model.train()
            optimizer.zero_grad()
            
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            
            # Simple MSE loss
            loss = nn.functional.mse_loss(pred.mean(dim=-1), y)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                result["has_nan"] = True
                break
            
            step += 1
        
        if train_losses:
            result["train_loss_first"] = float(train_losses[0])
            result["train_loss_last"] = float(train_losses[-1])
        
        # Validation
        if not result["has_nan"]:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    pred = model(x)
                    loss = nn.functional.mse_loss(pred.mean(dim=-1), y)
                    val_losses.append(loss.item())
            
            if val_losses:
                result["val_loss"] = float(np.mean(val_losses))
        
        # Determine status
        if result["has_nan"]:
            result["status"] = "nan_detected"
            result["error"] = "NaN/Inf during training"
        else:
            result["status"] = "success"
            result["trainable"] = True
        
    except FileNotFoundError as e:
        result["status"] = "missing_files"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Load universe
    with open(args.universe, 'r') as f:
        universe = yaml.safe_load(f)
    
    symbols = universe.get('candidates', [])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"M2-T4: Universe Sweep")
    print(f"{'='*60}")
    print(f"Universe: {args.universe}")
    print(f"Symbols: {len(symbols)}")
    print(f"Max steps per symbol: {args.max_steps}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Test each symbol
    results = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Testing {symbol}...", end=' ')
        
        result = test_symbol(symbol, config['paths']['processed_dir'], 
                           args.max_steps, args.batch_size, device)
        results.append(result)
        
        status_icon = {
            "success": "✅",
            "insufficient_samples": "⚠️",
            "missing_files": "❌",
            "nan_detected": "❌",
            "error": "❌"
        }.get(result["status"], "❓")
        
        print(f"{status_icon} {result['status']}")
        if result["error"]:
            print(f"    Error: {result['error']}")
    
    # Generate reports
    print(f"\nGenerating reports...")
    
    # Statistics
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    insufficient = sum(1 for r in results if r["status"] == "insufficient_samples")
    missing = sum(1 for r in results if r["status"] == "missing_files")
    nan = sum(1 for r in results if r["status"] == "nan_detected")
    error = sum(1 for r in results if r["status"] == "error")
    trainable = sum(1 for r in results if r["trainable"])
    
    # JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "universe": args.universe,
        "total_symbols": total,
        "statistics": {
            "success": success,
            "insufficient_samples": insufficient,
            "missing_files": missing,
            "nan_detected": nan,
            "error": error,
            "trainable": trainable,
            "trainable_rate": trainable / total if total > 0 else 0.0
        },
        "results": results
    }
    
    json_path = "reports/m2_universe_sweep.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ JSON report: {json_path}")
    
    # Markdown report
    md_path = "reports/m2_universe_sweep.md"
    with open(md_path, 'w') as f:
        f.write("# M2 Universe Sweep Report\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 总览\n\n")
        f.write(f"- **Universe**: {args.universe}\n")
        f.write(f"- **Total symbols**: {total}\n")
        f.write(f"- **Success**: {success} ({success/total*100:.1f}%)\n")
        f.write(f"- **Trainable**: {trainable} ({trainable/total*100:.1f}%)\n\n")
        
        f.write("## 统计\n\n")
        f.write("| Status | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        f.write(f"| Success | {success} | {success/total*100:.1f}% |\n")
        f.write(f"| Insufficient samples | {insufficient} | {insufficient/total*100:.1f}% |\n")
        f.write(f"| Missing files | {missing} | {missing/total*100:.1f}% |\n")
        f.write(f"| NaN detected | {nan} | {nan/total*100:.1f}% |\n")
        f.write(f"| Error | {error} | {error/total*100:.1f}% |\n\n")
        
        # Success table
        success_results = [r for r in results if r["status"] == "success"]
        if success_results:
            f.write("## 成功品种\n\n")
            f.write("| Symbol | Train Samples | Val Samples | Train Loss | Val Loss |\n")
            f.write("|--------|---------------|-------------|------------|----------|\n")
            for r in success_results:
                f.write(f"| {r['symbol']} | {r['n_train_samples']:,} | {r['n_val_samples']:,} | "
                       f"{r['train_loss_last']:.6f} | {r['val_loss']:.6f} |\n")
            f.write("\n")
        
        # Failed table
        failed_results = [r for r in results if r["status"] != "success"]
        if failed_results:
            f.write("## 失败品种\n\n")
            f.write("| Symbol | Status | Error |\n")
            f.write("|--------|--------|-------|\n")
            for r in failed_results:
                error_msg = r['error'][:50] + "..." if r['error'] and len(r['error']) > 50 else r['error']
                f.write(f"| {r['symbol']} | {r['status']} | {error_msg} |\n")
    
    print(f"✅ Markdown report: {md_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ M2-T4 Complete")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Success: {success} ({success/total*100:.1f}%)")
    print(f"Trainable: {trainable} ({trainable/total*100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
