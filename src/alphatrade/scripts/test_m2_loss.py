#!/usr/bin/env python3
"""
M2 Loss Sanity Check

Tests:
- Pinball loss computation
- Quantile crossing penalty
- By-horizon loss statistics
- Loss convergence over 200 steps
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.feature_schema import FEATURE_COLS, FEATURE_DIM
import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="M2 Loss Sanity Check")
    parser.add_argument("--config", type=str, default="configs/dataset/m2.yaml")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--smoke", action="store_true")
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Import dataset from train script
import pandas as pd

class M2Dataset:
    """Simplified dataset for testing."""
    
    def __init__(self, symbols, processed_root, split="train"):
        self.symbols = symbols
        self.features = FEATURE_COLS
        self.bars_dict = {}
        self.indices = []
        
        print(f"\nLoading {split} data...")
        for symbol in symbols:
            symbol_dir = Path(processed_root) / symbol
            bars_path = symbol_dir / "bars.parquet"
            bars_df = pd.read_parquet(bars_path)
            
            feature_data = bars_df[self.features].values.astype(np.float32)
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            self.bars_dict[symbol] = feature_data
            
            index_path = symbol_dir / f"index_{split}.parquet"
            index_df = pd.read_parquet(index_path)
            
            for _, row in index_df.iterrows():
                self.indices.append({
                    "symbol": symbol,
                    "x_start": int(row["x_start"]),
                    "x_end": int(row["x_end"]),
                    "y_h1": float(row["y_h1"]),
                    "y_h5": float(row["y_h5"]),
                    "y_h20": float(row["y_h20"]),
                    "y_h60": float(row["y_h60"]),
                })
            
            print(f"  {symbol}: {len(index_df):,} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        entry = self.indices[idx]
        x = self.bars_dict[entry["symbol"]][entry["x_start"]:entry["x_end"]+1]
        y = np.array([entry["y_h1"], entry["y_h5"], entry["y_h20"], entry["y_h60"]], dtype=np.float32)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}


class SimpleQuantileModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim=8, lookback=60, n_horizons=4, n_quantiles=5):
        super().__init__()
        flat_dim = lookback * input_dim
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_horizons * n_quantiles)
        )
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
    
    def forward(self, x):
        B, L, F = x.shape
        x_flat = x.view(B, -1)
        out = self.mlp(x_flat)
        return out.view(B, self.n_horizons, self.n_quantiles)


def pinball_loss_with_details(pred, target, quantiles, horizon_weights=None):
    """Pinball loss with per-horizon details."""
    B, H, Q = pred.shape
    target = target.unsqueeze(-1)
    errors = target - pred
    
    quantiles_t = torch.tensor(quantiles, device=pred.device).view(1, 1, Q)
    loss = torch.where(errors >= 0, quantiles_t * errors, (quantiles_t - 1) * errors)
    
    # Per-horizon loss
    loss_per_horizon = loss.mean(dim=(0, 2))  # [H]
    
    # Total loss
    if horizon_weights is not None:
        hw = torch.tensor(horizon_weights, device=pred.device)
        total_loss = (loss_per_horizon * hw).sum() / hw.sum()
    else:
        total_loss = loss_per_horizon.mean()
    
    return total_loss, loss_per_horizon


def quantile_crossing_penalty(pred):
    """Quantile crossing penalty."""
    diffs = pred[:, :, 1:] - pred[:, :, :-1]
    penalty = torch.relu(-diffs).mean()
    return penalty


def train_and_analyze(config, args):
    """Train for 200 steps and analyze loss."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get symbols
    if args.smoke:
        symbols = config['universe']['smoke_symbols']
    else:
        with open(config['universe']['candidates_file'], 'r') as f:
            universe = yaml.safe_load(f)
        symbols = universe['candidates']
    
    print(f"\n{'='*60}")
    print(f"M2-T2: Loss Sanity Check")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Max steps: {args.max_steps}")
    print(f"{'='*60}")
    
    # Create dataset
    train_dataset = M2Dataset(symbols, config['paths']['processed_dir'], "train")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    
    # Create model
    model = SimpleQuantileModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Training
    quantiles = config['model']['quantiles']['levels']
    horizon_weights = config['loss']['pinball']['horizon_weights']
    crossing_enabled = config['loss']['crossing_penalty']['enabled']
    crossing_weight = config['loss']['crossing_penalty']['weight']
    
    print(f"\nTraining...")
    
    train_losses = []
    pinball_losses = []
    crossing_penalties = []
    horizon_losses = {f"h{h}": [] for h in [1, 5, 20, 60]}
    
    step = 0
    for batch in train_loader:
        if step >= args.max_steps:
            break
        
        model.train()
        optimizer.zero_grad()
        
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        
        pred = model(x)
        
        # Pinball loss with details
        pinball, loss_per_h = pinball_loss_with_details(pred, y, quantiles, horizon_weights)
        
        # Crossing penalty
        crossing = quantile_crossing_penalty(pred)
        
        # Total loss
        if crossing_enabled:
            total_loss = pinball + crossing_weight * crossing
        else:
            total_loss = pinball
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record
        train_losses.append(total_loss.item())
        pinball_losses.append(pinball.item())
        crossing_penalties.append(crossing.item())
        
        for i, h in enumerate([1, 5, 20, 60]):
            horizon_losses[f"h{h}"].append(loss_per_h[i].item())
        
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}/{args.max_steps} | "
                  f"Loss: {total_loss.item():.6f} | "
                  f"Pinball: {pinball.item():.6f} | "
                  f"Crossing: {crossing.item():.6f}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    
    return {
        "train_losses": train_losses,
        "pinball_losses": pinball_losses,
        "crossing_penalties": crossing_penalties,
        "horizon_losses": horizon_losses
    }


def analyze_results(results, config, args):
    """Analyze and report results."""
    
    train_losses = results["train_losses"]
    pinball_losses = results["pinball_losses"]
    crossing_penalties = results["crossing_penalties"]
    horizon_losses = results["horizon_losses"]
    
    # Compute statistics
    stats = {
        "total_loss": {
            "first": float(train_losses[0]),
            "last": float(train_losses[-1]),
            "min": float(min(train_losses)),
            "max": float(max(train_losses)),
            "mean": float(np.mean(train_losses)),
            "std": float(np.std(train_losses)),
            "trend": "decreasing" if train_losses[-1] < train_losses[0] else "increasing"
        },
        "pinball_loss": {
            "first": float(pinball_losses[0]),
            "last": float(pinball_losses[-1]),
            "min": float(min(pinball_losses)),
            "mean": float(np.mean(pinball_losses))
        },
        "crossing_penalty": {
            "first": float(crossing_penalties[0]),
            "last": float(crossing_penalties[-1]),
            "min": float(min(crossing_penalties)),
            "mean": float(np.mean(crossing_penalties)),
            "active": crossing_penalties[-1] > 0
        },
        "by_horizon": {}
    }
    
    for h_name, losses in horizon_losses.items():
        stats["by_horizon"][h_name] = {
            "first": float(losses[0]),
            "last": float(losses[-1]),
            "min": float(min(losses)),
            "mean": float(np.mean(losses))
        }
    
    # Check convergence
    convergence = {
        "loss_decreased": train_losses[-1] < train_losses[0],
        "no_nan": not any(np.isnan(train_losses)),
        "no_inf": not any(np.isinf(train_losses)),
        "stable": np.std(train_losses[-20:]) < 0.01  # Last 20 steps stable
    }
    
    stats["convergence"] = convergence
    
    # Generate report
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    md_path = reports_dir / "m2_t2_loss_sanity.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(md_path, 'w') as f:
        f.write("# M2-T2 Loss Sanity Check Report\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 配置\n\n")
        f.write(f"- **Max steps**: {args.max_steps}\n")
        f.write(f"- **Quantiles**: {config['model']['quantiles']['levels']}\n")
        f.write(f"- **Horizon weights**: {config['loss']['pinball']['horizon_weights']}\n")
        f.write(f"- **Crossing penalty**: {config['loss']['crossing_penalty']['enabled']}\n")
        f.write(f"- **Crossing weight**: {config['loss']['crossing_penalty']['weight']}\n\n")
        
        f.write("## Loss 统计\n\n")
        f.write("### Total Loss\n\n")
        f.write(f"- **First**: {stats['total_loss']['first']:.6f}\n")
        f.write(f"- **Last**: {stats['total_loss']['last']:.6f}\n")
        f.write(f"- **Min**: {stats['total_loss']['min']:.6f}\n")
        f.write(f"- **Mean**: {stats['total_loss']['mean']:.6f}\n")
        f.write(f"- **Trend**: {stats['total_loss']['trend']}\n\n")
        
        f.write("### Pinball Loss\n\n")
        f.write(f"- **First**: {stats['pinball_loss']['first']:.6f}\n")
        f.write(f"- **Last**: {stats['pinball_loss']['last']:.6f}\n")
        f.write(f"- **Min**: {stats['pinball_loss']['min']:.6f}\n\n")
        
        f.write("### Crossing Penalty\n\n")
        f.write(f"- **First**: {stats['crossing_penalty']['first']:.6f}\n")
        f.write(f"- **Last**: {stats['crossing_penalty']['last']:.6f}\n")
        f.write(f"- **Min**: {stats['crossing_penalty']['min']:.6f}\n")
        f.write(f"- **Active**: {'✅' if stats['crossing_penalty']['active'] else '❌'}\n\n")
        
        f.write("### By-Horizon Loss\n\n")
        f.write("| Horizon | First | Last | Min | Mean |\n")
        f.write("|---------|-------|------|-----|------|\n")
        for h_name in ["h1", "h5", "h20", "h60"]:
            h_stats = stats["by_horizon"][h_name]
            f.write(f"| {h_name} | {h_stats['first']:.6f} | {h_stats['last']:.6f} | "
                   f"{h_stats['min']:.6f} | {h_stats['mean']:.6f} |\n")
        f.write("\n")
        
        f.write("## 收敛性检查\n\n")
        conv = stats["convergence"]
        f.write(f"- **Loss decreased**: {'✅' if conv['loss_decreased'] else '❌'}\n")
        f.write(f"- **No NaN**: {'✅' if conv['no_nan'] else '❌'}\n")
        f.write(f"- **No Inf**: {'✅' if conv['no_inf'] else '❌'}\n")
        f.write(f"- **Stable**: {'✅' if conv['stable'] else '❌'}\n\n")
        
        all_pass = all(conv.values())
        f.write(f"### 总体状态: {'✅ PASS' if all_pass else '❌ FAIL'}\n\n")
    
    print(f"\n✅ Report saved: {md_path}")
    
    return stats


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Train and collect results
    results = train_and_analyze(config, args)
    
    # Analyze results
    stats = analyze_results(results, config, args)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ M2-T2 Complete")
    print(f"{'='*60}")
    print(f"Loss trend: {stats['total_loss']['trend']}")
    print(f"Convergence: {'✅ PASS' if all(stats['convergence'].values()) else '❌ FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
