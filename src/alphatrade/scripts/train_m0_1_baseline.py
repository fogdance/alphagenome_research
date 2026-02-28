#!/usr/bin/env python3
"""
M0.1 Baseline Training Script

Minimal smoke training on M0.1 dataset with multi-horizon regression.
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="M0.1 Baseline Training")
    parser.add_argument("--config", type=str, default="src/alphatrade/configs/dataset/m0_1.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load dataset config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return "unknown"


class M0Dataset(Dataset):
    """Multi-symbol dataset with on-the-fly window slicing."""
    
    def __init__(
        self,
        symbols: List[str],
        processed_root: str,
        features: List[str],
        split: str = "train",
        limit_samples: int = None
    ):
        self.symbols = symbols
        self.processed_root = processed_root
        self.features = features
        self.split = split
        
        # Load bars and indices for all symbols
        self.bars_dict = {}
        self.indices = []
        
        for symbol in symbols:
            symbol_dir = Path(processed_root) / symbol
            
            # Load bars
            bars_path = symbol_dir / "bars.parquet"
            bars_df = pd.read_parquet(bars_path)
            
            # Extract feature columns as numpy array
            feature_data = bars_df[features].values.astype(np.float32)
            self.bars_dict[symbol] = feature_data
            
            # Load index
            index_path = symbol_dir / f"index_{split}.parquet"
            index_df = pd.read_parquet(index_path)
            
            # Add symbol column if not present
            if "symbol" not in index_df.columns:
                index_df["symbol"] = symbol
            
            self.indices.append(index_df)
        
        # Concatenate all indices
        self.index_df = pd.concat(self.indices, ignore_index=True)
        
        # Limit samples if requested
        if limit_samples and limit_samples < len(self.index_df):
            self.index_df = self.index_df.iloc[:limit_samples].reset_index(drop=True)
        
        # Sanity checks
        self._sanity_check()
    
    def _sanity_check(self):
        """Validate dataset integrity."""
        # Check feature columns exist in bars
        for symbol, bars in self.bars_dict.items():
            assert bars.shape[1] == len(self.features), \
                f"Feature dimension mismatch for {symbol}"
        
        # Check index columns
        required_cols = ["symbol", "t", "x_start", "x_end", "y_h1", "y_h5", "y_h20", "y_h60"]
        for col in required_cols:
            assert col in self.index_df.columns, f"Missing column: {col}"
        
        # Check t bounds
        for _, row in self.index_df.iterrows():
            symbol = row["symbol"]
            t = row["t"]
            x_start = row["x_start"]
            x_end = row["x_end"]
            bars_len = len(self.bars_dict[symbol])
            assert x_end < bars_len, f"Index x_end={x_end} out of bounds for {symbol} (len={bars_len})"
            assert x_end == t, f"x_end != t for sample"
    
    def __len__(self):
        return len(self.index_df)
    
    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        
        symbol = row["symbol"]
        x_start = int(row["x_start"])
        x_end = int(row["x_end"])
        
        # Get bars for this symbol
        bars = self.bars_dict[symbol]
        
        # Extract window
        X = bars[x_start:x_end+1]  # [lookback, feature_dim]
        
        # Extract labels
        y = np.array([
            row["y_h1"],
            row["y_h5"],
            row["y_h20"],
            row["y_h60"]
        ], dtype=np.float32)
        
        return torch.from_numpy(X), torch.from_numpy(y)


class BaselineMLP(nn.Module):
    """Simple MLP for multi-horizon regression."""
    
    def __init__(self, lookback: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        input_dim = lookback * feature_dim
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 4 horizons
        )
    
    def forward(self, x):
        # x: [B, lookback, feature_dim]
        return self.net(x)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_steps: int,
    current_step: int
) -> Tuple[float, int]:
    """Train for one epoch or until max_steps."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X, y in dataloader:
        if current_step >= max_steps:
            break
        
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        current_step += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, current_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Extract config
    # Extract config (support both old and new structure)
    symbols = config.get("universe", {}).get("symbols") or config.get("symbols", [])
    processed_root = config.get("paths", {}).get("processed_dir") or config.get("dataset", {}).get("processed_root", "data/processed/m0_1")
    features = config["features"]
    lookback = config["sampling"]["lookback"]
    horizons = config["sampling"]["horizons"]
    
    print(f"\nDataset: {config['dataset']['name']}")
    print(f"Symbols: {symbols}")
    print(f"Features: {len(features)}")
    print(f"Lookback: {lookback}")
    print(f"Horizons: {horizons}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = M0Dataset(
        symbols, processed_root, features, "train",
        limit_samples=args.limit_train_samples
    )
    val_dataset = M0Dataset(
        symbols, processed_root, features, "val",
        limit_samples=args.limit_val_samples
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    # Create model
    print("\nInitializing model...")
    model = BaselineMLP(lookback, len(features)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.max_steps} steps...")
    current_step = 0
    best_val_loss = float('inf')
    best_step = 0
    train_last_loss = 0.0
    train_best_loss = float('inf')
    val_last_loss = 0.0
    
    while current_step < args.max_steps:
        # Train
        train_loss, current_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.max_steps, current_step
        )
        train_last_loss = train_loss
        train_best_loss = min(train_best_loss, train_loss)
        
        # Validate periodically
        if current_step % args.val_every == 0 or current_step >= args.max_steps:
            val_loss = validate(model, val_loader, criterion, device)
            val_last_loss = val_loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = current_step
            
            print(f"Step {current_step}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, best_val={best_val_loss:.6f}")
    
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.6f} at step {best_step}")
    
    # Generate metrics JSON
    run_id = str(uuid.uuid4())[:8]
    git_sha = get_git_sha()
    
    # Count samples by symbol
    samples_by_symbol = {}
    for symbol in symbols:
        train_count = len(train_dataset.index_df[train_dataset.index_df["symbol"] == symbol])
        val_count = len(val_dataset.index_df[val_dataset.index_df["symbol"] == symbol])
        samples_by_symbol[symbol] = {"train": train_count, "val": val_count}
    
    metrics = {
        "run": {
            "run_id": run_id,
            "git_sha": git_sha,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "seed": args.seed
        },
        "dataset": {
            "name": config["dataset"]["name"],
            "config_path": args.config,
            "lookback": lookback,
            "stride": config["sampling"]["stride"],
            "horizons": horizons,
            "features": features,
            "symbols": symbols,
            "samples": {
                "train_total": len(train_dataset),
                "val_total": len(val_dataset),
                "by_symbol": samples_by_symbol
            }
        },
        "train": {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "loss": {
                "train_last": float(train_last_loss),
                "train_best": float(train_best_loss),
                "val_last": float(val_last_loss),
                "val_best": float(best_val_loss),
                "best_step": best_step
            }
        }
    }
    
    # Save metrics JSON
    metrics_path = "reports/m0_1_train_metrics.json"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved: {metrics_path}")
    
    # Generate markdown report
    md_path = "reports/m0_1_train_run.md"
    with open(md_path, 'w') as f:
        f.write("# M0.1 Baseline Training Run\n\n")
        f.write(f"生成时间: {metrics['run']['created_at']}\n\n")
        
        f.write("## 运行命令\n\n")
        f.write("```bash\n")
        f.write(f"python src/alphatrade/scripts/train_m0_1_baseline.py \\\n")
        f.write(f"  --config {args.config} \\\n")
        f.write(f"  --seed {args.seed} \\\n")
        f.write(f"  --max-steps {args.max_steps} \\\n")
        f.write(f"  --batch-size {args.batch_size} \\\n")
        f.write(f"  --num-workers {args.num_workers} \\\n")
        f.write(f"  --device {args.device}\n")
        f.write("```\n\n")
        
        f.write("## 配置摘要\n\n")
        f.write(f"- **Dataset**: {config['dataset']['name']}\n")
        f.write(f"- **Symbols**: {', '.join(symbols)}\n")
        f.write(f"- **Features**: {len(features)} ({', '.join(features[:3])}...)\n")
        f.write(f"- **Lookback**: {lookback}\n")
        f.write(f"- **Stride**: {config['sampling']['stride']}\n")
        f.write(f"- **Horizons**: {horizons}\n\n")
        
        f.write("## 数据量统计\n\n")
        f.write(f"- **Train total**: {len(train_dataset):,}\n")
        f.write(f"- **Val total**: {len(val_dataset):,}\n\n")
        f.write("### 按 Symbol\n\n")
        for symbol, counts in samples_by_symbol.items():
            f.write(f"- **{symbol}**: train={counts['train']:,}, val={counts['val']:,}\n")
        f.write("\n")
        
        f.write("## 训练结果\n\n")
        f.write(f"- **Device**: {device}\n")
        f.write(f"- **Max steps**: {args.max_steps}\n")
        f.write(f"- **Batch size**: {args.batch_size}\n")
        f.write(f"- **Learning rate**: {args.lr}\n\n")
        
        f.write("### Loss\n\n")
        f.write(f"- **Train last**: {train_last_loss:.6f}\n")
        f.write(f"- **Train best**: {train_best_loss:.6f}\n")
        f.write(f"- **Val last**: {val_last_loss:.6f}\n")
        f.write(f"- **Val best**: {best_val_loss:.6f} (step {best_step})\n\n")
        
        if args.limit_train_samples or args.limit_val_samples:
            f.write("## 注意事项\n\n")
            if args.limit_train_samples:
                f.write(f"- ⚠️ Train samples limited to {args.limit_train_samples}\n")
            if args.limit_val_samples:
                f.write(f"- ⚠️ Val samples limited to {args.limit_val_samples}\n")
    
    print(f"✅ Report saved: {md_path}")


if __name__ == "__main__":
    main()
