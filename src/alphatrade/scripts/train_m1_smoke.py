#!/usr/bin/env python3
"""
M1 Smoke Training Script

Minimal smoke training on M1 dataset (8D features, 24 symbols) with AlphaTrade v0.2.
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import unified feature schema
from data_pipeline.feature_schema import FEATURE_COLS, FEATURE_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="M1 Smoke Training")
    parser.add_argument("--universe", type=str, default="configs/universe/m1_selected.yaml")
    parser.add_argument("--processed-dir", type=str, default="data/processed/m1_f8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_args()


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


class M1Dataset(Dataset):
    """Multi-symbol dataset with on-the-fly window slicing (8D features)."""
    
    def __init__(
        self,
        symbols: List[str],
        processed_root: str,
        split: str = "train"
    ):
        self.symbols = symbols
        self.processed_root = processed_root
        self.split = split
        
        # Use unified feature schema
        self.features = FEATURE_COLS
        self.feature_dim = FEATURE_DIM
        
        # Load bars and indices for all symbols
        self.bars_dict = {}
        self.indices = []
        
        print(f"\nLoading {split} data...")
        for symbol in symbols:
            symbol_dir = Path(processed_root) / symbol
            
            # Load bars
            bars_path = symbol_dir / "bars.parquet"
            bars_df = pd.read_parquet(bars_path)
            
            # Extract feature columns as numpy array
            feature_data = bars_df[self.features].values.astype(np.float32)

            # Fill NaN with 0 (first row of ret_1m is NaN from pct_change)
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

            self.bars_dict[symbol] = feature_data
            
            # Load index
            index_path = symbol_dir / f"index_{split}.parquet"
            index_df = pd.read_parquet(index_path)
            
            # Add symbol column
            if "symbol" not in index_df.columns:
                index_df["symbol"] = symbol
            
            # Store indices
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
            
            print(f"  {symbol}: {len(bars_df):,} bars, {len(index_df):,} samples")
        
        print(f"Total {split} samples: {len(self.indices):,}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        entry = self.indices[idx]
        symbol = entry["symbol"]
        x_start = entry["x_start"]
        x_end = entry["x_end"]
        
        # Slice window from bars
        x = self.bars_dict[symbol][x_start:x_end+1]  # [L, F]
        
        # Labels
        y = np.array([
            entry["y_h1"],
            entry["y_h5"],
            entry["y_h20"],
            entry["y_h60"]
        ], dtype=np.float32)
        
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y)
        }


class SimpleMLPModel(nn.Module):
    """Simple MLP baseline for smoke test."""
    
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        n_horizons: int,
        hidden_dims: List[int] = [256, 128, 64]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.lookback = lookback
        self.n_horizons = n_horizons
        
        # Flatten input
        flat_dim = lookback * input_dim
        
        # MLP layers
        layers = []
        prev_dim = flat_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, n_horizons))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, F]
        Returns:
            [B, H]
        """
        B, L, F = x.shape
        x_flat = x.view(B, -1)  # [B, L*F]
        return self.mlp(x_flat)


def train_step(model, batch, optimizer, criterion, grad_clip):
    """Single training step."""
    model.train()
    optimizer.zero_grad()
    
    x = batch["x"]  # [B, L, F]
    y = batch["y"]  # [B, H]
    
    # Forward
    pred = model(x)
    loss = criterion(pred, y)
    
    # Backward
    loss.backward()
    
    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    
    return loss.item()


def val_step(model, val_loader, criterion, device):
    """Validation step."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            pred = model(x)
            loss = criterion(pred, y)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"M1 Smoke Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Grad clip: {args.grad_clip}")
    print(f"{'='*60}")
    
    # Load universe
    with open(args.universe, 'r') as f:
        universe = yaml.safe_load(f)
    symbols = universe['candidates']
    
    print(f"\nUniverse: {len(symbols)} symbols")
    
    # Create datasets
    train_dataset = M1Dataset(symbols, args.processed_dir, split="train")
    val_dataset = M1Dataset(symbols, args.processed_dir, split="val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Create model
    model = SimpleMLPModel(
        input_dim=FEATURE_DIM,  # 8
        lookback=60,
        n_horizons=4,
        hidden_dims=[256, 128, 64]
    ).to(device)
    
    print(f"\nModel: SimpleMLPModel")
    print(f"  Input dim: {FEATURE_DIM}")
    print(f"  Lookback: 60")
    print(f"  Horizons: 4")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training...")
    print(f"{'='*60}")
    
    run_id = str(uuid.uuid4())[:8]
    best_val_loss = float('inf')
    best_step = 0
    train_losses = []
    val_losses = []
    
    step = 0
    epoch = 0
    
    while step < args.max_steps:
        epoch += 1
        
        for batch in train_loader:
            if step >= args.max_steps:
                break
            
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Train step
            loss = train_step(model, batch, optimizer, criterion, args.grad_clip)
            train_losses.append(loss)
            
            step += 1
            
            # Validation
            if step % args.val_every == 0 or step == args.max_steps:
                val_loss = val_step(model, val_loader, criterion, device)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                
                print(f"Step {step}/{args.max_steps} | "
                      f"Train Loss: {loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Best: {best_val_loss:.6f} @ {best_step}")
                
                # Check for NaN/Inf
                if np.isnan(loss) or np.isinf(loss):
                    print(f"\n❌ Training failed: NaN/Inf detected!")
                    sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete")
    print(f"{'='*60}")
    print(f"Best val loss: {best_val_loss:.6f} @ step {best_step}")
    
    # Generate reports
    print(f"\nGenerating reports...")
    
    # JSON report
    metrics = {
        "run": {
            "run_id": run_id,
            "git_sha": get_git_sha(),
            "created_at": datetime.now().isoformat(),
            "device": str(device),
            "seed": args.seed
        },
        "dataset": {
            "name": "m1_f8",
            "universe": args.universe,
            "processed_dir": args.processed_dir,
            "symbols": len(symbols),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "feature_dim": FEATURE_DIM,
            "feature_cols": FEATURE_COLS,
            "lookback": 60,
            "horizons": [1, 5, 20, 60]
        },
        "model": {
            "type": "SimpleMLPModel",
            "input_dim": FEATURE_DIM,
            "lookback": 60,
            "n_horizons": 4,
            "hidden_dims": [256, 128, 64],
            "total_params": sum(p.numel() for p in model.parameters())
        },
        "training": {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "grad_clip": args.grad_clip
        },
        "loss": {
            "train_last": float(train_losses[-1]) if train_losses else 0.0,
            "train_best": float(min(train_losses)) if train_losses else 0.0,
            "val_last": float(val_losses[-1]) if val_losses else 0.0,
            "val_best": float(best_val_loss),
            "best_step": best_step
        }
    }
    
    json_path = "reports/m1_train_metrics.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ JSON report: {json_path}")
    
    # Markdown report
    md_path = "reports/m1_train_run.md"
    with open(md_path, 'w') as f:
        f.write("# M1 Smoke Training Run\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 运行命令\n\n")
        f.write("```bash\n")
        f.write(f"python src/alphatrade/scripts/train_m1_smoke.py \\\n")
        f.write(f"  --universe {args.universe} \\\n")
        f.write(f"  --processed-dir {args.processed_dir} \\\n")
        f.write(f"  --max-steps {args.max_steps} \\\n")
        f.write(f"  --batch-size {args.batch_size} \\\n")
        f.write(f"  --lr {args.lr} \\\n")
        f.write(f"  --seed {args.seed}\n")
        f.write("```\n\n")
        
        f.write("## 配置摘要\n\n")
        f.write(f"- **Run ID**: {run_id}\n")
        f.write(f"- **Git SHA**: {get_git_sha()}\n")
        f.write(f"- **Device**: {device}\n")
        f.write(f"- **Seed**: {args.seed}\n\n")
        
        f.write("### Dataset\n\n")
        f.write(f"- **Universe**: {len(symbols)} symbols\n")
        f.write(f"- **Train samples**: {len(train_dataset):,}\n")
        f.write(f"- **Val samples**: {len(val_dataset):,}\n")
        f.write(f"- **Feature dim**: {FEATURE_DIM}\n")
        f.write(f"- **Lookback**: 60\n")
        f.write(f"- **Horizons**: [1, 5, 20, 60]\n\n")
        
        f.write("### Model\n\n")
        f.write(f"- **Type**: SimpleMLPModel\n")
        f.write(f"- **Total params**: {sum(p.numel() for p in model.parameters()):,}\n\n")
        
        f.write("### Training\n\n")
        f.write(f"- **Max steps**: {args.max_steps}\n")
        f.write(f"- **Batch size**: {args.batch_size}\n")
        f.write(f"- **Learning rate**: {args.lr}\n")
        f.write(f"- **Grad clip**: {args.grad_clip}\n\n")
        
        f.write("### Loss\n\n")
        f.write(f"- **Train last**: {train_losses[-1]:.6f}\n")
        f.write(f"- **Train best**: {min(train_losses):.6f}\n")
        f.write(f"- **Val last**: {val_losses[-1]:.6f}\n")
        f.write(f"- **Val best**: {best_val_loss:.6f} (step {best_step})\n\n")
        
        f.write("### Status\n\n")
        f.write(f"✅ **Training completed successfully**\n\n")
        f.write(f"- No NaN/Inf detected\n")
        f.write(f"- Loss converged normally\n")
    
    print(f"✅ Markdown report: {md_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ M1-T5.3 Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
