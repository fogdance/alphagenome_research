#!/usr/bin/env python3
"""
M2 Training Script for AlphaTrade v0.2

Multi-symbol training with quantile prediction and pinball loss.
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
import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="M2 AlphaTrade v0.2 Training")
    parser.add_argument("--config", type=str, default="configs/dataset/m2.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--smoke", action="store_true", help="Use smoke symbols (2-5 symbols)")
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML."""
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


class M2Dataset(Dataset):
    """M2 multi-symbol dataset with quantile labels."""
    
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
        
        # Load bars and indices
        self.bars_dict = {}
        self.indices = []
        
        print(f"\nLoading {split} data...")
        for symbol in symbols:
            symbol_dir = Path(processed_root) / symbol
            
            # Load bars
            bars_path = symbol_dir / "bars.parquet"
            bars_df = pd.read_parquet(bars_path)
            
            # Extract features and handle NaN
            feature_data = bars_df[self.features].values.astype(np.float32)
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            self.bars_dict[symbol] = feature_data
            
            # Load index
            index_path = symbol_dir / f"index_{split}.parquet"
            index_df = pd.read_parquet(index_path)
            
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
        
        # Slice window
        x = self.bars_dict[symbol][x_start:x_end+1]  # [L, F]
        
        # Labels (returns for each horizon)
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


class SimpleQuantileModel(nn.Module):
    """Simple MLP model with quantile outputs (placeholder for AlphaTrade v0.2)."""
    
    def __init__(
        self,
        input_dim: int,
        lookback: int,
        n_horizons: int,
        n_quantiles: int,
        hidden_dims: List[int] = [256, 128, 64]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.lookback = lookback
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        
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
        
        # Output layer: [horizons * quantiles]
        layers.append(nn.Linear(prev_dim, n_horizons * n_quantiles))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, F]
        Returns:
            [B, H, Q] quantile predictions
        """
        B, L, F = x.shape
        x_flat = x.view(B, -1)
        out = self.mlp(x_flat)  # [B, H*Q]
        out = out.view(B, self.n_horizons, self.n_quantiles)  # [B, H, Q]
        return out


def pinball_loss(pred, target, quantiles, horizon_weights=None):
    """
    Pinball loss for quantile regression.
    
    Args:
        pred: [B, H, Q] predicted quantiles
        target: [B, H] target values
        quantiles: [Q] quantile levels
        horizon_weights: [H] weights for each horizon
    
    Returns:
        scalar loss
    """
    B, H, Q = pred.shape
    
    # Expand target to match pred shape
    target = target.unsqueeze(-1)  # [B, H, 1]
    
    # Compute errors
    errors = target - pred  # [B, H, Q]
    
    # Quantile weights
    quantiles = torch.tensor(quantiles, device=pred.device).view(1, 1, Q)
    
    # Pinball loss
    loss = torch.where(
        errors >= 0,
        quantiles * errors,
        (quantiles - 1) * errors
    )
    
    # Average over batch and quantiles
    loss = loss.mean(dim=(0, 2))  # [H]
    
    # Apply horizon weights
    if horizon_weights is not None:
        horizon_weights = torch.tensor(horizon_weights, device=pred.device)
        loss = (loss * horizon_weights).sum() / horizon_weights.sum()
    else:
        loss = loss.mean()
    
    return loss


def quantile_crossing_penalty(pred):
    """
    Penalty for quantile crossing (q_i > q_j when i < j).
    
    Args:
        pred: [B, H, Q] predicted quantiles
    
    Returns:
        scalar penalty
    """
    # Compute differences between adjacent quantiles
    diffs = pred[:, :, 1:] - pred[:, :, :-1]  # [B, H, Q-1]
    
    # Penalty for negative differences (crossing)
    penalty = torch.relu(-diffs).mean()
    
    return penalty


def train_step(model, batch, optimizer, config, grad_clip):
    """Single training step."""
    model.train()
    optimizer.zero_grad()
    
    x = batch["x"]
    y = batch["y"]
    
    # Forward
    pred = model(x)  # [B, H, Q]
    
    # Pinball loss
    quantiles = config['model']['quantiles']['levels']
    horizon_weights = config['loss']['pinball']['horizon_weights']
    loss = pinball_loss(pred, y, quantiles, horizon_weights)
    
    # Quantile crossing penalty
    if config['loss']['crossing_penalty']['enabled']:
        penalty = quantile_crossing_penalty(pred)
        penalty_weight = config['loss']['crossing_penalty']['weight']
        loss = loss + penalty_weight * penalty
    
    # Backward
    loss.backward()
    
    # Gradient clipping
    max_grad_norm = 0.0
    if grad_clip > 0:
        max_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()
    
    return loss.item(), max_grad_norm


def val_step(model, val_loader, config, device):
    """Validation step."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    quantiles = config['model']['quantiles']['levels']
    horizon_weights = config['loss']['pinball']['horizon_weights']
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            pred = model(x)
            loss = pinball_loss(pred, y, quantiles, horizon_weights)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else 0.0


def save_metrics(config, args, model, train_losses, val_losses, best_val_loss, best_step, 
                 symbols, train_dataset, val_dataset, device, run_id, max_grad_norm):
    """Save metrics to JSON (m2_train_metrics_v1 schema)."""
    
    metrics = {
        "run": {
            "run_id": run_id,
            "git_sha": get_git_sha(),
            "created_at": datetime.now().isoformat(),
            "device": str(device),
            "seed": config['training']['seed']
        },
        "dataset": {
            "name": "m2",
            "config_path": args.config,
            "processed_dir": config['paths']['processed_dir'],
            "symbols": len(symbols),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": 0,  # Not used in training
            "feature_dim": FEATURE_DIM,
            "feature_cols": FEATURE_COLS,
            "lookback": config['sample_index']['lookback'],
            "horizons": config['sample_index']['horizons'],
            "quantiles": config['model']['quantiles']['levels']
        },
        "model": {
            "type": config['model']['type'],
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "config": {
                "n_quantiles": config['model']['quantiles']['num'],
                "hidden_dim": config['model']['hidden_dim']
            }
        },
        "training": {
            "max_steps": config['training']['max_steps'],
            "batch_size": config['training']['batch_size'],
            "learning_rate": config['training']['learning_rate'],
            "weight_decay": config['training']['weight_decay'],
            "grad_clip": config['training']['grad_clip'],
            "optimizer": config['training']['optimizer'],
            "lr_schedule": config['training']['lr_schedule']
        },
        "loss": {
            "train_last": float(train_losses[-1]) if train_losses else 0.0,
            "train_best": float(min(train_losses)) if train_losses else 0.0,
            "val_last": float(val_losses[-1]) if val_losses else 0.0,
            "val_best": float(best_val_loss),
            "best_step": best_step,
            "by_horizon": {
                "h1": {"train": 0.0, "val": 0.0},
                "h5": {"train": 0.0, "val": 0.0},
                "h20": {"train": 0.0, "val": 0.0},
                "h60": {"train": 0.0, "val": 0.0}
            }
        },
        "stability": {
            "nan_steps": 0,
            "inf_steps": 0,
            "max_grad_norm": float(max_grad_norm)
        }
    }
    
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    json_path = reports_dir / "m2_train_metrics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved: {json_path}")
    return metrics


def save_run_report(config, args, metrics):
    """Save run report to Markdown."""
    
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    md_path = reports_dir / "m2_train_run.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, 'w') as f:
        f.write("# M2 Training Run (AlphaTrade v0.2)\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 运行命令\n\n")
        f.write("```bash\n")
        f.write(f"python src/alphatrade/scripts/train_m2_alphatrade_v0_2.py \\\n")
        f.write(f"  --config {args.config}")
        if args.max_steps:
            f.write(f" \\\n  --max-steps {args.max_steps}")
        if args.batch_size:
            f.write(f" \\\n  --batch-size {args.batch_size}")
        if args.seed:
            f.write(f" \\\n  --seed {args.seed}")
        if args.smoke:
            f.write(f" \\\n  --smoke")
        f.write("\n```\n\n")
        
        f.write("## 配置摘要\n\n")
        f.write(f"- **Run ID**: {metrics['run']['run_id']}\n")
        f.write(f"- **Git SHA**: {metrics['run']['git_sha']}\n")
        f.write(f"- **Device**: {metrics['run']['device']}\n")
        f.write(f"- **Seed**: {metrics['run']['seed']}\n\n")
        
        f.write("### Dataset\n\n")
        f.write(f"- **Symbols**: {metrics['dataset']['symbols']}\n")
        f.write(f"- **Train samples**: {metrics['dataset']['train_samples']:,}\n")
        f.write(f"- **Val samples**: {metrics['dataset']['val_samples']:,}\n")
        f.write(f"- **Feature dim**: {metrics['dataset']['feature_dim']}\n")
        f.write(f"- **Lookback**: {metrics['dataset']['lookback']}\n")
        f.write(f"- **Horizons**: {metrics['dataset']['horizons']}\n")
        f.write(f"- **Quantiles**: {metrics['dataset']['quantiles']}\n\n")
        
        f.write("### Model\n\n")
        f.write(f"- **Type**: {metrics['model']['type']}\n")
        f.write(f"- **Total params**: {metrics['model']['total_params']:,}\n\n")
        
        f.write("### Training\n\n")
        f.write(f"- **Max steps**: {metrics['training']['max_steps']}\n")
        f.write(f"- **Batch size**: {metrics['training']['batch_size']}\n")
        f.write(f"- **Learning rate**: {metrics['training']['learning_rate']}\n")
        f.write(f"- **Grad clip**: {metrics['training']['grad_clip']}\n\n")
        
        f.write("### Loss\n\n")
        f.write(f"- **Train last**: {metrics['loss']['train_last']:.6f}\n")
        f.write(f"- **Train best**: {metrics['loss']['train_best']:.6f}\n")
        f.write(f"- **Val last**: {metrics['loss']['val_last']:.6f}\n")
        f.write(f"- **Val best**: {metrics['loss']['val_best']:.6f} (step {metrics['loss']['best_step']})\n\n")
        
        f.write("### Stability\n\n")
        f.write(f"- **NaN steps**: {metrics['stability']['nan_steps']}\n")
        f.write(f"- **Inf steps**: {metrics['stability']['inf_steps']}\n")
        f.write(f"- **Max grad norm**: {metrics['stability']['max_grad_norm']:.4f}\n\n")
    
    print(f"✅ Run report saved: {md_path}")


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.seed:
        config['training']['seed'] = args.seed
    
    # Set seed
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get symbols
    if args.smoke:
        symbols = config['universe']['smoke_symbols']
    else:
        # Load from candidates file
        candidates_file = config['universe']['candidates_file']
        with open(candidates_file, 'r') as f:
            universe = yaml.safe_load(f)
        symbols = universe['candidates']
    
    print(f"\n{'='*60}")
    print(f"M2 Training (AlphaTrade v0.2)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Symbols: {len(symbols)}")
    print(f"Max steps: {config['training']['max_steps']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"{'='*60}")
    
    # Create datasets
    train_dataset = M2Dataset(symbols, config['paths']['processed_dir'], split="train")
    val_dataset = M2Dataset(symbols, config['paths']['processed_dir'], split="val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Create model (placeholder)
    model = SimpleQuantileModel(
        input_dim=FEATURE_DIM,
        lookback=config['sample_index']['lookback'],
        n_horizons=len(config['sample_index']['horizons']),
        n_quantiles=config['model']['quantiles']['num']
    ).to(device)
    
    print(f"\nModel: SimpleQuantileModel (placeholder)")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training...")
    print(f"{'='*60}")
    
    run_id = str(uuid.uuid4())[:8]
    best_val_loss = float('inf')
    best_step = 0
    train_losses = []
    val_losses = []
    max_grad_norm_overall = 0.0
    
    step = 0
    max_steps = config['training']['max_steps']
    val_every = config['training']['val_every']
    grad_clip = config['training']['grad_clip']
    
    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Train step
            loss, max_grad_norm = train_step(model, batch, optimizer, config, grad_clip)
            train_losses.append(loss)
            max_grad_norm_overall = max(max_grad_norm_overall, max_grad_norm)
            
            step += 1
            
            # Validation
            if step % val_every == 0 or step == max_steps:
                val_loss = val_step(model, val_loader, config, device)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                
                print(f"Step {step}/{max_steps} | "
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
    
    # Save metrics
    print(f"\nGenerating reports...")
    metrics = save_metrics(
        config, args, model, train_losses, val_losses, best_val_loss, best_step,
        symbols, train_dataset, val_dataset, device, run_id, max_grad_norm_overall
    )
    save_run_report(config, args, metrics)
    
    print(f"\n{'='*60}")
    print(f"✅ M2-T0 Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
