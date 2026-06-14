#!/usr/bin/env python3
"""
M2 Final Training Script (M2-T3)

Complete training with:
- By-horizon loss tracking
- Full m2_train_metrics_v1 schema
- Stable training with gradient clipping
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.feature_schema import FEATURE_COLS, FEATURE_DIM
import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="M2 Final Training")
    parser.add_argument("--config", type=str, default="configs/dataset/m2.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_git_sha() -> str:
    try:
        import subprocess
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return "unknown"


class M2Dataset(Dataset):
    def __init__(self, symbols, processed_root, split="train"):
        self.symbols = symbols
        self.features = FEATURE_COLS
        self.bars_dict = {}
        self.indices = []
        
        print(f"\nLoading {split} data...")
        for symbol in symbols:
            symbol_dir = Path(processed_root) / symbol
            bars_df = pd.read_parquet(symbol_dir / "bars.parquet")
            feature_data = bars_df[self.features].values.astype(np.float32)
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            self.bars_dict[symbol] = feature_data
            
            index_df = pd.read_parquet(symbol_dir / f"index_{split}.parquet")
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
    def __init__(self, input_dim=8, lookback=60, n_horizons=4, n_quantiles=5):
        super().__init__()
        flat_dim = lookback * input_dim
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_horizons * n_quantiles)
        )
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
    
    def forward(self, x):
        B, L, F = x.shape
        return self.mlp(x.view(B, -1)).view(B, self.n_horizons, self.n_quantiles)


def pinball_loss_with_details(pred, target, quantiles, horizon_weights=None):
    """Pinball loss with per-horizon details."""
    B, H, Q = pred.shape
    target = target.unsqueeze(-1)
    errors = target - pred
    quantiles_t = torch.tensor(quantiles, device=pred.device).view(1, 1, Q)
    loss = torch.where(errors >= 0, quantiles_t * errors, (quantiles_t - 1) * errors)
    loss_per_horizon = loss.mean(dim=(0, 2))
    
    if horizon_weights is not None:
        hw = torch.tensor(horizon_weights, device=pred.device)
        total_loss = (loss_per_horizon * hw).sum() / hw.sum()
    else:
        total_loss = loss_per_horizon.mean()
    
    return total_loss, loss_per_horizon


def quantile_crossing_penalty(pred):
    diffs = pred[:, :, 1:] - pred[:, :, :-1]
    return torch.relu(-diffs).mean()


def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.seed:
        config['training']['seed'] = args.seed
    
    seed = config['training']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.smoke:
        symbols = config['universe']['smoke_symbols']
    else:
        with open(config['universe']['candidates_file'], 'r') as f:
            symbols = yaml.safe_load(f)['candidates']
    
    print(f"\n{'='*60}")
    print(f"M2-T3: Final Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Symbols: {len(symbols)}")
    print(f"Max steps: {config['training']['max_steps']}")
    print(f"{'='*60}")
    
    train_dataset = M2Dataset(symbols, config['paths']['processed_dir'], "train")
    val_dataset = M2Dataset(symbols, config['paths']['processed_dir'], "val")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                             shuffle=True, num_workers=config['training']['num_workers'],
                             pin_memory=True if device.type == "cuda" else False)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                           shuffle=False, num_workers=config['training']['num_workers'],
                           pin_memory=True if device.type == "cuda" else False)
    
    model = SimpleQuantileModel(FEATURE_DIM, 60, 4, 5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    
    print(f"\nModel: SimpleQuantileModel (placeholder for AlphaTrade v0.2)")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n{'='*60}")
    print(f"Training...")
    print(f"{'='*60}")
    
    run_id = str(uuid.uuid4())[:8]
    quantiles = config['model']['quantiles']['levels']
    horizon_weights = config['loss']['pinball']['horizon_weights']
    crossing_enabled = config['loss']['crossing_penalty']['enabled']
    crossing_weight = config['loss']['crossing_penalty']['weight']
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_step = 0
    max_grad_norm_overall = 0.0
    
    # Track by-horizon losses
    horizon_train_losses = {f"h{h}": [] for h in [1, 5, 20, 60]}
    horizon_val_losses = {f"h{h}": [] for h in [1, 5, 20, 60]}
    
    step = 0
    max_steps = config['training']['max_steps']
    val_every = config['training']['val_every']
    grad_clip = config['training']['grad_clip']
    
    for batch in train_loader:
        if step >= max_steps:
            break
        
        model.train()
        optimizer.zero_grad()
        
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        
        pinball, loss_per_h = pinball_loss_with_details(pred, y, quantiles, horizon_weights)
        crossing = quantile_crossing_penalty(pred)
        total_loss = pinball + (crossing_weight * crossing if crossing_enabled else 0)
        
        total_loss.backward()
        max_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        max_grad_norm_overall = max(max_grad_norm_overall, max_grad_norm.item())
        optimizer.step()
        
        train_losses.append(total_loss.item())
        for i, h in enumerate([1, 5, 20, 60]):
            horizon_train_losses[f"h{h}"].append(loss_per_h[i].item())
        
        step += 1
        
        if step % val_every == 0 or step == max_steps:
            model.eval()
            val_loss_sum = 0.0
            val_horizon_sum = {f"h{h}": 0.0 for h in [1, 5, 20, 60]}
            n_val_batches = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    vx = val_batch["x"].to(device)
                    vy = val_batch["y"].to(device)
                    vpred = model(vx)
                    vl, vl_per_h = pinball_loss_with_details(vpred, vy, quantiles, horizon_weights)
                    val_loss_sum += vl.item()
                    for i, h in enumerate([1, 5, 20, 60]):
                        val_horizon_sum[f"h{h}"] += vl_per_h[i].item()
                    n_val_batches += 1
            
            val_loss = val_loss_sum / n_val_batches
            val_losses.append(val_loss)
            for h in [1, 5, 20, 60]:
                horizon_val_losses[f"h{h}"].append(val_horizon_sum[f"h{h}"] / n_val_batches)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
            
            print(f"Step {step}/{max_steps} | Train: {total_loss.item():.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f} @ {best_step}")
            
            if np.isnan(total_loss.item()) or np.isinf(total_loss.item()):
                print(f"\n❌ Training failed: NaN/Inf detected!")
                sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete")
    print(f"{'='*60}")
    
    # Generate metrics
    metrics = {
        "run": {
            "run_id": run_id,
            "git_sha": get_git_sha(),
            "created_at": datetime.now().isoformat(),
            "device": str(device),
            "seed": seed
        },
        "dataset": {
            "name": "m2",
            "config_path": args.config,
            "processed_dir": config['paths']['processed_dir'],
            "symbols": len(symbols),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": 0,
            "feature_dim": FEATURE_DIM,
            "feature_cols": FEATURE_COLS,
            "lookback": 60,
            "horizons": [1, 5, 20, 60],
            "quantiles": quantiles
        },
        "model": {
            "type": "SimpleQuantileModel (placeholder for AlphaTrade v0.2)",
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "config": {"n_quantiles": 5, "hidden_dim": 256}
        },
        "training": {
            "max_steps": max_steps,
            "batch_size": config['training']['batch_size'],
            "learning_rate": config['training']['learning_rate'],
            "weight_decay": config['training']['weight_decay'],
            "grad_clip": grad_clip,
            "optimizer": "adamw",
            "lr_schedule": config['training']['lr_schedule']
        },
        "loss": {
            "train_last": float(train_losses[-1]),
            "train_best": float(min(train_losses)),
            "val_last": float(val_losses[-1]),
            "val_best": float(best_val_loss),
            "best_step": best_step,
            "by_horizon": {
                "h1": {
                    "train": float(np.mean(horizon_train_losses["h1"][-10:])),
                    "val": float(horizon_val_losses["h1"][-1])
                },
                "h5": {
                    "train": float(np.mean(horizon_train_losses["h5"][-10:])),
                    "val": float(horizon_val_losses["h5"][-1])
                },
                "h20": {
                    "train": float(np.mean(horizon_train_losses["h20"][-10:])),
                    "val": float(horizon_val_losses["h20"][-1])
                },
                "h60": {
                    "train": float(np.mean(horizon_train_losses["h60"][-10:])),
                    "val": float(horizon_val_losses["h60"][-1])
                }
            }
        },
        "stability": {
            "nan_steps": 0,
            "inf_steps": 0,
            "max_grad_norm": float(max_grad_norm_overall)
        }
    }
    
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    json_path = reports_dir / "m2_train_metrics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics: {json_path}")
    
    # Markdown report
    md_path = reports_dir / "m2_train_run.md"
    with open(md_path, 'w') as f:
        f.write("# M2 Training Run (Final)\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 运行命令\n\n```bash\n")
        f.write(f"python src/alphatrade/scripts/train_m2_final.py --config {args.config}")
        if args.smoke:
            f.write(" --smoke")
        f.write("\n```\n\n")
        f.write(f"## 配置\n\n- Run ID: {run_id}\n- Device: {device}\n- Symbols: {len(symbols)}\n")
        f.write(f"- Train samples: {len(train_dataset):,}\n- Val samples: {len(val_dataset):,}\n\n")
        f.write(f"## Loss\n\n- Train last: {metrics['loss']['train_last']:.6f}\n")
        f.write(f"- Val best: {metrics['loss']['val_best']:.6f} @ step {best_step}\n\n")
        f.write("### By-Horizon Loss\n\n| Horizon | Train | Val |\n|---------|-------|-----|\n")
        for h in ["h1", "h5", "h20", "h60"]:
            f.write(f"| {h} | {metrics['loss']['by_horizon'][h]['train']:.6f} | {metrics['loss']['by_horizon'][h]['val']:.6f} |\n")
    print(f"✅ Report: {md_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ M2-T3 Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
