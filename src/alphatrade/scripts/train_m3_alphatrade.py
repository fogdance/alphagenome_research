#!/usr/bin/env python3
"""
M3 Training Script - AlphaTrade v0.2 (JAX)

Integrates real AlphaTrade v0.2 model into training pipeline.
Reuses M2 dataloader and reports schema.
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphatrade.core import losses as loss_lib
from alphatrade.core import model as model_lib
from alphatrade.core import schemas
from alphatrade.training import training
from alphatrade import runtime_paths
from data_pipeline.feature_schema import FEATURE_COLS, FEATURE_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="M3 AlphaTrade v0.2 Training")
    parser.add_argument("--config", type=str, default="configs/dataset/m2.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jit", type=int, default=1, help="Use JIT compilation (0/1)")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_git_sha() -> str:
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


class M3Dataset:
    """M3 dataset - loads M2 data and converts to JAX arrays."""

    def __init__(self, symbols: List[str], processed_root: str, split: str = "train"):
        self.symbols = symbols
        self.processed_root = processed_root
        self.split = split
        self.features = FEATURE_COLS

        # Load bars and indices
        self.bars_dict = {}
        self.indices = []

        print(f"\nLoading {split} data...")
        for symbol in symbols:
            symbol_dir = Path(processed_root) / symbol

            # Load bars
            bars_path = symbol_dir / "bars.parquet"
            if not bars_path.exists():
                print(f"  ⚠️  {symbol}: bars.parquet not found, skipping")
                continue

            bars_df = pd.read_parquet(bars_path)

            # Extract features and handle NaN
            feature_data = bars_df[self.features].values.astype(np.float32)
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            self.bars_dict[symbol] = feature_data

            # Load index
            index_path = symbol_dir / f"index_{split}.parquet"
            if not index_path.exists():
                print(f"  ⚠️  {symbol}: index_{split}.parquet not found, skipping")
                continue

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

    def get_batch(self, indices: List[int]):
        """Get a batch of samples as JAX arrays."""
        batch_x = []
        batch_y = []

        for idx in indices:
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

            batch_x.append(x)
            batch_y.append(y)

        # Stack to batch
        batch_x = np.stack(batch_x, axis=0)  # [B, L, F]
        batch_y = np.stack(batch_y, axis=0)  # [B, H]

        return jnp.array(batch_x), jnp.array(batch_y)


def create_batches(dataset: M3Dataset, batch_size: int, shuffle: bool = False):
    """Create batches from dataset."""
    indices = list(range(len(dataset)))

    if shuffle:
        np.random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        yield dataset.get_batch(batch_indices)


def make_train_step(config: schemas.AlphaTradeConfig, optimizer, use_jit: bool = True):
    """Create training step function."""

    quantiles_array = jnp.array(config.quantiles, dtype=jnp.float32)

    def train_step_fn(params, state, opt_state, rng, xb, yb):
        """Single training step."""

        def loss_fn(params):
            # Forward pass
            def forward(x):
                model = model_lib.AlphaTrade(config)
                return model(x)

            import haiku as hk
            forward_t = hk.transform_with_state(lambda x: forward(x))
            output, new_state = forward_t.apply(params, state, rng, xb)

            # Compute loss for each horizon
            total_loss = jnp.zeros((), dtype=jnp.float32)
            loss_dict = {}

            for i, horizon in enumerate(config.horizons):
                y_true = yb[:, i]  # [B]
                y_pred = output.log_return_quantiles[horizon]  # [B, Q]

                # Pinball loss
                pinball = loss_lib.quantile_pinball_loss(y_true, y_pred, quantiles_array)

                # Crossing penalty
                crossing = loss_lib.quantile_crossing_penalty(y_pred)

                horizon_loss = pinball + 0.1 * crossing
                total_loss += horizon_loss

                loss_dict[f'h{horizon}_pinball'] = pinball
                loss_dict[f'h{horizon}_crossing'] = crossing
                loss_dict[f'h{horizon}_total'] = horizon_loss

            loss_dict['total_loss'] = total_loss

            return total_loss, (new_state, loss_dict)

        # Compute gradients
        (loss, (new_state, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Compute gradient norm
        grad_norm = optax.global_norm(grads)

        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        metrics['grad_norm'] = grad_norm
        metrics['loss'] = loss

        return new_params, new_state, new_opt_state, metrics

    if use_jit:
        return jax.jit(train_step_fn)
    else:
        return train_step_fn


def make_eval_step(config: schemas.AlphaTradeConfig, use_jit: bool = True):
    """Create evaluation step function."""

    quantiles_array = jnp.array(config.quantiles, dtype=jnp.float32)

    def eval_step_fn(params, state, rng, xb, yb):
        """Single evaluation step."""

        # Forward pass
        def forward(x):
            model = model_lib.AlphaTrade(config)
            return model(x)

        import haiku as hk
        forward_t = hk.transform_with_state(lambda x: forward(x))
        output, _ = forward_t.apply(params, state, rng, xb)

        # Compute loss for each horizon
        total_loss = jnp.zeros((), dtype=jnp.float32)
        loss_dict = {}

        for i, horizon in enumerate(config.horizons):
            y_true = yb[:, i]  # [B]
            y_pred = output.log_return_quantiles[horizon]  # [B, Q]

            # Pinball loss
            pinball = loss_lib.quantile_pinball_loss(y_true, y_pred, quantiles_array)

            # Crossing penalty
            crossing = loss_lib.quantile_crossing_penalty(y_pred)

            horizon_loss = pinball + 0.1 * crossing
            total_loss += horizon_loss

            loss_dict[f'h{horizon}_pinball'] = pinball
            loss_dict[f'h{horizon}_crossing'] = crossing
            loss_dict[f'h{horizon}_total'] = horizon_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict

    if use_jit:
        return jax.jit(eval_step_fn)
    else:
        return eval_step_fn


def main():
    args = parse_args()
    config_dict = load_config(args.config)

    # Override config
    if args.max_steps:
        config_dict['training']['max_steps'] = args.max_steps
    if args.batch_size:
        config_dict['training']['batch_size'] = args.batch_size
    if args.seed:
        config_dict['training']['seed'] = args.seed

    seed = config_dict['training']['seed']
    np.random.seed(seed)

    # Select symbols
    if args.smoke:
        symbols = config_dict['universe']['smoke_symbols']
    else:
        with open(config_dict['universe']['candidates_file'], 'r') as f:
            symbols = yaml.safe_load(f)['candidates']

    print(f"\n{'='*60}")
    print(f"M3: AlphaTrade v0.2 Training")
    print(f"{'='*60}")
    print(f"Backend: JAX")
    print(f"JIT: {'enabled' if args.jit else 'disabled'}")
    print(f"Symbols: {len(symbols)}")
    print(f"Max steps: {config_dict['training']['max_steps']}")
    print(f"Batch size: {config_dict['training']['batch_size']}")
    print(f"Clip norm: {args.clip_norm}")
    print(f"{'='*60}")

    # Load datasets
    train_dataset = M3Dataset(symbols, config_dict['paths']['processed_dir'], "train")
    val_dataset = M3Dataset(symbols, config_dict['paths']['processed_dir'], "val")

    # Create AlphaTrade config
    # Note: lookback=60 is too short for 6 stages (requires 64x downsample)
    # Use 2 stages for 4x downsample (60/4 = 15 sequence length)
    alphatrade_config = schemas.AlphaTradeConfig(
        lookback_length=60,
        num_features=8,
        horizons=[1, 5, 20, 60],
        quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
        stem_channels=128,
        num_encoder_stages=2,  # 2 stages = 4x downsample (60/4=15)
        channel_increment=64,
        d_model=256,  # Reduced for shorter sequences
        num_transformer_layers=4,  # Reduced for shorter sequences
    )

    print(f"\nAlphaTrade v0.2 Config:")
    print(f"  Lookback: {alphatrade_config.lookback_length}")
    print(f"  Features: {alphatrade_config.num_features}")
    print(f"  Horizons: {alphatrade_config.horizons}")
    print(f"  Quantiles: {alphatrade_config.quantiles}")
    print(f"  d_model: {alphatrade_config.d_model}")
    print(f"  Transformer layers: {alphatrade_config.num_transformer_layers}")

    # Initialize model
    print(f"\nInitializing model...")
    rng = jax.random.PRNGKey(seed)

    # Create dummy batch for initialization
    dummy_x = jnp.zeros((1, alphatrade_config.lookback_length, alphatrade_config.num_features), dtype=jnp.float32)

    # Initialize parameters
    def forward(x):
        model = model_lib.AlphaTrade(alphatrade_config)
        return model(x)

    import haiku as hk
    forward_t = hk.transform_with_state(lambda x: forward(x))
    params, state = forward_t.init(rng, dummy_x)

    # Count parameters
    def count_params(pytree):
        return sum(x.size for x in jax.tree_util.tree_leaves(pytree))

    total_params = count_params(params)
    print(f"  ✓ Model initialized")
    print(f"    Total params: {total_params:,}")

    # Create optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.adamw(
            learning_rate=config_dict['training']['learning_rate'],
            weight_decay=config_dict['training']['weight_decay']
        )
    )
    opt_state = optimizer.init(params)

    # Create train/eval steps
    use_jit = bool(args.jit)
    train_step = make_train_step(alphatrade_config, optimizer, use_jit)
    eval_step = make_eval_step(alphatrade_config, use_jit)

    print(f"\n{'='*60}")
    print(f"Training...")
    print(f"{'='*60}")

    run_id = str(uuid.uuid4())[:8]
    max_steps = config_dict['training']['max_steps']
    batch_size = config_dict['training']['batch_size']
    val_every = config_dict['training']['val_every']

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_step = 0
    max_grad_norm_overall = 0.0
    nan_steps = 0
    inf_steps = 0

    # Track by-horizon losses
    horizon_train_losses = {f"h{h}": [] for h in [1, 5, 20, 60]}
    horizon_val_losses = {f"h{h}": [] for h in [1, 5, 20, 60]}

    step = 0

    while step < max_steps:
        # Training epoch
        for xb, yb in create_batches(train_dataset, batch_size, shuffle=True):
            if step >= max_steps:
                break

            # Training step
            step_rng = jax.random.fold_in(rng, step)
            params, state, opt_state, metrics = train_step(params, state, opt_state, step_rng, xb, yb)

            # Extract metrics
            loss = float(metrics['loss'])
            grad_norm = float(metrics['grad_norm'])

            # Check for NaN/Inf
            if np.isnan(loss):
                nan_steps += 1
                print(f"  ⚠️  Step {step}: NaN detected!")
            if np.isinf(loss):
                inf_steps += 1
                print(f"  ⚠️  Step {step}: Inf detected!")

            max_grad_norm_overall = max(max_grad_norm_overall, grad_norm)

            train_losses.append(loss)
            for h in [1, 5, 20, 60]:
                horizon_train_losses[f"h{h}"].append(float(metrics[f'h{h}_total']))

            step += 1

            # Validation
            if step % val_every == 0 or step == max_steps:
                val_loss_sum = 0.0
                val_horizon_sum = {f"h{h}": 0.0 for h in [1, 5, 20, 60]}
                n_val_batches = 0

                for vx, vy in create_batches(val_dataset, batch_size, shuffle=False):
                    val_rng = jax.random.fold_in(rng, step + 1000000)
                    val_metrics = eval_step(params, state, val_rng, vx, vy)

                    val_loss_sum += float(val_metrics['total_loss'])
                    for h in [1, 5, 20, 60]:
                        val_horizon_sum[f"h{h}"] += float(val_metrics[f'h{h}_total'])
                    n_val_batches += 1

                val_loss = val_loss_sum / n_val_batches
                val_losses.append(val_loss)
                for h in [1, 5, 20, 60]:
                    horizon_val_losses[f"h{h}"].append(val_horizon_sum[f"h{h}"] / n_val_batches)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step

                print(f"Step {step}/{max_steps} | Train: {loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f} @ {best_step} | Grad: {grad_norm:.4f}")

    print(f"\n{'='*60}")
    print(f"✅ Training Complete")
    print(f"{'='*60}")

    # Generate metrics (M2 schema compatible)
    metrics_output = {
        "run": {
            "run_id": run_id,
            "git_sha": get_git_sha(),
            "created_at": datetime.now().isoformat(),
            "device": "cuda" if jax.devices()[0].platform == "gpu" else "cpu",
            "seed": seed
        },
        "dataset": {
            "name": "m2",
            "config_path": args.config,
            "processed_dir": config_dict['paths']['processed_dir'],
            "symbols": len(symbols),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": 0,
            "feature_dim": FEATURE_DIM,
            "feature_cols": FEATURE_COLS,
            "lookback": 60,
            "horizons": [1, 5, 20, 60],
            "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        "model": {
            "type": "alphatrade_v0.2",
            "backend": "jax",
            "total_params": int(total_params),
            "trainable_params": int(total_params),
            "config": {
                "lookback_length": alphatrade_config.lookback_length,
                "num_features": alphatrade_config.num_features,
                "stem_channels": alphatrade_config.stem_channels,
                "num_encoder_stages": alphatrade_config.num_encoder_stages,
                "d_model": alphatrade_config.d_model,
                "num_transformer_layers": alphatrade_config.num_transformer_layers,
                "horizons": alphatrade_config.horizons,
                "quantiles": alphatrade_config.quantiles
            }
        },
        "training": {
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": config_dict['training']['learning_rate'],
            "weight_decay": config_dict['training']['weight_decay'],
            "grad_clip": args.clip_norm,
            "optimizer": "adamw",
            "compile_jit": use_jit,
            "lr_schedule": config_dict['training']['lr_schedule']
        },
        "loss": {
            "train_last": float(train_losses[-1]) if train_losses else 0.0,
            "train_best": float(min(train_losses)) if train_losses else 0.0,
            "val_last": float(val_losses[-1]) if val_losses else 0.0,
            "val_best": float(best_val_loss),
            "best_step": best_step,
            "by_horizon": {
                "h1": {
                    "train": float(np.mean(horizon_train_losses["h1"][-10:])) if horizon_train_losses["h1"] else 0.0,
                    "val": float(horizon_val_losses["h1"][-1]) if horizon_val_losses["h1"] else 0.0
                },
                "h5": {
                    "train": float(np.mean(horizon_train_losses["h5"][-10:])) if horizon_train_losses["h5"] else 0.0,
                    "val": float(horizon_val_losses["h5"][-1]) if horizon_val_losses["h5"] else 0.0
                },
                "h20": {
                    "train": float(np.mean(horizon_train_losses["h20"][-10:])) if horizon_train_losses["h20"] else 0.0,
                    "val": float(horizon_val_losses["h20"][-1]) if horizon_val_losses["h20"] else 0.0
                },
                "h60": {
                    "train": float(np.mean(horizon_train_losses["h60"][-10:])) if horizon_train_losses["h60"] else 0.0,
                    "val": float(horizon_val_losses["h60"][-1]) if horizon_val_losses["h60"] else 0.0
                }
            }
        },
        "stability": {
            "nan_steps": nan_steps,
            "inf_steps": inf_steps,
            "max_grad_norm": float(max_grad_norm_overall),
            "oom_count": 0
        }
    }

    # Save metrics JSON
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    json_path = reports_dir / "m3_train_metrics.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"\n✅ Metrics: {json_path}")

    # Save markdown report
    md_path = reports_dir / "m3_train_run.md"
    with open(md_path, 'w') as f:
        f.write("# M3 Training Run - AlphaTrade v0.2 (JAX)\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 运行命令\n\n```bash\n")
        f.write(f"python src/alphatrade/scripts/train_m3_alphatrade.py")
        f.write(f" --config {args.config}")
        f.write(f" --max-steps {max_steps}")
        f.write(f" --batch-size {batch_size}")
        f.write(f" --clip-norm {args.clip_norm}")
        f.write(f" --jit {args.jit}")
        if args.smoke:
            f.write(" --smoke")
        f.write("\n```\n\n")

        f.write(f"## 配置\n\n")
        f.write(f"- Run ID: {run_id}\n")
        f.write(f"- Backend: JAX\n")
        f.write(f"- JIT: {'enabled' if use_jit else 'disabled'}\n")
        f.write(f"- Device: {metrics_output['run']['device']}\n")
        f.write(f"- Symbols: {len(symbols)}\n")
        f.write(f"- Train samples: {len(train_dataset):,}\n")
        f.write(f"- Val samples: {len(val_dataset):,}\n\n")

        f.write(f"## 模型\n\n")
        f.write(f"- Type: AlphaTrade v0.2\n")
        f.write(f"- Backend: JAX\n")
        f.write(f"- Total params: {total_params:,}\n")
        f.write(f"- d_model: {alphatrade_config.d_model}\n")
        f.write(f"- Transformer layers: {alphatrade_config.num_transformer_layers}\n\n")

        f.write(f"## Loss\n\n")
        f.write(f"- Train last: {metrics_output['loss']['train_last']:.6f}\n")
        f.write(f"- Train best: {metrics_output['loss']['train_best']:.6f}\n")
        f.write(f"- Val last: {metrics_output['loss']['val_last']:.6f}\n")
        f.write(f"- Val best: {metrics_output['loss']['val_best']:.6f} @ step {best_step}\n\n")

        f.write("### By-Horizon Loss\n\n")
        f.write("| Horizon | Train | Val |\n")
        f.write("|---------|-------|-----|\n")
        for h in ["h1", "h5", "h20", "h60"]:
            train_val = metrics_output['loss']['by_horizon'][h]['train']
            val_val = metrics_output['loss']['by_horizon'][h]['val']
            f.write(f"| {h} | {train_val:.6f} | {val_val:.6f} |\n")

        f.write(f"\n## Stability\n\n")
        f.write(f"- NaN steps: {nan_steps}\n")
        f.write(f"- Inf steps: {inf_steps}\n")
        f.write(f"- Max grad norm: {max_grad_norm_overall:.4f}\n")
        f.write(f"- OOM count: 0\n")

    print(f"✅ Report: {md_path}")

    print(f"\n{'='*60}")
    print(f"✅ M3 Training Complete")
    print(f"{'='*60}")
    print(f"\nNext: Validate schema")
    print(f"  python src/alphatrade/scripts/validate_reports_schema.py \\")
    print(f"    --reports-dir {reports_dir} --schemas-dir src/alphatrade/schemas")


if __name__ == "__main__":
    main()
