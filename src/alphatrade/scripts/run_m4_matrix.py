#!/usr/bin/env python3
"""
M4 Matrix Runner

Runs multiple training experiments with different seeds/configs and generates comparison summary.

Usage:
    python src/alphatrade/scripts/run_m4_matrix.py \
        --config configs/dataset/m2.yaml \
        --seeds 42 43 44 \
        --max-steps 1000 \
        --smoke
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="M4 Matrix Runner")
    parser.add_argument("--config", type=str, default="configs/dataset/m2.yaml",
                        help="Dataset config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds to run (default: 42 43 44)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max training steps (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--smoke", action="store_true",
                        help="Use smoke test symbols")
    parser.add_argument("--jit", type=int, default=1,
                        help="Use JIT compilation (0/1, default: 1)")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation after training")
    parser.add_argument("--eval-split", type=str, default="val",
                        help="Evaluation split (default: val)")
    return parser.parse_args()


def run_training(config: str, seed: int, max_steps: int, batch_size: int,
                 smoke: bool, jit: int) -> Dict:
    """Run single training experiment."""

    print(f"\n{'='*60}")
    print(f"Training with seed={seed}")
    print(f"{'='*60}\n")

    cmd = [
        "python", "src/alphatrade/scripts/train_m4_alphatrade.py",
        "--config", config,
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--batch-size", str(batch_size),
        "--jit", str(jit)
    ]

    if smoke:
        cmd.append("--smoke")

    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  ⚠️  Training failed for seed={seed}")
        return None

    # Load metrics
    metrics_path = "reports/m4_train_metrics.json"
    if not os.path.exists(metrics_path):
        print(f"  ⚠️  Metrics file not found: {metrics_path}")
        return None

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Rename metrics file to include seed
    seed_metrics_path = f"reports/m4_train_metrics_seed{seed}.json"
    os.rename(metrics_path, seed_metrics_path)

    # Rename markdown report
    md_path = "reports/m4_train_run.md"
    if os.path.exists(md_path):
        seed_md_path = f"reports/m4_train_run_seed{seed}.md"
        os.rename(md_path, seed_md_path)

    print(f"  ✅ Training complete: {seed_metrics_path}")

    return {
        "seed": seed,
        "metrics": metrics,
        "metrics_path": seed_metrics_path
    }


def run_evaluation(config: str, seed: int, split: str) -> Dict:
    """Run evaluation for a trained model."""

    print(f"\n{'='*60}")
    print(f"Evaluating seed={seed}")
    print(f"{'='*60}\n")

    train_metrics_path = f"reports/m4_train_metrics_seed{seed}.json"

    cmd = [
        "python", "src/alphatrade/scripts/eval_m4_fast.py",
        "--train-metrics", train_metrics_path,
        "--dataset-config", config,
        "--split", split,
        "--batch-size", "128"
    ]

    # Run evaluation
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"  ⚠️  Evaluation failed for seed={seed}")
        return None

    # Load eval metrics
    eval_metrics_path = "reports/m4_eval_metrics_fast.json"
    if not os.path.exists(eval_metrics_path):
        print(f"  ⚠️  Eval metrics file not found: {eval_metrics_path}")
        return None

    with open(eval_metrics_path, 'r') as f:
        eval_metrics = json.load(f)

    # Rename eval metrics file to include seed
    seed_eval_path = f"reports/m4_eval_metrics_seed{seed}.json"
    os.rename(eval_metrics_path, seed_eval_path)

    # Rename eval markdown report
    eval_md_path = "reports/m4_eval_run_fast.md"
    if os.path.exists(eval_md_path):
        seed_eval_md_path = f"reports/m4_eval_run_seed{seed}.md"
        os.rename(eval_md_path, seed_eval_md_path)

    print(f"  ✅ Evaluation complete: {seed_eval_path}")

    return {
        "seed": seed,
        "eval_metrics": eval_metrics,
        "eval_path": seed_eval_path
    }


def generate_summary(results: List[Dict], eval_results: List[Dict], args) -> str:
    """Generate matrix summary markdown."""

    summary = []
    summary.append("# M4 Matrix Summary\n")
    summary.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary.append("---\n\n")

    # Configuration
    summary.append("## 配置\n\n")
    summary.append(f"- Dataset config: {args.config}\n")
    summary.append(f"- Seeds: {args.seeds}\n")
    summary.append(f"- Max steps: {args.max_steps}\n")
    summary.append(f"- Batch size: {args.batch_size}\n")
    summary.append(f"- JIT: {'enabled' if args.jit else 'disabled'}\n")
    summary.append(f"- Smoke test: {'yes' if args.smoke else 'no'}\n")
    if eval_results:
        summary.append(f"- Evaluation: yes (split: {args.eval_split})\n")
    summary.append("\n")

    # Training results
    summary.append("## 训练结果对比\n\n")
    summary.append("| Seed | Train Loss | Val Loss | Best Step | NaN Steps | Inf Steps | Grad Norm (pre/post) |\n")
    summary.append("|------|------------|----------|-----------|-----------|-----------|----------------------|\n")

    for r in results:
        if r is None:
            continue

        m = r['metrics']
        seed = r['seed']
        train_loss = m['loss']['train_last']
        val_loss = m['loss']['val_last']
        best_step = m['loss']['best_step']
        nan_steps = m['stability']['nan_steps']
        inf_steps = m['stability']['inf_steps']
        grad_pre = m['stability']['grad_norm_pre_clip_max']
        grad_post = m['stability']['grad_norm_post_clip_max']

        summary.append(f"| {seed} | {train_loss:.6f} | {val_loss:.6f} | {best_step} | {nan_steps} | {inf_steps} | {grad_pre:.4f}/{grad_post:.4f} |\n")

    summary.append("\n")

    # By-horizon training loss
    summary.append("### By-Horizon 训练损失\n\n")
    summary.append("| Seed | h1 | h5 | h20 | h60 |\n")
    summary.append("|------|----|----|-----|-----|\n")

    for r in results:
        if r is None:
            continue

        m = r['metrics']
        seed = r['seed']
        h1 = m['loss']['by_horizon']['h1']['train']
        h5 = m['loss']['by_horizon']['h5']['train']
        h20 = m['loss']['by_horizon']['h20']['train']
        h60 = m['loss']['by_horizon']['h60']['train']

        summary.append(f"| {seed} | {h1:.6f} | {h5:.6f} | {h20:.6f} | {h60:.6f} |\n")

    summary.append("\n")

    # Evaluation results (if available)
    if eval_results:
        summary.append("## 评估结果对比\n\n")
        summary.append("| Seed | Pinball Loss | IC | Rank IC | Quantile Crossing Rate |\n")
        summary.append("|------|--------------|----|---------|-----------------------|\n")

        for r in eval_results:
            if r is None:
                continue

            m = r['eval_metrics']
            seed = r['seed']
            pb_loss = m['pinball_loss']['overall']
            ic = m['ic_metrics']['ic']
            rank_ic = m['ic_metrics']['rank_ic']
            crossing_rate = m['quantile_crossing']['rate']

            summary.append(f"| {seed} | {pb_loss:.6f} | {ic:.4f} | {rank_ic:.4f} | {crossing_rate:.4f} |\n")

        summary.append("\n")

        # By-horizon evaluation
        summary.append("### By-Horizon 评估损失\n\n")
        summary.append("| Seed | h1 | h5 | h20 | h60 |\n")
        summary.append("|------|----|----|-----|-----|\n")

        for r in eval_results:
            if r is None:
                continue

            m = r['eval_metrics']
            seed = r['seed']
            h1 = m['pinball_loss']['by_horizon']['h1']
            h5 = m['pinball_loss']['by_horizon']['h5']
            h20 = m['pinball_loss']['by_horizon']['h20']
            h60 = m['pinball_loss']['by_horizon']['h60']

            summary.append(f"| {seed} | {h1:.6f} | {h5:.6f} | {h20:.6f} | {h60:.6f} |\n")

        summary.append("\n")

    # Statistics
    summary.append("## 统计分析\n\n")

    # Training loss statistics
    train_losses = [r['metrics']['loss']['train_last'] for r in results if r is not None]
    val_losses = [r['metrics']['loss']['val_last'] for r in results if r is not None]

    import numpy as np

    summary.append("### 训练损失统计\n\n")
    summary.append(f"- Train loss: {np.mean(train_losses):.6f} ± {np.std(train_losses):.6f}\n")
    summary.append(f"- Val loss: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}\n")
    summary.append(f"- Best val loss: {np.min(val_losses):.6f} (seed={results[np.argmin(val_losses)]['seed']})\n")
    summary.append("\n")

    if eval_results:
        pb_losses = [r['eval_metrics']['pinball_loss']['overall'] for r in eval_results if r is not None]
        ics = [r['eval_metrics']['ic_metrics']['ic'] for r in eval_results if r is not None]

        summary.append("### 评估指标统计\n\n")
        summary.append(f"- Pinball loss: {np.mean(pb_losses):.6f} ± {np.std(pb_losses):.6f}\n")
        summary.append(f"- IC: {np.mean(ics):.4f} ± {np.std(ics):.4f}\n")
        summary.append(f"- Best IC: {np.max(ics):.4f} (seed={eval_results[np.argmax(ics)]['seed']})\n")
        summary.append("\n")

    # Files
    summary.append("## 生成文件\n\n")
    summary.append("### 训练输出\n\n")
    for r in results:
        if r is None:
            continue
        seed = r['seed']
        summary.append(f"- `reports/m4_train_metrics_seed{seed}.json`\n")
        summary.append(f"- `reports/m4_train_run_seed{seed}.md`\n")

    if eval_results:
        summary.append("\n### 评估输出\n\n")
        for r in eval_results:
            if r is None:
                continue
            seed = r['seed']
            summary.append(f"- `reports/m4_eval_metrics_seed{seed}.json`\n")
            summary.append(f"- `reports/m4_eval_run_seed{seed}.md`\n")

    summary.append("\n---\n")
    summary.append(f"\n**总计**: {len([r for r in results if r is not None])} 个训练实验")
    if eval_results:
        summary.append(f", {len([r for r in eval_results if r is not None])} 个评估实验")
    summary.append("\n")

    return "".join(summary)


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"M4 Matrix Runner")
    print(f"{'='*60}")
    print(f"Seeds: {args.seeds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"JIT: {'enabled' if args.jit else 'disabled'}")
    print(f"Evaluation: {'yes' if args.eval else 'no'}")
    print(f"{'='*60}\n")

    # Run training for all seeds
    results = []
    for seed in args.seeds:
        result = run_training(
            args.config,
            seed,
            args.max_steps,
            args.batch_size,
            args.smoke,
            args.jit
        )
        results.append(result)

    # Run evaluation if requested
    eval_results = []
    if args.eval:
        for seed in args.seeds:
            eval_result = run_evaluation(args.config, seed, args.eval_split)
            eval_results.append(eval_result)

    # Generate summary
    summary_md = generate_summary(results, eval_results, args)

    # Save summary
    summary_path = "reports/m4_matrix_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary_md)

    print(f"\n{'='*60}")
    print(f"✅ Matrix Run Complete")
    print(f"{'='*60}")
    print(f"\nSummary: {summary_path}")
    print(f"Total experiments: {len([r for r in results if r is not None])}")
    if eval_results:
        print(f"Total evaluations: {len([r for r in eval_results if r is not None])}")
    print()


if __name__ == "__main__":
    main()
