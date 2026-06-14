#!/usr/bin/env python3
"""
Generate M6 Baseline Run Report.

Reads M5 sweep artifacts (manifest, leaderboard, per-seed eval files)
and configs to produce a single-page markdown summary in the reports directory.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: pyyaml not installed")
    sys.exit(1)

from alphatrade import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Generate M6 baseline run report")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs (default: ALPHATRADE_RUNS_ROOT or ../alphatrade_runs/default)")
    parser.add_argument("--reports-dir", type=str, default=None, help="Reports directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <reports-dir>/m6_baseline_run.md)")
    parser.add_argument("--sweep-config", type=str, default="configs/sweep/m5.yaml", help="Sweep config path")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset/m2.yaml", help="Dataset config path")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    output = Path(args.output) if args.output else reports_dir / "m6_baseline_run.md"

    # Load inputs
    manifest = load_json(reports_dir / "m5_sweep_manifest.json")
    leaderboard = load_json(reports_dir / "m5_leaderboard.json")
    sweep_cfg = load_yaml(args.sweep_config)
    dataset_cfg = load_yaml(args.dataset_config)

    git_sha = get_git_sha()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract config details
    defaults = sweep_cfg.get("defaults", {})
    sample_idx = dataset_cfg.get("sample_index", {})
    model_cfg = dataset_cfg.get("model", {})
    training_cfg = dataset_cfg.get("training", {})
    features = dataset_cfg.get("features", {})

    # Extract leaderboard experiment (baseline)
    exp = leaderboard["experiments"][0]
    metrics = exp["metrics"]

    # Per-seed details
    seed_rows = []
    for run_info in exp["artifacts"]["runs"]:
        seed = run_info["seed"]
        eval_path = reports_dir / Path(run_info["eval_metrics_path"]).name
        if eval_path.exists():
            eval_data = load_json(eval_path)
            pinball = eval_data.get("pinball_loss", {}).get("overall", "N/A")
            ic = eval_data.get("ic_metrics", {}).get("ic", "N/A")
            rank_ic = eval_data.get("ic_metrics", {}).get("rank_ic", "N/A")
            crossing = eval_data.get("quantile_crossing", {}).get("rate", "N/A")
        else:
            pinball = ic = rank_ic = crossing = "N/A"
        seed_rows.append({
            "seed": seed,
            "run_id": run_info["run_id"],
            "pinball": pinball,
            "ic": ic,
            "rank_ic": rank_ic,
            "crossing": crossing,
        })

    # Build markdown
    lines = []
    w = lines.append

    w("# M6 Baseline Run Report\n")
    w(f"Generated: {now}")
    w(f"git_sha: `{git_sha}`\n")

    w("## Reproduction\n")
    w("```bash")
    w("# Default outputs go to ALPHATRADE_RUNS_ROOT or ../alphatrade_runs/default")
    w("")
    w("# Run 3-seed sweep")
    w("conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \\")
    w("  python src/alphatrade/scripts/run_m5_sweep.py --sweep-config configs/sweep/m5.yaml")
    w("")
    w("# Generate this report")
    w("conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \\")
    w("  python src/alphatrade/scripts/gen_m6_baseline_report.py")
    w("```\n")

    w("## Data\n")
    w(f"- **Universe**: {sweep_cfg.get('universe', 'N/A')} ({len(load_yaml('configs/universe/m1_selected.yaml').get('candidates', []))} symbols)")
    w(f"- **Dataset config**: `{args.dataset_config}`")
    w(f"- **Lookback**: {sample_idx.get('lookback', 'N/A')}")
    w(f"- **Horizons**: {sample_idx.get('horizons', 'N/A')}")
    w(f"- **Quantiles**: {model_cfg.get('quantiles', {}).get('levels', 'N/A')}")
    w(f"- **Train**: {sample_idx.get('train_start', '?')} ~ {sample_idx.get('train_end', '?')}")
    w(f"- **Val**: {sample_idx.get('val_start', '?')} ~ {sample_idx.get('val_end', '?')}")
    w(f"- **Features**: {features.get('dim', '?')}D\n")

    w("## Training Config\n")
    w("| Parameter | Value |")
    w("|-----------|-------|")
    w(f"| max_steps | {defaults.get('max_steps', 'N/A')} |")
    w(f"| batch_size | {defaults.get('batch_size', 'N/A')} |")
    w(f"| jit | {defaults.get('jit', 'N/A')} |")
    w(f"| clip_norm | {defaults.get('clip_norm', 'N/A')} |")
    w(f"| optimizer | {training_cfg.get('optimizer', 'N/A')} |")
    w(f"| learning_rate | {training_cfg.get('learning_rate', 'N/A')} |")
    w(f"| eval_split | {defaults.get('eval_split', 'N/A')} |")
    w(f"| ckpt_step | {defaults.get('ckpt_step', 'N/A')} |")
    w(f"| seeds | {sweep_cfg.get('expected_seeds', 'N/A')} |")
    w("")

    w("## Metrics Summary\n")
    w(f"- **Primary metric**: `{leaderboard.get('primary_metric', 'N/A')}` (lower is better)")
    w(f"- **Mean**: {metrics['primary_mean']:.6f}")
    w(f"- **Std**: {metrics['primary_std']:.6f}")
    w(f"- **Best**: {metrics['primary_best']:.6f} (run_id: {metrics['best_run_id']})\n")

    w("## Per-Seed Results\n")
    w("| Seed | Run ID | Pinball Loss | IC | Rank IC | Crossing Rate |")
    w("|------|--------|--------------|-----|---------|---------------|")
    for r in seed_rows:
        pb = f"{r['pinball']:.6f}" if isinstance(r["pinball"], float) else str(r["pinball"])
        ic = f"{r['ic']:.6f}" if isinstance(r["ic"], float) else str(r["ic"])
        ric = f"{r['rank_ic']:.6f}" if isinstance(r["rank_ic"], float) else str(r["rank_ic"])
        cr = f"{r['crossing']:.4f}" if isinstance(r["crossing"], float) else str(r["crossing"])
        w(f"| {r['seed']} | {r['run_id']} | {pb} | {ic} | {ric} | {cr} |")
    w("")

    w("## Artifacts\n")
    w(f"- Sweep manifest: `{reports_dir}/m5_sweep_manifest.json`")
    w(f"- Leaderboard: `{reports_dir}/m5_leaderboard.json`")
    for run_info in exp["artifacts"]["runs"]:
        w(f"- Seed {run_info['seed']} train: `{run_info['train_metrics_path']}`")
        w(f"- Seed {run_info['seed']} eval: `{run_info['eval_metrics_path']}`")
    w("")

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"M6 baseline report written to: {output}")


if __name__ == "__main__":
    main()
