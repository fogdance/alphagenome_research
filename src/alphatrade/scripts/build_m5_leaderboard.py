#!/usr/bin/env python3
"""
Build M5 Leaderboard from sweep manifest.

Reads m5_sweep_manifest.json, aggregates per-experiment metrics,
and outputs m5_leaderboard.json + m5_leaderboard.md.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

from alphatrade import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Build M5 leaderboard from sweep manifest")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs (default: ALPHATRADE_RUNS_ROOT or ../alphatrade_runs/default)")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory (default: <output-root>/reports)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to m5_sweep_manifest.json")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    return parser.parse_args()


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _extract_primary_metric(eval_data: dict, metric_path: str) -> float | None:
    """Extract a metric value from eval data using dot-separated path."""
    keys = metric_path.split(".")
    obj = eval_data
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k)
        else:
            return None
    if isinstance(obj, (int, float)):
        return float(obj)
    return None


def build_leaderboard(manifest: dict) -> dict:
    """Build leaderboard dict from manifest data."""
    expected_seeds = manifest["expected_seeds"]
    primary_metric = manifest["primary_metric"]

    # Group runs by exp_id
    exp_runs: dict[str, list[dict]] = defaultdict(list)
    for run in manifest["runs"]:
        exp_runs[run["exp_id"]].append(run)

    experiments = []
    for exp_id, runs in exp_runs.items():
        config_hashes = {r["config_hash"] for r in runs}
        config_hash = runs[0]["config_hash"]
        dataset_configs = sorted({r.get("dataset_config", "") for r in runs})
        universes = sorted({r.get("universe", "") for r in runs})
        eval_splits = sorted({r.get("eval_split", "") for r in runs})
        ckpt_steps = sorted({str(r.get("ckpt_step", "")) for r in runs})

        seeds_done = sorted({r["seed"] for r in runs})
        seeds_missing = sorted(set(expected_seeds) - set(seeds_done))

        # Extract primary metric from each run's eval file
        metric_values = {}  # run_id -> value
        for run in runs:
            eval_path = run["eval_metrics_path"]
            if not os.path.exists(eval_path):
                continue
            try:
                with open(eval_path) as f:
                    eval_data = json.load(f)
                val = _extract_primary_metric(eval_data, primary_metric)
                if val is not None:
                    metric_values[run["run_id"]] = val
            except (json.JSONDecodeError, OSError):
                continue

        if metric_values:
            values = list(metric_values.values())
            mean_val = sum(values) / len(values)
            # std dev
            if len(values) > 1:
                variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
                std_val = variance ** 0.5
            else:
                std_val = 0.0
            best_val = min(values)  # pinball_loss: lower is better
            best_run_id = min(metric_values, key=metric_values.get)
        else:
            mean_val = float("nan")
            std_val = float("nan")
            best_val = float("nan")
            best_run_id = ""

        regression_flags = []
        if len(config_hashes) > 1:
            regression_flags.append(f"config_hash_mismatch: {config_hashes}")
        if seeds_missing:
            regression_flags.append(f"missing_seeds: {seeds_missing}")

        experiments.append({
            "exp_id": exp_id,
            "config_hash": config_hash,
            "dataset_configs": dataset_configs,
            "universes": universes,
            "eval_splits": eval_splits,
            "ckpt_steps": ckpt_steps,
            "seeds_done": seeds_done,
            "seeds_missing": seeds_missing,
            "n_runs": len(runs),
            "metrics": {
                "primary_mean": mean_val,
                "primary_std": std_val,
                "primary_best": best_val,
                "best_run_id": best_run_id,
            },
            "artifacts": {
                "runs": [
                    {
                        "seed": r["seed"],
                        "run_id": r["run_id"],
                        "dataset_config": r.get("dataset_config", ""),
                        "universe": r.get("universe", ""),
                        "eval_split": r.get("eval_split", ""),
                        "ckpt_step": r.get("ckpt_step", ""),
                        "train_metrics_path": r["train_metrics_path"],
                        "eval_metrics_path": r["eval_metrics_path"],
                    }
                    for r in sorted(runs, key=lambda r: r["seed"])
                ]
            },
            "regression_flags": regression_flags,
        })

    # Sort by primary_mean ascending (lower is better for pinball_loss)
    experiments.sort(key=lambda e: e["metrics"]["primary_mean"])

    return {
        "schema_version": "m5_leaderboard_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": manifest.get("git_sha", _get_git_sha()),
        "primary_metric": primary_metric,
        "expected_seeds": expected_seeds,
        "experiments": experiments,
    }


def generate_md(leaderboard: dict) -> str:
    """Generate human-readable markdown table."""
    lines = []
    w = lines.append

    w("# M5 Leaderboard\n")
    w(f"Generated: {leaderboard['generated_at']}")
    w(f"Primary metric: `{leaderboard['primary_metric']}` (lower is better)")
    w(f"Expected seeds: {leaderboard['expected_seeds']}\n")

    w("## Rankings\n")
    w("| Rank | Experiment | Mean | Std | Best | Best Run | Seeds | Flags |")
    w("|------|-----------|------|-----|------|----------|-------|-------|")

    for i, exp in enumerate(leaderboard["experiments"], 1):
        m = exp["metrics"]
        seeds_str = f"{len(exp['seeds_done'])}/{len(exp['seeds_done']) + len(exp['seeds_missing'])}"
        flags = ", ".join(exp.get("regression_flags", [])) or "-"
        w(f"| {i} | {exp['exp_id']} | {m['primary_mean']:.4f} | {m['primary_std']:.4f} | {m['primary_best']:.4f} | {m['best_run_id']} | {seeds_str} | {flags} |")

    w("")
    return "\n".join(lines)


def main():
    args = parse_args()
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    manifest_path = args.manifest or str(reports_dir / "m5_sweep_manifest.json")
    output_json = args.output_json or str(reports_dir / "m5_leaderboard.json")
    output_md = args.output_md or str(reports_dir / "m5_leaderboard.md")

    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Building leaderboard from {manifest_path}")
    print(f"  Runs: {len(manifest['runs'])}")
    print(f"  Expected seeds: {manifest['expected_seeds']}")
    print(f"  Primary metric: {manifest['primary_metric']}")

    leaderboard = build_leaderboard(manifest)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print(f"JSON: {output_json}")

    with open(output_md, "w") as f:
        f.write(generate_md(leaderboard))
    print(f"Markdown: {output_md}")

    print(f"Experiments: {len(leaderboard['experiments'])}")
    for exp in leaderboard["experiments"]:
        flags = " [FLAGS: " + ", ".join(exp["regression_flags"]) + "]" if exp["regression_flags"] else ""
        print(f"  {exp['exp_id']}: mean={exp['metrics']['primary_mean']:.4f}{flags}")


if __name__ == "__main__":
    main()
