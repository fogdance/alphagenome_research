#!/usr/bin/env python3
"""
M9: Export champion model bundle.

Reads the M5 leaderboard, selects the champion experiment (lowest primary_mean),
and packages the best seed's checkpoint into a self-contained bundle.

Usage:
    python src/alphatrade/scripts/export_model_bundle.py \
      [--leaderboard <runs>/reports/m5_leaderboard.json] \
      [--exp-id <override>] [--run-id <override>] \
      [--output-dir <runs>/artifacts/model_bundle] \
      [--output-manifest <runs>/reports/m9_model_bundle_manifest.json]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

from alphatrade import runtime_paths


def parse_args():
    parser = argparse.ArgumentParser(description="M9: Export champion model bundle")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs (default: ALPHATRADE_RUNS_ROOT or ../alphatrade_runs/default)")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory (default: <output-root>/reports)")
    parser.add_argument("--artifacts-dir", type=str, default=None,
                        help="Artifacts directory (default: <output-root>/artifacts)")
    parser.add_argument("--leaderboard", type=str, default=None,
                        help="Path to M5 leaderboard JSON (default: <reports-dir>/m5_leaderboard.json)")
    parser.add_argument("--exp-id", type=str, default=None,
                        help="Override: select specific experiment ID")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Override: select specific run ID")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for bundles (default: <artifacts-dir>/model_bundle)")
    parser.add_argument("--output-manifest", type=str, default=None,
                        help="Path for the reports manifest copy (default: <reports-dir>/m9_model_bundle_manifest.json)")
    return parser.parse_args()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def select_champion(leaderboard: dict, exp_id_override: str | None, run_id_override: str | None) -> dict:
    """Select champion experiment and run from leaderboard.

    Returns dict with: exp, run, seed
    """
    experiments = leaderboard["experiments"]
    if not experiments:
        sys.exit("ERROR: leaderboard has no experiments")

    # Select experiment
    if exp_id_override:
        exp = next((e for e in experiments if e["exp_id"] == exp_id_override), None)
        if exp is None:
            available = [e["exp_id"] for e in experiments]
            sys.exit(f"ERROR: exp_id '{exp_id_override}' not found. Available: {available}")
    else:
        # Default: first experiment (lowest primary_mean, leaderboard is pre-sorted)
        exp = experiments[0]

    # Select run
    runs = exp["artifacts"]["runs"]
    if run_id_override:
        run = next((r for r in runs if r["run_id"] == run_id_override), None)
        if run is None:
            available = [r["run_id"] for r in runs]
            sys.exit(f"ERROR: run_id '{run_id_override}' not in exp '{exp['exp_id']}'. Available: {available}")
    else:
        # Default: best_run_id
        best_run_id = exp["metrics"]["best_run_id"]
        run = next((r for r in runs if r["run_id"] == best_run_id), None)
        if run is None:
            sys.exit(f"ERROR: best_run_id '{best_run_id}' not found in runs")

    return {"exp": exp, "run": run, "seed": run["seed"]}


def main():
    args = parse_args()
    output_root = runtime_paths.resolve_output_root(args.output_root)
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    artifacts_dir = runtime_paths.artifacts_dir(args.output_root, args.artifacts_dir)
    leaderboard_path = args.leaderboard or str(reports_dir / "m5_leaderboard.json")
    output_dir = args.output_dir or str(artifacts_dir / "model_bundle")
    output_manifest = args.output_manifest or str(reports_dir / "m9_model_bundle_manifest.json")

    print(f"\n{'='*60}")
    print(f"M9: Export Champion Model Bundle")
    print(f"{'='*60}\n")
    print(f"Output root: {output_root}")
    print(f"Reports:     {reports_dir}")
    print(f"Artifacts:   {artifacts_dir}\n")

    # Load leaderboard
    if not os.path.exists(leaderboard_path):
        sys.exit(f"ERROR: leaderboard not found: {leaderboard_path}")
    with open(leaderboard_path, 'r') as f:
        leaderboard = json.load(f)

    # Select champion
    champion = select_champion(leaderboard, args.exp_id, args.run_id)
    exp = champion["exp"]
    run = champion["run"]
    seed = champion["seed"]

    exp_id = exp["exp_id"]
    run_id = run["run_id"]
    metrics = exp["metrics"]

    print(f"Champion: exp_id={exp_id}, run_id={run_id}, seed={seed}")
    print(f"  primary_mean={metrics['primary_mean']:.6f}")
    print(f"  primary_std={metrics['primary_std']:.6f}")
    print(f"  primary_best={metrics['primary_best']:.6f}")

    # Load train metrics to get checkpoint_dir
    train_metrics_path = run["train_metrics_path"]
    if not os.path.exists(train_metrics_path):
        sys.exit(f"ERROR: train metrics not found: {train_metrics_path}")
    with open(train_metrics_path, 'r') as f:
        train_metrics = json.load(f)

    checkpoint_dir = train_metrics["run"]["checkpoint_dir"]
    if not os.path.exists(checkpoint_dir):
        sys.exit(f"ERROR: checkpoint dir not found: {checkpoint_dir}")

    best_ckpt_dir = os.path.join(checkpoint_dir, "best")
    if not os.path.exists(best_ckpt_dir):
        sys.exit(f"ERROR: best checkpoint dir not found: {best_ckpt_dir}")

    artifacts_json_path = os.path.join(checkpoint_dir, "artifacts.json")
    if not os.path.exists(artifacts_json_path):
        sys.exit(f"ERROR: artifacts.json not found: {artifacts_json_path}")

    # Read artifacts.json
    with open(artifacts_json_path, 'r') as f:
        artifacts = json.load(f)

    # Prefer full model config from train_metrics (has stem_channels, num_encoder_stages)
    # Fall back to artifacts.json model_config
    model_config = train_metrics.get("model", {}).get("config", {})
    if not model_config:
        model_config = artifacts.get("model_config", {})

    # Construct model_version
    model_version = f"alphatrade_v0.2_{exp_id}_{run_id}"
    print(f"\nModel version: {model_version}")

    # Create bundle directory
    bundle_dir = os.path.join(output_dir, model_version)
    os.makedirs(bundle_dir, exist_ok=True)

    # Copy best/ checkpoint dir
    dest_best = os.path.join(bundle_dir, "best")
    if os.path.exists(dest_best):
        shutil.rmtree(dest_best)
    shutil.copytree(best_ckpt_dir, dest_best)
    print(f"  Copied best/ checkpoint")

    # Copy artifacts.json
    shutil.copy2(artifacts_json_path, os.path.join(bundle_dir, "artifacts.json"))
    print(f"  Copied artifacts.json")

    # Write model_config.json
    model_config_path = os.path.join(bundle_dir, "model_config.json")
    with open(model_config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"  Wrote model_config.json")

    # Build bundle manifest
    git_sha = get_git_sha()
    bundle_manifest = {
        "schema_version": "m9_model_bundle_manifest_v1",
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "model_version": model_version,
        "champion_selection": {
            "method": "leaderboard_primary_mean",
            "exp_id": exp_id,
            "run_id": run_id,
            "seed": seed,
            "primary_mean": metrics["primary_mean"],
            "primary_std": metrics["primary_std"],
            "primary_best": metrics["primary_best"],
            "config_hash": exp["config_hash"],
        },
        "model_config": model_config,
        "bundle_path": bundle_dir,
        "checkpoint_source": checkpoint_dir,
        "dataset_config": train_metrics.get("dataset", {}).get("config_path", ""),
        "universe": leaderboard.get("universe", ""),
        "expected_seeds": leaderboard.get("expected_seeds", []),
        "leaderboard_path": leaderboard_path,
    }

    # Write bundle_manifest.json inside bundle
    bundle_manifest_path = os.path.join(bundle_dir, "bundle_manifest.json")
    with open(bundle_manifest_path, 'w') as f:
        json.dump(bundle_manifest, f, indent=2)
    print(f"  Wrote bundle_manifest.json")

    # Copy manifest to the reports directory for validation and handoff.
    os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
    shutil.copy2(bundle_manifest_path, output_manifest)
    print(f"  Copied manifest -> {output_manifest}")

    print(f"\n{'='*60}")
    print(f"Bundle: {bundle_dir}")
    print(f"Manifest: {output_manifest}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
