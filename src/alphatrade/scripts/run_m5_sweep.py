#!/usr/bin/env python3
"""
M5 Sweep Runner

Runs the full sweep pipeline: train → eval → manifest → leaderboard → validator
for all (exp_id, seed) combinations defined in a sweep config YAML.

Usage:
    python src/alphatrade/scripts/run_m5_sweep.py \
        --sweep-config configs/sweep/m5.yaml \
        [--out-manifest reports/m5_sweep_manifest.json] \
        [--reports-dir reports] \
        [--dry-run] [--resume] [--smoke] [--gpu/--no-gpu]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="M5 Sweep Runner")
    parser.add_argument("--sweep-config", type=str, required=True,
                        help="Path to sweep config YAML (e.g. configs/sweep/m5.yaml)")
    parser.add_argument("--out-manifest", type=str, default="reports/m5_sweep_manifest.json",
                        help="Output manifest path")
    parser.add_argument("--reports-dir", type=str, default="reports",
                        help="Reports directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print run plan and exit")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose output files already exist")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test mode (3 symbols, 3 steps)")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Run on GPU (default)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu",
                        help="Force CPU mode")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_sweep_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_config_hash(overrides: dict, dataset_config: str) -> str:
    """SHA256[:8] of canonical JSON of config-relevant fields."""
    payload = json.dumps(
        {"dataset_config": dataset_config, "overrides": overrides},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def build_run_plan(config: dict, smoke: bool = False) -> List[dict]:
    """Return list of {exp_id, seed, config_hash, overrides, dataset_config, description}."""
    seeds = config["expected_seeds"]
    dataset_config = config["dataset_config"]
    defaults = config.get("defaults", {})
    plan = []
    for exp in config["experiments"]:
        exp_id = exp["exp_id"]
        overrides = {**defaults, **exp.get("overrides", {})}
        if smoke:
            overrides["max_steps"] = 3
            overrides["save_every"] = 1
        config_hash = compute_config_hash(exp.get("overrides", {}), dataset_config)
        for seed in seeds:
            plan.append({
                "exp_id": exp_id,
                "seed": seed,
                "config_hash": config_hash,
                "overrides": overrides,
                "dataset_config": dataset_config,
                "description": exp.get("description", ""),
            })
    return plan


# ---------------------------------------------------------------------------
# Resume check
# ---------------------------------------------------------------------------

def _run_file_paths(exp_id: str, seed: int, reports_dir: str) -> tuple:
    """Return (train_path, eval_path) for a run."""
    train = os.path.join(reports_dir, f"m5_{exp_id}_seed{seed}_train_metrics.json")
    eval_ = os.path.join(reports_dir, f"m5_{exp_id}_seed{seed}_eval_metrics.json")
    return train, eval_


def is_run_complete(exp_id: str, seed: int, reports_dir: str) -> bool:
    """Check if both train and eval JSON exist and are readable."""
    train_path, eval_path = _run_file_paths(exp_id, seed, reports_dir)
    for p in (train_path, eval_path):
        if not os.path.exists(p):
            return False
        try:
            with open(p) as f:
                json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
    return True


def _extract_run_id(metrics_path: str) -> str:
    """Read run_id from a metrics JSON file."""
    with open(metrics_path) as f:
        data = json.load(f)
    return data.get("run", {}).get("run_id", "unknown")


# ---------------------------------------------------------------------------
# GPU workaround (same as run_m4_matrix.py)
# ---------------------------------------------------------------------------

_GPU_ENV = {
    **os.environ,
    "JAX_PLATFORMS": "cuda",
    "XLA_FLAGS": "--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=",
}


def _build_cmd(module: str, cli_args: List[str], gpu: bool) -> tuple:
    """Return (cmd, env) for subprocess."""
    if gpu:
        argv_str = json.dumps(["run"] + cli_args)
        code = (
            f"import sys; sys.argv = {argv_str}; "
            f"from alphatrade.scripts.{module} import main; main()"
        )
        return ["python", "-c", code], _GPU_ENV
    else:
        script = f"src/alphatrade/scripts/{module}.py"
        return ["python", script] + cli_args, None


# ---------------------------------------------------------------------------
# Single-run executors
# ---------------------------------------------------------------------------

def run_single_train(run: dict, reports_dir: str, smoke: bool,
                     gpu: bool) -> Optional[dict]:
    """Run training for a single (exp_id, seed). Returns info dict or None."""
    exp_id = run["exp_id"]
    seed = run["seed"]
    ov = run["overrides"]

    print(f"\n{'='*60}")
    print(f"[TRAIN] exp={exp_id} seed={seed}")
    print(f"{'='*60}\n")

    ckpt_dir = f"checkpoints/m5/{exp_id}_seed{seed}"

    cli_args = [
        "--config", run["dataset_config"],
        "--seed", str(seed),
        "--max-steps", str(ov.get("max_steps", 500)),
        "--batch-size", str(ov.get("batch_size", 128)),
        "--jit", str(ov.get("jit", 0)),
        "--clip-norm", str(ov.get("clip_norm", 1.0)),
        "--ckpt-dir", ckpt_dir,
        "--save-every", str(ov.get("save_every", 100)),
        "--keep-last", str(ov.get("keep_last", 3)),
    ]
    if smoke:
        cli_args.append("--smoke")

    cmd, env = _build_cmd("train_m4_alphatrade", cli_args, gpu)
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"  [FAIL] Training failed for {exp_id} seed={seed}")
        return None

    # Rename m4 outputs to m5 per-run names
    src_json = os.path.join(reports_dir, "m4_train_metrics.json")
    if not os.path.exists(src_json):
        print(f"  [FAIL] Train metrics not found: {src_json}")
        return None

    with open(src_json) as f:
        metrics = json.load(f)

    run_id = metrics.get("run", {}).get("run_id", "unknown")
    dst_json = os.path.join(reports_dir, f"m5_{exp_id}_seed{seed}_train_metrics.json")
    os.rename(src_json, dst_json)

    src_md = os.path.join(reports_dir, "m4_train_run.md")
    dst_md = os.path.join(reports_dir, f"m5_{exp_id}_seed{seed}_train_run.md")
    if os.path.exists(src_md):
        os.rename(src_md, dst_md)

    print(f"  [OK] {dst_json} (run_id={run_id})")
    return {
        "run_id": run_id,
        "train_metrics_path": dst_json,
        "train_md_path": dst_md if os.path.exists(dst_md) else None,
    }


def run_single_eval(run: dict, train_info: dict, reports_dir: str,
                    smoke: bool, gpu: bool) -> Optional[dict]:
    """Run evaluation for a single (exp_id, seed). Returns info dict or None."""
    exp_id = run["exp_id"]
    seed = run["seed"]
    ov = run["overrides"]

    print(f"\n{'='*60}")
    print(f"[EVAL] exp={exp_id} seed={seed}")
    print(f"{'='*60}\n")

    cli_args = [
        "--train-metrics", train_info["train_metrics_path"],
        "--dataset-config", run["dataset_config"],
        "--split", ov.get("eval_split", "val"),
        "--batch-size", str(ov.get("batch_size", 128)),
        "--ckpt-step", str(ov.get("ckpt_step", "best")),
    ]
    if smoke:
        cli_args.append("--smoke")

    cmd, env = _build_cmd("eval_m4_fast", cli_args, gpu)
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"  [FAIL] Eval failed for {exp_id} seed={seed}")
        return None

    src_json = os.path.join(reports_dir, "m4_eval_metrics_fast.json")
    if not os.path.exists(src_json):
        print(f"  [FAIL] Eval metrics not found: {src_json}")
        return None

    dst_json = os.path.join(reports_dir, f"m5_{exp_id}_seed{seed}_eval_metrics.json")
    os.rename(src_json, dst_json)

    src_md = os.path.join(reports_dir, "m4_eval_run_fast.md")
    dst_md = os.path.join(reports_dir, f"m5_{exp_id}_seed{seed}_eval_run.md")
    if os.path.exists(src_md):
        os.rename(src_md, dst_md)

    print(f"  [OK] {dst_json}")
    return {
        "eval_metrics_path": dst_json,
        "eval_md_path": dst_md if os.path.exists(dst_md) else None,
    }


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_manifest(config: dict, completed_runs: List[dict], git_sha: str) -> dict:
    """Build schema-compliant m5_sweep_manifest.json."""
    runs = []
    for cr in completed_runs:
        entry = {
            "exp_id": cr["exp_id"],
            "seed": cr["seed"],
            "run_id": cr["run_id"],
            "config_hash": cr["config_hash"],
            "train_metrics_path": cr["train_metrics_path"],
            "eval_metrics_path": cr["eval_metrics_path"],
        }
        if cr.get("train_md_path"):
            entry["train_run_md_path"] = cr["train_md_path"]
        if cr.get("eval_md_path"):
            entry["eval_run_md_path"] = cr["eval_md_path"]
        runs.append(entry)

    return {
        "schema_version": "m5_sweep_manifest_v1",
        "generated_at": datetime.now().isoformat(),
        "profile": "m5",
        "git_sha": git_sha,
        "universe": config.get("universe", "unknown"),
        "dataset": config.get("dataset", "unknown"),
        "expected_seeds": config["expected_seeds"],
        "primary_metric": config["primary_metric"],
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Post-pipeline (leaderboard + validator)
# ---------------------------------------------------------------------------

def run_post_pipeline(manifest_path: str, gpu: bool) -> bool:
    """Run leaderboard builder and validator. Returns True if all pass."""
    all_ok = True

    # 1. Build leaderboard
    print(f"\n{'='*60}")
    print("[POST] Building M5 leaderboard")
    print(f"{'='*60}\n")

    cmd, env = _build_cmd("build_m5_leaderboard", ["--manifest", manifest_path], gpu)
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print("  [FAIL] Leaderboard build failed")
        all_ok = False

    # 2. Run validator (run twice: first pass creates output files that are
    #    themselves in the m5 profile, second pass validates everything)
    print(f"\n{'='*60}")
    print("[POST] Running M5 schema validator")
    print(f"{'='*60}\n")

    validator_args = [
        "--profile", "m5", "--strict",
        "--output-json", "reports/m5_schema_validation.json",
        "--output-md", "reports/m5_schema_validation.md",
    ]

    # First pass: creates m5_schema_validation.json/md (may fail)
    cmd, env = _build_cmd("validate_reports_schema", validator_args, gpu)
    subprocess.run(cmd, capture_output=True, text=True, env=env)

    # Second pass: now all files exist — this is the real result
    cmd, env = _build_cmd("validate_reports_schema", validator_args, gpu)
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print("  [FAIL] Validator failed or reported strict errors")
        all_ok = False
    else:
        print("  [OK] Validator passed")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_sweep_config(args.sweep_config)
    plan = build_run_plan(config, smoke=args.smoke)

    os.makedirs(args.reports_dir, exist_ok=True)

    # ── Header ──
    print(f"\n{'='*60}")
    print("M5 Sweep Runner")
    print(f"{'='*60}")
    print(f"Config:    {args.sweep_config}")
    print(f"Runs:      {len(plan)}")
    print(f"Seeds:     {config['expected_seeds']}")
    print(f"Metric:    {config['primary_metric']}")
    print(f"GPU:       {'yes' if args.gpu else 'no'}")
    print(f"Smoke:     {'yes' if args.smoke else 'no'}")
    print(f"Resume:    {'yes' if args.resume else 'no'}")
    print(f"{'='*60}\n")

    # ── Plan table ──
    print(f"{'exp_id':<20} {'seed':<6} {'config_hash':<10} {'description'}")
    print(f"{'-'*20} {'-'*6} {'-'*10} {'-'*30}")
    for r in plan:
        print(f"{r['exp_id']:<20} {r['seed']:<6} {r['config_hash']:<10} {r['description']}")
    print()

    if args.dry_run:
        print("[DRY-RUN] Plan printed. Exiting.")
        return

    # ── Execute runs ──
    completed_runs = []

    for r in plan:
        exp_id = r["exp_id"]
        seed = r["seed"]

        # Resume check
        if args.resume and is_run_complete(exp_id, seed, args.reports_dir):
            train_path, eval_path = _run_file_paths(exp_id, seed, args.reports_dir)
            run_id = _extract_run_id(train_path)
            print(f"[SKIP] {exp_id} seed={seed} — already complete (run_id={run_id})")
            completed_runs.append({
                "exp_id": exp_id,
                "seed": seed,
                "run_id": run_id,
                "config_hash": r["config_hash"],
                "train_metrics_path": train_path,
                "eval_metrics_path": eval_path,
                "train_md_path": None,
                "eval_md_path": None,
            })
            continue

        try:
            # Train
            train_info = run_single_train(r, args.reports_dir, args.smoke, args.gpu)
            if train_info is None:
                print(f"[SKIP] {exp_id} seed={seed} — training failed, skipping eval")
                continue

            # Eval
            eval_info = run_single_eval(r, train_info, args.reports_dir, args.smoke, args.gpu)
            if eval_info is None:
                print(f"[SKIP] {exp_id} seed={seed} — eval failed")
                continue

            completed_runs.append({
                "exp_id": exp_id,
                "seed": seed,
                "run_id": train_info["run_id"],
                "config_hash": r["config_hash"],
                "train_metrics_path": train_info["train_metrics_path"],
                "eval_metrics_path": eval_info["eval_metrics_path"],
                "train_md_path": train_info.get("train_md_path"),
                "eval_md_path": eval_info.get("eval_md_path"),
            })
        except Exception as e:
            print(f"[ERROR] {exp_id} seed={seed}: {e}")
            continue

    # ── Write manifest ──
    git_sha = _get_git_sha()
    manifest = build_manifest(config, completed_runs, git_sha)

    with open(args.out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[MANIFEST] {args.out_manifest} ({len(completed_runs)}/{len(plan)} runs)")

    # ── Post-pipeline ──
    if not completed_runs:
        print("\n[WARN] No completed runs — skipping post-pipeline")
        sys.exit(1)

    all_ok = run_post_pipeline(args.out_manifest, args.gpu)

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("M5 Sweep Complete")
    print(f"{'='*60}")
    print(f"  Completed: {len(completed_runs)}/{len(plan)} runs")
    print(f"  Manifest:  {args.out_manifest}")
    print(f"  Gate:      {'PASS' if all_ok else 'FAIL'}")
    print()

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
