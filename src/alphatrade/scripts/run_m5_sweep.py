#!/usr/bin/env python3
"""
M5 Sweep Runner

Runs the full sweep pipeline: train → eval → manifest → leaderboard → validator
for all (exp_id, seed) combinations defined in a sweep config YAML.

Usage:
    python src/alphatrade/scripts/run_m5_sweep.py \
        --sweep-config configs/sweep/m5.yaml \
        [--output-root ../alphatrade_runs/default] \
        [--out-manifest <runs>/reports/m5_sweep_manifest.json] \
        [--reports-dir <runs>/reports] \
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
from alphatrade import gpu_launcher
from alphatrade import runtime_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="M5 Sweep Runner")
    parser.add_argument("--sweep-config", type=str, required=True,
                        help="Path to sweep config YAML (e.g. configs/sweep/m5.yaml)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs (default: ALPHATRADE_RUNS_ROOT or ../alphatrade_runs/default)")
    parser.add_argument("--out-manifest", type=str, default=None,
                        help="Output manifest path")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory")
    parser.add_argument("--checkpoints-dir", type=str, default=None,
                        help="Checkpoints directory (default: <output-root>/checkpoints)")
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


def compute_config_hash(
    overrides: dict,
    dataset_config: str,
    smoke: bool = False,
    universe: str | None = None,
) -> str:
    """SHA256[:8] of canonical JSON of config-relevant fields."""
    payload = json.dumps(
        {
            "dataset_config": dataset_config,
            "universe": universe,
            "overrides": overrides,
            "smoke": bool(smoke),
        },
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def build_run_plan(config: dict, smoke: bool = False) -> List[dict]:
    """Return list of {exp_id, seed, config_hash, overrides, dataset_config, description}."""
    seeds = config["expected_seeds"]
    default_dataset_config = config["dataset_config"]
    universe = config.get("universe", "unknown")
    defaults = config.get("defaults", {})
    plan = []
    for exp in config["experiments"]:
        exp_id = exp["exp_id"]
        dataset_config = exp.get("dataset_config", default_dataset_config)
        overrides = {**defaults, **exp.get("overrides", {})}
        if smoke:
            overrides["max_steps"] = 3
            overrides["save_every"] = 1
        config_hash = compute_config_hash(
            overrides,
            dataset_config,
            smoke=smoke,
            universe=universe,
        )
        for seed in seeds:
            plan.append({
                "exp_id": exp_id,
                "seed": seed,
                "config_hash": config_hash,
                "overrides": overrides,
                "dataset_config": dataset_config,
                "universe": universe,
                "eval_split": overrides.get("eval_split", "val"),
                "ckpt_step": overrides.get("ckpt_step", "best"),
                "description": exp.get("description", ""),
                "smoke": bool(smoke),
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


def sweep_metadata(run: dict) -> dict:
    """Return the per-run metadata used to validate resume artifacts."""
    return {
        "exp_id": run["exp_id"],
        "seed": run["seed"],
        "config_hash": run["config_hash"],
        "dataset_config": run["dataset_config"],
        "universe": run.get("universe", "unknown"),
        "eval_split": run.get("eval_split", run.get("overrides", {}).get("eval_split", "val")),
        "ckpt_step": run.get("ckpt_step", run.get("overrides", {}).get("ckpt_step", "best")),
        "overrides": run["overrides"],
        "smoke": bool(run.get("smoke", False)),
    }


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _artifact_sweep_metadata(data: dict, kind: str) -> Optional[dict]:
    if kind == "train":
        return data.get("run", {}).get("sweep")
    if kind == "eval":
        return data.get("model", {}).get("sweep")
    raise ValueError(f"unknown artifact kind: {kind}")


_RESUME_COMPARE_FIELDS = (
    "exp_id",
    "seed",
    "config_hash",
    "dataset_config",
    "universe",
    "eval_split",
    "ckpt_step",
)


class ResumeConfigMismatch(RuntimeError):
    """Raised when --resume would reuse artifacts from a different config."""


def _artifact_matches_run(data: dict, kind: str, expected_run: Optional[dict]) -> tuple[bool, str]:
    if expected_run is None:
        return True, ""

    expected = sweep_metadata(expected_run)
    actual = _artifact_sweep_metadata(data, kind)
    if actual is None:
        return False, f"{kind} artifact has no sweep metadata"
    mismatches = []
    for field in _RESUME_COMPARE_FIELDS:
        if actual.get(field) != expected.get(field):
            mismatches.append(
                f"{field}: existing={actual.get(field)!r} expected={expected.get(field)!r}"
            )
    if actual.get("overrides") != expected.get("overrides"):
        mismatches.append("overrides/defaults changed")
    if bool(actual.get("smoke", False)) != bool(expected.get("smoke", False)):
        mismatches.append(
            f"smoke: existing={actual.get('smoke')!r} expected={expected.get('smoke')!r}"
        )
    if mismatches:
        return False, f"{kind} artifact sweep metadata mismatch ({'; '.join(mismatches)})"
    return True, ""


def fail_on_resume_config_mismatch(run_state: dict, exp_id: str, seed: int) -> None:
    """Fail loudly before --resume can reuse stale train/eval artifacts."""
    if not run_state.get("stale_reasons"):
        return
    reasons = "; ".join(run_state["stale_reasons"])
    raise ResumeConfigMismatch(
        f"resume_config_mismatch: exp_id={exp_id} seed={seed}: {reasons}"
    )


def get_run_state(exp_id: str, seed: int, reports_dir: str,
                  expected_run: Optional[dict] = None) -> dict:
    """Return readable train/eval state for finer resume handling."""
    train_path, eval_path = _run_file_paths(exp_id, seed, reports_dir)

    train_data = _read_json(train_path)
    eval_data = _read_json(eval_path)
    stale_reasons = []

    train_complete = train_data is not None
    eval_complete = eval_data is not None

    if train_data is not None:
        ok, reason = _artifact_matches_run(train_data, "train", expected_run)
        if not ok:
            train_complete = False
            stale_reasons.append(reason)

    if eval_data is not None:
        ok, reason = _artifact_matches_run(eval_data, "eval", expected_run)
        if not ok:
            eval_complete = False
            stale_reasons.append(reason)

    run_id = train_data.get("run", {}).get("run_id", "unknown") if train_complete else None
    if train_complete and eval_complete:
        eval_train_run_id = eval_data.get("model", {}).get("train_run_id")
        if eval_train_run_id != run_id:
            eval_complete = False
            stale_reasons.append("eval artifact points to a different train_run_id")

    return {
        "train_path": train_path,
        "eval_path": eval_path,
        "train_complete": train_complete,
        "eval_complete": eval_complete,
        "complete": train_complete and eval_complete,
        "run_id": run_id,
        "stale_reasons": stale_reasons,
    }


def _extract_run_id(metrics_path: str) -> str:
    """Read run_id from a metrics JSON file."""
    with open(metrics_path) as f:
        data = json.load(f)
    return data.get("run", {}).get("run_id", "unknown")


def _build_cmd(module: str, cli_args: List[str], gpu: bool) -> tuple:
    """Return (cmd, env) for subprocess."""
    return gpu_launcher.build_module_cmd(module, cli_args, gpu=gpu)


def _write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Sweep → train CLI parameter mapping
# ---------------------------------------------------------------------------

# Each entry: (sweep_yaml_key, cli_flag, default_or_None)
# - If default_or_None is not None: always pass the flag with override or default
# - If default_or_None is None: only pass the flag when present in overrides
#
# To support a new parameter:
#   1. Add a CLI arg in train_m4_alphatrade.py: parse_args()
#   2. Add one row here
#   3. Done — no other code changes needed
_TRAIN_PARAM_MAP = [
    ("max_steps",      "--max-steps",      500),
    ("batch_size",     "--batch-size",     128),
    ("jit",            "--jit",            1),
    ("clip_norm",      "--clip-norm",      1.0),
    ("save_every",     "--save-every",     100),
    ("keep_last",      "--keep-last",      3),
    ("window_cache",   "--window-cache",   "auto"),
    # Parameters that only pass when explicitly set (override config YAML defaults)
    ("learning_rate",  "--learning-rate",  None),
    ("weight_decay",   "--weight-decay",   None),
    ("val_every",      "--val-every",      None),
    ("window_cache_dir", "--window-cache-dir", None),
]

_EVAL_PARAM_MAP = [
    ("batch_size",     "--batch-size",     128),
    ("eval_split",     "--split",          "val"),
    ("ckpt_step",      "--ckpt-step",      "best"),
    ("window_cache",   "--window-cache",   "auto"),
    ("window_cache_dir", "--window-cache-dir", None),
]


def _build_cli_from_map(param_map: list, overrides: dict) -> List[str]:
    """Build CLI args list from a parameter mapping table and overrides dict."""
    cli = []
    for yaml_key, cli_flag, default in param_map:
        if yaml_key in overrides:
            cli.extend([cli_flag, str(overrides[yaml_key])])
        elif default is not None:
            cli.extend([cli_flag, str(default)])
        # else: not in overrides and no default → skip (train script uses its own default)
    return cli


# ---------------------------------------------------------------------------
# Single-run executors
# ---------------------------------------------------------------------------

def run_single_train(run: dict, output_root: str, reports_dir: str,
                     checkpoints_dir: str, smoke: bool, gpu: bool) -> Optional[dict]:
    """Run training for a single (exp_id, seed). Returns info dict or None."""
    exp_id = run["exp_id"]
    seed = run["seed"]
    ov = run["overrides"]

    print(f"\n{'='*60}")
    print(f"[TRAIN] exp={exp_id} seed={seed}")
    print(f"{'='*60}\n")

    ckpt_dir = str(Path(checkpoints_dir) / "m5" / f"{exp_id}_seed{seed}")

    cli_args = [
        "--config", run["dataset_config"],
        "--seed", str(seed),
        "--output-root", output_root,
        "--ckpt-dir", ckpt_dir,
        "--reports-dir", reports_dir,
    ]
    cli_args.extend(_build_cli_from_map(_TRAIN_PARAM_MAP, ov))
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
    metrics.setdefault("run", {})["sweep"] = sweep_metadata(run)
    _write_json(dst_json, metrics)
    os.remove(src_json)

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


def run_single_eval(run: dict, train_info: dict, output_root: str, reports_dir: str,
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
        "--output-root", output_root,
        "--reports-dir", reports_dir,
    ]
    cli_args.extend(_build_cli_from_map(_EVAL_PARAM_MAP, ov))
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
    with open(src_json) as f:
        metrics = json.load(f)
    metrics.setdefault("model", {})["sweep"] = sweep_metadata(run)
    _write_json(dst_json, metrics)
    os.remove(src_json)

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
            "dataset_config": cr.get("dataset_config", ""),
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
        "dataset_config": config.get("dataset_config", ""),
        "dataset_configs": sorted({r.get("dataset_config", "") for r in completed_runs}),
        "expected_seeds": config["expected_seeds"],
        "primary_metric": config["primary_metric"],
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# Post-pipeline (leaderboard + validator)
# ---------------------------------------------------------------------------

def run_post_pipeline(manifest_path: str, output_root: str, reports_dir: str,
                      gpu: bool) -> bool:
    """Run leaderboard builder and validator. Returns True if all pass."""
    all_ok = True

    # 1. Build leaderboard
    print(f"\n{'='*60}")
    print("[POST] Building M5 leaderboard")
    print(f"{'='*60}\n")

    cmd, env = _build_cmd(
        "build_m5_leaderboard",
        [
            "--manifest", manifest_path,
            "--output-root", output_root,
            "--reports-dir", reports_dir,
            "--output-json", os.path.join(reports_dir, "m5_leaderboard.json"),
            "--output-md", os.path.join(reports_dir, "m5_leaderboard.md"),
        ],
        gpu,
    )
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
        "--output-root", output_root,
        "--reports-dir", reports_dir,
        "--output-json", os.path.join(reports_dir, "m5_schema_validation.json"),
        "--output-md", os.path.join(reports_dir, "m5_schema_validation.md"),
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
    output_root = runtime_paths.resolve_output_root(args.output_root)
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    checkpoints_dir = runtime_paths.checkpoints_dir(
        args.output_root, args.checkpoints_dir
    )
    out_manifest = Path(args.out_manifest) if args.out_manifest else reports_dir / "m5_sweep_manifest.json"

    os.makedirs(reports_dir, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    # ── Header ──
    print(f"\n{'='*60}")
    print("M5 Sweep Runner")
    print(f"{'='*60}")
    print(f"Config:    {args.sweep_config}")
    print(f"Output:    {output_root}")
    print(f"Reports:   {reports_dir}")
    print(f"Ckpts:     {checkpoints_dir}")
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

        run_state = get_run_state(exp_id, seed, str(reports_dir), r) if args.resume else None
        if run_state:
            try:
                fail_on_resume_config_mismatch(run_state, exp_id, seed)
            except ResumeConfigMismatch as e:
                print(f"ERROR: {e}")
                sys.exit(1)

        # Resume check: fully complete run.
        if run_state and run_state["complete"]:
            train_path = run_state["train_path"]
            eval_path = run_state["eval_path"]
            run_id = run_state["run_id"]
            print(f"[SKIP] {exp_id} seed={seed} — already complete (run_id={run_id})")
            completed_runs.append({
                "exp_id": exp_id,
                "seed": seed,
                "run_id": run_id,
                "config_hash": r["config_hash"],
                "dataset_config": r["dataset_config"],
                "train_metrics_path": train_path,
                "eval_metrics_path": eval_path,
                "train_md_path": None,
                "eval_md_path": None,
            })
            continue

        try:
            if run_state and run_state["train_complete"]:
                print(
                    f"[RESUME] {exp_id} seed={seed} — train complete, "
                    "running missing eval only"
                )
                train_info = {
                    "run_id": run_state["run_id"],
                    "train_metrics_path": run_state["train_path"],
                    "train_md_path": None,
                }
            else:
                # Train
                train_info = run_single_train(
                    r, str(output_root), str(reports_dir), str(checkpoints_dir),
                    args.smoke, args.gpu
                )
                if train_info is None:
                    print(f"[SKIP] {exp_id} seed={seed} — training failed, skipping eval")
                    continue

            # Eval
            eval_info = run_single_eval(
                r, train_info, str(output_root), str(reports_dir), args.smoke, args.gpu
            )
            if eval_info is None:
                print(f"[SKIP] {exp_id} seed={seed} — eval failed")
                continue

            completed_runs.append({
                "exp_id": exp_id,
                "seed": seed,
                "run_id": train_info["run_id"],
                "config_hash": r["config_hash"],
                "dataset_config": r["dataset_config"],
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

    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[MANIFEST] {out_manifest} ({len(completed_runs)}/{len(plan)} runs)")

    # ── Post-pipeline ──
    if not completed_runs:
        print("\n[WARN] No completed runs — skipping post-pipeline")
        sys.exit(1)

    all_ok = run_post_pipeline(
        str(out_manifest), str(output_root), str(reports_dir), args.gpu
    )

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("M5 Sweep Complete")
    print(f"{'='*60}")
    print(f"  Completed: {len(completed_runs)}/{len(plan)} runs")
    print(f"  Manifest:  {out_manifest}")
    print(f"  Gate:      {'PASS' if all_ok else 'FAIL'}")
    print()

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
