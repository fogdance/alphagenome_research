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
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from alphatrade import prediction_schema
from alphatrade import runtime_paths
from alphatrade.data_pipeline.feature_profiles import (
    feature_profile_from_mapping,
    feature_profile_to_dict,
)


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
    parser.add_argument("--model-version", type=str, default=None,
                        help="Override model_version directory/name")
    parser.add_argument("--overwrite", action="store_true", default=True,
                        help="Overwrite an existing bundle with the same model_version (default)")
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite",
                        help="Fail if the target bundle directory already exists")
    return parser.parse_args()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_file_manifest(bundle_dir: Path) -> list[dict]:
    """Return content hashes for all versioned bundle payload files."""
    files = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(bundle_dir).as_posix()
        if rel_path == "bundle_manifest.json":
            continue
        files.append(
            {
                "path": rel_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def compute_bundle_id(model_version: str, files: list[dict]) -> str:
    payload = json.dumps(
        {"model_version": model_version, "files": files},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_model_version(model_version: str) -> str:
    """Validate model_version is a single safe directory name."""
    if not model_version or model_version.strip() != model_version:
        sys.exit(f"ERROR: invalid model_version: {model_version!r}")
    if model_version in (".", ".."):
        sys.exit(f"ERROR: invalid model_version: {model_version!r}")
    if "/" in model_version or "\\" in model_version:
        sys.exit("ERROR: model_version must not contain path separators")
    return model_version


def prepare_bundle_dir(output_dir: str, model_version: str, overwrite: bool) -> Path:
    """Create an empty target bundle directory."""
    model_version = validate_model_version(model_version)
    bundle_dir = Path(output_dir).expanduser().resolve() / model_version
    if bundle_dir.exists():
        if not overwrite:
            sys.exit(f"ERROR: bundle already exists: {bundle_dir}")
        if bundle_dir.is_dir():
            shutil.rmtree(bundle_dir)
        else:
            bundle_dir.unlink()
    bundle_dir.mkdir(parents=True, exist_ok=False)
    return bundle_dir


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
    feature_profile_data = (
        train_metrics.get("dataset", {}).get("feature_profile")
        or train_metrics.get("model", {}).get("config", {}).get("feature_profile")
        or artifacts.get("feature_profile")
        or model_config.get("feature_profile")
    )
    if not feature_profile_data:
        sys.exit("ERROR: train metrics/artifacts missing required feature_profile")
    try:
        feature_profile = feature_profile_from_mapping(feature_profile_data)
    except ValueError as exc:
        sys.exit(f"ERROR: invalid feature_profile: {exc}")
    if int(model_config.get("num_features", -1)) != feature_profile.feature_dim:
        sys.exit(
            "ERROR: model_config.num_features does not match feature_profile.feature_dim: "
            f"{model_config.get('num_features')} vs {feature_profile.feature_dim}"
        )
    feature_profile_dict = feature_profile_to_dict(feature_profile)
    model_config = {
        **model_config,
        "feature_profile": feature_profile_dict,
        "feature_profile_id": feature_profile.profile_id,
        "feature_cols": list(feature_profile.feature_cols),
    }

    # Construct model_version
    model_version = validate_model_version(args.model_version or f"alphatrade_v0.2_{exp_id}_{run_id}")
    print(f"\nModel version: {model_version}")

    # Create bundle directory
    bundle_dir = prepare_bundle_dir(output_dir, model_version, args.overwrite)

    # Copy best/ checkpoint dir
    dest_best = bundle_dir / "best"
    shutil.copytree(best_ckpt_dir, dest_best)
    print(f"  Copied best/ checkpoint")

    # Copy artifacts.json
    shutil.copy2(artifacts_json_path, bundle_dir / "artifacts.json")
    print(f"  Copied artifacts.json")

    # Write model_config.json
    model_config_path = bundle_dir / "model_config.json"
    with model_config_path.open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"  Wrote model_config.json")
    bundle_files = bundle_file_manifest(bundle_dir)
    bundle_id = compute_bundle_id(model_version, bundle_files)

    # Build bundle manifest
    git_sha = get_git_sha()
    bundle_manifest = {
        "schema_version": "m9_model_bundle_manifest_v1",
        "bundle_format_version": "alphatrade_model_bundle_v1",
        "prediction_schema_version": prediction_schema.PREDICTION_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "git_sha": git_sha,
        "model_version": model_version,
        "bundle_id": bundle_id,
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
        "feature_profile": feature_profile_dict,
        "model_config": model_config,
        "bundle_path": str(bundle_dir),
        "bundle_files": bundle_files,
        "checkpoint_source": checkpoint_dir,
        "dataset_config": train_metrics.get("dataset", {}).get("config_path", ""),
        "universe": leaderboard.get("universe", ""),
        "expected_seeds": leaderboard.get("expected_seeds", []),
        "leaderboard_path": leaderboard_path,
    }

    # Write bundle_manifest.json inside bundle
    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    with bundle_manifest_path.open("w", encoding="utf-8") as f:
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
