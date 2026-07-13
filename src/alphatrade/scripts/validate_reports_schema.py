#!/usr/bin/env python3
"""
Reports Schema & Semantic Validator

Phase 1: JSON Schema validation (structure) — driven by manifest profiles
Phase 2: Semantic checks (M4 eval↔train cross-validation)
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from alphatrade import prediction_schema
from alphatrade import runtime_paths

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    print("Error: jsonschema package not installed")
    print("Install with: pip install jsonschema")
    sys.exit(1)


_DEFAULT_MANIFEST = "src/alphatrade/schemas/contracts_manifest.yaml"
_DEFAULT_PROFILE = "m4"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate reports: schema + semantic")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs (default: ALPHATRADE_RUNS_ROOT or ../alphatrade_runs/default)")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory (default: <output-root>/reports)")
    parser.add_argument("--schemas-dir", type=str, default="src/alphatrade/schemas", help="Schemas directory")
    parser.add_argument("--manifest", type=str, default=_DEFAULT_MANIFEST, help="Path to contracts_manifest.yaml")
    parser.add_argument("--profile", type=str, default=_DEFAULT_PROFILE, help="Manifest profile to validate")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on required missing/fail")
    parser.add_argument("--output-json", type=str, default=None, help="JSON output path (default: <reports-dir>/{profile}_schema_validation.json)")
    parser.add_argument("--output-md", type=str, default=None, help="Markdown output path (default: <reports-dir>/{profile}_schema_validation.md)")
    return parser.parse_args()


def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    """Return sha256 for a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: str) -> dict | None:
    """Load manifest YAML. Returns None if file missing or yaml unavailable."""
    if not os.path.exists(manifest_path):
        return None
    try:
        import yaml
    except ImportError:
        warnings.warn("pyyaml not installed, cannot load manifest — falling back to legacy mode")
        return None
    with open(manifest_path, 'r') as f:
        return yaml.safe_load(f)


def get_profile_items(manifest: dict, profile_name: str) -> list[dict] | None:
    """Extract items list from a manifest profile. Returns None if profile not found."""
    profiles = manifest.get("profiles", {})
    profile = profiles.get(profile_name)
    if profile is None:
        return None
    return profile.get("items", [])


def get_profile_reports_dir(manifest: dict, profile_name: str) -> str | None:
    """Get reports_dir for a profile."""
    profiles = manifest.get("profiles", {})
    profile = profiles.get(profile_name)
    if profile is None:
        return None
    return profile.get("reports_dir")


# ---------------------------------------------------------------------------
# Legacy hardcoded validations (fallback when manifest is unavailable)
# ---------------------------------------------------------------------------

def _legacy_validations(reports_dir: str, schemas_dir: str) -> list[dict]:
    """Return the old hardcoded report→schema mapping."""
    validations = [
        {"name": "m2_train_metrics", "report": os.path.join(reports_dir, "m2_train_metrics.json"), "schema": os.path.join(schemas_dir, "m2_train_metrics.schema.json"), "required": True},
        {"name": "m2_universe_sweep", "report": os.path.join(reports_dir, "m2_universe_sweep.json"), "schema": os.path.join(schemas_dir, "m2_universe_sweep.schema.json"), "required": True},
        {"name": "m2_t1_dataloader_check", "report": os.path.join(reports_dir, "m2_t1_dataloader_check.json"), "schema": os.path.join(schemas_dir, "m2_t1_dataloader_check.schema.json"), "required": True},
        {"name": "m3_train_metrics", "report": os.path.join(reports_dir, "m3_train_metrics.json"), "schema": os.path.join(schemas_dir, "m2_train_metrics.schema.json"), "required": True},
        {"name": "m4_train_metrics", "report": os.path.join(reports_dir, "m4_train_metrics.json"), "schema": os.path.join(schemas_dir, "m2_train_metrics.schema.json"), "required": True},
        {"name": "m4_eval_metrics", "report": os.path.join(reports_dir, "m4_eval_metrics.json"), "schema": os.path.join(schemas_dir, "m4_eval_metrics.schema.json"), "required": True},
    ]
    # Also discover seed-specific files
    eval_schema = os.path.join(schemas_dir, "m4_eval_metrics.schema.json")
    train_schema = os.path.join(schemas_dir, "m2_train_metrics.schema.json")
    seed_pairs = discover_m4_seed_pairs(reports_dir)
    for sp in seed_pairs:
        validations.append({"name": f"m4_eval_metrics_seed{sp['seed']}", "report": sp["eval_path"], "schema": eval_schema, "required": True})
        if os.path.exists(sp["train_path"]):
            validations.append({"name": f"m4_train_metrics_seed{sp['seed']}", "report": sp["train_path"], "schema": train_schema, "required": True})
    return validations


# ---------------------------------------------------------------------------
# Phase 1: Schema validation
# ---------------------------------------------------------------------------

def validate_report(report_path: str, schema_path: str) -> dict:
    """Validate a single report against its JSON schema."""
    result = {
        "report": report_path,
        "schema": schema_path,
        "status": "unknown",
        "error": None,
    }

    try:
        if not os.path.exists(report_path):
            result["status"] = "missing_report"
            result["error"] = f"Report file not found: {report_path}"
            return result

        if not schema_path:
            # No schema — existence-only check (works for .md and other non-JSON files)
            result["status"] = "pass"
            return result

        if not os.path.exists(schema_path):
            result["status"] = "missing_schema"
            result["error"] = f"Schema file not found: {schema_path}"
            return result

        report = load_json(report_path)
        schema = load_json(schema_path)
        validate(instance=report, schema=schema)
        result["status"] = "pass"

    except ValidationError as e:
        result["status"] = "fail"
        result["error"] = str(e.message)
        result["error_path"] = list(e.path)
    except json.JSONDecodeError as e:
        result["status"] = "invalid_json"
        result["error"] = f"Invalid JSON: {str(e)}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def discover_m4_seed_pairs(reports_dir: str):
    """Find all m4_eval_metrics_seed*.json and pair with train metrics."""
    eval_pattern = os.path.join(reports_dir, "m4_eval_metrics_seed*.json")
    pairs = []
    for eval_path in sorted(glob.glob(eval_pattern)):
        match = re.search(r'seed(\d+)', os.path.basename(eval_path))
        if not match:
            continue
        seed = match.group(1)
        train_path = os.path.join(reports_dir, f"m4_train_metrics_seed{seed}.json")
        pairs.append({
            "seed": seed,
            "eval_path": eval_path,
            "train_path": train_path,
        })
    return pairs


# ---------------------------------------------------------------------------
# Phase 2: Semantic checks (M4 eval↔train)
# ---------------------------------------------------------------------------

def semantic_check_m4(eval_path: str, train_path: str) -> list:
    """Run semantic checks on an M4 eval↔train pair."""
    checks = []

    def add(name, passed, detail="", observed=None):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
            "observed": observed,
        })

    if not os.path.exists(eval_path):
        add("eval_exists", False, f"File not found: {eval_path}")
        return checks
    if not os.path.exists(train_path):
        add("train_exists", False, f"File not found: {train_path}")
        return checks

    try:
        eval_data = load_json(eval_path)
        train_data = load_json(train_path)
    except Exception as e:
        add("json_load", False, str(e))
        return checks

    model = eval_data.get("model", {})
    source = model.get("source")
    add("model.source == 'checkpoint'",
        source == "checkpoint",
        f"got '{source}'" if source != "checkpoint" else "")

    eval_run_id = model.get("train_run_id")
    train_run_id = train_data.get("run", {}).get("run_id")
    matched = eval_run_id == train_run_id and eval_run_id is not None
    add("model.train_run_id == train.run.run_id",
        matched,
        f"eval='{eval_run_id}' vs train='{train_run_id}'" if not matched else "")

    step = model.get("checkpoint_step")
    step_ok = isinstance(step, int) and step >= 1
    add("model.checkpoint_step >= 1",
        step_ok,
        f"got {step!r}" if not step_ok else "")

    ckpt_dir = model.get("checkpoint_dir")
    dir_ok = isinstance(ckpt_dir, str) and len(ckpt_dir) > 0
    add("model.checkpoint_dir non-empty",
        dir_ok,
        f"got {ckpt_dir!r}" if not dir_ok else "")

    return checks


# ---------------------------------------------------------------------------
# Phase 2: Semantic checks (M5 sweep)
# ---------------------------------------------------------------------------

_M5_SWEEP_COMPARE_FIELDS = (
    "exp_id",
    "seed",
    "config_hash",
    "dataset_config",
    "universe",
    "eval_split",
    "ckpt_step",
)


def _m5_artifact_sweep_metadata(data: dict, kind: str) -> dict | None:
    if kind == "train":
        return data.get("run", {}).get("sweep")
    if kind == "eval":
        return data.get("model", {}).get("sweep")
    raise ValueError(f"unknown M5 artifact kind: {kind}")


def _m5_expected_sweep_metadata(run: dict, manifest: dict) -> dict:
    return {
        "exp_id": run.get("exp_id"),
        "seed": run.get("seed"),
        "config_hash": run.get("config_hash"),
        "dataset_config": run.get("dataset_config", manifest.get("dataset_config", "")),
        "universe": run.get("universe", manifest.get("universe", "unknown")),
        "eval_split": run.get("eval_split", "val"),
        "ckpt_step": run.get("ckpt_step", "best"),
    }


def _m5_sweep_metadata_mismatch(data: dict, kind: str, run: dict, manifest: dict) -> str:
    expected = _m5_expected_sweep_metadata(run, manifest)
    actual = _m5_artifact_sweep_metadata(data, kind)
    if actual is None:
        return f"{kind} artifact has no sweep metadata"

    mismatches = []
    for field in _M5_SWEEP_COMPARE_FIELDS:
        if actual.get(field) != expected.get(field):
            mismatches.append(
                f"{field}: existing={actual.get(field)!r} expected={expected.get(field)!r}"
            )
    return "; ".join(mismatches)


def semantic_check_m5(reports_dir: str, schemas_dir: str = "src/alphatrade/schemas") -> dict:
    """Run semantic checks on M5 sweep manifest runs.

    Returns dict with:
      - checks: list of individual check results
      - by_exp: per-experiment details
      - all_pass: bool
    """
    manifest_path = os.path.join(reports_dir, "m5_sweep_manifest.json")
    train_schema_path = os.path.join(schemas_dir, "m2_train_metrics.schema.json")
    eval_schema_path = os.path.join(schemas_dir, "m4_eval_metrics.schema.json")

    checks = []

    def add(name, passed, detail="", observed=None):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
            "observed": observed,
        })

    # Load manifest
    if not os.path.exists(manifest_path):
        add("manifest_exists", False, f"File not found: {manifest_path}")
        return {"checks": checks, "by_exp": {}, "all_pass": False}

    try:
        manifest = load_json(manifest_path)
    except Exception as e:
        add("manifest_json_load", False, str(e))
        return {"checks": checks, "by_exp": {}, "all_pass": False}

    add("manifest_exists", True)

    expected_seeds = set(manifest.get("expected_seeds", []))
    runs = manifest.get("runs", [])

    # Load schemas for validation
    train_schema = None
    eval_schema = None
    if os.path.exists(train_schema_path):
        train_schema = load_json(train_schema_path)
    if os.path.exists(eval_schema_path):
        eval_schema = load_json(eval_schema_path)

    # Check run_id uniqueness
    run_ids = [r["run_id"] for r in runs]
    duplicates = [rid for rid in set(run_ids) if run_ids.count(rid) > 1]
    add("run_id_unique", len(duplicates) == 0,
        f"duplicates: {duplicates}" if duplicates else "")

    # Group by exp_id
    from collections import defaultdict
    exp_runs: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        exp_runs[run["exp_id"]].append(run)

    by_exp = {}

    for exp_id, exp_run_list in exp_runs.items():
        exp_checks = []

        def add_exp(name, passed, detail=""):
            exp_checks.append({
                "check": name,
                "status": "pass" if passed else "fail",
                "detail": detail,
            })

        # Config hash consistency
        config_hashes = {r["config_hash"] for r in exp_run_list}
        add_exp(f"[{exp_id}] config_hash_consistent",
                len(config_hashes) == 1,
                f"found {len(config_hashes)} distinct hashes: {config_hashes}" if len(config_hashes) > 1 else "")

        # Seed coverage
        seeds_present = {r["seed"] for r in exp_run_list}
        missing_seeds = expected_seeds - seeds_present
        add_exp(f"[{exp_id}] seeds_complete",
                len(missing_seeds) == 0,
                f"missing seeds: {sorted(missing_seeds)}" if missing_seeds else "")

        # Per-run file checks
        for run in exp_run_list:
            run_label = f"[{exp_id}/seed{run['seed']}]"

            # Train metrics exists + schema
            train_path = run["train_metrics_path"]
            train_exists = os.path.exists(train_path)
            add_exp(f"{run_label} train_exists", train_exists,
                    f"not found: {train_path}" if not train_exists else "")
            train_data = None
            if train_exists:
                try:
                    train_data = load_json(train_path)
                except Exception as e:
                    add_exp(f"{run_label} train_json_load", False, str(e)[:80])
            if train_data is not None and train_schema:
                try:
                    validate(instance=train_data, schema=train_schema)
                    add_exp(f"{run_label} train_schema", True)
                except ValidationError as e:
                    add_exp(f"{run_label} train_schema", False, str(e.message)[:80])
                except Exception as e:
                    add_exp(f"{run_label} train_schema", False, str(e)[:80])
            if train_data is not None:
                train_run_id = train_data.get("run", {}).get("run_id")
                add_exp(
                    f"{run_label} train_run_id_matches_manifest",
                    train_run_id == run.get("run_id"),
                    f"train={train_run_id!r} manifest={run.get('run_id')!r}",
                )
                mismatch = _m5_sweep_metadata_mismatch(train_data, "train", run, manifest)
                add_exp(
                    f"{run_label} train_sweep_metadata_matches_manifest",
                    not mismatch,
                    mismatch,
                )

            # Eval metrics exists + schema
            eval_path = run["eval_metrics_path"]
            eval_exists = os.path.exists(eval_path)
            add_exp(f"{run_label} eval_exists", eval_exists,
                    f"not found: {eval_path}" if not eval_exists else "")
            eval_data = None
            if eval_exists:
                try:
                    eval_data = load_json(eval_path)
                except Exception as e:
                    add_exp(f"{run_label} eval_json_load", False, str(e)[:80])
            if eval_data is not None and eval_schema:
                try:
                    validate(instance=eval_data, schema=eval_schema)
                    add_exp(f"{run_label} eval_schema", True)
                except ValidationError as e:
                    add_exp(f"{run_label} eval_schema", False, str(e.message)[:80])
                except Exception as e:
                    add_exp(f"{run_label} eval_schema", False, str(e)[:80])
            if eval_data is not None:
                eval_run_id = eval_data.get("run_id")
                eval_train_run_id = eval_data.get("model", {}).get("train_run_id")
                add_exp(
                    f"{run_label} eval_run_id_matches_manifest",
                    eval_run_id == run.get("run_id"),
                    f"eval={eval_run_id!r} manifest={run.get('run_id')!r}",
                )
                add_exp(
                    f"{run_label} eval_train_run_id_matches_manifest",
                    eval_train_run_id == run.get("run_id"),
                    f"eval_model_train={eval_train_run_id!r} manifest={run.get('run_id')!r}",
                )
                mismatch = _m5_sweep_metadata_mismatch(eval_data, "eval", run, manifest)
                add_exp(
                    f"{run_label} eval_sweep_metadata_matches_manifest",
                    not mismatch,
                    mismatch,
                )

        by_exp[exp_id] = exp_checks
        checks.extend(exp_checks)

    all_pass = all(c["status"] == "pass" for c in checks)
    return {"checks": checks, "by_exp": by_exp, "all_pass": all_pass}


# ---------------------------------------------------------------------------
# Phase 2: Semantic checks (M7 regression report)
# ---------------------------------------------------------------------------

def semantic_check_m7_regression(reports_dir: str) -> dict:
    """Run semantic checks on M7 regression report.

    Returns dict with:
      - checks: list of individual check results
      - all_pass: bool
    """
    regression_path = os.path.join(reports_dir, "m7_regression_report.json")
    leaderboard_path = os.path.join(reports_dir, "m5_leaderboard.json")

    checks = []

    def add(name, passed, detail=""):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
        })

    # Load files
    if not os.path.exists(regression_path):
        add("regression_report_exists", False, f"File not found: {regression_path}")
        return {"checks": checks, "all_pass": False}
    if not os.path.exists(leaderboard_path):
        add("leaderboard_exists", False, f"File not found: {leaderboard_path}")
        return {"checks": checks, "all_pass": False}

    try:
        regression = load_json(regression_path)
        leaderboard = load_json(leaderboard_path)
    except Exception as e:
        add("json_load", False, str(e))
        return {"checks": checks, "all_pass": False}

    add("regression_report_exists", True)
    add("leaderboard_exists", True)

    # Collect leaderboard exp_ids
    lb_exp_ids = {exp["exp_id"] for exp in leaderboard.get("experiments", [])}

    # Check baseline_exp_id exists in leaderboard
    baseline_exp_id = regression.get("baseline_exp_id", "")
    add("baseline_exp_in_leaderboard",
        baseline_exp_id in lb_exp_ids,
        f"'{baseline_exp_id}' not in leaderboard exp_ids: {sorted(lb_exp_ids)}" if baseline_exp_id not in lb_exp_ids else "")

    # Check each comparison exp_id exists in leaderboard
    comparisons = regression.get("comparisons", [])
    for comp in comparisons:
        exp_id = comp.get("exp_id", "")
        add(f"[{exp_id}] exp_in_leaderboard",
            exp_id in lb_exp_ids,
            f"'{exp_id}' not in leaderboard" if exp_id not in lb_exp_ids else "")

    # Check verdicts are valid
    valid_verdicts = {"improved", "neutral", "regressed"}
    for comp in comparisons:
        verdict = comp.get("verdict", "")
        add(f"[{comp.get('exp_id', '?')}] verdict_valid",
            verdict in valid_verdicts,
            f"invalid verdict: '{verdict}'" if verdict not in valid_verdicts else "")

    # Check summary counts match comparisons
    summary = regression.get("summary", {})
    expected_total = len(comparisons)
    expected_improved = sum(1 for c in comparisons if c.get("verdict") == "improved")
    expected_neutral = sum(1 for c in comparisons if c.get("verdict") == "neutral")
    expected_regressed = sum(1 for c in comparisons if c.get("verdict") == "regressed")

    actual_total = summary.get("total_ablations", -1)
    actual_improved = summary.get("improved", -1)
    actual_neutral = summary.get("neutral", -1)
    actual_regressed = summary.get("regressed", -1)

    counts_match = (actual_total == expected_total and
                    actual_improved == expected_improved and
                    actual_neutral == expected_neutral and
                    actual_regressed == expected_regressed)
    add("summary_counts_consistent", counts_match,
        f"expected total={expected_total} improved={expected_improved} neutral={expected_neutral} regressed={expected_regressed}, "
        f"got total={actual_total} improved={actual_improved} neutral={actual_neutral} regressed={actual_regressed}"
        if not counts_match else "")

    all_pass = all(c["status"] == "pass" for c in checks)
    return {"checks": checks, "all_pass": all_pass}


# ---------------------------------------------------------------------------
# Phase 2: Semantic checks (M9 champion bundle + inference)
# ---------------------------------------------------------------------------

def semantic_check_m9(reports_dir: str, schemas_dir: str = "src/alphatrade/schemas") -> dict:
    """Run semantic checks on M9 bundle manifest and predictions parquet.

    Returns dict with:
      - checks: list of individual check results
      - all_pass: bool
    """
    manifest_path = os.path.join(reports_dir, "m9_model_bundle_manifest.json")
    predictions_path = os.path.join(reports_dir, "m9_predictions.parquet")
    backtest_path = os.path.join(reports_dir, "m9_backtest_metrics.json")

    checks = []

    def add(name, passed, detail="", observed=None):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
            "observed": observed,
        })

    # Check manifest exists and has valid bundle_path
    if not os.path.exists(manifest_path):
        add("manifest_exists", False, f"File not found: {manifest_path}")
        return {"checks": checks, "all_pass": False}

    try:
        manifest = load_json(manifest_path)
    except Exception as e:
        add("manifest_json_load", False, str(e))
        return {"checks": checks, "all_pass": False}

    add("manifest_exists", True)

    bundle_path = manifest.get("bundle_path", "")
    add("bundle_path_exists", os.path.exists(bundle_path),
        f"bundle not found: {bundle_path}" if not os.path.exists(bundle_path) else "")

    model_version = manifest.get("model_version", "")
    add("model_version_non_empty", bool(model_version),
        "model_version is empty" if not model_version else "")
    add("prediction_schema_version_current",
        manifest.get("prediction_schema_version") == prediction_schema.PREDICTION_SCHEMA_VERSION,
        f"got {manifest.get('prediction_schema_version')!r}" if manifest.get("prediction_schema_version") != prediction_schema.PREDICTION_SCHEMA_VERSION else "")

    feature_profile = manifest.get("feature_profile", {})
    model_config = manifest.get("model_config", {})
    profile_cols = feature_profile.get("feature_cols", [])
    feature_profile_ok = (
        bool(feature_profile.get("profile_id"))
        and isinstance(profile_cols, list)
        and len(profile_cols) == int(feature_profile.get("feature_dim", -1))
        and int(model_config.get("num_features", -2)) == int(feature_profile.get("feature_dim", -1))
    )
    add(
        "feature_profile_matches_model_config",
        feature_profile_ok,
        observed={
            "profile_id": feature_profile.get("profile_id"),
            "profile_dim": feature_profile.get("feature_dim"),
            "feature_cols": len(profile_cols) if isinstance(profile_cols, list) else None,
            "model_num_features": model_config.get("num_features"),
        },
    )

    bundle_files = manifest.get("bundle_files", [])
    if bundle_files and os.path.exists(bundle_path):
        file_errors = []
        bundle_root = Path(bundle_path).resolve()
        for item in bundle_files:
            rel_path = item.get("path", "")
            path = (bundle_root / rel_path).resolve()
            if not path.is_relative_to(bundle_root):
                file_errors.append(f"path:{rel_path}")
                continue
            if not path.exists():
                file_errors.append(f"missing:{rel_path}")
                continue
            if path.stat().st_size != item.get("size_bytes"):
                file_errors.append(f"size:{rel_path}")
                continue
            digest = sha256_file(path)
            if digest != item.get("sha256"):
                file_errors.append(f"sha256:{rel_path}")
        add("bundle_files_match_manifest", len(file_errors) == 0,
            f"errors: {file_errors[:5]}" if file_errors else "")
    else:
        add("bundle_files_present", bool(bundle_files),
            "bundle_files is empty" if not bundle_files else "")

    # Check predictions parquet
    if not os.path.exists(predictions_path):
        add("predictions_exists", False, f"File not found: {predictions_path}")
        all_pass = all(c["status"] == "pass" for c in checks)
        return {"checks": checks, "all_pass": all_pass}

    add("predictions_exists", True)

    # Load predictions and check columns/types
    try:
        import pandas as pd
        df = pd.read_parquet(predictions_path)
    except Exception as e:
        add("predictions_parquet_load", False, str(e)[:80])
        all_pass = all(c["status"] == "pass" for c in checks)
        return {"checks": checks, "all_pass": all_pass}

    add("predictions_parquet_load", True)
    add("predictions_rows_gt_0", len(df) > 0,
        f"0 rows" if len(df) == 0 else "")

    issues = prediction_schema.validate_prediction_frame(
        df,
        expected_model_version=model_version or None,
    )
    missing_issue = next((issue for issue in issues if issue.field == "columns"), None)
    add("predictions_columns_complete", missing_issue is None,
        missing_issue.message if missing_issue else "")
    other_issues = [f"{issue.field}: {issue.message}" for issue in issues if issue.field != "columns"]
    add("predictions_schema_invariants", len(other_issues) == 0,
        "; ".join(other_issues[:5]) if other_issues else "")

    # Check backtest handoff metrics if present.
    if not os.path.exists(backtest_path):
        add("backtest_metrics_exists", False, f"File not found: {backtest_path}")
        all_pass = all(c["status"] == "pass" for c in checks)
        return {"checks": checks, "all_pass": all_pass}

    add("backtest_metrics_exists", True)
    try:
        backtest = load_json(backtest_path)
    except Exception as e:
        add("backtest_metrics_json_load", False, str(e))
        all_pass = all(c["status"] == "pass" for c in checks)
        return {"checks": checks, "all_pass": all_pass}

    add("backtest_n_predictions_matches",
        backtest.get("data", {}).get("n_predictions") == len(df),
        f"backtest={backtest.get('data', {}).get('n_predictions')} predictions={len(df)}")
    add("backtest_n_matched_gt_0",
        backtest.get("data", {}).get("n_matched", 0) > 0,
        f"n_matched={backtest.get('data', {}).get('n_matched')}")
    bt_versions = set(backtest.get("model_versions", []))
    pred_versions = set(df["model_version"].astype(str).unique().tolist())
    add("backtest_model_versions_match_predictions",
        bt_versions == pred_versions,
        f"backtest={sorted(bt_versions)} predictions={sorted(pred_versions)}" if bt_versions != pred_versions else "")

    all_pass = all(c["status"] == "pass" for c in checks)
    return {"checks": checks, "all_pass": all_pass}


# ---------------------------------------------------------------------------
# Phase 2: Semantic checks (M11 closure)
# ---------------------------------------------------------------------------

def _m11_get(data: dict, path: list[str], default=None):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def semantic_check_m11(reports_dir: str) -> dict:
    """Run M11 closure semantic checks.

    These checks verify that the M11 decision is internally consistent. They
    intentionally keep schema/contract pass-fail separate from model quality:
    a rejected champion can be a valid M11 closure outcome.
    """
    checks = []

    def add(name, passed, detail="", observed=None):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
            "observed": observed,
        })

    paths = {
        "audit": os.path.join(reports_dir, "m11_target_scale_audit.json"),
        "quality": os.path.join(reports_dir, "m11_model_quality_validation.json"),
        "calibration": os.path.join(reports_dir, "m11_calibration_comparison.json"),
        "metadata": os.path.join(reports_dir, "m11_calibrated_prediction_metadata.json"),
        "memo": os.path.join(reports_dir, "M11_DECISION_MEMO.md"),
    }
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        add("m11_closure_files_exist", False, f"missing: {missing}", observed=paths)
        return {
            "checks": checks,
            "all_pass": False,
            "model_quality_status": "UNKNOWN",
            "separation_note": "Schema validation and model-quality status are reported separately.",
        }
    add("m11_closure_files_exist", True, observed=paths)

    try:
        audit = load_json(paths["audit"])
        quality = load_json(paths["quality"])
        calibration = load_json(paths["calibration"])
        metadata = load_json(paths["metadata"])
        with open(paths["memo"], "r", encoding="utf-8") as f:
            memo = f.read()
    except Exception as e:
        add("m11_closure_files_load", False, str(e))
        return {
            "checks": checks,
            "all_pass": False,
            "model_quality_status": "UNKNOWN",
            "separation_note": "Schema validation and model-quality status are reported separately.",
        }
    add("m11_closure_files_load", True)

    target = audit.get("target_unit_conclusion", {})
    target_scale_ruled_out = (
        target.get("target_unit") == "raw_log_return"
        and target.get("engineering_scale_mismatch_found") is False
        and target.get("inference_inverse_transform_required") is False
    )
    add(
        "target_scale_bug_ruled_out",
        target_scale_ruled_out,
        observed={
            "target_unit": target.get("target_unit"),
            "engineering_scale_mismatch_found": target.get("engineering_scale_mismatch_found"),
            "inference_inverse_transform_required": target.get("inference_inverse_transform_required"),
        },
    )

    output_problem_confirmed = (
        target.get("model_output_scale_mismatch_found") is True
        and quality.get("overall_status") in ("WARN", "FAIL_MODEL_QUALITY")
    )
    add(
        "output_calibration_problem_confirmed",
        output_problem_confirmed,
        observed={
            "model_output_scale_mismatch_found": target.get("model_output_scale_mismatch_found"),
            "model_quality_status": quality.get("overall_status"),
        },
    )

    raw_mae = _m11_get(calibration, ["raw", "quantile_coverage", "overall_mae"])
    calibrated_mae = _m11_get(calibration, ["calibrated", "quantile_coverage", "overall_mae"])
    coverage_delta = _m11_get(
        calibration,
        ["raw_vs_calibrated", "coverage_mae_delta_raw_minus_calibrated"],
    )
    coverage_improved = (
        raw_mae is not None
        and calibrated_mae is not None
        and coverage_delta is not None
        and calibrated_mae < raw_mae
        and coverage_delta > 0.01
    )
    add(
        "posthoc_calibration_improves_coverage",
        coverage_improved,
        observed={
            "raw_coverage_mae": raw_mae,
            "calibrated_coverage_mae": calibrated_mae,
            "coverage_delta": coverage_delta,
        },
    )

    rolling_pinball = _m11_get(
        calibration,
        ["rolling_historical_recomparison", "rolling_historical_quantile", "pinball_loss", "overall"],
    )
    calibrated_common_pinball = _m11_get(
        calibration,
        ["rolling_historical_recomparison", "calibrated_model_on_common", "pinball_loss", "overall"],
    )
    rolling_minus_calibrated = _m11_get(
        calibration,
        ["rolling_historical_recomparison", "pinball_delta_rolling_minus_calibrated"],
    )
    loses_to_rolling = (
        rolling_pinball is not None
        and calibrated_common_pinball is not None
        and calibrated_common_pinball > rolling_pinball
        and rolling_minus_calibrated is not None
        and rolling_minus_calibrated < 0.0
    )
    add(
        "calibrated_model_still_loses_to_rolling_historical",
        loses_to_rolling,
        observed={
            "calibrated_pinball": calibrated_common_pinball,
            "rolling_historical_pinball": rolling_pinball,
            "rolling_minus_calibrated": rolling_minus_calibrated,
        },
    )

    promotion = metadata.get("promotion", {})
    metadata_not_promoted = (
        promotion.get("promoted") is False
        and promotion.get("promotion_status") == "not_promoted"
        and _m11_get(
            metadata,
            ["calibrated_prediction_identity", "calibrated_predictions_written"],
        ) is False
    )
    add(
        "calibrated_candidate_not_promoted",
        metadata_not_promoted,
        observed={
            "promoted": promotion.get("promoted"),
            "promotion_status": promotion.get("promotion_status"),
            "calibrated_predictions_written": _m11_get(
                metadata,
                ["calibrated_prediction_identity", "calibrated_predictions_written"],
            ),
        },
    )

    champion_rejected = (
        _m11_get(calibration, ["conclusion", "champion_status"])
        == "rejected_pending_retrain_or_stronger_fix"
        and promotion.get("champion_status") == "rejected_pending_retrain_or_stronger_fix"
    )
    add(
        "champion_remains_rejected",
        champion_rejected,
        observed={
            "calibration_champion_status": _m11_get(calibration, ["conclusion", "champion_status"]),
            "metadata_champion_status": promotion.get("champion_status"),
        },
    )

    memo_has_m12 = "M12 feature/label/baseline expansion" in memo
    add(
        "m12_next_milestone_recorded",
        memo_has_m12,
        observed={"memo": paths["memo"]},
    )

    all_pass = all(c["status"] == "pass" for c in checks)
    return {
        "checks": checks,
        "all_pass": all_pass,
        "model_quality_status": quality.get("overall_status", "UNKNOWN"),
        "separation_note": (
            "Schema ALL PASS means report contracts are valid. Model-quality "
            "PASS/WARN/FAIL_MODEL_QUALITY is reported separately and can be "
            "FAIL_MODEL_QUALITY for a valid rejected-champion closure."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 2: Semantic checks (M12 Chendage processed features)
# ---------------------------------------------------------------------------

def semantic_check_m12(reports_dir: str) -> dict:
    """Run semantic checks on the M12 Chendage feature contract."""
    path = os.path.join(reports_dir, "m12_chendage_feature_contract.json")
    checks = []

    def add(name, passed, detail="", observed=None):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
            "observed": observed,
        })

    if not os.path.exists(path):
        add("contract_exists", False, f"File not found: {path}")
        return {"checks": checks, "all_pass": False}
    add("contract_exists", True, observed=path)

    try:
        report = load_json(path)
    except Exception as e:
        add("contract_json_load", False, str(e))
        return {"checks": checks, "all_pass": False}
    add("contract_json_load", True)

    add(
        "overall_status_pass",
        report.get("overall_status") == "PASS",
        observed=report.get("overall_status"),
    )

    failed_contract_checks = [
        c for c in report.get("checks", [])
        if c.get("status") not in ("PASS", "WARN")
    ]
    add(
        "contract_checks_no_fail",
        len(failed_contract_checks) == 0,
        observed=failed_contract_checks[:10],
    )

    profile = report.get("feature_profile", {})
    chg_cols = profile.get("chendage_feature_cols", [])
    add(
        "feature_profile_wider_than_base8",
        profile.get("feature_dim", 0) > 8 and len(chg_cols) > 0,
        observed={"feature_dim": profile.get("feature_dim"), "chendage_feature_cols": len(chg_cols)},
    )
    normalization = report.get("normalization", {})
    add(
        "train_only_scaler_recorded",
        normalization.get("train_only") is True
        and bool(normalization.get("scaler_hash"))
        and profile.get("scaler_hash") == normalization.get("scaler_hash"),
        observed={
            "profile_scaler_hash": profile.get("scaler_hash"),
            "normalization": normalization,
        },
    )

    boundary = report.get("source_boundary", {})
    expected_versions = boundary.get("expected_source_schema_versions", {})
    observed_versions = boundary.get("observed_source_schema_versions", {})
    source_versions_ok = (
        set(observed_versions.get("schema_version", [])) == {expected_versions.get("schema_version")}
        and set(observed_versions.get("feature_vector_version", []))
        == {expected_versions.get("feature_vector_version")}
    )
    add(
        "source_schema_versions_match_expected",
        source_versions_ok,
        observed={"expected": expected_versions, "observed": observed_versions},
    )
    rule_only_found = boundary.get("rule_only_fields_found", [])
    add(
        "rule_only_fields_absent",
        not rule_only_found,
        observed=rule_only_found,
    )
    add(
        "source_manifest_provenance_recorded",
        bool(boundary.get("input_files"))
        and bool(boundary.get("input_hashes"))
        and bool(boundary.get("chendage_commit")),
        observed={
            "input_files": boundary.get("input_files"),
            "input_hashes": boundary.get("input_hashes"),
            "chendage_commit": boundary.get("chendage_commit"),
        },
    )

    causality = report.get("causality_test", {})
    causality_checks = causality.get("checks", {})
    causality_pass = bool(causality_checks) and all(
        item.get("status") == "PASS" for item in causality_checks.values()
    )
    add(
        "causality_checks_pass",
        causality_pass,
        observed=causality_checks,
    )

    symbol_results = report.get("symbols", [])
    all_symbols_success = bool(symbol_results) and all(
        item.get("status") == "SUCCESS" for item in symbol_results
    )
    add("symbols_all_success", all_symbols_success, observed=symbol_results[:10])

    total_common_rows = sum(int(item.get("common_rows", 0)) for item in symbol_results)
    total_samples = sum(
        int(item.get("train_samples", 0))
        + int(item.get("val_samples", 0))
        + int(item.get("test_samples", 0))
        for item in symbol_results
    )
    add(
        "common_rows_and_samples_positive",
        total_common_rows > 0 and total_samples > 0,
        observed={"common_rows": total_common_rows, "samples": total_samples},
    )
    coverage_ok = bool(symbol_results) and all(
        item.get("feature_coverage", {}).get("status") == "PASS"
        for item in symbol_results
    )
    add(
        "feature_coverage_within_threshold",
        coverage_ok,
        observed={
            item.get("symbol"): item.get("feature_coverage")
            for item in symbol_results
        },
    )
    distribution_recorded = bool(symbol_results) and all(
        all(split in item.get("feature_distribution_by_split", {}) for split in ("train", "val", "test"))
        for item in symbol_results
        if item.get("status") == "SUCCESS"
    )
    add(
        "feature_distribution_by_split_recorded",
        distribution_recorded,
        observed={
            item.get("symbol"): list(item.get("feature_distribution_by_split", {}).keys())
            for item in symbol_results
            if item.get("status") == "SUCCESS"
        },
    )

    outputs = report.get("outputs", {})
    candidate_root = outputs.get("candidate_processed_root")
    control_root = outputs.get("control_processed_root")
    add(
        "candidate_and_control_roots_exist",
        bool(candidate_root) and os.path.exists(candidate_root)
        and bool(control_root) and os.path.exists(control_root),
        observed={"candidate": candidate_root, "control": control_root},
    )

    all_pass = all(c["status"] == "pass" for c in checks)
    return {"checks": checks, "all_pass": all_pass}


# ---------------------------------------------------------------------------
# Phase 2: Semantic checks (MG0-A frozen audit)
# ---------------------------------------------------------------------------

def semantic_check_mg0(
    reports_dir: str,
    schemas_dir: str = "src/alphatrade/schemas",
) -> dict:
    """Verify the frozen MG0-A payload, artifacts, and deferred-work policy."""
    audit_path = Path(reports_dir) / "mg0_alphagenome_parity_audit.json"
    markdown_path = Path(reports_dir) / "mg0_alphagenome_parity_audit.md"
    schema_path = Path(schemas_dir) / "mg0_alphagenome_parity_audit.schema.json"
    checks = []

    def add(name, passed, detail="", observed=None):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
            "observed": observed,
        })

    artifact_paths = {
        "audit": audit_path,
        "markdown": markdown_path,
        "schema": schema_path,
    }
    invalid_artifacts = [
        name for name, path in artifact_paths.items() if not path.is_file()
    ]
    observed_paths = {name: str(path) for name, path in artifact_paths.items()}
    add(
        "freeze_artifacts_exist",
        not invalid_artifacts,
        f"missing or non-file: {invalid_artifacts}" if invalid_artifacts else "",
        observed_paths,
    )
    if invalid_artifacts:
        return {"checks": checks, "all_pass": False}

    try:
        audit = load_json(str(audit_path))
    except Exception as e:
        add("freeze_payload_load", False, str(e))
        return {"checks": checks, "all_pass": False}
    if not isinstance(audit, dict):
        add("freeze_payload_load", False, "audit payload must be a JSON object")
        return {"checks": checks, "all_pass": False}
    freeze = audit.get("freeze")
    if not isinstance(freeze, dict):
        add("freeze_payload_load", False, "freeze must be a JSON object")
        return {"checks": checks, "all_pass": False}
    add("freeze_payload_load", True)

    baseline_payload = dict(audit)
    baseline_payload.pop("freeze", None)
    canonical_payload = json.dumps(
        baseline_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    observed_payload_hash = hashlib.sha256(canonical_payload).hexdigest()
    add(
        "baseline_payload_hash_matches",
        observed_payload_hash == freeze.get("baseline_payload_sha256"),
        observed={
            "expected": freeze.get("baseline_payload_sha256"),
            "observed": observed_payload_hash,
        },
    )

    def add_file_hash_check(check_name, path, expected):
        try:
            observed = sha256_file(path)
        except OSError as e:
            add(
                check_name,
                False,
                f"{type(e).__name__}: {e}",
                {"expected": expected, "observed": None},
            )
            return None
        add(
            check_name,
            observed == expected,
            observed={"expected": expected, "observed": observed},
        )
        return observed

    add_file_hash_check(
        "markdown_hash_matches", markdown_path, freeze.get("markdown_sha256")
    )
    add_file_hash_check(
        "schema_hash_matches", schema_path, freeze.get("schema_sha256")
    )

    validator_path = Path(__file__).resolve()
    add_file_hash_check(
        "validator_hash_matches",
        validator_path,
        freeze.get("validator_sha256"),
    )

    repo_root = validator_path.parents[3]
    git_tag = freeze.get("git_tag")
    tag_files = {
        "audit": (
            "docs/alphaTrade/market_genome/MG0_ALPHAGENOME_PARITY_AUDIT.json",
            audit_path,
        ),
        "markdown": (
            "docs/alphaTrade/market_genome/MG0_ALPHAGENOME_PARITY_AUDIT.md",
            markdown_path,
        ),
        "schema": (
            "src/alphatrade/schemas/mg0_alphagenome_parity_audit.schema.json",
            schema_path,
        ),
        "validator": (
            "src/alphatrade/scripts/validate_reports_schema.py",
            validator_path,
        ),
        "manifest": (
            "src/alphatrade/schemas/contracts_manifest.yaml",
            repo_root / "src/alphatrade/schemas/contracts_manifest.yaml",
        ),
        "validator_tests": (
            "src/alphatrade/tests/test_reports_validator.py",
            repo_root / "src/alphatrade/tests/test_reports_validator.py",
        ),
    }
    tag_commit = None
    tag_mismatches = []
    tag_error = ""
    try:
        if not isinstance(git_tag, str) or not git_tag:
            raise ValueError("freeze.git_tag must be a non-empty string")
        resolved = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--verify",
                f"refs/tags/{git_tag}^{{commit}}",
            ],
            capture_output=True,
            check=False,
        )
        if resolved.returncode != 0:
            raise RuntimeError(
                resolved.stderr.decode("utf-8", errors="replace").strip()
                or f"unable to resolve tag {git_tag}"
            )
        tag_commit = resolved.stdout.decode("ascii").strip()
        for label, (repo_path, current_path) in tag_files.items():
            tagged = subprocess.run(
                ["git", "-C", str(repo_root), "show", f"{tag_commit}:{repo_path}"],
                capture_output=True,
                check=False,
            )
            if tagged.returncode != 0:
                tag_mismatches.append(f"{label}:missing_in_tag")
            elif not current_path.exists():
                tag_mismatches.append(f"{label}:missing_current")
            elif tagged.stdout != current_path.read_bytes():
                tag_mismatches.append(f"{label}:content_mismatch")
    except Exception as e:
        tag_error = str(e)
    add(
        "git_tag_anchor_matches",
        not tag_error and not tag_mismatches,
        tag_error or (f"mismatches: {tag_mismatches}" if tag_mismatches else ""),
        {"tag": git_tag, "commit": tag_commit, "mismatches": tag_mismatches},
    )

    metadata_value = audit.get("metadata")
    metadata = metadata_value if isinstance(metadata_value, dict) else {}
    gate_value = audit.get("gate")
    gate = gate_value if isinstance(gate_value, dict) else {}
    conclusion_ok = (
        metadata.get("audit_status") == "FROZEN_WITH_NEEDS_VERIFICATION"
        and metadata.get("parity_verdict") == "PARTIAL_PUBLISHED_CODE_PARITY"
        and metadata.get("gate_decision") == "PASS_WITH_NEEDS_VERIFICATION"
        and gate.get("overall") == metadata.get("gate_decision")
        and freeze.get("status") == "FROZEN"
        and freeze.get("review_decision") == "APPROVED_WITH_DOCUMENTED_GAPS"
    )
    add(
        "frozen_conclusion_consistent",
        conclusion_ok,
        observed={
            "audit_status": metadata.get("audit_status"),
            "parity_verdict": metadata.get("parity_verdict"),
            "metadata_gate": metadata.get("gate_decision"),
            "gate_overall": gate.get("overall"),
            "freeze_status": freeze.get("status"),
            "review_decision": freeze.get("review_decision"),
        },
    )

    source_baseline_ok = (
        freeze.get("official_commit") == metadata.get("official_commit")
        and freeze.get("local_source_head") == metadata.get("local_head")
    )
    add(
        "source_baseline_fields_consistent",
        source_baseline_ok,
        observed={
            "metadata_official_commit": metadata.get("official_commit"),
            "freeze_official_commit": freeze.get("official_commit"),
            "metadata_local_head": metadata.get("local_head"),
            "freeze_local_source_head": freeze.get("local_source_head"),
            "official_source_tree": freeze.get("official_source_tree"),
        },
    )

    def list_count(field_name):
        value = audit.get(field_name)
        return len(value) if isinstance(value, list) else None

    observed_counts = {
        "component_count": list_count("component_map"),
        "uncertainty_count": list_count("uncertainties"),
        "non_negotiable_principle_count": list_count(
            "non_negotiable_alpha_genome_principles"
        ),
        "non_isomorphic_component_count": list_count(
            "non_isomorphic_components"
        ),
    }
    expected_counts = {name: freeze.get(name) for name in observed_counts}
    add(
        "frozen_counts_match",
        observed_counts == expected_counts,
        observed={"expected": expected_counts, "observed": observed_counts},
    )

    try:
        generated_at = datetime.fromisoformat(
            metadata["generated_at"].replace("Z", "+00:00")
        )
        frozen_at = datetime.fromisoformat(
            freeze["frozen_at"].replace("Z", "+00:00")
        )
        timestamp_ok = (
            generated_at.tzinfo is not None
            and frozen_at.tzinfo is not None
            and frozen_at >= generated_at
        )
        timestamp_detail = ""
    except Exception as e:
        timestamp_ok = False
        timestamp_detail = str(e)
    add(
        "freeze_timestamp_valid",
        timestamp_ok,
        timestamp_detail,
        {
            "generated_at": metadata.get("generated_at"),
            "frozen_at": freeze.get("frozen_at"),
        },
    )

    execution_value = audit.get("execution")
    execution = execution_value if isinstance(execution_value, dict) else {}
    deferred_ok = (
        freeze.get("training_status") == "DEFERRED_BY_USER"
        and freeze.get("gpu_status")
        == "DEFERRED_UNTIL_EXPLICIT_USER_REACTIVATION"
        and freeze.get("resume_requires_explicit_user_confirmation") is True
        and execution.get("training_execution_policy") == "DEFERRED_BY_USER"
        and execution.get("training_workload_after_defer") is False
        and execution.get("gpu_execution_policy") == "DEFERRED_BY_USER"
        and execution.get("gpu_workload_after_defer") is False
        and execution.get("resume_requires_explicit_user_confirmation") is True
    )
    add(
        "training_and_gpu_remain_deferred",
        deferred_ok,
        observed={
            "freeze_training": freeze.get("training_status"),
            "freeze_gpu": freeze.get("gpu_status"),
            "gpu_revisit_not_before": freeze.get("gpu_revisit_not_before"),
            "execution_training": execution.get("training_execution_policy"),
            "execution_gpu": execution.get("gpu_execution_policy"),
            "explicit_confirmation": execution.get(
                "resume_requires_explicit_user_confirmation"
            ),
        },
    )

    uncertainties_value = audit.get("uncertainties")
    uncertainties = uncertainties_value if isinstance(uncertainties_value, list) else []
    uncertainty_statuses = [
        item.get("status") if isinstance(item, dict) else None
        for item in uncertainties
    ]
    add(
        "uncertainties_remain_open",
        bool(uncertainties)
        and all(status == "NEEDS_VERIFICATION" for status in uncertainty_statuses),
        observed=uncertainty_statuses,
    )

    all_pass = all(c["status"] == "pass" for c in checks)
    return {"checks": checks, "all_pass": all_pass}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_STATUS_ICON = {
    "pass": "\u2705",
    "fail": "\u274c",
    "missing_report": "\u26a0\ufe0f",
    "missing_schema": "\u26a0\ufe0f",
    "invalid_json": "\u274c",
    "error": "\u274c",
    "warning": "\u26a0\ufe0f",
}


def generate_md(schema_results, semantic_results, profile_name: str, strict: bool) -> str:
    lines = []
    w = lines.append

    w(f"# Reports Schema & Semantic Validation (profile: {profile_name})\n")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Strict: {strict}\n")

    # --- Phase 1 ---
    w("## Phase 1: Schema Validation\n")

    total = len(schema_results)
    passed = sum(1 for r in schema_results if r["status"] == "pass")
    failed = total - passed

    w(f"- Total: {total}")
    w(f"- Passed: {passed}")
    w(f"- Failed/Missing: {failed}\n")

    w("| Report | Required | Status | Error |")
    w("|--------|----------|--------|-------|")
    for r in schema_results:
        icon = _STATUS_ICON.get(r["status"], "\u2753")
        err = r["error"] or "-"
        if len(err) > 60:
            err = err[:57] + "..."
        req = "\u2705" if r.get("required") else "-"
        w(f"| {r['name']} | {req} | {icon} {r['status']} | {err} |")
    w("")

    # --- Phase 2 ---
    if profile_name == "mg0":
        _generate_md_phase2_mg0(w, semantic_results)
    elif profile_name == "m9":
        _generate_md_phase2_m9(w, semantic_results)
    elif profile_name == "m10":
        _generate_md_phase2_schema_only(w, profile_name)
    elif profile_name == "m11":
        _generate_md_phase2_m11(w, semantic_results)
    elif profile_name == "m12":
        _generate_md_phase2_m12(w, semantic_results)
    elif profile_name in ("m5", "m6", "m7"):
        _generate_md_phase2_m5(w, semantic_results)
        if profile_name == "m7":
            _generate_md_phase2_m7(w, semantic_results)
    else:
        _generate_md_phase2_m4(w, semantic_results)

    # --- Overall ---
    schema_ok = all(r["status"] == "pass" for r in schema_results if r.get("required"))
    sem_ok = _semantic_all_pass(semantic_results, profile_name)

    w("## Overall\n")
    schema_verdict = "\u2705 pass" if schema_ok else "\u274c fail"
    sem_verdict = "\u2705 pass" if sem_ok else "\u274c fail"
    w(f"- Schema (required): {schema_verdict}")
    w(f"- Semantic: {sem_verdict}")
    overall = "\u2705 ALL PASS" if (schema_ok and sem_ok) else "\u274c FAIL"
    w(f"- **Overall: {overall}**")
    w("")

    return "\n".join(lines)


def _semantic_all_pass(semantic_results, profile_name: str) -> bool:
    """Check if all semantic checks passed."""
    if profile_name == "mg0":
        if not semantic_results:
            return True
        return semantic_results.get("all_pass", True)
    elif profile_name == "m9":
        if not semantic_results:
            return True
        return semantic_results.get("all_pass", True)
    elif profile_name == "m10":
        return True
    elif profile_name == "m11":
        if not semantic_results:
            return True
        return semantic_results.get("all_pass", True)
    elif profile_name == "m12":
        if not semantic_results:
            return True
        return semantic_results.get("all_pass", True)
    elif profile_name in ("m5", "m6", "m7"):
        # semantic_results is a dict from semantic_check_m5
        if not semantic_results:
            return True
        return semantic_results.get("all_pass", True)
    else:
        # semantic_results is a list of {seed, checks} dicts
        if not semantic_results:
            return True
        return all(
            c["status"] == "pass"
            for sr in semantic_results
            for c in sr["checks"]
        )


def _generate_md_phase2_m4(w, semantic_results):
    """Generate Phase 2 markdown for M4."""
    w("## Phase 2: M4 Semantic Checks\n")

    if not semantic_results:
        w("_No M4 eval\u2194train pairs found._\n")
    else:
        all_checks = []
        for sr in semantic_results:
            w(f"### Seed {sr['seed']}\n")
            w(f"- eval: `{sr['eval_path']}`")
            w(f"- train: `{sr['train_path']}`\n")
            w("| Check | Status | Detail |")
            w("|-------|--------|--------|")
            for c in sr["checks"]:
                icon = "\u2705" if c["status"] == "pass" else "\u274c"
                w(f"| {c['check']} | {icon} | {c['detail'] or '-'} |")
            w("")
            all_checks.extend(sr["checks"])

        sem_pass = sum(1 for c in all_checks if c["status"] == "pass")
        sem_total = len(all_checks)
        w(f"**Semantic summary**: {sem_pass}/{sem_total} checks passed\n")


def _generate_md_phase2_m5(w, semantic_results):
    """Generate Phase 2 markdown for M5 sweep."""
    w("## Phase 2: M5 Sweep Semantic Checks\n")

    if not semantic_results or not semantic_results.get("checks"):
        w("_No M5 sweep semantic checks run._\n")
        return

    checks = semantic_results["checks"]
    by_exp = semantic_results.get("by_exp", {})

    if by_exp:
        for exp_id, exp_checks in by_exp.items():
            w(f"### Experiment: {exp_id}\n")
            w("| Check | Status | Detail |")
            w("|-------|--------|--------|")
            for c in exp_checks:
                icon = "\u2705" if c["status"] == "pass" else "\u274c"
                w(f"| {c['check']} | {icon} | {c['detail'] or '-'} |")
            w("")
    else:
        w("| Check | Status | Detail |")
        w("|-------|--------|--------|")
        for c in checks:
            icon = "\u2705" if c["status"] == "pass" else "\u274c"
            w(f"| {c['check']} | {icon} | {c['detail'] or '-'} |")
        w("")

    sem_pass = sum(1 for c in checks if c["status"] == "pass")
    sem_total = len(checks)
    w(f"**Semantic summary**: {sem_pass}/{sem_total} checks passed\n")


def _generate_md_phase2_m7(w, semantic_results):
    """Generate Phase 2 markdown for M7 regression checks."""
    regression_result = semantic_results.get("m7_regression")
    if not regression_result or not regression_result.get("checks"):
        return

    w("## Phase 2b: M7 Regression Report Checks\n")
    checks = regression_result["checks"]
    w("| Check | Status | Detail |")
    w("|-------|--------|--------|")
    for c in checks:
        icon = "\u2705" if c["status"] == "pass" else "\u274c"
        w(f"| {c['check']} | {icon} | {c['detail'] or '-'} |")
    w("")

    sem_pass = sum(1 for c in checks if c["status"] == "pass")
    sem_total = len(checks)
    w(f"**Regression checks**: {sem_pass}/{sem_total} passed\n")


def _generate_md_phase2_m9(w, semantic_results):
    """Generate Phase 2 markdown for M9."""
    w("## Phase 2: M9 Bundle + Predictions Checks\n")

    if not semantic_results or not semantic_results.get("checks"):
        w("_No M9 semantic checks run._\n")
        return

    checks = semantic_results["checks"]
    w("| Check | Status | Detail |")
    w("|-------|--------|--------|")
    for c in checks:
        icon = "\u2705" if c["status"] == "pass" else "\u274c"
        w(f"| {c['check']} | {icon} | {c['detail'] or '-'} |")
    w("")

    sem_pass = sum(1 for c in checks if c["status"] == "pass")
    sem_total = len(checks)
    w(f"**M9 checks**: {sem_pass}/{sem_total} passed\n")


def _generate_md_phase2_m11(w, semantic_results):
    """Generate Phase 2 markdown for M11 closure checks."""
    w("## Phase 2: M11 Closure Semantic Checks\n")

    if not semantic_results or not semantic_results.get("checks"):
        w("_No M11 closure semantic checks run._\n")
        return

    w(f"- Model-quality status: `{semantic_results.get('model_quality_status', 'UNKNOWN')}`")
    w(f"- Separation note: {semantic_results.get('separation_note', '')}\n")
    w("| Check | Status | Detail | Observed |")
    w("|-------|--------|--------|----------|")
    for c in semantic_results["checks"]:
        icon = "\u2705" if c["status"] == "pass" else "\u274c"
        observed = c.get("observed")
        observed_text = "-" if observed is None else json.dumps(observed, sort_keys=True)[:180]
        w(f"| {c['check']} | {icon} {c['status']} | {c.get('detail') or '-'} | `{observed_text}` |")
    w("")

    sem_pass = sum(1 for c in semantic_results["checks"] if c["status"] == "pass")
    sem_total = len(semantic_results["checks"])
    w(f"**M11 closure checks**: {sem_pass}/{sem_total} passed\n")


def _generate_md_phase2_m12(w, semantic_results):
    """Generate Phase 2 markdown for M12 Chendage feature checks."""
    w("## Phase 2: M12 Chendage Feature Semantic Checks\n")

    if not semantic_results or not semantic_results.get("checks"):
        w("_No M12 semantic checks run._\n")
        return

    w("| Check | Status | Detail | Observed |")
    w("|-------|--------|--------|----------|")
    for c in semantic_results["checks"]:
        icon = "\u2705" if c["status"] == "pass" else "\u274c"
        observed = c.get("observed")
        observed_text = "-" if observed is None else json.dumps(observed, sort_keys=True)[:180]
        w(f"| {c['check']} | {icon} {c['status']} | {c.get('detail') or '-'} | `{observed_text}` |")
    w("")

    sem_pass = sum(1 for c in semantic_results["checks"] if c["status"] == "pass")
    sem_total = len(semantic_results["checks"])
    w(f"**M12 checks**: {sem_pass}/{sem_total} passed\n")


def _generate_md_phase2_schema_only(w, profile_name: str):
    """Generate Phase 2 markdown for schema-only profiles."""
    w(f"## Phase 2: {profile_name.upper()} Semantic Checks\n")
    w("_No additional semantic checks are defined for this profile._\n")


def _generate_md_phase2_mg0(w, semantic_results):
    """Generate Phase 2 markdown for the MG0-A freeze contract."""
    w("## Phase 2: MG0 Freeze Semantic Checks\n")
    if not semantic_results or not semantic_results.get("checks"):
        w("_No MG0 freeze checks run._\n")
        return

    w("| Check | Status | Detail | Observed |")
    w("|-------|--------|--------|----------|")
    for check in semantic_results["checks"]:
        icon = "\u2705" if check["status"] == "pass" else "\u274c"
        observed = check.get("observed")
        observed_text = (
            "-" if observed is None else json.dumps(observed, sort_keys=True)[:180]
        )
        w(
            f"| {check['check']} | {icon} {check['status']} | "
            f"{check.get('detail') or '-'} | `{observed_text}` |"
        )
    w("")

    passed = sum(
        1 for check in semantic_results["checks"] if check["status"] == "pass"
    )
    total = len(semantic_results["checks"])
    w(f"**MG0 freeze checks**: {passed}/{total} passed\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    profile_name = args.profile
    strict = args.strict
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    reports_dir_str = str(reports_dir)

    # Output paths
    json_out = args.output_json or str(reports_dir / f"{profile_name}_schema_validation.json")
    md_out = args.output_md or str(reports_dir / f"{profile_name}_schema_validation.md")

    print(f"\n{'='*60}")
    print(f"Reports Schema & Semantic Validation")
    print(f"{'='*60}")
    print(f"Profile:     {profile_name}")
    print(f"Strict:      {strict}")
    print(f"Reports dir: {reports_dir_str}")
    print(f"Manifest:    {args.manifest}")
    print(f"{'='*60}\n")

    # ── Load manifest ──
    manifest = load_manifest(args.manifest)
    use_manifest = manifest is not None

    if use_manifest:
        items = get_profile_items(manifest, profile_name)
        if items is None:
            print(f"ERROR: profile '{profile_name}' not found in manifest")
            available = list(manifest.get("profiles", {}).keys())
            print(f"Available profiles: {available}")
            sys.exit(1)
        validations = []
        for item in items:
            report_path = os.path.join(reports_dir_str, item["path"])
            schema_path = item.get("schema", "")
            validations.append({
                "name": item["name"],
                "report": report_path,
                "schema": schema_path if schema_path else "",
                "required": item.get("required", True),
            })
        print(f"Loaded manifest profile '{profile_name}': {len(validations)} items\n")
    else:
        warnings.warn(f"Manifest not found at {args.manifest} — using legacy hardcoded validations")
        validations = _legacy_validations(reports_dir_str, args.schemas_dir)
        print(f"Legacy mode: {len(validations)} items\n")

    # ── Phase 1: Schema validation ──
    print("Phase 1: Schema validation\n")
    schema_results = []
    for val in validations:
        print(f"  {val['name']}...", end=" ")
        result = validate_report(val["report"], val["schema"] if val.get("schema") else "")
        result["name"] = val["name"]
        result["required"] = val.get("required", True)
        schema_results.append(result)
        icon = _STATUS_ICON.get(result["status"], "\u2753")
        req_tag = " [required]" if result["required"] else " [optional]"
        print(f"{icon} {result['status']}{req_tag}")
        if result["error"]:
            print(f"    Error: {result['error'][:80]}")

    schema_pass = sum(1 for r in schema_results if r["status"] == "pass")
    schema_total = len(schema_results)
    schema_required_fail = sum(
        1 for r in schema_results
        if r.get("required") and r["status"] != "pass"
    )
    print(f"\n  Schema: {schema_pass}/{schema_total} passed ({schema_required_fail} required failures)\n")

    # ── Phase 2: Semantic checks ──
    if profile_name == "mg0":
        print("Phase 2: MG0 freeze semantic checks\n")
        semantic_results = semantic_check_mg0(reports_dir_str, args.schemas_dir)
        for check in semantic_results["checks"]:
            icon = "\u2705" if check["status"] == "pass" else "\u274c"
            detail = f" ({check['detail']})" if check.get("detail") else ""
            print(f"  {icon} {check['check']}{detail}")
        sem_all_pass = semantic_results["all_pass"]
        sem_total = len(semantic_results["checks"])
        sem_pass = sum(
            1 for check in semantic_results["checks"]
            if check["status"] == "pass"
        )
        print(f"\n  Semantic: {sem_pass}/{sem_total} passed\n")
    elif profile_name == "m9":
        print("Phase 2: M9 bundle + predictions checks\n")
        semantic_results = semantic_check_m9(reports_dir_str, args.schemas_dir)
        for c in semantic_results["checks"]:
            icon = "\u2705" if c["status"] == "pass" else "\u274c"
            detail = f" ({c['detail']})" if c["detail"] else ""
            print(f"  {icon} {c['check']}{detail}")
        sem_all_pass = semantic_results["all_pass"]
        sem_total = len(semantic_results["checks"])
        sem_pass = sum(1 for c in semantic_results["checks"] if c["status"] == "pass")
        print(f"\n  Semantic: {sem_pass}/{sem_total} passed\n")
    elif profile_name == "m10":
        print("Phase 2: M10 semantic checks\n")
        print("  No additional semantic checks are defined for M10.")
        semantic_results = {"checks": [], "all_pass": True}
        sem_all_pass = True
        sem_total = 0
        sem_pass = 0
        print(f"\n  Semantic: {sem_pass}/{sem_total} passed\n")
    elif profile_name == "m11":
        print("Phase 2: M11 closure semantic checks\n")
        semantic_results = semantic_check_m11(reports_dir_str)
        print(f"  Model-quality status: {semantic_results.get('model_quality_status', 'UNKNOWN')}")
        print(f"  {semantic_results.get('separation_note', '')}")
        for c in semantic_results["checks"]:
            icon = "\u2705" if c["status"] == "pass" else "\u274c"
            detail = f" ({c['detail']})" if c.get("detail") else ""
            print(f"  {icon} {c['check']}{detail}")
        sem_all_pass = semantic_results["all_pass"]
        sem_total = len(semantic_results["checks"])
        sem_pass = sum(1 for c in semantic_results["checks"] if c["status"] == "pass")
        print(f"\n  M11 closure semantic: {sem_pass}/{sem_total} passed\n")
    elif profile_name == "m12":
        print("Phase 2: M12 Chendage feature semantic checks\n")
        semantic_results = semantic_check_m12(reports_dir_str)
        for c in semantic_results["checks"]:
            icon = "\u2705" if c["status"] == "pass" else "\u274c"
            detail = f" ({c['detail']})" if c.get("detail") else ""
            print(f"  {icon} {c['check']}{detail}")
        sem_all_pass = semantic_results["all_pass"]
        sem_total = len(semantic_results["checks"])
        sem_pass = sum(1 for c in semantic_results["checks"] if c["status"] == "pass")
        print(f"\n  M12 semantic: {sem_pass}/{sem_total} passed\n")
    elif profile_name in ("m5", "m6", "m7"):
        print("Phase 2: M5 sweep semantic checks\n")
        semantic_results = semantic_check_m5(reports_dir_str, args.schemas_dir)
        for c in semantic_results["checks"]:
            icon = "\u2705" if c["status"] == "pass" else "\u274c"
            detail = f" ({c['detail']})" if c["detail"] else ""
            print(f"  {icon} {c['check']}{detail}")
        sem_all_pass = semantic_results["all_pass"]
        sem_total = len(semantic_results["checks"])
        sem_pass = sum(1 for c in semantic_results["checks"] if c["status"] == "pass")
        print(f"\n  Semantic: {sem_pass}/{sem_total} passed\n")

        # M7: additional regression report checks
        if profile_name == "m7":
            print("Phase 2b: M7 regression report checks\n")
            m7_result = semantic_check_m7_regression(reports_dir_str)
            semantic_results["m7_regression"] = m7_result
            for c in m7_result["checks"]:
                icon = "\u2705" if c["status"] == "pass" else "\u274c"
                detail = f" ({c['detail']})" if c["detail"] else ""
                print(f"  {icon} {c['check']}{detail}")
            if not m7_result["all_pass"]:
                sem_all_pass = False
            m7_total = len(m7_result["checks"])
            m7_pass = sum(1 for c in m7_result["checks"] if c["status"] == "pass")
            sem_total += m7_total
            sem_pass += m7_pass
            print(f"\n  Regression checks: {m7_pass}/{m7_total} passed\n")
    else:
        print("Phase 2: M4 semantic checks\n")
        semantic_results = []
        seed_pairs = discover_m4_seed_pairs(reports_dir_str)

        for sp in seed_pairs:
            print(f"  Seed {sp['seed']}:")
            checks = semantic_check_m4(sp["eval_path"], sp["train_path"])
            for c in checks:
                icon = "\u2705" if c["status"] == "pass" else "\u274c"
                detail = f" ({c['detail']})" if c["detail"] else ""
                print(f"    {icon} {c['check']}{detail}")
            semantic_results.append({
                "seed": sp["seed"],
                "eval_path": sp["eval_path"],
                "train_path": sp["train_path"],
                "checks": checks,
            })

        if not seed_pairs:
            print("  No M4 eval\u2194train seed pairs found.")

        sem_all_pass = all(
            c["status"] == "pass"
            for sr in semantic_results
            for c in sr["checks"]
        ) if semantic_results else True
        sem_total = sum(len(sr["checks"]) for sr in semantic_results)
        sem_pass = sum(
            1 for sr in semantic_results for c in sr["checks"] if c["status"] == "pass"
        )
        print(f"\n  Semantic: {sem_pass}/{sem_total} passed\n")

    # ── Determine pass/fail ──
    if strict:
        # Strict: required items must all pass
        overall_ok = (schema_required_fail == 0) and sem_all_pass
    else:
        # Non-strict: only count actual schema+semantic failures (missing optional is OK)
        overall_ok = (schema_required_fail == 0) and sem_all_pass

    # ── Generate outputs ──
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile_name,
        "strict": strict,
        "manifest": args.manifest if use_manifest else None,
        "reports_dir": reports_dir_str,
        "summary": {
            "schema_total": schema_total,
            "schema_passed": schema_pass,
            "schema_required_fail": schema_required_fail,
            "semantic_total": sem_total,
            "semantic_passed": sem_pass,
            "overall": "pass" if overall_ok else "fail",
            "model_quality_status": (
                semantic_results.get("model_quality_status")
                if isinstance(semantic_results, dict)
                else None
            ),
        },
        "schema_results": schema_results,
        "semantic_results": semantic_results,
    }

    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON report: {json_out}")

    with open(md_out, "w") as f:
        f.write(generate_md(schema_results, semantic_results, profile_name, strict))
    print(f"Markdown report: {md_out}")

    print(f"\n{'='*60}")
    if overall_ok:
        print(f"ALL PASS (schema {schema_pass}/{schema_total}, required_fail={schema_required_fail}, semantic {sem_pass}/{sem_total})")
    else:
        print(f"FAIL (schema {schema_pass}/{schema_total}, required_fail={schema_required_fail}, semantic {sem_pass}/{sem_total})")
    print(f"{'='*60}")

    sys.exit(0 if overall_ok else 1 if strict else 2)


if __name__ == "__main__":
    main()
