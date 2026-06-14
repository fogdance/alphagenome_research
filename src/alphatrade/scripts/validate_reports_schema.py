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
import sys
import warnings
from datetime import datetime
from pathlib import Path

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

    def add(name, passed, detail=""):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
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

    def add(name, passed, detail=""):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
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
            if train_exists and train_schema:
                try:
                    train_data = load_json(train_path)
                    validate(instance=train_data, schema=train_schema)
                    add_exp(f"{run_label} train_schema", True)
                except ValidationError as e:
                    add_exp(f"{run_label} train_schema", False, str(e.message)[:80])
                except Exception as e:
                    add_exp(f"{run_label} train_schema", False, str(e)[:80])

            # Eval metrics exists + schema
            eval_path = run["eval_metrics_path"]
            eval_exists = os.path.exists(eval_path)
            add_exp(f"{run_label} eval_exists", eval_exists,
                    f"not found: {eval_path}" if not eval_exists else "")
            if eval_exists and eval_schema:
                try:
                    eval_data = load_json(eval_path)
                    validate(instance=eval_data, schema=eval_schema)
                    add_exp(f"{run_label} eval_schema", True)
                except ValidationError as e:
                    add_exp(f"{run_label} eval_schema", False, str(e.message)[:80])
                except Exception as e:
                    add_exp(f"{run_label} eval_schema", False, str(e)[:80])

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

    def add(name, passed, detail=""):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
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
    if profile_name == "m9":
        _generate_md_phase2_m9(w, semantic_results)
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
    if profile_name == "m9":
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
    if profile_name == "m9":
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
