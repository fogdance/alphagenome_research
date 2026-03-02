#!/usr/bin/env python3
"""
Reports Schema & Semantic Validator

Phase 1: JSON Schema validation (structure) — driven by manifest profiles
Phase 2: Semantic checks (M4 eval↔train cross-validation)
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

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
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    parser.add_argument("--schemas-dir", type=str, default="src/alphatrade/schemas", help="Schemas directory")
    parser.add_argument("--manifest", type=str, default=_DEFAULT_MANIFEST, help="Path to contracts_manifest.yaml")
    parser.add_argument("--profile", type=str, default=_DEFAULT_PROFILE, help="Manifest profile to validate")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on required missing/fail")
    parser.add_argument("--output-json", type=str, default=None, help="JSON output path (default: reports/{profile}_schema_validation.json)")
    parser.add_argument("--output-md", type=str, default=None, help="Markdown output path (default: reports/{profile}_schema_validation.md)")
    return parser.parse_args()


def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


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

        if schema_path and not os.path.exists(schema_path):
            result["status"] = "missing_schema"
            result["error"] = f"Schema file not found: {schema_path}"
            return result

        report = load_json(report_path)

        if not schema_path:
            # No schema — existence-only check
            result["status"] = "pass"
            return result

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

    # --- Overall ---
    schema_ok = all(r["status"] == "pass" for r in schema_results if r.get("required"))
    sem_ok = all(
        c["status"] == "pass"
        for sr in semantic_results
        for c in sr["checks"]
    ) if semantic_results else True

    w("## Overall\n")
    schema_verdict = "\u2705 pass" if schema_ok else "\u274c fail"
    sem_verdict = "\u2705 pass" if sem_ok else "\u274c fail"
    w(f"- Schema (required): {schema_verdict}")
    w(f"- Semantic: {sem_verdict}")
    overall = "\u2705 ALL PASS" if (schema_ok and sem_ok) else "\u274c FAIL"
    w(f"- **Overall: {overall}**")
    w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    profile_name = args.profile
    strict = args.strict

    # Output paths
    json_out = args.output_json or os.path.join(args.reports_dir, f"{profile_name}_schema_validation.json")
    md_out = args.output_md or os.path.join(args.reports_dir, f"{profile_name}_schema_validation.md")

    print(f"\n{'='*60}")
    print(f"Reports Schema & Semantic Validation")
    print(f"{'='*60}")
    print(f"Profile:     {profile_name}")
    print(f"Strict:      {strict}")
    print(f"Reports dir: {args.reports_dir}")
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
        reports_dir = get_profile_reports_dir(manifest, profile_name) or args.reports_dir
        validations = []
        for item in items:
            report_path = os.path.join(reports_dir, item["path"])
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
        validations = _legacy_validations(args.reports_dir, args.schemas_dir)
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
    print("Phase 2: M4 semantic checks\n")
    semantic_results = []
    seed_pairs = discover_m4_seed_pairs(args.reports_dir)

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
        "reports_dir": args.reports_dir,
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
