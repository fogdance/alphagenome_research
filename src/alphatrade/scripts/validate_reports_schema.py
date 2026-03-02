#!/usr/bin/env python3
"""
Reports Schema & Semantic Validator

Phase 1: JSON Schema validation (structure)
Phase 2: Semantic checks (M4 eval↔train cross-validation)
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

try:
    import jsonschema
    from jsonschema import validate, ValidationError
except ImportError:
    print("Error: jsonschema package not installed")
    print("Install with: pip install jsonschema")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate reports: schema + semantic")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    parser.add_argument("--schemas-dir", type=str, default="src/alphatrade/schemas", help="Schemas directory")
    return parser.parse_args()


def load_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)


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
        # Extract seed number
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
    """Run semantic checks on an M4 eval↔train pair.

    Returns a list of check dicts: {check, status, detail}.
    """
    checks = []

    # Helper
    def add(name, passed, detail=""):
        checks.append({
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
        })

    # Load files
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

    # Check 1: model.source == "checkpoint"
    model = eval_data.get("model", {})
    source = model.get("source")
    add("model.source == 'checkpoint'",
        source == "checkpoint",
        f"got '{source}'" if source != "checkpoint" else "")

    # Check 2: model.train_run_id matches train.run.run_id
    eval_run_id = model.get("train_run_id")
    train_run_id = train_data.get("run", {}).get("run_id")
    matched = eval_run_id == train_run_id and eval_run_id is not None
    add("model.train_run_id == train.run.run_id",
        matched,
        f"eval='{eval_run_id}' vs train='{train_run_id}'" if not matched else "")

    # Check 3: model.checkpoint_step is a positive integer
    step = model.get("checkpoint_step")
    step_ok = isinstance(step, int) and step >= 1
    add("model.checkpoint_step >= 1",
        step_ok,
        f"got {step!r}" if not step_ok else "")

    # Check 4: model.checkpoint_dir is non-empty string
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
    "pass": "✅",
    "fail": "❌",
    "missing_report": "⚠️",
    "missing_schema": "⚠️",
    "invalid_json": "❌",
    "error": "❌",
}


def generate_md(schema_results, semantic_results, args) -> str:
    lines = []
    w = lines.append

    w("# Reports Schema & Semantic Validation\n")
    w(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- Phase 1 ---
    w("## Phase 1: Schema Validation\n")

    total = len(schema_results)
    passed = sum(1 for r in schema_results if r["status"] == "pass")
    failed = total - passed

    w(f"- Total: {total}")
    w(f"- Passed: {passed}")
    w(f"- Failed/Missing: {failed}\n")

    w("| Report | Status | Error |")
    w("|--------|--------|-------|")
    for r in schema_results:
        icon = _STATUS_ICON.get(r["status"], "❓")
        err = r["error"] or "-"
        if len(err) > 60:
            err = err[:57] + "..."
        w(f"| {r['name']} | {icon} {r['status']} | {err} |")
    w("")

    # --- Phase 2 ---
    w("## Phase 2: M4 Semantic Checks\n")

    if not semantic_results:
        w("_No M4 eval↔train pairs found._\n")
    else:
        all_checks = []
        for sr in semantic_results:
            w(f"### Seed {sr['seed']}\n")
            w(f"- eval: `{sr['eval_path']}`")
            w(f"- train: `{sr['train_path']}`\n")
            w("| Check | Status | Detail |")
            w("|-------|--------|--------|")
            for c in sr["checks"]:
                icon = "✅" if c["status"] == "pass" else "❌"
                w(f"| {c['check']} | {icon} | {c['detail'] or '-'} |")
            w("")
            all_checks.extend(sr["checks"])

        sem_pass = sum(1 for c in all_checks if c["status"] == "pass")
        sem_total = len(all_checks)
        w(f"**Semantic summary**: {sem_pass}/{sem_total} checks passed\n")

    # --- Overall ---
    schema_ok = all(r["status"] == "pass" for r in schema_results)
    sem_ok = all(
        c["status"] == "pass"
        for sr in semantic_results
        for c in sr["checks"]
    ) if semantic_results else True

    w("## 总体状态\n")
    w(f"- Schema: {'✅ pass' if schema_ok else '❌ fail'}")
    w(f"- Semantic: {'✅ pass' if sem_ok else '❌ fail'}")
    overall = "✅ ALL PASS" if (schema_ok and sem_ok) else "❌ FAIL"
    w(f"- **Overall: {overall}**")
    w("")

    return "\n".join(lines)


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Reports Schema & Semantic Validation")
    print(f"{'='*60}")
    print(f"Reports dir: {args.reports_dir}")
    print(f"Schemas dir: {args.schemas_dir}")
    print(f"{'='*60}\n")

    # ── Phase 1: Schema validation ──
    validations = [
        {
            "name": "m2_train_metrics",
            "report": os.path.join(args.reports_dir, "m2_train_metrics.json"),
            "schema": os.path.join(args.schemas_dir, "m2_train_metrics.schema.json"),
        },
        {
            "name": "m2_universe_sweep",
            "report": os.path.join(args.reports_dir, "m2_universe_sweep.json"),
            "schema": os.path.join(args.schemas_dir, "m2_universe_sweep.schema.json"),
        },
        {
            "name": "m2_t1_dataloader_check",
            "report": os.path.join(args.reports_dir, "m2_t1_dataloader_check.json"),
            "schema": os.path.join(args.schemas_dir, "m2_t1_dataloader_check.schema.json"),
        },
        {
            "name": "m3_train_metrics",
            "report": os.path.join(args.reports_dir, "m3_train_metrics.json"),
            "schema": os.path.join(args.schemas_dir, "m2_train_metrics.schema.json"),
        },
        {
            "name": "m4_train_metrics",
            "report": os.path.join(args.reports_dir, "m4_train_metrics.json"),
            "schema": os.path.join(args.schemas_dir, "m2_train_metrics.schema.json"),
        },
        {
            "name": "m4_eval_metrics",
            "report": os.path.join(args.reports_dir, "m4_eval_metrics.json"),
            "schema": os.path.join(args.schemas_dir, "m4_eval_metrics.schema.json"),
        },
    ]

    # Also validate seed-specific eval files
    eval_schema = os.path.join(args.schemas_dir, "m4_eval_metrics.schema.json")
    train_schema = os.path.join(args.schemas_dir, "m2_train_metrics.schema.json")
    seed_pairs = discover_m4_seed_pairs(args.reports_dir)
    for sp in seed_pairs:
        validations.append({
            "name": f"m4_eval_metrics_seed{sp['seed']}",
            "report": sp["eval_path"],
            "schema": eval_schema,
        })
        # Also validate the paired train metrics if present
        if os.path.exists(sp["train_path"]):
            validations.append({
                "name": f"m4_train_metrics_seed{sp['seed']}",
                "report": sp["train_path"],
                "schema": train_schema,
            })

    print("Phase 1: Schema validation\n")
    schema_results = []
    for val in validations:
        print(f"  {val['name']}...", end=" ")
        result = validate_report(val["report"], val["schema"])
        result["name"] = val["name"]
        schema_results.append(result)
        icon = _STATUS_ICON.get(result["status"], "❓")
        print(f"{icon} {result['status']}")
        if result["error"]:
            print(f"    Error: {result['error'][:80]}")

    schema_pass = sum(1 for r in schema_results if r["status"] == "pass")
    schema_total = len(schema_results)
    print(f"\n  Schema: {schema_pass}/{schema_total} passed\n")

    # ── Phase 2: Semantic checks ──
    print("Phase 2: M4 semantic checks\n")
    semantic_results = []

    for sp in seed_pairs:
        print(f"  Seed {sp['seed']}:")
        checks = semantic_check_m4(sp["eval_path"], sp["train_path"])
        for c in checks:
            icon = "✅" if c["status"] == "pass" else "❌"
            detail = f" ({c['detail']})" if c["detail"] else ""
            print(f"    {icon} {c['check']}{detail}")
        semantic_results.append({
            "seed": sp["seed"],
            "eval_path": sp["eval_path"],
            "train_path": sp["train_path"],
            "checks": checks,
        })

    if not seed_pairs:
        print("  No M4 eval↔train seed pairs found.")

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

    # ── Generate outputs ──
    schema_ok = schema_pass == schema_total
    overall_ok = schema_ok and sem_all_pass

    json_report = {
        "timestamp": datetime.now().isoformat(),
        "reports_dir": args.reports_dir,
        "schemas_dir": args.schemas_dir,
        "summary": {
            "schema_total": schema_total,
            "schema_passed": schema_pass,
            "semantic_total": sem_total,
            "semantic_passed": sem_pass,
            "overall": "pass" if overall_ok else "fail",
        },
        "schema_results": schema_results,
        "semantic_results": semantic_results,
    }

    json_path = os.path.join(args.reports_dir, "m4_schema_validation.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON report: {json_path}")

    md_path = os.path.join(args.reports_dir, "m4_schema_validation.md")
    with open(md_path, "w") as f:
        f.write(generate_md(schema_results, semantic_results, args))
    print(f"Markdown report: {md_path}")

    print(f"\n{'='*60}")
    if overall_ok:
        print(f"ALL PASS (schema {schema_pass}/{schema_total}, semantic {sem_pass}/{sem_total})")
    else:
        print(f"FAIL (schema {schema_pass}/{schema_total}, semantic {sem_pass}/{sem_total})")
    print(f"{'='*60}")

    sys.exit(0 if overall_ok else 2)


if __name__ == "__main__":
    main()
