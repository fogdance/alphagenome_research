#!/usr/bin/env python3
"""
M2 Reports Schema Validator

Validates M2 report JSON files against their fixed schemas.
"""

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Validate M2 reports against schemas")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    parser.add_argument("--schemas-dir", type=str, default="src/alphatrade/schemas", help="Schemas directory")
    return parser.parse_args()


def load_json(file_path: str) -> dict:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_schema(schema_path: str) -> dict:
    """Load JSON schema."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def validate_report(report_path: str, schema_path: str) -> dict:
    """Validate a report against its schema."""
    result = {
        "report": report_path,
        "schema": schema_path,
        "status": "unknown",
        "error": None
    }
    
    try:
        # Check if files exist
        if not os.path.exists(report_path):
            result["status"] = "missing_report"
            result["error"] = f"Report file not found: {report_path}"
            return result
        
        if not os.path.exists(schema_path):
            result["status"] = "missing_schema"
            result["error"] = f"Schema file not found: {schema_path}"
            return result
        
        # Load files
        report = load_json(report_path)
        schema = load_schema(schema_path)
        
        # Validate
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


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"Reports Schema Validation")
    print(f"{'='*60}")
    print(f"Reports dir: {args.reports_dir}")
    print(f"Schemas dir: {args.schemas_dir}")
    print(f"{'='*60}\n")

    # Define report-schema pairs
    validations = [
        {
            "name": "m2_train_metrics",
            "report": os.path.join(args.reports_dir, "m2_train_metrics.json"),
            "schema": os.path.join(args.schemas_dir, "m2_train_metrics.schema.json")
        },
        {
            "name": "m2_universe_sweep",
            "report": os.path.join(args.reports_dir, "m2_universe_sweep.json"),
            "schema": os.path.join(args.schemas_dir, "m2_universe_sweep.schema.json")
        },
        {
            "name": "m2_t1_dataloader_check",
            "report": os.path.join(args.reports_dir, "m2_t1_dataloader_check.json"),
            "schema": os.path.join(args.schemas_dir, "m2_t1_dataloader_check.schema.json")
        },
        {
            "name": "m3_train_metrics",
            "report": os.path.join(args.reports_dir, "m3_train_metrics.json"),
            "schema": os.path.join(args.schemas_dir, "m2_train_metrics.schema.json")
        },
        {
            "name": "m4_train_metrics",
            "report": os.path.join(args.reports_dir, "m4_train_metrics.json"),
            "schema": os.path.join(args.schemas_dir, "m2_train_metrics.schema.json")
        },
        {
            "name": "m4_eval_metrics",
            "report": os.path.join(args.reports_dir, "m4_eval_metrics.json"),
            "schema": os.path.join(args.schemas_dir, "m4_eval_metrics.schema.json")
        }
    ]
    
    # Validate each report
    results = []
    
    for val in validations:
        print(f"Validating {val['name']}...", end=' ')
        result = validate_report(val['report'], val['schema'])
        result["name"] = val["name"]
        results.append(result)
        
        status_icon = {
            "pass": "✅",
            "fail": "❌",
            "missing_report": "⚠️",
            "missing_schema": "⚠️",
            "invalid_json": "❌",
            "error": "❌"
        }.get(result["status"], "❓")
        
        print(f"{status_icon} {result['status']}")
        
        if result["error"]:
            print(f"  Error: {result['error']}")
            if "error_path" in result:
                print(f"  Path: {'.'.join(map(str, result['error_path']))}")
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] in ["fail", "invalid_json", "error"])
    missing = sum(1 for r in results if r["status"] in ["missing_report", "missing_schema"])
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Missing: {missing}")
    print(f"{'='*60}\n")
    
    # Generate JSON report
    json_report = {
        "timestamp": datetime.now().isoformat(),
        "reports_dir": args.reports_dir,
        "schemas_dir": args.schemas_dir,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "missing": missing,
            "success_rate": passed / total if total > 0 else 0.0
        },
        "results": results
    }
    
    json_path = os.path.join(args.reports_dir, "m4_schema_validation.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)

    print(f"✅ JSON report: {json_path}")

    # Generate Markdown report
    md_path = os.path.join(args.reports_dir, "m4_schema_validation.md")
    with open(md_path, 'w') as f:
        f.write("# Reports Schema Validation\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 配置\n\n")
        f.write(f"- **Reports dir**: `{args.reports_dir}`\n")
        f.write(f"- **Schemas dir**: `{args.schemas_dir}`\n\n")
        
        f.write("## 总览\n\n")
        f.write(f"- **Total**: {total}\n")
        f.write(f"- **Passed**: {passed} ({passed/total*100:.1f}%)\n")
        f.write(f"- **Failed**: {failed}\n")
        f.write(f"- **Missing**: {missing}\n\n")
        
        f.write("## 验证结果\n\n")
        f.write("| Report | Status | Error |\n")
        f.write("|--------|--------|-------|\n")
        
        for r in results:
            status_icon = {
                "pass": "✅",
                "fail": "❌",
                "missing_report": "⚠️",
                "missing_schema": "⚠️",
                "invalid_json": "❌",
                "error": "❌"
            }.get(r["status"], "❓")
            
            error_msg = r["error"][:50] + "..." if r["error"] and len(r["error"]) > 50 else (r["error"] or "-")
            f.write(f"| {r['name']} | {status_icon} {r['status']} | {error_msg} |\n")
        
        f.write("\n")
        
        # Details for failed validations
        failed_results = [r for r in results if r["status"] in ["fail", "invalid_json", "error"]]
        if failed_results:
            f.write("## 失败详情\n\n")
            for r in failed_results:
                f.write(f"### {r['name']}\n\n")
                f.write(f"- **Status**: {r['status']}\n")
                f.write(f"- **Error**: {r['error']}\n")
                if "error_path" in r:
                    f.write(f"- **Path**: {'.'.join(map(str, r['error_path']))}\n")
                f.write("\n")
        
        # Schema files
        f.write("## Schema 文件\n\n")
        for val in validations:
            f.write(f"- **{val['name']}**: `{val['schema']}`\n")
        f.write("\n")
        
        # Overall status
        if passed == total:
            f.write("## 总体状态\n\n")
            f.write("✅ **所有验证通过**\n\n")
        else:
            f.write("## 总体状态\n\n")
            f.write(f"❌ **{failed + missing} 个验证失败**\n\n")
    
    print(f"✅ Markdown report: {md_path}")
    
    # Exit code
    print(f"\n{'='*60}")
    if passed == total:
        print(f"✅ All validations passed")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"❌ {failed + missing} validation(s) failed")
        print(f"{'='*60}")
        sys.exit(2)


if __name__ == "__main__":
    main()
