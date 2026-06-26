#!/usr/bin/env python3
"""Build the M12 formal data-contract spotcheck report.

This is a post-build semantic report over every formal M12 dataset root. It
does not train or evaluate a model; it verifies that the common-row ablation
inputs are aligned and profile-consistent before the formal sweep runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_m12_chendage_features as m12


SPOTCHECK_SCHEMA_VERSION = "m12_formal_data_contract_spotcheck_v1"
INDEX_COMPARE_COLS = ("eob", "target_eob", "_m12_source_pos", "segment_id")
REQUIRED_INDEX_COMPARE_COLS = ("eob", "target_eob", "_m12_source_pos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M12 formal data-contract spotcheck")
    parser.add_argument("--run-root", required=True, help="Formal M12 run root")
    parser.add_argument(
        "--contract",
        default=None,
        help="Path to m12_chendage_feature_contract.json. Defaults to <run-root>/reports/...",
    )
    parser.add_argument("--reports-dir", default=None, help="Output reports directory")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on failed checks")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    detail: str = "",
    observed: Any = None,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "severity": "ERROR" if not passed else "INFO",
            "detail": detail,
            "observed": observed,
        }
    )


def feature_profile_from_manifest(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "feature_manifest.json")
    profile = manifest.get("feature_profile") or {}
    feature_cols = list(profile.get("feature_cols") or [])
    return {
        "manifest": manifest,
        "profile": profile,
        "feature_cols": feature_cols,
        "chendage_feature_keys": list(manifest.get("chendage_feature_keys") or []),
        "chendage_feature_cols": list(manifest.get("chendage_feature_cols") or []),
    }


def load_symbols(root: Path) -> list[str]:
    config_path = root / "universe.yaml"
    if not config_path.exists():
        return sorted(path.name for path in root.iterdir() if path.is_dir())
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return list(data.get("candidates") or [])


def compare_indices(candidate_root: Path, control_root: Path, symbols: list[str]) -> dict[str, Any]:
    failures = []
    summary: dict[str, Any] = {}
    for symbol in symbols:
        summary[symbol] = {}
        for split in ("train", "val", "test"):
            cand_path = candidate_root / symbol / f"index_{split}.parquet"
            ctrl_path = control_root / symbol / f"index_{split}.parquet"
            if not cand_path.exists() or not ctrl_path.exists():
                failures.append(
                    {
                        "symbol": symbol,
                        "split": split,
                        "reason": "missing_index_file",
                        "candidate_exists": cand_path.exists(),
                        "control_exists": ctrl_path.exists(),
                    }
                )
                continue
            cand = pd.read_parquet(cand_path)
            ctrl = pd.read_parquet(ctrl_path)
            cand_missing_required = [
                col for col in REQUIRED_INDEX_COMPARE_COLS if col not in cand.columns
            ]
            ctrl_missing_required = [
                col for col in REQUIRED_INDEX_COMPARE_COLS if col not in ctrl.columns
            ]
            cols = [col for col in INDEX_COMPARE_COLS if col in cand.columns and col in ctrl.columns]
            if cand_missing_required or ctrl_missing_required:
                summary[symbol][split] = {
                    "candidate_rows": int(len(cand)),
                    "control_rows": int(len(ctrl)),
                    "same_order": False,
                    "compared_cols": cols,
                    "candidate_missing_required_cols": cand_missing_required,
                    "control_missing_required_cols": ctrl_missing_required,
                }
                failures.append(
                    {
                        "symbol": symbol,
                        "split": split,
                        "reason": "missing_required_index_compare_columns",
                        "candidate_missing_required_cols": cand_missing_required,
                        "control_missing_required_cols": ctrl_missing_required,
                    }
                )
                continue
            same = len(cand) == len(ctrl) and cand[cols].reset_index(drop=True).equals(
                ctrl[cols].reset_index(drop=True)
            )
            summary[symbol][split] = {
                "candidate_rows": int(len(cand)),
                "control_rows": int(len(ctrl)),
                "same_order": bool(same),
                "compared_cols": cols,
            }
            if not same:
                failures.append(
                    {
                        "symbol": symbol,
                        "split": split,
                        "reason": "index_mismatch",
                        "candidate_rows": int(len(cand)),
                        "control_rows": int(len(ctrl)),
                    }
                )
    return {"status": "PASS" if not failures else "FAIL", "summary": summary, "failures": failures[:20]}


def split_feature_frame(symbol_root: Path, split: str, bars: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    index_path = symbol_root / f"index_{split}.parquet"
    if not index_path.exists():
        return bars.iloc[0:0][feature_cols]
    index_df = pd.read_parquet(index_path)
    if index_df.empty:
        return bars.iloc[0:0][feature_cols]
    positions = index_df["t"].astype(int).to_numpy()
    positions = positions[(positions >= 0) & (positions < len(bars))]
    return bars.iloc[positions][feature_cols]


def feature_missing_and_constant_stats(
    *,
    root: Path,
    symbols: list[str],
    feature_cols: list[str],
    chendage_feature_cols: list[str],
    col_to_group: dict[str, str],
) -> dict[str, Any]:
    numeric_failures = []
    finite_failures = []
    missing_rates: dict[str, Any] = {}
    near_constant_by_group: dict[str, dict[str, int]] = {}
    feature_dim_failures = []

    for symbol in symbols:
        bars_path = root / symbol / "bars.parquet"
        if not bars_path.exists():
            feature_dim_failures.append({"symbol": symbol, "reason": "missing_bars"})
            continue
        bars = pd.read_parquet(bars_path)
        missing_cols = [col for col in feature_cols if col not in bars.columns]
        if missing_cols:
            feature_dim_failures.append({"symbol": symbol, "missing_cols": missing_cols[:20]})
            continue
        if len(feature_cols) != len(set(feature_cols)):
            feature_dim_failures.append({"symbol": symbol, "reason": "duplicate_feature_cols"})

        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(bars[col]):
                numeric_failures.append({"symbol": symbol, "feature": col, "dtype": str(bars[col].dtype)})
        values = bars[feature_cols].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            bad_cols = []
            for col in feature_cols:
                arr = bars[col].to_numpy(dtype=np.float64, copy=False)
                if not np.isfinite(arr).all():
                    bad_cols.append(col)
            finite_failures.append({"symbol": symbol, "features": bad_cols[:20]})

        missing_rates[symbol] = {}
        near_constant_by_group[symbol] = {}
        for split in ("train", "val", "test"):
            split_frame = split_feature_frame(root / symbol, split, bars, feature_cols)
            if split_frame.empty:
                missing_rates[symbol][split] = {"rows": 0, "by_feature": {}}
                continue
            rates = {
                col: float(split_frame[col].isna().mean())
                for col in feature_cols
            }
            missing_rates[symbol][split] = {"rows": int(len(split_frame)), "by_feature": rates}

            for group in sorted(set(col_to_group.values()) or {"base8"}):
                group_cols = [col for col in chendage_feature_cols if col_to_group.get(col) == group]
                if not group_cols:
                    continue
                constant_count = 0
                for col in group_cols:
                    arr = split_frame[col].to_numpy(dtype=np.float64, copy=False)
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0 or float(np.nanstd(arr)) <= 1e-12:
                        constant_count += 1
                key = f"{split}:{group}"
                near_constant_by_group[symbol][key] = constant_count

    return {
        "feature_dim_failures": feature_dim_failures,
        "numeric_failures": numeric_failures[:50],
        "finite_failures": finite_failures[:50],
        "missing_rates": missing_rates,
        "near_constant_by_group": near_constant_by_group,
    }


def raw_price_like_selected(keys: list[str]) -> list[str]:
    return [key for key in keys if m12._looks_like_raw_price_feature(key)]


def rule_only_selected(keys: list[str], cols: list[str]) -> list[str]:
    names = list(keys) + list(cols)
    return [name for name in names if m12._is_rule_only_name(name)]


def dataset_spotcheck(
    *,
    name: str,
    root: Path,
    control_root: Path,
    symbols: list[str],
    is_control: bool,
) -> dict[str, Any]:
    profile_payload = feature_profile_from_manifest(root)
    manifest = profile_payload["manifest"]
    profile = profile_payload["profile"]
    feature_cols = profile_payload["feature_cols"]
    chg_keys = profile_payload["chendage_feature_keys"]
    chg_cols = profile_payload["chendage_feature_cols"]
    col_to_group = {
        col: m12.feature_group_for_key(key)
        for key, col in zip(chg_keys, chg_cols)
    }
    source_manifest = load_json(root / "source_manifest.json")
    index_check = (
        {"status": "PASS", "summary": {}, "failures": []}
        if is_control
        else compare_indices(root, control_root, symbols)
    )
    stats = feature_missing_and_constant_stats(
        root=root,
        symbols=symbols,
        feature_cols=feature_cols,
        chendage_feature_cols=chg_cols,
        col_to_group=col_to_group,
    )
    rule_only = rule_only_selected(chg_keys, chg_cols)
    raw_price_like = raw_price_like_selected(chg_keys)
    normalization = (
        profile.get("normalization")
        if is_control
        else (manifest.get("normalization") or profile.get("normalization") or {})
    )
    scaler_ok = (
        normalization.get("train_only") is True
        and bool(normalization.get("scaler_hash"))
    )
    if is_control:
        scaler_ok = normalization.get("policy") == "precomputed_in_bars"
    source_versions = manifest.get("source_schema_versions") or {}
    provenance_ok = (
        bool(source_manifest.get("chendage_commit"))
        and bool(source_manifest.get("input_files"))
        and bool(source_manifest.get("input_hashes"))
        and bool(source_manifest.get("symbol_map"))
    )

    return {
        "name": name,
        "root": str(root),
        "is_control": is_control,
        "feature_profile": profile,
        "feature_dim": int(profile.get("feature_dim") or 0),
        "feature_count": len(feature_cols),
        "chendage_feature_count": len(chg_cols),
        "feature_groups": manifest.get("feature_groups") or {},
        "index_equality_vs_control": index_check,
        "feature_dim_equals_manifest": int(profile.get("feature_dim") or -1) == len(feature_cols)
        and not stats["feature_dim_failures"],
        "rule_only_selected": rule_only,
        "raw_price_like_selected": raw_price_like,
        "all_selected_numeric": not stats["numeric_failures"],
        "all_selected_finite": not stats["finite_failures"],
        "missing_rates": stats["missing_rates"],
        "near_constant_by_group": stats["near_constant_by_group"],
        "train_only_scaler_metadata_ok": scaler_ok,
        "source_schema_versions": source_versions,
        "source_provenance_ok": provenance_ok,
        "failures": {
            "feature_dim": stats["feature_dim_failures"],
            "numeric": stats["numeric_failures"],
            "finite": stats["finite_failures"],
        },
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# M12 Formal Data Contract Spotcheck\n\n")
        handle.write("AlphaTrade is development-stage. This report checks M12 processed-feature ")
        handle.write("data contracts before training; it is not model-quality evidence.\n\n")
        handle.write(f"- Generated: {report['generated_at']}\n")
        handle.write(f"- Overall status: `{report['overall_status']}`\n")
        handle.write(f"- Run root: `{report['run_root']}`\n")
        handle.write(f"- Control root: `{report['control_root']}`\n\n")
        handle.write("## Dataset Status\n\n")
        handle.write("| Dataset | Dim | Chendage Features | Common Rows | Status |\n")
        handle.write("|---------|-----|-------------------|-------------|--------|\n")
        for name, item in report["datasets"].items():
            status = "PASS" if item["dataset_pass"] else "FAIL"
            common_rows = sum(
                split.get("candidate_rows", 0)
                for by_split in item.get("index_equality_vs_control", {}).get("summary", {}).values()
                for split in by_split.values()
            )
            if item.get("is_control"):
                common_rows = "-"
            handle.write(
                f"| {name} | {item['feature_dim']} | {item['chendage_feature_count']} | "
                f"{common_rows} | {status} |\n"
            )
        handle.write("\n## Checks\n\n")
        handle.write("| Check | Status | Detail |\n")
        handle.write("|-------|--------|--------|\n")
        for check in report["checks"]:
            handle.write(f"| {check['name']} | {check['status']} | {check.get('detail') or '-'} |\n")


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    reports_dir = Path(args.reports_dir).expanduser().resolve() if args.reports_dir else run_root / "reports"
    contract_path = (
        Path(args.contract).expanduser().resolve()
        if args.contract
        else reports_dir / "m12_chendage_feature_contract.json"
    )
    contract = load_json(contract_path)
    outputs = contract.get("outputs") or {}
    control_root = Path(outputs["control_processed_root"]).expanduser().resolve()
    dataset_roots = {
        name: Path(path).expanduser().resolve()
        for name, path in (outputs.get("dataset_roots") or {}).items()
    }
    if not dataset_roots:
        dataset_roots = {
            "candidate": Path(outputs["candidate_processed_root"]).expanduser().resolve()
        }
    symbols = list(contract.get("inputs", {}).get("symbols") or load_symbols(control_root))

    checks: list[dict[str, Any]] = []
    datasets: dict[str, Any] = {}
    datasets["base8_control_common_rows"] = dataset_spotcheck(
        name="base8_control_common_rows",
        root=control_root,
        control_root=control_root,
        symbols=symbols,
        is_control=True,
    )
    for name, root in dataset_roots.items():
        datasets[name] = dataset_spotcheck(
            name=name,
            root=root,
            control_root=control_root,
            symbols=symbols,
            is_control=False,
        )

    for name, item in datasets.items():
        add_check(
            checks,
            f"{name}.feature_dim_equals_manifest",
            item["feature_dim_equals_manifest"],
            observed={"feature_dim": item["feature_dim"], "feature_count": item["feature_count"]},
        )
        add_check(checks, f"{name}.rule_only_selected_absent", not item["rule_only_selected"], observed=item["rule_only_selected"])
        add_check(checks, f"{name}.raw_price_like_selected_absent", not item["raw_price_like_selected"], observed=item["raw_price_like_selected"])
        add_check(checks, f"{name}.all_selected_numeric", item["all_selected_numeric"], observed=item["failures"]["numeric"])
        add_check(checks, f"{name}.all_selected_finite", item["all_selected_finite"], observed=item["failures"]["finite"])
        add_check(checks, f"{name}.train_only_scaler_metadata", item["train_only_scaler_metadata_ok"], observed=item["feature_profile"].get("normalization"))
        add_check(checks, f"{name}.source_provenance_recorded", item["source_provenance_ok"], observed=item["source_schema_versions"])
        if not item["is_control"]:
            add_check(
                checks,
                f"{name}.common_row_index_equals_control",
                item["index_equality_vs_control"]["status"] == "PASS",
                observed=item["index_equality_vs_control"].get("failures"),
            )
        add_check(
            checks,
            f"{name}.near_constant_counts_recorded",
            bool(item.get("near_constant_by_group")),
            observed=item.get("near_constant_by_group"),
        )

    for item in datasets.values():
        item["dataset_pass"] = all(
            check["status"] == "PASS"
            for check in checks
            if check["name"].startswith(f"{item['name']}.")
        )
    overall_status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    report = {
        "schema_version": SPOTCHECK_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "run_root": str(run_root),
        "contract": str(contract_path),
        "control_root": str(control_root),
        "symbols": symbols,
        "overall_status": overall_status,
        "datasets": datasets,
        "checks": checks,
    }
    json_path = reports_dir / "m12_formal_data_contract_spotcheck.json"
    md_path = reports_dir / "m12_formal_data_contract_spotcheck.md"
    write_json(json_path, report)
    write_markdown(md_path, report)
    print(f"M12 formal data contract spotcheck: {overall_status}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 1 if args.strict and overall_status != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
