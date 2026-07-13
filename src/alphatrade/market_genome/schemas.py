"""MG0-B audit contract and strict, CPU-only report validation.

This module defines no model tensor schemas. Those depend on the MG1 ontology
and MG3 dense-track contract and therefore remain explicitly deferred.
"""

from __future__ import annotations

import argparse
from collections import Counter
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any


class CompatibilityDisposition(str, Enum):
  """Allowed outcomes for an audited AlphaTrade subsystem."""

  DIRECT_REUSE = "DIRECT_REUSE"
  REUSE_WITH_ADAPTER = "REUSE_WITH_ADAPTER"
  ISOLATE = "ISOLATE"
  REPLACE_FOR_MARKETGENOME = "REPLACE_FOR_MARKETGENOME"
  DEFER = "DEFER"


PROFILE_NAME = "mg0b"
REPORT_BASENAME = "mg0_alphatrade_compatibility_audit"
REPORT_SCHEMA_VERSION = "mg0b.alphatrade_compatibility_audit.v1"

REQUIRED_SUBSYSTEMS = (
    "data_ingestion",
    "continuous_contract_mapping",
    "feature_profiles",
    "window_cache",
    "long_sequence_storage",
    "model_config",
    "training_launcher",
    "checkpointing",
    "bundle_format",
    "offline_inference",
    "m10_m11_evaluation",
    "report_governance",
    "resume_safety",
)

REQUIRED_INCOMPATIBLE_ASSUMPTIONS = (
    "fixed_short_lookback",
    "last_timestep_only_readout",
    "terminal_return_only_labels",
    "fixed_window_cache_layout",
    "current_quantile_head_parameterization",
    "hard_coded_architecture_behavior",
)

REQUIRED_SCAFFOLD_PATHS = (
    "configs/market_genome/README.md",
    "configs/market_genome/contracts_manifest.yaml",
    "docs/alphaTrade/market_genome/MG0_ALPHATRADE_COMPATIBILITY_AUDIT.json",
    "docs/alphaTrade/market_genome/MG0_ALPHATRADE_COMPATIBILITY_AUDIT.md",
    "src/alphatrade/market_genome/README.md",
    "src/alphatrade/market_genome/__init__.py",
    "src/alphatrade/market_genome/config.py",
    "src/alphatrade/market_genome/datasets/__init__.py",
    "src/alphatrade/market_genome/evaluation/__init__.py",
    "src/alphatrade/market_genome/heads/__init__.py",
    "src/alphatrade/market_genome/losses/__init__.py",
    "src/alphatrade/market_genome/mg0b_compatibility_audit.schema.json",
    "src/alphatrade/market_genome/ontology/__init__.py",
    "src/alphatrade/market_genome/perturbation/__init__.py",
    "src/alphatrade/market_genome/schemas.py",
    "src/alphatrade/market_genome/synthetic/__init__.py",
    "src/alphatrade/market_genome/training/__init__.py",
    "tests/market_genome/__init__.py",
    "tests/market_genome/test_mg0b_scaffold.py",
)

_EXPECTED_AUDIT_PAYLOAD_SHA256 = (
    "0497842c5cc1f23414da05e5813b5b915aea4b253cfc416a1532a14d6e811ca5"
)
_EXPECTED_MARKDOWN_SHA256 = (
    "b11c2dc7c5784d635480df7c3f5031c4b036a47a0e089f74ec45dac26df5c7a5"
)
_EXPECTED_M12_ARTIFACTS = {
    (
        "/data/alphatrade/runs/m12_chg_formal5_20260626_parallel/"
        "reports/M12_CHG_FORMAL_DECISION_MEMO.md"
    ): "04e5df55d8e74c983a5bb838cf7040af772a5215fbb53401222688a8fdff3afc",
    (
        "/data/alphatrade/runs/m12_chg_formal5_20260626_parallel/"
        "reports/m12_formal_decision_summary.json"
    ): "9e3cc22a3646a9a1e8945631628c51dbfbce96b6feaea8795f90e9f61ce9cfb5",
    (
        "/data/alphatrade/runs/m12_chg_formal5_20260626_parallel/"
        "reports/m12_schema_validation.json"
    ): "98d9b73f8ad877622f21fbcefcfcdc631e54cf88b9b5d9280387701cfc8eec0c",
    (
        "/data/alphatrade/runs/m12_chg_formal5_20260626_parallel/"
        "reports/m12_formal_data_contract_spotcheck.json"
    ): "4c6465e8d3f4124301f0a467b8ec4c484cb7c13b8f5a1c72d0d84b951a0e0462",
    (
        "/data/alphatrade/runs/m12_chg_formal5_20260626_parallel/"
        "reports/formal_sweep/m5_sweep_manifest.json"
    ): "b220773b76b9ef75bc0d546d464affcbe3a8767670a6706c02a5b5f87d407b77",
}


def repository_root() -> Path:
  """Return the repository containing this source package."""
  return Path(__file__).resolve().parents[3]


def report_schema_path() -> Path:
  """Return the independent MG0-B JSON Schema path."""
  return Path(__file__).with_name("mg0b_compatibility_audit.schema.json")


def _canonical_sha256(value: Any) -> str:
  encoded = json.dumps(
      value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
  ).encode("ascii")
  return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _semantic_check(
    name: str, passed: bool, detail: str = "", observed: Any = None
) -> dict[str, Any]:
  return {
      "name": name,
      "status": "pass" if passed else "fail",
      "detail": detail,
      "observed": observed,
  }


def _evidence_items(report: dict[str, Any]) -> list[dict[str, Any]]:
  groups: list[Any] = []
  for key in (
      "compatibility_matrix",
      "architecture_boundaries",
      "incompatible_assumptions",
      "metric_reuse",
  ):
    groups.extend(report.get(key, []) if isinstance(report.get(key), list) else [])
  evidence: list[dict[str, Any]] = []
  for item in groups:
    if isinstance(item, dict) and isinstance(item.get("evidence"), list):
      evidence.extend(x for x in item["evidence"] if isinstance(x, dict))
  return evidence


def semantic_checks(
    report: dict[str, Any], repo_root: Path | None = None
) -> list[dict[str, Any]]:
  """Run fail-closed MG0-B checks not expressible in JSON Schema."""
  root = (repo_root or repository_root()).resolve()
  results: list[dict[str, Any]] = []

  observed_payload_hash = _canonical_sha256(report)
  results.append(
      _semantic_check(
          "canonical_audit_payload_hash_matches",
          observed_payload_hash == _EXPECTED_AUDIT_PAYLOAD_SHA256,
          "MG0-B v1 is a frozen source-grounded audit payload.",
          {
              "expected": _EXPECTED_AUDIT_PAYLOAD_SHA256,
              "observed": observed_payload_hash,
          },
      )
  )

  matrix = report.get("compatibility_matrix")
  matrix_items = matrix if isinstance(matrix, list) else []
  subsystem_ids = [
      item.get("subsystem_id") for item in matrix_items if isinstance(item, dict)
  ]
  results.append(
      _semantic_check(
          "required_subsystems_exact_and_unique",
          Counter(subsystem_ids) == Counter(REQUIRED_SUBSYSTEMS),
          "MG0-B requires exactly the 13 plan-listed subsystem boundaries.",
          subsystem_ids,
      )
  )

  allowed = {item.value for item in CompatibilityDisposition}
  observed_dispositions = [
      item.get("classification")
      for item in matrix_items
      if isinstance(item, dict)
  ]
  results.append(
      _semantic_check(
          "classification_enum_enforced",
          bool(observed_dispositions)
          and all(value in allowed for value in observed_dispositions),
          observed=observed_dispositions,
      )
  )

  assumptions = report.get("incompatible_assumptions")
  assumption_items = assumptions if isinstance(assumptions, list) else []
  assumption_ids = [
      item.get("assumption_id")
      for item in assumption_items
      if isinstance(item, dict)
  ]
  results.append(
      _semantic_check(
          "required_incompatible_assumptions_exact_and_unique",
          Counter(assumption_ids)
          == Counter(REQUIRED_INCOMPATIBLE_ASSUMPTIONS),
          observed=assumption_ids,
      )
  )

  evidence_errors: list[str] = []
  for evidence in _evidence_items(report):
    relative = evidence.get("path")
    start = evidence.get("start_line")
    end = evidence.get("end_line")
    if not isinstance(relative, str) or not relative:
      evidence_errors.append("missing_path")
      continue
    path = (root / relative).resolve()
    try:
      path.relative_to(root)
    except ValueError:
      evidence_errors.append(f"{relative}:outside_repository")
      continue
    if not path.is_file():
      evidence_errors.append(f"{relative}:missing")
      continue
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or start < 1
        or end < start
    ):
      evidence_errors.append(f"{relative}:invalid_line_range")
      continue
    with path.open("r", encoding="utf-8", errors="replace") as handle:
      line_count = sum(1 for _ in handle)
    if end > line_count:
      evidence_errors.append(f"{relative}:{start}-{end}>line_count:{line_count}")
  results.append(
      _semantic_check(
          "repository_evidence_paths_and_lines_valid",
          bool(_evidence_items(report)) and not evidence_errors,
          observed=evidence_errors,
      )
  )

  m12 = report.get("m12_evidence")
  m12 = m12 if isinstance(m12, dict) else {}
  separated = (
      "overall" not in m12
      and m12.get("engineering_contract_status") == "PASS"
      and m12.get("schema_validation_status") == "PASS"
      and m12.get("formal_sweep_completion") == "PASS"
      and m12.get("model_quality_status") == "FAIL_MODEL_QUALITY"
      and m12.get("formal_decision") == "FAIL"
      and m12.get("promotion_status") == "REJECTED"
  )
  results.append(
      _semantic_check(
          "m12_contract_and_model_quality_separated",
          separated,
          "Engineering/schema PASS must not overwrite model-quality FAIL.",
          m12,
      )
  )

  artifacts = (
      m12.get("artifacts") if isinstance(m12.get("artifacts"), list) else []
  )
  artifact_errors: list[str] = []
  artifact_pairs: dict[str, str] = {}
  for item in artifacts:
    if not isinstance(item, dict):
      artifact_errors.append("artifact_not_object")
      continue
    path_value = item.get("path")
    hash_value = item.get("sha256")
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
      artifact_errors.append("artifact_missing_path_or_sha256")
      continue
    if path_value in artifact_pairs:
      artifact_errors.append(f"{path_value}:duplicate")
    artifact_pairs[path_value] = hash_value

  if artifact_pairs != _EXPECTED_M12_ARTIFACTS:
    artifact_errors.append("artifact_path_or_recorded_hash_mismatch")

  verified_paths: list[str] = []
  unavailable_paths: list[str] = []
  for path_value, expected_hash in _EXPECTED_M12_ARTIFACTS.items():
    path = Path(path_value)
    if not path.exists():
      unavailable_paths.append(path_value)
      continue
    if not path.is_file():
      artifact_errors.append(f"{path_value}:not_a_file")
      continue
    try:
      observed_hash = _sha256_file(path)
    except OSError as exc:
      artifact_errors.append(f"{path_value}:unreadable:{exc}")
      continue
    if observed_hash != expected_hash:
      artifact_errors.append(f"{path_value}:content_hash_mismatch")
    else:
      verified_paths.append(path_value)

  results.append(
      _semantic_check(
          "m12_artifacts_locked_and_verified_when_available",
          not artifact_errors,
          (
              "Recorded paths/hashes are frozen. Locally available source "
              "artifacts must also match their content hashes."
          ),
          {
              "verified": verified_paths,
              "unavailable": unavailable_paths,
              "errors": artifact_errors,
          },
      )
  )

  inventory = report.get("scaffold_inventory")
  inventory_items = inventory if isinstance(inventory, list) else []
  inventory_paths = [
      item.get("path") for item in inventory_items if isinstance(item, dict)
  ]
  expected_inventory = Counter(REQUIRED_SCAFFOLD_PATHS)
  inventory_exact = Counter(inventory_paths) == expected_inventory
  inventory_exists = inventory_exact and all(
      (root / path).is_file() for path in inventory_paths
  )
  results.append(
      _semantic_check(
          "scaffold_inventory_exact_and_present",
          inventory_exists,
          observed=inventory_paths,
      )
  )

  preservation = report.get("preservation")
  preservation = preservation if isinstance(preservation, dict) else {}
  preservation_ok = (
      preservation.get("legacy_model_files_modified") is False
      and preservation.get("legacy_report_fields_changed") is False
      and preservation.get("legacy_manifest_modified") is False
      and preservation.get("legacy_validator_modified") is False
      and preservation.get("mg0a_freeze_preserved") is True
      and preservation.get("mg0b_profile_path")
      == "configs/market_genome/contracts_manifest.yaml"
  )
  results.append(
      _semantic_check(
          "legacy_contracts_preserved_with_independent_profile",
          preservation_ok,
          observed=preservation,
      )
  )

  resolved = report.get("resolved_config")
  config_hash = report.get("resolved_config_sha256")
  results.append(
      _semantic_check(
          "resolved_config_hash_matches",
          isinstance(resolved, dict)
          and isinstance(config_hash, str)
          and config_hash == _canonical_sha256(resolved),
          observed={"declared": config_hash, "computed": _canonical_sha256(resolved)},
      )
  )

  execution = report.get("execution")
  execution = execution if isinstance(execution, dict) else {}
  execution_ok = (
      execution.get("training_status") == "DEFERRED_BY_USER"
      and execution.get("gpu_status") == "DEFERRED_BY_USER"
      and execution.get("training_executed") is False
      and execution.get("jax_imported_by_cpu_only_test") is True
      and execution.get("jax_computation_executed") is False
      and execution.get("gpu_executed") is False
      and execution.get("full_backbone_implemented") is False
      and execution.get("resume_requires_explicit_user_confirmation") is True
  )
  results.append(
      _semantic_check(
          "training_gpu_and_backbone_remain_deferred",
          execution_ok,
          observed=execution,
      )
  )

  metadata = report.get("metadata")
  metadata = metadata if isinstance(metadata, dict) else {}
  gate = report.get("gate")
  gate = gate if isinstance(gate, dict) else {}
  gate_checks = gate.get("checks") if isinstance(gate.get("checks"), list) else []
  conclusion_ok = (
      metadata.get("implementation_scope") == "SCAFFOLD_ONLY"
      and metadata.get("audit_status") == "COMPLETE_WITH_DOCUMENTED_BOUNDARIES"
      and metadata.get("gate_decision") == "PASS_WITH_DOCUMENTED_BOUNDARIES"
      and gate.get("overall") == metadata.get("gate_decision")
      and bool(gate_checks)
      and all(
          isinstance(item, dict) and item.get("passed") is True
          for item in gate_checks
      )
  )
  results.append(
      _semantic_check(
          "scaffold_only_gate_conclusion_consistent",
          conclusion_ok,
          observed={"metadata": metadata, "gate": gate},
      )
  )

  uncertainties = report.get("uncertainties")
  uncertainty_items = uncertainties if isinstance(uncertainties, list) else []
  uncertainties_open = bool(uncertainty_items) and all(
      isinstance(item, dict) and item.get("status") == "NEEDS_VERIFICATION"
      for item in uncertainty_items
  )
  results.append(
      _semantic_check(
          "uncertainties_remain_open", uncertainties_open, observed=uncertainty_items
      )
  )
  return results


def schema_errors(report: Any) -> list[str]:
  """Return stable JSON Schema error strings for an audit payload."""
  try:
    import jsonschema
  except ImportError as exc:  # pragma: no cover - environment dependency guard
    return [f"jsonschema unavailable: {exc}"]
  schema = json.loads(report_schema_path().read_text(encoding="utf-8"))
  validator = jsonschema.Draft7Validator(schema)
  errors = sorted(
      validator.iter_errors(report),
      key=lambda error: tuple(str(part) for part in error.path),
  )
  return [
      f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
      for error in errors
  ]


def validate_report_bundle(
    report_path: Path,
    markdown_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
  """Validate the MG0-B JSON/Markdown pair and return a serializable result."""
  load_error = ""
  report: Any = None
  try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    load_error = str(exc)

  schema_error_list = schema_errors(report) if not load_error else [load_error]
  semantic = (
      semantic_checks(report, repo_root=repo_root)
      if isinstance(report, dict) and not schema_error_list
      else [
          _semantic_check(
              "report_payload_load", False, load_error or "schema failed"
          )
      ]
  )
  markdown_hash = _sha256_file(markdown_path) if markdown_path.is_file() else None
  declared_markdown_hash = (
      report.get("markdown_sha256") if isinstance(report, dict) else None
  )
  markdown_ok = (
      markdown_hash == _EXPECTED_MARKDOWN_SHA256
      and declared_markdown_hash == _EXPECTED_MARKDOWN_SHA256
  )
  semantic.append(
      _semantic_check(
          "markdown_hash_matches",
          markdown_ok,
          "The Markdown companion is frozen and bound to the JSON payload.",
          {
              "path": str(markdown_path),
              "expected": _EXPECTED_MARKDOWN_SHA256,
              "declared": declared_markdown_hash,
              "observed": markdown_hash,
          },
      )
  )
  semantic_passed = sum(item["status"] == "pass" for item in semantic)
  all_pass = not schema_error_list and semantic_passed == len(semantic)
  return {
      "profile": PROFILE_NAME,
      "report": str(report_path),
      "markdown": str(markdown_path),
      "schema": str(report_schema_path()),
      "schema_status": "pass" if not schema_error_list else "fail",
      "schema_errors": schema_error_list,
      "semantic_results": semantic,
      "summary": {
          "schema_total": 1,
          "schema_passed": 0 if schema_error_list else 1,
          "semantic_total": len(semantic),
          "semantic_passed": semantic_passed,
          "overall": "pass" if all_pass else "fail",
      },
  }


def _write_validation_markdown(result: dict[str, Any], path: Path) -> None:
  summary = result["summary"]
  lines = [
      "# MG0-B Schema Validation",
      "",
      f"- Profile: `{result['profile']}`",
      f"- Schema: `{summary['schema_passed']}/{summary['schema_total']}`",
      f"- Semantic: `{summary['semantic_passed']}/{summary['semantic_total']}`",
      f"- Overall: `{summary['overall'].upper()}`",
      "",
      "## Semantic Checks",
      "",
      "| Check | Status |",
      "|---|---|",
  ]
  lines.extend(
      f"| `{item['name']}` | `{item['status'].upper()}` |"
      for item in result["semantic_results"]
  )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Strict MG0-B report validation")
  parser.add_argument("--profile", choices=[PROFILE_NAME], default=PROFILE_NAME)
  parser.add_argument("--reports-dir", type=Path, required=True)
  parser.add_argument("--output-json", type=Path)
  parser.add_argument("--output-md", type=Path)
  parser.add_argument("--strict", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  report_path = args.reports_dir / f"{REPORT_BASENAME}.json"
  markdown_path = args.reports_dir / f"{REPORT_BASENAME}.md"
  result = validate_report_bundle(report_path, markdown_path)
  output_json = args.output_json or args.reports_dir / "mg0b_schema_validation.json"
  output_md = args.output_md or args.reports_dir / "mg0b_schema_validation.md"
  output_json.parent.mkdir(parents=True, exist_ok=True)
  output_md.parent.mkdir(parents=True, exist_ok=True)
  output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  _write_validation_markdown(result, output_md)
  summary = result["summary"]
  print(
      "MG0-B validation: "
      f"schema {summary['schema_passed']}/{summary['schema_total']}, "
      f"semantic {summary['semantic_passed']}/{summary['semantic_total']}, "
      f"overall={summary['overall']}"
  )
  return 1 if args.strict and summary["overall"] != "pass" else 0


if __name__ == "__main__":
  raise SystemExit(main())
