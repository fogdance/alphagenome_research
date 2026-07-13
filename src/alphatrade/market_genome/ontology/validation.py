"""Strict CPU-only report generation and validation for MG1-A."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from alphatrade.market_genome.ontology import schema as ontology_schema


PROFILE_NAME = "mg1a"
REPORT_BASENAME = "mg1_pattern_ontology"
VALIDATION_BASENAME = "mg1a_schema_validation"
INTERPRETATION = "draft contract valid; human freeze pending"

_REQUIRED_MARKDOWN_SECTIONS = (
    "## Glossary",
    "## Family-to-Anchor Diagrams",
    "## Phase Transition Tables",
    "## Open Questions",
    "## Rejected Shortcuts",
    "## Validation Command",
    "## Human Review Checklist",
)

_MG0B_FROZEN_HASHES = {
    "configs/market_genome/README.md": (
        "12ea30870a73e38d42dba070e349474a84a84d43103b1251b343e8c8a0fe3561"
    ),
    "configs/market_genome/contracts_manifest.yaml": (
        "f79ece24e8132d829c9d637420a33c8c9960dc42bbc77c6b718faf6d8ef28922"
    ),
    "src/alphatrade/market_genome/schemas.py": (
        "dd1fa3981ff8474e025d4c343909cc8167765ee525a4867224ce7f3e3b2c2448"
    ),
    "src/alphatrade/market_genome/mg0b_compatibility_audit.schema.json": (
        "223c2b976e972b8f67402c9938a6d8e491cb02c86ddc66bff07148c6e8c9088d"
    ),
    "tests/market_genome/test_mg0b_scaffold.py": (
        "07415f9b59771f797784bf4f5a52ee959f73c3618cfb445dbec0384f85ce9a56"
    ),
    "docs/alphaTrade/market_genome/MG0_ALPHATRADE_COMPATIBILITY_AUDIT.json": (
        "a52f1dbc8d2138f0eca81c15ad514ac00f6ff5a6002a6044769607bed9218e64"
    ),
    "docs/alphaTrade/market_genome/MG0_ALPHATRADE_COMPATIBILITY_AUDIT.md": (
        "b11c2dc7c5784d635480df7c3f5031c4b036a47a0e089f74ec45dac26df5c7a5"
    ),
    "src/alphatrade/market_genome/README.md": (
        "ff40a15065c5d69bfee9c86c9a0b1f1de67b417dbfaff177f8e340641faa730d"
    ),
    "src/alphatrade/market_genome/__init__.py": (
        "b836a6eecd1d5d632d544903910ffc3fefa4ed93c7ac89747c45a08f25e2f97a"
    ),
    "src/alphatrade/market_genome/config.py": (
        "851218be776393c4d107ec06285c479a6feeefa135efe0dec0a8c23bb1482518"
    ),
    "src/alphatrade/market_genome/datasets/__init__.py": (
        "4744d77344d6b1cc590b5c6fe0389e2d574faf55f21d97fc26ef86b3a45f8039"
    ),
    "src/alphatrade/market_genome/evaluation/__init__.py": (
        "6433ec534a486e18fa79d9d3a39aa6e85b33c3a38d4f7b5f4e15188540fd6237"
    ),
    "src/alphatrade/market_genome/heads/__init__.py": (
        "61595209bc4e74cd13694b0b201655b4cc81dc138dfa77e5f6db30e035cc4255"
    ),
    "src/alphatrade/market_genome/losses/__init__.py": (
        "02d0ecf260ae6e49113ee7156b33f21503f9475fd054a931b4cc471b2c7f54fd"
    ),
    "src/alphatrade/market_genome/ontology/__init__.py": (
        "f67e90fc85cd5a59d1e636c18ff5a5145bec558dbabefb30e35767d963d46bb9"
    ),
    "src/alphatrade/market_genome/perturbation/__init__.py": (
        "f4e9b606bf71ab50a5a70f0994606f5ad75c7b3cddd3c693608df23c8dd7e4e4"
    ),
    "src/alphatrade/market_genome/synthetic/__init__.py": (
        "1d2e492a1a4d56f994a6276124d1b1e9c735de4c3be21bd8496187807e8c877b"
    ),
    "src/alphatrade/market_genome/training/__init__.py": (
        "e12592269a8dee389ddfc83114cbf4c6600ee81d3baa8842375aa67935ec3ab6"
    ),
    "tests/market_genome/__init__.py": (
        "283915d15826fc61dcd8016389c4d53ec6bca42290773f99a2bcbe62711ae99c"
    ),
}

_MG1A_ARTIFACTS = (
    "configs/market_genome/pattern_ontology_v1.yaml",
    "configs/market_genome/mg1a_contracts_manifest.yaml",
    "docs/alphaTrade/market_genome/MG1_PATTERN_ONTOLOGY.md",
    "src/alphatrade/market_genome/ontology/schema.py",
    "src/alphatrade/market_genome/ontology/validation.py",
    "src/alphatrade/market_genome/ontology/mg1a_pattern_ontology.schema.json",
    "tests/market_genome/test_mg1a_ontology.py",
)


def repository_root() -> Path:
  """Return the repository containing the ontology package."""
  return Path(__file__).resolve().parents[4]


def ontology_config_path(repo_root: Path | None = None) -> Path:
  root = (repo_root or repository_root()).resolve()
  return root / "configs/market_genome/pattern_ontology_v1.yaml"


def ontology_document_path(repo_root: Path | None = None) -> Path:
  root = (repo_root or repository_root()).resolve()
  return root / "docs/alphaTrade/market_genome/MG1_PATTERN_ONTOLOGY.md"


def report_schema_path(repo_root: Path | None = None) -> Path:
  if repo_root is None:
    return Path(__file__).with_name("mg1a_pattern_ontology.schema.json")
  return (
      repo_root.resolve()
      / "src/alphatrade/market_genome/ontology/"
      "mg1a_pattern_ontology.schema.json"
  )


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


def schema_errors(
    report: Any, repo_root: Path | None = None
) -> list[str]:
  """Return stable Draft-07 validation errors for an MG1-A report."""
  return ontology_schema.json_schema_errors(
      report,
      schema_path=report_schema_path(repo_root),
  )


def _load_yaml(path: Path) -> tuple[Any, str]:
  try:
    return ontology_schema.loads_yaml_unique(
        path.read_text(encoding="utf-8")
    ), ""
  except (
      OSError,
      UnicodeError,
      RuntimeError,
      ontology_schema.OntologyValidationError,
  ) as exc:
    return None, str(exc)
  except ValueError as exc:
    return None, str(exc)


def _line_range_errors(path: Path, line_range: str) -> list[str]:
  try:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
      line_count = sum(1 for _ in handle)
  except OSError as exc:
    return [f"unreadable:{exc}"]
  errors: list[str] = []
  for raw_part in line_range.split(","):
    part = raw_part.strip()
    values = part.split("-", maxsplit=1)
    try:
      start = int(values[0])
      end = int(values[-1])
    except ValueError:
      errors.append(f"invalid_range:{part}")
      continue
    if start < 1 or end < start or end > line_count:
      errors.append(f"out_of_bounds:{part}/{line_count}")
  return errors


def _evidence_checks(
    report: dict[str, Any],
    root: Path,
    evidence_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
  evidence = report.get("source_evidence")
  items = evidence if isinstance(evidence, list) else []
  identifiers: list[str] = []
  errors: list[str] = []
  verified: list[str] = []
  unavailable: list[str] = []
  configured_roots: dict[str, Path | None] = {
      "REPOSITORY": root.resolve(),
      "CHENDAGE_SIGNAL": None,
      "CHEN_PACK": None,
  }
  env_roots = {
      "CHENDAGE_SIGNAL": os.environ.get("CHENDAGE_SIGNAL_ROOT"),
      "CHEN_PACK": os.environ.get("CHEN_REFERENCE_ROOT"),
  }
  for name, value in env_roots.items():
    if value:
      configured_roots[name] = Path(value).expanduser().resolve()
  for name, value in (evidence_roots or {}).items():
    if name in {"CHENDAGE_SIGNAL", "CHEN_PACK"}:
      configured_roots[name] = Path(value).expanduser().resolve()

  for index, raw in enumerate(items):
    if not isinstance(raw, dict):
      errors.append(f"{index}:not_object")
      continue
    identifier = raw.get("evidence_id")
    source_root = raw.get("source_root")
    path_value = raw.get("path")
    expected_hash = raw.get("sha256")
    availability = raw.get("availability")
    line_range = raw.get("line_range")
    if isinstance(identifier, str):
      identifiers.append(identifier)
    if (
        not isinstance(identifier, str)
        or not identifier
        or source_root not in configured_roots
        or not isinstance(path_value, str)
        or not path_value
        or Path(path_value).is_absolute()
        or not isinstance(expected_hash, str)
        or not isinstance(line_range, str)
        or availability not in {"REQUIRED", "VERIFY_WHEN_AVAILABLE"}
    ):
      errors.append(f"{identifier or index}:invalid_evidence_location")
      continue

    base = configured_roots[source_root]
    if base is None:
      if availability == "REQUIRED":
        errors.append(f"{identifier}:required_source_root_unavailable")
      else:
        unavailable.append(str(identifier))
      continue
    resolved = (base / path_value).resolve()
    try:
      resolved.relative_to(base)
    except ValueError:
      errors.append(f"{identifier}:outside_source_root")
      continue
    if not resolved.is_file():
      if availability == "REQUIRED" or source_root != "REPOSITORY":
        errors.append(f"{identifier}:missing:{resolved}")
      else:
        unavailable.append(str(identifier))
      continue
    try:
      observed_hash = _sha256_file(resolved)
    except OSError as exc:
      errors.append(f"{identifier}:unreadable:{exc}")
      continue
    if observed_hash != expected_hash:
      errors.append(f"{identifier}:content_hash_mismatch")
      continue
    range_errors = _line_range_errors(resolved, line_range)
    errors.extend(f"{identifier}:{error}" for error in range_errors)
    if not range_errors:
      verified.append(str(identifier))
  if len(identifiers) != len(set(identifiers)):
    errors.append("duplicate_evidence_id")

  referenced: list[str] = []
  contract = report.get("resolved_contract")
  if isinstance(contract, dict):
    for assumption in contract.get("assumptions", []):
      if isinstance(assumption, dict):
        values = assumption.get("source_evidence", [])
        if isinstance(values, list):
          referenced.extend(value for value in values if isinstance(value, str))
  for shortcut in report.get("rejected_shortcuts", []):
    if isinstance(shortcut, dict):
      values = shortcut.get("evidence_ids", [])
      if isinstance(values, list):
        referenced.extend(value for value in values if isinstance(value, str))
  unknown = sorted(set(referenced) - set(identifiers))
  if unknown:
    errors.append(f"unknown_evidence_references:{unknown}")
  return {
      "evidence_count": len(items),
      "referenced_count": len(set(referenced)),
      "verified": verified,
      "unavailable": unavailable,
      "configured_roots": {
          name: str(value) if value is not None else None
          for name, value in configured_roots.items()
      },
      "errors": errors,
  }


def _mg0b_preservation(root: Path) -> dict[str, Any]:
  errors: list[str] = []
  observed: dict[str, str | None] = {}
  frozen_paths = set(_MG0B_FROZEN_HASHES)
  for relative, expected in _MG0B_FROZEN_HASHES.items():
    path = root / relative
    try:
      value = _sha256_file(path) if path.is_file() else None
    except OSError:
      value = None
    observed[relative] = value
    if value != expected:
      errors.append(relative)
  return {
      "expected": _MG0B_FROZEN_HASHES,
      "observed": observed,
      "expected_paths": sorted(frozen_paths),
      "errors": errors,
  }


def _mg0b_bundle_revalidation(
    root: Path, preservation: dict[str, Any]
) -> dict[str, Any]:
  if preservation["errors"]:
    return {
        "valid": False,
        "skipped": True,
        "reason": "frozen byte validation failed",
    }
  from alphatrade.market_genome import schemas as mg0b_schemas

  frozen_paths = set(_MG0B_FROZEN_HASHES)
  declared_paths = set(mg0b_schemas.REQUIRED_SCAFFOLD_PATHS)
  if declared_paths != frozen_paths:
    return {
        "valid": False,
        "skipped": False,
        "reason": "frozen_path_set_mismatch",
        "declared_paths": sorted(declared_paths),
        "expected_paths": sorted(frozen_paths),
    }

  report = (
      root
      / "docs/alphaTrade/market_genome/"
      "MG0_ALPHATRADE_COMPATIBILITY_AUDIT.json"
  )
  markdown = (
      root
      / "docs/alphaTrade/market_genome/"
      "MG0_ALPHATRADE_COMPATIBILITY_AUDIT.md"
  )
  try:
    result = mg0b_schemas.validate_report_bundle(
        report, markdown, repo_root=root
    )
  except (OSError, TypeError, ValueError) as exc:
    return {
        "valid": False,
        "skipped": False,
        "error": f"{type(exc).__name__}: {exc}",
    }
  return {
      "valid": result.get("summary", {}).get("overall") == "pass",
      "skipped": False,
      "summary": result.get("summary"),
      "schema_errors": result.get("schema_errors"),
      "failed_semantic_checks": [
          item.get("name")
          for item in result.get("semantic_results", [])
          if item.get("status") != "pass"
      ],
  }


def _manifest_check(root: Path) -> dict[str, Any]:
  path = root / "configs/market_genome/mg1a_contracts_manifest.yaml"
  payload, load_error = _load_yaml(path)
  expected_items = [
      {
          "name": "mg1_pattern_ontology",
          "path": "mg1_pattern_ontology.json",
          "schema": (
              "src/alphatrade/market_genome/ontology/"
              "mg1a_pattern_ontology.schema.json"
          ),
          "required": True,
      },
      {
          "name": "mg1_pattern_ontology_md",
          "path": "mg1_pattern_ontology.md",
          "schema": "",
          "required": True,
      },
  ]
  expected_profile = {
      "description": (
          "MG1-A parameterized pattern ontology review-draft contract."
      ),
      "reports_dir": "$ALPHATRADE_RUNS_ROOT/reports",
      "decision": ontology_schema.DECISION,
      "items": expected_items,
  }
  expected_payload = {"version": 1, "profiles": {"mg1a": expected_profile}}
  profile = (
      payload.get("profiles", {}).get("mg1a", {})
      if isinstance(payload, dict)
      else {}
  )
  valid = not load_error and payload == expected_payload
  return {
      "path": str(path),
      "load_error": load_error,
      "profile": profile,
      "expected": expected_payload,
      "valid": valid,
  }


def _manifest_report_paths(
    reports_dir: Path, root: Path
) -> tuple[Path, Path]:
  manifest = _manifest_check(root)
  if not manifest["valid"]:
    raise ontology_schema.OntologyValidationError(
        ["MG1-A contracts manifest is invalid"]
    )
  items = {
      item["name"]: item
      for item in manifest["profile"]["items"]
  }
  return (
      reports_dir / items["mg1_pattern_ontology"]["path"],
      reports_dir / items["mg1_pattern_ontology_md"]["path"],
  )


def _default_reports_dir(root: Path) -> Path:
  manifest = _manifest_check(root)
  if not manifest["valid"]:
    raise ontology_schema.OntologyValidationError(
        ["MG1-A contracts manifest is invalid"]
    )
  runs_root = os.environ.get("ALPHATRADE_RUNS_ROOT")
  if not runs_root:
    raise ontology_schema.OntologyValidationError(
        ["ALPHATRADE_RUNS_ROOT is required when --reports-dir is omitted"]
    )
  template = manifest["profile"]["reports_dir"]
  if template != "$ALPHATRADE_RUNS_ROOT/reports":
    raise ontology_schema.OntologyValidationError(
        ["unsupported MG1-A reports_dir template"]
    )
  return Path(runs_root).expanduser().resolve() / "reports"


def _require_distinct_report_paths(paths: dict[str, Path]) -> None:
  resolved: dict[Path, list[str]] = {}
  for name, path in paths.items():
    resolved.setdefault(path.expanduser().resolve(), []).append(name)
  collisions = [names for names in resolved.values() if len(names) > 1]
  if collisions:
    raise ontology_schema.OntologyValidationError(
        [f"report path collision: {collisions}"]
    )


def semantic_checks(
    report: dict[str, Any],
    markdown_path: Path,
    repo_root: Path | None = None,
    evidence_roots: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
  """Run semantic checks that JSON Schema cannot express."""
  root = (repo_root or repository_root()).resolve()
  results: list[dict[str, Any]] = []

  ontology_errors = ontology_schema.validate_ontology_payload(report)
  results.append(
      _semantic_check(
          "ontology_semantics_valid",
          not ontology_errors,
          "Parameterized draft invariants must fail closed.",
          ontology_errors,
      )
  )

  yaml_payload, yaml_error = _load_yaml(ontology_config_path(root))
  results.append(
      _semantic_check(
          "report_matches_resolved_yaml",
          not yaml_error and report == yaml_payload,
          "The JSON report is the fully resolved YAML contract.",
          {"load_error": yaml_error, "matches": report == yaml_payload},
      )
  )

  contract = report.get("resolved_contract")
  declared_hash = report.get("resolved_contract_sha256")
  computed_hash = ontology_schema.canonical_sha256(contract)
  results.append(
      _semantic_check(
          "resolved_contract_hash_matches",
          declared_hash == computed_hash,
          observed={"declared": declared_hash, "computed": computed_hash},
      )
  )

  human_review = report.get("human_review")
  draft_boundary_ok = (
      report.get("decision") == ontology_schema.DECISION
      and isinstance(human_review, dict)
      and human_review.get("status") == "PENDING"
      and human_review.get("frozen") is False
      and human_review.get("approver") is None
      and human_review.get("approved_at") is None
      and human_review.get("frozen_sha256") is None
  )
  results.append(
      _semantic_check(
          "draft_decision_and_human_freeze_boundary",
          draft_boundary_ok,
          INTERPRETATION,
          {"decision": report.get("decision"), "human_review": human_review},
      )
  )

  expected_execution = {
      "detector_implemented": False,
      "labeler_implemented": False,
      "real_data_labels_created": False,
      "synthetic_samples_created": False,
      "trading_rules_implemented": False,
      "model_training_executed": False,
      "jax_computation_executed": False,
      "gpu_executed": False,
  }
  results.append(
      _semantic_check(
          "detector_labels_training_and_gpu_deferred",
          report.get("execution") == expected_execution,
          observed=report.get("execution"),
      )
  )

  source_document = ontology_document_path(root)
  try:
    markdown_bytes = (
        markdown_path.read_bytes() if markdown_path.is_file() else b""
    )
    source_bytes = (
        source_document.read_bytes() if source_document.is_file() else b""
    )
    markdown_read_error = ""
  except OSError as exc:
    markdown_bytes = b""
    source_bytes = b""
    markdown_read_error = str(exc)
  markdown_text = markdown_bytes.decode("utf-8", errors="replace")
  missing_sections = [
      section
      for section in _REQUIRED_MARKDOWN_SECTIONS
      if section not in markdown_text
  ]
  results.append(
      _semantic_check(
          "markdown_matches_source_and_required_sections",
          bool(markdown_bytes)
          and markdown_bytes == source_bytes
          and not missing_sections,
          observed={
              "report_sha256": (
                  hashlib.sha256(markdown_bytes).hexdigest()
                  if markdown_bytes
                  else None
              ),
              "source_sha256": (
                  hashlib.sha256(source_bytes).hexdigest()
                  if source_bytes
                  else None
              ),
              "missing_sections": missing_sections,
              "read_error": markdown_read_error,
          },
      )
  )

  evidence = _evidence_checks(
      report, root, evidence_roots=evidence_roots
  )
  results.append(
      _semantic_check(
          "source_evidence_paths_lines_and_references_valid",
          evidence["evidence_count"] > 0 and not evidence["errors"],
          observed=evidence,
      )
  )

  manifest = _manifest_check(root)
  results.append(
      _semantic_check(
          "independent_mg1a_profile_valid",
          manifest["valid"],
          "MG1-A must not extend the frozen MG0-B manifest.",
          manifest,
      )
  )

  artifact_presence = {
      relative: (root / relative).is_file() for relative in _MG1A_ARTIFACTS
  }
  results.append(
      _semantic_check(
          "required_mg1a_artifacts_present",
          all(artifact_presence.values()),
          observed=artifact_presence,
      )
  )

  preservation = _mg0b_preservation(root)
  results.append(
      _semantic_check(
          "mg0b_frozen_bytes_preserved",
          not preservation["errors"],
          observed=preservation,
      )
  )
  mg0b_bundle = _mg0b_bundle_revalidation(root, preservation)
  results.append(
      _semantic_check(
          "mg0b_frozen_bundle_revalidated",
          mg0b_bundle["valid"],
          observed=mg0b_bundle,
      )
  )
  return results


def artifact_hashes(
    report_path: Path, markdown_path: Path, repo_root: Path | None = None
) -> list[dict[str, Any]]:
  root = (repo_root or repository_root()).resolve()
  records: list[dict[str, Any]] = []
  for relative in _MG1A_ARTIFACTS:
    path = root / relative
    try:
      value = _sha256_file(path) if path.is_file() else None
    except OSError:
      value = None
    records.append(
        {
            "path": relative,
            "sha256": value,
        }
    )
  for path in (report_path, markdown_path):
    try:
      value = _sha256_file(path) if path.is_file() else None
    except OSError:
      value = None
    records.append(
        {
            "path": str(path),
            "sha256": value,
        }
    )
  return records


def generate_report_bundle(
    reports_dir: Path, repo_root: Path | None = None
) -> tuple[Path, Path]:
  """Generate the required JSON/Markdown reports from canonical sources."""
  root = (repo_root or repository_root()).resolve()
  draft = ontology_schema.load_ontology(ontology_config_path(root))
  reports_dir.mkdir(parents=True, exist_ok=True)
  report_path, markdown_path = _manifest_report_paths(reports_dir, root)
  report_path.write_text(
      json.dumps(draft.to_dict(), indent=2, sort_keys=True, ensure_ascii=True)
      + "\n",
      encoding="utf-8",
  )
  shutil.copyfile(ontology_document_path(root), markdown_path)
  return report_path, markdown_path


def validate_report_bundle(
    report_path: Path,
    markdown_path: Path,
    repo_root: Path | None = None,
    evidence_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
  """Validate an MG1-A JSON/Markdown pair and return a serializable result."""
  load_error = ""
  report: Any = None
  try:
    report = ontology_schema.loads_json_unique(
        report_path.read_text(encoding="utf-8")
    )
  except (OSError, UnicodeError, ValueError) as exc:
    load_error = str(exc)

  schema_error_list = (
      schema_errors(report, repo_root=repo_root)
      if not load_error
      else [load_error]
  )
  semantic = (
      semantic_checks(
          report,
          markdown_path,
          repo_root=repo_root,
          evidence_roots=evidence_roots,
      )
      if isinstance(report, dict) and not schema_error_list
      else [
          _semantic_check(
              "report_payload_load", False, load_error or "schema failed"
          )
      ]
  )
  semantic_passed = sum(item["status"] == "pass" for item in semantic)
  all_pass = not schema_error_list and semantic_passed == len(semantic)
  observed_decision = (
      report.get("decision") if isinstance(report, dict) else None
  )
  interpretation = (
      INTERPRETATION
      if all_pass
      else "invalid draft bundle; human freeze prohibited"
  )
  return {
      "profile": PROFILE_NAME,
      "decision": ontology_schema.DECISION,
      "observed_decision": observed_decision,
      "interpretation": interpretation,
      "report": str(report_path),
      "markdown": str(markdown_path),
      "schema": str(report_schema_path(repo_root)),
      "schema_status": "pass" if not schema_error_list else "fail",
      "schema_errors": schema_error_list,
      "semantic_results": semantic,
      "artifact_hashes": artifact_hashes(
          report_path, markdown_path, repo_root=repo_root
      ),
      "summary": {
          "schema_total": 1,
          "schema_passed": 0 if schema_error_list else 1,
          "semantic_total": len(semantic),
          "semantic_passed": semantic_passed,
          "overall": "pass" if all_pass else "fail",
          "human_freeze_pending": all_pass,
          "human_freeze_allowed": False,
      },
  }


def _write_validation_markdown(result: dict[str, Any], path: Path) -> None:
  summary = result["summary"]
  lines = [
      "# MG1-A Schema Validation",
      "",
      f"- Profile: `{result['profile']}`",
      f"- Decision: `{result['decision']}`",
      f"- Interpretation: `{result['interpretation']}`",
      f"- Schema: `{summary['schema_passed']}/{summary['schema_total']}`",
      f"- Semantic: `{summary['semantic_passed']}/{summary['semantic_total']}`",
      f"- Overall: `{summary['overall'].upper()}`",
      (
          "- Human freeze: `PENDING`"
          if summary["human_freeze_pending"]
          else "- Human freeze: `PROHIBITED_INVALID_DRAFT`"
      ),
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
  parser = argparse.ArgumentParser(
      description="Generate and strictly validate the MG1-A review draft"
  )
  parser.add_argument("--profile", choices=[PROFILE_NAME], default=PROFILE_NAME)
  parser.add_argument("--reports-dir", type=Path)
  parser.add_argument("--generate", action="store_true")
  parser.add_argument("--output-json", type=Path)
  parser.add_argument("--output-md", type=Path)
  parser.add_argument("--strict", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  root = repository_root()
  try:
    reports_dir = args.reports_dir or _default_reports_dir(root)
    report_path, markdown_path = _manifest_report_paths(
        reports_dir, root
    )
  except ontology_schema.OntologyValidationError as exc:
    print(f"MG1-A validation configuration error: {exc}")
    return 2
  try:
    if args.generate:
      report_path, markdown_path = generate_report_bundle(reports_dir)
    result = validate_report_bundle(report_path, markdown_path)
    output_json = (
        args.output_json
        or reports_dir / f"{VALIDATION_BASENAME}.json"
    )
    output_md = args.output_md or reports_dir / f"{VALIDATION_BASENAME}.md"
    _require_distinct_report_paths(
        {
            "ontology_json": report_path,
            "ontology_markdown": markdown_path,
            "validation_json": output_json,
            "validation_markdown": output_md,
        }
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_validation_markdown(result, output_md)
  except (
      OSError,
      UnicodeError,
      TypeError,
      ValueError,
      ontology_schema.OntologyValidationError,
  ) as exc:
    print(f"MG1-A validation runtime error: {type(exc).__name__}: {exc}")
    return 2
  summary = result["summary"]
  print(
      "MG1-A validation: "
      f"schema {summary['schema_passed']}/{summary['schema_total']}, "
      f"semantic {summary['semantic_passed']}/{summary['semantic_total']}, "
      f"overall={summary['overall']}, decision={result['decision']}; "
      f"{result['interpretation']}"
  )
  return 1 if args.strict and summary["overall"] != "pass" else 0


if __name__ == "__main__":
  raise SystemExit(main())
