"""CPU-only contract tests for the MG0-B scaffold."""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from alphatrade.market_genome import config
from alphatrade.market_genome import schemas


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_JSON = (
    REPO_ROOT
    / "docs/alphaTrade/market_genome/MG0_ALPHATRADE_COMPATIBILITY_AUDIT.json"
)
CANONICAL_MD = CANONICAL_JSON.with_suffix(".md")


def _load_report() -> dict:
  return json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))


def _semantic_status(report: dict, name: str) -> str:
  checks = schemas.semantic_checks(report, repo_root=REPO_ROOT)
  return next(item["status"] for item in checks if item["name"] == name)


def test_package_import_is_lightweight_and_gpu_neutral() -> None:
  script = """
import json
import sys
import alphatrade.market_genome as mg
blocked = sorted(
    name for name in sys.modules
    if name == "jax"
    or name.startswith("jax.")
    or name == "haiku"
    or name.startswith("haiku.")
    or name in {"alphatrade.core.model", "alphatrade.training"}
)
print(json.dumps({
    "status": mg.IMPLEMENTATION_STATUS,
    "backbone": mg.FULL_BACKBONE_IMPLEMENTED,
    "training": mg.TRAINING_ENABLED,
    "blocked": blocked,
}))
"""
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  completed = subprocess.run(
      [sys.executable, "-c", script],
      cwd=REPO_ROOT,
      env=env,
      check=True,
      capture_output=True,
      text=True,
  )
  observed = json.loads(completed.stdout)
  assert observed == {
      "status": "SCAFFOLD_ONLY",
      "backbone": False,
      "training": False,
      "blocked": [],
  }


@pytest.mark.parametrize(
    "subpackage",
    [
        "ontology",
        "synthetic",
        "datasets",
        "heads",
        "losses",
        "training",
        "perturbation",
        "evaluation",
    ],
)
def test_placeholder_packages_are_import_safe(subpackage: str) -> None:
  module = importlib.import_module(f"alphatrade.market_genome.{subpackage}")
  assert module.IMPLEMENTATION_STATUS == "NOT_IMPLEMENTED"
  assert module.__all__ == ()


def test_scaffold_config_is_resolved_and_model_config_is_deferred() -> None:
  scaffold = config.MarketGenomeScaffoldConfig()
  assert scaffold.to_dict() == config.resolved_scaffold_config()
  assert (
      scaffold.sha256()
      == "927c4775b7a857b2346dbfbe44c101cac9d8471b795a8f0733a02201dab9d451"
  )
  with pytest.raises(NotImplementedError, match="not implemented in MG0-B"):
    config.load_model_config()


def test_classification_contract_is_exact() -> None:
  assert {item.value for item in schemas.CompatibilityDisposition} == {
      "DIRECT_REUSE",
      "REUSE_WITH_ADAPTER",
      "ISOLATE",
      "REPLACE_FOR_MARKETGENOME",
      "DEFER",
  }


def test_independent_profile_does_not_extend_legacy_manifest() -> None:
  independent = yaml.safe_load(
      (REPO_ROOT / "configs/market_genome/contracts_manifest.yaml").read_text(
          encoding="utf-8"
      )
  )
  legacy = yaml.safe_load(
      (REPO_ROOT / "src/alphatrade/schemas/contracts_manifest.yaml").read_text(
          encoding="utf-8"
      )
  )
  profile = independent["profiles"]["mg0b"]
  assert [item["required"] for item in profile["items"]] == [True, True]
  assert profile["items"][0]["schema"].startswith(
      "src/alphatrade/market_genome/"
  )
  assert "mg0b" not in legacy["profiles"]


def test_canonical_report_passes_schema_and_semantics() -> None:
  report = _load_report()
  assert schemas.schema_errors(report) == []
  checks = schemas.semantic_checks(report, repo_root=REPO_ROOT)
  assert checks
  assert [item for item in checks if item["status"] != "pass"] == []

  result = schemas.validate_report_bundle(
      CANONICAL_JSON, CANONICAL_MD, repo_root=REPO_ROOT
  )
  assert result["summary"] == {
      "schema_total": 1,
      "schema_passed": 1,
      "semantic_total": len(checks) + 1,
      "semantic_passed": len(checks) + 1,
      "overall": "pass",
  }


def test_subsystem_deletion_and_duplication_fail_closed() -> None:
  missing = _load_report()
  missing["compatibility_matrix"].pop()
  assert (
      _semantic_status(missing, "required_subsystems_exact_and_unique") == "fail"
  )

  duplicate = _load_report()
  duplicate["compatibility_matrix"][-1] = copy.deepcopy(
      duplicate["compatibility_matrix"][0]
  )
  assert (
      _semantic_status(duplicate, "required_subsystems_exact_and_unique")
      == "fail"
  )


def test_evidence_path_and_line_tampering_fails_closed() -> None:
  report = _load_report()
  evidence = report["compatibility_matrix"][0]["evidence"][0]
  evidence["end_line"] = 10_000_000
  assert (
      _semantic_status(report, "repository_evidence_paths_and_lines_valid")
      == "fail"
  )

  report = _load_report()
  evidence = report["compatibility_matrix"][0]["evidence"][0]
  evidence["path"] = "../outside.py"
  assert (
      _semantic_status(report, "repository_evidence_paths_and_lines_valid")
      == "fail"
  )


def test_m12_schema_pass_cannot_mask_model_quality_fail() -> None:
  report = _load_report()
  report["m12_evidence"]["model_quality_status"] = "PASS"
  report["m12_evidence"]["formal_decision"] = "PASS"
  report["m12_evidence"]["promotion_status"] = "PROMOTED"
  assert schemas.schema_errors(report)
  assert (
      _semantic_status(report, "m12_contract_and_model_quality_separated")
      == "fail"
  )

  report = _load_report()
  report["compatibility_matrix"][0]["evidence"][0]["claim"] = (
      "This source proves an unrelated false statement."
  )
  assert schemas.schema_errors(report) == []
  assert _semantic_status(report, "canonical_audit_payload_hash_matches") == "fail"

  report = _load_report()
  report["m12_evidence"]["artifacts"][0]["path"] = "/definitely/missing"
  report["m12_evidence"]["artifacts"][0]["sha256"] = "0" * 64
  report["m12_evidence"]["leaderboard_summary"]["control_primary_mean"] = 999.0
  assert schemas.schema_errors(report)
  assert _semantic_status(report, "canonical_audit_payload_hash_matches") == "fail"
  assert (
      _semantic_status(
          report, "m12_artifacts_locked_and_verified_when_available"
      )
      == "fail"
  )

  report = _load_report()
  report["m12_evidence"]["overall"] = "PASS"
  assert schemas.schema_errors(report)
  assert (
      _semantic_status(report, "m12_contract_and_model_quality_separated")
      == "fail"
  )


def test_inventory_config_execution_and_gate_tampering_fail_closed() -> None:
  report = _load_report()
  report["scaffold_inventory"].pop()
  assert _semantic_status(report, "scaffold_inventory_exact_and_present") == "fail"

  report = _load_report()
  report["resolved_config"]["training_enabled"] = True
  assert _semantic_status(report, "resolved_config_hash_matches") == "fail"

  report = _load_report()
  report["execution"]["gpu_executed"] = True
  assert (
      _semantic_status(report, "training_gpu_and_backbone_remain_deferred")
      == "fail"
  )

  report = _load_report()
  report["gate"]["checks"][0]["passed"] = False
  assert (
      _semantic_status(report, "scaffold_only_gate_conclusion_consistent")
      == "fail"
  )


def test_malformed_json_and_empty_markdown_fail_closed(tmp_path: Path) -> None:
  report_path = tmp_path / "report.json"
  markdown_path = tmp_path / "report.md"
  report_path.write_text("{not-json", encoding="utf-8")
  markdown_path.write_text("", encoding="utf-8")
  result = schemas.validate_report_bundle(
      report_path, markdown_path, repo_root=REPO_ROOT
  )
  assert result["summary"]["overall"] == "fail"
  assert result["schema_errors"]
  assert result["semantic_results"][-1]["name"] == "markdown_hash_matches"
  assert result["semantic_results"][-1]["status"] == "fail"

  fake_markdown = tmp_path / "nonempty-unrelated.md"
  fake_markdown.write_text("not the MG0-B audit\n", encoding="utf-8")
  result = schemas.validate_report_bundle(
      CANONICAL_JSON, fake_markdown, repo_root=REPO_ROOT
  )
  assert result["summary"]["overall"] == "fail"
  markdown_check = result["semantic_results"][-1]
  assert markdown_check["name"] == "markdown_hash_matches"
  assert markdown_check["status"] == "fail"


def test_strict_cli_writes_validation_reports(tmp_path: Path) -> None:
  reports_dir = tmp_path / "reports"
  reports_dir.mkdir()
  shutil.copyfile(
      CANONICAL_JSON,
      reports_dir / "mg0_alphatrade_compatibility_audit.json",
  )
  shutil.copyfile(
      CANONICAL_MD,
      reports_dir / "mg0_alphatrade_compatibility_audit.md",
  )
  output_json = reports_dir / "validation.json"
  output_md = reports_dir / "validation.md"
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.schemas",
          "--profile",
          "mg0b",
          "--reports-dir",
          str(reports_dir),
          "--output-json",
          str(output_json),
          "--output-md",
          str(output_md),
          "--strict",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=True,
      capture_output=True,
      text=True,
  )
  result = json.loads(output_json.read_text(encoding="utf-8"))
  assert result["summary"]["overall"] == "pass"
  assert "overall=pass" in completed.stdout
  assert "Overall: `PASS`" in output_md.read_text(encoding="utf-8")
