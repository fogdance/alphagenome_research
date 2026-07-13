"""CPU-only tests for the MG1-A.1 AI-simulated dual review."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from alphatrade.market_genome.ontology import revision
from alphatrade.market_genome.ontology import simulated_review


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / simulated_review.BUNDLE_PATH
ARCHIVE = BUNDLE.parent


def _bundle() -> dict:
  return simulated_review.load_bundle(BUNDLE)


def _json(name: str) -> dict:
  return simulated_review.loads_json_unique(
      (ARCHIVE / name).read_text(encoding="utf-8")
  )


def _decision_materials() -> tuple[dict, dict, dict, dict, dict, dict]:
  payload = _bundle()
  adjudication = _json("adjudication.json")
  reviewer_a = _json("reviewer_a.raw.json")
  reviewer_b = _json("reviewer_b.raw.json")
  v2 = revision.load_revision(
      REPO_ROOT / "configs/market_genome/pattern_ontology_v2.yaml"
  )
  ledger = revision.loads_yaml_unique(
      (
          REPO_ROOT
          / "configs/market_genome/pattern_ontology_v1_review.yaml"
      ).read_text(encoding="utf-8")
  )
  return payload, adjudication, reviewer_a, reviewer_b, v2, ledger


def test_import_is_lightweight_and_gpu_neutral() -> None:
  script = """
import importlib
import json
import sys
module = importlib.import_module(
    "alphatrade.market_genome.ontology.simulated_review"
)
blocked = sorted(
    name for name in sys.modules
    if name == "jax"
    or name.startswith("jax.")
    or name == "haiku"
    or name.startswith("haiku.")
    or name in {"alphatrade.core.model", "alphatrade.training"}
)
print(json.dumps({"profile": module.PROFILE_NAME, "blocked": blocked}))
"""
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [sys.executable, "-c", script],
      cwd=REPO_ROOT,
      env=env,
      check=True,
      capture_output=True,
      text=True,
  )
  assert json.loads(completed.stdout) == {
      "profile": "mg1a-simulated-dual-review",
      "blocked": [],
  }


def test_bundle_schema_semantics_and_hash_chain_pass() -> None:
  payload = _bundle()
  assert simulated_review.json_schema_errors(
      payload, repo_root=REPO_ROOT
  ) == []
  assert simulated_review.validate_bundle_payload(
      payload, repo_root=REPO_ROOT
  ) == []
  loaded, errors = simulated_review.validate_bundle(
      BUNDLE, repo_root=REPO_ROOT
  )
  assert errors == []
  assert loaded == payload


def test_exact_coverage_counts_and_non_human_boundary() -> None:
  payload = _bundle()
  adjudication = _json("adjudication.json")
  decisions = adjudication["decisions"]
  assert len(decisions) == 70
  assert len({item["assumption_id"] for item in decisions}) == 70
  assert payload["summary"]["adjudicated_counts"] == {
      "ACCEPT": 13,
      "REVISE": 53,
      "DEFER": 4,
      "REJECT": 0,
  }
  assert payload["summary"]["a_b_agreement_count"] == 42
  assert payload["summary"]["a_b_disagreement_count"] == 28
  assert all(item["human_review_required"] is True for item in decisions)
  assert all(item["freeze_eligible"] is False for item in decisions)
  reviewer_a = _json("reviewer_a.raw.json")
  assert reviewer_a["summary"]["freeze_ready"] == (
      reviewer_a["summary"]["counts"]["ACCEPT"]
  )
  assert payload["governance"][
      "reviewer_a_freeze_ready_is_technical_only"
  ] is True
  assert payload["governance"] == {
      "actual_human_reviewer_count": 0,
      "simulated_reviewer_count": 2,
      "simulated_adjudicator_count": 1,
      "model_assisted_independent_truth_allowed": False,
      "reviewer_a_freeze_ready_is_technical_only": True,
      "human_review_required": True,
      "human_freeze_allowed": False,
      "v2_human_review_fields_modified": False,
      "mg1b_allowed": False,
  }


def test_adjudicator_can_override_reviewer_consensus() -> None:
  decisions = {
      item["assumption_id"]: item
      for item in _json("adjudication.json")["decisions"]
  }
  range_phase = decisions["MG1A.RANGE.PHASE.001"]
  assert range_phase["reviewer_a_disposition"] == "ACCEPT"
  assert range_phase["reviewer_b_disposition"] == "ACCEPT"
  assert range_phase["adjudicated_disposition"] == "REVISE"
  assert range_phase["adjudicator_alignment"] == "NEITHER"
  assert "RANGE_TERMINATION_RECORDED" in range_phase["rationale"]


def test_bundle_and_archive_tampering_fail_closed() -> None:
  payload = _bundle()
  payload["artifacts"]["reviewer_a"]["sha256"] = "0" * 64
  errors = simulated_review.validate_bundle_payload(
      payload, repo_root=REPO_ROOT
  )
  assert any("artifacts/reviewer_a/sha256" in error for error in errors)

  payload = _bundle()
  payload["decisions_sha256"] = "0" * 64
  errors = simulated_review.validate_bundle_payload(
      payload, repo_root=REPO_ROOT
  )
  assert any("decisions_sha256" in error for error in errors)

  payload = _bundle()
  payload["summary"]["adjudicated_counts"]["ACCEPT"] = 14
  errors = simulated_review.validate_bundle_payload(
      payload, repo_root=REPO_ROOT
  )
  assert any("summary: does not match" in error for error in errors)


def test_false_human_freeze_fails_schema_and_record_validation() -> None:
  payload = _bundle()
  payload["governance"]["human_freeze_allowed"] = True
  errors = simulated_review.json_schema_errors(
      payload, repo_root=REPO_ROOT
  )
  assert any("human_freeze_allowed" in error for error in errors)

  (
      payload,
      adjudication,
      reviewer_a,
      reviewer_b,
      v2,
      ledger,
  ) = _decision_materials()
  adjudication["decisions"][0]["freeze_eligible"] = True
  errors = []
  simulated_review._validate_adjudication_records(
      adjudication,
      reviewer_a,
      reviewer_b,
      v2,
      ledger,
      payload,
      errors,
  )
  assert any("freeze_eligible: false required" in error for error in errors)


def test_nested_human_authority_claims_fail_closed() -> None:
  payload = _bundle()
  payload["governance"]["human_freeze_approved"] = True
  semantic_errors = simulated_review.validate_bundle_payload(
      payload, repo_root=REPO_ROOT
  )
  assert any("bundle/governance" in error for error in semantic_errors)

  reviewer_a = _json("reviewer_a.raw.json")
  reviewer_b = _json("reviewer_b.raw.json")
  reviewer_a["reviewer"]["human_reviewer"] = True
  reviewer_a["summary"]["overall_decision"] = "HUMAN_FREEZE_APPROVED"
  reviewer_b["summary"]["mg1b_recommendation"] = "UNBLOCKED"
  errors: list[str] = []
  simulated_review._validate_reviewer_metadata(
      reviewer_a, reviewer_b, _bundle()["target"], errors
  )
  simulated_review._validate_reviewer_summaries(
      reviewer_a,
      reviewer_b,
      reviewer_a["decisions"],
      reviewer_b["decisions"],
      errors,
  )
  assert any("reviewer_a/reviewer" in error for error in errors)
  assert any("reviewer_a/summary/overall_decision" in error for error in errors)
  assert any("reviewer_b/summary/mg1b_recommendation" in error for error in errors)

  adjudication = _json("adjudication.json")
  adjudication["adjudicator"]["authority"] = "HUMAN_FREEZE_AUTHORITY"
  adjudication["human_review_and_freeze_limitations"][
      "human_freeze_approved"
  ] = True
  adjudication["human_review_and_freeze_limitations"]["mg1b_allowed"] = True
  errors = []
  simulated_review._validate_adjudication_metadata(
      adjudication, _bundle(), errors
  )
  assert any("adjudication/adjudicator" in error for error in errors)
  assert any("human-freeze limitations" in error for error in errors)


def test_reviewer_and_v1_blocker_mismatch_fail() -> None:
  (
      payload,
      adjudication,
      reviewer_a,
      reviewer_b,
      v2,
      ledger,
  ) = _decision_materials()
  adjudication["decisions"][0]["reviewer_a_disposition"] = "DEFER"
  adjudication["decisions"][0]["v1_blocker_assessments"].pop()
  errors: list[str] = []
  simulated_review._validate_adjudication_records(
      adjudication,
      reviewer_a,
      reviewer_b,
      v2,
      ledger,
      payload,
      errors,
  )
  assert any("reviewer_a_disposition: raw input mismatch" in error for error in errors)
  assert any("exact coverage required" in error for error in errors)


def test_b_attestation_detachment_fails() -> None:
  payload = _bundle()
  reviewer_b = _json("reviewer_b.raw.json")
  attestation = _json("reviewer_b.attestation.json")
  attestation["supplement_of"]["sha256"] = "0" * 64
  errors: list[str] = []
  simulated_review._validate_attestation(
      attestation, reviewer_b, payload, errors
  )
  assert any("raw B hash mismatch" in error for error in errors)


def test_duplicate_json_key_is_rejected() -> None:
  with pytest.raises(
      simulated_review.SimulatedReviewValidationError,
      match="duplicate JSON key",
  ):
    simulated_review.loads_json_unique(
        '{"schema_version":"one","schema_version":"two"}'
    )


def test_malformed_bound_data_fails_closed_without_traceback() -> None:
  payload = _bundle()
  payload["artifacts"] = []
  errors = simulated_review.validate_bundle_payload(
      payload, repo_root=REPO_ROOT
  )
  assert len(errors) == 1
  assert errors[0].startswith("semantic validation failed closed")


def test_bundle_snapshot_change_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
  original = simulated_review._sha256_file

  def changed_hash(path: Path) -> str:
    if path.resolve() == BUNDLE.resolve():
      return "0" * 64
    return original(path)

  monkeypatch.setattr(simulated_review, "_sha256_file", changed_hash)
  _, errors = simulated_review.validate_bundle(BUNDLE, repo_root=REPO_ROOT)
  assert any("bundle: input changed during validation" in error for error in errors)


def test_cli_strict_validation_passes() -> None:
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.simulated_review",
          "--bundle",
          str(BUNDLE),
          "--strict",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=True,
      capture_output=True,
      text=True,
  )
  assert "assumptions=70" in completed.stdout
  assert "'ACCEPT': 13" in completed.stdout
  assert "errors=0" in completed.stdout
