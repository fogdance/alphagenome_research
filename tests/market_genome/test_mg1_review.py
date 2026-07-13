"""CPU-only tests for the MG1 expert assumption review ledger."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from alphatrade.market_genome.ontology import review
from alphatrade.market_genome.ontology import schema as ontology_schema


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = (
    REPO_ROOT / "configs/market_genome/pattern_ontology_v1_review.yaml"
)
COMMON_FINDINGS = (
    REPO_ROOT / "configs/market_genome/reviews/mg1_common_review_v1.yaml"
)


def _ledger() -> dict:
  return review.load_review_ledger(LEDGER)


def _direct_cli_args(ledger: Path, **overrides: object) -> SimpleNamespace:
  values = {
      "ledger": ledger,
      "initialize": False,
      "apply_findings": None,
      "base_commit": None,
      "output_json": None,
      "output_md": None,
      "strict": True,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def test_review_import_is_lightweight_and_gpu_neutral() -> None:
  script = """
import importlib
import json
import sys
module = importlib.import_module(
    "alphatrade.market_genome.ontology.review"
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
  completed = subprocess.run(
      [sys.executable, "-c", script],
      cwd=REPO_ROOT,
      env=env,
      check=True,
      capture_output=True,
      text=True,
  )
  assert json.loads(completed.stdout) == {
      "profile": "mg1-review",
      "blocked": [],
  }


def test_pending_ledger_is_exact_and_valid() -> None:
  payload = review.build_pending_ledger(
      repo_root=REPO_ROOT,
      base_commit="ab6a6f79a81c7cb5fc183dfb49953425846f4aa1",
  )
  assert review.json_schema_errors(payload, repo_root=REPO_ROOT) == []
  assert review.validate_review_ledger(payload, repo_root=REPO_ROOT) == []
  assert payload["summary"] == {
      "assumptions_total": 61,
      "reviewed": 0,
      "pending": 61,
      "accepted": 0,
      "revised": 0,
      "rejected": 0,
      "deferred": 0,
      "freeze_ready_records": 0,
      "freeze_blocked_records": 61,
      "resolved_parameter_refs": 0,
      "unresolved_parameter_refs": 114,
      "wildcard_expansions": 0,
      "freeze_ready": False,
  }


def test_committed_ledger_records_complete_first_review() -> None:
  payload = _ledger()
  assert review.json_schema_errors(payload, repo_root=REPO_ROOT) == []
  assert review.validate_review_ledger(payload, repo_root=REPO_ROOT) == []
  assert payload["decision"] == (
      "DOMAIN_REVIEW_COMPLETED_WITH_REQUIRED_REVISIONS"
  )
  assert payload["draft_assessment"] == "APPROVED_AS_DRAFT"
  assert payload["freeze_assessment"] == (
      "REVISION_REQUIRED_BEFORE_FREEZE"
  )
  assert payload["mg1b_status"] == "BLOCKED"
  assert payload["next_milestone"] == "MG1-A.1"
  assert payload["summary"] == {
      "assumptions_total": 61,
      "reviewed": 61,
      "pending": 0,
      "accepted": 0,
      "revised": 51,
      "rejected": 0,
      "deferred": 10,
      "freeze_ready_records": 0,
      "freeze_blocked_records": 61,
      "resolved_parameter_refs": 0,
      "unresolved_parameter_refs": 114,
      "wildcard_expansions": 0,
      "freeze_ready": False,
  }
  assert all(record["review_state"] == "REVIEWED" for record in payload["records"])
  assert all(record["freeze_gate"] == "BLOCKED" for record in payload["records"])
  assert all(record["expert_decision"] for record in payload["records"])
  assert all(record["approved_invariants"] for record in payload["records"])
  assert all(record["blocking_items"] for record in payload["records"])


def test_findings_overlay_applies_once_and_fails_closed() -> None:
  payload = review.build_pending_ledger(
      repo_root=REPO_ROOT,
      base_commit="ab6a6f79a81c7cb5fc183dfb49953425846f4aa1",
  )
  findings = ontology_schema.loads_yaml_unique(
      COMMON_FINDINGS.read_text(encoding="utf-8")
  )
  result = review.apply_review_findings(payload, findings)
  assert result["summary"]["reviewed"] == 13
  assert result["summary"]["pending"] == 48
  assert review.validate_review_ledger(result, repo_root=REPO_ROOT) == []

  with pytest.raises(ValueError, match="already reviewed"):
    review.apply_review_findings(result, findings)

  wrong_hash = copy.deepcopy(findings)
  wrong_hash["base_contract_sha256"] = "0" * 64
  pending = review.build_pending_ledger(
      repo_root=REPO_ROOT,
      base_commit="ab6a6f79a81c7cb5fc183dfb49953425846f4aa1",
  )
  with pytest.raises(ValueError, match="different draft contract"):
    review.apply_review_findings(pending, wrong_hash)


def test_draft_metadata_and_hash_tampering_fail() -> None:
  payload = _ledger()
  payload["base_draft"]["file_sha256"] = "0" * 64
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("current draft bytes do not match" in error for error in errors)

  payload = _ledger()
  payload["records"][0]["question"] = "Different question"
  review.refresh_derived_fields(payload)
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("draft metadata must match exactly" in error for error in errors)

  payload = _ledger()
  payload["records"].reverse()
  review.refresh_derived_fields(payload)
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("exact draft assumption order" in error for error in errors)

  payload = _ledger()
  payload["base_draft"]["git_commit"] = "0" * 40
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("unavailable in Git" in error for error in errors)


def test_derived_hash_summary_and_decision_tampering_fail() -> None:
  payload = _ledger()
  payload["records_sha256"] = "0" * 64
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("records_sha256" in error for error in errors)

  payload = _ledger()
  payload["summary"]["reviewed"] = 60
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("summary: does not match" in error for error in errors)

  payload = _ledger()
  payload["decision"] = "APPROVED_FOR_FREEZE"
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("decision: expected" in error for error in errors)


def test_parameter_partition_and_wildcard_resolution_fail_closed() -> None:
  payload = _ledger()
  record = payload["records"][0]
  record["resolved_parameter_refs"] = list(record["parameter_refs"])
  record["unresolved_parameter_refs"] = list(record["parameter_refs"])
  review.refresh_derived_fields(payload)
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("must partition parameter_refs" in error for error in errors)

  payload = _ledger()
  record = next(
      item
      for item in payload["records"]
      if item["assumption_id"] == "MG1A.TREND_UP.PHASE.001"
  )
  record["resolved_parameter_refs"] = list(record["parameter_refs"])
  record["unresolved_parameter_refs"] = []
  record["required_changes"] = []
  record["blocking_items"] = []
  record["disposition"] = "ACCEPT"
  record["freeze_gate"] = "READY"
  review.refresh_derived_fields(payload)
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("resolved wildcard requires" in error for error in errors)


@pytest.mark.parametrize(
    "decision, expected",
    [
        ("Use an absolute 5 point threshold.", "absolute numeric threshold"),
        (
            "Confirm using next-day profit above two percent.",
            "future outcome cannot define ontology state",
        ),
        (
            "Confirm phase using next week's return.",
            "future outcome cannot define ontology state",
        ),
        (
            "Classify the state from the following bar return.",
            "future outcome cannot define ontology state",
        ),
        (
            "Set the class using forward return.",
            "future outcome cannot define ontology state",
        ),
        ("Emit a buy entry after confirmation.", "trading action semantics"),
        ("Emit a long entry after confirmation.", "trading action semantics"),
        ("Open a short position after confirmation.", "trading action semantics"),
        (
            "Do not ignore volume, confirm phase using future profit.",
            "future outcome cannot define ontology state",
        ),
        (
            "Never drop volume, classify state from future return.",
            "future outcome cannot define ontology state",
        ),
    ],
)
def test_unsafe_review_narrative_is_rejected(
    decision: str, expected: str
) -> None:
  payload = _ledger()
  payload["records"][0]["expert_decision"] = decision
  review.refresh_derived_fields(payload)
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any(expected in error for error in errors)


def test_prohibition_does_not_mask_unsafe_later_clauses() -> None:
  payload = _ledger()
  payload["records"][0]["expert_decision"] = (
      "Do not ignore volume. Confirm phase using future profit; "
      "emit a buy signal."
  )
  review.refresh_derived_fields(payload)
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("future outcome cannot define ontology state" in error for error in errors)
  assert any("trading action semantics" in error for error in errors)


def test_local_prohibition_of_trade_semantics_remains_valid() -> None:
  payload = _ledger()
  record = next(
      item
      for item in payload["records"]
      if item["assumption_id"] == "MG1A.TREND_TRANSITION.STRUCTURE.001"
  )
  assert (
      "Side describes structural counter direction, not a trade action."
      in record["approved_invariants"]
  )
  assert review.validate_review_ledger(payload, repo_root=REPO_ROOT) == []


def test_direct_prohibition_of_future_semantics_remains_valid() -> None:
  payload = _ledger()
  payload["records"][0]["expert_decision"] = (
      "Do not use future profit to confirm phase."
  )
  review.refresh_derived_fields(payload)
  assert review.validate_review_ledger(payload, repo_root=REPO_ROOT) == []


def test_false_ready_gate_is_rejected() -> None:
  payload = _ledger()
  payload["records"][0]["freeze_gate"] = "READY"
  review.refresh_derived_fields(payload)
  errors = review.validate_review_ledger(payload, repo_root=REPO_ROOT)
  assert any("freeze_gate: inconsistent" in error for error in errors)


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
  duplicate = tmp_path / "duplicate.yaml"
  duplicate.write_text("schema_version: one\nschema_version: two\n", encoding="utf-8")
  with pytest.raises(ontology_schema.OntologyValidationError, match="duplicate key"):
    review.load_review_ledger(duplicate)


def test_cli_strict_validation_and_report_generation(tmp_path: Path) -> None:
  output_json = tmp_path / "review.json"
  output_md = tmp_path / "review.md"
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.review",
          "--ledger",
          str(LEDGER),
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
  assert "reviewed=61/61" in completed.stdout
  assert (
      "decision=DOMAIN_REVIEW_COMPLETED_WITH_REQUIRED_REVISIONS"
      in completed.stdout
  )
  assert json.loads(output_json.read_text(encoding="utf-8"))["summary"][
      "reviewed"
  ] == 61
  markdown = output_md.read_text(encoding="utf-8")
  assert "# MG1 Assumption Review Ledger" in markdown
  assert "MG1A.COMMON.LIFECYCLE.001" in markdown
  assert "MG1A.SUPPORT_RESISTANCE_CONVERSION.INVALIDATION.001" in markdown


def test_cli_invalid_ledger_does_not_write_reports(tmp_path: Path) -> None:
  payload = _ledger()
  payload["records_sha256"] = "0" * 64
  ledger = tmp_path / "invalid.yaml"
  ledger.write_text(
      yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
  )
  output_json = tmp_path / "review.json"
  output_md = tmp_path / "review.md"
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.review",
          "--ledger",
          str(ledger),
          "--output-json",
          str(output_json),
          "--output-md",
          str(output_md),
          "--strict",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert completed.returncode == 1
  assert "schema_semantic=fail" in completed.stdout
  assert not output_json.exists()
  assert not output_md.exists()


def test_cli_invalid_ledger_exits_nonzero_without_strict(tmp_path: Path) -> None:
  payload = _ledger()
  payload["records_sha256"] = "0" * 64
  ledger = tmp_path / "invalid.yaml"
  ledger.write_text(
      yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
  )
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.review",
          "--ledger",
          str(ledger),
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert completed.returncode == 1
  assert "schema_semantic=fail" in completed.stdout


def test_cli_rejects_output_path_collision_without_mutating_ledger(
    tmp_path: Path,
) -> None:
  ledger = tmp_path / "ledger.yaml"
  ledger.write_bytes(LEDGER.read_bytes())
  before = ledger.read_bytes()
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.review",
          "--ledger",
          str(ledger),
          "--output-md",
          str(ledger),
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert completed.returncode == 2
  assert "must not replace validation inputs" in completed.stdout
  assert ledger.read_bytes() == before


@pytest.mark.parametrize(
    "protected_path",
    [
        Path(ontology_schema.__file__),
        ontology_schema.report_schema_path(),
    ],
)
def test_cli_protects_transitive_validation_inputs(
    tmp_path: Path, protected_path: Path
) -> None:
  ledger = tmp_path / "ledger.yaml"
  ledger.write_bytes(LEDGER.read_bytes())
  before = protected_path.read_bytes()
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.review",
          "--ledger",
          str(ledger),
          "--output-json",
          str(protected_path),
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert completed.returncode == 2
  assert "must not replace validation inputs" in completed.stdout
  assert protected_path.read_bytes() == before


def test_cli_invalid_findings_do_not_mutate_ledger(tmp_path: Path) -> None:
  pending = review.build_pending_ledger(
      repo_root=REPO_ROOT,
      base_commit="ab6a6f79a81c7cb5fc183dfb49953425846f4aa1",
  )
  ledger = tmp_path / "pending.yaml"
  ledger.write_text(
      yaml.safe_dump(pending, sort_keys=False), encoding="utf-8"
  )
  before = ledger.read_bytes()
  findings = ontology_schema.loads_yaml_unique(
      COMMON_FINDINGS.read_text(encoding="utf-8")
  )
  findings["findings"][0]["expert_decision"] = 5
  invalid_findings = tmp_path / "invalid-findings.yaml"
  invalid_findings.write_text(
      yaml.safe_dump(findings, sort_keys=False), encoding="utf-8"
  )
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.review",
          "--ledger",
          str(ledger),
          "--apply-findings",
          str(invalid_findings),
          "--strict",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert completed.returncode == 1
  assert ledger.read_bytes() == before


def test_cli_initialize_and_apply_findings_success_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  initialized = tmp_path / "initialized.yaml"
  monkeypatch.setattr(
      review,
      "_parse_args",
      lambda: _direct_cli_args(initialized, initialize=True),
  )
  assert review.main() == 0
  initialized_payload = review.load_review_ledger(initialized)
  assert initialized_payload["summary"]["pending"] == 61
  capsys.readouterr()

  findings = tmp_path / "findings.yaml"
  findings.write_bytes(COMMON_FINDINGS.read_bytes())
  monkeypatch.setattr(
      review,
      "_parse_args",
      lambda: _direct_cli_args(
          initialized,
          apply_findings=findings,
      ),
  )
  assert review.main() == 0
  applied_payload = review.load_review_ledger(initialized)
  assert applied_payload["summary"]["reviewed"] == 13
  assert applied_payload["summary"]["pending"] == 48


def test_cli_ledger_snapshot_change_fails_without_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  ledger = tmp_path / "ledger.yaml"
  ledger.write_bytes(LEDGER.read_bytes())
  output_json = tmp_path / "must_not_exist.json"
  output_md = tmp_path / "must_not_exist.md"
  concurrent = ledger.read_bytes() + b"# concurrent change\n"
  original_validate = review.validate_review_ledger

  def validate_then_change(payload: object, **kwargs: object) -> list[str]:
    errors = original_validate(payload, **kwargs)
    ledger.write_bytes(concurrent)
    return errors

  monkeypatch.setattr(review, "validate_review_ledger", validate_then_change)
  monkeypatch.setattr(
      review,
      "_parse_args",
      lambda: _direct_cli_args(
          ledger,
          output_json=output_json,
          output_md=output_md,
      ),
  )
  assert review.main() == 2
  assert "review ledger input changed during validation" in capsys.readouterr().out
  assert ledger.read_bytes() == concurrent
  assert not output_json.exists()
  assert not output_md.exists()


def test_cli_apply_findings_change_fails_without_mutating_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  pending = review.build_pending_ledger(
      repo_root=REPO_ROOT,
      base_commit="ab6a6f79a81c7cb5fc183dfb49953425846f4aa1",
  )
  ledger = tmp_path / "pending.yaml"
  ledger.write_text(
      yaml.safe_dump(pending, sort_keys=False), encoding="utf-8"
  )
  ledger_before = ledger.read_bytes()
  findings = tmp_path / "findings.yaml"
  findings.write_bytes(COMMON_FINDINGS.read_bytes())
  findings_concurrent = findings.read_bytes() + b"# concurrent change\n"
  original_apply = review.apply_review_findings

  def apply_then_change(payload: dict, overlay: object) -> dict:
    result = original_apply(payload, overlay)
    findings.write_bytes(findings_concurrent)
    return result

  monkeypatch.setattr(review, "apply_review_findings", apply_then_change)
  monkeypatch.setattr(
      review,
      "_parse_args",
      lambda: _direct_cli_args(ledger, apply_findings=findings),
  )
  assert review.main() == 2
  assert "review findings input changed during validation" in capsys.readouterr().out
  assert ledger.read_bytes() == ledger_before
  assert findings.read_bytes() == findings_concurrent


def test_cli_apply_does_not_overwrite_concurrently_changed_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  pending = review.build_pending_ledger(
      repo_root=REPO_ROOT,
      base_commit="ab6a6f79a81c7cb5fc183dfb49953425846f4aa1",
  )
  ledger = tmp_path / "pending.yaml"
  ledger.write_text(
      yaml.safe_dump(pending, sort_keys=False), encoding="utf-8"
  )
  concurrent = ledger.read_bytes() + b"# concurrent change\n"
  findings = tmp_path / "findings.yaml"
  findings.write_bytes(COMMON_FINDINGS.read_bytes())
  original_validate = review.validate_review_ledger

  def validate_then_change(payload: object, **kwargs: object) -> list[str]:
    errors = original_validate(payload, **kwargs)
    ledger.write_bytes(concurrent)
    return errors

  monkeypatch.setattr(review, "validate_review_ledger", validate_then_change)
  monkeypatch.setattr(
      review,
      "_parse_args",
      lambda: _direct_cli_args(ledger, apply_findings=findings),
  )
  assert review.main() == 2
  assert "review ledger input changed during validation" in capsys.readouterr().out
  assert ledger.read_bytes() == concurrent


def test_cli_initialize_does_not_clobber_concurrently_created_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  ledger = tmp_path / "new-ledger.yaml"
  concurrent = b"owner: concurrent writer\n"
  original_validate = review.validate_review_ledger

  def validate_then_create(payload: object, **kwargs: object) -> list[str]:
    errors = original_validate(payload, **kwargs)
    ledger.write_bytes(concurrent)
    return errors

  monkeypatch.setattr(review, "validate_review_ledger", validate_then_create)
  monkeypatch.setattr(
      review,
      "_parse_args",
      lambda: _direct_cli_args(ledger, initialize=True),
  )
  assert review.main() == 2
  assert "ledger appeared during initialization" in capsys.readouterr().out
  assert ledger.read_bytes() == concurrent
