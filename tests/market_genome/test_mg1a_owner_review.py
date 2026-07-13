"""CPU-only tests for the MG1-A.1 single-owner review workflow."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import pytest

from alphatrade.market_genome.ontology import owner_review
from alphatrade.market_genome.ontology import revision
from alphatrade.market_genome.ontology import simulated_review


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / owner_review.LEDGER_PATH
WORKBOOK = REPO_ROOT / owner_review.WORKBOOK_PATH
REVIEWED_AT = "2026-07-13T20:00:00+08:00"


def _ledger() -> dict:
  return owner_review.load_ledger(LEDGER)


def _pending_ledger() -> dict:
  return owner_review.build_pending_ledger(REPO_ROOT)


def _advisory() -> list[dict]:
  path = REPO_ROOT / owner_review.AI_REFERENCE["adjudication_path"]
  payload = simulated_review.loads_json_unique(
      path.read_text(encoding="utf-8")
  )
  return payload["decisions"]


def _refresh(payload: dict) -> dict:
  return owner_review.refresh_derived_fields(payload)


def _approve_all(payload: dict) -> None:
  advisory_by_id = {
      item["assumption_id"]: item["adjudicated_disposition"]
      for item in _advisory()
  }
  for record in payload["records"]:
    record["owner_decision"] = "APPROVE_AS_IS"
    record["owner_rationale"] = None
    record["required_changes"] = []
    record["evidence_needed_to_resume"] = []
    record["replacement_proposition"] = []
    record["reviewed_at"] = REVIEWED_AT
    if advisory_by_id[record["assumption_id"]] != "ACCEPT":
      record["owner_rationale"] = (
          "Project owner explicitly accepts the current hash-bound text."
      )


def _sign(payload: dict, decision: str) -> None:
  _refresh(payload)
  payload["owner_signoff"] = {
      "decision": decision,
      "signed_at": REVIEWED_AT,
      "target_resolved_contract_sha256": owner_review.TARGET[
          "resolved_contract_sha256"
      ],
      "review_records_sha256": payload["records_sha256"],
      "attestation": owner_review.SIGNOFF_ATTESTATIONS[decision],
  }
  _refresh(payload)


def test_import_is_lightweight_and_gpu_neutral() -> None:
  script = """
import importlib
import json
import sys
module = importlib.import_module(
    "alphatrade.market_genome.ontology.owner_review"
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
      "profile": "mg1a-v2-owner-review",
      "blocked": [],
  }


def test_live_completed_ledger_schema_semantics_and_exact_coverage_pass() -> None:
  payload = _ledger()
  assert owner_review.json_schema_errors(payload, repo_root=REPO_ROOT) == []
  loaded, errors = owner_review.validate_ledger(
      LEDGER, repo_root=REPO_ROOT
  )
  assert errors == []
  assert loaded == payload
  expected_ids = [item["assumption_id"] for item in _advisory()]
  assert [item["assumption_id"] for item in payload["records"]] == expected_ids
  assert len(expected_ids) == len(set(expected_ids)) == 70
  assert payload["summary"] == {
      "assumptions_total": 70,
      "pending": 0,
      "approved_as_is": 8,
      "request_revision": 61,
      "deferred": 1,
      "rejected": 0,
      "missing_contract_approval_conflicts": 0,
      "review_complete": True,
      "owner_signoff_complete": False,
      "ready_for_mg1b_freeze": False,
  }
  assert payload["status"] == "AWAITING_OWNER_SIGNOFF"


def test_single_owner_policy_records_gold_annotation_change() -> None:
  payload = _ledger()
  assert payload["owner"] == owner_review.OWNER
  assert payload["owner"]["appointed_owner_count"] == 1
  assert payload["owner"]["second_review_required"] is False
  assert payload["owner"]["adjudication_required"] is False
  assert payload["policy"][
      "downstream_gold_annotation_policy_changed"
  ] is True

  v2 = revision.load_revision(REPO_ROOT / owner_review.TARGET["path"])
  annotation_policy = v2["resolved_contract"]["review_protocol"]
  assert annotation_policy["minimum_independent_reviewers"] == "TWO"
  assert annotation_policy["independent_review_policy"] == (
      "TWO_REVIEWERS_THEN_ADJUDICATE_DISAGREEMENT"
  )


def test_target_and_ai_archive_remain_exactly_hash_bound() -> None:
  for relative, expected in (
      (owner_review.TARGET["path"], owner_review.TARGET["file_sha256"]),
      (
          owner_review.AI_REFERENCE["bundle_path"],
          owner_review.AI_REFERENCE["bundle_sha256"],
      ),
      (
          owner_review.AI_REFERENCE["adjudication_path"],
          owner_review.AI_REFERENCE["adjudication_sha256"],
      ),
  ):
    observed = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
    assert observed == expected


def test_workbook_is_grouped_complete_and_matches_renderer() -> None:
  rendered = owner_review.render_workbook(_ledger(), repo_root=REPO_ROOT)
  assert set(rendered) == {
      "README.md",
      "01_common_contracts.md",
      "02_trend_range.md",
      "03_reversal_morphologies.md",
      "04_break_retest_conversion.md",
      "05_final_integration.md",
  }
  for name, content in rendered.items():
    assert (WORKBOOK / name).read_text(encoding="utf-8") == content
  combined = "\n".join(
      content for name, content in rendered.items() if name != "README.md"
  )
  headings = re.findall(
      r"^## \d{2}\. `([^`]+)`$", combined, flags=re.MULTILINE
  )
  assert len(headings) == len(set(headings)) == 70
  assert "AI 参考意见（无审批权）" in combined
  assert "PROJECT_OWNER" in rendered["README.md"]
  assert rendered["README.md"].count("| [`MG1A") == 70
  assert "13" in rendered["README.md"]
  assert "53" in rendered["README.md"]
  assert "不要直接编辑" in rendered["README.md"]
  assert "REFERENCE_ONLY" not in combined

  questions_path = REPO_ROOT / owner_review.QUESTIONS_ZH_PATH
  assert hashlib.sha256(questions_path.read_bytes()).hexdigest() == (
      owner_review.QUESTIONS_ZH_SHA256
  )
  questions = revision.loads_json_unique(
      questions_path.read_text(encoding="utf-8")
  )
  assert set(questions) == set(headings)
  assert all(
      any("\u4e00" <= char <= "\u9fff" for char in value)
      for value in questions.values()
  )


def test_missing_roll_and_zone_contracts_are_visible() -> None:
  common = (WORKBOOK / "01_common_contracts.md").read_text(
      encoding="utf-8"
  )
  assert common.count("**当前定义缺失。**") == 2
  assert "resolved_contract.roll_contract" in common
  assert "resolved_contract.zone_contract" in common


def test_non_accept_ai_override_requires_owner_rationale() -> None:
  payload = _pending_ledger()
  record = payload["records"][0]
  assert _advisory()[0]["adjudicated_disposition"] != "ACCEPT"
  record["owner_decision"] = "APPROVE_AS_IS"
  record["reviewed_at"] = REVIEWED_AT
  _refresh(payload)
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert any("overriding non-ACCEPT AI advice" in error for error in errors)

  record["owner_rationale"] = "Owner accepts the current contract explicitly."
  _refresh(payload)
  assert owner_review.validate_ledger_payload(
      payload, repo_root=REPO_ROOT
  ) == []


@pytest.mark.parametrize(
    ("decision", "field", "expected_error"),
    [
        (
            "REQUEST_REVISION",
            "required_changes",
            "revision requires rationale and required_changes",
        ),
        (
            "DEFER",
            "evidence_needed_to_resume",
            "defer requires rationale and evidence_needed_to_resume",
        ),
        (
            "REJECT",
            "replacement_proposition",
            "rejection requires rationale and replacement_proposition",
        ),
    ],
)
def test_non_approval_decisions_require_actionable_input(
    decision: str, field: str, expected_error: str
) -> None:
  payload = _pending_ledger()
  record = payload["records"][0]
  record["owner_decision"] = decision
  record["owner_rationale"] = "Owner action is required."
  record["reviewed_at"] = REVIEWED_AT
  _refresh(payload)
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert any(expected_error in error for error in errors)

  record[field] = ["Concrete owner instruction or required evidence."]
  _refresh(payload)
  assert owner_review.validate_ledger_payload(
      payload, repo_root=REPO_ROOT
  ) == []


def test_whitespace_only_action_is_rejected() -> None:
  payload = _pending_ledger()
  record = payload["records"][0]
  record["owner_decision"] = "REQUEST_REVISION"
  record["owner_rationale"] = "A concrete change is required."
  record["required_changes"] = ["   "]
  record["reviewed_at"] = REVIEWED_AT
  _refresh(payload)
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert any("nonblank strings required" in error for error in errors)


@pytest.mark.parametrize(
    "timestamp",
    [
        "9999-99-99T99:99:99Z",
        "2026-07-13T20:00:00+08:60",
        "2026-07-13T18:00:00+08:00",
    ],
)
def test_invalid_or_pre_target_review_timestamp_is_rejected(
    timestamp: str,
) -> None:
  payload = _pending_ledger()
  accept_index = next(
      index
      for index, item in enumerate(_advisory())
      if item["adjudicated_disposition"] == "ACCEPT"
  )
  record = payload["records"][accept_index]
  record["owner_decision"] = "APPROVE_AS_IS"
  record["reviewed_at"] = timestamp
  _refresh(payload)
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert any("reviewed_at" in error for error in errors)


def test_record_order_hash_summary_and_status_tampering_fail() -> None:
  payload = _pending_ledger()
  payload["records"][0], payload["records"][1] = (
      payload["records"][1],
      payload["records"][0],
  )
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert any("exact 70-item AI advisory order" in error for error in errors)
  assert any("records_sha256" in error for error in errors)

  payload = _pending_ledger()
  payload["summary"]["pending"] = 69
  payload["status"] = "APPROVED_FOR_MG1B_FREEZE"
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert any("summary: derived values" in error for error in errors)
  assert any("status: derived status" in error for error in errors)


def test_false_second_reviewer_or_adjudicator_fails_closed_schema() -> None:
  payload = _pending_ledger()
  payload["owner"]["second_review_required"] = True
  payload["owner"]["adjudication_required"] = True
  errors = owner_review.json_schema_errors(payload, repo_root=REPO_ROOT)
  assert any("second_review_required" in error for error in errors)
  assert any("adjudication_required" in error for error in errors)

  payload = _pending_ledger()
  payload["owner"]["second_reviewer_id"] = "SOMEONE"
  errors = owner_review.json_schema_errors(payload, repo_root=REPO_ROOT)
  assert any("Additional properties are not allowed" in error for error in errors)


def test_signoff_cannot_approve_incomplete_review() -> None:
  payload = _pending_ledger()
  payload["owner_signoff"] = {
      "decision": "APPROVE_FOR_MG1B_FREEZE",
      "signed_at": REVIEWED_AT,
      "target_resolved_contract_sha256": owner_review.TARGET[
          "resolved_contract_sha256"
      ],
      "review_records_sha256": payload["records_sha256"],
      "attestation": owner_review.OWNER_SIGNOFF_ATTESTATION,
  }
  _refresh(payload)
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert any("all 70 owner decisions required" in error for error in errors)
  assert any("approval requires 70 APPROVE_AS_IS" in error for error in errors)


def test_missing_contracts_cannot_be_approved_as_is() -> None:
  payload = _pending_ledger()
  _approve_all(payload)
  _sign(payload, "APPROVE_FOR_MG1B_FREEZE")
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert sum("v2 contract is missing" in error for error in errors) == 2
  assert payload["status"] == "AWAITING_OWNER_SIGNOFF"
  assert payload["summary"]["missing_contract_approval_conflicts"] == 2
  assert payload["summary"]["ready_for_mg1b_freeze"] is False
  assert "frozen" not in payload
  assert "mg1b_allowed" not in payload


def test_signoff_rejects_false_freeze_or_unblock_claim() -> None:
  payload = _pending_ledger()
  _approve_all(payload)
  _sign(payload, "APPROVE_FOR_MG1B_FREEZE")
  payload["owner_signoff"]["attestation"] = (
      "I certify that the ontology is already frozen and MG1-B is unblocked."
  )
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert any("attestation" in error for error in errors)


def test_completed_review_with_owner_action_returns_for_revision() -> None:
  payload = _pending_ledger()
  _approve_all(payload)
  first = payload["records"][0]
  first["owner_decision"] = "REQUEST_REVISION"
  first["owner_rationale"] = "A contract edit is required before freeze."
  first["required_changes"] = ["Define the missing event predicate."]
  for record in payload["records"]:
    if record["assumption_id"] in owner_review.MISSING_CONTRACT_IDS:
      record["owner_decision"] = "REQUEST_REVISION"
      record["owner_rationale"] = "The current v2 contract is missing."
      record["required_changes"] = ["Add the missing contract definition."]
  _sign(payload, "RETURN_FOR_REVISION")
  assert owner_review.validate_ledger_payload(
      payload, repo_root=REPO_ROOT
  ) == []
  assert payload["status"] == "OWNER_REVIEW_COMPLETED_WITH_ACTIONS"
  assert payload["summary"]["ready_for_mg1b_freeze"] is False


def test_signoff_cannot_precede_record_review_times() -> None:
  payload = _pending_ledger()
  _approve_all(payload)
  for record in payload["records"]:
    if record["assumption_id"] in owner_review.MISSING_CONTRACT_IDS:
      record["owner_decision"] = "REQUEST_REVISION"
      record["owner_rationale"] = "The current v2 contract is missing."
      record["required_changes"] = ["Add the missing contract definition."]
  _sign(payload, "RETURN_FOR_REVISION")
  payload["owner_signoff"]["signed_at"] = "2026-07-13T19:00:00+08:00"
  errors = owner_review.validate_ledger_payload(payload, repo_root=REPO_ROOT)
  assert any("cannot precede record review times" in error for error in errors)


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
  duplicate = tmp_path / "duplicate.yaml"
  duplicate.write_text(
      "schema_version: one\nschema_version: two\n", encoding="utf-8"
  )
  with pytest.raises(
      owner_review.OwnerReviewValidationError, match="duplicate YAML key"
  ):
    owner_review.load_ledger(duplicate)


def test_unquoted_yaml_timestamps_are_normalized_to_strings(
    tmp_path: Path,
) -> None:
  content = owner_review._dump_ledger(_pending_ledger()).replace(
      "reviewed_at: null",
      "reviewed_at: 2026-07-13T20:00:00+08:00",
      1,
  )
  ledger = tmp_path / "review.yaml"
  ledger.write_text(content, encoding="ascii")
  payload = owner_review.load_ledger(ledger)
  assert payload["records"][0]["reviewed_at"] == REVIEWED_AT


def test_strict_cli_passes_without_gpu_visibility() -> None:
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.owner_review",
          "--strict",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=True,
      capture_output=True,
      text=True,
  )
  assert "status=AWAITING_OWNER_SIGNOFF" in completed.stdout
  assert "reviewed=70/70" in completed.stdout
  assert "ready_for_mg1b_freeze=False" in completed.stdout
  assert "errors=0" in completed.stdout


def test_refresh_rejects_invalid_edits_without_overwriting(
    tmp_path: Path,
) -> None:
  payload = _pending_ledger()
  payload["records"][0]["owner_decision"] = "REQUEST_REVISION"
  ledger = tmp_path / "review.yaml"
  ledger.write_text(owner_review._dump_ledger(payload), encoding="ascii")
  before = ledger.read_bytes()
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.owner_review",
          "--ledger",
          str(ledger),
          "--refresh",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert completed.returncode == 1
  assert "revision requires rationale" in completed.stdout
  assert ledger.read_bytes() == before


def test_stale_lock_recovery_requires_dead_recorded_process(
    tmp_path: Path,
) -> None:
  ledger = tmp_path / "review.yaml"
  lock = tmp_path / ".review.yaml.lock"
  lock.write_text("pid=999999999\n", encoding="ascii")
  owner_review._recover_stale_lock(ledger)
  assert not lock.exists()

  lock.write_text(f"pid={os.getpid()}\n", encoding="ascii")
  with pytest.raises(
      owner_review.OwnerReviewValidationError,
      match="process is still active",
  ):
    owner_review._recover_stale_lock(ledger)
  assert lock.exists()


def test_stale_recovery_cannot_remove_a_replacement_active_lock(
    tmp_path: Path,
) -> None:
  ledger = tmp_path / "review.yaml"
  lock = tmp_path / ".review.yaml.lock"
  lock.write_text("pid=999999999\n", encoding="ascii")
  recovery_errors: list[Exception] = []

  def recover() -> None:
    try:
      owner_review._recover_stale_lock(ledger)
    except Exception as exc:  # pragma: no branch - expected active-lock error
      recovery_errors.append(exc)

  with owner_review._lock_guard(ledger):
    worker = threading.Thread(target=recover)
    worker.start()
    time.sleep(0.05)
    assert worker.is_alive()
    lock.write_text(f"pid={os.getpid()}\n", encoding="ascii")
  worker.join(timeout=2)

  assert not worker.is_alive()
  assert len(recovery_errors) == 1
  assert "process is still active" in str(recovery_errors[0])
  assert lock.read_text(encoding="ascii") == f"pid={os.getpid()}\n"


def test_lock_initialization_failure_cleans_owned_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  ledger = tmp_path / "review.yaml"
  lock = tmp_path / ".review.yaml.lock"

  def fail_fsync(_descriptor: int) -> None:
    raise OSError("injected fsync failure")

  monkeypatch.setattr(owner_review.os, "fsync", fail_fsync)
  with pytest.raises(
      owner_review.OwnerReviewValidationError,
      match="lock initialization failed",
  ):
    with owner_review._exclusive_update_lock(ledger):
      pytest.fail("lock body must not run after initialization failure")
  assert not lock.exists()


def test_partial_lock_write_failure_cleans_owned_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  ledger = tmp_path / "review.yaml"
  lock = tmp_path / ".review.yaml.lock"
  real_write = owner_review.os.write
  write_calls = 0

  def partial_then_fail(descriptor: int, content: bytes) -> int:
    nonlocal write_calls
    write_calls += 1
    if write_calls == 1:
      return real_write(descriptor, content[:2])
    raise OSError("injected write failure")

  monkeypatch.setattr(owner_review.os, "write", partial_then_fail)
  with pytest.raises(
      owner_review.OwnerReviewValidationError,
      match="lock initialization failed",
  ):
    with owner_review._exclusive_update_lock(ledger):
      pytest.fail("lock body must not run after initialization failure")
  assert write_calls == 2
  assert not lock.exists()


def test_update_lock_cleanup_preserves_replacement_lock(tmp_path: Path) -> None:
  ledger = tmp_path / "review.yaml"
  lock = tmp_path / ".review.yaml.lock"
  expected_content = f"pid={os.getpid()}\n"

  with owner_review._exclusive_update_lock(ledger):
    assert lock.read_text(encoding="ascii") == expected_content
    lock.unlink()
    lock.write_text(expected_content, encoding="ascii")

  assert lock.read_text(encoding="ascii") == expected_content
