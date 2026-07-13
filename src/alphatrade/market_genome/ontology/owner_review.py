"""CPU-only owner review workflow for the MG1-A.1 ontology draft.

The owner ledger is the machine-readable source of truth.  The generated
Markdown workbook is a human interface and must not be parsed as authority.
This workflow governs ontology approval only.  The owner ledger also records
the approved replacement of mandatory downstream dual review with canonical
algorithm labels and optional append-only human audits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
from typing import Any, Iterable

from alphatrade.market_genome.ontology import revision
from alphatrade.market_genome.ontology import simulated_review


SCHEMA_VERSION = "mg1a.owner_review.v1"
PROFILE_NAME = "mg1a-v2-owner-review"
MILESTONE = "MG1-A.1"
REVIEW_SCOPE = "ONTOLOGY_DRAFT_APPROVAL"

LEDGER_PATH = "configs/market_genome/reviews/mg1a_v2_owner/review.yaml"
SCHEMA_PATH = (
    "src/alphatrade/market_genome/ontology/mg1a_owner_review.schema.json"
)
WORKBOOK_PATH = "docs/alphaTrade/market_genome/MG1A_V2_OWNER_REVIEW"
QUESTIONS_ZH_PATH = (
    "configs/market_genome/reviews/mg1a_v2_owner/questions_zh.json"
)
QUESTIONS_ZH_SHA256 = (
    "28f048815d4011ae023acc3d34c5421a0fec9bb0ff05a8fc1ac8a3e5b80fd3df"
)

TARGET = {
    "git_commit": "0d378954d49b6467d05fdef152c522e94b660978",
    "path": "configs/market_genome/pattern_ontology_v2.yaml",
    "file_sha256": (
        "cd569c19a73768b19b8f9ded2b570dc3fc93796a87b97ecd27becdd32ba5978e"
    ),
    "resolved_contract_sha256": (
        "ad3228a608998cc0823c440af0013b12658c9f8dcda4b858ea719be879389ebd"
    ),
}

AI_REFERENCE = {
    "role": "ADVISORY_ONLY",
    "bundle_path": simulated_review.BUNDLE_PATH,
    "bundle_sha256": (
        "0e9a4f798a540e86901757ebf2633a51dcbab217ea5d9ea38900289937b7ec4c"
    ),
    "adjudication_path": (
        "configs/market_genome/reviews/mg1a_v2_ai_simulated/"
        "adjudication.json"
    ),
    "adjudication_sha256": (
        "b10638c96338556feaa6c8e1766e19c1baf456390220281ceb792a5b0db07df3"
    ),
    "decisions_sha256": (
        "01a9c889c94ac47a9707bee1cc49d9b09b0d412eb5632f140fa39bfeeabc15d1"
    ),
}

OWNER = {
    "reviewer_id": "PROJECT_OWNER",
    "role": "PROJECT_OWNER",
    "appointment": "EXPLICIT_USER_INSTRUCTION",
    "decision_mode": "SINGLE_ACCOUNTABLE_OWNER",
    "appointed_owner_count": 1,
    "second_review_required": False,
    "adjudication_required": False,
    "personal_data_recorded": False,
}

POLICY = {
    "all_assumptions_required": True,
    "approve_as_is_only_freeze_ready": True,
    "non_approval_blocks_freeze": True,
    "owner_may_override_ai_advisory": True,
    "non_accept_advisory_override_requires_rationale": True,
    "draft_remains_immutable": True,
    "downstream_gold_annotation_policy_changed": True,
    "frozen_change_requires_new_version": True,
    "training_and_gpu_deferred": True,
}

EXECUTION = {
    "gpu_executed": False,
    "jax_executed": False,
    "training_executed": False,
    "labels_created": False,
    "synthetic_samples_created": False,
    "trading_execution_performed": False,
}

OWNER_DECISIONS = (
    "PENDING",
    "APPROVE_AS_IS",
    "REQUEST_REVISION",
    "DEFER",
    "REJECT",
)
SIGNOFF_DECISIONS = (
    "PENDING",
    "RETURN_FOR_REVISION",
    "APPROVE_FOR_MG1B_FREEZE",
)
OWNER_SIGNOFF_ATTESTATION = (
    "I approve the hash-bound MG1-A.1 draft for the MG1-B freeze step; "
    "this signoff does not freeze the ontology or unblock gold-set production."
)
OWNER_RETURN_ATTESTATION = (
    "I return the hash-bound MG1-A.1 draft for revision; this signoff does not "
    "approve or freeze the ontology or unblock gold-set production."
)
SIGNOFF_ATTESTATIONS = {
    "RETURN_FOR_REVISION": OWNER_RETURN_ATTESTATION,
    "APPROVE_FOR_MG1B_FREEZE": OWNER_SIGNOFF_ATTESTATION,
}

MISSING_CONTRACT_IDS = {
    "MG1A.COMMON.ROLL.001",
    "MG1A.COMMON.ZONE.001",
}
TARGET_REVIEW_AVAILABLE_AT = "2026-07-13T18:21:03+08:00"

_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d"
    r"(?:\.\d+)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$"
)

_CHAPTERS = (
    (
        "01_common_contracts.md",
        "公共合同",
        lambda assumption_id: (
            ".COMMON." in assumption_id
            and assumption_id != "MG1A1.COMMON.FAMILY_REVISIONS.001"
        ),
    ),
    (
        "02_trend_range.md",
        "趋势与区间结构",
        lambda assumption_id: _family_token(assumption_id)
        in {"TREND_UP", "TREND_DOWN", "TREND_TRANSITION", "RANGE"},
    ),
    (
        "03_reversal_morphologies.md",
        "反转形态",
        lambda assumption_id: _family_token(assumption_id)
        in {
            "DOUBLE_TOP",
            "DOUBLE_BOTTOM",
            "HEAD_SHOULDERS_TOP",
            "INVERSE_HEAD_SHOULDERS",
        },
    ),
    (
        "04_break_retest_conversion.md",
        "突破、回踩与角色转换",
        lambda assumption_id: _family_token(assumption_id)
        in {
            "BREAKOUT",
            "FAILED_BREAKOUT",
            "RETEST",
            "SUPPORT_RESISTANCE_CONVERSION",
        },
    ),
    (
        "05_final_integration.md",
        "整体一致性确认",
        lambda assumption_id: (
            assumption_id == "MG1A1.COMMON.FAMILY_REVISIONS.001"
        ),
    ),
)

_ADVISORY_GUIDANCE_ZH = {
    "ACCEPT": "AI 技术预审未提出阻断项；仍需负责人确认领域语义。",
    "REVISE": "AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。",
    "DEFER": "AI 技术预审认为缺少外部证据或注册表；请判断是否延后。",
    "REJECT": "AI 技术预审建议否决当前命题；请检查替代方案是否充分。",
}


class OwnerReviewValidationError(ValueError):
  """Raised when owner-review material cannot be loaded safely."""

  def __init__(self, errors: list[str]):
    super().__init__("; ".join(errors))
    self.errors = tuple(errors)


def repository_root() -> Path:
  """Return the repository containing this module."""
  return Path(__file__).resolve().parents[4]


def default_ledger_path(repo_root: Path | None = None) -> Path:
  root = (repo_root or repository_root()).resolve()
  return root / LEDGER_PATH


def owner_review_schema_path(repo_root: Path | None = None) -> Path:
  root = (repo_root or repository_root()).resolve()
  return root / SCHEMA_PATH


def default_workbook_path(repo_root: Path | None = None) -> Path:
  root = (repo_root or repository_root()).resolve()
  return root / WORKBOOK_PATH


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_file_bytes(repo_root: Path, commit: str, path: str) -> bytes:
  completed = subprocess.run(
      ["git", "show", f"{commit}:{path}"],
      cwd=repo_root,
      check=False,
      capture_output=True,
  )
  if completed.returncode:
    raise ValueError("commit or target path is unavailable in Git")
  return completed.stdout


def _format_jsonschema_path(parts: Iterable[Any]) -> str:
  rendered = "/".join(str(part) for part in parts)
  return rendered or "<root>"


def _schema_errors_from_snapshot(payload: Any, snapshot: bytes) -> list[str]:
  try:
    schema_payload = revision.loads_json_unique(snapshot.decode("utf-8"))
    import jsonschema
  except (UnicodeError, revision.RevisionValidationError, ImportError) as exc:
    return [f"schema: unavailable or invalid ({exc})"]
  try:
    jsonschema.Draft7Validator.check_schema(schema_payload)
    validator = jsonschema.Draft7Validator(schema_payload)
  except jsonschema.exceptions.SchemaError as exc:
    return [f"schema: unavailable or invalid ({exc.message})"]
  errors = sorted(
      validator.iter_errors(payload),
      key=lambda item: tuple(str(part) for part in item.absolute_path),
  )
  return [
      f"{_format_jsonschema_path(error.absolute_path)}: {error.message}"
      for error in errors
  ]


def json_schema_errors(
    payload: Any, *, repo_root: Path | None = None
) -> list[str]:
  """Validate a ledger against the independent closed JSON Schema."""
  root = (repo_root or repository_root()).resolve()
  try:
    snapshot = owner_review_schema_path(root).read_bytes()
  except OSError as exc:
    return [f"schema: unavailable or invalid ({exc})"]
  return _schema_errors_from_snapshot(payload, snapshot)


def _load_yaml_snapshot(path: Path) -> tuple[dict[str, Any], bytes]:
  try:
    snapshot = path.read_bytes()
    payload = revision.loads_yaml_unique(snapshot.decode("utf-8"))
  except (OSError, UnicodeError, revision.RevisionValidationError) as exc:
    raise OwnerReviewValidationError(
        [f"owner review ledger unavailable or invalid ({exc})"]
    ) from exc
  if not isinstance(payload, dict):
    raise OwnerReviewValidationError(["<root>: expected a mapping"])
  records = payload.get("records")
  if isinstance(records, list):
    for record in records:
      if isinstance(record, dict) and isinstance(
          record.get("reviewed_at"), datetime
      ):
        record["reviewed_at"] = record["reviewed_at"].isoformat()
  signoff = payload.get("owner_signoff")
  if isinstance(signoff, dict) and isinstance(signoff.get("signed_at"), datetime):
    signoff["signed_at"] = signoff["signed_at"].isoformat()
  return payload, snapshot


def load_ledger(path: Path) -> dict[str, Any]:
  """Load a duplicate-key-safe owner-review ledger."""
  return _load_yaml_snapshot(path)[0]


def _load_adjudication(repo_root: Path) -> tuple[dict[str, Any], bytes]:
  path = repo_root / AI_REFERENCE["adjudication_path"]
  try:
    snapshot = path.read_bytes()
    payload = simulated_review.loads_json_unique(snapshot.decode("utf-8"))
  except (
      OSError,
      UnicodeError,
      simulated_review.SimulatedReviewValidationError,
  ) as exc:
    raise OwnerReviewValidationError(
        [f"AI advisory unavailable or invalid ({exc})"]
    ) from exc
  if not isinstance(payload, dict):
    raise OwnerReviewValidationError(["AI advisory root must be an object"])
  return payload, snapshot


def _load_questions_zh(
    repo_root: Path, expected_ids: list[str]
) -> dict[str, str]:
  path = repo_root / QUESTIONS_ZH_PATH
  try:
    snapshot = path.read_bytes()
    payload = revision.loads_json_unique(snapshot.decode("utf-8"))
  except (OSError, UnicodeError, revision.RevisionValidationError) as exc:
    raise OwnerReviewValidationError(
        [f"Chinese review prompts unavailable or invalid ({exc})"]
    ) from exc
  errors: list[str] = []
  if _sha256_bytes(snapshot) != QUESTIONS_ZH_SHA256:
    errors.append("Chinese review prompts: immutable hash mismatch")
  if not isinstance(payload, dict):
    errors.append("Chinese review prompts: expected an object")
  else:
    if list(payload) != expected_ids:
      errors.append("Chinese review prompts: exact 70-item order required")
    if any(not _is_nonempty_string(value) for value in payload.values()):
      errors.append("Chinese review prompts: nonblank text required")
  if errors:
    raise OwnerReviewValidationError(errors)
  return payload


def _expected_review_decisions(repo_root: Path) -> list[dict[str, Any]]:
  adjudication, _snapshot = _load_adjudication(repo_root)
  decisions = adjudication.get("decisions")
  if not isinstance(decisions, list) or any(
      not isinstance(item, dict) for item in decisions
  ):
    raise OwnerReviewValidationError(
        ["AI advisory decisions must be a list of objects"]
    )
  return decisions


def _empty_record(assumption_id: str) -> dict[str, Any]:
  return {
      "assumption_id": assumption_id,
      "owner_decision": "PENDING",
      "owner_rationale": None,
      "required_changes": [],
      "evidence_needed_to_resume": [],
      "replacement_proposition": [],
      "reviewed_at": None,
  }


def _derived_summary(
    records: list[dict[str, Any]], signoff: dict[str, Any]
) -> dict[str, Any]:
  counts = Counter(record.get("owner_decision") for record in records)
  missing_contract_conflicts = sum(
      record.get("assumption_id") in MISSING_CONTRACT_IDS
      and record.get("owner_decision") == "APPROVE_AS_IS"
      for record in records
  )
  complete = bool(records) and counts["PENDING"] == 0
  signoff_complete = signoff.get("decision") != "PENDING"
  ready = (
      len(records) == 70
      and counts["APPROVE_AS_IS"] == 70
      and missing_contract_conflicts == 0
      and signoff.get("decision") == "APPROVE_FOR_MG1B_FREEZE"
  )
  return {
      "assumptions_total": len(records),
      "pending": counts["PENDING"],
      "approved_as_is": counts["APPROVE_AS_IS"],
      "request_revision": counts["REQUEST_REVISION"],
      "deferred": counts["DEFER"],
      "rejected": counts["REJECT"],
      "missing_contract_approval_conflicts": missing_contract_conflicts,
      "review_complete": complete,
      "owner_signoff_complete": signoff_complete,
      "ready_for_mg1b_freeze": ready,
  }


def _derived_status(
    summary: dict[str, Any], signoff: dict[str, Any]
) -> str:
  if summary["ready_for_mg1b_freeze"]:
    return "APPROVED_FOR_MG1B_FREEZE"
  if signoff.get("decision") == "RETURN_FOR_REVISION":
    return "OWNER_REVIEW_COMPLETED_WITH_ACTIONS"
  if summary["review_complete"]:
    return "AWAITING_OWNER_SIGNOFF"
  return "IN_PROGRESS"


def refresh_derived_fields(payload: dict[str, Any]) -> dict[str, Any]:
  """Refresh deterministic ledger hash, summary, and status in place."""
  records = payload.get("records")
  signoff = payload.get("owner_signoff")
  if not isinstance(records, list) or not isinstance(signoff, dict):
    raise OwnerReviewValidationError(
        ["records must be a list and owner_signoff must be a mapping"]
    )
  payload["records_sha256"] = revision.canonical_sha256(records)
  payload["summary"] = _derived_summary(records, signoff)
  payload["status"] = _derived_status(payload["summary"], signoff)
  return payload


def build_pending_ledger(repo_root: Path | None = None) -> dict[str, Any]:
  """Build the initial 70-item, all-pending owner-review ledger."""
  root = (repo_root or repository_root()).resolve()
  decisions = _expected_review_decisions(root)
  records = [_empty_record(item["assumption_id"]) for item in decisions]
  payload = {
      "schema_version": SCHEMA_VERSION,
      "profile": PROFILE_NAME,
      "milestone": MILESTONE,
      "review_scope": REVIEW_SCOPE,
      "status": "IN_PROGRESS",
      "target": dict(TARGET),
      "ai_reference": dict(AI_REFERENCE),
      "owner": dict(OWNER),
      "policy": dict(POLICY),
      "records_sha256": "0" * 64,
      "records": records,
      "summary": {},
      "owner_signoff": {
          "decision": "PENDING",
          "signed_at": None,
          "target_resolved_contract_sha256": None,
          "review_records_sha256": None,
          "attestation": None,
      },
      "execution": dict(EXECUTION),
  }
  return refresh_derived_fields(payload)


def _is_nonempty_string(value: Any) -> bool:
  return isinstance(value, str) and bool(value.strip())


def _parse_timestamp(value: Any) -> datetime | None:
  if not isinstance(value, str) or not _RFC3339_RE.fullmatch(value):
    return None
  normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
  try:
    return datetime.fromisoformat(normalized)
  except ValueError:
    return None


def _valid_timestamp(value: Any) -> bool:
  parsed = _parse_timestamp(value)
  return parsed is not None and parsed.tzinfo is not None


def _valid_string_list(value: list[Any]) -> bool:
  return all(_is_nonempty_string(item) for item in value)


def _record_errors(
    records: list[dict[str, Any]],
    advisory: list[dict[str, Any]],
) -> list[str]:
  errors: list[str] = []
  expected_ids = [item["assumption_id"] for item in advisory]
  observed_ids = [record.get("assumption_id") for record in records]
  if observed_ids != expected_ids:
    errors.append("records: exact 70-item AI advisory order required")
  if len(observed_ids) != len(set(observed_ids)):
    errors.append("records: duplicate assumption_id")
  advisory_by_id = {item["assumption_id"]: item for item in advisory}

  for index, record in enumerate(records):
    prefix = f"records/{index}"
    assumption_id = record.get("assumption_id")
    decision = record.get("owner_decision")
    rationale = record.get("owner_rationale")
    changes = record.get("required_changes")
    evidence = record.get("evidence_needed_to_resume")
    replacement = record.get("replacement_proposition")
    reviewed_at = record.get("reviewed_at")
    if decision not in OWNER_DECISIONS:
      continue
    if not all(isinstance(value, list) for value in (changes, evidence, replacement)):
      continue
    for field_name, values in (
        ("required_changes", changes),
        ("evidence_needed_to_resume", evidence),
        ("replacement_proposition", replacement),
    ):
      if not _valid_string_list(values):
        errors.append(f"{prefix}/{field_name}: nonblank strings required")

    if decision == "PENDING":
      if (
          rationale is not None
          or changes
          or evidence
          or replacement
          or reviewed_at is not None
      ):
        errors.append(f"{prefix}: pending record must have empty inputs")
      continue

    if not _valid_timestamp(reviewed_at):
      errors.append(f"{prefix}/reviewed_at: RFC3339 timestamp required")
    else:
      reviewed_time = _parse_timestamp(reviewed_at)
      target_time = _parse_timestamp(TARGET_REVIEW_AVAILABLE_AT)
      if reviewed_time is not None and target_time is not None:
        if reviewed_time < target_time:
          errors.append(
              f"{prefix}/reviewed_at: cannot precede the review target"
          )
    if decision == "APPROVE_AS_IS":
      if changes or evidence or replacement:
        errors.append(
            f"{prefix}: approval cannot carry revision/defer/reject inputs"
        )
      advisory_record = advisory_by_id.get(assumption_id, {})
      if (
          advisory_record.get("adjudicated_disposition") != "ACCEPT"
          and not _is_nonempty_string(rationale)
      ):
        errors.append(
            f"{prefix}/owner_rationale: required when overriding non-ACCEPT AI advice"
        )
      if assumption_id in MISSING_CONTRACT_IDS:
        errors.append(
            f"{prefix}: APPROVE_AS_IS forbidden because the v2 contract is missing"
        )
    elif decision == "REQUEST_REVISION":
      if not _is_nonempty_string(rationale) or not changes:
        errors.append(
            f"{prefix}: revision requires rationale and required_changes"
        )
      if evidence or replacement:
        errors.append(
            f"{prefix}: revision cannot carry defer/reject-only inputs"
        )
    elif decision == "DEFER":
      if not _is_nonempty_string(rationale) or not evidence:
        errors.append(
            f"{prefix}: defer requires rationale and evidence_needed_to_resume"
        )
      if changes or replacement:
        errors.append(
            f"{prefix}: defer cannot carry revision/reject-only inputs"
        )
    elif decision == "REJECT":
      if not _is_nonempty_string(rationale) or not replacement:
        errors.append(
            f"{prefix}: rejection requires rationale and replacement_proposition"
        )
      if changes or evidence:
        errors.append(
            f"{prefix}: rejection cannot carry revision/defer-only inputs"
        )
  return errors


def _signoff_errors(
    payload: dict[str, Any], expected_summary: dict[str, Any]
) -> list[str]:
  errors: list[str] = []
  signoff = payload["owner_signoff"]
  decision = signoff.get("decision")
  details = (
      signoff.get("signed_at"),
      signoff.get("target_resolved_contract_sha256"),
      signoff.get("review_records_sha256"),
      signoff.get("attestation"),
  )
  if decision == "PENDING":
    if any(value is not None for value in details):
      errors.append("owner_signoff: pending signoff must have null details")
    return errors
  if decision not in SIGNOFF_DECISIONS:
    return errors
  if not expected_summary["review_complete"]:
    errors.append("owner_signoff: all 70 owner decisions required first")
  if not _valid_timestamp(signoff.get("signed_at")):
    errors.append("owner_signoff/signed_at: RFC3339 timestamp required")
  else:
    signed_at = _parse_timestamp(signoff["signed_at"])
    reviewed_times = [
        parsed
        for record in payload["records"]
        if (parsed := _parse_timestamp(record.get("reviewed_at"))) is not None
    ]
    if signed_at is not None and reviewed_times and signed_at < max(reviewed_times):
      errors.append("owner_signoff/signed_at: cannot precede record review times")
  if signoff.get("target_resolved_contract_sha256") != TARGET[
      "resolved_contract_sha256"
  ]:
    errors.append("owner_signoff: target contract hash mismatch")
  if signoff.get("review_records_sha256") != payload.get("records_sha256"):
    errors.append("owner_signoff: review records hash mismatch")
  if signoff.get("attestation") != SIGNOFF_ATTESTATIONS.get(decision):
    errors.append("owner_signoff/attestation: exact controlled statement required")

  approved = expected_summary["approved_as_is"] == 70
  has_actions = any(
      expected_summary[field]
      for field in ("request_revision", "deferred", "rejected")
  )
  if decision == "APPROVE_FOR_MG1B_FREEZE" and not approved:
    errors.append("owner_signoff: approval requires 70 APPROVE_AS_IS records")
  if decision == "RETURN_FOR_REVISION" and not has_actions:
    errors.append("owner_signoff: return requires at least one owner action")
  return errors


def _validate_bound_inputs(
    root: Path,
    errors: list[str],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
  snapshots: dict[str, bytes] = {}
  paths = {
      TARGET["path"]: root / TARGET["path"],
      AI_REFERENCE["bundle_path"]: root / AI_REFERENCE["bundle_path"],
      AI_REFERENCE["adjudication_path"]: root
      / AI_REFERENCE["adjudication_path"],
  }
  for relative, path in paths.items():
    try:
      snapshots[relative] = path.read_bytes()
    except OSError as exc:
      errors.append(f"bound input {relative}: unavailable ({exc})")
  if len(snapshots) != len(paths):
    return [], snapshots

  if _sha256_bytes(snapshots[TARGET["path"]]) != TARGET["file_sha256"]:
    errors.append("target/file_sha256: current v2 bytes do not match")
  if _sha256_bytes(snapshots[AI_REFERENCE["bundle_path"]]) != AI_REFERENCE[
      "bundle_sha256"
  ]:
    errors.append("ai_reference/bundle_sha256: current bytes do not match")
  if _sha256_bytes(snapshots[AI_REFERENCE["adjudication_path"]]) != AI_REFERENCE[
      "adjudication_sha256"
  ]:
    errors.append("ai_reference/adjudication_sha256: current bytes do not match")

  try:
    v2 = revision.loads_yaml_unique(
        snapshots[TARGET["path"]].decode("utf-8")
    )
    adjudication = simulated_review.loads_json_unique(
        snapshots[AI_REFERENCE["adjudication_path"]].decode("utf-8")
    )
  except (
      UnicodeError,
      revision.RevisionValidationError,
      simulated_review.SimulatedReviewValidationError,
  ) as exc:
    errors.append(f"bound input: invalid ({exc})")
    return [], snapshots
  if not isinstance(v2, dict) or not isinstance(adjudication, dict):
    errors.append("bound input: target and adjudication must be mappings")
    return [], snapshots
  if revision.canonical_sha256(v2.get("resolved_contract")) != TARGET[
      "resolved_contract_sha256"
  ]:
    errors.append("target/resolved_contract_sha256: semantic hash mismatch")
  decisions = adjudication.get("decisions")
  if not isinstance(decisions, list) or any(
      not isinstance(item, dict) for item in decisions
  ):
    errors.append("ai_reference/decisions: object list required")
    decisions = []
  elif simulated_review.canonical_sha256(decisions) != AI_REFERENCE[
      "decisions_sha256"
  ]:
    errors.append("ai_reference/decisions_sha256: semantic hash mismatch")

  try:
    committed = _git_file_bytes(root, TARGET["git_commit"], TARGET["path"])
  except ValueError as exc:
    errors.append(f"target/git_commit: {exc}")
  else:
    if _sha256_bytes(committed) != TARGET["file_sha256"]:
      errors.append("target/git_commit: committed v2 hash mismatch")

  try:
    _bundle, bundle_errors = simulated_review.validate_bundle(
        root / AI_REFERENCE["bundle_path"], repo_root=root
    )
  except simulated_review.SimulatedReviewValidationError as exc:
    bundle_errors = list(exc.errors)
  errors.extend(f"ai_reference/bundle: {item}" for item in bundle_errors)
  return decisions, snapshots


def _validate_ledger_payload(
    payload: dict[str, Any], *, repo_root: Path | None = None
) -> list[str]:
  root = (repo_root or repository_root()).resolve()
  errors = json_schema_errors(payload, repo_root=root)
  if errors:
    return errors

  if payload["target"] != TARGET:
    errors.append("target: exact reviewed artifact binding required")
  if payload["ai_reference"] != AI_REFERENCE:
    errors.append("ai_reference: exact immutable advisory binding required")
  if payload["owner"] != OWNER:
    errors.append("owner: single accountable project owner required")
  if payload["policy"] != POLICY:
    errors.append("policy: exact ontology-only owner policy required")
  if payload["execution"] != EXECUTION:
    errors.append("execution: no GPU/training/labels/trading required")

  advisory, snapshots = _validate_bound_inputs(root, errors)
  records = payload["records"]
  errors.extend(_record_errors(records, advisory))
  expected_records_sha = revision.canonical_sha256(records)
  if payload["records_sha256"] != expected_records_sha:
    errors.append("records_sha256: canonical record hash mismatch")
  expected_summary = _derived_summary(records, payload["owner_signoff"])
  if payload["summary"] != expected_summary:
    errors.append("summary: derived values do not match records/signoff")
  expected_status = _derived_status(expected_summary, payload["owner_signoff"])
  if payload["status"] != expected_status:
    errors.append("status: derived status does not match records/signoff")
  errors.extend(_signoff_errors(payload, expected_summary))

  for relative, snapshot in snapshots.items():
    try:
      observed = _sha256_file(root / relative)
    except OSError as exc:
      errors.append(f"snapshot/{relative}: unavailable after validation ({exc})")
      continue
    if observed != _sha256_bytes(snapshot):
      errors.append(f"snapshot/{relative}: input changed during validation")
  return errors


def validate_ledger_payload(
    payload: dict[str, Any], *, repo_root: Path | None = None
) -> list[str]:
  """Validate owner-review semantics and fail closed on malformed input."""
  try:
    return _validate_ledger_payload(payload, repo_root=repo_root)
  except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
    return [
        "semantic validation failed closed for malformed data "
        f"({type(exc).__name__}: {exc})"
    ]


def validate_ledger(
    path: Path, *, repo_root: Path | None = None
) -> tuple[dict[str, Any], list[str]]:
  """Load and validate an owner-review ledger with snapshot checks."""
  payload, ledger_snapshot = _load_yaml_snapshot(path)
  root = (repo_root or repository_root()).resolve()
  schema_path = owner_review_schema_path(root)
  try:
    schema_snapshot = schema_path.read_bytes()
  except OSError as exc:
    schema_snapshot = b""
    errors = [f"schema: unavailable before validation ({exc})"]
  else:
    errors = _schema_errors_from_snapshot(payload, schema_snapshot)
  if not errors:
    errors.extend(validate_ledger_payload(payload, repo_root=root))

  try:
    if _sha256_file(path) != _sha256_bytes(ledger_snapshot):
      errors.append("ledger: input changed during validation")
  except OSError as exc:
    errors.append(f"ledger: unavailable after validation ({exc})")
  if schema_snapshot:
    try:
      if _sha256_file(schema_path) != _sha256_bytes(schema_snapshot):
        errors.append("schema: input changed during validation")
    except OSError as exc:
      errors.append(f"schema: unavailable after validation ({exc})")
  return payload, errors


def _family_token(assumption_id: str) -> str:
  parts = assumption_id.split(".")
  return parts[1] if len(parts) > 1 else ""


def _family_by_name(contract: dict[str, Any], family_name: str) -> dict[str, Any]:
  for family in contract["families"]:
    if family["family"] == family_name:
      return family
  raise KeyError(f"unknown family {family_name}")


def _selected(source: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
  return {field: source[field] for field in fields if field in source}


def _contract_excerpt(
    assumption_id: str, v2: dict[str, Any]
) -> tuple[str, Any]:
  contract = v2["resolved_contract"]
  parts = assumption_id.split(".")
  family_token = parts[1]
  component = parts[2]

  if family_token != "COMMON":
    family = _family_by_name(contract, family_token.lower())
    if component == "PHASE":
      return "EXACT", {"phase_contract": family["phase_contract"]}
    if component == "CONFIRMATION":
      return "EXACT", {
          "confirmation_rules": family["confirmation_rules"],
          "branch_transitions": family["phase_contract"][
              "branch_transitions"
          ],
      }
    if component == "INVALIDATION":
      return "EXACT", {
          "invalidation_rules": family["invalidation_rules"],
          "invalidation_transitions": family["phase_contract"][
              "invalidation_transitions"
          ],
      }
    excluded = {
        "phase_contract",
        "confirmation_rules",
        "invalidation_rules",
        "assumption_ids",
    }
    return "EXACT", {
        key: value for key, value in family.items() if key not in excluded
    }

  mappings: dict[str, tuple[str, ...]] = {
      "MG1A.COMMON.LIFECYCLE.001": ("common_lifecycle",),
      "MG1A.COMMON.SCALE.001": ("scale_spec_contract",),
      "MG1A.COMMON.NORMALIZATION.001": ("geometry_formula_contract",),
      "MG1A.COMMON.SWING_CAUSALITY.001": ("anchor_contract",),
      "MG1A.COMMON.OVERLAP.001": ("relationship_graph",),
      "MG1A.COMMON.FILTER_MISSINGNESS.001": (
          "geometry_formula_contract",
      ),
      "MG1A.COMMON.LABEL_TIMING.001": ("label_namespaces",),
      "MG1A1.COMMON.OBJECT_MODEL.001": ("object_model",),
      "MG1A1.COMMON.LIFECYCLE_MAPPING.001": ("common_lifecycle",),
      "MG1A1.COMMON.SCALE_SPEC.001": ("scale_spec_contract",),
      "MG1A1.COMMON.ANCHOR_REVISION.001": ("anchor_contract",),
      "MG1A1.COMMON.RELATION_GRAPH.001": ("relationship_graph",),
      "MG1A1.COMMON.FORMULA_CONTRACT.001": (
          "geometry_formula_contract",
      ),
      "MG1A1.COMMON.REVIEW_PROTOCOL.001": ("review_protocol",),
  }
  if assumption_id in mappings:
    fields = mappings[assumption_id]
    return "EXACT", {field: contract[field] for field in fields}
  if assumption_id == "MG1A.COMMON.PARTIAL_BAR.001":
    return "EXACT", {
        "anchor_contract": _selected(
            contract["anchor_contract"],
            (
                "committed_anchor_requires_closed_bar",
                "partial_bar_namespace",
                "committed_history_is_append_only",
            ),
        )
    }
  if assumption_id == "MG1A.COMMON.SESSION.001":
    return "EXACT", {
        "scale_spec_contract": _selected(
            contract["scale_spec_contract"],
            ("aggregation_policy", "session_alignment"),
        )
    }
  if assumption_id == "MG1A.COMMON.PROVENANCE.001":
    return "EXACT", {
        "anchor_required_fields": contract["anchor_contract"][
            "required_fields"
        ],
        "relation_required_fields": contract["relationship_graph"][
            "required_fields"
        ],
        "review_required_fields": contract["review_protocol"][
            "required_fields"
        ],
    }
  if assumption_id == "MG1A.COMMON.MIRROR.001":
    return "EXACT", {
        "family_orientation_partition": [
            _selected(
                family,
                (
                    "family",
                    "object_class",
                    "structural_orientation",
                    "orientation_binding",
                    "allowed_instance_orientations",
                    "semantic_roles",
                ),
            )
            for family in contract["families"]
        ]
    }
  if assumption_id == "MG1A1.COMMON.MORPHOLOGY_EFFECT.001":
    return "EXACT", {
        "morphology_effect_separation": contract["object_model"][
            "morphology_effect_separation"
        ],
        "label_namespaces": contract["label_namespaces"],
    }
  if assumption_id == "MG1A1.COMMON.FAMILY_REVISIONS.001":
    return "EXACT", {
        "family_review_index": [
            _selected(
                family,
                (
                    "family",
                    "object_class",
                    "structural_orientation",
                    "assumption_ids",
                ),
            )
            for family in contract["families"]
        ]
    }
  if assumption_id == "MG1A.COMMON.ROLL.001":
    return "MISSING", {
        "missing_contract": "resolved_contract.roll_contract",
        "meaning": "Current v2 contains no roll contract to approve.",
    }
  if assumption_id == "MG1A.COMMON.ZONE.001":
    return "MISSING", {
        "missing_contract": "resolved_contract.zone_contract",
        "meaning": "Current v2 contains no common zone contract to approve.",
    }
  return "REFERENCE_ONLY", {
      "meaning": "No dedicated exact-excerpt mapping is defined.",
  }


def _yaml_text(value: Any) -> str:
  try:
    import yaml
  except ImportError as exc:  # pragma: no cover - dependency guard
    raise OwnerReviewValidationError([f"PyYAML is required: {exc}"]) from exc
  return yaml.safe_dump(
      value,
      sort_keys=False,
      allow_unicode=True,
      width=100,
      default_flow_style=False,
  ).rstrip()


def _wrap(value: str) -> str:
  return textwrap.fill(value, width=100, break_long_words=False)


def _bullet_list(values: list[str], empty: str = "无") -> list[str]:
  if not values:
    return [f"- {empty}"]
  return [
      textwrap.fill(
          value,
          width=100,
          initial_indent="- ",
          subsequent_indent="  ",
          break_long_words=False,
      )
      for value in values
  ]


def _labeled_bullet(label: str, value: str) -> str:
  return textwrap.fill(
      value,
      width=100,
      initial_indent=f"- {label}：",
      subsequent_indent="  ",
      break_long_words=False,
  )


def _labeled_paragraph(label: str, value: str) -> str:
  return textwrap.fill(
      value,
      width=100,
      initial_indent=f"**{label}**：",
      subsequent_indent="  ",
      break_long_words=False,
  )


def _record_markdown(
    number: int,
    record: dict[str, Any],
    advisory: dict[str, Any],
    question_zh: str,
    v2: dict[str, Any],
) -> str:
  assumption_id = record["assumption_id"]
  excerpt_status, excerpt = _contract_excerpt(assumption_id, v2)
  paths = advisory.get("v2_contract_evidence_paths", [])
  lines = [
      f'<a id="item-{number:02d}"></a>',
      "",
      f"## {number:02d}. `{assumption_id}`",
      "",
      _labeled_paragraph("要确认的问题", question_zh),
      "",
      "<details><summary>查看英文原始问题</summary>",
      "",
      _wrap(advisory["question"]),
      "",
      "</details>",
      "",
      "**当前定义定位**：",
      *_bullet_list([f"`{path}`" for path in paths], empty="未提供定位"),
      "",
  ]
  if excerpt_status == "MISSING":
    lines.extend(
        [
            "> **当前定义缺失。** 这一项不能按现有 v2 文本直接批准；请要求补充、延后或拒绝。",
            "",
        ]
    )
  if assumption_id == "MG1A1.COMMON.REVIEW_PROTOCOL.001":
    lines.extend(
        [
            "> **范围说明：** 摘录中的 TWO_REVIEWERS 是当前 v2 的待审核规则；"
            "它不为本次 ontology owner review 增加第二审核者，且负责人已要求在修订版中取消。",
            "",
        ]
    )
  lines.extend(
      [
          f"<details><summary>查看当前 v2 选定字段原文（{excerpt_status}）</summary>",
          "",
          "```yaml",
          _yaml_text(excerpt),
          "```",
          "",
          "</details>",
          "",
          "### AI 参考意见（无审批权）",
          "",
          f"- 建议：`{advisory['adjudicated_disposition']}`",
          f"- 置信度：`{advisory['confidence']}`",
          _labeled_bullet(
              "中文提示",
              _ADVISORY_GUIDANCE_ZH[advisory["adjudicated_disposition"]],
          ),
          _labeled_bullet("风险摘要", advisory["rationale"]),
          "- 建议修改：",
          *_bullet_list(advisory["required_changes"]),
          "- 恢复审核所需证据：",
          *_bullet_list(advisory["evidence_needed_to_resume"]),
          "",
          "### 项目负责人决定",
          "",
          "以下块来自机器台账；正式结果以 `review.yaml` 为准。",
          "",
          "```yaml",
          _yaml_text(record),
          "```",
          "",
          "---",
          "",
      ]
  )
  return "\n".join(lines)


def _read_render_inputs(
    payload: dict[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
  try:
    v2 = revision.load_revision(root / TARGET["path"])
  except revision.RevisionValidationError as exc:
    raise OwnerReviewValidationError(list(exc.errors)) from exc
  advisory = _expected_review_decisions(root)
  expected_ids = [item["assumption_id"] for item in advisory]
  questions_zh = _load_questions_zh(root, expected_ids)
  return v2, advisory, questions_zh


def render_workbook(
    payload: dict[str, Any], *, repo_root: Path | None = None
) -> dict[str, str]:
  """Render a Chinese owner workbook from the canonical ledger and inputs."""
  root = (repo_root or repository_root()).resolve()
  v2, advisory, questions_zh = _read_render_inputs(payload, root)
  records = payload["records"]
  record_by_id = {record["assumption_id"]: record for record in records}
  advisory_by_id = {item["assumption_id"]: item for item in advisory}
  summary = payload["summary"]

  readme = [
      "# MG1-A.1 项目负责人审核工作簿",
      "",
      "本目录是人工审核入口。正式机器记录位于",
      "`configs/market_genome/reviews/mg1a_v2_owner/review.yaml`。",
      "",
      "## 当前状态",
      "",
      f"- 状态：`{payload['status']}`",
      f"- 已处理：`{70 - summary['pending']}/70`",
      f"- 通过当前文本：`{summary['approved_as_is']}`",
      f"- 要求修改：`{summary['request_revision']}`",
      f"- 延后：`{summary['deferred']}`",
      f"- 否决：`{summary['rejected']}`",
      f"- 缺失合同误批准：`{summary['missing_contract_approval_conflicts']}`",
      f"- 可进入 MG1-B freeze：`{str(summary['ready_for_mg1b_freeze']).lower()}`",
      "",
      "## 审核权限",
      "",
      "- 本次范围仅为 `MG1-A ontology draft approval`。",
      "- `PROJECT_OWNER` 是唯一终审人，不要求第二审核者，不设置分歧裁决者。",
      "- AI 双审仅提供风险提示，没有审批权；负责人可以覆盖其意见。",
      "- 修订版必须取消后续标签治理中的强制双审；正式标签由版本化算法生成，人工抽查可选。",
      "- 审核通过只表示可以启动 MG1-B freeze；当前 v2 文件仍是未冻结草案。",
      "- AI/Agent 只能转录负责人明确给出的决定，不得代替负责人审核或签署。",
      "",
      "## 决定含义",
      "",
      "| 决定 | 含义 | 必填内容 |",
      "|---|---|---|",
      "| `APPROVE_AS_IS` | 批准当前精确 v2 hash 对应文本 | 覆盖 AI 非 ACCEPT 时填写理由 |",
      "| `REQUEST_REVISION` | 修改后再审 | 理由、`required_changes` |",
      "| `DEFER` | 等待外部证据 | 理由、`evidence_needed_to_resume` |",
      "| `REJECT` | 否决当前命题 | 理由、`replacement_proposition` |",
      "| `PENDING` | 尚未审核 | 保持其他字段为空 |",
      "",
      "只有 70 条全部 `APPROVE_AS_IS`，并由同一 `PROJECT_OWNER` 另行最终签署，才会得到",
      "`APPROVED_FOR_MG1B_FREEZE`。任何其他决定都保持 gate 阻塞。",
      "",
      "## 先处理的硬阻塞",
      "",
      "- [`MG1A.COMMON.ROLL.001`](01_common_contracts.md#item-07)：v2 缺少 `roll_contract`。",
      "- [`MG1A.COMMON.ZONE.001`](01_common_contracts.md#item-11)：v2 缺少通用 `zone_contract`。",
      "",
      "这两条不能选择 `APPROVE_AS_IS`；必须要求修改、延后或拒绝。",
      "",
      "## 审核操作",
      "",
      "1. 按章节阅读中文问题、当前 v2 摘录和 AI 风险提示。",
      "2. 直接在对话中按 `assumption_id + 决定 + 理由/动作` 给出结论；工具只转录显式决定。",
      "3. 也可以直接编辑机器台账；非 `PENDING` 项必须填写 RFC3339 `reviewed_at`。",
      "4. Markdown 是生成视图，不要直接编辑；台账更新后运行下列 CPU-only 命令刷新。",
      "",
      "```bash",
      "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \\",
      "  python -m alphatrade.market_genome.ontology.owner_review --refresh --render --strict",
      "```",
      "",
      "5. 70 条全部处理后，由同一 `PROJECT_OWNER` 填写 `owner_signoff`。",
      "",
      "若更新进程异常中断并遗留 `.review.yaml.lock`，先确认原进程已结束，再运行",
      "`python -m alphatrade.market_genome.ontology.owner_review --recover-stale-lock`。",
      "",
      "## 目标绑定",
      "",
      f"- Git commit：`{TARGET['git_commit']}`",
      f"- v2 file SHA-256：`{TARGET['file_sha256']}`",
      f"- resolved contract SHA-256：`{TARGET['resolved_contract_sha256']}`",
      f"- AI decisions SHA-256：`{AI_REFERENCE['decisions_sha256']}`",
      "",
      "## 分组",
      "",
  ]
  for file_name, title, predicate in _CHAPTERS:
    count = sum(predicate(record["assumption_id"]) for record in records)
    readme.append(f"- [{title}]({file_name})：{count} 条")
  readme.extend(
      [
          "",
          "## AI 参考分布",
          "",
      ]
  )
  advisory_counts = Counter(
      item["adjudicated_disposition"] for item in advisory
  )
  readme.extend(
      [
          f"- `ACCEPT`：{advisory_counts['ACCEPT']}",
          f"- `REVISE`：{advisory_counts['REVISE']}",
          f"- `DEFER`：{advisory_counts['DEFER']}",
          f"- `REJECT`：{advisory_counts['REJECT']}",
          "",
          "## 70 条索引",
          "",
          "| # | Assumption | 分组 | AI | Owner | 标记 |",
          "|---:|---|---|---|---|---|",
      ]
  )
  for index, item in enumerate(advisory, start=1):
    assumption_id = item["assumption_id"]
    chapter_file = next(
        file_name
        for file_name, _title, predicate in _CHAPTERS
        if predicate(assumption_id)
    )
    record = record_by_id[assumption_id]
    marker = "MISSING_CONTRACT" if assumption_id in MISSING_CONTRACT_IDS else "-"
    readme.append(
        f"| {index:02d} | [`{assumption_id}`]({chapter_file}#item-{index:02d}) | "
        f"{_family_token(assumption_id)} | {item['adjudicated_disposition']} | "
        f"{record['owner_decision']} | {marker} |"
    )
  readme.extend(
      [
          "",
          "## 最终签署",
          "",
          "逐条审核完成后再填写 `review.yaml` 的 `owner_signoff`。签署不会原地修改或",
          "伪装冻结当前 v2；冻结文件和最终 hash 由后续 MG1-B 步骤生成。",
          "签署声明使用工具内置的受控文本，不能自由改写为已冻结或已解锁。",
          "",
      ]
  )

  rendered = {"README.md": "\n".join(readme)}
  sequence = {
      item["assumption_id"]: index
      for index, item in enumerate(advisory, start=1)
  }
  for file_name, title, predicate in _CHAPTERS:
    selected = [
        item for item in advisory if predicate(item["assumption_id"])
    ]
    chapter = [
        f"# {title}",
        "",
        "[返回审核总览](README.md)",
        "",
        f"本章共 {len(selected)} 条。AI 内容仅作参考，最终决定由 `PROJECT_OWNER` 作出。",
        "",
    ]
    for item in selected:
      assumption_id = item["assumption_id"]
      chapter.append(
          _record_markdown(
              sequence[assumption_id],
              record_by_id[assumption_id],
              advisory_by_id[assumption_id],
              questions_zh[assumption_id],
              v2,
          )
      )
    rendered[file_name] = "\n".join(chapter).rstrip() + "\n"
  return rendered


def _dump_ledger(payload: dict[str, Any]) -> str:
  try:
    import yaml
  except ImportError as exc:  # pragma: no cover - dependency guard
    raise OwnerReviewValidationError([f"PyYAML is required: {exc}"]) from exc
  return yaml.safe_dump(
      payload,
      sort_keys=False,
      allow_unicode=False,
      width=120,
      default_flow_style=False,
  )


def _write_text_atomic(
    path: Path,
    content: str,
    *,
    encoding: str,
    require_absent: bool = False,
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary_name: str | None = None
  try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
      temporary_name = handle.name
      handle.write(content)
      handle.flush()
      os.fsync(handle.fileno())
    if require_absent:
      try:
        os.link(temporary_name, path)
      except FileExistsError as exc:
        raise OwnerReviewValidationError(
            [f"refusing to overwrite existing ledger: {path}"]
        ) from exc
      Path(temporary_name).unlink()
    else:
      Path(temporary_name).replace(path)
    temporary_name = None
  finally:
    if temporary_name is not None:
      Path(temporary_name).unlink(missing_ok=True)


@contextmanager
def _lock_guard(path: Path) -> Iterable[None]:
  """Serialize lock-file lifecycle operations and survive process death."""
  try:
    import fcntl
  except ImportError as exc:  # pragma: no cover - Linux project environment
    raise OwnerReviewValidationError([f"fcntl is required: {exc}"]) from exc
  guard_key = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
  guard_path = Path(tempfile.gettempdir()) / (
      f"alphatrade-owner-review-{guard_key[:24]}.guard"
  )
  with guard_path.open("a+b") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _exclusive_update_lock(path: Path) -> Iterable[None]:
  """Serialize owner-ledger writers that use this workflow."""
  lock_path = path.with_name(f".{path.name}.lock")
  expected_content = f"pid={os.getpid()}\n"
  descriptor: int | None = None
  lock_identity: tuple[int, int] | None = None
  try:
    with _lock_guard(path):
      path.parent.mkdir(parents=True, exist_ok=True)
      try:
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
      except FileExistsError as exc:
        raise OwnerReviewValidationError(
            [f"owner review update is already in progress: {lock_path}"]
        ) from exc
      lock_stat = os.fstat(descriptor)
      lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
    try:
      content = expected_content.encode("ascii")
      offset = 0
      while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
          raise OSError("short write while initializing owner review lock")
        offset += written
      os.fsync(descriptor)
    except OSError as exc:
      raise OwnerReviewValidationError(
          [f"owner review lock initialization failed: {exc}"]
      ) from exc
    yield
  finally:
    if descriptor is not None:
      try:
        os.close(descriptor)
      finally:
        with _lock_guard(path):
          try:
            observed_stat = os.lstat(lock_path)
          except FileNotFoundError:
            observed_identity = None
          else:
            observed_identity = (observed_stat.st_dev, observed_stat.st_ino)
          if observed_identity == lock_identity:
            lock_path.unlink()


def _recover_stale_lock(path: Path) -> None:
  """Remove this workflow's lock only after its recorded process is gone."""
  lock_path = path.with_name(f".{path.name}.lock")
  with _lock_guard(path):
    try:
      content = lock_path.read_text(encoding="ascii").strip()
    except OSError as exc:
      raise OwnerReviewValidationError(
          [f"owner review lock is unavailable: {lock_path} ({exc})"]
      ) from exc
    match = re.fullmatch(r"pid=(\d+)", content)
    if match is None:
      raise OwnerReviewValidationError(
          [f"owner review lock has invalid content: {lock_path}"]
      )
    process_id = int(match.group(1))
    try:
      os.kill(process_id, 0)
    except ProcessLookupError:
      lock_path.unlink()
      return
    except PermissionError:
      pass
    raise OwnerReviewValidationError(
        [f"owner review update process is still active: pid={process_id}"]
    )


def write_workbook(
    payload: dict[str, Any],
    output_dir: Path,
    *,
    repo_root: Path | None = None,
) -> None:
  """Write the deterministic human workbook to an existing repository."""
  rendered = render_workbook(payload, repo_root=repo_root)
  output_dir.mkdir(parents=True, exist_ok=True)
  expected_names = set(rendered)
  existing_names = {path.name for path in output_dir.glob("*.md")}
  unexpected = existing_names - expected_names
  if unexpected:
    raise OwnerReviewValidationError(
        [f"workbook contains unexpected Markdown files: {sorted(unexpected)}"]
    )
  for name, content in rendered.items():
    _write_text_atomic(output_dir / name, content, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Validate or render the MG1-A.1 project-owner review."
  )
  parser.add_argument("--ledger", type=Path, default=default_ledger_path())
  parser.add_argument(
      "--workbook", type=Path, default=default_workbook_path()
  )
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument("--initialize", action="store_true")
  mode.add_argument("--refresh", action="store_true")
  mode.add_argument("--recover-stale-lock", action="store_true")
  parser.add_argument("--render", action="store_true")
  parser.add_argument("--strict", action="store_true")
  return parser


def main(argv: list[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  root = repository_root()
  try:
    if args.recover_stale_lock:
      _recover_stale_lock(args.ledger)
      print(f"recovered_stale_lock={args.ledger}")
      return 0
    if args.initialize:
      with _exclusive_update_lock(args.ledger):
        if args.ledger.exists():
          raise OwnerReviewValidationError(
              [f"refusing to overwrite existing ledger: {args.ledger}"]
          )
        payload = build_pending_ledger(root)
        initialize_errors = validate_ledger_payload(payload, repo_root=root)
        if initialize_errors:
          raise OwnerReviewValidationError(initialize_errors)
        _write_text_atomic(
            args.ledger,
            _dump_ledger(payload),
            encoding="ascii",
            require_absent=True,
        )
    elif args.refresh:
      with _exclusive_update_lock(args.ledger):
        payload, input_snapshot = _load_yaml_snapshot(args.ledger)
        refresh_derived_fields(payload)
        refresh_errors = validate_ledger_payload(payload, repo_root=root)
        if refresh_errors:
          raise OwnerReviewValidationError(refresh_errors)
        if _sha256_file(args.ledger) != _sha256_bytes(input_snapshot):
          raise OwnerReviewValidationError(
              ["ledger: input changed before refreshed content was written"]
          )
        _write_text_atomic(args.ledger, _dump_ledger(payload), encoding="ascii")
    else:
      payload = load_ledger(args.ledger)
    payload, errors = validate_ledger(args.ledger, repo_root=root)
    if (args.initialize or args.refresh or args.render) and not errors:
      write_workbook(payload, args.workbook, repo_root=root)
  except OwnerReviewValidationError as exc:
    payload = {}
    errors = list(exc.errors)
  summary = payload.get("summary", {})
  print(
      f"profile={payload.get('profile', PROFILE_NAME)} "
      f"status={payload.get('status', 'INVALID')} "
      f"reviewed={70 - summary.get('pending', 70)}/70 "
      f"ready_for_mg1b_freeze={summary.get('ready_for_mg1b_freeze', False)} "
      f"errors={len(errors)}"
  )
  for error in errors:
    print(f"ERROR {error}")
  if errors:
    return 1
  return 0


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
