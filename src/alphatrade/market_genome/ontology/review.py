"""CPU-only review ledger for the MG1 pattern ontology draft.

The ledger records expert findings without mutating or falsely freezing the
MG1-A draft.  A separate MG1-B contract must consume an approved ledger before
creating an immutable ontology.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from alphatrade.market_genome.ontology import schema as ontology_schema


SCHEMA_VERSION = "mg1.assumption_review.v1"
PROFILE_NAME = "mg1-review"
MILESTONE = "MG1-REVIEW"
BASE_DRAFT_PATH = "configs/market_genome/pattern_ontology_v1.yaml"
REVIEWER_ID = "CODEX_DOMAIN_EXPERT_01"
REVIEWER_ROLE = "USER_APPOINTED_AI_DOMAIN_EXPERT"

DECISION_IN_PROGRESS = "REVIEW_IN_PROGRESS"
DECISION_NEEDS_REVISION = "DOMAIN_REVIEW_COMPLETED_WITH_REQUIRED_REVISIONS"
DECISION_APPROVED = "APPROVED_FOR_FREEZE"

REVIEW_STATES = ("PENDING", "REVIEWED")
DISPOSITIONS = ("PENDING", "ACCEPT", "REVISE", "REJECT", "DEFER")
FREEZE_GATES = ("BLOCKED", "READY")

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_ABSOLUTE_THRESHOLD_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:points?|ticks?|percent|%|usd|cny|eur)\b",
    re.IGNORECASE,
)
_FUTURE_DECISION_RE = re.compile(
    r"\b(?:use\w*|confirm\w*|invalidat\w*|detect\w*|classif\w*|"
    r"deriv\w*|assign\w*|set)\b"
    r".{0,100}\b(?:next[-_ ]?(?:day|bar|week)(?:[-_ 's]*return)?|"
    r"following[-_ ]?bar(?:[-_ ]?return)?|forward[-_ ]?return|"
    r"future[-_ ]?(?:profit|return|outcome)|expected[-_ ]?return|"
    r"profit|pnl|return|outcome)\b"
    r"|\b(?:next[-_ ]?(?:day|bar|week)(?:[-_ 's]*return)?|"
    r"following[-_ ]?bar(?:[-_ ]?return)?|forward[-_ ]?return|"
    r"future[-_ ]?(?:profit|return|outcome)|expected[-_ ]?return|"
    r"profit|pnl|return|outcome)\b.{0,100}"
    r"\b(?:defin\w*|confirm\w*|invalidat\w*|detect\w*|classif\w*|"
    r"deriv\w*|assign\w*|set)\b.{0,60}\b(?:ontology|state|phase|class)\b",
    re.IGNORECASE,
)
_TRADING_ACTION_RE = re.compile(
    r"\b(?:buy|sell|take[-_ ]?profit|stop[-_ ]?loss|"
    r"trade[-_ ]?action|(?:long|short)[-_ ]+(?:entry|position)|"
    r"(?:entry|position|order)[-_ ]?(?:signal|intent))\b",
    re.IGNORECASE,
)
_DIRECT_PROHIBITION_PREFIX_RE = re.compile(
    r"\b(?:do\s+not|must\s+not|never|cannot|forbid\w*|prohibit\w*|not)"
    r"\s+(?:(?:ever|to|a|an|the)\s+){0,3}$",
    re.IGNORECASE,
)
_INTERNAL_PROHIBITION_RE = re.compile(
    r"^\s*(?:[A-Za-z_]+[-_ ]+){0,3}"
    r"(?:cannot|must\s+not|does\s+not|never)\b",
    re.IGNORECASE,
)
_DIRECT_PROHIBITION_SUFFIX_RE = re.compile(
    r"^\s+(?:is|are|remains?|must\s+be)?\s*"
    r"(?:forbidden|prohibited|not\s+allowed)\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"[.;\n]+")


def repository_root() -> Path:
  """Return the repository containing this module."""
  return Path(__file__).resolve().parents[4]


def review_schema_path(repo_root: Path | None = None) -> Path:
  """Return the independent Draft-07 schema for the review ledger."""
  if repo_root is None:
    return Path(__file__).with_name("mg1_assumption_review.schema.json")
  return (
      repo_root.resolve()
      / "src/alphatrade/market_genome/ontology/"
      "mg1_assumption_review.schema.json"
  )


def default_ledger_path(repo_root: Path | None = None) -> Path:
  root = (repo_root or repository_root()).resolve()
  return root / "configs/market_genome/pattern_ontology_v1_review.yaml"


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
  completed = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=repo_root,
      check=True,
      capture_output=True,
      text=True,
  )
  return completed.stdout.strip()


def _git_file_sha256(repo_root: Path, commit: str, path: str) -> str:
  completed = subprocess.run(
      ["git", "show", f"{commit}:{path}"],
      cwd=repo_root,
      check=False,
      capture_output=True,
  )
  if completed.returncode:
    raise ValueError("commit or draft path is unavailable in Git")
  return hashlib.sha256(completed.stdout).hexdigest()


def _empty_record(assumption: dict[str, Any]) -> dict[str, Any]:
  parameter_refs = list(assumption["parameter_refs"])
  return {
      "assumption_id": assumption["assumption_id"],
      "category": assumption["category"],
      "question": assumption["question"],
      "affected_families": list(assumption["affected_families"]),
      "parameter_refs": parameter_refs,
      "source_evidence": list(assumption["source_evidence"]),
      "review_state": "PENDING",
      "disposition": "PENDING",
      "freeze_gate": "BLOCKED",
      "expert_decision": None,
      "approved_invariants": [],
      "resolved_parameter_refs": [],
      "unresolved_parameter_refs": list(parameter_refs),
      "wildcard_expansions": [],
      "required_changes": [],
      "blocking_items": ["REVIEW_NOT_STARTED"],
      "evidence_ids": [],
      "reviewer_id": None,
      "reviewed_at": None,
  }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
  dispositions = Counter(record.get("disposition") for record in records)
  review_states = Counter(record.get("review_state") for record in records)
  gates = Counter(record.get("freeze_gate") for record in records)
  return {
      "assumptions_total": len(records),
      "reviewed": review_states["REVIEWED"],
      "pending": review_states["PENDING"],
      "accepted": dispositions["ACCEPT"],
      "revised": dispositions["REVISE"],
      "rejected": dispositions["REJECT"],
      "deferred": dispositions["DEFER"],
      "freeze_ready_records": gates["READY"],
      "freeze_blocked_records": gates["BLOCKED"],
      "resolved_parameter_refs": sum(
          len(record.get("resolved_parameter_refs", [])) for record in records
      ),
      "unresolved_parameter_refs": sum(
          len(record.get("unresolved_parameter_refs", []))
          for record in records
      ),
      "wildcard_expansions": sum(
          len(record.get("wildcard_expansions", [])) for record in records
      ),
      "freeze_ready": bool(records) and gates["READY"] == len(records),
  }


def _decision(summary: dict[str, Any]) -> str:
  if summary["pending"]:
    return DECISION_IN_PROGRESS
  if summary["freeze_ready"]:
    return DECISION_APPROVED
  return DECISION_NEEDS_REVISION


def _assessments(summary: dict[str, Any]) -> dict[str, str]:
  if summary["pending"]:
    return {
        "draft_assessment": "PENDING",
        "freeze_assessment": "NOT_EVALUATED",
        "mg1b_status": "BLOCKED",
        "next_milestone": "MG1-REVIEW",
    }
  if summary["freeze_ready"]:
    return {
        "draft_assessment": "APPROVED_FOR_FREEZE",
        "freeze_assessment": "READY_FOR_FREEZE",
        "mg1b_status": "READY_AFTER_FREEZE",
        "next_milestone": "MG1-B",
    }
  return {
      "draft_assessment": "APPROVED_AS_DRAFT",
      "freeze_assessment": "REVISION_REQUIRED_BEFORE_FREEZE",
      "mg1b_status": "BLOCKED",
      "next_milestone": "MG1-A.1",
  }


def refresh_derived_fields(payload: dict[str, Any]) -> dict[str, Any]:
  """Refresh hash, summary, and decision after an intentional ledger edit."""
  records = payload.get("records", [])
  if not isinstance(records, list):
    raise ValueError("records must be a list")
  payload["records_sha256"] = ontology_schema.canonical_sha256(records)
  payload["summary"] = _summary(records)
  payload["decision"] = _decision(payload["summary"])
  payload.update(_assessments(payload["summary"]))
  return payload


def apply_review_findings(
    payload: dict[str, Any], findings_payload: Any
) -> dict[str, Any]:
  """Apply a fail-closed expert findings overlay to a pending ledger."""
  if not isinstance(findings_payload, dict):
    raise ValueError("findings overlay must be a mapping")
  expected_root_keys = {
      "schema_version",
      "base_contract_sha256",
      "reviewer_id",
      "reviewed_at",
      "findings",
  }
  if set(findings_payload) != expected_root_keys:
    raise ValueError("findings overlay has missing or unknown root fields")
  if findings_payload["schema_version"] != "mg1.review_findings.v1":
    raise ValueError("unsupported findings overlay schema_version")
  if (
      findings_payload["base_contract_sha256"]
      != payload["base_draft"]["resolved_contract_sha256"]
  ):
    raise ValueError("findings overlay targets a different draft contract")
  if findings_payload["reviewer_id"] != payload["reviewer"]["reviewer_id"]:
    raise ValueError("findings overlay reviewer does not match ledger")
  if not _valid_timestamp(findings_payload["reviewed_at"]):
    raise ValueError("findings overlay reviewed_at must be RFC3339")
  findings = findings_payload["findings"]
  if not isinstance(findings, list) or not findings:
    raise ValueError("findings overlay requires non-empty findings")

  expected_finding_keys = {
      "assumption_id",
      "disposition",
      "expert_decision",
      "approved_invariants",
      "resolved_parameter_refs",
      "wildcard_expansions",
      "required_changes",
      "blocking_items",
      "evidence_ids",
  }
  records = {
      record["assumption_id"]: record for record in payload["records"]
  }
  observed_ids: set[str] = set()
  for index, finding in enumerate(findings):
    if not isinstance(finding, dict) or set(finding) != expected_finding_keys:
      raise ValueError(
          f"findings/{index}: missing or unknown finding fields"
      )
    assumption_id = finding["assumption_id"]
    if assumption_id in observed_ids:
      raise ValueError(f"findings/{index}: duplicate assumption_id")
    observed_ids.add(assumption_id)
    record = records.get(assumption_id)
    if record is None:
      raise ValueError(f"findings/{index}: unknown assumption_id")
    if record["review_state"] != "PENDING":
      raise ValueError(f"findings/{index}: record is already reviewed")
    disposition = finding["disposition"]
    if disposition not in DISPOSITIONS or disposition == "PENDING":
      raise ValueError(f"findings/{index}: invalid reviewed disposition")
    list_fields = (
        "approved_invariants",
        "resolved_parameter_refs",
        "wildcard_expansions",
        "required_changes",
        "blocking_items",
        "evidence_ids",
    )
    if any(not isinstance(finding[field], list) for field in list_fields):
      raise ValueError(f"findings/{index}: review collections must be lists")

    resolved = finding["resolved_parameter_refs"]
    parameter_refs = record["parameter_refs"]
    if len(resolved) != len(set(resolved)) or not set(resolved).issubset(
        set(parameter_refs)
    ):
      raise ValueError(
          f"findings/{index}: resolved_parameter_refs are invalid"
      )
    unresolved = [ref for ref in parameter_refs if ref not in set(resolved)]
    ready = (
        disposition == "ACCEPT"
        and not unresolved
        and not finding["required_changes"]
        and not finding["blocking_items"]
    )
    record.update(
        {
            "review_state": "REVIEWED",
            "disposition": disposition,
            "freeze_gate": "READY" if ready else "BLOCKED",
            "expert_decision": finding["expert_decision"],
            "approved_invariants": finding["approved_invariants"],
            "resolved_parameter_refs": resolved,
            "unresolved_parameter_refs": unresolved,
            "wildcard_expansions": finding["wildcard_expansions"],
            "required_changes": finding["required_changes"],
            "blocking_items": finding["blocking_items"],
            "evidence_ids": finding["evidence_ids"],
            "reviewer_id": findings_payload["reviewer_id"],
            "reviewed_at": findings_payload["reviewed_at"],
        }
    )
  return refresh_derived_fields(payload)


def build_pending_ledger(
    *, repo_root: Path | None = None, base_commit: str | None = None
) -> dict[str, Any]:
  """Build a 61-record pending ledger bound to the validated MG1-A draft."""
  root = (repo_root or repository_root()).resolve()
  draft_path = root / BASE_DRAFT_PATH
  draft = ontology_schema.load_ontology(draft_path).to_dict()
  records = [
      _empty_record(item)
      for item in draft["resolved_contract"]["assumptions"]
  ]
  payload = {
      "schema_version": SCHEMA_VERSION,
      "profile": PROFILE_NAME,
      "milestone": MILESTONE,
      "decision": DECISION_IN_PROGRESS,
      "draft_assessment": "PENDING",
      "freeze_assessment": "NOT_EVALUATED",
      "mg1b_status": "BLOCKED",
      "next_milestone": "MG1-REVIEW",
      "base_draft": {
          "path": BASE_DRAFT_PATH,
          "git_commit": base_commit or _git_head(root),
          "file_sha256": _sha256_file(draft_path),
          "resolved_contract_sha256": draft["resolved_contract_sha256"],
      },
      "reviewer": {
          "reviewer_id": REVIEWER_ID,
          "role": REVIEWER_ROLE,
          "appointment": "EXPLICIT_USER_APPOINTMENT",
          "personal_data_recorded": False,
      },
      "review_policy": {
          "draft_remains_immutable": True,
          "all_assumptions_required": True,
          "all_parameter_refs_must_be_partitioned": True,
          "unresolved_refs_block_freeze": True,
          "wildcard_resolution_requires_expansion": True,
          "defer_or_reject_blocks_freeze": True,
          "causal_prefix_only_required": True,
          "absolute_price_thresholds_forbidden": True,
          "training_and_gpu_deferred": True,
      },
      "records_sha256": ontology_schema.canonical_sha256(records),
      "records": records,
      "summary": _summary(records),
      "execution": {
          "detector_implemented": False,
          "labeler_implemented": False,
          "real_data_labels_created": False,
          "synthetic_samples_created": False,
          "model_training_executed": False,
          "jax_computation_executed": False,
          "gpu_executed": False,
      },
  }
  return payload


def _load_yaml_mapping_snapshot(
    path: Path, *, label: str
) -> tuple[dict[str, Any], bytes]:
  try:
    snapshot = path.read_bytes()
    payload = ontology_schema.loads_yaml_unique(snapshot.decode("utf-8"))
  except (OSError, UnicodeError, RuntimeError) as exc:
    raise ontology_schema.OntologyValidationError(
        [f"<root>: {label} unavailable or invalid: {exc}"]
    ) from exc
  if not isinstance(payload, dict):
    raise ontology_schema.OntologyValidationError(
        ["<root>: expected a mapping"]
    )
  return payload, snapshot


def _load_review_ledger_snapshot(path: Path) -> tuple[dict[str, Any], bytes]:
  return _load_yaml_mapping_snapshot(path, label="review ledger")


def load_review_ledger(path: Path) -> dict[str, Any]:
  """Load a duplicate-key-safe YAML review ledger."""
  return _load_review_ledger_snapshot(path)[0]


def _assert_snapshot_unchanged(
    path: Path, snapshot: bytes, *, label: str
) -> None:
  try:
    observed = path.read_bytes()
  except OSError as exc:
    raise ValueError(f"{label} input unavailable after snapshot: {exc}") from exc
  if observed != snapshot:
    raise ValueError(f"{label} input changed during validation")


def json_schema_errors(
    payload: Any, *, repo_root: Path | None = None
) -> list[str]:
  """Return structural errors for a review ledger."""
  return ontology_schema.json_schema_errors(
      payload, review_schema_path(repo_root)
  )


def _record_metadata(assumption: dict[str, Any]) -> dict[str, Any]:
  return {
      key: assumption[key]
      for key in (
          "assumption_id",
          "category",
          "question",
          "affected_families",
          "parameter_refs",
          "source_evidence",
      )
  }


def _valid_timestamp(value: Any) -> bool:
  if not isinstance(value, str) or not _RFC3339_RE.fullmatch(value):
    return False
  try:
    datetime.fromisoformat(value.replace("Z", "+00:00"))
  except ValueError:
    return False
  return True


def _match_is_explicitly_prohibited(
    clause: str, match: re.Match[str]
) -> bool:
  prefix = clause[: match.start()]
  matched_text = match.group(0)
  suffix = clause[match.end() :]
  return bool(
      _DIRECT_PROHIBITION_PREFIX_RE.search(prefix)
      or _INTERNAL_PROHIBITION_RE.search(matched_text)
      or _DIRECT_PROHIBITION_SUFFIX_RE.search(suffix)
  )


def validate_review_ledger(
    payload: Any, *, repo_root: Path | None = None
) -> list[str]:
  """Return fail-closed semantic errors for the expert review ledger."""
  if not isinstance(payload, dict):
    return ["<root>: expected a mapping"]
  root = (repo_root or repository_root()).resolve()
  errors = list(json_schema_errors(payload, repo_root=root))
  if errors:
    return errors

  if payload.get("schema_version") != SCHEMA_VERSION:
    errors.append(f"schema_version: expected {SCHEMA_VERSION}")
  if payload.get("profile") != PROFILE_NAME:
    errors.append(f"profile: expected {PROFILE_NAME}")
  if payload.get("milestone") != MILESTONE:
    errors.append(f"milestone: expected {MILESTONE}")

  base = payload["base_draft"]
  if base["path"] != BASE_DRAFT_PATH:
    errors.append(f"base_draft/path: expected {BASE_DRAFT_PATH}")
  draft_path = root / BASE_DRAFT_PATH
  try:
    draft = ontology_schema.load_ontology(draft_path).to_dict()
    file_sha256 = _sha256_file(draft_path)
  except (OSError, ontology_schema.OntologyValidationError) as exc:
    return [*errors, f"base_draft: unavailable or invalid ({exc})"]
  if base["file_sha256"] != file_sha256:
    errors.append("base_draft/file_sha256: current draft bytes do not match")
  if base["resolved_contract_sha256"] != draft["resolved_contract_sha256"]:
    errors.append(
        "base_draft/resolved_contract_sha256: current contract does not match"
    )
  if not _COMMIT_RE.fullmatch(base["git_commit"]):
    errors.append("base_draft/git_commit: full lowercase Git commit required")
  else:
    try:
      committed_sha256 = _git_file_sha256(
          root, base["git_commit"], BASE_DRAFT_PATH
      )
    except (OSError, ValueError) as exc:
      errors.append(f"base_draft/git_commit: {exc}")
    else:
      if committed_sha256 != base["file_sha256"]:
        errors.append(
            "base_draft/git_commit: committed draft bytes do not match"
        )
  for field in ("file_sha256", "resolved_contract_sha256"):
    if not _SHA256_RE.fullmatch(base[field]):
      errors.append(f"base_draft/{field}: lowercase SHA256 required")

  assumptions = draft["resolved_contract"]["assumptions"]
  expected_ids = [item["assumption_id"] for item in assumptions]
  records = payload["records"]
  observed_ids = [record["assumption_id"] for record in records]
  if observed_ids != expected_ids:
    errors.append("records: exact draft assumption order and IDs required")

  expected_by_id = {
      item["assumption_id"]: _record_metadata(item) for item in assumptions
  }
  for index, record in enumerate(records):
    path = f"records/{index}"
    assumption_id = record["assumption_id"]
    expected = expected_by_id.get(assumption_id)
    if expected is None:
      errors.append(f"{path}/assumption_id: unknown draft assumption")
      continue
    observed = {key: record[key] for key in expected}
    if observed != expected:
      errors.append(f"{path}: draft metadata must match exactly")

    parameter_refs = record["parameter_refs"]
    resolved = record["resolved_parameter_refs"]
    unresolved = record["unresolved_parameter_refs"]
    if (
        len(resolved) != len(set(resolved))
        or len(unresolved) != len(set(unresolved))
        or set(resolved) & set(unresolved)
        or resolved + unresolved != parameter_refs
    ):
      errors.append(
          f"{path}: resolved/unresolved refs must partition parameter_refs "
          "in source order"
      )

    expansions = record["wildcard_expansions"]
    expansion_refs = [item["parameter_ref"] for item in expansions]
    resolved_wildcards = [ref for ref in resolved if "*" in ref]
    if expansion_refs != resolved_wildcards:
      errors.append(
          f"{path}/wildcard_expansions: every resolved wildcard requires "
          "one ordered expansion"
      )
    for expansion_index, expansion in enumerate(expansions):
      rules = expansion["concrete_rules"]
      if any("*" in rule for rule in rules):
        errors.append(
            f"{path}/wildcard_expansions/{expansion_index}: concrete rules "
            "cannot contain wildcards"
        )

    pending = record["review_state"] == "PENDING"
    if pending:
      if (
          record["disposition"] != "PENDING"
          or record["freeze_gate"] != "BLOCKED"
          or record["expert_decision"] is not None
          or record["reviewer_id"] is not None
          or record["reviewed_at"] is not None
          or resolved
          or unresolved != parameter_refs
          or record["approved_invariants"]
          or record["required_changes"]
          or record["evidence_ids"]
          or record["blocking_items"] != ["REVIEW_NOT_STARTED"]
      ):
        errors.append(f"{path}: pending record fields are inconsistent")
      continue

    if record["disposition"] == "PENDING":
      errors.append(f"{path}/disposition: reviewed record cannot be pending")
    if record["reviewer_id"] != payload["reviewer"]["reviewer_id"]:
      errors.append(f"{path}/reviewer_id: must match ledger reviewer")
    if not _valid_timestamp(record["reviewed_at"]):
      errors.append(f"{path}/reviewed_at: valid RFC3339 timestamp required")
    if not record["expert_decision"]:
      errors.append(f"{path}/expert_decision: reviewed decision required")
    if not record["approved_invariants"]:
      errors.append(f"{path}/approved_invariants: non-empty review required")
    if not record["evidence_ids"]:
      errors.append(f"{path}/evidence_ids: review evidence required")
    if not set(record["evidence_ids"]).issubset(
        set(record["source_evidence"])
    ):
      errors.append(f"{path}/evidence_ids: unknown source evidence")
    review_texts = [
        record["expert_decision"],
        *record["approved_invariants"],
        *record["required_changes"],
    ]
    for text in review_texts:
      if not isinstance(text, str):
        continue
      if _ABSOLUTE_THRESHOLD_RE.search(text):
        errors.append(f"{path}: absolute numeric threshold is forbidden")
      clauses = [
          clause.strip()
          for clause in _CLAUSE_SPLIT_RE.split(text)
          if clause.strip()
      ]
      unsafe_trading = False
      unsafe_future = False
      for clause in clauses:
        trading_match = _TRADING_ACTION_RE.search(clause)
        if trading_match and not _match_is_explicitly_prohibited(
            clause, trading_match
        ):
          unsafe_trading = True
        future_match = _FUTURE_DECISION_RE.search(clause)
        if future_match and not _match_is_explicitly_prohibited(
            clause, future_match
        ):
          unsafe_future = True
      if unsafe_trading:
        errors.append(f"{path}: trading action semantics are forbidden")
      if unsafe_future:
        errors.append(f"{path}: future outcome cannot define ontology state")

    ready = record["freeze_gate"] == "READY"
    expected_ready = (
        record["disposition"] == "ACCEPT"
        and not unresolved
        and not record["required_changes"]
        and not record["blocking_items"]
    )
    if ready != expected_ready:
      errors.append(f"{path}/freeze_gate: inconsistent with review findings")

  expected_records_hash = ontology_schema.canonical_sha256(records)
  if payload["records_sha256"] != expected_records_hash:
    errors.append("records_sha256: does not match canonical records")
  expected_summary = _summary(records)
  if payload["summary"] != expected_summary:
    errors.append("summary: does not match review records")
  expected_decision = _decision(expected_summary)
  if payload["decision"] != expected_decision:
    errors.append(f"decision: expected {expected_decision}")
  expected_assessments = _assessments(expected_summary)
  for field, expected in expected_assessments.items():
    if payload[field] != expected:
      errors.append(f"{field}: expected {expected}")
  return errors


def _render_markdown(payload: dict[str, Any]) -> str:
  summary = payload["summary"]
  base = payload["base_draft"]
  blocker_counts = Counter(
      blocker
      for record in payload["records"]
      for blocker in record["blocking_items"]
  )
  lines = [
      "# MG1 Assumption Review Ledger",
      "",
      f"- Decision: `{payload['decision']}`",
      f"- Draft assessment: `{payload['draft_assessment']}`",
      f"- Freeze assessment: `{payload['freeze_assessment']}`",
      f"- MG1-B: `{payload['mg1b_status']}`",
      f"- Next milestone: `{payload['next_milestone']}`",
      f"- Reviewer: `{payload['reviewer']['reviewer_id']}`",
      f"- Base commit: `{base['git_commit']}`",
      f"- Draft contract SHA256: `{base['resolved_contract_sha256']}`",
      f"- Records SHA256: `{payload['records_sha256']}`",
      f"- Reviewed: `{summary['reviewed']}/{summary['assumptions_total']}`",
      f"- Dispositions: `{summary['revised']} REVISE`, "
      f"`{summary['deferred']} DEFER`, `{summary['accepted']} ACCEPT`",
      f"- Freeze-ready: `{summary['freeze_ready_records']}/"
      f"{summary['assumptions_total']}`",
      f"- Unresolved parameter refs: `{summary['unresolved_parameter_refs']}`",
      "",
      "## Review Records",
      "",
      "| Assumption | State | Disposition | Freeze | Unresolved |",
      "|---|---|---|---|---:|",
  ]
  for record in payload["records"]:
    lines.append(
        "| `{}` | `{}` | `{}` | `{}` | {} |".format(
            record["assumption_id"],
            record["review_state"],
            record["disposition"],
            record["freeze_gate"],
            len(record["unresolved_parameter_refs"]),
        )
    )
  lines.extend(
      [
          "",
          "## Aggregate Blockers",
          "",
          "| Blocker | Records |",
          "|---|---:|",
      ]
  )
  lines.extend(
      f"| `{blocker}` | {count} |"
      for blocker, count in sorted(blocker_counts.items())
  )
  lines.extend(["", "## Findings", ""])
  for record in payload["records"]:
    if record["review_state"] != "REVIEWED":
      continue
    lines.extend(
        [
            f"### {record['assumption_id']}",
            "",
            f"**Disposition:** `{record['disposition']}`; "
            f"**Freeze gate:** `{record['freeze_gate']}`",
            "",
            f"**Question:** {record['question']}",
            "",
            "**Affected families:** "
            + ", ".join(f"`{item}`" for item in record["affected_families"]),
            "",
            "**Parameter refs:** "
            + ", ".join(f"`{item}`" for item in record["parameter_refs"]),
            "",
            "**Evidence:** "
            + ", ".join(f"`{item}`" for item in record["evidence_ids"]),
            "",
            record["expert_decision"],
            "",
            "**Approved invariants**",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in record["approved_invariants"])
    if record["required_changes"]:
      lines.extend(["", "**Required changes**", ""])
      lines.extend(f"- {item}" for item in record["required_changes"])
    if record["blocking_items"]:
      lines.extend(["", "**Blocking items**", ""])
      lines.extend(f"- {item}" for item in record["blocking_items"])
    lines.append("")
  return "\n".join(lines).rstrip() + "\n"


def _write_yaml(
    payload: dict[str, Any],
    path: Path,
    *,
    expected_snapshot: bytes | None = None,
    require_absent: bool = False,
) -> bytes:
  try:
    import yaml
  except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        f"PyYAML is required to write the ledger: {exc}"
    ) from exc
  path.parent.mkdir(parents=True, exist_ok=True)
  text = yaml.safe_dump(
      payload, sort_keys=False, allow_unicode=False, width=100
  )
  encoded = text.encode("utf-8")
  temporary_path: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
      handle.write(text)
      temporary_path = Path(handle.name)
    if require_absent:
      try:
        os.link(temporary_path, path)
      except FileExistsError as exc:
        raise ValueError(
            f"ledger appeared during initialization: {path}"
        ) from exc
    else:
      if expected_snapshot is None:
        raise ValueError("existing ledger write requires an input snapshot")
      _assert_snapshot_unchanged(
          path, expected_snapshot, label="review ledger"
      )
      temporary_path.replace(path)
  finally:
    if temporary_path is not None and temporary_path.exists():
      temporary_path.unlink()
  return encoded


def _atomic_write_text(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path: Path | None = None
  try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
      handle.write(text)
      temporary_path = Path(handle.name)
    temporary_path.replace(path)
  finally:
    if temporary_path is not None and temporary_path.exists():
      temporary_path.unlink()


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Initialize or validate the CPU-only MG1 review ledger"
  )
  parser.add_argument("--ledger", type=Path, default=default_ledger_path())
  parser.add_argument("--initialize", action="store_true")
  parser.add_argument("--apply-findings", type=Path)
  parser.add_argument("--base-commit")
  parser.add_argument("--output-json", type=Path)
  parser.add_argument("--output-md", type=Path)
  parser.add_argument("--strict", action="store_true")
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  root = repository_root()
  output_paths = [
      path.resolve() for path in (args.output_json, args.output_md) if path
  ]
  protected_paths = {
      args.ledger.resolve(),
      (root / BASE_DRAFT_PATH).resolve(),
      review_schema_path(root).resolve(),
      ontology_schema.report_schema_path().resolve(),
      Path(ontology_schema.__file__).resolve(),
      Path(__file__).resolve(),
  }
  if args.apply_findings:
    protected_paths.add(args.apply_findings.resolve())
  if (
      len(output_paths) != len(set(output_paths))
      or not protected_paths.isdisjoint(output_paths)
  ):
    print(
        "MG1 review runtime error: output paths must be distinct and must "
        "not replace validation inputs"
    )
    return 2
  try:
    state_changed = False
    ledger_snapshot: bytes | None = None
    findings_snapshot: bytes | None = None
    initialize_requires_absent = False
    if args.initialize:
      if args.ledger.exists():
        raise ValueError(
            f"refusing to overwrite existing ledger: {args.ledger}"
        )
      payload = build_pending_ledger(
          repo_root=root, base_commit=args.base_commit
      )
      state_changed = True
      initialize_requires_absent = True
    else:
      payload, ledger_snapshot = _load_review_ledger_snapshot(args.ledger)
    if args.apply_findings:
      findings_payload, findings_snapshot = _load_yaml_mapping_snapshot(
          args.apply_findings, label="review findings"
      )
      payload = apply_review_findings(payload, findings_payload)
      state_changed = True
    errors = validate_review_ledger(payload, repo_root=root)
    if ledger_snapshot is not None:
      _assert_snapshot_unchanged(
          args.ledger, ledger_snapshot, label="review ledger"
      )
    elif initialize_requires_absent and args.ledger.exists():
      raise ValueError(f"ledger appeared during initialization: {args.ledger}")
    if findings_snapshot is not None:
      _assert_snapshot_unchanged(
          args.apply_findings, findings_snapshot, label="review findings"
      )
    if state_changed and not errors:
      ledger_snapshot = _write_yaml(
          payload,
          args.ledger,
          expected_snapshot=ledger_snapshot,
          require_absent=initialize_requires_absent,
      )
      initialize_requires_absent = False
    if not errors:
      if args.output_json:
        _atomic_write_text(
            args.output_json,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
      if args.output_md:
        _atomic_write_text(args.output_md, _render_markdown(payload))
    if ledger_snapshot is not None:
      _assert_snapshot_unchanged(
          args.ledger, ledger_snapshot, label="review ledger"
      )
    elif initialize_requires_absent and args.ledger.exists():
      raise ValueError(f"ledger appeared during initialization: {args.ledger}")
    if findings_snapshot is not None:
      _assert_snapshot_unchanged(
          args.apply_findings, findings_snapshot, label="review findings"
      )
  except (
      OSError,
      RuntimeError,
      TypeError,
      ValueError,
      ontology_schema.OntologyValidationError,
  ) as exc:
    print(f"MG1 review runtime error: {type(exc).__name__}: {exc}")
    return 2
  summary = payload.get("summary", {})
  print(
      "MG1 review: "
      f"schema_semantic={'pass' if not errors else 'fail'}, "
      f"reviewed={summary.get('reviewed', 0)}/"
      f"{summary.get('assumptions_total', 0)}, "
      f"freeze_ready={summary.get('freeze_ready_records', 0)}/"
      f"{summary.get('assumptions_total', 0)}, "
      f"decision={payload.get('decision')}"
  )
  if errors:
    for error in errors:
      print(f"- {error}")
  return 1 if errors else 0


if __name__ == "__main__":
  raise SystemExit(main())


__all__ = (
    "BASE_DRAFT_PATH",
    "DECISION_APPROVED",
    "DECISION_IN_PROGRESS",
    "DECISION_NEEDS_REVISION",
    "DISPOSITIONS",
    "FREEZE_GATES",
    "MILESTONE",
    "PROFILE_NAME",
    "REVIEWER_ID",
    "REVIEW_STATES",
    "SCHEMA_VERSION",
    "apply_review_findings",
    "build_pending_ledger",
    "default_ledger_path",
    "json_schema_errors",
    "load_review_ledger",
    "refresh_derived_fields",
    "repository_root",
    "review_schema_path",
    "validate_review_ledger",
)
