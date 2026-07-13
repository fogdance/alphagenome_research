"""CPU-only validation for the non-human MG1-A.1 dual-review bundle."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from alphatrade.market_genome.ontology import revision


SCHEMA_VERSION = "mg1a.simulated_dual_review_bundle.v1"
PROFILE_NAME = "mg1a-simulated-dual-review"
BUNDLE_PATH = (
    "configs/market_genome/reviews/mg1a_v2_ai_simulated/bundle.json"
)
SCHEMA_PATH = (
    "src/alphatrade/market_genome/ontology/"
    "mg1a_simulated_dual_review.schema.json"
)

DISPOSITIONS = ("ACCEPT", "REVISE", "DEFER", "REJECT")
ALIGNMENTS = ("BOTH", "A_ONLY", "B_ONLY", "NEITHER")
EMPIRICAL_DEPENDENCIES = (
    "NONE",
    "DOWNSTREAM_DEFERRED_NOT_ESSENTIAL",
    "ESSENTIAL_EXTERNAL_REGISTRY",
    "ESSENTIAL_TRAIN_FITTED",
    "ESSENTIAL_REAL_DATA_OR_ANNOTATION",
    "ESSENTIAL_HUMAN_IDENTITY_OR_AUTHORITY",
)

_ARTIFACT_PATHS = {
    "reviewer_a": (
        "configs/market_genome/reviews/mg1a_v2_ai_simulated/"
        "reviewer_a.raw.json"
    ),
    "reviewer_b": (
        "configs/market_genome/reviews/mg1a_v2_ai_simulated/"
        "reviewer_b.raw.json"
    ),
    "reviewer_b_attestation": (
        "configs/market_genome/reviews/mg1a_v2_ai_simulated/"
        "reviewer_b.attestation.json"
    ),
    "adjudication_rubric": (
        "configs/market_genome/reviews/mg1a_v2_ai_simulated/"
        "adjudication_rubric.json"
    ),
    "adjudication": (
        "configs/market_genome/reviews/mg1a_v2_ai_simulated/"
        "adjudication.json"
    ),
}

_ARTIFACT_SHA256 = {
    "reviewer_a": (
        "29fb4565aa9ef34f6808207fd2667ffc921c78255626f4e62b8a24395f62eea9"
    ),
    "reviewer_b": (
        "b8af2a7fe318d334289b2dbb03fc6465955a837e703f85ce76ab5df7a11e1457"
    ),
    "reviewer_b_attestation": (
        "ab7d80845b4c8a47d6ed1ac9e853451a7077c339ca9bbf424ef81b43dbd083a5"
    ),
    "adjudication_rubric": (
        "dfbd316f9aa08850c7ecdc6c2e007a99cc18078bfcf3c51c110f4f712ac85d69"
    ),
    "adjudication": (
        "b10638c96338556feaa6c8e1766e19c1baf456390220281ceb792a5b0db07df3"
    ),
}

_TARGET_BINDING = {
    "git_commit": "0d378954d49b6467d05fdef152c522e94b660978",
    "path": "configs/market_genome/pattern_ontology_v2.yaml",
    "file_sha256": (
        "cd569c19a73768b19b8f9ded2b570dc3fc93796a87b97ecd27becdd32ba5978e"
    ),
    "resolved_contract_sha256": (
        "ad3228a608998cc0823c440af0013b12658c9f8dcda4b858ea719be879389ebd"
    ),
    "v1_review_ledger_path": (
        "configs/market_genome/pattern_ontology_v1_review.yaml"
    ),
    "v1_review_ledger_sha256": (
        "a4941b85fd92ada4e58429e2ad0ce0b73d371ac35caa7fe99847a4688c16e90b"
    ),
    "v1_review_records_sha256": (
        "fdf2328d1fb8b63c6c342498ade11f29c47db467a28c17a190a0290cdad8132c"
    ),
}

_DECISIONS_SHA256 = (
    "01a9c889c94ac47a9707bee1cc49d9b09b0d412eb5632f140fa39bfeeabc15d1"
)

_SOURCE_PATHS = (
    "configs/market_genome/pattern_ontology_v1_review.yaml",
    "configs/market_genome/pattern_ontology_v2.yaml",
    "configs/market_genome/reviews/mg1_common_review_v1.yaml",
    "configs/market_genome/reviews/mg1_trend_level_family_review_v1.yaml",
    (
        "configs/market_genome/reviews/"
        "mg1_reversal_conversion_family_review_v1.yaml"
    ),
    "docs/alphaTrade/market_genome/MG1_PATTERN_ONTOLOGY_V2_REVIEW.md",
    "docs/alphaTrade/market_genome/MG1A_V1_TO_V2_DIFF.md",
)

_ADJUDICATION_RECORD_KEYS = {
    "assumption_id",
    "question",
    "generation",
    "reviewer_a_disposition",
    "reviewer_a_rationale",
    "reviewer_b_disposition",
    "reviewer_b_rationale",
    "adjudicated_disposition",
    "component_findings",
    "v1_historical_disposition",
    "v1_blocker_assessments",
    "v2_contract_evidence_paths",
    "validator_or_test_evidence",
    "empirical_dependency",
    "contradictions",
    "rationale",
    "required_changes",
    "evidence_needed_to_resume",
    "replacement_proposition",
    "confidence",
    "human_review_required",
    "freeze_eligible",
    "reviewer_agreement",
    "adjudicator_alignment",
}

_V1_BLOCKER_KEYS = {
    "v1_item_id_or_text",
    "status",
    "v2_evidence_paths",
    "validator_or_test_evidence",
    "explanation",
}

_ADJUDICATION_ROOT_KEYS = {
    "schema_version",
    "generated_at",
    "adjudicator",
    "rubric_binding",
    "input_provenance",
    "decision_policy",
    "summary",
    "decisions",
    "human_review_and_freeze_limitations",
    "execution_boundary",
    "verification",
}

_RUBRIC_ROOT_KEYS = {
    "schema_version",
    "rubric_status",
    "created_at",
    "adjudicator",
    "independence",
    "scope",
    "source_artifacts",
    "observed_baseline",
    "evidence_hierarchy",
    "core_invariants",
    "dispositions",
    "decision_algorithm",
    "v1_to_v2_resolution_policy",
    "execution_deferral_policy",
    "human_identity_and_authority_policy",
    "reviewer_output_ingestion_and_disagreement_policy",
    "per_assumption_output_contract",
    "completion_gate_for_later_70_item_adjudication",
    "freeze_gate",
    "prohibited_reasoning",
    "next_state",
}


class SimulatedReviewValidationError(ValueError):
  """Raised when the simulated-review bundle cannot be loaded."""

  def __init__(self, errors: Iterable[str]):
    self.errors = tuple(errors)
    super().__init__("; ".join(self.errors))


def repository_root() -> Path:
  """Return the repository containing this module."""
  return Path(__file__).resolve().parents[4]


def default_bundle_path(repo_root: Path | None = None) -> Path:
  root = (repo_root or repository_root()).resolve()
  return root / BUNDLE_PATH


def bundle_schema_path(repo_root: Path | None = None) -> Path:
  if repo_root is None:
    return Path(__file__).with_name(
        "mg1a_simulated_dual_review.schema.json"
    )
  return repo_root.resolve() / SCHEMA_PATH


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def canonical_json(value: Any) -> str:
  """Return deterministic ASCII JSON for review-record hashing."""
  return json.dumps(
      value,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=True,
      allow_nan=False,
  )


def canonical_sha256(value: Any) -> str:
  return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def loads_json_unique(text: str) -> Any:
  """Parse JSON while rejecting duplicate object keys."""

  def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise SimulatedReviewValidationError(
            [f"duplicate JSON key: {key}"]
        )
      result[key] = value
    return result

  try:
    return json.loads(text, object_pairs_hook=unique_object)
  except json.JSONDecodeError as exc:
    raise SimulatedReviewValidationError([f"invalid JSON: {exc}"]) from exc


def _load_json_snapshot(path: Path) -> tuple[dict[str, Any], bytes]:
  try:
    snapshot = path.read_bytes()
    payload = loads_json_unique(snapshot.decode("utf-8"))
  except (OSError, UnicodeError) as exc:
    raise SimulatedReviewValidationError(
        [f"JSON artifact unavailable: {exc}"]
    ) from exc
  if not isinstance(payload, dict):
    raise SimulatedReviewValidationError(["<root>: expected an object"])
  return payload, snapshot


def load_bundle(path: Path) -> dict[str, Any]:
  """Load a duplicate-key-safe simulated-review bundle manifest."""
  return _load_json_snapshot(path)[0]


def _format_jsonschema_path(parts: Iterable[Any]) -> str:
  rendered = "/".join(str(part) for part in parts)
  return rendered or "<root>"


def json_schema_errors(
    payload: Any, *, repo_root: Path | None = None
) -> list[str]:
  """Return closed-shape Draft-07 errors for a bundle manifest."""
  try:
    schema_snapshot = bundle_schema_path(repo_root).read_bytes()
    schema = loads_json_unique(schema_snapshot.decode("utf-8"))
    import jsonschema
  except (
      OSError,
      UnicodeError,
      ImportError,
      SimulatedReviewValidationError,
  ) as exc:
    return [f"schema: unavailable or invalid ({exc})"]
  try:
    jsonschema.Draft7Validator.check_schema(schema)
    validator = jsonschema.Draft7Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
  except jsonschema.exceptions.SchemaError as exc:
    return [f"schema: unavailable or invalid ({exc.message})"]
  errors = sorted(
      validator.iter_errors(payload),
      key=lambda item: tuple(str(part) for part in item.absolute_path),
  )
  return [
      f"schema/{_format_jsonschema_path(error.absolute_path)}: "
      f"{error.message}"
      for error in errors
  ]


def _git_file_bytes(root: Path, commit: str, relative: str) -> bytes:
  completed = subprocess.run(
      ["git", "show", f"{commit}:{relative}"],
      cwd=root,
      check=False,
      capture_output=True,
  )
  if completed.returncode:
    raise ValueError(f"{commit}:{relative} is unavailable in Git")
  return completed.stdout


def _counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
  counter = Counter(record[key] for record in records)
  return {name: counter[name] for name in DISPOSITIONS}


def _exact_mapping(
    value: Any, expected_keys: set[str], label: str, errors: list[str]
) -> bool:
  if not isinstance(value, dict):
    errors.append(f"{label}: expected an object")
    return False
  if set(value) != expected_keys:
    errors.append(f"{label}: exact field set required")
    return False
  return True


def _decision_matrix(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
  matrix = {
      left: {right: 0 for right in DISPOSITIONS}
      for left in DISPOSITIONS
  }
  for record in records:
    matrix[record["reviewer_a_disposition"]][
        record["reviewer_b_disposition"]
    ] += 1
  return matrix


def _alignment(a: str, b: str, adjudicated: str) -> str:
  if adjudicated == a == b:
    return "BOTH"
  if adjudicated == a and adjudicated != b:
    return "A_ONLY"
  if adjudicated == b and adjudicated != a:
    return "B_ONLY"
  return "NEITHER"


def _artifact_payloads(
    payload: dict[str, Any], root: Path, errors: list[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
  parsed: dict[str, dict[str, Any]] = {}
  snapshots: dict[str, bytes] = {}
  for name, expected_path in _ARTIFACT_PATHS.items():
    binding = payload["artifacts"][name]
    if binding["path"] != expected_path:
      errors.append(f"artifacts/{name}/path: exact archive path required")
      continue
    path = root / expected_path
    try:
      artifact, snapshot = _load_json_snapshot(path)
    except SimulatedReviewValidationError as exc:
      errors.extend(f"artifacts/{name}: {item}" for item in exc.errors)
      continue
    actual_hash = _sha256_bytes(snapshot)
    if actual_hash != binding["sha256"]:
      errors.append(
          f"artifacts/{name}/sha256: expected {binding['sha256']}, "
          f"observed {actual_hash}"
      )
    parsed[name] = artifact
    snapshots[expected_path] = snapshot
  return parsed, snapshots


def _validate_bundle_metadata(
    payload: dict[str, Any], errors: list[str]
) -> None:
  _exact_mapping(
      payload,
      {
          "schema_version",
          "profile",
          "milestone",
          "review_type",
          "status",
          "created_at",
          "target",
          "artifacts",
          "decisions_sha256",
          "summary",
          "governance",
          "execution",
      },
      "bundle",
      errors,
  )
  expected_scalars = {
      "schema_version": SCHEMA_VERSION,
      "profile": PROFILE_NAME,
      "milestone": "MG1-A.1",
      "review_type": "AI_SIMULATED_NON_HUMAN",
      "status": (
          "REVISION_AND_EXTERNAL_EVIDENCE_REQUIRED_BEFORE_HUMAN_FREEZE"
      ),
  }
  for field, expected in expected_scalars.items():
    if payload.get(field) != expected:
      errors.append(f"bundle/{field}: exact value required")
  target = payload.get("target")
  if _exact_mapping(
      target,
      {
          "git_commit",
          "path",
          "file_sha256",
          "resolved_contract_sha256",
          "v1_review_ledger_path",
          "v1_review_ledger_sha256",
          "v1_review_records_sha256",
      },
      "bundle/target",
      errors,
  ):
    if target["path"] != "configs/market_genome/pattern_ontology_v2.yaml":
      errors.append("bundle/target/path: exact v2 path required")
    if target["v1_review_ledger_path"] != (
        "configs/market_genome/pattern_ontology_v1_review.yaml"
    ):
      errors.append("bundle/target/v1_review_ledger_path: exact path required")
    if target != _TARGET_BINDING:
      errors.append("bundle/target: exact reviewed hash binding required")
  artifacts = payload.get("artifacts")
  if _exact_mapping(
      artifacts, set(_ARTIFACT_PATHS), "bundle/artifacts", errors
  ):
    for name, binding in artifacts.items():
      _exact_mapping(
          binding, {"path", "sha256"}, f"bundle/artifacts/{name}", errors
      )
      if binding.get("sha256") != _ARTIFACT_SHA256[name]:
        errors.append(f"bundle/artifacts/{name}: immutable hash required")
  if payload.get("decisions_sha256") != _DECISIONS_SHA256:
    errors.append("bundle/decisions_sha256: immutable decision hash required")
  expected_governance = {
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
  if payload.get("governance") != expected_governance:
    errors.append("bundle/governance: exact non-human boundary required")
  expected_execution = {
      "gpu_executed": False,
      "jax_computation_executed": False,
      "model_training_executed": False,
      "detector_implemented_or_executed": False,
      "labeler_implemented_or_executed": False,
      "real_data_labels_created": False,
      "synthetic_samples_created": False,
      "trading_rules_implemented": False,
  }
  if payload.get("execution") != expected_execution:
    errors.append("bundle/execution: exact no-execution boundary required")


def _validate_review_input(
    review: dict[str, Any],
    expected_ids: list[str],
    label: str,
    errors: list[str],
) -> list[dict[str, Any]]:
  decisions = review.get("decisions")
  if not isinstance(decisions, list):
    errors.append(f"{label}/decisions: expected a list")
    return []
  if any(not isinstance(item, dict) for item in decisions):
    errors.append(f"{label}/decisions: every record must be an object")
    return []
  ids = [item.get("assumption_id") for item in decisions]
  if ids != expected_ids:
    errors.append(f"{label}/decisions: exact v2 assumption order required")
  if len(ids) != len(set(ids)):
    errors.append(f"{label}/decisions: duplicate assumption_id")
  for index, record in enumerate(decisions):
    if set(record) != {"assumption_id", "decision", "evidence", "blockers"}:
      errors.append(f"{label}/decisions/{index}: exact field set required")
    if record.get("decision") not in DISPOSITIONS:
      errors.append(f"{label}/decisions/{index}/decision: invalid")
    evidence = record.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
      errors.append(f"{label}/decisions/{index}/evidence: required")
    blockers = record.get("blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item for item in blockers
    ):
      errors.append(f"{label}/decisions/{index}/blockers: invalid")
  return decisions


def _validate_reviewer_metadata(
    reviewer_a: dict[str, Any],
    reviewer_b: dict[str, Any],
    target: dict[str, Any],
    errors: list[str],
) -> None:
  a_reviewer = reviewer_a.get("reviewer")
  if _exact_mapping(
      a_reviewer,
      {
          "reviewer_id",
          "reviewer_type",
          "reviewed_commit",
          "review_scope",
          "gpu_used",
          "training_executed",
      },
      "reviewer_a/reviewer",
      errors,
  ):
    expected_a = {
        "reviewer_id": "AI_SIMULATED_REVIEWER_A",
        "reviewer_type": "AI_SIMULATION_NOT_A_HUMAN_DOMAIN_EXPERT",
        "reviewed_commit": target["git_commit"],
        "review_scope": (
            "MG1-A.1 ontology v2, 61 inherited assumptions and "
            "9 new assumptions"
        ),
        "gpu_used": False,
        "training_executed": False,
    }
    if a_reviewer != expected_a:
      errors.append("reviewer_a/reviewer: exact non-human identity required")
  a_independence = reviewer_a.get("independence")
  if _exact_mapping(
      a_independence,
      {"declaration", "limitation", "decision_standard"},
      "reviewer_a/independence",
      errors,
  ):
    expected_limitation = (
        "This is an AI-simulated review and cannot itself satisfy a "
        "governance requirement for two real human domain experts."
    )
    if a_independence["limitation"] != expected_limitation:
      errors.append("reviewer_a/independence: human-review limitation changed")

  b_reviewer = reviewer_b.get("reviewer")
  if _exact_mapping(
      b_reviewer,
      {
          "reviewer_id",
          "reviewer_type",
          "human_reviewer",
          "human_freeze_authority",
      },
      "reviewer_b/reviewer",
      errors,
  ):
    expected_b = {
        "reviewer_id": "AI_SIMULATED_REVIEWER_B",
        "reviewer_type": "AI_SIMULATED_DOMAIN_REVIEW",
        "human_reviewer": False,
        "human_freeze_authority": False,
    }
    if b_reviewer != expected_b:
      errors.append("reviewer_b/reviewer: exact non-human identity required")
  b_independence = reviewer_b.get("independence")
  if _exact_mapping(
      b_independence,
      {
          "statement",
          "reviewer_a_material_consulted",
          "does_not_constitute_human_review",
          "execution_boundary",
          "source_files",
      },
      "reviewer_b/independence",
      errors,
  ):
    if b_independence["reviewer_a_material_consulted"] is not False:
      errors.append("reviewer_b/independence: Reviewer A must not be consulted")
    if b_independence["does_not_constitute_human_review"] is not True:
      errors.append("reviewer_b/independence: non-human declaration required")
    expected_boundary = (
        "Read-only ontology review; no detector, labeler, training, JAX, "
        "synthetic-data, or GPU execution."
    )
    if b_independence["execution_boundary"] != expected_boundary:
      errors.append("reviewer_b/independence: execution boundary changed")
  _exact_mapping(
      reviewer_b.get("decision_policy"),
      set(DISPOSITIONS),
      "reviewer_b/decision_policy",
      errors,
  )


def _validate_reviewer_summaries(
    reviewer_a: dict[str, Any],
    reviewer_b: dict[str, Any],
    a_records: list[dict[str, Any]],
    b_records: list[dict[str, Any]],
    errors: list[str],
) -> None:
  a_counts = _counts(a_records, "decision")
  a_summary = reviewer_a.get("summary")
  if _exact_mapping(
      a_summary,
      {
          "total",
          "counts",
          "freeze_ready",
          "freeze_blocked",
          "overall_decision",
          "mg1b_decision",
          "highest_priority_blockers",
      },
      "reviewer_a/summary",
      errors,
  ):
    expected_a_fields = {
        "total": 70,
        "counts": a_counts,
        "freeze_ready": a_counts["ACCEPT"],
        "freeze_blocked": 70 - a_counts["ACCEPT"],
        "overall_decision": "REVISION_REQUIRED_BEFORE_FREEZE",
        "mg1b_decision": "BLOCKED",
    }
    for field, expected in expected_a_fields.items():
      if a_summary[field] != expected:
        errors.append(f"reviewer_a/summary/{field}: mismatch")
    blockers = a_summary["highest_priority_blockers"]
    if not isinstance(blockers, list) or not blockers:
      errors.append("reviewer_a/summary/highest_priority_blockers: required")

  b_counts = _counts(b_records, "decision")
  b_summary = reviewer_b.get("summary")
  if _exact_mapping(
      b_summary,
      {
          "assumptions_total",
          "ACCEPT",
          "REVISE",
          "DEFER",
          "REJECT",
          "decisions_recorded",
          "freeze_ready",
          "freeze_recommendation",
          "mg1b_recommendation",
          "blocking_themes",
      },
      "reviewer_b/summary",
      errors,
  ):
    expected_b_fields = {
        "assumptions_total": 70,
        **b_counts,
        "decisions_recorded": 70,
        "freeze_ready": False,
        "freeze_recommendation": "DO_NOT_FREEZE",
        "mg1b_recommendation": "REMAIN_BLOCKED",
    }
    for field, expected in expected_b_fields.items():
      if b_summary[field] != expected:
        errors.append(f"reviewer_b/summary/{field}: mismatch")
    themes = b_summary["blocking_themes"]
    if not isinstance(themes, list) or not themes:
      errors.append("reviewer_b/summary/blocking_themes: required")


def _validate_target(
    payload: dict[str, Any], root: Path, errors: list[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
  target = payload["target"]
  snapshots: dict[str, bytes] = {}
  v2_path = root / target["path"]
  ledger_path = root / target["v1_review_ledger_path"]
  try:
    v2_snapshot = v2_path.read_bytes()
    ledger_snapshot = ledger_path.read_bytes()
    snapshots[target["path"]] = v2_snapshot
    snapshots[target["v1_review_ledger_path"]] = ledger_snapshot
    v2 = revision.loads_yaml_unique(v2_snapshot.decode("utf-8"))
    ledger = revision.loads_yaml_unique(ledger_snapshot.decode("utf-8"))
  except (OSError, UnicodeError, revision.RevisionValidationError) as exc:
    errors.append(f"target: unavailable or invalid ({exc})")
    return {}, {}, snapshots

  if _sha256_bytes(v2_snapshot) != target["file_sha256"]:
    errors.append("target/file_sha256: current v2 bytes do not match")
  if _sha256_bytes(ledger_snapshot) != target["v1_review_ledger_sha256"]:
    errors.append("target/v1_review_ledger_sha256: current bytes do not match")
  if canonical_sha256(v2["resolved_contract"]) != target[
      "resolved_contract_sha256"
  ]:
    errors.append("target/resolved_contract_sha256: semantic hash mismatch")
  if canonical_sha256(ledger["records"]) != target[
      "v1_review_records_sha256"
  ]:
    errors.append("target/v1_review_records_sha256: semantic hash mismatch")

  try:
    committed_v2 = _git_file_bytes(
        root, target["git_commit"], target["path"]
    )
    committed_ledger = _git_file_bytes(
        root, target["git_commit"], target["v1_review_ledger_path"]
    )
  except ValueError as exc:
    errors.append(f"target/git_commit: {exc}")
  else:
    if _sha256_bytes(committed_v2) != target["file_sha256"]:
      errors.append("target/git_commit: committed v2 hash mismatch")
    if _sha256_bytes(committed_ledger) != target[
        "v1_review_ledger_sha256"
    ]:
      errors.append("target/git_commit: committed v1 ledger hash mismatch")

  expected_human_review = {
      "status": "PENDING",
      "frozen": False,
      "reviewer_decisions_recorded": False,
      "approver": None,
      "approved_at": None,
      "frozen_sha256": None,
  }
  if v2.get("decision") != revision.DECISION:
    errors.append("target/decision: v2 must remain DRAFT_NEEDS_HUMAN_FREEZE")
  if v2.get("human_review") != expected_human_review:
    errors.append("target/human_review: pending non-frozen state required")
  if v2.get("review_outcome_v1", {}).get("mg1b_status") != "BLOCKED":
    errors.append("target/review_outcome_v1/mg1b_status: must remain BLOCKED")
  if v2.get("resolved_contract", {}).get("review_protocol", {}).get(
      "model_assisted_independent_truth_allowed"
  ) is not False:
    errors.append("target/review_protocol: model-assisted truth must be false")

  revision_errors = revision.validate_revision_payload(v2, repo_root=root)
  errors.extend(f"target/v2_validation: {item}" for item in revision_errors)
  return v2, ledger, snapshots


def _validate_attestation(
    attestation: dict[str, Any],
    reviewer_b: dict[str, Any],
    payload: dict[str, Any],
    errors: list[str],
) -> None:
  b_binding = payload["artifacts"]["reviewer_b"]
  target = payload["target"]
  _exact_mapping(
      attestation,
      {
          "schema_version",
          "attestation_id",
          "attested_at",
          "reviewer",
          "supplement_of",
          "review_target",
          "source_file_attestations",
          "repository_evidence",
          "declarations",
      },
      "reviewer_b_attestation",
      errors,
  )
  supplement = attestation.get("supplement_of", {})
  _exact_mapping(
      supplement,
      {
          "path",
          "sha256",
          "original_bytes_unchanged",
          "decisions_unchanged",
          "decision_count",
          "decision_summary",
      },
      "reviewer_b_attestation/supplement_of",
      errors,
  )
  if supplement.get("sha256") != b_binding["sha256"]:
    errors.append("reviewer_b_attestation: raw B hash mismatch")
  if supplement.get("path") != "/tmp/mg1_dual_review_b.json":
    errors.append("reviewer_b_attestation: original path changed")
  if supplement.get("original_bytes_unchanged") is not True:
    errors.append("reviewer_b_attestation: original bytes must remain unchanged")
  if supplement.get("decisions_unchanged") is not True:
    errors.append("reviewer_b_attestation: decisions must remain unchanged")
  if supplement.get("decision_count") != 70:
    errors.append("reviewer_b_attestation: decision_count must be 70")
  if attestation.get("review_target") != {
      "reviewed_commit": target["git_commit"],
      "v2_path": target["path"],
      "v2_file_sha256": target["file_sha256"],
      "resolved_contract_sha256": target["resolved_contract_sha256"],
  }:
    errors.append("reviewer_b_attestation: target binding mismatch")
  reviewer = attestation.get("reviewer", {})
  _exact_mapping(
      reviewer,
      {
          "reviewer_id",
          "reviewer_type",
          "human_reviewer",
          "human_freeze_authority",
      },
      "reviewer_b_attestation/reviewer",
      errors,
  )
  if reviewer != {
      "reviewer_id": "AI_SIMULATED_REVIEWER_B",
      "reviewer_type": "AI_SIMULATED_DOMAIN_REVIEW",
      "human_reviewer": False,
      "human_freeze_authority": False,
  }:
    errors.append("reviewer_b_attestation: exact non-human identity required")
  declarations = attestation.get("declarations", {})
  expected_declarations = {
      "reviewer_a_material_consulted": False,
      "independent_review_statement_unchanged": True,
      "does_not_constitute_human_review": True,
      "does_not_authorize_freeze": True,
      "does_not_authorize_mg1b": True,
      "no_decision_was_added_removed_or_modified": True,
      "source_files_are_exactly_the_files_read_during_review": True,
      "repository_files_modified_by_this_attestation": False,
      "gpu_or_training_executed": False,
  }
  if declarations != expected_declarations:
    errors.append("reviewer_b_attestation/declarations: exact boundary required")
  repository_evidence = attestation.get("repository_evidence")
  expected_repository_evidence = {
      "head_at_attestation": target["git_commit"],
      "head_matches_reviewed_commit": True,
      "source_files_have_no_working_tree_diff": True,
      "working_tree_clean_at_attestation": True,
  }
  if repository_evidence != expected_repository_evidence:
    errors.append("reviewer_b_attestation/repository_evidence: mismatch")
  expected_b_counts = _counts(reviewer_b["decisions"], "decision")
  if supplement.get("decision_summary") != expected_b_counts:
    errors.append("reviewer_b_attestation: decision summary mismatch")


def _validate_rubric(
    rubric: dict[str, Any], payload: dict[str, Any], errors: list[str]
) -> None:
  _exact_mapping(
      rubric, _RUBRIC_ROOT_KEYS, "adjudication_rubric", errors
  )
  if rubric.get("rubric_status") != "LOCKED_BEFORE_REVIEWER_OUTPUT_ACCESS":
    errors.append("adjudication_rubric: pre-review lock required")
  independence = rubric.get("independence", {})
  if _exact_mapping(
      independence,
      {
          "reviewer_a_output_seen",
          "reviewer_b_output_seen",
          "reviewer_output_paths_searched",
          "rubric_fixed_before_reviewer_output_access",
          "post_lock_policy",
      },
      "adjudication_rubric/independence",
      errors,
  ):
    if any(
        independence[field] is not False
        for field in (
            "reviewer_a_output_seen",
            "reviewer_b_output_seen",
            "reviewer_output_paths_searched",
        )
    ):
      errors.append("adjudication_rubric: reviewer output was seen before lock")
    if independence["rubric_fixed_before_reviewer_output_access"] is not True:
      errors.append("adjudication_rubric: rubric was not fixed before access")
  adjudicator = rubric.get("adjudicator", {})
  expected_adjudicator = {
      "adjudicator_id": "AI_SIMULATED_ADJUDICATOR",
      "identity_type": "AI_SIMULATED_ADJUDICATOR",
      "authority": "NON_BINDING_TECHNICAL_ADJUDICATION_ONLY",
      "human_identity_verified": False,
      "personal_data_recorded": False,
      "may_issue_simulated_technical_dispositions": True,
      "may_count_as_human_reviewer": False,
      "may_approve_or_freeze_ontology": False,
      "may_unblock_mg1b": False,
      "may_write_human_review_or_freeze_fields": False,
  }
  if adjudicator != expected_adjudicator:
    errors.append("adjudication_rubric/adjudicator: exact AI authority required")
  scope = rubric.get("scope", {})
  expected_scope_fields = {
      "milestone": "MG1-A.1",
      "target_artifact": payload["target"]["path"],
      "target_decision": "DRAFT_NEEDS_HUMAN_FREEZE",
      "assumption_total": 70,
      "inherited_v1_assumptions": 61,
      "new_v2_assumptions": 9,
      "currently_frozen_assumptions": 0,
      "mg1b_status": "BLOCKED",
      "adjudication_scope": "Symbolic ontology and governance contract only",
  }
  if not _exact_mapping(
      scope,
      set(expected_scope_fields) | {"out_of_scope"},
      "adjudication_rubric/scope",
      errors,
  ) or any(scope.get(key) != value for key, value in expected_scope_fields.items()):
    errors.append("adjudication_rubric/scope: exact blocked scope required")
  output_contract = rubric.get("per_assumption_output_contract", {})
  _exact_mapping(
      output_contract,
      {
          "required_fields",
          "generation_values",
          "disposition_values",
          "empirical_dependency_values",
          "confidence_values",
          "constant_fields_for_this_simulation",
      },
      "adjudication_rubric/per_assumption_output_contract",
      errors,
  )
  constants = output_contract.get("constant_fields_for_this_simulation")
  if constants != {"human_review_required": True, "freeze_eligible": False}:
    errors.append("adjudication_rubric: per-assumption freeze constants changed")
  freeze_gate = rubric.get("freeze_gate", {})
  _exact_mapping(
      freeze_gate,
      {
          "current_status",
          "simulated_results_can_freeze",
          "all_non_accept_technical_dispositions_block_freeze",
          "even_all_simulated_accepts_are_insufficient",
          "required_before_real_freeze",
          "fields_that_must_not_be_changed_by_this_adjudication",
      },
      "adjudication_rubric/freeze_gate",
      errors,
  )
  if freeze_gate.get("current_status") != "BLOCKED":
    errors.append("adjudication_rubric: freeze gate must remain BLOCKED")
  if freeze_gate.get("simulated_results_can_freeze") is not False:
    errors.append("adjudication_rubric: simulated results cannot freeze")
  if freeze_gate.get("even_all_simulated_accepts_are_insufficient") is not True:
    errors.append("adjudication_rubric: human freeze boundary changed")
  execution = rubric.get("execution_deferral_policy", {})
  if _exact_mapping(
      execution,
      {
          "gpu_allowed_for_adjudication",
          "training_allowed_for_adjudication",
          "detector_or_labeler_execution_required",
          "symbolic_review_requires_gpu",
          "rules",
      },
      "adjudication_rubric/execution_deferral_policy",
      errors,
  ):
    for field in (
        "gpu_allowed_for_adjudication",
        "training_allowed_for_adjudication",
        "detector_or_labeler_execution_required",
        "symbolic_review_requires_gpu",
    ):
      if execution[field] is not False:
        errors.append(f"adjudication_rubric/{field}: false required")
  target_source = next(
      (
          item
          for item in rubric.get("source_artifacts", [])
          if item.get("source_id") == "V2_ONTOLOGY"
      ),
      {},
  )
  target = payload["target"]
  if target_source.get("file_sha256") != target["file_sha256"]:
    errors.append("adjudication_rubric: v2 file hash mismatch")
  if target_source.get("embedded_resolved_contract_sha256") != target[
      "resolved_contract_sha256"
  ]:
    errors.append("adjudication_rubric: v2 contract hash mismatch")


def _validate_source_evidence(
    reviewer_a: dict[str, Any],
    reviewer_b: dict[str, Any],
    attestation: dict[str, Any],
    adjudication: dict[str, Any],
    payload: dict[str, Any],
    root: Path,
    snapshots: dict[str, bytes],
    errors: list[str],
) -> None:
  evidence_snapshot = reviewer_a.get("evidence_snapshot", {})
  attestations = attestation.get("source_file_attestations", [])
  attestation_by_path = {item.get("path"): item for item in attestations}
  if tuple(evidence_snapshot) != _SOURCE_PATHS:
    errors.append("source_evidence: exact source path order required")
  if set(attestation_by_path) != set(evidence_snapshot):
    errors.append("source_evidence: Reviewer A/B source path sets differ")
  b_sources = reviewer_b.get("independence", {}).get("source_files")
  if b_sources != list(evidence_snapshot):
    errors.append("source_evidence: Reviewer B source order differs")
  provenance_snapshot = adjudication.get("input_provenance", {}).get(
      "reviewer_a", {}
  ).get("evidence_snapshot")
  if provenance_snapshot != evidence_snapshot:
    errors.append("source_evidence: adjudication snapshot differs from A")

  commit = payload["target"]["git_commit"]
  for relative, expected_hash in evidence_snapshot.items():
    if relative not in _SOURCE_PATHS:
      continue
    try:
      snapshot = (root / relative).read_bytes()
      committed = _git_file_bytes(root, commit, relative)
    except (OSError, ValueError) as exc:
      errors.append(f"source_evidence/{relative}: unavailable ({exc})")
      continue
    snapshots[relative] = snapshot
    if _sha256_bytes(snapshot) != expected_hash:
      errors.append(f"source_evidence/{relative}: current hash mismatch")
    if _sha256_bytes(committed) != expected_hash:
      errors.append(f"source_evidence/{relative}: commit hash mismatch")
    item = attestation_by_path.get(relative, {})
    expected_item = {
        "path": relative,
        "sha256_at_review": expected_hash,
        "sha256_at_reviewed_commit": expected_hash,
        "read_during_review": True,
        "working_tree_matches_reviewed_commit": True,
    }
    if item != expected_item:
      errors.append(f"source_evidence/{relative}: B attestation mismatch")


def _validate_adjudication_metadata(
    adjudication: dict[str, Any],
    payload: dict[str, Any],
    errors: list[str],
) -> None:
  _exact_mapping(
      adjudication, _ADJUDICATION_ROOT_KEYS, "adjudication", errors
  )
  expected_adjudicator = {
      "adjudicator_id": "AI_SIMULATED_ADJUDICATOR",
      "identity_type": "AI_SIMULATED_ADJUDICATOR",
      "authority": "NON_BINDING_TECHNICAL_ADJUDICATION_ONLY",
      "human_reviewer": False,
      "human_freeze_authority": False,
  }
  if adjudication.get("adjudicator") != expected_adjudicator:
    errors.append("adjudication/adjudicator: exact non-human authority required")

  expected_rubric_binding = {
      "path": "/tmp/mg1_adjudication_rubric.json",
      "sha256": payload["artifacts"]["adjudication_rubric"]["sha256"],
      "status": "LOCKED_BEFORE_REVIEWER_OUTPUT_ACCESS",
      "modified_after_reviewer_access": False,
  }
  if adjudication.get("rubric_binding") != expected_rubric_binding:
    errors.append("adjudication/rubric_binding: exact lock binding required")

  expected_policy = {
      "no_majority_vote": True,
      "a_b_agreement_rechecked_against_evidence": True,
      "symbolic_contract_gap_is_revise_not_gpu_defer": True,
      "essential_registry_or_train_fitted_evidence_can_defer": True,
      "v1_disposition_not_inherited_automatically": True,
  }
  if adjudication.get("decision_policy") != expected_policy:
    errors.append("adjudication/decision_policy: exact evidence policy required")

  expected_limitations = {
      "all_results_non_binding": True,
      "all_assumptions_human_review_required": True,
      "simulated_accept_is_not_human_accept": True,
      "two_independent_real_human_reviews_still_required": True,
      "hash_bound_human_adjudication_still_required": True,
      "v2_fields_modified": False,
      "v2_remains": "DRAFT_NEEDS_HUMAN_FREEZE",
      "mg1b_remains": "BLOCKED",
  }
  if adjudication.get("human_review_and_freeze_limitations") != (
      expected_limitations
  ):
    errors.append("adjudication: exact human-freeze limitations required")

  expected_execution = {
      "gpu_used": False,
      "jax_executed": False,
      "training_executed": False,
      "detector_or_labeler_executed": False,
      "cpu_contract_tests_only": True,
  }
  if adjudication.get("execution_boundary") != expected_execution:
    errors.append("adjudication/execution_boundary: exact CPU-only boundary required")

  expected_verification = {
      "input_hashes": "PASS",
      "reviewer_b_raw_to_attestation_binding": "PASS",
      "review_target_commit_and_canonical_hash": "PASS",
      "cpu_targeted_tests": "53 passed in 21.50s",
      "assumption_id_coverage": "70 unique IDs: 61 inherited and 9 new",
      "per_assumption_contract": "PASS",
      "completion_gate": "PASS",
      "repository_files_modified_by_generator": False,
  }
  if adjudication.get("verification") != expected_verification:
    errors.append("adjudication/verification: exact recorded evidence required")

  provenance = adjudication.get("input_provenance")
  if not _exact_mapping(
      provenance,
      {"reviewer_a", "reviewer_b", "review_target"},
      "adjudication/input_provenance",
      errors,
  ):
    return
  expected_a = {
      "path": "/tmp/mg1_dual_review_a.json",
      "sha256": payload["artifacts"]["reviewer_a"]["sha256"],
      "reviewer_id": "AI_SIMULATED_REVIEWER_A",
      "reviewed_commit": payload["target"]["git_commit"],
  }
  a_provenance = provenance["reviewer_a"]
  if not _exact_mapping(
      a_provenance,
      set(expected_a) | {"evidence_snapshot"},
      "adjudication/input_provenance/reviewer_a",
      errors,
  ) or any(a_provenance.get(key) != value for key, value in expected_a.items()):
    errors.append("adjudication/input_provenance/reviewer_a: mismatch")
  expected_b = {
      "raw_path": "/tmp/mg1_dual_review_b.json",
      "raw_sha256": payload["artifacts"]["reviewer_b"]["sha256"],
      "attestation_path": "/tmp/mg1_dual_review_b_attestation.json",
      "attestation_sha256": payload["artifacts"]["reviewer_b_attestation"][
          "sha256"
      ],
      "reviewer_id": "AI_SIMULATED_REVIEWER_B",
      "reviewed_commit": payload["target"]["git_commit"],
      "raw_decisions_unchanged_by_attestation": True,
  }
  if provenance["reviewer_b"] != expected_b:
    errors.append("adjudication/input_provenance/reviewer_b: mismatch")
  target = payload["target"]
  expected_target = {
      "commit": target["git_commit"],
      "v2_path": target["path"],
      "v2_file_sha256": target["file_sha256"],
      "v2_resolved_contract_sha256": target["resolved_contract_sha256"],
      "v1_review_ledger_path": target["v1_review_ledger_path"],
      "v1_review_ledger_sha256": target["v1_review_ledger_sha256"],
      "v1_review_records_sha256": target["v1_review_records_sha256"],
  }
  if provenance["review_target"] != expected_target:
    errors.append("adjudication/input_provenance/review_target: mismatch")


def _validate_adjudication_records(
    adjudication: dict[str, Any],
    reviewer_a: dict[str, Any],
    reviewer_b: dict[str, Any],
    v2: dict[str, Any],
    ledger: dict[str, Any],
    payload: dict[str, Any],
    errors: list[str],
) -> None:
  registry = v2["resolved_contract"]["assumption_registry"]
  inherited_ids = list(registry["inherited_from_v1"]["assumption_ids"])
  new_records = list(registry["new_v2_assumptions"])
  new_ids = [item["assumption_id"] for item in new_records]
  expected_ids = inherited_ids + new_ids

  a_records = _validate_review_input(
      reviewer_a, expected_ids, "reviewer_a", errors
  )
  b_records = _validate_review_input(
      reviewer_b, expected_ids, "reviewer_b", errors
  )
  decisions = adjudication.get("decisions")
  if not isinstance(decisions, list) or len(decisions) != 70:
    errors.append("adjudication/decisions: exactly 70 records required")
    return
  if any(not isinstance(item, dict) for item in decisions):
    errors.append("adjudication/decisions: every record must be an object")
    return
  if [item.get("assumption_id") for item in decisions] != expected_ids:
    errors.append("adjudication/decisions: exact v2 assumption order required")
  if len(a_records) != 70 or len(b_records) != 70:
    return
  _validate_reviewer_summaries(
      reviewer_a, reviewer_b, a_records, b_records, errors
  )

  ledger_by_id = {item["assumption_id"]: item for item in ledger["records"]}
  new_by_id = {item["assumption_id"]: item for item in new_records}
  for index, (record, a_record, b_record) in enumerate(
      zip(decisions, a_records, b_records)
  ):
    prefix = f"adjudication/decisions/{index}"
    if set(record) != _ADJUDICATION_RECORD_KEYS:
      errors.append(f"{prefix}: exact 24-field record shape required")
      continue
    assumption_id = record["assumption_id"]
    inherited = assumption_id in ledger_by_id
    source = ledger_by_id.get(assumption_id) or new_by_id[assumption_id]
    expected_generation = "INHERITED_V1" if inherited else "NEW_V2"
    if record["generation"] != expected_generation:
      errors.append(f"{prefix}/generation: mismatch")
    if record["question"] != source["question"]:
      errors.append(f"{prefix}/question: source question mismatch")

    a_disposition = a_record["decision"]
    b_disposition = b_record["decision"]
    adjudicated = record["adjudicated_disposition"]
    if record["reviewer_a_disposition"] != a_disposition:
      errors.append(f"{prefix}/reviewer_a_disposition: raw input mismatch")
    if record["reviewer_b_disposition"] != b_disposition:
      errors.append(f"{prefix}/reviewer_b_disposition: raw input mismatch")
    if record["reviewer_a_rationale"] != {
        "evidence": a_record["evidence"],
        "blockers": a_record["blockers"],
    }:
      errors.append(f"{prefix}/reviewer_a_rationale: raw input mismatch")
    if record["reviewer_b_rationale"] != {
        "evidence": b_record["evidence"],
        "blockers": b_record["blockers"],
    }:
      errors.append(f"{prefix}/reviewer_b_rationale: raw input mismatch")
    if adjudicated not in DISPOSITIONS:
      errors.append(f"{prefix}/adjudicated_disposition: invalid")
      continue
    expected_agreement = (
        "A_B_SAME" if a_disposition == b_disposition else "A_B_DIFFERENT"
    )
    if record["reviewer_agreement"] != expected_agreement:
      errors.append(f"{prefix}/reviewer_agreement: mismatch")
    expected_alignment = _alignment(
        a_disposition, b_disposition, adjudicated
    )
    if record["adjudicator_alignment"] != expected_alignment:
      errors.append(f"{prefix}/adjudicator_alignment: mismatch")

    expected_v1_disposition = source["disposition"] if inherited else None
    if record["v1_historical_disposition"] != expected_v1_disposition:
      errors.append(f"{prefix}/v1_historical_disposition: mismatch")
    blocker_assessments = record["v1_blocker_assessments"]
    if not isinstance(blocker_assessments, list) or any(
        not isinstance(item, dict) for item in blocker_assessments
    ):
      errors.append(f"{prefix}/v1_blocker_assessments: invalid collection")
      continue
    expected_blockers = source["blocking_items"] if inherited else []
    if [item.get("v1_item_id_or_text") for item in blocker_assessments] != (
        expected_blockers
    ):
      errors.append(f"{prefix}/v1_blocker_assessments: exact coverage required")
    for blocker_index, blocker in enumerate(blocker_assessments):
      if set(blocker) != _V1_BLOCKER_KEYS:
        errors.append(
            f"{prefix}/v1_blocker_assessments/{blocker_index}: invalid shape"
        )
      if blocker.get("status") not in {
          "RESOLVED",
          "PARTIALLY_RESOLVED",
          "OPEN",
          "NOT_APPLICABLE",
      }:
        errors.append(
            f"{prefix}/v1_blocker_assessments/{blocker_index}/status: invalid"
        )
      for field in ("v2_evidence_paths", "validator_or_test_evidence"):
        values = blocker.get(field)
        if not isinstance(values, list) or any(
            not isinstance(item, str) or not item for item in values
        ):
          errors.append(
              f"{prefix}/v1_blocker_assessments/{blocker_index}/"
              f"{field}: invalid"
          )
      if not isinstance(blocker.get("explanation"), str) or not blocker[
          "explanation"
      ]:
        errors.append(
            f"{prefix}/v1_blocker_assessments/{blocker_index}/"
            "explanation: required"
        )

    list_fields = (
        "component_findings",
        "v1_blocker_assessments",
        "v2_contract_evidence_paths",
        "validator_or_test_evidence",
        "contradictions",
        "required_changes",
        "evidence_needed_to_resume",
        "replacement_proposition",
    )
    if any(not isinstance(record[field], list) for field in list_fields):
      errors.append(f"{prefix}: adjudication collections must be lists")
      continue
    if not record["component_findings"]:
      errors.append(f"{prefix}/component_findings: required")
    for component_index, component in enumerate(record["component_findings"]):
      if not isinstance(component, dict) or set(component) != {
          "component",
          "status",
          "finding",
          "evidence_paths",
      }:
        errors.append(
            f"{prefix}/component_findings/{component_index}: invalid shape"
        )
        continue
      if component["status"] not in {"COMPLETE", "PARTIAL", "DEFERRED"}:
        errors.append(
            f"{prefix}/component_findings/{component_index}/status: invalid"
        )
      if any(
          not isinstance(component[field], str) or not component[field]
          for field in ("component", "finding")
      ):
        errors.append(
            f"{prefix}/component_findings/{component_index}: text required"
        )
      paths = component["evidence_paths"]
      if not isinstance(paths, list) or any(
          not isinstance(item, str) or not item for item in paths
      ):
        errors.append(
            f"{prefix}/component_findings/{component_index}/"
            "evidence_paths: invalid"
        )
    for field in (
        "v2_contract_evidence_paths",
        "validator_or_test_evidence",
        "contradictions",
        "required_changes",
        "evidence_needed_to_resume",
        "replacement_proposition",
    ):
      if any(not isinstance(item, str) or not item for item in record[field]):
        errors.append(f"{prefix}/{field}: non-empty strings required")
    if not record["v2_contract_evidence_paths"]:
      errors.append(f"{prefix}/v2_contract_evidence_paths: required")
    if not record["validator_or_test_evidence"]:
      errors.append(f"{prefix}/validator_or_test_evidence: required")
    if not isinstance(record["rationale"], str) or not record["rationale"]:
      errors.append(f"{prefix}/rationale: required")
    if record["confidence"] not in {"HIGH", "MEDIUM", "LOW"}:
      errors.append(f"{prefix}/confidence: invalid")
    if record["empirical_dependency"] not in EMPIRICAL_DEPENDENCIES:
      errors.append(f"{prefix}/empirical_dependency: invalid")
    if record["human_review_required"] is not True:
      errors.append(f"{prefix}/human_review_required: true required")
    if record["freeze_eligible"] is not False:
      errors.append(f"{prefix}/freeze_eligible: false required")

    if adjudicated == "ACCEPT":
      if any(
          record[field]
          for field in (
              "contradictions",
              "required_changes",
              "evidence_needed_to_resume",
              "replacement_proposition",
          )
      ):
        errors.append(f"{prefix}: ACCEPT cannot retain blocking work")
      if any(
          item["status"] not in {"RESOLVED", "NOT_APPLICABLE"}
          for item in blocker_assessments
      ):
        errors.append(f"{prefix}: ACCEPT has an unresolved v1 blocker")
    elif adjudicated == "REVISE" and not record["required_changes"]:
      errors.append(f"{prefix}/required_changes: required for REVISE")
    elif adjudicated == "DEFER" and not record["evidence_needed_to_resume"]:
      errors.append(f"{prefix}/evidence_needed_to_resume: required for DEFER")
    elif adjudicated == "REJECT":
      if not record["contradictions"] or not record["replacement_proposition"]:
        errors.append(f"{prefix}: REJECT needs contradiction and replacement")

  if canonical_sha256(decisions) != payload["decisions_sha256"]:
    errors.append("decisions_sha256: canonical decision hash mismatch")

  a_counts = _counts(decisions, "reviewer_a_disposition")
  b_counts = _counts(decisions, "reviewer_b_disposition")
  adjudicated_counts = _counts(decisions, "adjudicated_disposition")
  agreement_count = sum(
      item["reviewer_a_disposition"] == item["reviewer_b_disposition"]
      for item in decisions
  )
  alignment_counter = Counter(
      item["adjudicator_alignment"] for item in decisions
  )
  expected_summary = {
      "assumptions_total": 70,
      "inherited_v1": 61,
      "new_v2": 9,
      "reviewer_a_counts": a_counts,
      "reviewer_b_counts": b_counts,
      "adjudicated_counts": adjudicated_counts,
      "a_b_agreement_count": agreement_count,
      "a_b_disagreement_count": 70 - agreement_count,
      "decision_matrix_rows_a_columns_b": _decision_matrix(decisions),
      "adjudicator_alignment_counts": {
          name: alignment_counter[name] for name in ALIGNMENTS
      },
      "freeze_eligible_count": sum(
          item["freeze_eligible"] for item in decisions
      ),
  }
  if payload["summary"] != expected_summary:
    errors.append("summary: does not match adjudication records")

  adjudication_summary = adjudication.get("summary", {})
  comparable = {
      "assumption_total": 70,
      "inherited_v1": 61,
      "new_v2": 9,
      "reviewer_a_counts": a_counts,
      "reviewer_b_counts": b_counts,
      "adjudicated_counts": adjudicated_counts,
      "a_b_agreement_count": agreement_count,
      "a_b_disagreement_count": 70 - agreement_count,
      "adjudicator_alignment_counts": expected_summary[
          "adjudicator_alignment_counts"
      ],
      "technical_accept_count": adjudicated_counts["ACCEPT"],
      "freeze_eligible_count": 0,
      "overall_decision": payload["status"],
      "mg1b_decision": "REMAIN_BLOCKED",
  }
  if adjudication_summary != comparable:
    errors.append("adjudication/summary: does not match records")


def _validate_bundle_payload(
    payload: dict[str, Any], *, repo_root: Path | None = None
) -> list[str]:
  """Validate a schema-conformant bundle and its bound artifacts."""
  root = (repo_root or repository_root()).resolve()
  errors: list[str] = []
  _validate_bundle_metadata(payload, errors)
  artifacts, artifact_snapshots = _artifact_payloads(payload, root, errors)
  v2, ledger, target_snapshots = _validate_target(payload, root, errors)
  required_artifacts = set(_ARTIFACT_PATHS)
  if set(artifacts) != required_artifacts or not v2 or not ledger:
    return errors

  reviewer_a = artifacts["reviewer_a"]
  reviewer_b = artifacts["reviewer_b"]
  attestation = artifacts["reviewer_b_attestation"]
  rubric = artifacts["adjudication_rubric"]
  adjudication = artifacts["adjudication"]
  target = payload["target"]

  if set(reviewer_a) != {
      "reviewer",
      "independence",
      "evidence_snapshot",
      "summary",
      "decisions",
  }:
    errors.append("reviewer_a: exact root shape required")
  if set(reviewer_b) != {
      "schema_version",
      "reviewer",
      "independence",
      "decision_policy",
      "summary",
      "decisions",
  }:
    errors.append("reviewer_b: exact root shape required")
  if reviewer_b.get("schema_version") != "mg1a.simulated_dual_review.v1":
    errors.append("reviewer_b: schema_version mismatch")
  if attestation.get("schema_version") != (
      "mg1a.simulated_review_attestation.v1"
  ):
    errors.append("reviewer_b_attestation: schema_version mismatch")
  if rubric.get("schema_version") != "mg1.adjudication_rubric.v1":
    errors.append("adjudication_rubric: schema_version mismatch")
  if adjudication.get("schema_version") != "mg1.dual_adjudication.v1":
    errors.append("adjudication: schema_version mismatch")
  if set(adjudication) != _ADJUDICATION_ROOT_KEYS:
    errors.append("adjudication: exact root shape required")

  _validate_reviewer_metadata(reviewer_a, reviewer_b, target, errors)
  _validate_attestation(attestation, reviewer_b, payload, errors)
  _validate_rubric(rubric, payload, errors)
  _validate_adjudication_metadata(adjudication, payload, errors)
  _validate_source_evidence(
      reviewer_a,
      reviewer_b,
      attestation,
      adjudication,
      payload,
      root,
      target_snapshots,
      errors,
  )
  _validate_adjudication_records(
      adjudication, reviewer_a, reviewer_b, v2, ledger, payload, errors
  )

  for relative, snapshot in {
      **artifact_snapshots,
      **target_snapshots,
  }.items():
    try:
      observed = _sha256_file(root / relative)
    except OSError as exc:
      errors.append(f"snapshot/{relative}: unavailable after validation ({exc})")
      continue
    expected = _sha256_bytes(snapshot)
    if observed != expected:
      errors.append(
          f"snapshot/{relative}: input changed during validation "
          f"(snapshot {expected}, observed {observed})"
      )
  return errors


def validate_bundle_payload(
    payload: dict[str, Any], *, repo_root: Path | None = None
) -> list[str]:
  """Validate bundle semantics and fail closed on malformed bound data."""
  try:
    return _validate_bundle_payload(payload, repo_root=repo_root)
  except (
      AttributeError,
      IndexError,
      KeyError,
      TypeError,
      ValueError,
  ) as exc:
    return [
        "semantic validation failed closed for malformed data "
        f"({type(exc).__name__}: {exc})"
    ]


def validate_bundle(
    path: Path, *, repo_root: Path | None = None
) -> tuple[dict[str, Any], list[str]]:
  """Load and validate a bundle manifest without GPU dependencies."""
  payload, snapshot = _load_json_snapshot(path)
  root = (repo_root or repository_root()).resolve()
  schema_path = bundle_schema_path(root)
  try:
    schema_snapshot = schema_path.read_bytes()
  except OSError as exc:
    schema_snapshot = b""
    schema_snapshot_error = f"schema: unavailable before validation ({exc})"
  else:
    schema_snapshot_error = None
  errors = json_schema_errors(payload, repo_root=root)
  if schema_snapshot_error is not None:
    errors.append(schema_snapshot_error)
  if not errors:
    errors.extend(validate_bundle_payload(payload, repo_root=root))
  try:
    observed = _sha256_file(path)
  except OSError as exc:
    errors.append(f"bundle: unavailable after validation ({exc})")
  else:
    expected = _sha256_bytes(snapshot)
    if observed != expected:
      errors.append(
          "bundle: input changed during validation "
          f"(snapshot {expected}, observed {observed})"
      )
  if schema_snapshot:
    try:
      schema_observed = _sha256_file(schema_path)
    except OSError as exc:
      errors.append(f"schema: unavailable after validation ({exc})")
    else:
      schema_expected = _sha256_bytes(schema_snapshot)
      if schema_observed != schema_expected:
        errors.append(
            "schema: input changed during validation "
            f"(snapshot {schema_expected}, observed {schema_observed})"
        )
  return payload, errors


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      description="Validate the non-human MG1-A.1 dual-review bundle."
  )
  parser.add_argument("--bundle", type=Path, default=default_bundle_path())
  parser.add_argument("--strict", action="store_true")
  return parser


def main(argv: list[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  try:
    payload, errors = validate_bundle(
        args.bundle, repo_root=repository_root()
    )
  except SimulatedReviewValidationError as exc:
    payload = {}
    errors = list(exc.errors)
  summary = payload.get("summary", {})
  counts = summary.get("adjudicated_counts", {})
  print(
      f"profile={payload.get('profile', PROFILE_NAME)} "
      f"assumptions={summary.get('assumptions_total', 0)} "
      f"adjudicated={counts} errors={len(errors)}"
  )
  for error in errors:
    print(f"ERROR {error}")
  if errors and args.strict:
    return 1
  return 0


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
