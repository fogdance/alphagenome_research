"""CPU-only validation for the MG1-A.1 pattern ontology revision draft.

This module is intentionally independent from the MG1-A v1 implementation.
The v1 byte gate runs without importing any v1 Python module, so a modified
frozen validator cannot execute before its bytes have been checked.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = "mg1a.pattern_ontology.v2"
PROFILE_NAME = "mg1a-v2"
MILESTONE = "MG1-A.1"
DECISION = "DRAFT_NEEDS_HUMAN_FREEZE"
INTERPRETATION = "revision draft valid; separate human freeze still required"
CONFIG_PATH = "configs/market_genome/pattern_ontology_v2.yaml"
SCHEMA_PATH = (
    "src/alphatrade/market_genome/ontology/"
    "mg1a_pattern_ontology_v2.schema.json"
)

_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NUMBER_LITERAL_PATTERN = (
    r"(?:[+-]?(?:\d+(?:\.\d+)?|\.\d+)|zero|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million)"
)
_NUMERIC_THRESHOLD_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    r"(?:threshold|cutoff|tolerance|min|max|greater(?:[_\s-]+)than|"
    r"less(?:[_\s-]+)than)"
    r"[_\s:=<>-]+"
    rf"{_NUMBER_LITERAL_PATTERN}"
    r"(?:[_\s-]+(?:percent|points?|ticks?|bars?))?"
    r"(?=$|[^A-Za-z0-9])",
    flags=re.IGNORECASE,
)
_NUMERIC_UNIT_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    rf"{_NUMBER_LITERAL_PATTERN}"
    r"[_\s-]+(?:percent|points?|ticks?|bars?)"
    r"(?=$|[^A-Za-z0-9])",
    flags=re.IGNORECASE,
)
_NUMERIC_COMPARATOR_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])"
    r"(?:gte|lte|gt|lt|above|below|at(?:[_\s-]+)(?:least|most))"
    r"[_\s:=<>-]+"
    rf"{_NUMBER_LITERAL_PATTERN}"
    r"(?:[_\s-]+(?:percent|points?|ticks?|bars?))?"
    r"(?=$|[^A-Za-z0-9])",
    flags=re.IGNORECASE,
)

_V1_FROZEN_HASHES = {
    "configs/market_genome/pattern_ontology_v1.yaml": (
        "3ab4ce88dd63dbb5ec484b6ca420ce9a3ab06579f63b53776252216a10012b90"
    ),
    "configs/market_genome/mg1a_contracts_manifest.yaml": (
        "02820dca83420add736a1fa6168d2acc33e7f81ae6f3629d6340500afc636ac2"
    ),
    "docs/alphaTrade/market_genome/MG1_PATTERN_ONTOLOGY.md": (
        "d205d1511dc9750f031030a15f139b1cb3220bba0dcfc375212ee20a0d670779"
    ),
    "src/alphatrade/market_genome/ontology/schema.py": (
        "3143100cc22f9674539ef9a3f4abe72d9698a2d2d1de72acf9d0178d391d493e"
    ),
    "src/alphatrade/market_genome/ontology/validation.py": (
        "251764717de39e6c6470c3ef5fff716209676e3db6052de4018839477dfeb72d"
    ),
    "src/alphatrade/market_genome/ontology/mg1a_pattern_ontology.schema.json": (
        "badd21886961c2dbe59e12b24b5682110529b61be187736d91d6b233c699037e"
    ),
    "tests/market_genome/test_mg1a_ontology.py": (
        "b31f341389215574c80ddfa7ad84f3c59609d9c4c6179ab085f6bf9bbf1f4a40"
    ),
}

_MG0_FROZEN_HASHES = {
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

_REVIEW_INPUT_HASHES = {
    "configs/market_genome/pattern_ontology_v1_review.yaml": (
        "a4941b85fd92ada4e58429e2ad0ce0b73d371ac35caa7fe99847a4688c16e90b"
    ),
    "configs/market_genome/reviews/mg1_common_review_v1.yaml": (
        "1f0f2bf82cc542a433aa26e98a2c7a6475b4e713ed3fd30a5ba4d6130bc949aa"
    ),
    (
        "configs/market_genome/reviews/"
        "mg1_reversal_conversion_family_review_v1.yaml"
    ): (
        "625bdec5ffc86ed9816cf3b767bf94d9e8844890204c83d1b64d6b645ce8367e"
    ),
    "configs/market_genome/reviews/mg1_trend_level_family_review_v1.yaml": (
        "c67a04655c8042ec1ae5b32d1c2400cd39ab5829022af91123c1c648f02c089e"
    ),
}

_V1_RESOLVED_CONTRACT_SHA256 = (
    "9c832971fcb6a88a67493aef3e81a81ce1aabb8e739b4c6575d96925b706dc36"
)
_REVIEW_RECORDS_SHA256 = (
    "fdf2328d1fb8b63c6c342498ade11f29c47db467a28c17a190a0290cdad8132c"
)

_FAMILY_ORDER = (
    "trend_up",
    "trend_down",
    "trend_transition",
    "double_top",
    "double_bottom",
    "head_shoulders_top",
    "inverse_head_shoulders",
    "breakout",
    "failed_breakout",
    "retest",
    "range",
    "support_resistance_conversion",
)
_OBJECT_CLASS_BY_FAMILY = {
    "trend_up": "STRUCTURAL_STATE",
    "trend_down": "STRUCTURAL_STATE",
    "trend_transition": "LEVEL_EVENT",
    "double_top": "MORPHOLOGY",
    "double_bottom": "MORPHOLOGY",
    "head_shoulders_top": "MORPHOLOGY",
    "inverse_head_shoulders": "MORPHOLOGY",
    "breakout": "LEVEL_EVENT",
    "failed_breakout": "LEVEL_EVENT",
    "retest": "LEVEL_EVENT",
    "range": "STRUCTURAL_STATE",
    "support_resistance_conversion": "STRUCTURAL_RELATION",
}
_ORIENTATION_BY_FAMILY = {
    "trend_up": ("UPWARD", "FIXED", ("UPWARD",)),
    "trend_down": ("DOWNWARD", "FIXED", ("DOWNWARD",)),
    "trend_transition": (None, "INSTANCE_BOUND", ("UPWARD", "DOWNWARD")),
    "double_top": ("UPPER", "FIXED", ("UPPER",)),
    "double_bottom": ("LOWER", "FIXED", ("LOWER",)),
    "head_shoulders_top": ("UPPER", "FIXED", ("UPPER",)),
    "inverse_head_shoulders": ("LOWER", "FIXED", ("LOWER",)),
    "breakout": (None, "INSTANCE_BOUND", ("UPWARD", "DOWNWARD")),
    "failed_breakout": (None, "INSTANCE_BOUND", ("UPWARD", "DOWNWARD")),
    "retest": (None, "INSTANCE_BOUND", ("UPWARD", "DOWNWARD")),
    "range": ("NEUTRAL", "FIXED", ("NEUTRAL",)),
    "support_resistance_conversion": (
        None,
        "INSTANCE_BOUND",
        ("UPWARD", "DOWNWARD"),
    ),
}
_SEMANTIC_ROLES_BY_FAMILY = {
    "trend_up": ("TREND_STATE",),
    "trend_down": ("TREND_STATE",),
    "trend_transition": ("TRANSITION_EVENT",),
    "double_top": (
        "REVERSAL_CANDIDATE",
        "RANGE_BOUNDARY_REJECTION",
        "CONTINUATION_STRUCTURE",
        "UNCLASSIFIED",
    ),
    "double_bottom": (
        "REVERSAL_CANDIDATE",
        "RANGE_BOUNDARY_REJECTION",
        "CONTINUATION_STRUCTURE",
        "UNCLASSIFIED",
    ),
    "head_shoulders_top": (
        "REVERSAL_CANDIDATE",
        "RANGE_BOUNDARY_REJECTION",
        "CONTINUATION_STRUCTURE",
        "UNCLASSIFIED",
    ),
    "inverse_head_shoulders": (
        "REVERSAL_CANDIDATE",
        "RANGE_BOUNDARY_REJECTION",
        "CONTINUATION_STRUCTURE",
        "UNCLASSIFIED",
    ),
    "breakout": ("BREAK_EVENT",),
    "failed_breakout": ("FAILURE_EVENT",),
    "retest": ("RETEST_EVENT",),
    "range": ("RANGE_STATE",),
    "support_resistance_conversion": ("ROLE_CONVERSION",),
}
_PERSISTENT_BY_FAMILY = {
    "trend_up": True,
    "trend_down": True,
    "trend_transition": False,
    "double_top": False,
    "double_bottom": False,
    "head_shoulders_top": False,
    "inverse_head_shoulders": False,
    "breakout": False,
    "failed_breakout": False,
    "retest": False,
    "range": True,
    "support_resistance_conversion": False,
}
_COMMON_FAMILY_KEYS = frozenset(
    {
        "family",
        "object_class",
        "structural_orientation",
        "orientation_binding",
        "allowed_instance_orientations",
        "semantic_roles",
        "persistent_state",
        "scale_spec",
        "anchors",
        "phase_contract",
        "topology_rules",
        "formula_contracts",
        "confirmation_rules",
        "invalidation_rules",
        "relationship_types",
        "assumption_ids",
    }
)
_FAMILY_SPECIFIC_KEYS = {
    "breakout": frozenset({"instance_fields"}),
    "failed_breakout": frozenset(
        {
            "instance_fields",
            "direction_contract",
            "object_creation_event",
            "historical_attempt_relationship",
            "historical_source_evidence_available_at_creation_only",
        }
    ),
    "retest": frozenset({"instance_fields"}),
    "range": frozenset({"range_geometry"}),
    "support_resistance_conversion": frozenset(
        {"instance_fields", "role_conversion_contract"}
    ),
}
_INSTANCE_FIELDS_BY_FAMILY = {
    "breakout": (
        "source_pattern_ids",
        "source_zone_id",
        "boundary_geometry",
        "cross_direction",
    ),
    "failed_breakout": (
        "source_breakout_attempt_id",
        "source_zone_id",
        "attempt_direction",
        "reentry_direction",
        "resolution_direction",
    ),
    "retest": (
        "source_event_id",
        "source_pattern_id",
        "source_zone_id",
        "origin_break_direction",
        "retest_ordinal",
    ),
    "support_resistance_conversion": (
        "source_breakout_id",
        "source_retest_id",
        "source_zone_id",
        "original_role",
        "converted_role",
    ),
}
_FAILED_BREAKOUT_DIRECTION_CONTRACT = {
    "attempt_direction": ["UPWARD", "DOWNWARD"],
    "reentry_direction": ["UPWARD", "DOWNWARD"],
    "resolution_direction": ["UPWARD", "DOWNWARD", "NEUTRAL", "UNRESOLVED"],
    "structural_orientation_basis": "ATTEMPT_DIRECTION",
    "reentry_direction_is_opposite_attempt": True,
    "directions_must_not_be_collapsed": True,
}
_ROLE_CONVERSION_CONTRACT = {
    "upward_break": "RESISTANCE_TO_SUPPORT",
    "downward_break": "SUPPORT_TO_RESISTANCE",
    "source_breakout_and_retest_required": True,
    "independent_morphology_label": False,
}

_STRUCTURAL_ORIENTATIONS = ("UPPER", "LOWER", "UPWARD", "DOWNWARD", "NEUTRAL")
_ORIENTATION_BINDING_VALUES = ("FIXED", "INSTANCE_BOUND")
_SEMANTIC_ROLE_REGISTRY = (
    "TREND_STATE",
    "RANGE_STATE",
    "REVERSAL_CANDIDATE",
    "RANGE_BOUNDARY_REJECTION",
    "CONTINUATION_STRUCTURE",
    "TRANSITION_EVENT",
    "BREAK_EVENT",
    "FAILURE_EVENT",
    "RETEST_EVENT",
    "ROLE_CONVERSION",
    "UNCLASSIFIED",
)
_SCALE_REQUIRED_FIELDS = (
    "scale_id",
    "scale_registry_version",
    "scale_registry_sha256",
    "bar_resolution",
    "aggregation_policy",
    "session_alignment",
    "pivot_prominence_scale_ref",
    "formation_span_bars",
    "formation_span_session_bars",
    "scale_octave",
    "encoder_scale_ref",
)
_ANCHOR_STATES = ("PROVISIONAL", "CONFIRMED", "REVISED", "SUPERSEDED", "REVOKED")
_ANCHOR_REQUIRED_FIELDS = (
    "anchor_id",
    "anchor_lineage_id",
    "role",
    "event_eob",
    "available_as_of_eob",
    "as_of_eob",
    "source_bar_ref",
    "source_bar_closed",
    "anchor_state",
    "anchor_revision",
    "supersedes_anchor_id",
    "revision_reason",
    "normalized_price",
    "price_basis",
    "confirmation_basis",
    "scale_spec_ref",
    "source_method",
    "source_method_version",
    "assumption_ids",
)
_REVIEW_MODES = ("ONLINE_BLINDED", "RETROSPECTIVE", "OUTCOME")
_REVIEW_REQUIRED_FIELDS = (
    "review_mode",
    "review_id",
    "sample_id",
    "namespace",
    "reviewer_ids",
    "reviewer_decisions",
    "adjudicator_id",
    "adjudication_status",
    "ambiguity_status",
    "future_visibility_policy",
    "evidence_snapshot_sha256",
    "reviewed_at",
    "provenance",
)

_COMMON_STATUSES = ("INACTIVE", "ACTIVE", "RESOLVED", "INVALIDATED")
_PROJECTED_EDGES = (
    "INACTIVE_TO_ACTIVE",
    "ACTIVE_TO_ACTIVE",
    "ACTIVE_TO_RESOLVED",
    "ACTIVE_TO_INVALIDATED",
)
_ALLOWED_PROJECTED_PAIRS = {
    ("INACTIVE", "ACTIVE"),
    ("ACTIVE", "ACTIVE"),
    ("ACTIVE", "RESOLVED"),
    ("ACTIVE", "INVALIDATED"),
}
_RELATION_TYPES = (
    "NESTED_IN",
    "CONTAINS",
    "CONFIRMED_BY",
    "INVALIDATED_BY",
    "RETESTS",
    "ORIGINATES_FROM",
    "CONVERTS_ROLE_OF",
    "SAME_STRUCTURE_DIFFERENT_SCALE",
)
_RELATION_REQUIRED_FIELDS = {
    "relation_id",
    "relation_type",
    "source_ref",
    "target_ref",
    "relation_event_id",
    "source_event_ids",
    "relation_ordinal",
    "event_eob",
    "available_as_of_eob",
    "as_of_eob",
    "relation_provenance",
    "assumption_ids",
}
_EXPECTED_EXECUTION = {
    "detector_implemented": False,
    "labeler_implemented": False,
    "real_data_labels_created": False,
    "synthetic_samples_created": False,
    "trading_rules_implemented": False,
    "model_training_executed": False,
    "jax_computation_executed": False,
    "gpu_executed": False,
}
_EXPECTED_NEW_ASSUMPTION_IDS = (
    "MG1A1.COMMON.OBJECT_MODEL.001",
    "MG1A1.COMMON.MORPHOLOGY_EFFECT.001",
    "MG1A1.COMMON.LIFECYCLE_MAPPING.001",
    "MG1A1.COMMON.SCALE_SPEC.001",
    "MG1A1.COMMON.ANCHOR_REVISION.001",
    "MG1A1.COMMON.RELATION_GRAPH.001",
    "MG1A1.COMMON.FORMULA_CONTRACT.001",
    "MG1A1.COMMON.REVIEW_PROTOCOL.001",
    "MG1A1.COMMON.FAMILY_REVISIONS.001",
)

# These are negative assertions, not future-derived family data.
_ALLOWED_FAMILY_FUTURE_ASSERTIONS = {
    "CONTINUATION_IS_LATER_RESOLUTION",
    "SHAPE_IDENTITY_INDEPENDENT_OF_FUTURE_EFFECT",
    "ROLE_CONVERSION_IS_RELATION_NOT_FUTURE_EFFECT",
}
_ALLOWED_FAMILY_CAUSAL_RETURN_TERMS = {
    "MAXIMUM_RETURN_PENETRATION_DISTANCE",
    "OPPOSITE_SIDE_RETURN",
    "OPPOSITE_SIDE_RETURN_DEPTH",
    "RETURN_AND_CONVERTED_ROLE_TEST_REQUIRED",
    "RETURN_DEPTH_TO_BREAK_LEG",
    "RETURN_DURATION_TO_BREAK_LEG_DURATION",
    "RETURN_LEG_OBSERVED",
    "RETURN_OBSERVED",
    "RETURN_TOUCH",
    "REVISED_UNTIL_RETURN",
    "SCALE_RELATIVE_RETURN_EXPIRY",
}
_FORBIDDEN_FAMILY_TOKENS = {
    "ACTION",
    "BUY",
    "ENTRY",
    "EXPECTED",
    "FOLLOWING",
    "FORWARD",
    "FUTURE",
    "LATER",
    "LONG",
    "SELL",
    "SHORT",
    "SIGNAL",
    "TRADE",
    "NEXT",
    "OUTCOME",
    "POSITION",
    "PROFIT",
    "PROFITABLE",
    "RETURN",
    "PNL",
    "MFE",
    "MAE",
    "TARGET",
    "LOOKAHEAD",
    "TOMORROW",
    "SUBSEQUENT",
}
_FORBIDDEN_FAMILY_SEQUENCES = (
    "FUTURE_OUTCOME",
    "FUTURE_RETURN",
    "FORWARD_RETURN",
    "NEXT_BAR",
    "NEXT_DAY",
    "ORDER_INTENT",
    "TRADE_INTENT",
    "SIGNAL_INTENT",
)


class RevisionValidationError(ValueError):
  """Raised when the MG1-A.1 revision cannot be loaded or validated."""

  def __init__(self, errors: Iterable[str]):
    self.errors = tuple(errors)
    super().__init__("; ".join(self.errors))


def repository_root() -> Path:
  """Return the repository containing this module."""
  return Path(__file__).resolve().parents[4]


def revision_config_path(repo_root: Path | None = None) -> Path:
  root = (repo_root or repository_root()).resolve()
  return root / CONFIG_PATH


def revision_schema_path(repo_root: Path | None = None) -> Path:
  if repo_root is None:
    return Path(__file__).with_name("mg1a_pattern_ontology_v2.schema.json")
  return repo_root.resolve() / SCHEMA_PATH


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
  return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
  """Return deterministic ASCII JSON suitable for contract hashing."""
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
        raise RevisionValidationError([f"duplicate JSON key: {key}"])
      result[key] = value
    return result

  try:
    return json.loads(text, object_pairs_hook=unique_object)
  except json.JSONDecodeError as exc:
    raise RevisionValidationError([f"invalid JSON: {exc}"]) from exc


def loads_yaml_unique(text: str) -> Any:
  """Parse YAML while rejecting duplicate mapping keys."""
  try:
    import yaml
  except ImportError as exc:  # pragma: no cover - dependency guard
    raise RevisionValidationError([f"PyYAML is required: {exc}"]) from exc

  class UniqueKeySafeLoader(yaml.SafeLoader):
    pass

  def construct_mapping(
      loader: UniqueKeySafeLoader, node: Any, deep: bool = False
  ) -> dict[Any, Any]:
    explicit_keys: set[Any] = set()
    for key_node, _value_node in node.value:
      if key_node.tag == "tag:yaml.org,2002:merge":
        continue
      key = loader.construct_object(key_node, deep=deep)
      try:
        duplicate = key in explicit_keys
      except TypeError as exc:
        raise RevisionValidationError(
            [
                "unhashable YAML mapping key at line "
                f"{key_node.start_mark.line + 1}"
            ]
        ) from exc
      if duplicate:
        raise RevisionValidationError(
            [
                "duplicate YAML key "
                f"{key!r} at line {key_node.start_mark.line + 1}"
            ]
        )
      explicit_keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

  UniqueKeySafeLoader.add_constructor(
      yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
      construct_mapping,
  )
  try:
    return yaml.load(text, Loader=UniqueKeySafeLoader)
  except RevisionValidationError:
    raise
  except yaml.YAMLError as exc:
    raise RevisionValidationError([f"invalid YAML: {exc}"]) from exc


def _load_revision_snapshot(path: Path) -> tuple[dict[str, Any], bytes]:
  try:
    snapshot = path.read_bytes()
    payload = loads_yaml_unique(snapshot.decode("utf-8"))
  except (OSError, UnicodeError) as exc:
    raise RevisionValidationError([f"revision unavailable: {exc}"]) from exc
  if not isinstance(payload, dict):
    raise RevisionValidationError(["<root>: expected a mapping"])
  return payload, snapshot


def load_revision(path: Path) -> dict[str, Any]:
  """Load a duplicate-key-safe MG1-A.1 YAML payload."""
  return _load_revision_snapshot(path)[0]


def _format_jsonschema_path(parts: Iterable[Any]) -> str:
  rendered = "/".join(str(item) for item in parts)
  return rendered or "<root>"


def _json_schema_errors_from_snapshot(
    payload: Any, schema_snapshot: bytes
) -> list[str]:
  try:
    schema_payload = loads_json_unique(schema_snapshot.decode("utf-8"))
    import jsonschema
  except (UnicodeError, RevisionValidationError, ImportError) as exc:
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
      f"schema/{_format_jsonschema_path(error.absolute_path)}: {error.message}"
      for error in errors
  ]


def json_schema_errors(
    payload: Any, *, repo_root: Path | None = None
) -> list[str]:
  """Return closed-shape Draft-07 errors for a v2 payload."""
  path = revision_schema_path(repo_root)
  try:
    schema_snapshot = path.read_bytes()
  except OSError as exc:
    return [f"schema: unavailable or invalid ({exc})"]
  return _json_schema_errors_from_snapshot(payload, schema_snapshot)


def _check_hash_map(
    root: Path, expected: dict[str, str], label: str
) -> tuple[list[str], dict[str, bytes]]:
  errors: list[str] = []
  snapshots: dict[str, bytes] = {}
  for relative, expected_hash in expected.items():
    path = root / relative
    try:
      snapshot = path.read_bytes()
    except OSError as exc:
      errors.append(f"{label}/{relative}: unavailable ({exc})")
      continue
    snapshots[relative] = snapshot
    actual = _sha256_bytes(snapshot)
    if actual != expected_hash:
      errors.append(
          f"{label}/{relative}: byte hash mismatch "
          f"(expected {expected_hash}, observed {actual})"
      )
  return errors, snapshots


def _snapshot_consistency_errors(
    root: Path, snapshots: dict[str, bytes], label: str
) -> list[str]:
  errors: list[str] = []
  for relative, snapshot in snapshots.items():
    expected_hash = _sha256_bytes(snapshot)
    try:
      observed_hash = _sha256_file(root / relative)
    except OSError as exc:
      errors.append(
          f"{label}/{relative}: unavailable after snapshot ({exc})"
      )
      continue
    if observed_hash != expected_hash:
      errors.append(
          f"{label}/{relative}: input changed during validation "
          f"(snapshot {expected_hash}, observed {observed_hash})"
      )
  return errors


def _validate_frozen_inputs(
    payload: dict[str, Any], root: Path
) -> tuple[list[str], list[str]]:
  errors, mg0_snapshots = _check_hash_map(
      root, _MG0_FROZEN_HASHES, "mg0_frozen"
  )
  v1_errors, v1_snapshots = _check_hash_map(
      root, _V1_FROZEN_HASHES, "v1_frozen"
  )
  review_errors, review_snapshots = _check_hash_map(
      root, _REVIEW_INPUT_HASHES, "review_input"
  )
  errors.extend(v1_errors)
  errors.extend(review_errors)

  declared = payload["v1_artifact_hashes"]
  expected_declared = [
      {"path": path, "sha256": digest}
      for path, digest in _V1_FROZEN_HASHES.items()
  ]
  if declared != expected_declared:
    errors.append("v1_artifact_hashes: exact ordered frozen map required")

  base = payload["base_v1"]
  expected_base = {
      "config_path": "configs/market_genome/pattern_ontology_v1.yaml",
      "config_sha256": _V1_FROZEN_HASHES[
          "configs/market_genome/pattern_ontology_v1.yaml"
      ],
      "resolved_contract_sha256": _V1_RESOLVED_CONTRACT_SHA256,
      "review_ledger_path": (
          "configs/market_genome/pattern_ontology_v1_review.yaml"
      ),
      "review_records_sha256": _REVIEW_RECORDS_SHA256,
  }
  if base != expected_base:
    errors.append("base_v1: exact frozen source binding required")

  try:
    v1_payload = loads_yaml_unique(
        v1_snapshots[expected_base["config_path"]].decode("utf-8")
    )
    v1_contract_hash = canonical_sha256(v1_payload["resolved_contract"])
  except (
      OSError,
      UnicodeError,
      KeyError,
      TypeError,
      RevisionValidationError,
  ) as exc:
    errors.append(f"base_v1: cannot verify resolved contract ({exc})")
  else:
    if v1_contract_hash != _V1_RESOLVED_CONTRACT_SHA256:
      errors.append("base_v1/resolved_contract_sha256: semantic hash mismatch")

  review_ids: list[str] = []
  try:
    review_payload = loads_yaml_unique(
        review_snapshots[expected_base["review_ledger_path"]].decode("utf-8")
    )
    records = review_payload["records"]
    records_hash = canonical_sha256(records)
    review_ids = [record["assumption_id"] for record in records]
  except (
      OSError,
      UnicodeError,
      KeyError,
      TypeError,
      RevisionValidationError,
  ) as exc:
    errors.append(f"base_v1/review_ledger: cannot verify records ({exc})")
  else:
    if records_hash != _REVIEW_RECORDS_SHA256:
      errors.append("base_v1/review_records_sha256: canonical hash mismatch")
    if review_payload.get("records_sha256") != records_hash:
      errors.append("base_v1/review_ledger: stored records hash mismatch")
    if review_payload.get("decision") != (
        "DOMAIN_REVIEW_COMPLETED_WITH_REQUIRED_REVISIONS"
    ):
      errors.append("base_v1/review_ledger: required revision decision missing")
    summary = review_payload.get("summary", {})
    if not (
        isinstance(summary, dict)
        and summary.get("assumptions_total") == 61
        and summary.get("reviewed") == 61
        and summary.get("freeze_ready_records") == 0
        and summary.get("freeze_blocked_records") == 61
        and summary.get("freeze_ready") is False
    ):
      errors.append(
          "base_v1/review_ledger: exact freeze-blocked summary required"
      )
  errors.extend(
      _snapshot_consistency_errors(root, mg0_snapshots, "mg0_frozen")
  )
  errors.extend(
      _snapshot_consistency_errors(root, v1_snapshots, "v1_frozen")
  )
  errors.extend(
      _snapshot_consistency_errors(root, review_snapshots, "review_input")
  )
  return errors, review_ids


def _walk(
    value: Any, path: str = "resolved_contract"
) -> Iterable[tuple[str, Any]]:
  yield path, value
  if isinstance(value, dict):
    for key, nested in value.items():
      yield from _walk(nested, f"{path}/{key}")
  elif isinstance(value, list):
    for index, nested in enumerate(value):
      yield from _walk(nested, f"{path}/{index}")


def _family_language_errors(families: list[dict[str, Any]]) -> list[str]:
  errors: list[str] = []
  for family_index, family in enumerate(families):
    for path, value in _walk(family, f"families/{family_index}"):
      if not isinstance(value, str):
        continue
      if value in _ALLOWED_FAMILY_FUTURE_ASSERTIONS:
        continue
      upper = value.upper()
      if upper in _ALLOWED_FAMILY_CAUSAL_RETURN_TERMS:
        continue
      tokens = set(_IDENTIFIER_TOKEN_RE.findall(upper))
      bad_tokens = sorted(tokens & _FORBIDDEN_FAMILY_TOKENS)
      bad_sequences = [
          item for item in _FORBIDDEN_FAMILY_SEQUENCES if item in upper
      ]
      if bad_tokens or bad_sequences:
        observed = ",".join([*bad_tokens, *bad_sequences])
        errors.append(
            f"{path}: forbidden future/trading semantics ({observed})"
        )
  return errors


def _numeric_contract_errors(contract: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  for path, value in _walk(contract):
    if isinstance(value, bool) or value is None:
      continue
    if isinstance(value, (int, float)):
      errors.append(f"{path}: numeric detector threshold/value is forbidden")
    elif isinstance(value, str) and any(
        pattern.search(value)
        for pattern in (
            _NUMERIC_THRESHOLD_RE,
            _NUMERIC_UNIT_RE,
            _NUMERIC_COMPARATOR_RE,
        )
    ):
      errors.append(f"{path}: embedded numeric detector threshold is forbidden")
  return errors


def _draft_governance_errors(
    payload: dict[str, Any], review_ids: list[str]
) -> list[str]:
  errors: list[str] = []
  expected_human_review = {
      "status": "PENDING",
      "frozen": False,
      "reviewer_decisions_recorded": False,
      "approver": None,
      "approved_at": None,
      "frozen_sha256": None,
  }
  if payload["human_review"] != expected_human_review:
    errors.append("human_review: v2 must remain an unfrozen pending draft")
  expected_outcome = {
      "decision": "DOMAIN_REVIEW_COMPLETED_WITH_REQUIRED_REVISIONS",
      "draft_assessment": "APPROVED_AS_DRAFT",
      "freeze_assessment": "REVISION_REQUIRED_BEFORE_FREEZE",
      "mg1b_status": "BLOCKED",
      "next_milestone": "MG1-A.1",
  }
  if payload["review_outcome_v1"] != expected_outcome:
    errors.append("review_outcome_v1: MG1-B must remain blocked")
  if payload["execution"] != _EXPECTED_EXECUTION:
    errors.append(
        "execution: detector/labeler/training/GPU flags must all be false"
    )

  contract = payload["resolved_contract"]
  if contract.get("contract_status") != "REVISED_REVIEW_DRAFT":
    errors.append("resolved_contract/contract_status: review draft required")
  registry = contract["assumption_registry"]
  inherited = registry["inherited_from_v1"]
  if inherited.get("status") != "NEEDS_HUMAN_REVIEW":
    errors.append(
        "assumption_registry/inherited_from_v1: review must remain open"
    )
  if inherited.get("proposed_default") is not None:
    errors.append(
        "assumption_registry/inherited_from_v1: default must remain null"
    )
  if inherited.get("reviewer_decision") is not None:
    errors.append(
        "assumption_registry/inherited_from_v1: decision must remain null"
    )
  if inherited.get("assumption_ids") != review_ids:
    errors.append(
        "assumption_registry/inherited_from_v1: exact reviewed assumption "
        "order required"
    )

  new_assumptions = registry["new_v2_assumptions"]
  new_ids = tuple(item["assumption_id"] for item in new_assumptions)
  if new_ids != _EXPECTED_NEW_ASSUMPTION_IDS:
    errors.append(
        "assumption_registry/new_v2_assumptions: exact open set required"
    )
  for index, item in enumerate(new_assumptions):
    if (
        item.get("status") != "NEEDS_HUMAN_REVIEW"
        or item.get("proposed_default") is not None
        or item.get("reviewer_decision") is not None
    ):
      errors.append(
          f"assumption_registry/new_v2_assumptions/{index}: "
          "must remain open with null decision/default"
      )
  return errors


def _object_model_errors(contract: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  model = contract["object_model"]
  if model["class_by_family"] != _OBJECT_CLASS_BY_FAMILY:
    errors.append(
        "object_model/class_by_family: exact family object mapping required"
    )
  if tuple(model["object_classes"]) != (
      "STRUCTURAL_STATE",
      "MORPHOLOGY",
      "LEVEL_EVENT",
      "STRUCTURAL_RELATION",
  ):
    errors.append("object_model/object_classes: exact closed registry required")
  if tuple(model["structural_orientations"]) != _STRUCTURAL_ORIENTATIONS:
    errors.append(
        "object_model/structural_orientations: exact closed registry required"
    )
  if tuple(model["orientation_binding_values"]) != _ORIENTATION_BINDING_VALUES:
    errors.append(
        "object_model/orientation_binding_values: exact closed registry required"
    )
  if tuple(model["semantic_roles"]) != _SEMANTIC_ROLE_REGISTRY:
    errors.append("object_model/semantic_roles: exact closed registry required")
  separation = model["morphology_effect_separation"]
  expected_separation = {
      "structural_orientation_is_future_effect": False,
      "semantic_role_is_future_outcome": False,
      "classic_reversal_role_requires_preceding_context": True,
      "future_effect_namespace": "future_outcome",
      "future_effect_allowed_in_family_contract": False,
  }
  if separation != expected_separation:
    errors.append(
        "object_model/morphology_effect_separation: exact separation policy "
        "required"
    )

  families = contract["families"]
  observed_order = tuple(family["family"] for family in families)
  if observed_order != _FAMILY_ORDER:
    errors.append("families: exact order and unique family coverage required")
    return errors
  for index, family in enumerate(families):
    name = family["family"]
    path = f"families/{index}"
    expected_keys = _COMMON_FAMILY_KEYS | _FAMILY_SPECIFIC_KEYS.get(
        name, frozenset()
    )
    observed_keys = set(family)
    if observed_keys != expected_keys:
      errors.append(
          f"{path}: exact common/specific key set required; "
          f"missing={sorted(expected_keys - observed_keys)}, "
          f"unexpected={sorted(observed_keys - expected_keys)}"
      )
    if family["object_class"] != _OBJECT_CLASS_BY_FAMILY[name]:
      errors.append(f"{path}/object_class: inconsistent with object model")
    orientation, binding, allowed = _ORIENTATION_BY_FAMILY[name]
    observed_orientation = (
        family.get("structural_orientation"),
        family.get("orientation_binding"),
        tuple(family.get("allowed_instance_orientations", [])),
    )
    if observed_orientation != (orientation, binding, allowed):
      errors.append(f"{path}: exact structural orientation binding required")
    if tuple(family["semantic_roles"]) != _SEMANTIC_ROLES_BY_FAMILY[name]:
      errors.append(
          f"{path}/semantic_roles: exact object/effect roles required"
      )
    if family["persistent_state"] is not _PERSISTENT_BY_FAMILY[name]:
      errors.append(f"{path}/persistent_state: inconsistent object lifetime")
  return errors


def _common_registry_errors(contract: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  scale = contract["scale_spec_contract"]
  if tuple(scale["required_fields"]) != _SCALE_REQUIRED_FIELDS:
    errors.append(
        "scale_spec_contract/required_fields: exact closed registry required"
    )
  if tuple(scale["nullable_fields"]) != ("scale_octave", "encoder_scale_ref"):
    errors.append(
        "scale_spec_contract/nullable_fields: exact closed registry required"
    )

  anchor = contract["anchor_contract"]
  if tuple(anchor["states"]) != _ANCHOR_STATES:
    errors.append("anchor_contract/states: exact closed registry required")
  expected_state_transitions = {
      "PROVISIONAL": ["CONFIRMED", "REVISED", "REVOKED"],
      "REVISED": ["CONFIRMED", "SUPERSEDED", "REVOKED"],
      "CONFIRMED": ["SUPERSEDED"],
      "SUPERSEDED": [],
      "REVOKED": [],
  }
  if anchor["state_transitions"] != expected_state_transitions:
    errors.append(
        "anchor_contract/state_transitions: exact closed graph required"
    )
  if tuple(anchor["required_fields"]) != _ANCHOR_REQUIRED_FIELDS:
    errors.append(
        "anchor_contract/required_fields: exact closed registry required"
    )
  if tuple(anchor["price_basis_values"]) != (
      "HIGH",
      "LOW",
      "CLOSE",
      "BODY_HIGH",
      "BODY_LOW",
      "SETTLEMENT",
      "VWAP",
  ):
    errors.append(
        "anchor_contract/price_basis_values: exact closed registry required"
    )
  if tuple(anchor["confirmation_basis_values"]) != (
      "PIVOT_SOURCE_RULE",
      "CLOSED_BAR_CROSS",
      "CLOSED_BAR_HOLD",
      "ANCHOR_SEQUENCE",
      "ZONE_ROLE_TEST",
  ):
    errors.append(
        "anchor_contract/confirmation_basis_values: exact closed registry "
        "required"
    )

  review = contract["review_protocol"]
  if tuple(review["review_modes"]) != _REVIEW_MODES:
    errors.append("review_protocol/review_modes: exact closed registry required")
  if tuple(review["required_fields"]) != _REVIEW_REQUIRED_FIELDS:
    errors.append(
        "review_protocol/required_fields: exact closed registry required"
    )
  if tuple(review["adjudication_statuses"]) != (
      "NOT_REQUIRED",
      "PENDING",
      "ADJUDICATED",
  ):
    errors.append(
        "review_protocol/adjudication_statuses: exact closed registry required"
    )
  if tuple(review["ambiguity_statuses"]) != (
      "UNAMBIGUOUS",
      "AMBIGUOUS",
      "INSUFFICIENT_EVIDENCE",
  ):
    errors.append(
        "review_protocol/ambiguity_statuses: exact closed registry required"
    )
  return errors


def _assumption_reference_errors(
    contract: dict[str, Any], inherited_source_ids: Iterable[str]
) -> list[str]:
  registry = contract["assumption_registry"]
  registered_ids = set(registry["inherited_from_v1"]["assumption_ids"])
  registered_ids.update(
      item["assumption_id"] for item in registry["new_v2_assumptions"]
  )
  errors: list[str] = []
  # Inherited assumptions remain concretely bound to the byte-checked v1 ledger.
  used_ids = set(inherited_source_ids)
  for path, value in _walk(contract):
    if not path.endswith("/assumption_ids") or not isinstance(value, list):
      continue
    if path.startswith("resolved_contract/assumption_registry/"):
      continue
    used_ids.update(value)
    unknown = sorted(set(value) - registered_ids)
    if unknown:
      errors.append(f"{path}: unregistered assumption references {unknown}")
  unused = sorted(registered_ids - used_ids)
  if unused:
    errors.append(
        "assumption_registry: every registered assumption must be referenced "
        "by the resolved contract or hash-bound inherited review ledger; "
        f"unused={unused}"
    )
  return errors


def _family_p0_errors(contract: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  families = contract["families"]
  if tuple(family["family"] for family in families) != _FAMILY_ORDER:
    return errors
  by_name = {family["family"]: family for family in families}

  transition = by_name["trend_transition"]
  expected_first_transition = {
      "from_phase": "INACTIVE",
      "to_phase": "STRUCTURE_BREAK_OBSERVED",
      "event": "SHARED_DEFENDING_PIVOT_BREAK",
  }
  if (
      transition["phase_contract"]["branch_transitions"][0]
      != expected_first_transition
      or "PRIOR_TREND_WEAKENING"
      in transition["phase_contract"]["family_phases"]
  ):
    errors.append(
        "families/trend_transition: shared defending-pivot break must be the "
        "first event and prior weakening is not a phase"
    )

  for family_name in ("double_top", "double_bottom"):
    family = by_name[family_name]
    if any(
        isinstance(value, str) and "NECKLINE_SLOPE" in value.upper()
        for _, value in _walk(family)
    ):
      errors.append(
          f"families/{family_name}: horizontal neckline cannot declare "
          "neckline_slope"
      )

  required_head_shoulders_anchors = {
      "head_shoulders_top": {
          "left_neck_trough",
          "right_neck_trough",
          "neckline_segment",
          "neckline_break",
          "neckline_retest",
          "continuation_anchor",
      },
      "inverse_head_shoulders": {
          "left_neck_peak",
          "right_neck_peak",
          "neckline_segment",
          "neckline_break",
          "neckline_retest",
          "continuation_anchor",
      },
  }
  for family_name, required_roles in required_head_shoulders_anchors.items():
    observed_roles = {item["role"] for item in by_name[family_name]["anchors"]}
    missing = sorted(required_roles - observed_roles)
    if missing:
      errors.append(
          f"families/{family_name}/anchors: required neckline, retest, and "
          f"continuation anchors missing={missing}"
      )

  failed = by_name["failed_breakout"]
  expected_failed_creation = {
      "from_phase": "INACTIVE",
      "to_phase": "REENTRY_OBSERVED",
      "event": "SHARED_BOUNDARY_REENTRY",
  }
  if (
      failed.get("direction_contract")
      != _FAILED_BREAKOUT_DIRECTION_CONTRACT
      or failed.get("object_creation_event") != "SHARED_BOUNDARY_REENTRY"
      or failed.get("historical_attempt_relationship") != "ORIGINATES_FROM"
      or failed.get("historical_source_evidence_available_at_creation_only")
      is not True
      or failed["phase_contract"]["branch_transitions"][0]
      != expected_failed_creation
  ):
    errors.append(
        "families/failed_breakout: exact direction, creation, and historical "
        "source contract required"
    )

  retest = by_name["retest"]
  required_retest_phases = {
      "HOLD_CONFIRMED",
      "PENETRATE_RECLAIMED",
      "FULL_RECROSS_FAILURE",
      "CONTINUATION",
  }
  if not required_retest_phases.issubset(
      set(retest["phase_contract"]["family_phases"])
  ):
    errors.append(
        "families/retest/phase_contract: hold, penetrate-reclaim, full "
        "recross, and continuation phases must remain distinct"
    )

  range_family = by_name["range"]
  range_mapping = range_family["phase_contract"]["common_mapping"]
  if (
      range_family.get("range_geometry") != "HORIZONTAL_ZONE_PAIR"
      or "ACCEPTED_BREAK" not in range_mapping["ACTIVE"]
      or range_mapping["RESOLVED"] != ["TERMINATED"]
  ):
    errors.append(
        "families/range: horizontal zone-pair geometry and ACTIVE "
        "ACCEPTED_BREAK mapping required"
    )

  conversion = by_name["support_resistance_conversion"]
  if conversion.get("role_conversion_contract") != _ROLE_CONVERSION_CONTRACT:
    errors.append(
        "families/support_resistance_conversion: exact role conversion "
        "contract required"
    )
  return errors


def _relation_errors(contract: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  graph = contract["relationship_graph"]
  if tuple(graph["relation_types"]) != _RELATION_TYPES:
    errors.append(
        "relationship_graph/relation_types: exact closed registry required"
    )
  required = graph["required_fields"]
  if set(required) != _RELATION_REQUIRED_FIELDS or len(required) != len(
      _RELATION_REQUIRED_FIELDS
  ):
    missing = sorted(_RELATION_REQUIRED_FIELDS - set(required))
    unexpected = sorted(set(required) - _RELATION_REQUIRED_FIELDS)
    errors.append(
        "relationship_graph/required_fields: event-bound binary identity fields "
        f"required; missing={missing}, unexpected={unexpected}"
    )
  exact_flags = {
      "cross_family_objects_remain_distinct": True,
      "cross_scale_objects_remain_distinct": True,
      "destructive_merge_allowed": False,
      "source_event_identity_required": True,
      "binary_edges_only": True,
  }
  for key, expected in exact_flags.items():
    if graph.get(key) is not expected:
      errors.append(f"relationship_graph/{key}: expected {expected}")
  if tuple(graph["derived_views"]) != (
      "source_pattern_ids",
      "target_pattern_ids",
      "related_level_ids",
  ):
    errors.append(
        "relationship_graph/derived_views: exact closed registry required"
    )
  relation_types = set(graph["relation_types"])
  for index, family in enumerate(contract["families"]):
    used = family["relationship_types"]
    if len(used) != len(set(used)) or not set(used).issubset(relation_types):
      errors.append(
          f"families/{index}/relationship_types: unknown or duplicate type"
      )

  by_name = {item["family"]: item for item in contract["families"]}
  if set(by_name) != set(_FAMILY_ORDER) or len(by_name) != len(_FAMILY_ORDER):
    return errors
  for family_name, expected_fields in _INSTANCE_FIELDS_BY_FAMILY.items():
    observed = tuple(by_name[family_name].get("instance_fields", []))
    if observed != expected_fields:
      errors.append(
          f"families/{family_name}/instance_fields: exact fields required; "
          f"expected={list(expected_fields)}, observed={list(observed)}"
      )
  return errors


def _expanded_edges(
    phase_contract: dict[str, Any], active_phases: set[str]
) -> tuple[list[tuple[str, str, str]], list[str]]:
  errors: list[str] = []
  edges: list[tuple[str, str, str]] = []
  for edge in phase_contract["branch_transitions"]:
    edges.append((edge["from_phase"], edge["to_phase"], edge["event"]))
  for index, transition in enumerate(
      phase_contract.get("invalidation_transitions", [])
  ):
    source_spec = transition["from_phases"]
    if source_spec == "ALL_ACTIVE_FAMILY_PHASES":
      sources = sorted(active_phases)
    elif isinstance(source_spec, list):
      sources = source_spec
    else:
      errors.append(
          f"invalidation_transitions/{index}/from_phases: invalid source "
          "selector"
      )
      continue
    for source in sources:
      for event in transition["events"]:
        edges.append((source, transition["to_phase"], event))
  return edges, errors


def _phase_graph_errors(contract: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  lifecycle = contract["common_lifecycle"]
  if tuple(lifecycle["statuses"]) != _COMMON_STATUSES:
    errors.append("common_lifecycle/statuses: exact common statuses required")
  if tuple(lifecycle["projected_edges"]) != _PROJECTED_EDGES:
    errors.append("common_lifecycle/projected_edges: exact graph required")
  expected_flags = {
      "mapping_policy": "VARIABLE_LENGTH_MANY_TO_ONE",
      "branch_transitions_required": True,
      "retest_and_continuation_must_be_separate": True,
      "terminal_objects_are_immutable": True,
      "reactivation_requires_new_pattern_id": True,
      "every_invalidated_phase_requires_explicit_causal_in_edge": True,
  }
  for key, expected in expected_flags.items():
    if lifecycle.get(key) != expected:
      errors.append(f"common_lifecycle/{key}: expected {expected}")

  for family_index, family in enumerate(contract["families"]):
    path = f"families/{family_index}/phase_contract"
    phase_contract = family["phase_contract"]
    phases = phase_contract["family_phases"]
    if len(phases) != len(set(phases)):
      errors.append(f"{path}/family_phases: phases must be unique")
    mapping = phase_contract["common_mapping"]
    if tuple(mapping) != _COMMON_STATUSES:
      errors.append(f"{path}/common_mapping: exact common status keys required")
      continue
    mapped = [
        phase
        for status in _COMMON_STATUSES
        for phase in mapping[status]
    ]
    if len(mapped) != len(set(mapped)) or set(mapped) != set(phases):
      errors.append(
          f"{path}/common_mapping: exact disjoint phase partition required"
      )
      continue
    phase_status = {
        phase: status
        for status in _COMMON_STATUSES
        for phase in mapping[status]
    }
    active = set(mapping["ACTIVE"])
    inactive = set(mapping["INACTIVE"])
    resolved = set(mapping["RESOLVED"])
    invalidated = set(mapping["INVALIDATED"])
    terminal = resolved | invalidated
    if not all((inactive, active, resolved, invalidated)):
      errors.append(f"{path}/common_mapping: every common status must be used")
    for transition_index, transition in enumerate(
        phase_contract["invalidation_transitions"]
    ):
      target = transition["to_phase"]
      if phase_status.get(target) != "INVALIDATED":
        errors.append(
            f"{path}/invalidation_transitions/{transition_index}/to_phase: "
            "invalidation transition target must map to common INVALIDATED"
        )

    edges, expansion_errors = _expanded_edges(phase_contract, active)
    errors.extend(f"{path}/{error}" for error in expansion_errors)
    edge_keys = [(source, target, event) for source, target, event in edges]
    if len(edge_keys) != len(set(edge_keys)):
      errors.append(f"{path}: duplicate expanded transition")

    adjacency: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    projected_pairs: set[tuple[str, str]] = set()
    invalidated_incoming: set[str] = set()
    for edge_index, (source, target, event) in enumerate(edges):
      if source not in phase_status or target not in phase_status:
        errors.append(
            f"{path}/transitions/{edge_index}: unknown endpoint "
            f"{source}->{target}"
        )
        continue
      projected = (phase_status[source], phase_status[target])
      projected_pairs.add(projected)
      if projected not in _ALLOWED_PROJECTED_PAIRS:
        errors.append(
            f"{path}/transitions/{edge_index}: illegal common projection "
            f"{projected[0]}->{projected[1]}"
        )
      if source in terminal:
        errors.append(
            f"{path}/transitions/{edge_index}: terminal phase {source} "
            "has an out-edge"
        )
      if target in invalidated:
        invalidated_incoming.add(target)
      adjacency[source].add(target)
      reverse[target].add(source)

    if projected_pairs != _ALLOWED_PROJECTED_PAIRS:
      missing = sorted(_ALLOWED_PROJECTED_PAIRS - projected_pairs)
      unexpected = sorted(projected_pairs - _ALLOWED_PROJECTED_PAIRS)
      errors.append(
          f"{path}: actual common projected edges must be exact; "
          f"missing={missing}, unexpected={unexpected}"
      )
    if invalidated_incoming != invalidated:
      errors.append(
          f"{path}: every INVALIDATED phase requires an explicit causal in-edge"
      )
    reachable = set(inactive)
    queue = deque(inactive)
    while queue:
      source = queue.popleft()
      for target in adjacency.get(source, set()):
        if target not in reachable:
          reachable.add(target)
          queue.append(target)
    unreachable = sorted(set(phases) - reachable)
    if unreachable:
      errors.append(f"{path}: unreachable phases {unreachable}")

    can_terminate = set(terminal)
    queue = deque(terminal)
    while queue:
      target = queue.popleft()
      for source in reverse.get(target, set()):
        if source not in can_terminate:
          can_terminate.add(source)
          queue.append(source)
    stranded = sorted((inactive | active) - can_terminate)
    if stranded:
      errors.append(f"{path}: phases have no terminal path {stranded}")

    combined = [phase for phase in phases if "RETEST_OR_CONTINUATION" in phase]
    if combined:
      errors.append(f"{path}: retest and continuation phases must be separate")
  return errors


def _is_raw_absolute_price_denominator(value: str) -> bool:
  tokens = set(_IDENTIFIER_TOKEN_RE.findall(value.upper()))
  if "PRICE" not in tokens:
    return False
  if "RAW" in tokens:
    return True
  difference_tokens = {
      "AMPLITUDE",
      "CHANGE",
      "DELTA",
      "DIFFERENCE",
      "DISPLACEMENT",
      "DISTANCE",
      "RANGE",
      "WIDTH",
  }
  return "ABSOLUTE" in tokens and not tokens.intersection(difference_tokens)


def _namespace_and_formula_errors(contract: dict[str, Any]) -> list[str]:
  errors: list[str] = []
  geometry = contract["geometry_formula_contract"]
  expected_geometry_registries = {
      "required_fields": (
          "name",
          "numerator",
          "denominator",
          "sign",
          "unit",
          "missingness",
          "clipping_policy",
          "fit_boundary",
      ),
      "sign_values": ("SIGNED", "NONNEGATIVE", "UNIT_INTERVAL_WHEN_DEFINED"),
      "missingness_values": (
          "UNKNOWN_WITH_MASK",
          "NOT_APPLICABLE_WITH_MASK",
      ),
      "clipping_policy_values": ("NO_CLIPPING", "TRAIN_FROZEN_CLIP_POLICY"),
      "fit_boundary_values": ("NOT_APPLICABLE", "TRAIN_PARTITION_ONLY"),
  }
  for key, expected in expected_geometry_registries.items():
    if tuple(geometry[key]) != expected:
      errors.append(
          f"geometry_formula_contract/{key}: exact closed registry required"
      )
  expected_geometry_flags = {
      "absolute_price_denominator_allowed": False,
      "numeric_detector_thresholds_allowed": False,
      "causality": "PREFIX_ONLY",
      "denominator_zero_policy": "UNKNOWN_WITH_MASK",
      "insufficient_history_policy": "UNKNOWN_WITH_MASK",
      "formula_dependencies_must_be_acyclic": True,
      "every_output_has_exactly_one_formula": True,
  }
  for key, expected in expected_geometry_flags.items():
    if geometry.get(key) != expected:
      errors.append(f"geometry_formula_contract/{key}: expected {expected}")
  for family_index, family in enumerate(contract["families"]):
    names = [item["name"] for item in family["formula_contracts"]]
    if len(names) != len(set(names)):
      errors.append(
          f"families/{family_index}/formula_contracts: duplicate output"
      )
    output_names = set(names)
    dependencies = {
        formula["name"]: {
            operand
            for operand in (formula["numerator"], formula["denominator"])
            if operand in output_names
        }
        for formula in family["formula_contracts"]
    }
    resolved_outputs: set[str] = set()
    remaining_outputs = set(dependencies)
    while remaining_outputs:
      ready = {
          name
          for name in remaining_outputs
          if dependencies[name].issubset(resolved_outputs)
      }
      if not ready:
        errors.append(
            f"families/{family_index}/formula_contracts: formula dependency "
            f"cycle detected {sorted(remaining_outputs)}"
        )
        break
      resolved_outputs.update(ready)
      remaining_outputs.difference_update(ready)
    for formula_index, formula in enumerate(family["formula_contracts"]):
      if formula["sign"] not in geometry["sign_values"]:
        errors.append(
            f"families/{family_index}/formula_contracts/{formula_index}/"
            "sign: unknown"
        )
      if formula["missingness"] not in geometry["missingness_values"]:
        errors.append(
            f"families/{family_index}/formula_contracts/{formula_index}/"
            "missingness: unknown"
        )
      if formula["clipping_policy"] not in geometry["clipping_policy_values"]:
        errors.append(
            f"families/{family_index}/formula_contracts/{formula_index}/"
            "clipping_policy: unknown"
        )
      if formula["fit_boundary"] not in geometry["fit_boundary_values"]:
        errors.append(
            f"families/{family_index}/formula_contracts/{formula_index}/"
            "fit_boundary: unknown"
        )
      if _is_raw_absolute_price_denominator(formula["denominator"]):
        errors.append(
            f"families/{family_index}/formula_contracts/{formula_index}/"
            "denominator: raw/absolute price level is forbidden"
        )

  namespaces = contract["label_namespaces"]
  expected_flags = {
      "online_causal_phase": (False, False),
      "retrospective_completion": (True, False),
      "future_outcome": (True, False),
  }
  expected_namespace_fields = {
      "online_causal_phase": (
          "phase_at_as_of",
          "causal_evidence",
          "ambiguity_status",
      ),
      "retrospective_completion": (
          "completion_state",
          "completion_event_eob",
          "completion_censor_state",
      ),
      "future_outcome": (
          "outcome_profile_ref",
          "outcome_horizon_ref",
          "outcome_censor_state",
      ),
  }
  observed_fields: dict[str, set[str]] = {}
  for namespace, (future_visible, runtime_allowed) in expected_flags.items():
    value = namespaces[namespace]
    if value["future_visible"] is not future_visible:
      errors.append(
          f"label_namespaces/{namespace}/future_visible: inconsistent"
      )
    if value["runtime_input_allowed"] is not runtime_allowed:
      errors.append(
          f"label_namespaces/{namespace}/runtime_input_allowed: must be false"
      )
    if tuple(value["fields"]) != expected_namespace_fields[namespace]:
      errors.append(
          f"label_namespaces/{namespace}/fields: exact closed registry required"
      )
    observed_fields[namespace] = set(value["fields"])
  names = tuple(observed_fields)
  for index, left in enumerate(names):
    for right in names[index + 1 :]:
      overlap = observed_fields[left] & observed_fields[right]
      if overlap:
        errors.append(
            f"label_namespaces: {left}/{right} fields overlap {sorted(overlap)}"
        )
  family_strings = {
      value
      for family in contract["families"]
      for _, value in _walk(family)
      if isinstance(value, str)
  }
  leaked = sorted(
      (
          observed_fields["retrospective_completion"]
          | observed_fields["future_outcome"]
      )
      & family_strings
  )
  if leaked:
    errors.append(
        f"label_namespaces: future fields leaked into families {leaked}"
    )
  return errors


def validate_revision_payload(
    payload: Any,
    *,
    repo_root: Path | None = None,
    _schema_snapshot: bytes | None = None,
) -> list[str]:
  """Return fail-closed schema and semantic errors for MG1-A.1."""
  if not isinstance(payload, dict):
    return ["<root>: expected a mapping"]
  root = (repo_root or repository_root()).resolve()
  schema_error_list = (
      json_schema_errors(payload, repo_root=root)
      if _schema_snapshot is None
      else _json_schema_errors_from_snapshot(payload, _schema_snapshot)
  )
  if schema_error_list:
    raw_contract = payload.get("resolved_contract")
    if isinstance(raw_contract, dict):
      schema_error_list.extend(_numeric_contract_errors(raw_contract))
      raw_families = raw_contract.get("families")
      if isinstance(raw_families, list):
        schema_error_list.extend(_family_language_errors(raw_families))
    return schema_error_list

  errors, review_ids = _validate_frozen_inputs(payload, root)
  errors.extend(_draft_governance_errors(payload, review_ids))
  contract = payload["resolved_contract"]
  try:
    observed_hash = canonical_sha256(contract)
  except (TypeError, ValueError) as exc:
    errors.append(f"resolved_contract_sha256: cannot canonicalize ({exc})")
  else:
    if payload["resolved_contract_sha256"] != observed_hash:
      errors.append(
          "resolved_contract_sha256: does not match canonical resolved contract"
      )
  errors.extend(_numeric_contract_errors(contract))
  errors.extend(_object_model_errors(contract))
  errors.extend(_common_registry_errors(contract))
  errors.extend(_assumption_reference_errors(contract, review_ids))
  errors.extend(_family_p0_errors(contract))
  errors.extend(_relation_errors(contract))
  errors.extend(_phase_graph_errors(contract))
  errors.extend(_namespace_and_formula_errors(contract))
  errors.extend(_family_language_errors(contract["families"]))
  return errors


def validate_revision(
    path: Path | None = None, *, repo_root: Path | None = None
) -> tuple[dict[str, Any], list[str]]:
  root = (repo_root or repository_root()).resolve()
  config = path or revision_config_path(root)
  payload, _config_snapshot = _load_revision_snapshot(config)
  try:
    schema_snapshot = revision_schema_path(root).read_bytes()
  except OSError as exc:
    return payload, [f"schema: unavailable or invalid ({exc})"]
  return payload, validate_revision_payload(
      payload, repo_root=root, _schema_snapshot=schema_snapshot
  )


def build_validation_result(
    payload: dict[str, Any],
    errors: list[str],
    *,
    config_path: Path,
    repo_root: Path,
    config_sha256: str | None = None,
    schema_sha256: str | None = None,
) -> dict[str, Any]:
  """Build a validation envelope; this never masquerades as ontology output."""
  config_hash = config_sha256
  if config_hash is None:
    try:
      config_hash = _sha256_file(config_path)
    except OSError:
      config_hash = None
  schema_hash = schema_sha256
  if schema_hash is None:
    try:
      schema_hash = _sha256_file(revision_schema_path(repo_root))
    except OSError:
      schema_hash = None
  return {
      "profile": PROFILE_NAME,
      "milestone": MILESTONE,
      "decision": DECISION,
      "observed_decision": payload.get("decision"),
      "overall": "pass" if not errors else "fail",
      "interpretation": (
          INTERPRETATION
          if not errors
          else "invalid revision draft; MG1-B and human freeze prohibited"
      ),
      "config": str(config_path),
      "config_sha256": config_hash,
      "schema": str(revision_schema_path(repo_root)),
      "schema_sha256": schema_hash,
      "resolved_contract_sha256": payload.get("resolved_contract_sha256"),
      "errors": errors,
      "gates": {
          "v1_bytes_preserved": not errors and not any(
              error.startswith("v1_frozen/") for error in errors
          ),
          "mg0_bytes_preserved": not errors and not any(
              error.startswith("mg0_frozen/") for error in errors
          ),
          "draft_only": not errors and not any(
              error.startswith(("human_review:", "review_outcome_v1:"))
              for error in errors
          ),
          "human_freeze_allowed": False,
          "mg1b_allowed": False,
      },
  }


def render_validation_markdown(result: dict[str, Any]) -> str:
  lines = [
      "# MG1-A.1 Revision Validation",
      "",
      f"- Overall: `{'PASS' if result['overall'] == 'pass' else 'FAIL'}`",
      f"- Decision: `{result['decision']}`",
      f"- Interpretation: {result['interpretation']}",
      f"- Config SHA256: `{result['config_sha256']}`",
      f"- Resolved contract SHA256: `{result['resolved_contract_sha256']}`",
      "- Human freeze allowed: `false`",
      "- MG1-B allowed: `false`",
      "",
      "## Errors",
      "",
  ]
  if result["errors"]:
    lines.extend(f"- {error}" for error in result["errors"])
  else:
    lines.append("- None")
  return "\n".join(lines).rstrip() + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary_name = tempfile.mkstemp(
      prefix=f".{path.name}.", dir=path.parent, text=True
  )
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
      handle.write(text)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary_name, path)
  except BaseException:
    try:
      os.unlink(temporary_name)
    except OSError:
      pass
    raise


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Strictly validate the CPU-only MG1-A.1 revision draft"
  )
  parser.add_argument("--profile", default=PROFILE_NAME)
  parser.add_argument("--repo-root", type=Path)
  parser.add_argument("--config", type=Path)
  parser.add_argument("--output-json", type=Path)
  parser.add_argument("--output-md", type=Path)
  parser.add_argument(
      "--strict",
      action="store_true",
      help="Compatibility flag; validation is always strict and fail-closed.",
  )
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  if args.profile != PROFILE_NAME:
    print(f"MG1-A.1 runtime error: unsupported profile {args.profile}")
    return 2
  root = (args.repo_root or repository_root()).resolve()
  config_path = (args.config or revision_config_path(root)).resolve()
  output_paths = [
      path.resolve() for path in (args.output_json, args.output_md) if path
  ]
  protected_paths = {
      config_path,
      revision_config_path(root).resolve(),
      revision_schema_path(root).resolve(),
      Path(__file__).resolve(),
  }
  protected_paths.update(
      (root / path).resolve()
      for path in (
          *_MG0_FROZEN_HASHES,
          *_V1_FROZEN_HASHES,
          *_REVIEW_INPUT_HASHES,
      )
  )
  if (
      len(output_paths) != len(set(output_paths))
      or not protected_paths.isdisjoint(output_paths)
  ):
    print(
        "MG1-A.1 runtime error: output paths must be distinct and must not "
        "replace validation inputs"
    )
    return 2
  try:
    payload, config_snapshot = _load_revision_snapshot(config_path)
    schema_path = revision_schema_path(root)
    schema_snapshot = schema_path.read_bytes()
    config_hash = _sha256_bytes(config_snapshot)
    schema_hash = _sha256_bytes(schema_snapshot)
    errors = validate_revision_payload(
        payload, repo_root=root, _schema_snapshot=schema_snapshot
    )
    if _sha256_file(config_path) != config_hash:
      raise RevisionValidationError(
          ["config input changed during validation"]
      )
    if _sha256_file(schema_path) != schema_hash:
      raise RevisionValidationError(
          ["schema input changed during validation"]
      )
    result = build_validation_result(
        payload,
        errors,
        config_path=config_path,
        repo_root=root,
        config_sha256=config_hash,
        schema_sha256=schema_hash,
    )
    # Invalid drafts never create or replace report artifacts.
    if not errors and args.output_json:
      _atomic_write_text(
          args.output_json.resolve(),
          json.dumps(
              result, indent=2, sort_keys=True, ensure_ascii=True
          ) + "\n",
      )
    if not errors and args.output_md:
      _atomic_write_text(
          args.output_md.resolve(), render_validation_markdown(result)
      )
  except (
      OSError,
      RuntimeError,
      TypeError,
      ValueError,
      RevisionValidationError,
  ) as exc:
    print(f"MG1-A.1 runtime error: {type(exc).__name__}: {exc}")
    return 2

  print(
      "MG1-A.1 revision: "
      f"overall={result['overall']}, decision={result['decision']}, "
      "human_freeze_allowed=false, mg1b_allowed=false"
  )
  for error in errors:
    print(f"- {error}")
  return 0 if not errors else 1


if __name__ == "__main__":
  raise SystemExit(main())


__all__ = (
    "CONFIG_PATH",
    "DECISION",
    "INTERPRETATION",
    "MILESTONE",
    "PROFILE_NAME",
    "RevisionValidationError",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "build_validation_result",
    "canonical_json",
    "canonical_sha256",
    "json_schema_errors",
    "load_revision",
    "loads_json_unique",
    "loads_yaml_unique",
    "render_validation_markdown",
    "repository_root",
    "revision_config_path",
    "revision_schema_path",
    "validate_revision",
    "validate_revision_payload",
)
