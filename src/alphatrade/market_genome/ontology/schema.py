"""CPU-only schema for the MG1-A pattern ontology review draft.

This module describes a parameterized vocabulary. It does not detect patterns,
create labels, encode trading actions, or implement a frozen ontology.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mg1a.pattern_ontology.v1"
PROFILE_NAME = "mg1a"
MILESTONE = "MG1-A"
DECISION = "DRAFT_NEEDS_HUMAN_FREEZE"
REVIEW_STATUS = "NEEDS_HUMAN_REVIEW"


class LifecyclePhase(str, Enum):
  """Common lifecycle phases required by the MG1-A plan."""

  INACTIVE = "INACTIVE"
  CANDIDATE = "CANDIDATE"
  DEVELOPING = "DEVELOPING"
  MATURE = "MATURE"
  CONFIRMED = "CONFIRMED"
  RETEST_OR_CONTINUATION = "RETEST_OR_CONTINUATION"
  COMPLETED = "COMPLETED"
  INVALIDATED = "INVALIDATED"


class PatternFamily(str, Enum):
  """Initial pattern families required by the MG1-A plan."""

  TREND_UP = "trend_up"
  TREND_DOWN = "trend_down"
  TREND_TRANSITION = "trend_transition"
  DOUBLE_TOP = "double_top"
  DOUBLE_BOTTOM = "double_bottom"
  HEAD_SHOULDERS_TOP = "head_shoulders_top"
  INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
  BREAKOUT = "breakout"
  FAILED_BREAKOUT = "failed_breakout"
  RETEST = "retest"
  RANGE = "range"
  SUPPORT_RESISTANCE_CONVERSION = "support_resistance_conversion"


class PatternSide(str, Enum):
  """Structural direction only; values are not trading actions."""

  BULLISH = "BULLISH"
  BEARISH = "BEARISH"
  NEUTRAL = "NEUTRAL"


class LabelProvenance(str, Enum):
  """Allowed provenance categories from the technical report."""

  RULE_DERIVED = "RULE_DERIVED"
  HUMAN_REVIEWED = "HUMAN_REVIEWED"
  SYNTHETIC = "SYNTHETIC"
  AUTO_MODEL_ASSISTED = "AUTO_MODEL_ASSISTED"


COMMON_PHASES = tuple(phase.value for phase in LifecyclePhase)
REQUIRED_FAMILIES = tuple(family.value for family in PatternFamily)
ALLOWED_PROVENANCE = tuple(item.value for item in LabelProvenance)
ALLOWED_PATTERN_SIDES = tuple(item.value for item in PatternSide)
SIDE_INSTANCE_PARAMETER = "INSTANCE_PARAMETER"

PRICE_UNITS = (
    "ATR_MULTIPLE",
    "REALIZED_VOL_MULTIPLE",
    "TRAIN_FROZEN_SCALE_MULTIPLE",
    "DIMENSIONLESS_RATIO",
)
DURATION_UNIT = "SCALE_WINDOW_STATISTIC_RATIO"

REQUIRED_ANCHOR_FIELDS = (
    "role",
    "event_eob",
    "available_as_of_eob",
    "normalized_price",
    "scale_id",
    "source_method",
    "source_bar_closed",
    "anchor_state",
    "label_provenance",
    "assumption_ids",
)
REQUIRED_PATTERN_INSTANCE_FIELDS = (
    "pattern_id",
    "family",
    "side",
    "scale_id",
    "base_timeframe",
    "start_eob",
    "as_of_eob",
    "phase_observed_at",
    "phase",
    "anchors",
    "topology",
    "normalized_geometry",
    "duration_ratios",
    "price_ratios",
    "prior_context",
    "online_confirmation",
    "invalidation",
    "maturity_score",
    "label_provenance",
    "detector_version",
    "assumption_ids",
    "session_calendar_version",
    "active_contract",
    "roll_state",
)
LABEL_NAMESPACE_FIELDS = {
    "online_causal_phase": (
        "family",
        "phase",
        "phase_observed_at",
        "maturity_score",
    ),
    "retrospective_completion": (
        "completion_status",
        "completed_at",
        "final_phase",
    ),
    "future_outcome": ("target_class", "target_horizon"),
}

_FAMILY_SIDE_BASIS = {
    "trend_up": "FIXED_FAMILY_SIDE",
    "trend_down": "FIXED_FAMILY_SIDE",
    "trend_transition": "COUNTER_SEQUENCE_DIRECTION",
    "double_top": "FIXED_FAMILY_SIDE",
    "double_bottom": "FIXED_FAMILY_SIDE",
    "head_shoulders_top": "FIXED_FAMILY_SIDE",
    "inverse_head_shoulders": "FIXED_FAMILY_SIDE",
    "breakout": "BOUNDARY_CROSS_DIRECTION",
    "failed_breakout": "ATTEMPT_CROSS_DIRECTION",
    "retest": "ORIGIN_BREAK_DIRECTION",
    "range": "FIXED_FAMILY_SIDE",
    "support_resistance_conversion": "ROLE_BREAK_DIRECTION",
}
_FAMILY_ALLOWED_SIDES = {
    "trend_up": ("BULLISH",),
    "trend_down": ("BEARISH",),
    "trend_transition": ("BULLISH", "BEARISH"),
    "double_top": ("BEARISH",),
    "double_bottom": ("BULLISH",),
    "head_shoulders_top": ("BEARISH",),
    "inverse_head_shoulders": ("BULLISH",),
    "breakout": ("BULLISH", "BEARISH"),
    "failed_breakout": ("BULLISH", "BEARISH"),
    "retest": ("BULLISH", "BEARISH"),
    "range": ("NEUTRAL",),
    "support_resistance_conversion": ("BULLISH", "BEARISH"),
}
_INSTANCE_SIDE_FAMILIES = {
    "trend_transition",
    "breakout",
    "failed_breakout",
    "retest",
    "support_resistance_conversion",
}
_REQUIRED_TOPOLOGY_RELATIONS = {
    "trend_transition": {
        "STRUCTURE_BREAK_CROSSES_LAST_DEFENDING_PIVOT",
    },
    "head_shoulders_top": {
        "HEAD_PEAK_ABOVE_BOTH_SHOULDERS",
        "SHOULDER_PEAK_LEVELS_COMPATIBLE",
    },
    "inverse_head_shoulders": {
        "HEAD_TROUGH_BELOW_BOTH_SHOULDERS",
        "SHOULDER_TROUGH_LEVELS_COMPATIBLE",
    },
}
_REQUIRED_TOPOLOGY_ANCHORS = {
    "trend_transition": {
        "STRUCTURE_BREAK_CROSSES_LAST_DEFENDING_PIVOT": (
            "last_defending_pivot",
            "structure_break",
        ),
    },
    "head_shoulders_top": {
        "HEAD_PEAK_ABOVE_BOTH_SHOULDERS": (
            "head_peak",
            "left_shoulder_peak",
            "right_shoulder_peak",
        ),
        "SHOULDER_PEAK_LEVELS_COMPATIBLE": (
            "left_shoulder_peak",
            "right_shoulder_peak",
        ),
    },
    "inverse_head_shoulders": {
        "HEAD_TROUGH_BELOW_BOTH_SHOULDERS": (
            "head_trough",
            "left_shoulder_trough",
            "right_shoulder_trough",
        ),
        "SHOULDER_TROUGH_LEVELS_COMPATIBLE": (
            "left_shoulder_trough",
            "right_shoulder_trough",
        ),
    },
}
_SIDE_BASIS_REQUIREMENTS = {
    "trend_transition": {
        "anchors": ("structure_break", "first_counter_pivot"),
        "relations": ("COUNTER_SEQUENCE_EMERGES_AFTER_BREAK",),
    },
    "breakout": {
        "anchors": ("boundary_cross",),
        "relations": ("CROSSES_BOUNDARY_OUTWARD_WITH_OPTIONAL_HOLD",),
    },
    "failed_breakout": {
        "anchors": ("attempt_cross",),
        "relations": ("ATTEMPTS_OUTWARD_CROSS",),
    },
    "retest": {
        "anchors": ("origin_break",),
        "relations": ("FOLLOWS_SOURCE_BREAK_AND_RETURNS_FROM_OUTSIDE",),
    },
    "support_resistance_conversion": {
        "anchors": ("role_break",),
        "relations": ("ORIGINAL_ROLE_ZONE_IS_CROSSED",),
    },
}
_FAMILY_ASSUMPTION_CATEGORIES = (
    "STRUCTURE",
    "PHASE",
    "CONFIRMATION",
    "INVALIDATION",
)
_FAMILY_STRUCTURE_PARAMETER_REFS = {
    "trend_up": (
        "trend_up.sequence_order_policy",
        "trend_up.normalized_order_tolerance",
        "trend_up.geometry_policy",
        "trend_up.duration_policy",
        "trend_up.price_ratio_policy",
    ),
    "trend_down": (
        "trend_down.sequence_order_policy",
        "trend_down.normalized_order_tolerance",
        "trend_down.geometry_policy",
        "trend_down.duration_policy",
        "trend_down.price_ratio_policy",
    ),
    "trend_transition": (
        "trend_transition.side_basis_policy",
        "trend_transition.order_loss_policy",
        "trend_transition.break_and_counter_policy",
        "trend_transition.geometry_policy",
        "trend_transition.duration_policy",
        "trend_transition.price_ratio_policy",
    ),
    "double_top": (
        "double_top.peak_compatibility_policy",
        "double_top.neckline_policy",
        "double_top.geometry_policy",
        "double_top.duration_policy",
        "double_top.price_ratio_policy",
    ),
    "double_bottom": (
        "double_bottom.trough_compatibility_policy",
        "double_bottom.neckline_policy",
        "double_bottom.geometry_policy",
        "double_bottom.duration_policy",
        "double_bottom.price_ratio_policy",
    ),
    "head_shoulders_top": (
        "head_shoulders_top.shoulder_head_policy",
        "head_shoulders_top.neckline_policy",
        "head_shoulders_top.geometry_policy",
        "head_shoulders_top.duration_policy",
        "head_shoulders_top.price_ratio_policy",
    ),
    "inverse_head_shoulders": (
        "inverse_head_shoulders.shoulder_head_policy",
        "inverse_head_shoulders.neckline_policy",
        "inverse_head_shoulders.geometry_policy",
        "inverse_head_shoulders.duration_policy",
        "inverse_head_shoulders.price_ratio_policy",
    ),
    "breakout": (
        "breakout.side_basis_policy",
        "breakout.zone_construction_policy",
        "breakout.cross_and_hold_policy",
        "breakout.geometry_policy",
        "breakout.duration_policy",
        "breakout.price_ratio_policy",
    ),
    "failed_breakout": (
        "failed_breakout.side_basis_policy",
        "failed_breakout.attempt_policy",
        "failed_breakout.reentry_policy",
        "failed_breakout.geometry_policy",
        "failed_breakout.duration_policy",
        "failed_breakout.price_ratio_policy",
    ),
    "retest": (
        "retest.side_basis_policy",
        "retest.source_and_return_policy",
        "retest.hold_policy",
        "retest.geometry_policy",
        "retest.duration_policy",
        "retest.price_ratio_policy",
    ),
    "range": (
        "range.boundary_and_touch_policy",
        "range.containment_policy",
        "range.geometry_policy",
        "range.duration_policy",
        "range.price_ratio_policy",
    ),
    "support_resistance_conversion": (
        "support_resistance_conversion.side_basis_policy",
        "support_resistance_conversion.break_policy",
        "support_resistance_conversion.role_conversion_policy",
        "support_resistance_conversion.geometry_policy",
        "support_resistance_conversion.duration_policy",
        "support_resistance_conversion.price_ratio_policy",
    ),
}
_FAMILY_IDENTIFIER_SURFACE_HASHES = {
    "trend_up": (
        "647c8d0af3f166c895c8a3cc47178c8106a9046c0defee2aafdd6767a9226c3d"
    ),
    "trend_down": (
        "0376289555fedb8292c6f38bab7645b974d3bf64f22aa85113ac5e2b80db0fe4"
    ),
    "trend_transition": (
        "6f4f5e75ca98079ac2ad443a17b0411c472a295ec9d6cc02761435a61e57d66f"
    ),
    "double_top": (
        "163fb80230b61c0d6011604bf7bcd94582b3c2ce53729fe4c530536cd6b4b608"
    ),
    "double_bottom": (
        "b8ed1b5ce94be4cf41bd0203bfb23bb25600c127736ff183fce4de710d71b2da"
    ),
    "head_shoulders_top": (
        "2554da7a41f181b2ac205c6993007c96176d9db8733c6d267a177f1e3538102c"
    ),
    "inverse_head_shoulders": (
        "ce89f9da9c438898589c237dc8554f5fa9d0f2e6af2615d401a6517c1d391391"
    ),
    "breakout": (
        "3b243f1dd14239834526f2db5baceb6f70ec09cd74d21392faeda4c00d52fe25"
    ),
    "failed_breakout": (
        "5c6152171384186de34bafb084a21292ca7305f9b07725f4005ea7f8e78623ff"
    ),
    "retest": (
        "96f0cd680a554540c193745cd00fe5cfeae92966683fc1c8d22447bb972b58c5"
    ),
    "range": (
        "9dcc4dbdfaeb84c72e2c4af3ab608424f9bcab00c2c54b509f1cd16f3b39c881"
    ),
    "support_resistance_conversion": (
        "6a43713f5eeb0357a10ba9e87c6308a7c25ebf0d881a6bfaa3e9be915669677c"
    ),
}
_FAMILY_ROOT_COMMON_ASSUMPTIONS = {
    "trend_up": (
        "MG1A.COMMON.SCALE.001",
        "MG1A.COMMON.NORMALIZATION.001",
    ),
    "trend_down": (
        "MG1A.COMMON.MIRROR.001",
        "MG1A.COMMON.SCALE.001",
    ),
    "trend_transition": (
        "MG1A.COMMON.OVERLAP.001",
        "MG1A.COMMON.SCALE.001",
    ),
    "double_top": (
        "MG1A.COMMON.ZONE.001",
        "MG1A.COMMON.SCALE.001",
    ),
    "double_bottom": (
        "MG1A.COMMON.MIRROR.001",
        "MG1A.COMMON.ZONE.001",
    ),
    "head_shoulders_top": (
        "MG1A.COMMON.ZONE.001",
        "MG1A.COMMON.SCALE.001",
    ),
    "inverse_head_shoulders": (
        "MG1A.COMMON.MIRROR.001",
        "MG1A.COMMON.ZONE.001",
    ),
    "breakout": (
        "MG1A.COMMON.ZONE.001",
        "MG1A.COMMON.OVERLAP.001",
    ),
    "failed_breakout": (
        "MG1A.COMMON.ZONE.001",
        "MG1A.COMMON.OVERLAP.001",
    ),
    "retest": (
        "MG1A.COMMON.ZONE.001",
        "MG1A.COMMON.OVERLAP.001",
    ),
    "range": (
        "MG1A.COMMON.ZONE.001",
        "MG1A.COMMON.OVERLAP.001",
    ),
    "support_resistance_conversion": (
        "MG1A.COMMON.ZONE.001",
        "MG1A.COMMON.OVERLAP.001",
    ),
}
_COMMON_ASSUMPTION_CONTRACTS = {
    "MG1A.COMMON.LIFECYCLE.001": (
        "PHASE",
        ("ALL",),
        ("common_lifecycle",),
    ),
    "MG1A.COMMON.SCALE.001": (
        "SCALE",
        ("ALL",),
        ("scale_contract",),
    ),
    "MG1A.COMMON.NORMALIZATION.001": (
        "NORMALIZATION",
        ("ALL",),
        ("normalization_contract",),
    ),
    "MG1A.COMMON.SWING_CAUSALITY.001": (
        "CAUSALITY",
        ("ALL",),
        ("anchor_instance_contract",),
    ),
    "MG1A.COMMON.PARTIAL_BAR.001": (
        "CAUSALITY",
        ("ALL",),
        ("anchor_instance_contract.partial_bar_policy",),
    ),
    "MG1A.COMMON.SESSION.001": (
        "CONTEXT",
        ("ALL",),
        ("pattern_instance_contract.session_calendar_version",),
    ),
    "MG1A.COMMON.ROLL.001": (
        "CONTEXT",
        ("ALL",),
        ("pattern_instance_contract.roll_state",),
    ),
    "MG1A.COMMON.PROVENANCE.001": (
        "PROVENANCE",
        ("ALL",),
        ("label_provenance_contract",),
    ),
    "MG1A.COMMON.MIRROR.001": (
        "STRUCTURE",
        ("trend_down", "double_bottom", "inverse_head_shoulders"),
        ("mirror_transform",),
    ),
    "MG1A.COMMON.OVERLAP.001": (
        "IDENTITY",
        (
            "trend_transition",
            "breakout",
            "failed_breakout",
            "retest",
            "range",
            "support_resistance_conversion",
        ),
        ("overlap_precedence",),
    ),
    "MG1A.COMMON.ZONE.001": (
        "STRUCTURE",
        (
            "double_top",
            "double_bottom",
            "head_shoulders_top",
            "inverse_head_shoulders",
            "breakout",
            "failed_breakout",
            "retest",
            "range",
            "support_resistance_conversion",
        ),
        ("zone_construction",),
    ),
    "MG1A.COMMON.FILTER_MISSINGNESS.001": (
        "CONFIRMATION",
        ("ALL",),
        ("optional_volume_oi_confirmations",),
    ),
    "MG1A.COMMON.LABEL_TIMING.001": (
        "LABEL",
        ("ALL",),
        ("label_namespaces",),
    ),
}

COMMON_TRANSITIONS = (
    ("INACTIVE", "CANDIDATE"),
    ("CANDIDATE", "DEVELOPING"),
    ("DEVELOPING", "MATURE"),
    ("MATURE", "CONFIRMED"),
    ("CONFIRMED", "RETEST_OR_CONTINUATION"),
    ("RETEST_OR_CONTINUATION", "COMPLETED"),
    ("CANDIDATE", "INVALIDATED"),
    ("DEVELOPING", "INVALIDATED"),
    ("MATURE", "INVALIDATED"),
    ("CONFIRMED", "INVALIDATED"),
    ("RETEST_OR_CONTINUATION", "INVALIDATED"),
    ("COMPLETED", "INACTIVE"),
    ("INVALIDATED", "INACTIVE"),
)

_FIELD_SECTIONS = (
    "normalized_geometry_fields",
    "duration_ratios",
    "price_ratios",
    "prior_context_fields",
    "online_confirmation_fields",
    "invalidation_fields",
)
_FORBIDDEN_KEY_FRAGMENTS = (
    "absolute_price",
    "raw_price",
    "buy",
    "sell",
    "entry",
    "trade",
    "action",
    "profit",
    "rating",
    "stop_price",
    "teacher",
    "mfe",
    "mae",
    "long_signal",
    "short_signal",
    "long_position",
    "short_position",
    "lookahead",
    "pnl",
    "order_intent",
    "trade_intent",
    "signal_intent",
    "close_price",
    "open_price",
    "high_price",
    "low_price",
    "settlement_price",
    "last_price",
    "usd",
    "eur",
    "cny",
    "jpy",
    "gbp",
    "hkd",
    "seconds",
    "minutes",
    "hours",
    "days",
    "milliseconds",
)
_FORBIDDEN_FIELD_FRAGMENTS = _FORBIDDEN_KEY_FRAGMENTS
_FUTURE_CAUSAL_TOKENS = (
    "future",
    "outcome",
    "retrospective",
    "completion",
    "target",
    "next",
    "forward",
    "lookahead",
    "post_event",
    "tomorrow",
)
_VOLUME_TOPOLOGY_TOKENS = ("volume", "open_interest", "oi_")


class OntologyValidationError(ValueError):
  """Raised when an ontology payload violates the MG1-A draft contract."""

  def __init__(self, errors: list[str]):
    self.errors = tuple(errors)
    super().__init__("; ".join(errors))


def canonical_json(value: Any) -> str:
  """Return deterministic ASCII JSON for hashing and round trips."""
  return json.dumps(
      value,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=True,
      allow_nan=False,
  )


def canonical_sha256(value: Any) -> str:
  """Return SHA256 of :func:`canonical_json`."""
  return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def loads_json_unique(text: str) -> Any:
  """Parse JSON while rejecting duplicate object keys."""

  def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f"duplicate mapping key: {key}")
      result[key] = value
    return result

  return json.loads(text, object_pairs_hook=unique_object)


def loads_yaml_unique(text: str) -> Any:
  """Parse safe YAML while rejecting explicit duplicate mapping keys."""
  try:
    import yaml
  except ImportError as exc:  # pragma: no cover - dependency guard
    message = f"PyYAML is required to load the ontology: {exc}"
    raise RuntimeError(message) from exc

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
        raise yaml.constructor.ConstructorError(
            "while constructing a mapping",
            node.start_mark,
            "found an unhashable mapping key",
            key_node.start_mark,
        ) from exc
      if duplicate:
        raise yaml.constructor.ConstructorError(
            "while constructing a mapping",
            node.start_mark,
            f"found duplicate key {key!r}",
            key_node.start_mark,
        )
      explicit_keys.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

  UniqueKeySafeLoader.add_constructor(
      yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
      construct_mapping,
  )
  try:
    return yaml.load(text, Loader=UniqueKeySafeLoader)
  except yaml.YAMLError as exc:
    raise OntologyValidationError([f"invalid YAML: {exc}"]) from exc


@dataclass(frozen=True)
class ScaleBoundFamily:
  """A family definition plus instance-supplied scale metadata."""

  family: str
  side: str
  scale_id: str
  base_timeframe: str
  definition_sha256: str
  definition: dict[str, Any]

  def to_dict(self) -> dict[str, Any]:
    return {
        "family": self.family,
        "side": self.side,
        "scale_id": self.scale_id,
        "base_timeframe": self.base_timeframe,
        "definition_sha256": self.definition_sha256,
        "definition": copy.deepcopy(self.definition),
    }


@dataclass(frozen=True)
class PatternOntologyDraft:
  """Validated MG1-A review draft with deterministic serialization."""

  _payload: dict[str, Any]

  @classmethod
  def from_dict(cls, payload: dict[str, Any]) -> PatternOntologyDraft:
    try:
      normalized = json.loads(canonical_json(payload))
    except (TypeError, ValueError, OverflowError) as exc:
      raise OntologyValidationError(
          [f"<root>: payload is not canonical JSON: {exc}"]
      ) from exc
    structural_errors = json_schema_errors(normalized)
    if structural_errors:
      raise OntologyValidationError(structural_errors)
    errors = validate_ontology_payload(normalized)
    if errors:
      raise OntologyValidationError(errors)
    return cls(_payload=normalized)

  def to_dict(self) -> dict[str, Any]:
    return copy.deepcopy(self._payload)

  def to_json(self) -> str:
    return canonical_json(self._payload)

  @property
  def resolved_contract_sha256(self) -> str:
    return str(self._payload["resolved_contract_sha256"])

  @property
  def families(self) -> tuple[str, ...]:
    return tuple(
        item["family"]
        for item in self._payload["resolved_contract"]["families"]
    )

  @property
  def phases(self) -> tuple[str, ...]:
    return tuple(
        self._payload["resolved_contract"]["common_lifecycle"]["phases"]
    )

  def transition_allowed(self, from_phase: str, to_phase: str) -> bool:
    transitions = self._payload["resolved_contract"]["common_lifecycle"][
        "transitions"
    ]
    return any(
        item["from_phase"] == from_phase and item["to_phase"] == to_phase
        for item in transitions
    )

  def require_transition(self, from_phase: str, to_phase: str) -> None:
    if from_phase not in self.phases or to_phase not in self.phases:
      raise OntologyValidationError(
          [f"unknown phase transition endpoint: {from_phase}->{to_phase}"]
      )
    if not self.transition_allowed(from_phase, to_phase):
      raise OntologyValidationError(
          [f"undeclared phase transition: {from_phase}->{to_phase}"]
      )

  def bind_family(
      self,
      family: str,
      *,
      scale_id: str,
      base_timeframe: str,
      side: str | None = None,
  ) -> ScaleBoundFamily:
    if (
        not isinstance(scale_id, str)
        or not scale_id
        or scale_id == SIDE_INSTANCE_PARAMETER
    ):
      raise ValueError("scale_id must be an instance-supplied non-placeholder")
    if (
        not isinstance(base_timeframe, str)
        or not base_timeframe
        or base_timeframe == SIDE_INSTANCE_PARAMETER
    ):
      raise ValueError(
          "base_timeframe must be an instance-supplied non-placeholder"
      )
    definitions = {
        item["family"]: item
        for item in self._payload["resolved_contract"]["families"]
    }
    if family not in definitions:
      raise KeyError(f"unknown family: {family}")
    definition = copy.deepcopy(definitions[family])
    declared_side = definition["side"]
    allowed_sides = tuple(definition["allowed_instance_sides"])
    if declared_side == SIDE_INSTANCE_PARAMETER:
      if side is None:
        raise ValueError(f"side is required for family: {family}")
      bound_side = side
    else:
      bound_side = declared_side if side is None else side
    if not isinstance(bound_side, str) or bound_side not in allowed_sides:
      raise ValueError(
          f"side {bound_side!r} is not allowed for family: {family}"
      )
    return ScaleBoundFamily(
        family=family,
        side=bound_side,
        scale_id=scale_id,
        base_timeframe=base_timeframe,
        definition_sha256=canonical_sha256(definition),
        definition=definition,
    )


def load_ontology(path: Path) -> PatternOntologyDraft:
  """Load and validate a YAML ontology without importing model libraries."""
  try:
    payload = loads_yaml_unique(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, RuntimeError) as exc:
    raise OntologyValidationError(
        [f"<root>: ontology YAML unavailable or invalid: {exc}"]
    ) from exc
  if not isinstance(payload, dict):
    raise OntologyValidationError(["<root>: expected a mapping"])
  return PatternOntologyDraft.from_dict(payload)


def report_schema_path() -> Path:
  """Return the independent Draft-07 schema for the MG1-A payload."""
  return Path(__file__).with_name("mg1a_pattern_ontology.schema.json")


def json_schema_errors(
    payload: Any, schema_path: Path | None = None
) -> list[str]:
  """Return stable structural errors without importing model libraries."""
  try:
    import jsonschema
  except ImportError as exc:  # pragma: no cover - dependency guard
    return [f"jsonschema unavailable: {exc}"]
  try:
    schema_payload = loads_json_unique(
        (schema_path or report_schema_path()).read_text(encoding="utf-8")
    )
    jsonschema.Draft7Validator.check_schema(schema_payload)
  except (
      OSError,
      UnicodeError,
      ValueError,
      jsonschema.SchemaError,
  ) as exc:
    return [f"ontology schema unavailable or invalid: {exc}"]
  validator = jsonschema.Draft7Validator(schema_payload)
  errors = sorted(
      validator.iter_errors(payload),
      key=lambda error: tuple(str(part) for part in error.path),
  )
  return [
      (
          f"{'/'.join(str(part) for part in error.path) or '<root>'}: "
          f"{error.message}"
      )
      for error in errors
  ]


def _as_mapping(value: Any) -> dict[str, Any]:
  return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
  return value if isinstance(value, list) else []


def _flatten_strings(value: Any) -> list[str]:
  if isinstance(value, str):
    return [value]
  if isinstance(value, dict):
    result: list[str] = []
    for key, nested in value.items():
      result.extend([str(key), *_flatten_strings(nested)])
    return result
  if isinstance(value, list):
    result = []
    for nested in value:
      result.extend(_flatten_strings(nested))
    return result
  return []


def _contains_identifier_fragment(value: str, fragment: str) -> bool:
  tokens = value.lower().replace(".", "_").split("_")
  expected = fragment.split("_")
  width = len(expected)
  return any(
      tokens[index : index + width] == expected
      for index in range(len(tokens) - width + 1)
  )


def _find_forbidden_keys(value: Any, path: str = "<root>") -> list[str]:
  errors: list[str] = []
  if isinstance(value, dict):
    for key, nested in value.items():
      normalized = str(key).lower()
      if any(
          _contains_identifier_fragment(normalized, fragment)
          for fragment in _FORBIDDEN_KEY_FRAGMENTS
      ):
        errors.append(f"{path}/{key}: forbidden ontology key")
      errors.extend(_find_forbidden_keys(nested, f"{path}/{key}"))
  elif isinstance(value, list):
    for index, nested in enumerate(value):
      errors.extend(_find_forbidden_keys(nested, f"{path}/{index}"))
  return errors


def _find_forbidden_strings(
    value: Any,
    fragments: tuple[str, ...],
    path: str,
    detail: str,
) -> list[str]:
  errors: list[str] = []
  if isinstance(value, str):
    normalized = value.lower()
    if any(
        _contains_identifier_fragment(normalized, fragment)
        for fragment in fragments
    ):
      errors.append(f"{path}: {detail}: {value}")
  elif isinstance(value, dict):
    for key, nested in value.items():
      errors.extend(
          _find_forbidden_strings(
              nested, fragments, f"{path}/{key}", detail
          )
      )
  elif isinstance(value, list):
    for index, nested in enumerate(value):
      errors.extend(
          _find_forbidden_strings(
              nested, fragments, f"{path}/{index}", detail
          )
      )
  return errors


def _assumption_ids(
    item: dict[str, Any], path: str, errors: list[str]
) -> list[str]:
  values = item.get("assumption_ids")
  if not isinstance(values, list) or not values:
    errors.append(f"{path}/assumption_ids: non-empty provenance is required")
    return []
  if not all(isinstance(value, str) and value for value in values):
    errors.append(f"{path}/assumption_ids: values must be non-empty strings")
    return []
  if len(values) != len(set(values)):
    errors.append(f"{path}/assumption_ids: duplicate references")
  return values


def _require_assumption_reference(
    item: dict[str, Any],
    identifier: str,
    path: str,
    errors: list[str],
) -> None:
  if identifier not in _as_list(item.get("assumption_ids")):
    errors.append(f"{path}/assumption_ids: required reference {identifier}")


def _require_exact_assumption_references(
    item: dict[str, Any],
    identifiers: tuple[str, ...],
    path: str,
    errors: list[str],
) -> None:
  if tuple(_as_list(item.get("assumption_ids"))) != identifiers:
    errors.append(f"{path}/assumption_ids: exact provenance required")


def _valid_parameter_reference(value: Any, *, allow_wildcard: bool) -> bool:
  if not isinstance(value, str) or not value:
    return False
  wildcard = value.endswith(".*")
  if wildcard and not allow_wildcard:
    return False
  base = value[:-2] if wildcard else value
  if "*" in base:
    return False
  parts = base.split(".")
  if wildcard and len(parts) < 2:
    return False
  return bool(parts) and all(
      part
      and part[0].isalpha()
      and part[0].isascii()
      and all(
          char.isascii() and (char.isalnum() or char == "_")
          for char in part
      )
      for part in parts
  )


def _parameter_refs(
    item: dict[str, Any],
    path: str,
    assumption_registry: dict[str, dict[str, Any]],
    registration_ids: tuple[str, ...],
    errors: list[str],
) -> list[str]:
  values = item.get("parameter_refs")
  if not isinstance(values, list) or not values:
    errors.append(f"{path}/parameter_refs: non-empty references required")
    return []
  if not all(
      _valid_parameter_reference(value, allow_wildcard=False)
      for value in values
  ):
    errors.append(f"{path}/parameter_refs: invalid concrete reference")
    return []
  if len(values) != len(set(values)):
    errors.append(f"{path}/parameter_refs: duplicate references")

  patterns: list[str] = []
  for identifier in registration_ids:
    assumption = assumption_registry.get(identifier)
    if assumption:
      patterns.extend(
          pattern
          for pattern in _as_list(assumption.get("parameter_refs"))
          if _valid_parameter_reference(pattern, allow_wildcard=True)
      )

  def registered(reference: str) -> bool:
    return any(
        reference == pattern
        or (
            isinstance(pattern, str)
            and pattern.endswith(".*")
            and reference.startswith(pattern[:-1])
        )
        for pattern in patterns
    )

  for reference in values:
    if isinstance(reference, str) and not registered(reference):
      errors.append(
          f"{path}/parameter_refs: unregistered reference {reference}"
      )
  return values


def _validate_label_namespaces(
    contract: dict[str, Any], errors: list[str]
) -> None:
  namespaces = _as_mapping(contract.get("label_namespaces"))
  expected = {
      "online_causal_phase",
      "retrospective_completion",
      "future_outcome",
  }
  if set(namespaces) != expected:
    errors.append(
        "resolved_contract/label_namespaces: exact namespaces required"
    )
    return

  online = _as_mapping(namespaces.get("online_causal_phase"))
  retrospective = _as_mapping(namespaces.get("retrospective_completion"))
  outcome = _as_mapping(namespaces.get("future_outcome"))
  if (
      online.get("information_cutoff") != "PREFIX_THROUGH_AS_OF"
      or online.get("runtime_input_allowed") is not False
      or online.get("target_only") is not False
  ):
    errors.append("label_namespaces/online_causal_phase: causal flags invalid")
  for name, item, cutoff in (
      (
          "retrospective_completion",
          retrospective,
          "FUTURE_ALLOWED_TARGET_ONLY",
      ),
      ("future_outcome", outcome, "POST_CONFIRMATION_FUTURE_TARGET_ONLY"),
  ):
    if (
        item.get("information_cutoff") != cutoff
        or item.get("runtime_input_allowed") is not False
        or item.get("target_only") is not True
    ):
      errors.append(f"label_namespaces/{name}: target isolation flags invalid")
  observed_field_sets: list[set[Any]] = []
  for name, item in namespaces.items():
    _assumption_ids(_as_mapping(item), f"label_namespaces/{name}", errors)
    _require_assumption_reference(
        _as_mapping(item),
        "MG1A.COMMON.LABEL_TIMING.001",
        f"label_namespaces/{name}",
        errors,
    )
    _require_exact_assumption_references(
        _as_mapping(item),
        ("MG1A.COMMON.LABEL_TIMING.001",),
        f"label_namespaces/{name}",
        errors,
    )
    fields = tuple(_as_list(_as_mapping(item).get("fields")))
    observed_field_sets.append(
        {value for value in fields if isinstance(value, str)}
    )
    if fields != LABEL_NAMESPACE_FIELDS[name]:
      errors.append(f"label_namespaces/{name}: exact fields required")

  if any(
      left & right
      for index, left in enumerate(observed_field_sets)
      for right in observed_field_sets[index + 1 :]
  ):
    errors.append("label_namespaces: namespace fields must be disjoint")


def _validate_assumptions(
    contract: dict[str, Any], errors: list[str]
) -> set[str]:
  items = _as_list(contract.get("assumptions"))
  identifiers = [
      item.get("assumption_id")
      for item in items
      if isinstance(item, dict)
  ]
  if not items or len(identifiers) != len(items):
    errors.append(
        "resolved_contract/assumptions: records with IDs are required"
    )
  if any(not isinstance(value, str) or not value for value in identifiers):
    errors.append(
        "resolved_contract/assumptions: IDs must be non-empty strings"
    )
  if len(identifiers) != len(set(identifiers)):
    errors.append("resolved_contract/assumptions: duplicate assumption_id")
  family_by_prefix = {
      family.upper(): family for family in REQUIRED_FAMILIES
  }
  expected_identifiers = set(_COMMON_ASSUMPTION_CONTRACTS)
  expected_identifiers.update(
      f"MG1A.{family.upper()}.{category}.001"
      for family in REQUIRED_FAMILIES
      for category in _FAMILY_ASSUMPTION_CATEGORIES
  )
  if set(identifiers) != expected_identifiers:
    errors.append(
        "resolved_contract/assumptions: exact common and family IDs required"
    )
  for index, raw in enumerate(items):
    item = _as_mapping(raw)
    path = f"resolved_contract/assumptions/{index}"
    if item.get("status") != REVIEW_STATUS:
      errors.append(f"{path}/status: must remain {REVIEW_STATUS}")
    if item.get("proposed_default") is not None:
      errors.append(f"{path}/proposed_default: must remain null")
    if item.get("reviewer_decision") is not None:
      errors.append(f"{path}/reviewer_decision: must remain null")
    for field_name in ("parameter_refs", "source_evidence"):
      values = _as_list(item.get(field_name))
      if not values or not all(
          isinstance(value, str) and value for value in values
      ):
        errors.append(f"{path}/{field_name}: non-empty strings required")
      elif len(values) != len(set(values)):
        errors.append(f"{path}/{field_name}: duplicate values")
    parameter_refs = _as_list(item.get("parameter_refs"))
    if not all(
        _valid_parameter_reference(value, allow_wildcard=True)
        for value in parameter_refs
    ):
      errors.append(f"{path}/parameter_refs: invalid registration pattern")
    affected = _as_list(item.get("affected_families"))
    if not affected or not all(
        isinstance(value, str)
        and (value == "ALL" or value in REQUIRED_FAMILIES)
        for value in affected
    ):
      errors.append(f"{path}/affected_families: unknown family reference")
    if "ALL" in affected and affected != ["ALL"]:
      errors.append(f"{path}/affected_families: ALL must be used alone")
    identifier = item.get("assumption_id")
    parts = identifier.split(".") if isinstance(identifier, str) else []
    common_contract = _COMMON_ASSUMPTION_CONTRACTS.get(identifier)
    if common_contract:
      expected_category, expected_affected, expected_parameters = (
          common_contract
      )
      if item.get("category") != expected_category:
        errors.append(f"{path}/category: exact common category required")
      if tuple(affected) != expected_affected:
        errors.append(
            f"{path}/affected_families: exact common ownership required"
        )
      if tuple(parameter_refs) != expected_parameters:
        errors.append(
            f"{path}/parameter_refs: exact common registrations required"
        )
    elif len(parts) == 4 and parts[1] in family_by_prefix:
      expected_family = family_by_prefix[parts[1]]
      expected_category = parts[2]
      if expected_category not in _FAMILY_ASSUMPTION_CATEGORIES:
        errors.append(f"{path}/assumption_id: invalid family category")
      if item.get("category") != expected_category:
        errors.append(f"{path}/category: must match assumption_id")
      if affected != [expected_family]:
        errors.append(
            f"{path}/affected_families: must match assumption_id family"
        )
      expected_parameters = (
          _FAMILY_STRUCTURE_PARAMETER_REFS[expected_family]
          if expected_category == "STRUCTURE"
          else (f"{expected_family}.{expected_category.lower()}.*",)
      )
      if tuple(parameter_refs) != expected_parameters:
        errors.append(
            f"{path}/parameter_refs: exact family registrations required"
        )
  return {value for value in identifiers if isinstance(value, str) and value}


def _validate_assumption_references(
    value: Any,
    known: set[str],
    errors: list[str],
    path: str = "<root>",
) -> None:
  if isinstance(value, dict):
    for key, nested in value.items():
      nested_path = f"{path}/{key}"
      if key == "assumption_ids" and isinstance(nested, list):
        for identifier in nested:
          if isinstance(identifier, str) and identifier not in known:
            errors.append(f"{nested_path}: unknown assumption {identifier}")
      else:
        _validate_assumption_references(nested, known, errors, nested_path)
  elif isinstance(value, list):
    for index, nested in enumerate(value):
      _validate_assumption_references(nested, known, errors, f"{path}/{index}")


def _validate_family(
    family: dict[str, Any],
    phases: tuple[str, ...],
    assumption_registry: dict[str, dict[str, Any]],
    path: str,
    errors: list[str],
) -> None:
  family_name = family.get("family")
  family_prefix = str(family_name).upper()
  required_assumption_ids = {
      category: f"MG1A.{family_prefix}.{category}.001"
      for category in _FAMILY_ASSUMPTION_CATEGORIES
  }
  if family.get("review_status") != REVIEW_STATUS:
    errors.append(f"{path}/review_status: must remain {REVIEW_STATUS}")
  expected_sides = _FAMILY_ALLOWED_SIDES.get(family_name)
  observed_sides = tuple(_as_list(family.get("allowed_instance_sides")))
  if expected_sides is None or observed_sides != expected_sides:
    errors.append(f"{path}/allowed_instance_sides: exact sides required")
  expected_side = (
      SIDE_INSTANCE_PARAMETER
      if family_name in _INSTANCE_SIDE_FAMILIES
      else expected_sides[0] if expected_sides else None
  )
  if family.get("side") != expected_side:
    errors.append(f"{path}/side: invalid fixed or instance-bound side")
  if family.get("instance_side_basis") != _FAMILY_SIDE_BASIS.get(family_name):
    errors.append(f"{path}/instance_side_basis: exact basis required")
  if family.get("scale_id") != "INSTANCE_PARAMETER":
    errors.append(
        f"{path}/scale_id: family definitions must be scale-independent"
    )
  if family.get("base_timeframe") != "INSTANCE_PARAMETER":
    errors.append(f"{path}/base_timeframe: must be instance-supplied")

  family_assumptions = _assumption_ids(family, path, errors)
  expected_family_assumptions = tuple(
      required_assumption_ids[category]
      for category in _FAMILY_ASSUMPTION_CATEGORIES
  ) + _FAMILY_ROOT_COMMON_ASSUMPTIONS.get(
      family_name,
      (),
  )
  if tuple(family_assumptions) != expected_family_assumptions:
    errors.append(
        f"{path}/assumption_ids: exact family-root provenance required"
    )

  anchors = _as_list(family.get("anchor_roles"))
  anchor_names = [
      item.get("role") for item in anchors if isinstance(item, dict)
  ]
  anchor_cardinalities = {
      item.get("role"): item.get("cardinality")
      for item in anchors
      if isinstance(item, dict)
  }
  if not anchors or len(anchor_names) != len(anchors):
    errors.append(f"{path}/anchor_roles: non-empty role objects required")
  if len(anchor_names) != len(set(anchor_names)):
    errors.append(f"{path}/anchor_roles: roles must be unique")
  for index, raw_anchor in enumerate(anchors):
    anchor = _as_mapping(raw_anchor)
    anchor_path = f"{path}/anchor_roles/{index}"
    _assumption_ids(anchor, anchor_path, errors)
    _require_assumption_reference(
        anchor,
        required_assumption_ids["STRUCTURE"],
        anchor_path,
        errors,
    )
    _require_exact_assumption_references(
        anchor,
        (required_assumption_ids["STRUCTURE"],),
        anchor_path,
        errors,
    )

  phase_contract = _as_mapping(family.get("phase_contract"))
  allowed_sequence = tuple(
      _as_list(phase_contract.get("allowed_phase_sequence"))
  )
  mappings = _as_list(phase_contract.get("common_mapping"))
  family_phases = [
      item.get("family_phase")
      for item in mappings
      if isinstance(item, dict)
  ]
  common_phases = [
      item.get("common_phase")
      for item in mappings
      if isinstance(item, dict)
  ]
  if not allowed_sequence or allowed_sequence != tuple(family_phases):
    errors.append(
        f"{path}/phase_contract/allowed_phase_sequence: mapping order required"
    )
  if len(family_phases) != len(set(family_phases)):
    errors.append(
        f"{path}/phase_contract/common_mapping: family phases not unique"
    )
  if set(common_phases) != set(phases):
    errors.append(
        f"{path}/phase_contract/common_mapping: all common phases required"
    )
  _assumption_ids(phase_contract, f"{path}/phase_contract", errors)
  _require_assumption_reference(
      phase_contract,
      required_assumption_ids["PHASE"],
      f"{path}/phase_contract",
      errors,
  )
  _require_exact_assumption_references(
      phase_contract,
      (required_assumption_ids["PHASE"],),
      f"{path}/phase_contract",
      errors,
  )

  topology = _as_list(family.get("topology_constraints"))
  if not topology:
    errors.append(
        f"{path}/topology_constraints: at least one relation required"
    )
  declared_anchors = set(anchor_names)
  observed_relations = {
      item.get("relation")
      for item in topology
      if isinstance(item, dict) and isinstance(item.get("relation"), str)
  }
  required_relations = _REQUIRED_TOPOLOGY_RELATIONS.get(family_name, set())
  if not required_relations.issubset(observed_relations):
    errors.append(f"{path}/topology_constraints: defining relations missing")
  side_basis = _SIDE_BASIS_REQUIREMENTS.get(family_name)
  if side_basis and (
      not set(side_basis["anchors"]).issubset(declared_anchors)
      or not set(side_basis["relations"]).issubset(observed_relations)
      or any(
          anchor_cardinalities.get(name) != "ONE"
          for name in side_basis["anchors"]
      )
  ):
    errors.append(
        f"{path}/instance_side_basis: required anchors/relations missing"
    )
  if side_basis:
    basis_anchors = set(side_basis["anchors"])
    for relation_name in side_basis["relations"]:
      matching_relations = [
          _as_mapping(item)
          for item in topology
          if _as_mapping(item).get("relation") == relation_name
      ]
      if not any(
          basis_anchors.issubset(
              set(_as_list(item.get("anchor_roles")))
          )
          for item in matching_relations
      ):
        errors.append(
            f"{path}/instance_side_basis: basis relation roles missing"
        )
  if side_basis:
    side_policy = f"{family_name}.side_basis_policy"
    structure_assumption = assumption_registry.get(
        required_assumption_ids["STRUCTURE"], {}
    )
    if side_policy not in _as_list(structure_assumption.get("parameter_refs")):
      errors.append(
          f"{path}/instance_side_basis: unregistered review parameter "
          f"{side_policy}"
      )
  for index, raw_constraint in enumerate(topology):
    constraint = _as_mapping(raw_constraint)
    constraint_path = f"{path}/topology_constraints/{index}"
    references = set(_as_list(constraint.get("anchor_roles")))
    if not references or not references.issubset(declared_anchors):
      errors.append(
          f"{constraint_path}/anchor_roles: undeclared anchor reference"
      )
    required_anchor_roles = _REQUIRED_TOPOLOGY_ANCHORS.get(
        family_name, {}
    ).get(constraint.get("relation"))
    if required_anchor_roles and references != set(required_anchor_roles):
      errors.append(
          f"{constraint_path}/anchor_roles: exact defining roles required"
      )
    _assumption_ids(constraint, constraint_path, errors)
    _require_assumption_reference(
        constraint,
        required_assumption_ids["STRUCTURE"],
        constraint_path,
        errors,
    )
    _require_exact_assumption_references(
        constraint,
        (required_assumption_ids["STRUCTURE"],),
        constraint_path,
        errors,
    )
    _parameter_refs(
        constraint,
        constraint_path,
        assumption_registry,
        (required_assumption_ids["STRUCTURE"],),
        errors,
    )
    topology_text = " ".join(_flatten_strings(constraint)).lower()
    if any(token in topology_text for token in _VOLUME_TOPOLOGY_TOKENS):
      errors.append(f"{constraint_path}: volume/OI cannot define topology")

  for section_name in _FIELD_SECTIONS:
    section = _as_mapping(family.get(section_name))
    section_path = f"{path}/{section_name}"
    fields = _as_list(section.get("fields"))
    if not fields or not all(
        isinstance(field, str) and field for field in fields
    ):
      errors.append(f"{section_path}/fields: non-empty field names required")
    if len(fields) != len(set(fields)):
      errors.append(f"{section_path}/fields: field names must be unique")
    _assumption_ids(section, section_path, errors)
    required_category = {
        "online_confirmation_fields": "CONFIRMATION",
        "invalidation_fields": "INVALIDATION",
    }.get(section_name, "STRUCTURE")
    _require_assumption_reference(
        section,
        required_assumption_ids[required_category],
        section_path,
        errors,
    )
    common_references = {
        "normalized_geometry_fields": (
            "MG1A.COMMON.NORMALIZATION.001",
        ),
        "duration_ratios": ("MG1A.COMMON.SCALE.001",),
        "price_ratios": ("MG1A.COMMON.NORMALIZATION.001",),
        "prior_context_fields": (
            "MG1A.COMMON.SESSION.001",
            "MG1A.COMMON.ROLL.001",
        ),
    }.get(section_name, ())
    _require_exact_assumption_references(
        section,
        (required_assumption_ids[required_category], *common_references),
        section_path,
        errors,
    )

    if section_name in {
        "normalized_geometry_fields",
        "duration_ratios",
        "price_ratios",
    }:
      _parameter_refs(
          section,
          section_path,
          assumption_registry,
          (required_assumption_ids["STRUCTURE"],),
          errors,
      )

  normalized = _as_mapping(family.get("normalized_geometry_fields"))
  if tuple(_as_list(normalized.get("allowed_units"))) != PRICE_UNITS:
    errors.append(
        f"{path}/normalized_geometry_fields/allowed_units: "
        "exact normalized price units required"
    )
  duration = _as_mapping(family.get("duration_ratios"))
  if duration.get("unit") != DURATION_UNIT:
    errors.append(
        f"{path}/duration_ratios/unit: exact duration unit required"
    )
  price = _as_mapping(family.get("price_ratios"))
  if price.get("unit") != "DIMENSIONLESS_RATIO":
    errors.append(
        f"{path}/price_ratios/unit: DIMENSIONLESS_RATIO required"
    )

  if (
      _as_mapping(family.get("prior_context_fields")).get("prefix_only")
      is not True
  ):
    errors.append(f"{path}/prior_context_fields: must be prefix-only")
  online = _as_mapping(family.get("online_confirmation_fields"))
  if online.get("prefix_only") is not True:
    errors.append(f"{path}/online_confirmation_fields: must be prefix-only")
  if (
      _as_mapping(family.get("invalidation_fields")).get("prefix_only")
      is not True
  ):
    errors.append(f"{path}/invalidation_fields: must be prefix-only")

  optional = _as_mapping(family.get("optional_volume_oi_confirmations"))
  if (
      optional.get("separate_from_topology") is not True
      or optional.get("required_for_topology") is not False
      or not _as_list(optional.get("fields"))
  ):
    errors.append(
        f"{path}/optional_volume_oi_confirmations: separation invalid"
    )
  _assumption_ids(optional, f"{path}/optional_volume_oi_confirmations", errors)
  _require_assumption_reference(
      optional,
      required_assumption_ids["CONFIRMATION"],
      f"{path}/optional_volume_oi_confirmations",
      errors,
  )
  _require_exact_assumption_references(
      optional,
      (
          required_assumption_ids["CONFIRMATION"],
          "MG1A.COMMON.FILTER_MISSINGNESS.001",
      ),
      f"{path}/optional_volume_oi_confirmations",
      errors,
  )
  _require_assumption_reference(
      optional,
      "MG1A.COMMON.FILTER_MISSINGNESS.001",
      f"{path}/optional_volume_oi_confirmations",
      errors,
  )
  provenance = _as_mapping(family.get("label_provenance"))
  if tuple(_as_list(provenance.get("allowed_values"))) != ALLOWED_PROVENANCE:
    errors.append(f"{path}/label_provenance: exact provenance values required")
  _assumption_ids(provenance, f"{path}/label_provenance", errors)
  _require_assumption_reference(
      provenance,
      "MG1A.COMMON.PROVENANCE.001",
      f"{path}/label_provenance",
      errors,
  )
  _require_exact_assumption_references(
      provenance,
      ("MG1A.COMMON.PROVENANCE.001",),
      f"{path}/label_provenance",
      errors,
  )

  identifier_surface = {
      "anchor_roles": [
          {
              "role": _as_mapping(item).get("role"),
              "cardinality": _as_mapping(item).get("cardinality"),
          }
          for item in anchors
      ],
      "topology_constraints": [
          {
              "relation": _as_mapping(item).get("relation"),
              "anchor_roles": _as_mapping(item).get("anchor_roles"),
              "parameter_refs": _as_mapping(item).get("parameter_refs"),
          }
          for item in topology
      ],
      "field_sections": {
          name: _as_mapping(family.get(name)).get("fields")
          for name in _FIELD_SECTIONS
      },
      "optional_volume_oi_confirmations": optional.get("fields"),
  }
  expected_identifier_hash = _FAMILY_IDENTIFIER_SURFACE_HASHES.get(
      family_name
  )
  if canonical_sha256(identifier_surface) != expected_identifier_hash:
    errors.append(f"{path}: exact family identifier surface required")

  structural_sections = {
      "anchor_roles": family.get("anchor_roles"),
      "topology_constraints": family.get("topology_constraints"),
      **{
          name: family.get(name)
          for name in _FIELD_SECTIONS
      },
      "optional_volume_oi_confirmations": family.get(
          "optional_volume_oi_confirmations"
      ),
  }
  errors.extend(
      _find_forbidden_strings(
          structural_sections,
          _FORBIDDEN_FIELD_FRAGMENTS,
          path,
          "forbidden ontology identifier",
      )
  )
  errors.extend(
      _find_forbidden_strings(
          {
              name: family.get(name)
              for name in (
                  "phase_contract",
                  "topology_constraints",
                  *_FIELD_SECTIONS,
              )
          },
          _FUTURE_CAUSAL_TOKENS,
          path,
          "future target leakage",
      )
  )


def _validate_ontology_payload(payload: Any) -> list[str]:
  """Return stable, fail-closed semantic errors for an MG1-A payload."""
  if not isinstance(payload, dict):
    return ["<root>: expected a mapping"]
  errors: list[str] = []
  if payload.get("schema_version") != SCHEMA_VERSION:
    errors.append(f"schema_version: expected {SCHEMA_VERSION}")
  if payload.get("profile") != PROFILE_NAME:
    errors.append(f"profile: expected {PROFILE_NAME}")
  if payload.get("milestone") != MILESTONE:
    errors.append(f"milestone: expected {MILESTONE}")
  if payload.get("decision") != DECISION:
    errors.append(f"decision: expected {DECISION}")

  human_review = _as_mapping(payload.get("human_review"))
  if human_review != {
      "status": "PENDING",
      "frozen": False,
      "approver": None,
      "approved_at": None,
      "frozen_sha256": None,
  }:
    errors.append("human_review: draft must remain pending and unfrozen")

  execution = _as_mapping(payload.get("execution"))
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
  if execution != expected_execution:
    errors.append(
        "execution: MG1-A implementation deferrals must remain explicit"
    )

  contract = _as_mapping(payload.get("resolved_contract"))
  observed_hash = payload.get("resolved_contract_sha256")
  computed_hash = canonical_sha256(contract)
  if observed_hash != computed_hash:
    errors.append(
        "resolved_contract_sha256: declared hash does not match "
        "canonical contract"
    )

  lifecycle = _as_mapping(contract.get("common_lifecycle"))
  phases = tuple(_as_list(lifecycle.get("phases")))
  if phases != COMMON_PHASES:
    errors.append(
        "resolved_contract/common_lifecycle/phases: exact order required"
    )
  transitions = _as_list(lifecycle.get("transitions"))
  transition_pairs = [
      (item.get("from_phase"), item.get("to_phase"))
      for item in transitions
      if isinstance(item, dict)
  ]
  if Counter(transition_pairs) != Counter(COMMON_TRANSITIONS):
    errors.append(
        "resolved_contract/common_lifecycle/transitions: exact graph required"
    )
  for index, transition in enumerate(transitions):
    transition_path = (
        f"resolved_contract/common_lifecycle/transitions/{index}"
    )
    transition_mapping = _as_mapping(transition)
    _assumption_ids(transition_mapping, transition_path, errors)
    _require_assumption_reference(
        transition_mapping,
        "MG1A.COMMON.LIFECYCLE.001",
        transition_path,
        errors,
    )
    _require_exact_assumption_references(
        transition_mapping,
        ("MG1A.COMMON.LIFECYCLE.001",),
        transition_path,
        errors,
    )

  for name in (
      "scale_contract",
      "normalization_contract",
      "anchor_instance_contract",
      "pattern_instance_contract",
      "label_provenance_contract",
  ):
    _assumption_ids(
        _as_mapping(contract.get(name)), f"resolved_contract/{name}", errors
    )

  required_common_references = {
      "scale_contract": ("MG1A.COMMON.SCALE.001",),
      "normalization_contract": ("MG1A.COMMON.NORMALIZATION.001",),
      "anchor_instance_contract": (
          "MG1A.COMMON.SWING_CAUSALITY.001",
          "MG1A.COMMON.PARTIAL_BAR.001",
      ),
      "pattern_instance_contract": (
          "MG1A.COMMON.SESSION.001",
          "MG1A.COMMON.ROLL.001",
          "MG1A.COMMON.LABEL_TIMING.001",
      ),
      "label_provenance_contract": ("MG1A.COMMON.PROVENANCE.001",),
  }
  for name, identifiers in required_common_references.items():
    item = _as_mapping(contract.get(name))
    for identifier in identifiers:
      _require_assumption_reference(
          item, identifier, f"resolved_contract/{name}", errors
      )
    _require_exact_assumption_references(
        item,
        identifiers,
        f"resolved_contract/{name}",
        errors,
    )

  scale_contract = _as_mapping(contract.get("scale_contract"))
  if (
      scale_contract.get("scale_id") != "INSTANCE_PARAMETER"
      or scale_contract.get("base_timeframe") != "INSTANCE_PARAMETER"
      or scale_contract.get("family_definition_scale_invariant") is not True
  ):
    errors.append(
        "resolved_contract/scale_contract: scale binding is not generic"
    )
  normalization = _as_mapping(contract.get("normalization_contract"))
  if (
      tuple(_as_list(normalization.get("price_units"))) != PRICE_UNITS
      or normalization.get("parameter_freeze_policy") != "TRAIN_FROZEN"
      or normalization.get("absolute_thresholds_allowed") is not False
      or normalization.get("duration_unit") != DURATION_UNIT
  ):
    errors.append(
        "resolved_contract/normalization_contract: normalization invalid"
    )

  anchors = _as_mapping(contract.get("anchor_instance_contract"))
  required_anchor_fields = tuple(_as_list(anchors.get("required_fields")))
  if required_anchor_fields != REQUIRED_ANCHOR_FIELDS:
    errors.append(
        "anchor_instance_contract: causal anchor fields are incomplete"
    )
  if (
      anchors.get("event_availability_order")
      != "AVAILABLE_AS_OF_NOT_BEFORE_EVENT"
      or anchors.get("online_use_rule")
      != "AVAILABLE_AS_OF_NOT_AFTER_OBSERVATION"
  ):
    errors.append("anchor_instance_contract: availability semantics invalid")

  instance = _as_mapping(contract.get("pattern_instance_contract"))
  if instance.get("target_namespaces_separate") is not True:
    errors.append(
        "pattern_instance_contract: target namespaces must be separate"
    )
  required_instance_fields = set(_as_list(instance.get("required_fields")))
  if tuple(_as_list(instance.get("required_fields"))) != (
      REQUIRED_PATTERN_INSTANCE_FIELDS
  ):
    errors.append(
        "pattern_instance_contract: exact required instance fields required"
    )
  target_fields = tuple(_as_list(instance.get("target_fields")))
  if target_fields != (
      "retrospective_completion_target",
      "future_outcome_target",
  ):
    errors.append("pattern_instance_contract: exact target fields required")
  isolated_target_fields = set(target_fields)
  isolated_target_fields.update(
      LABEL_NAMESPACE_FIELDS["retrospective_completion"]
  )
  isolated_target_fields.update(LABEL_NAMESPACE_FIELDS["future_outcome"])
  if required_instance_fields & isolated_target_fields:
    errors.append(
        "pattern_instance_contract: target fields cannot be runtime fields"
    )

  _validate_label_namespaces(contract, errors)

  provenance = _as_mapping(contract.get("label_provenance_contract"))
  if (
      tuple(_as_list(provenance.get("allowed_values"))) != ALLOWED_PROVENANCE
      or provenance.get("auto_model_assisted_independent_truth_allowed")
      is not False
  ):
    errors.append("label_provenance_contract: provenance policy invalid")

  known_assumptions = _validate_assumptions(contract, errors)
  assumption_registry = {
      item["assumption_id"]: item
      for item in _as_list(contract.get("assumptions"))
      if isinstance(item, dict)
      and isinstance(item.get("assumption_id"), str)
  }

  families = _as_list(contract.get("families"))
  family_names = [
      item.get("family") for item in families if isinstance(item, dict)
  ]
  if Counter(family_names) != Counter(REQUIRED_FAMILIES):
    errors.append("resolved_contract/families: exact 12-family set required")
  for index, raw_family in enumerate(families):
    _validate_family(
        _as_mapping(raw_family),
        phases,
        assumption_registry,
        f"resolved_contract/families/{index}",
        errors,
    )

  _validate_assumption_references(payload, known_assumptions, errors)
  errors.extend(_find_forbidden_keys(payload))
  return errors


def validate_ontology_payload(payload: Any) -> list[str]:
  """Return stable, fail-closed semantic errors for an MG1-A payload."""
  try:
    return _validate_ontology_payload(payload)
  except (AttributeError, KeyError, TypeError, ValueError) as exc:
    return [f"<root>: malformed payload ({type(exc).__name__}: {exc})"]


__all__ = (
    "ALLOWED_PATTERN_SIDES",
    "ALLOWED_PROVENANCE",
    "COMMON_PHASES",
    "COMMON_TRANSITIONS",
    "DECISION",
    "DURATION_UNIT",
    "LABEL_NAMESPACE_FIELDS",
    "LabelProvenance",
    "LifecyclePhase",
    "OntologyValidationError",
    "PatternFamily",
    "PatternOntologyDraft",
    "PatternSide",
    "PRICE_UNITS",
    "REQUIRED_ANCHOR_FIELDS",
    "REQUIRED_FAMILIES",
    "REQUIRED_PATTERN_INSTANCE_FIELDS",
    "REVIEW_STATUS",
    "ScaleBoundFamily",
    "canonical_json",
    "canonical_sha256",
    "json_schema_errors",
    "load_ontology",
    "loads_json_unique",
    "loads_yaml_unique",
    "report_schema_path",
    "validate_ontology_payload",
)
