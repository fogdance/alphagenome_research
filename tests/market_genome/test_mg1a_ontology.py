"""CPU-only contract tests for the MG1-A pattern ontology review draft."""

from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from alphatrade.market_genome.ontology import schema
from alphatrade.market_genome.ontology import validation


REPO_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_YAML = REPO_ROOT / "configs/market_genome/pattern_ontology_v1.yaml"
ONTOLOGY_MD = (
    REPO_ROOT / "docs/alphaTrade/market_genome/MG1_PATTERN_ONTOLOGY.md"
)
REQUIRED_DESIGN_EVIDENCE = (
    REPO_ROOT
    / "docs/market_genome/"
    "AlphaTrade_MarketGenome_Implementation_Plan_and_Codex_Prompts.md",
    REPO_ROOT
    / "docs/market_genome/AlphaTrade_MarketGenome_Technical_Report.md",
)


def _raw_payload() -> dict:
  return yaml.safe_load(ONTOLOGY_YAML.read_text(encoding="utf-8"))


def _rehash(payload: dict) -> dict:
  payload["resolved_contract_sha256"] = schema.canonical_sha256(
      payload["resolved_contract"]
  )
  return payload


def _semantic_errors(payload: dict) -> list[str]:
  return schema.validate_ontology_payload(_rehash(payload))


def _required_design_evidence_available() -> bool:
  return all(path.is_file() for path in REQUIRED_DESIGN_EVIDENCE)


def test_imports_are_lightweight_and_gpu_neutral() -> None:
  script = """
import importlib
import json
import sys
root = importlib.import_module("alphatrade.market_genome.ontology")
schema = importlib.import_module("alphatrade.market_genome.ontology.schema")
validation = importlib.import_module(
    "alphatrade.market_genome.ontology.validation"
)
blocked = sorted(
    name for name in sys.modules
    if name == "jax"
    or name.startswith("jax.")
    or name == "haiku"
    or name.startswith("haiku.")
    or name in {"alphatrade.core.model", "alphatrade.training"}
)
print(json.dumps({
    "root_status": root.IMPLEMENTATION_STATUS,
    "decision": schema.DECISION,
    "profile": validation.PROFILE_NAME,
    "blocked": blocked,
}))
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
      "root_status": "NOT_IMPLEMENTED",
      "decision": "DRAFT_NEEDS_HUMAN_FREEZE",
      "profile": "mg1a",
      "blocked": [],
  }


def test_schema_round_trip_and_json_schema() -> None:
  raw = _raw_payload()
  draft = schema.PatternOntologyDraft.from_dict(raw)
  assert draft.to_dict() == raw
  assert (
      schema.PatternOntologyDraft.from_dict(
          json.loads(json.dumps(draft.to_dict()))
      ).to_dict()
      == raw
  )
  assert validation.schema_errors(raw) == []
  assert schema.validate_ontology_payload(raw) == []


def test_canonical_hash_is_independent_of_mapping_order() -> None:
  raw = _raw_payload()["resolved_contract"]
  reversed_top_level = dict(reversed(list(raw.items())))
  assert schema.canonical_sha256(raw) == schema.canonical_sha256(
      reversed_top_level
  )


def test_exact_family_and_phase_sets() -> None:
  draft = schema.load_ontology(ONTOLOGY_YAML)
  assert draft.families == schema.REQUIRED_FAMILIES
  assert draft.phases == schema.COMMON_PHASES
  assert len(draft.families) == 12
  assert len(draft.phases) == 8

  required_family_keys = {
      "family",
      "side",
      "allowed_instance_sides",
      "instance_side_basis",
      "scale_id",
      "base_timeframe",
      "anchor_roles",
      "phase_contract",
      "topology_constraints",
      "normalized_geometry_fields",
      "duration_ratios",
      "price_ratios",
      "prior_context_fields",
      "online_confirmation_fields",
      "invalidation_fields",
      "optional_volume_oi_confirmations",
      "label_provenance",
      "assumption_ids",
  }
  for family in draft.to_dict()["resolved_contract"]["families"]:
    assert required_family_keys.issubset(family)

  expected_sides = {
      "trend_up": ["BULLISH"],
      "trend_down": ["BEARISH"],
      "trend_transition": ["BULLISH", "BEARISH"],
      "double_top": ["BEARISH"],
      "double_bottom": ["BULLISH"],
      "head_shoulders_top": ["BEARISH"],
      "inverse_head_shoulders": ["BULLISH"],
      "breakout": ["BULLISH", "BEARISH"],
      "failed_breakout": ["BULLISH", "BEARISH"],
      "retest": ["BULLISH", "BEARISH"],
      "range": ["NEUTRAL"],
      "support_resistance_conversion": ["BULLISH", "BEARISH"],
  }
  assert {
      item["family"]: item["allowed_instance_sides"]
      for item in draft.to_dict()["resolved_contract"]["families"]
  } == expected_sides


def test_invalid_phase_transitions_are_rejected() -> None:
  draft = schema.load_ontology(ONTOLOGY_YAML)
  assert draft.transition_allowed("CANDIDATE", "DEVELOPING")
  draft.require_transition("CANDIDATE", "DEVELOPING")
  with pytest.raises(schema.OntologyValidationError, match="undeclared"):
    draft.require_transition("CANDIDATE", "CONFIRMED")
  with pytest.raises(schema.OntologyValidationError, match="unknown phase"):
    draft.require_transition("UNKNOWN", "CANDIDATE")

  tampered = _raw_payload()
  tampered["resolved_contract"]["common_lifecycle"]["transitions"][1][
      "to_phase"
  ] = "CONFIRMED"
  errors = _semantic_errors(tampered)
  assert any("exact graph required" in error for error in errors)


def test_family_specific_phase_names_map_to_common_lifecycle() -> None:
  specialized = _raw_payload()
  phase_contract = specialized["resolved_contract"]["families"][0][
      "phase_contract"
  ]
  mature_index = phase_contract["allowed_phase_sequence"].index("MATURE")
  phase_contract["allowed_phase_sequence"][mature_index] = "STRUCTURE_READY"
  phase_contract["common_mapping"][mature_index][
      "family_phase"
  ] = "STRUCTURE_READY"
  _rehash(specialized)
  assert schema.validate_ontology_payload(specialized) == []
  assert validation.schema_errors(specialized) == []

  leaked = _raw_payload()
  phase_contract = leaked["resolved_contract"]["families"][0][
      "phase_contract"
  ]
  phase_contract["allowed_phase_sequence"][0] = "FUTURE_OUTCOME_KNOWN"
  phase_contract["common_mapping"][0][
      "family_phase"
  ] = "FUTURE_OUTCOME_KNOWN"
  _rehash(leaked)
  assert validation.schema_errors(leaked)
  errors = schema.validate_ontology_payload(leaked)
  assert any("future target leakage" in error for error in errors)


def test_defining_topology_relations_are_explicit() -> None:
  payload = _raw_payload()
  families = {
      item["family"]: item
      for item in payload["resolved_contract"]["families"]
  }
  expected = {
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
  for family_name, relations in expected.items():
    observed = {
        item["relation"]
        for item in families[family_name]["topology_constraints"]
    }
    assert relations.issubset(observed)

  tampered = _raw_payload()
  transition = next(
      item
      for item in tampered["resolved_contract"]["families"]
      if item["family"] == "trend_transition"
  )
  transition["topology_constraints"] = [
      item
      for item in transition["topology_constraints"]
      if item["relation"]
      != "STRUCTURE_BREAK_CROSSES_LAST_DEFENDING_PIVOT"
  ]
  errors = _semantic_errors(tampered)
  assert any("defining relations missing" in error for error in errors)

  wrong_roles = _raw_payload()
  head_shoulders = next(
      item
      for item in wrong_roles["resolved_contract"]["families"]
      if item["family"] == "head_shoulders_top"
  )
  defining_relation = next(
      item
      for item in head_shoulders["topology_constraints"]
      if item["relation"] == "HEAD_PEAK_ABOVE_BOTH_SHOULDERS"
  )
  defining_relation["anchor_roles"] = ["left_neck_trough"]
  errors = _semantic_errors(wrong_roles)
  assert any("exact defining roles required" in error for error in errors)


def test_instance_side_basis_is_bound_to_structure_and_review_slot() -> None:
  payload = _raw_payload()
  instance_bound = {
      "trend_transition",
      "breakout",
      "failed_breakout",
      "retest",
      "support_resistance_conversion",
  }
  for family in payload["resolved_contract"]["families"]:
    if family["family"] not in instance_bound:
      continue
    identifier = f"MG1A.{family['family'].upper()}.STRUCTURE.001"
    assumption = next(
        item
        for item in payload["resolved_contract"]["assumptions"]
        if item["assumption_id"] == identifier
    )
    assert f"{family['family']}.side_basis_policy" in assumption[
        "parameter_refs"
    ]

  renamed = _raw_payload()
  breakout = next(
      item
      for item in renamed["resolved_contract"]["families"]
      if item["family"] == "breakout"
  )
  for anchor in breakout["anchor_roles"]:
    if anchor["role"] == "boundary_cross":
      anchor["role"] = "cross_event"
  for relation in breakout["topology_constraints"]:
    relation["anchor_roles"] = [
        "cross_event" if role == "boundary_cross" else role
        for role in relation["anchor_roles"]
    ]
  errors = _semantic_errors(renamed)
  assert any("required anchors/relations missing" in error for error in errors)

  optional_basis_anchor = _raw_payload()
  breakout = next(
      item
      for item in optional_basis_anchor["resolved_contract"]["families"]
      if item["family"] == "breakout"
  )
  boundary_cross = next(
      item
      for item in breakout["anchor_roles"]
      if item["role"] == "boundary_cross"
  )
  boundary_cross["cardinality"] = "ZERO_OR_ONE"
  errors = _semantic_errors(optional_basis_anchor)
  assert any("required anchors/relations missing" in error for error in errors)

  missing_review_slot = _raw_payload()
  structure = next(
      item
      for item in missing_review_slot["resolved_contract"]["assumptions"]
      if item["assumption_id"] == "MG1A.BREAKOUT.STRUCTURE.001"
  )
  structure["parameter_refs"].remove("breakout.side_basis_policy")
  errors = _semantic_errors(missing_review_slot)
  assert any("unregistered review parameter" in error for error in errors)


def test_instance_contract_carries_declared_normalized_values() -> None:
  payload = _raw_payload()
  required = payload["resolved_contract"]["pattern_instance_contract"][
      "required_fields"
  ]
  for name in ("normalized_geometry", "duration_ratios", "price_ratios"):
    assert name in required

  missing = _raw_payload()
  missing["resolved_contract"]["pattern_instance_contract"][
      "required_fields"
  ].remove("normalized_geometry")
  _rehash(missing)
  assert validation.schema_errors(missing)
  assert any(
      "exact required instance fields" in error
      for error in schema.validate_ontology_payload(missing)
  )


def test_missing_unknown_and_duplicate_assumption_provenance_rejected() -> None:
  missing = _raw_payload()
  missing["resolved_contract"]["families"][0]["topology_constraints"][0].pop(
      "assumption_ids"
  )
  errors = _semantic_errors(missing)
  assert any("non-empty provenance is required" in error for error in errors)

  unknown = _raw_payload()
  unknown["resolved_contract"]["families"][0]["assumption_ids"].append(
      "MG1A.TREND_UP.UNKNOWN.001"
  )
  errors = _semantic_errors(unknown)
  assert any("unknown assumption" in error for error in errors)

  duplicate = _raw_payload()
  duplicate["resolved_contract"]["assumptions"].append(
      copy.deepcopy(duplicate["resolved_contract"]["assumptions"][0])
  )
  errors = _semantic_errors(duplicate)
  assert any("duplicate assumption_id" in error for error in errors)


def test_assumption_categories_and_owners_are_bound() -> None:
  family_missing = _raw_payload()
  family_missing["resolved_contract"]["families"][0]["assumption_ids"] = [
      "MG1A.TREND_UP.STRUCTURE.001",
      "MG1A.COMMON.LIFECYCLE.001",
      "MG1A.COMMON.PROVENANCE.001",
      "MG1A.COMMON.LABEL_TIMING.001",
  ]
  errors = _semantic_errors(family_missing)
  assert any("exact family-root provenance" in error for error in errors)

  section_cases = (
      ("phase_contract", "PHASE"),
      ("online_confirmation_fields", "CONFIRMATION"),
      ("invalidation_fields", "INVALIDATION"),
  )
  for section, category in section_cases:
    wrong_category = _raw_payload()
    wrong_category["resolved_contract"]["families"][0][section][
        "assumption_ids"
    ] = ["MG1A.TREND_UP.STRUCTURE.001"]
    errors = _semantic_errors(wrong_category)
    assert any(
        f"required reference MG1A.TREND_UP.{category}.001" in error
        for error in errors
    )

  wrong_record = _raw_payload()
  phase_assumption = next(
      item
      for item in wrong_record["resolved_contract"]["assumptions"]
      if item["assumption_id"] == "MG1A.TREND_UP.PHASE.001"
  )
  phase_assumption["category"] = "STRUCTURE"
  errors = _semantic_errors(wrong_record)
  assert any("category: must match assumption_id" in error for error in errors)

  wrong_namespace = _raw_payload()
  wrong_namespace["resolved_contract"]["label_namespaces"][
      "online_causal_phase"
  ]["assumption_ids"] = ["MG1A.TREND_DOWN.STRUCTURE.001"]
  _rehash(wrong_namespace)
  assert validation.schema_errors(wrong_namespace)
  errors = schema.validate_ontology_payload(wrong_namespace)
  assert any("MG1A.COMMON.LABEL_TIMING.001" in error for error in errors)

  foreign_topology = _raw_payload()
  foreign_topology["resolved_contract"]["families"][0][
      "topology_constraints"
  ][0]["assumption_ids"].append("MG1A.TREND_DOWN.STRUCTURE.001")
  errors = _semantic_errors(foreign_topology)
  assert any("exact provenance required" in error for error in errors)

  foreign_label_provenance = _raw_payload()
  foreign_label_provenance["resolved_contract"]["families"][0][
      "label_provenance"
  ]["assumption_ids"].append("MG1A.TREND_DOWN.STRUCTURE.001")
  errors = _semantic_errors(foreign_label_provenance)
  assert any("exact provenance required" in error for error in errors)

  foreign_family_root = _raw_payload()
  foreign_family_root["resolved_contract"]["families"][0][
      "assumption_ids"
  ].append("MG1A.COMMON.MIRROR.001")
  errors = _semantic_errors(foreign_family_root)
  assert any("exact family-root provenance" in error for error in errors)

  wrong_common_category = _raw_payload()
  common_label = next(
      item
      for item in wrong_common_category["resolved_contract"]["assumptions"]
      if item["assumption_id"] == "MG1A.COMMON.LABEL_TIMING.001"
  )
  common_label["category"] = "STRUCTURE"
  common_label["affected_families"] = ["trend_up"]
  errors = _semantic_errors(wrong_common_category)
  assert any("exact common category" in error for error in errors)
  assert any("exact common ownership" in error for error in errors)

  rogue_common = _raw_payload()
  rogue = copy.deepcopy(
      rogue_common["resolved_contract"]["assumptions"][0]
  )
  rogue["assumption_id"] = "MG1A.COMMON.ROGUE.001"
  rogue_common["resolved_contract"]["assumptions"].append(rogue)
  errors = _semantic_errors(rogue_common)
  assert any("exact common and family IDs" in error for error in errors)

  for contract_name in (
      "anchor_instance_contract",
      "pattern_instance_contract",
      "label_provenance_contract",
  ):
    foreign_contract = _raw_payload()
    foreign_contract["resolved_contract"][contract_name][
        "assumption_ids"
    ].append("MG1A.TREND_DOWN.STRUCTURE.001")
    errors = _semantic_errors(foreign_contract)
    assert any("exact provenance required" in error for error in errors)


def test_all_assumptions_remain_open_without_defaults() -> None:
  assumptions = _raw_payload()["resolved_contract"]["assumptions"]
  assert len(assumptions) == 61
  assert all(item["status"] == "NEEDS_HUMAN_REVIEW" for item in assumptions)
  assert all(item["proposed_default"] is None for item in assumptions)
  assert all(item["reviewer_decision"] is None for item in assumptions)

  resolved = _raw_payload()
  resolved["resolved_contract"]["assumptions"][0]["status"] = "APPROVED"
  resolved["resolved_contract"]["assumptions"][0]["reviewer_decision"] = {
      "value": "invented"
  }
  errors = _semantic_errors(resolved)
  assert any("must remain NEEDS_HUMAN_REVIEW" in error for error in errors)
  assert any("reviewer_decision: must remain null" in error for error in errors)


def test_resolved_contract_contains_no_numeric_detector_values() -> None:
  numeric_paths: list[str] = []

  def visit(value: object, path: str) -> None:
    if isinstance(value, bool) or value is None:
      return
    if isinstance(value, (int, float)):
      numeric_paths.append(path)
      return
    if isinstance(value, dict):
      for key, nested in value.items():
        visit(nested, f"{path}/{key}")
    elif isinstance(value, list):
      for index, nested in enumerate(value):
        visit(nested, f"{path}/{index}")

  visit(_raw_payload()["resolved_contract"], "resolved_contract")
  assert numeric_paths == []


def test_scale_independent_family_serialization() -> None:
  draft = schema.load_ontology(ONTOLOGY_YAML)
  internal = draft.bind_family(
      "double_top", scale_id="internal/16", base_timeframe="1m"
  )
  standard = draft.bind_family(
      "double_top", scale_id="standard/daily", base_timeframe="daily"
  )
  assert internal.definition_sha256 == standard.definition_sha256
  assert internal.definition == standard.definition
  assert internal.scale_id != standard.scale_id
  assert internal.base_timeframe != standard.base_timeframe
  assert internal.definition["scale_id"] == "INSTANCE_PARAMETER"
  assert internal.definition["base_timeframe"] == "INSTANCE_PARAMETER"
  assert internal.side == "BEARISH"


def test_instance_bound_sides_are_required_and_validated() -> None:
  draft = schema.load_ontology(ONTOLOGY_YAML)
  bound = draft.bind_family(
      "breakout",
      scale_id="internal/16",
      base_timeframe="1m",
      side="BULLISH",
  )
  assert bound.side == "BULLISH"
  assert bound.to_dict()["side"] == "BULLISH"
  with pytest.raises(ValueError, match="side is required"):
    draft.bind_family(
        "breakout", scale_id="internal/16", base_timeframe="1m"
    )
  with pytest.raises(ValueError, match="not allowed"):
    draft.bind_family(
        "breakout",
        scale_id="internal/16",
        base_timeframe="1m",
        side="NEUTRAL",
    )
  with pytest.raises(ValueError, match="scale_id"):
    draft.bind_family(
        "double_top", scale_id=1, base_timeframe="1m"
    )


def test_label_namespace_leakage_rejected() -> None:
  runtime_future = _raw_payload()
  runtime_future["resolved_contract"]["label_namespaces"]["future_outcome"][
      "runtime_input_allowed"
  ] = True
  errors = _semantic_errors(runtime_future)
  assert any("target isolation flags invalid" in error for error in errors)

  online_future = _raw_payload()
  online_future["resolved_contract"]["families"][0][
      "online_confirmation_fields"
  ]["fields"].append("future_outcome_state")
  errors = _semantic_errors(online_future)
  assert any("future target leakage" in error for error in errors)

  online_target = _raw_payload()
  online_target["resolved_contract"]["label_namespaces"][
      "online_causal_phase"
  ]["fields"].append("target_class")
  _rehash(online_target)
  assert validation.schema_errors(online_target)
  errors = schema.validate_ontology_payload(online_target)
  assert any("exact fields required" in error for error in errors)
  assert any("namespace fields must be disjoint" in error for error in errors)


def test_topology_volume_oi_and_forbidden_core_keys_rejected() -> None:
  mixed = _raw_payload()
  mixed["resolved_contract"]["families"][0]["topology_constraints"][0][
      "parameter_refs"
  ].append("volume_threshold")
  errors = _semantic_errors(mixed)
  assert any("volume/OI cannot define topology" in error for error in errors)

  absolute = _raw_payload()
  absolute["resolved_contract"]["families"][0][
      "normalized_geometry_fields"
  ]["absolute_price_threshold"] = 1.0
  errors = _semantic_errors(absolute)
  assert any("forbidden ontology key" in error for error in errors)

  action = _raw_payload()
  action["resolved_contract"]["families"][0]["entry_action"] = "BUY"
  errors = _semantic_errors(action)
  assert any("forbidden ontology key" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "expected"),
    [
        ("prior_context_fields", "future_return", "future target leakage"),
        (
            "normalized_geometry_fields",
            "absolute_price_level",
            "forbidden ontology identifier",
        ),
        (
            "online_confirmation_fields",
            "buy_signal",
            "forbidden ontology identifier",
        ),
        (
            "online_confirmation_fields",
            "target_class",
            "future target leakage",
        ),
        (
            "invalidation_fields",
            "entry_signal",
            "forbidden ontology identifier",
        ),
        (
            "invalidation_fields",
            "profit_target_state",
            "forbidden ontology identifier",
        ),
        (
            "online_confirmation_fields",
            "next_bar_return",
            "future target leakage",
        ),
        (
            "online_confirmation_fields",
            "forward_return",
            "future target leakage",
        ),
        (
            "online_confirmation_fields",
            "tomorrow_return",
            "future target leakage",
        ),
        (
            "online_confirmation_fields",
            "lookahead_label",
            "forbidden ontology identifier",
        ),
        (
            "online_confirmation_fields",
            "post_event_pnl",
            "forbidden ontology identifier",
        ),
        (
            "online_confirmation_fields",
            "long_signal",
            "forbidden ontology identifier",
        ),
        (
            "online_confirmation_fields",
            "short_position",
            "forbidden ontology identifier",
        ),
        (
            "online_confirmation_fields",
            "order_intent",
            "forbidden ontology identifier",
        ),
        (
            "normalized_geometry_fields",
            "close_usd",
            "forbidden ontology identifier",
        ),
        (
            "normalized_geometry_fields",
            "close_price_level",
            "forbidden ontology identifier",
        ),
        (
            "price_ratios",
            "elapsed_seconds",
            "forbidden ontology identifier",
        ),
    ],
)
def test_forbidden_family_field_values_fail_both_validators(
    section: str, field: str, expected: str
) -> None:
  payload = _raw_payload()
  payload["resolved_contract"]["families"][0][section]["fields"].append(
      field
  )
  _rehash(payload)
  assert validation.schema_errors(payload)
  errors = schema.validate_ontology_payload(payload)
  assert any(expected in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("online_confirmation_fields", "subsequent_return"),
        ("online_confirmation_fields", "following_bar_return"),
        ("online_confirmation_fields", "later_return"),
        ("online_confirmation_fields", "order_request"),
        ("online_confirmation_fields", "position_instruction"),
        ("online_confirmation_fields", "execution_instruction"),
        ("normalized_geometry_fields", "spot_price_level"),
        ("normalized_geometry_fields", "unscaled_price_level"),
        ("normalized_geometry_fields", "nominal_price_level"),
    ],
)
def test_closed_identifier_surface_rejects_unlisted_synonyms(
    section: str, field: str
) -> None:
  payload = _raw_payload()
  payload["resolved_contract"]["families"][0][section]["fields"].append(
      field
  )
  errors = _semantic_errors(payload)
  assert any("exact family identifier surface" in error for error in errors)


def test_family_units_and_parameter_refs_fail_closed() -> None:
  missing_refs = _raw_payload()
  missing_refs["resolved_contract"]["families"][0][
      "normalized_geometry_fields"
  ].pop("parameter_refs")
  _rehash(missing_refs)
  assert validation.schema_errors(missing_refs)
  assert any(
      "non-empty references required" in error
      for error in schema.validate_ontology_payload(missing_refs)
  )

  bad_units = _raw_payload()
  bad_units["resolved_contract"]["families"][0][
      "normalized_geometry_fields"
  ]["allowed_units"] = ["USD"]
  bad_units["resolved_contract"]["families"][0]["duration_ratios"][
      "unit"
  ] = "SECONDS"
  bad_units["resolved_contract"]["families"][0]["price_ratios"][
      "unit"
  ] = "USD"
  _rehash(bad_units)
  assert validation.schema_errors(bad_units)
  errors = schema.validate_ontology_payload(bad_units)
  assert any("normalized price units" in error for error in errors)
  assert any("duration unit" in error for error in errors)
  assert any("DIMENSIONLESS_RATIO" in error for error in errors)

  unregistered = _raw_payload()
  unregistered["resolved_contract"]["families"][0][
      "topology_constraints"
  ][0]["parameter_refs"] = ["trend_up.unregistered_policy"]
  errors = _semantic_errors(unregistered)
  assert any("unregistered reference" in error for error in errors)

  wrong_category = _raw_payload()
  geometry = wrong_category["resolved_contract"]["families"][0][
      "normalized_geometry_fields"
  ]
  geometry["assumption_ids"].append("MG1A.TREND_UP.PHASE.001")
  geometry["parameter_refs"] = ["trend_up.phase.rogue_geometry_policy"]
  errors = _semantic_errors(wrong_category)
  assert any("unregistered reference" in error for error in errors)

  global_wildcard = _raw_payload()
  structure = next(
      item
      for item in global_wildcard["resolved_contract"]["assumptions"]
      if item["assumption_id"] == "MG1A.TREND_UP.STRUCTURE.001"
  )
  structure["parameter_refs"] = ["*"]
  global_wildcard["resolved_contract"]["families"][0][
      "normalized_geometry_fields"
  ]["parameter_refs"] = ["unregistered.anything"]
  _rehash(global_wildcard)
  assert validation.schema_errors(global_wildcard)
  errors = schema.validate_ontology_payload(global_wildcard)
  assert any("invalid registration pattern" in error for error in errors)
  assert any("unregistered reference" in error for error in errors)

  category_wildcard = _raw_payload()
  structure = next(
      item
      for item in category_wildcard["resolved_contract"]["assumptions"]
      if item["assumption_id"] == "MG1A.TREND_UP.STRUCTURE.001"
  )
  structure["parameter_refs"].append("trend_up.geometry.*")
  category_wildcard["resolved_contract"]["families"][0][
      "normalized_geometry_fields"
  ]["parameter_refs"].append("trend_up.geometry.rogue")
  errors = _semantic_errors(category_wildcard)
  assert any("exact family registrations" in error for error in errors)

  foreign_registration = _raw_payload()
  structure = next(
      item
      for item in foreign_registration["resolved_contract"]["assumptions"]
      if item["assumption_id"] == "MG1A.TREND_UP.STRUCTURE.001"
  )
  structure["parameter_refs"].append("trend_down.rogue")
  foreign_registration["resolved_contract"]["families"][0][
      "normalized_geometry_fields"
  ]["parameter_refs"].append("trend_down.rogue")
  errors = _semantic_errors(foreign_registration)
  assert any("exact family registrations" in error for error in errors)

  wrong_phase_owner = _raw_payload()
  phase = next(
      item
      for item in wrong_phase_owner["resolved_contract"]["assumptions"]
      if item["assumption_id"] == "MG1A.TREND_UP.PHASE.001"
  )
  phase["parameter_refs"] = ["trend_down.phase.*"]
  errors = _semantic_errors(wrong_phase_owner)
  assert any("exact family registrations" in error for error in errors)

  scoped_wildcard = _raw_payload()
  structure = next(
      item
      for item in scoped_wildcard["resolved_contract"]["assumptions"]
      if item["assumption_id"] == "MG1A.TREND_UP.STRUCTURE.001"
  )
  structure["parameter_refs"] = ["trend_up.*"]
  scoped_wildcard["resolved_contract"]["families"][0][
      "normalized_geometry_fields"
  ]["parameter_refs"] = ["trend_up.undeclared_policy"]
  _rehash(scoped_wildcard)
  assert validation.schema_errors(scoped_wildcard)
  errors = schema.validate_ontology_payload(scoped_wildcard)
  assert any("invalid registration pattern" in error for error in errors)
  assert any("unregistered reference" in error for error in errors)

  canonical = _raw_payload()
  breakout = next(
      item
      for item in canonical["resolved_contract"]["families"]
      if item["family"] == "breakout"
  )
  assert "boundary_reentry_state" in breakout["invalidation_fields"][
      "fields"
  ]
  assert schema.validate_ontology_payload(canonical) == []


def test_exact_anchor_and_instance_field_contracts() -> None:
  missing_anchor = _raw_payload()
  missing_anchor["resolved_contract"]["anchor_instance_contract"][
      "required_fields"
  ].remove("source_method")
  _rehash(missing_anchor)
  assert validation.schema_errors(missing_anchor)
  assert any(
      "causal anchor fields are incomplete" in error
      for error in schema.validate_ontology_payload(missing_anchor)
  )

  extra_instance = _raw_payload()
  extra_instance["resolved_contract"]["pattern_instance_contract"][
      "required_fields"
  ].append("future_outcome_target")
  _rehash(extra_instance)
  assert validation.schema_errors(extra_instance)
  errors = schema.validate_ontology_payload(extra_instance)
  assert any("exact required instance fields" in error for error in errors)
  assert any("cannot be runtime fields" in error for error in errors)


def test_false_human_freeze_is_rejected() -> None:
  payload = _raw_payload()
  payload["decision"] = "FROZEN"
  payload["human_review"] = {
      "status": "APPROVED",
      "frozen": True,
      "approver": "unverified",
      "approved_at": "2026-07-13",
      "frozen_sha256": "0" * 64,
  }
  with pytest.raises(schema.OntologyValidationError):
    schema.PatternOntologyDraft.from_dict(payload)
  errors = schema.validate_ontology_payload(payload)
  assert any("draft must remain pending" in error for error in errors)


def test_loader_rejects_structural_drift_and_malformed_values() -> None:
  extra = _raw_payload()
  extra["unexpected_root_key"] = True
  with pytest.raises(
      schema.OntologyValidationError, match="unexpected_root_key"
  ):
    schema.PatternOntologyDraft.from_dict(extra)

  malformed = _raw_payload()
  malformed["resolved_contract"]["common_lifecycle"]["phases"][0] = {
      "not": "hashable"
  }
  errors = schema.validate_ontology_payload(_rehash(malformed))
  assert errors
  assert "malformed payload" in errors[0]
  with pytest.raises(schema.OntologyValidationError):
    schema.PatternOntologyDraft.from_dict(malformed)

  non_finite = _raw_payload()
  non_finite["unexpected"] = float("nan")
  with pytest.raises(schema.OntologyValidationError, match="canonical JSON"):
    schema.PatternOntologyDraft.from_dict(non_finite)


def test_yaml_and_json_loaders_reject_duplicate_mapping_keys(
    tmp_path: Path,
) -> None:
  duplicate_yaml = tmp_path / "duplicate.yaml"
  duplicate_yaml.write_text(
      "decision: FROZEN_SHADOWED_VALUE\n"
      + ONTOLOGY_YAML.read_text(encoding="utf-8"),
      encoding="utf-8",
  )
  with pytest.raises(schema.OntologyValidationError, match="duplicate key"):
    schema.load_ontology(duplicate_yaml)
  payload, error = validation._load_yaml(duplicate_yaml)
  assert payload is None
  assert "duplicate key" in error

  reports_dir = tmp_path / "reports"
  report_path, markdown_path = validation.generate_report_bundle(reports_dir)
  duplicate_json = report_path.read_text(encoding="utf-8").replace(
      "{",
      '{"decision":"FROZEN_SHADOWED_VALUE",',
      1,
  )
  report_path.write_text(duplicate_json, encoding="utf-8")
  result = validation.validate_report_bundle(report_path, markdown_path)
  assert result["summary"]["overall"] == "fail"
  assert any(
      "duplicate mapping key" in error
      for error in result["schema_errors"]
  )


def test_missing_json_schema_returns_a_stable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  missing = tmp_path / "missing.schema.json"
  monkeypatch.setattr(schema, "report_schema_path", lambda: missing)
  errors = schema.json_schema_errors(_raw_payload())
  assert len(errors) == 1
  assert errors[0].startswith("ontology schema unavailable or invalid:")


def test_repo_root_override_uses_schema_from_that_root(tmp_path: Path) -> None:
  schema_path = (
      tmp_path
      / "src/alphatrade/market_genome/ontology/"
      "mg1a_pattern_ontology.schema.json"
  )
  schema_path.parent.mkdir(parents=True)
  schema_path.write_text("{broken schema", encoding="utf-8")
  errors = validation.schema_errors(_raw_payload(), repo_root=tmp_path)
  assert errors
  assert errors[0].startswith("ontology schema unavailable or invalid:")

  report_path = tmp_path / "report.json"
  markdown_path = tmp_path / "report.md"
  report_path.write_text(json.dumps(_raw_payload()), encoding="utf-8")
  markdown_path.write_text("not reached", encoding="utf-8")
  result = validation.validate_report_bundle(
      report_path,
      markdown_path,
      repo_root=tmp_path,
  )
  assert result["summary"]["overall"] == "fail"
  assert result["schema"] == str(schema_path)


def test_unknown_assumption_family_is_rejected() -> None:
  payload = _raw_payload()
  payload["resolved_contract"]["assumptions"][0][
      "affected_families"
  ] = ["not_a_family"]
  _rehash(payload)
  assert validation.schema_errors(payload)
  assert any(
      "unknown family reference" in error
      for error in schema.validate_ontology_payload(payload)
  )


def test_independent_manifest_does_not_modify_mg0b_profile() -> None:
  mg1a = yaml.safe_load(
      (
          REPO_ROOT / "configs/market_genome/mg1a_contracts_manifest.yaml"
      ).read_text(encoding="utf-8")
  )
  mg0b = yaml.safe_load(
      (
          REPO_ROOT / "configs/market_genome/contracts_manifest.yaml"
      ).read_text(encoding="utf-8")
  )
  legacy = yaml.safe_load(
      (
          REPO_ROOT / "src/alphatrade/schemas/contracts_manifest.yaml"
      ).read_text(encoding="utf-8")
  )
  assert set(mg1a["profiles"]) == {"mg1a"}
  assert mg1a["profiles"]["mg1a"]["decision"] == schema.DECISION
  assert set(mg0b["profiles"]) == {"mg0b"}
  assert "mg1a" not in legacy["profiles"]


def test_manifest_validation_is_closed_shape(tmp_path: Path) -> None:
  manifest = REPO_ROOT / "configs/market_genome/mg1a_contracts_manifest.yaml"
  destination = tmp_path / "configs/market_genome"
  destination.mkdir(parents=True)
  (destination / manifest.name).write_bytes(manifest.read_bytes())
  assert validation._manifest_check(tmp_path)["valid"] is True

  payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
  payload["version"] = 999
  (destination / manifest.name).write_text(
      yaml.safe_dump(payload), encoding="utf-8"
  )
  assert validation._manifest_check(tmp_path)["valid"] is False

  (destination / manifest.name).write_bytes(b"\xff")
  invalid_utf = validation._manifest_check(tmp_path)
  assert invalid_utf["valid"] is False
  assert invalid_utf["load_error"]


def test_manifest_reports_dir_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  runs_root = REPO_ROOT.parent / "alphatrade_runs" / "manifest_test"
  monkeypatch.setenv("ALPHATRADE_RUNS_ROOT", str(runs_root))
  assert validation._default_reports_dir(REPO_ROOT) == (
      runs_root.resolve() / "reports"
  )
  monkeypatch.delenv("ALPHATRADE_RUNS_ROOT")
  with pytest.raises(schema.OntologyValidationError, match="is required"):
    validation._default_reports_dir(REPO_ROOT)


def test_evidence_content_hash_is_enforced(tmp_path: Path) -> None:
  source = tmp_path / "source.md"
  source.write_text("first\nsecond\n", encoding="utf-8")
  report = {
      "source_evidence": [
          {
              "evidence_id": "SOURCE",
              "source_root": "REPOSITORY",
              "path": "source.md",
              "sha256": validation._sha256_file(source),
              "availability": "REQUIRED",
              "line_range": "1-2",
          }
      ],
      "resolved_contract": {
          "assumptions": [{"source_evidence": ["SOURCE"]}]
      },
      "rejected_shortcuts": [],
  }
  check = validation._evidence_checks(report, tmp_path)
  assert check["errors"] == []
  assert check["verified"] == ["SOURCE"]

  source.write_text("changed\nsecond\n", encoding="utf-8")
  check = validation._evidence_checks(report, tmp_path)
  assert check["errors"] == ["SOURCE:content_hash_mismatch"]


def test_optional_external_evidence_root_is_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.delenv("CHENDAGE_SIGNAL_ROOT", raising=False)
  report = {
      "source_evidence": [
          {
              "evidence_id": "EXTERNAL",
              "source_root": "CHENDAGE_SIGNAL",
              "path": "src/example.py",
              "sha256": "0" * 64,
              "availability": "VERIFY_WHEN_AVAILABLE",
              "line_range": "1",
          }
      ],
      "resolved_contract": {
          "assumptions": [{"source_evidence": ["EXTERNAL"]}]
      },
      "rejected_shortcuts": [],
  }
  check = validation._evidence_checks(report, tmp_path)
  assert check["errors"] == []
  assert check["unavailable"] == ["EXTERNAL"]

  configured = tmp_path / "external"
  configured.mkdir()
  check = validation._evidence_checks(
      report,
      tmp_path,
      evidence_roots={"CHENDAGE_SIGNAL": configured},
  )
  assert any(":missing:" in error for error in check["errors"])


def test_required_design_evidence_fails_closed_and_repo_root_is_fixed(
    tmp_path: Path,
) -> None:
  alternate = tmp_path / "alternate"
  alternate.mkdir()
  report = _raw_payload()
  check = validation._evidence_checks(
      report,
      tmp_path,
      evidence_roots={"REPOSITORY": alternate},
  )
  assert check["configured_roots"]["REPOSITORY"] == str(tmp_path.resolve())
  assert any(
      error.startswith("PLAN_MG1A:missing:") for error in check["errors"]
  )
  assert any(
      error.startswith("TECH_ONTOLOGY:missing:")
      for error in check["errors"]
  )


def test_report_generation_and_strict_validation(tmp_path: Path) -> None:
  report_path, markdown_path = validation.generate_report_bundle(tmp_path)
  report = json.loads(report_path.read_text(encoding="utf-8"))
  assert report == _raw_payload()
  assert markdown_path.read_bytes() == ONTOLOGY_MD.read_bytes()

  result = validation.validate_report_bundle(report_path, markdown_path)
  if not _required_design_evidence_available():
    assert result["summary"]["overall"] == "fail"
    assert result["summary"]["human_freeze_pending"] is False
    evidence_check = next(
        item
        for item in result["semantic_results"]
        if item["name"]
        == "source_evidence_paths_lines_and_references_valid"
    )
    assert evidence_check["status"] == "fail"
    return
  assert result["schema_errors"] == []
  assert result["summary"] == {
      "schema_total": 1,
      "schema_passed": 1,
      "semantic_total": 11,
      "semantic_passed": 11,
      "overall": "pass",
      "human_freeze_pending": True,
      "human_freeze_allowed": False,
  }
  assert result["decision"] == "DRAFT_NEEDS_HUMAN_FREEZE"
  assert result["interpretation"] == (
      "draft contract valid; human freeze pending"
  )


def test_report_or_markdown_tampering_fails_closed(tmp_path: Path) -> None:
  report_path, markdown_path = validation.generate_report_bundle(tmp_path)
  report = json.loads(report_path.read_text(encoding="utf-8"))
  report["resolved_contract"]["families"].pop()
  report_path.write_text(json.dumps(report), encoding="utf-8")
  result = validation.validate_report_bundle(report_path, markdown_path)
  assert result["summary"]["overall"] == "fail"
  assert result["summary"]["human_freeze_pending"] is False
  assert result["interpretation"] == (
      "invalid draft bundle; human freeze prohibited"
  )
  assert result["schema_errors"]

  report_path, markdown_path = validation.generate_report_bundle(tmp_path)
  markdown_path.write_text("unrelated\n", encoding="utf-8")
  result = validation.validate_report_bundle(report_path, markdown_path)
  assert result["summary"]["overall"] == "fail"
  markdown_check = next(
      item
      for item in result["semantic_results"]
      if item["name"] == "markdown_matches_source_and_required_sections"
  )
  assert markdown_check["status"] == "fail"

  report_path.write_text("{not-json", encoding="utf-8")
  result = validation.validate_report_bundle(report_path, markdown_path)
  assert result["summary"]["overall"] == "fail"
  assert result["schema_errors"]


def test_strict_cli_writes_validation_reports(tmp_path: Path) -> None:
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.validation",
          "--profile",
          "mg1a",
          "--reports-dir",
          str(tmp_path),
          "--generate",
          "--strict",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  result = json.loads(
      (tmp_path / "mg1a_schema_validation.json").read_text(encoding="utf-8")
  )
  if not _required_design_evidence_available():
    assert completed.returncode == 1
    assert result["summary"]["overall"] == "fail"
    assert result["summary"]["human_freeze_pending"] is False
    return
  assert completed.returncode == 0
  assert result["summary"]["overall"] == "pass"
  assert result["summary"]["human_freeze_pending"] is True
  assert "draft contract valid; human freeze pending" in completed.stdout
  validation_md = (
      tmp_path / "mg1a_schema_validation.md"
  ).read_text(encoding="utf-8")
  assert "Overall: `PASS`" in validation_md
  assert "Human freeze: `PENDING`" in validation_md


def test_cli_rejects_output_path_collision(tmp_path: Path) -> None:
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  ontology_report = tmp_path / "mg1_pattern_ontology.json"
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.validation",
          "--profile",
          "mg1a",
          "--reports-dir",
          str(tmp_path),
          "--generate",
          "--strict",
          "--output-json",
          str(ontology_report),
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert completed.returncode == 2
  assert "report path collision" in completed.stdout
  assert json.loads(ontology_report.read_text(encoding="utf-8"))[
      "schema_version"
  ] == schema.SCHEMA_VERSION


def test_strict_cli_exits_nonzero_for_semantic_failure(
    tmp_path: Path
) -> None:
  report_path, _ = validation.generate_report_bundle(tmp_path)
  report = json.loads(report_path.read_text(encoding="utf-8"))
  report["resolved_contract_sha256"] = "0" * 64
  report_path.write_text(json.dumps(report), encoding="utf-8")
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  completed = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.validation",
          "--profile",
          "mg1a",
          "--reports-dir",
          str(tmp_path),
          "--strict",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert completed.returncode == 1
  result = json.loads(
      (tmp_path / "mg1a_schema_validation.json").read_text(encoding="utf-8")
  )
  assert result["summary"]["overall"] == "fail"
  assert result["summary"]["human_freeze_pending"] is False


def test_validator_confirms_mg0b_frozen_hashes_and_bundle() -> None:
  report = _raw_payload()
  checks = validation.semantic_checks(report, ONTOLOGY_MD, repo_root=REPO_ROOT)
  preservation = next(
      item
      for item in checks
      if item["name"] == "mg0b_frozen_bytes_preserved"
  )
  assert preservation["status"] == "pass"
  assert preservation["observed"]["errors"] == []
  assert len(preservation["observed"]["expected"]) == 19
  bundle = next(
      item
      for item in checks
      if item["name"] == "mg0b_frozen_bundle_revalidated"
  )
  assert bundle["status"] == "pass"
  assert bundle["observed"]["summary"]["overall"] == "pass"


def test_mg0b_byte_gate_runs_before_schema_module_import() -> None:
  script = """
import sys
from pathlib import Path
from alphatrade.market_genome.ontology import validation
name = "alphatrade.market_genome.schemas"
assert name not in sys.modules
result = validation._mg0b_preservation(Path.cwd())
assert result["errors"] == []
assert name not in sys.modules
"""
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  subprocess.run(
      [sys.executable, "-c", script],
      cwd=REPO_ROOT,
      env=env,
      check=True,
      capture_output=True,
      text=True,
  )


def test_mg0b_hash_gate_covers_previously_omitted_file(
    tmp_path: Path
) -> None:
  for relative in validation._MG0B_FROZEN_HASHES:
    source = REPO_ROOT / relative
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
  assert validation._mg0b_preservation(tmp_path)["errors"] == []

  omitted_before = (
      tmp_path / "src/alphatrade/market_genome/ontology/__init__.py"
  )
  omitted_before.write_text("CHANGED\n", encoding="utf-8")
  assert (
      "src/alphatrade/market_genome/ontology/__init__.py"
      in validation._mg0b_preservation(tmp_path)["errors"]
  )


def test_module_reload_does_not_change_draft_status() -> None:
  ontology_root = importlib.import_module("alphatrade.market_genome.ontology")
  importlib.reload(schema)
  assert ontology_root.IMPLEMENTATION_STATUS == "NOT_IMPLEMENTED"
  assert schema.DECISION == "DRAFT_NEEDS_HUMAN_FREEZE"
