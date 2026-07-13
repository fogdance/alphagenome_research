"""CPU-only contract tests for the MG1-A.1 ontology revision draft."""

from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from alphatrade.market_genome.ontology import revision


REPO_ROOT = Path(__file__).resolve().parents[2]
REVISION_YAML = REPO_ROOT / "configs/market_genome/pattern_ontology_v2.yaml"

EXPECTED_CLASS_BY_FAMILY = {
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


def _payload() -> dict:
  return revision.load_revision(REVISION_YAML)


def _rehash(payload: dict) -> dict:
  payload["resolved_contract_sha256"] = revision.canonical_sha256(
      payload["resolved_contract"]
  )
  return payload


def _errors(payload: dict) -> list[str]:
  return revision.validate_revision_payload(_rehash(payload), repo_root=REPO_ROOT)


def _families(payload: dict) -> dict[str, dict]:
  return {
      item["family"]: item
      for item in payload["resolved_contract"]["families"]
  }


def _expanded_edges(family: dict) -> list[tuple[str, str, str]]:
  contract = family["phase_contract"]
  edges = [
      (item["from_phase"], item["to_phase"], item["event"])
      for item in contract["branch_transitions"]
  ]
  active = contract["common_mapping"]["ACTIVE"]
  for item in contract["invalidation_transitions"]:
    sources = (
        active
        if item["from_phases"] == "ALL_ACTIVE_FAMILY_PHASES"
        else item["from_phases"]
    )
    edges.extend(
        (source, item["to_phase"], event)
        for source in sources
        for event in item["events"]
    )
  return edges


def test_revision_import_is_lightweight_and_gpu_neutral() -> None:
  script = """
import importlib
import json
import sys
module = importlib.import_module(
    "alphatrade.market_genome.ontology.revision"
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
      "profile": "mg1a-v2",
      "blocked": [],
  }


def test_independent_schema_semantics_and_loader_pass() -> None:
  payload = _payload()
  assert revision.json_schema_errors(payload, repo_root=REPO_ROOT) == []
  assert revision.validate_revision_payload(payload, repo_root=REPO_ROOT) == []
  loaded, errors = revision.validate_revision(
      REVISION_YAML, repo_root=REPO_ROOT
  )
  assert errors == []
  assert loaded == payload


def test_resolved_hash_and_all_declared_v1_bytes_are_preserved() -> None:
  payload = _payload()
  assert payload["resolved_contract_sha256"] == revision.canonical_sha256(
      payload["resolved_contract"]
  )
  for record in payload["v1_artifact_hashes"]:
    observed = hashlib.sha256(
        (REPO_ROOT / record["path"]).read_bytes()
    ).hexdigest()
    assert observed == record["sha256"], record["path"]
  assert payload["base_v1"]["config_sha256"] == (
      "3ab4ce88dd63dbb5ec484b6ca420ce9a3ab06579f63b53776252216a10012b90"
  )


def test_exact_family_object_orientation_and_effect_separation() -> None:
  payload = _payload()
  contract = payload["resolved_contract"]
  model = contract["object_model"]
  families = _families(payload)
  assert model["class_by_family"] == EXPECTED_CLASS_BY_FAMILY
  assert set(families) == set(EXPECTED_CLASS_BY_FAMILY)
  assert len(families) == 12
  assert {
      name: item["object_class"] for name, item in families.items()
  } == EXPECTED_CLASS_BY_FAMILY
  assert model["structural_orientations"] == [
      "UPPER",
      "LOWER",
      "UPWARD",
      "DOWNWARD",
      "NEUTRAL",
  ]
  assert "INSTANCE_PARAMETER" not in model["structural_orientations"]
  assert "TRANSITIONAL" not in model["structural_orientations"]
  assert model["morphology_effect_separation"] == {
      "structural_orientation_is_future_effect": False,
      "semantic_role_is_future_outcome": False,
      "classic_reversal_role_requires_preceding_context": True,
      "future_effect_namespace": "future_outcome",
      "future_effect_allowed_in_family_contract": False,
  }
  assert all(
      item["structural_orientation"] is None
      for item in families.values()
      if item["orientation_binding"] == "INSTANCE_BOUND"
  )


def test_family_phase_graphs_are_total_reachable_and_terminal() -> None:
  for family in _payload()["resolved_contract"]["families"]:
    contract = family["phase_contract"]
    mapping = contract["common_mapping"]
    phases = contract["family_phases"]
    mapped = [
        phase
        for status in ("INACTIVE", "ACTIVE", "RESOLVED", "INVALIDATED")
        for phase in mapping[status]
    ]
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == set(phases)

    edges = _expanded_edges(family)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target, _ in edges:
      adjacency[source].add(target)
    reachable = set(mapping["INACTIVE"])
    queue = deque(reachable)
    while queue:
      for target in adjacency[queue.popleft()]:
        if target not in reachable:
          reachable.add(target)
          queue.append(target)
    assert reachable == set(phases), family["family"]

    terminal = set(mapping["RESOLVED"] + mapping["INVALIDATED"])
    assert not any(source in terminal for source, _, _ in edges), family["family"]
    for invalidated in mapping["INVALIDATED"]:
      assert any(target == invalidated for _, target, _ in edges), family["family"]


def test_common_scale_anchor_relation_formula_and_review_contracts() -> None:
  contract = _payload()["resolved_contract"]
  assert contract["common_lifecycle"]["projected_edges"] == [
      "INACTIVE_TO_ACTIVE",
      "ACTIVE_TO_ACTIVE",
      "ACTIVE_TO_RESOLVED",
      "ACTIVE_TO_INVALIDATED",
  ]
  assert set(contract["scale_spec_contract"]["required_fields"]) == {
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
  }
  anchor = contract["anchor_contract"]
  assert anchor["states"] == [
      "PROVISIONAL",
      "CONFIRMED",
      "REVISED",
      "SUPERSEDED",
      "REVOKED",
  ]
  assert {
      "anchor_id",
      "anchor_lineage_id",
      "anchor_revision",
      "supersedes_anchor_id",
      "revision_reason",
      "price_basis",
      "confirmation_basis",
  }.issubset(anchor["required_fields"])
  graph = contract["relationship_graph"]
  assert graph["binary_edges_only"] is True
  assert {
      "source_ref",
      "target_ref",
      "relation_event_id",
      "source_event_ids",
      "relation_ordinal",
  }.issubset(graph["required_fields"])
  formula = contract["geometry_formula_contract"]
  assert formula["absolute_price_denominator_allowed"] is False
  assert formula["numeric_detector_thresholds_allowed"] is False
  assert formula["denominator_zero_policy"] == "UNKNOWN_WITH_MASK"
  assert formula["insufficient_history_policy"] == "UNKNOWN_WITH_MASK"
  review = contract["review_protocol"]
  assert review["review_modes"] == [
      "ONLINE_BLINDED",
      "RETROSPECTIVE",
      "OUTCOME",
  ]
  assert review["future_visibility_by_mode"] == {
      "ONLINE_BLINDED": "PREFIX_ONLY",
      "RETROSPECTIVE": "COMPLETION_WINDOW_ONLY",
      "OUTCOME": "DECLARED_OUTCOME_WINDOW_ONLY",
  }
  assert review["minimum_independent_reviewers"] == "TWO"
  assert review["ambiguity_blocks_gold_eligibility"] is True
  assert all(
      namespace["runtime_input_allowed"] is False
      for name, namespace in contract["label_namespaces"].items()
      if name != "assumption_ids"
  )


def test_family_specific_p0_revisions_are_explicit() -> None:
  families = _families(_payload())
  transition = families["trend_transition"]
  assert "PRIOR_TREND_WEAKENING" not in transition["phase_contract"][
      "family_phases"
  ]
  assert transition["phase_contract"]["branch_transitions"][0]["event"] == (
      "SHARED_DEFENDING_PIVOT_BREAK"
  )

  for name in (
      "double_top",
      "double_bottom",
      "head_shoulders_top",
      "inverse_head_shoulders",
  ):
    roles = {item["role"] for item in families[name]["anchors"]}
    assert {"neckline_retest", "continuation_anchor"}.issubset(roles)
    assert "neckline_retest_or_continuation" not in roles
  for name in ("double_top", "double_bottom"):
    formula_names = {
        item["name"] for item in families[name]["formula_contracts"]
    }
    assert "neckline_slope" not in formula_names

  failed = families["failed_breakout"]
  assert failed["object_creation_event"] == "SHARED_BOUNDARY_REENTRY"
  assert failed["historical_attempt_relationship"] == "ORIGINATES_FROM"
  assert failed["historical_source_evidence_available_at_creation_only"] is True
  assert failed["phase_contract"]["branch_transitions"][0] == {
      "from_phase": "INACTIVE",
      "to_phase": "REENTRY_OBSERVED",
      "event": "SHARED_BOUNDARY_REENTRY",
  }
  assert {
      "attempt_direction",
      "reentry_direction",
      "resolution_direction",
  }.issubset(failed["instance_fields"])
  assert failed["direction_contract"]["directions_must_not_be_collapsed"] is True

  retest = families["retest"]
  assert "retest_ordinal" in retest["instance_fields"]
  assert {
      "HOLD_CONFIRMED",
      "PENETRATE_RECLAIMED",
      "FULL_RECROSS_FAILURE",
  }.issubset(retest["phase_contract"]["family_phases"])
  range_mapping = families["range"]["phase_contract"]["common_mapping"]
  assert "ACCEPTED_BREAK" in range_mapping["ACTIVE"]
  assert range_mapping["RESOLVED"] == ["TERMINATED"]
  conversion = families["support_resistance_conversion"]
  assert conversion["object_class"] == "STRUCTURAL_RELATION"
  assert conversion["role_conversion_contract"][
      "source_breakout_and_retest_required"
  ] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "transition_first_event",
        "double_top_neckline_slope",
        "head_shoulders_left_neck",
        "breakout_boundary_geometry",
        "failed_direction_contract",
        "failed_creation_event",
        "failed_history_contract",
        "retest_ordinal",
        "range_geometry_missing",
        "conversion_role_contract",
        "range_geometry_misplaced",
    ],
)
def test_family_specific_p0_mutations_fail_closed(mutation: str) -> None:
  payload = _payload()
  families = _families(payload)
  if mutation == "transition_first_event":
    families["trend_transition"]["phase_contract"]["branch_transitions"][0][
        "event"
    ] = "LINKED_TREND_WEAKENING"
  elif mutation == "double_top_neckline_slope":
    families["double_top"]["formula_contracts"][0]["name"] = (
        "neckline_slope_norm"
    )
  elif mutation == "head_shoulders_left_neck":
    anchors = families["head_shoulders_top"]["anchors"]
    anchors[:] = [item for item in anchors if item["role"] != "left_neck_trough"]
  elif mutation == "breakout_boundary_geometry":
    families["breakout"]["instance_fields"].remove("boundary_geometry")
  elif mutation == "failed_direction_contract":
    del families["failed_breakout"]["direction_contract"]
  elif mutation == "failed_creation_event":
    families["failed_breakout"]["object_creation_event"] = (
        "SOURCE_BREAKOUT_ATTEMPT_AVAILABLE"
    )
  elif mutation == "failed_history_contract":
    del families["failed_breakout"][
        "historical_source_evidence_available_at_creation_only"
    ]
  elif mutation == "retest_ordinal":
    families["retest"]["instance_fields"].remove("retest_ordinal")
  elif mutation == "range_geometry_missing":
    del families["range"]["range_geometry"]
  elif mutation == "conversion_role_contract":
    del families["support_resistance_conversion"]["role_conversion_contract"]
  elif mutation == "range_geometry_misplaced":
    families["trend_up"]["range_geometry"] = "HORIZONTAL_ZONE_PAIR"
  else:  # pragma: no cover - parameter list is closed above
    raise AssertionError(f"unknown mutation {mutation}")

  assert _errors(payload), mutation


def test_all_70_assumptions_remain_open_and_execution_is_false() -> None:
  payload = _payload()
  registry = payload["resolved_contract"]["assumption_registry"]
  inherited = registry["inherited_from_v1"]
  new = registry["new_v2_assumptions"]
  ids = inherited["assumption_ids"] + [item["assumption_id"] for item in new]
  assert len(inherited["assumption_ids"]) == 61
  assert len(new) == 9
  assert len(ids) == len(set(ids)) == 70
  assert inherited["status"] == "NEEDS_HUMAN_REVIEW"
  assert inherited["proposed_default"] is None
  assert inherited["reviewer_decision"] is None
  assert all(
      item["status"] == "NEEDS_HUMAN_REVIEW"
      and item["proposed_default"] is None
      and item["reviewer_decision"] is None
      for item in new
  )
  assert payload["execution"]
  assert all(value is False for value in payload["execution"].values())
  assert payload["review_outcome_v1"]["mg1b_status"] == "BLOCKED"


def test_duplicate_yaml_and_json_keys_are_rejected(tmp_path: Path) -> None:
  duplicate = tmp_path / "duplicate.yaml"
  duplicate.write_text(
      "schema_version: shadowed\n" + REVISION_YAML.read_text(encoding="utf-8"),
      encoding="utf-8",
  )
  with pytest.raises(revision.RevisionValidationError, match="duplicate YAML key"):
    revision.load_revision(duplicate)
  with pytest.raises(revision.RevisionValidationError, match="duplicate JSON key"):
    revision.loads_json_unique('{"decision":"one","decision":"two"}')


def test_closed_schema_and_false_freeze_fail_closed() -> None:
  payload = _payload()
  payload["unexpected_root_key"] = True
  assert revision.json_schema_errors(payload, repo_root=REPO_ROOT)

  frozen = _payload()
  frozen["decision"] = "FROZEN"
  frozen["human_review"]["status"] = "APPROVED"
  frozen["human_review"]["frozen"] = True
  errors = revision.validate_revision_payload(frozen, repo_root=REPO_ROOT)
  assert errors
  assert any("FROZEN" in error or "PENDING" in error for error in errors)


def test_v1_byte_tamper_is_detected_without_touching_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  original = revision._sha256_file
  target = REPO_ROOT / "configs/market_genome/pattern_ontology_v1.yaml"

  def tampered_hash(path: Path) -> str:
    return "0" * 64 if path.resolve() == target.resolve() else original(path)

  monkeypatch.setattr(revision, "_sha256_file", tampered_hash)
  errors = revision.validate_revision_payload(_payload(), repo_root=REPO_ROOT)
  assert any(
      error.startswith("v1_frozen/configs/market_genome/pattern_ontology_v1.yaml")
      for error in errors
  )


def test_stale_resolved_hash_is_rejected() -> None:
  payload = _payload()
  payload["resolved_contract"]["object_model"]["object_classes"].reverse()
  errors = revision.validate_revision_payload(payload, repo_root=REPO_ROOT)
  assert any("resolved_contract_sha256" in error for error in errors)


def test_object_mapping_relation_identity_and_variable_mapping_fail_closed() -> None:
  wrong_class = _payload()
  wrong_class["resolved_contract"]["families"][0]["object_class"] = "MORPHOLOGY"
  assert any("object_class" in error for error in _errors(wrong_class))

  missing_event_identity = _payload()
  missing_event_identity["resolved_contract"]["relationship_graph"][
      "required_fields"
  ].remove("source_event_ids")
  assert any("source_event" in error for error in _errors(missing_event_identity))

  overlapping_mapping = _payload()
  phase_contract = overlapping_mapping["resolved_contract"]["families"][0][
      "phase_contract"
  ]
  phase_contract["common_mapping"]["ACTIVE"].append(
      phase_contract["common_mapping"]["INACTIVE"][0]
  )
  assert any("disjoint phase partition" in error for error in _errors(overlapping_mapping))


def test_unreachable_terminal_and_projected_edge_fail_closed() -> None:
  unreachable = _payload()
  contract = unreachable["resolved_contract"]["families"][0]["phase_contract"]
  contract["branch_transitions"][0]["to_phase"] = contract["common_mapping"][
      "ACTIVE"
  ][1]
  assert any("unreachable phases" in error for error in _errors(unreachable))

  terminal_edge = _payload()
  contract = terminal_edge["resolved_contract"]["families"][0]["phase_contract"]
  contract["branch_transitions"].append(
      {
          "from_phase": contract["common_mapping"]["RESOLVED"][0],
          "to_phase": contract["common_mapping"]["ACTIVE"][0],
          "event": "ILLEGAL_REACTIVATION",
      }
  )
  errors = _errors(terminal_edge)
  assert any("terminal phase" in error for error in errors)
  assert any("illegal common projection" in error for error in errors)

  invalid_projection = _payload()
  contract = invalid_projection["resolved_contract"]["families"][0][
      "phase_contract"
  ]
  contract["branch_transitions"].append(
      {
          "from_phase": contract["common_mapping"]["ACTIVE"][0],
          "to_phase": contract["common_mapping"]["INACTIVE"][0],
          "event": "ILLEGAL_RESET",
      }
  )
  assert any(
      "illegal common projection" in error for error in _errors(invalid_projection)
  )


def test_invalidation_transition_must_target_invalidated_phase() -> None:
  payload = _payload()
  contract = payload["resolved_contract"]["families"][0]["phase_contract"]
  contract["invalidation_transitions"][0]["to_phase"] = contract[
      "common_mapping"
  ]["RESOLVED"][0]
  assert any(
      "invalidation transition target" in error for error in _errors(payload)
  )


@pytest.mark.parametrize(
    "mutation",
    [
        "anchor_state",
        "review_mode",
        "scale_required_field",
        "unknown_assumption_id",
    ],
)
def test_exact_registries_and_assumption_refs_fail_closed(mutation: str) -> None:
  payload = _payload()
  contract = payload["resolved_contract"]
  if mutation == "anchor_state":
    contract["anchor_contract"]["states"].append("ARCHIVED")
  elif mutation == "review_mode":
    contract["review_protocol"]["review_modes"].append("MODEL_ASSISTED")
  elif mutation == "scale_required_field":
    contract["scale_spec_contract"]["required_fields"].append(
        "unreviewed_scale_field"
    )
  elif mutation == "unknown_assumption_id":
    contract["families"][0]["assumption_ids"][0] = (
        "MG1A1.UNKNOWN.CONTRACT.001"
    )
  else:  # pragma: no cover - parameter list is closed above
    raise AssertionError(f"unknown mutation {mutation}")

  assert _errors(payload), mutation


def test_every_registered_assumption_has_a_bound_usage_source() -> None:
  payload = _payload()
  payload["resolved_contract"]["object_model"]["assumption_ids"].remove(
      "MG1A1.COMMON.OBJECT_MODEL.001"
  )
  errors = _errors(payload)
  assert any(
      "assumption" in error and "referenc" in error for error in errors
  )


@pytest.mark.parametrize(
    "embedded_value",
    [
        "normalized_price_change_threshold_2",
        "absolute_5_point_threshold",
        "normalized_price_change_above_2_percent",
        "normalized_price_change_gt_2",
        "at_least_2_bars",
        "normalized_price_change_threshold_two",
        "at_least_two_bars",
    ],
)
def test_numeric_mutations_fail_closed(embedded_value: str) -> None:
  numeric_scalar = _payload()
  numeric_scalar["resolved_contract"]["families"][0]["formula_contracts"][0][
      "numerator"
  ] = 2
  assert revision.validate_revision_payload(numeric_scalar, repo_root=REPO_ROOT)

  embedded_numeric = _payload()
  embedded_numeric["resolved_contract"]["families"][0]["formula_contracts"][0][
      "numerator"
  ] = embedded_value
  assert any(
      "embedded numeric detector threshold" in error
      for error in _errors(embedded_numeric)
  )


@pytest.mark.parametrize(
    "forbidden_value",
    [
        "FUTURE_OUTCOME_BUY_SIGNAL",
        "next_week_return",
        "following_bar_return",
        "long_entry_signal",
        "expected_return_label",
        "outcome_classification",
        "profitable_outcome",
    ],
)
def test_future_trading_mutations_fail_closed(forbidden_value: str) -> None:
  future_trade = _payload()
  future_trade["resolved_contract"]["families"][0]["topology_rules"].append(
      forbidden_value
  )
  assert any(
      "forbidden future/trading semantics" in error
      for error in _errors(future_trade)
  )


def test_versioned_identifier_is_not_a_numeric_threshold() -> None:
  payload = _payload()
  payload["resolved_contract"]["families"][0]["formula_contracts"][0][
      "numerator"
  ] = "normalized_price_change_2"
  assert _errors(payload) == []


@pytest.mark.parametrize("cycle_kind", ["self", "two_node"])
def test_formula_dependency_cycles_fail_closed(cycle_kind: str) -> None:
  payload = _payload()
  formulas = _families(payload)["trend_up"]["formula_contracts"]
  if cycle_kind == "self":
    formulas[0]["numerator"] = formulas[0]["name"]
  else:
    formulas[0]["numerator"] = formulas[1]["name"]
    formulas[1]["numerator"] = formulas[0]["name"]
  errors = _errors(payload)
  assert any("cycle" in error.lower() or "acyclic" in error.lower() for error in errors)


def test_absolute_price_denominator_fails_closed() -> None:
  payload = _payload()
  _families(payload)["trend_up"]["formula_contracts"][0]["denominator"] = (
      "absolute_price"
  )
  errors = _errors(payload)
  assert any(
      "absolute" in error.lower() and "denominator" in error.lower()
      for error in errors
  )


def test_snapshot_consistency_detects_changed_frozen_input(tmp_path: Path) -> None:
  relative = "frozen/input.yaml"
  target = tmp_path / relative
  target.parent.mkdir(parents=True)
  snapshot = b"decision: ORIGINAL\n"
  target.write_bytes(snapshot)
  target.write_bytes(b"decision: CHANGED\n")
  errors = revision._snapshot_consistency_errors(
      tmp_path, {relative: snapshot}, "v1_frozen"
  )
  assert any("input changed during validation" in error for error in errors)


def test_cli_config_snapshot_change_fails_without_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  config = tmp_path / "revision.yaml"
  config.write_bytes(REVISION_YAML.read_bytes())
  output_json = tmp_path / "must_not_exist.json"
  output_md = tmp_path / "must_not_exist.md"
  original_loader = revision._load_revision_snapshot

  def load_then_change(path: Path) -> tuple[dict, bytes]:
    payload, snapshot = original_loader(path)
    path.write_bytes(snapshot + b"# concurrent change\n")
    return payload, snapshot

  monkeypatch.setattr(revision, "_load_revision_snapshot", load_then_change)
  monkeypatch.setattr(
      revision,
      "_parse_args",
      lambda: SimpleNamespace(
          profile=revision.PROFILE_NAME,
          repo_root=REPO_ROOT,
          config=config,
          output_json=output_json,
          output_md=output_md,
          strict=True,
      ),
  )
  assert revision.main() == 2
  assert "config input changed during validation" in capsys.readouterr().out
  assert not output_json.exists()
  assert not output_md.exists()


def test_cli_schema_snapshot_change_fails_without_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  config = tmp_path / "revision.yaml"
  config.write_bytes(REVISION_YAML.read_bytes())
  schema = tmp_path / "schema.json"
  schema_snapshot = revision.revision_schema_path(REPO_ROOT).read_bytes()
  schema.write_bytes(schema_snapshot)
  output_json = tmp_path / "must_not_exist.json"
  output_md = tmp_path / "must_not_exist.md"
  original_validate = revision.validate_revision_payload

  def validate_then_change(payload: dict, **kwargs: object) -> list[str]:
    errors = original_validate(payload, **kwargs)
    schema.write_bytes(schema_snapshot + b"\n")
    return errors

  monkeypatch.setattr(revision, "revision_schema_path", lambda _root=None: schema)
  monkeypatch.setattr(revision, "validate_revision_payload", validate_then_change)
  monkeypatch.setattr(
      revision,
      "_parse_args",
      lambda: SimpleNamespace(
          profile=revision.PROFILE_NAME,
          repo_root=REPO_ROOT,
          config=config,
          output_json=output_json,
          output_md=output_md,
          strict=True,
      ),
  )
  assert revision.main() == 2
  assert "schema input changed during validation" in capsys.readouterr().out
  assert not output_json.exists()
  assert not output_md.exists()


def test_cli_passes_cpu_only_and_invalid_input_writes_no_outputs(
    tmp_path: Path,
) -> None:
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  valid_json = tmp_path / "valid.json"
  valid_md = tmp_path / "valid.md"
  valid = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.revision",
          "--repo-root",
          str(REPO_ROOT),
          "--output-json",
          str(valid_json),
          "--output-md",
          str(valid_md),
          "--strict",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert valid.returncode == 0, valid.stdout + valid.stderr
  assert valid_json.is_file()
  assert valid_md.is_file()
  result = json.loads(valid_json.read_text(encoding="utf-8"))
  assert result["overall"] == "pass"
  assert result["gates"]["human_freeze_allowed"] is False
  assert result["gates"]["mg1b_allowed"] is False

  invalid_payload = _payload()
  contract = invalid_payload["resolved_contract"]["families"][0][
      "phase_contract"
  ]
  contract["branch_transitions"].append(
      {
          "from_phase": contract["common_mapping"]["RESOLVED"][0],
          "to_phase": contract["common_mapping"]["ACTIVE"][0],
          "event": "ILLEGAL_REACTIVATION",
      }
  )
  invalid_path = tmp_path / "invalid.json"
  invalid_path.write_text(
      json.dumps(_rehash(invalid_payload), sort_keys=True), encoding="utf-8"
  )
  invalid_json = tmp_path / "must_not_exist.json"
  invalid_md = tmp_path / "must_not_exist.md"
  invalid = subprocess.run(
      [
          sys.executable,
          "-m",
          "alphatrade.market_genome.ontology.revision",
          "--repo-root",
          str(REPO_ROOT),
          "--config",
          str(invalid_path),
          "--output-json",
          str(invalid_json),
          "--output-md",
          str(invalid_md),
          "--strict",
      ],
      cwd=REPO_ROOT,
      env=env,
      check=False,
      capture_output=True,
      text=True,
  )
  assert invalid.returncode == 1, invalid.stdout + invalid.stderr
  assert not invalid_json.exists()
  assert not invalid_md.exists()
