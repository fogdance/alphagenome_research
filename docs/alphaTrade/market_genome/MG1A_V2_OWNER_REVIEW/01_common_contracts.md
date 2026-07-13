# 公共合同

[返回审核总览](README.md)

本章共 21 条。AI 内容仅作参考，最终决定由 `PROJECT_OWNER` 作出。

<a id="item-01"></a>

## 01. `MG1A.COMMON.LIFECYCLE.001`

**要确认的问题**：是否同意冻结通用状态转换、终止态重置行为及各形态族特有的阶段语义？

<details><summary>查看英文原始问题</summary>

Freeze common transitions, terminal reset behavior, and family-specific phase semantics.

</details>

**当前定义定位**：
- `resolved_contract.common_lifecycle`
- `resolved_contract.families[*].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
common_lifecycle:
  assumption_ids:
  - MG1A.COMMON.LIFECYCLE.001
  - MG1A1.COMMON.LIFECYCLE_MAPPING.001
  statuses:
  - INACTIVE
  - ACTIVE
  - RESOLVED
  - INVALIDATED
  projected_edges:
  - INACTIVE_TO_ACTIVE
  - ACTIVE_TO_ACTIVE
  - ACTIVE_TO_RESOLVED
  - ACTIVE_TO_INVALIDATED
  mapping_policy: VARIABLE_LENGTH_MANY_TO_ONE
  branch_transitions_required: true
  retest_and_continuation_must_be_separate: true
  terminal_objects_are_immutable: true
  reactivation_requires_new_pattern_id: true
  every_invalidated_phase_requires_explicit_causal_in_edge: true
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Terminal immutability, new identity on reactivation, and structurally valid family graphs
  close the reset defect, but event labels such as TERMINATION_RECORDED and
  RANGE_TERMINATION_RECORDED do not define earliest causal evidence or persistent-state completion.
- 建议修改：
- Add a versioned phase-event predicate registry with observation basis, availability timestamp,
  precedence, and causal persistent-state termination; validate every edge reference against it.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.LIFECYCLE.001
owner_decision: REQUEST_REVISION
owner_rationale: The four coarse lifecycle statuses may remain, but named transition events are not yet
  operationally defined for every pattern family.
required_changes:
- Define every family-specific phase and transition event with executable entry, continuation, resolution,
  invalidation, and expiry predicates.
- For each event, specify its observation basis, earliest availability timestamp, thresholds or tolerances,
  time window, precedence, and causal evidence.
- Validate every branch and invalidation transition against a versioned phase-event predicate registry.
- Add a concrete lifecycle example for every pattern family.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:19:14+08:00'
```

---

<a id="item-02"></a>

## 02. `MG1A.COMMON.SCALE.001`

**要确认的问题**：是否同意冻结尺度标识、尺度窗口统计及尺度特定的参数策略？

<details><summary>查看英文原始问题</summary>

Freeze scale identifiers, scale-window statistics, and scale-specific parameter policy.

</details>

**当前定义定位**：
- `resolved_contract.scale_spec_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
scale_spec_contract:
  assumption_ids:
  - MG1A.COMMON.SCALE.001
  - MG1A.COMMON.SESSION.001
  - MG1A1.COMMON.SCALE_SPEC.001
  required_fields:
  - scale_id
  - scale_registry_version
  - scale_registry_sha256
  - bar_resolution
  - aggregation_policy
  - session_alignment
  - pivot_prominence_scale_ref
  - formation_span_bars
  - formation_span_session_bars
  - scale_octave
  - encoder_scale_ref
  bar_resolution: INSTANCE_PARAMETER
  aggregation_policy: SESSION_ALIGNED_CLOSED_BAR
  session_alignment: VERSIONED_INSTANCE_PARAMETER
  pivot_prominence_scale_ref: VERSIONED_CAUSAL_INSTANCE_PARAMETER
  formation_span_bars: INSTANCE_OBSERVATION
  formation_span_session_bars: INSTANCE_OBSERVATION
  scale_octave: OPTIONAL_INSTANCE_PARAMETER
  encoder_scale_ref: OPTIONAL_MODEL_METADATA_NOT_FAMILY_IDENTITY
  nullable_fields:
  - scale_octave
  - encoder_scale_ref
  scale_id_must_resolve_in_versioned_registry: true
  topology_scale_invariant: true
  scale_specific_parameter_policy: TRAIN_PARTITION_ONLY
  test_selection_allowed: false
```

</details>

### AI 参考意见（无审批权）

- 建议：`DEFER`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为缺少外部证据或注册表；请判断是否延后。
- 风险摘要：ScaleSpec requires versioned scale resolution and train-only selection, but no legal
  scale/base-timeframe registry, causal scale-window statistic definitions, or shared-versus-
  specific parameter matrix is available.
- 建议修改：
- Define the registry schema and exact shared-versus-scale-specific parameter matrix without
  selecting values from validation or test data.
- 恢复审核所需证据：
- Provide content-addressed scale and base-timeframe registries plus causal statistic definitions;
  train-selected values may be supplied later from the training partition.

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.SCALE.001
owner_decision: REQUEST_REVISION
owner_rationale: Each timeframe must detect and record technical patterns independently; a lower-timeframe
  pattern is not a higher-timeframe pattern merely because it falls inside one higher-timeframe bar.
required_changes:
- Run pattern recognition independently on the closed bars of every registered timeframe.
- Preserve the detecting timeframe as part of each pattern instance identity; do not promote, collapse,
  or relabel a lower-timeframe pattern as a higher-timeframe pattern.
- Require a higher-timeframe pattern to satisfy the same family definition using that timeframe's own
  bars and anchors.
- Keep any cross-timeframe association as a separate optional relationship that cannot change either pattern's
  scale identity.
- Define the legal timeframe registry, session-aligned bar construction, and detection parameters for
  each timeframe.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:23:44+08:00'
```

---

<a id="item-03"></a>

## 03. `MG1A.COMMON.NORMALIZATION.001`

**要确认的问题**：是否同意选定由训练数据拟合后冻结的价格归一化器及其拟合边界？

<details><summary>查看英文原始问题</summary>

Select the train-frozen price normalizer and its fitting boundary.

</details>

**当前定义定位**：
- `resolved_contract.geometry_formula_contract`
- `resolved_contract.families[*].formula_contracts`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
geometry_formula_contract:
  assumption_ids:
  - MG1A.COMMON.NORMALIZATION.001
  - MG1A.COMMON.FILTER_MISSINGNESS.001
  - MG1A1.COMMON.FORMULA_CONTRACT.001
  required_fields:
  - name
  - numerator
  - denominator
  - sign
  - unit
  - missingness
  - clipping_policy
  - fit_boundary
  sign_values:
  - SIGNED
  - NONNEGATIVE
  - UNIT_INTERVAL_WHEN_DEFINED
  missingness_values:
  - UNKNOWN_WITH_MASK
  - NOT_APPLICABLE_WITH_MASK
  clipping_policy_values:
  - NO_CLIPPING
  - TRAIN_FROZEN_CLIP_POLICY
  fit_boundary_values:
  - NOT_APPLICABLE
  - TRAIN_PARTITION_ONLY
  absolute_price_denominator_allowed: false
  numeric_detector_thresholds_allowed: false
  causality: PREFIX_ONLY
  denominator_zero_policy: UNKNOWN_WITH_MASK
  insufficient_history_policy: UNKNOWN_WITH_MASK
  formula_dependencies_must_be_acyclic: true
  every_output_has_exactly_one_formula: true
```

</details>

### AI 参考意见（无审批权）

- 建议：`DEFER`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为缺少外部证据或注册表；请判断是否延后。
- 风险摘要：The formula contract bans absolute-price denominators and fixes train-only fit boundaries,
  but causal_price_scale, lookback, stability floor, field bindings, and fitted artifact
  version/hash are absent; the v1 question explicitly asks to select the train-frozen normalizer.
- 建议修改：
- Add a typed normalizer registry with field bindings, lookback, stability floor, fit partition,
  artifact version, and hash requirements.
- 恢复审核所需证据：
- Produce the selected normalizer and hash-bound fit metadata from training-partition evidence; no
  GPU is required to review the resulting artifact.

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.NORMALIZATION.001
owner_decision: DEFER
owner_rationale: Review must wait until training data and a concrete normalization design are available.
required_changes: []
evidence_needed_to_resume:
- Provide the selected normalizer, field bindings, fitting window, stability floor, and missing-value
  behavior.
- Provide training-partition fit metadata together with the normalizer version and content hash.
replacement_proposition: []
reviewed_at: '2026-07-13T20:25:15+08:00'
```

---

<a id="item-04"></a>

## 04. `MG1A.COMMON.SWING_CAUSALITY.001`

**要确认的问题**：是否同意冻结锚点来源方法，以及从 event_eob 到 available_as_of_eob 的延迟？

<details><summary>查看英文原始问题</summary>

Freeze anchor source methods and delay between event_eob and available_as_of_eob.

</details>

**当前定义定位**：
- `resolved_contract.anchor_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
anchor_contract:
  assumption_ids:
  - MG1A.COMMON.SWING_CAUSALITY.001
  - MG1A.COMMON.PARTIAL_BAR.001
  - MG1A1.COMMON.ANCHOR_REVISION.001
  states:
  - PROVISIONAL
  - CONFIRMED
  - REVISED
  - SUPERSEDED
  - REVOKED
  state_transitions:
    PROVISIONAL:
    - CONFIRMED
    - REVISED
    - REVOKED
    REVISED:
    - CONFIRMED
    - SUPERSEDED
    - REVOKED
    CONFIRMED:
    - SUPERSEDED
    SUPERSEDED: []
    REVOKED: []
  required_fields:
  - anchor_id
  - anchor_lineage_id
  - role
  - event_eob
  - available_as_of_eob
  - as_of_eob
  - source_bar_ref
  - source_bar_closed
  - anchor_state
  - anchor_revision
  - supersedes_anchor_id
  - revision_reason
  - normalized_price
  - price_basis
  - confirmation_basis
  - scale_spec_ref
  - source_method
  - source_method_version
  - assumption_ids
  price_basis_values:
  - HIGH
  - LOW
  - CLOSE
  - BODY_HIGH
  - BODY_LOW
  - SETTLEMENT
  - VWAP
  confirmation_basis_values:
  - PIVOT_SOURCE_RULE
  - CLOSED_BAR_CROSS
  - CLOSED_BAR_HOLD
  - ANCHOR_SEQUENCE
  - ZONE_ROLE_TEST
  time_order: EVENT_NOT_AFTER_AVAILABILITY_NOT_AFTER_OBSERVATION
  committed_anchor_requires_closed_bar: true
  partial_bar_namespace: PROVISIONAL_RUNTIME_ONLY
  committed_history_is_append_only: true
  revision_zero_requires_no_predecessor: true
  revision_successor_requires_same_lineage_role_and_scale: true
  confirmed_revision_reason: SOURCE_CORRECTION_ONLY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Anchor time ordering and revision lineage are explicit, but the allowed source-method
  registry, per-method availability delay, pivot tie rule, and full replacement/retraction semantics
  are not defined.
- 建议修改：
- Add a versioned anchor-source registry with causal delay, tie, replacement, retraction, and
  source-correction rules, then validate source_method and source_method_version against it.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.SWING_CAUSALITY.001
owner_decision: APPROVE_AS_IS
owner_rationale: Approve the occurrence-time, availability-time, closed-bar, and revision-lineage principles;
  specific pivot algorithms and their confirmation delays may be defined later.
required_changes: []
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:26:34+08:00'
```

---

<a id="item-05"></a>

## 05. `MG1A.COMMON.PARTIAL_BAR.001`

**要确认的问题**：是否同意明确各阶段可使用会被修订的未完成 bar，还是只能使用已收盘 bar？

<details><summary>查看英文原始问题</summary>

Decide whether phases may use revisable partial bars or only closed bars.

</details>

**当前定义定位**：
- `resolved_contract.anchor_contract.committed_anchor_requires_closed_bar`
- `resolved_contract.anchor_contract.partial_bar_namespace`
- `resolved_contract.families[family=trend_up|trend_down|range].anchors[role=current_frontier]`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
anchor_contract:
  committed_anchor_requires_closed_bar: true
  partial_bar_namespace: PROVISIONAL_RUNTIME_ONLY
  committed_history_is_append_only: true
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：Committed anchors require closed bars, partial bars are confined to PROVISIONAL_RUNTIME_ONLY,
  and committed history is append-only; family frontier anchors use the provisional runtime
  requirement.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.PARTIAL_BAR.001
owner_decision: APPROVE_AS_IS
owner_rationale: null
required_changes: []
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:27:27+08:00'
```

---

<a id="item-06"></a>

## 06. `MG1A.COMMON.SESSION.001`

**要确认的问题**：是否同意冻结交易时段日历、节假日、缺失 bar 及时间桶边界语义？

<details><summary>查看英文原始问题</summary>

Freeze session calendar, holiday, missing-bar, and bucket-boundary semantics.

</details>

**当前定义定位**：
- `resolved_contract.scale_spec_contract.aggregation_policy`
- `resolved_contract.scale_spec_contract.session_alignment`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
scale_spec_contract:
  aggregation_policy: SESSION_ALIGNED_CLOSED_BAR
  session_alignment: VERSIONED_INSTANCE_PARAMETER
```

</details>

### AI 参考意见（无审批权）

- 建议：`DEFER`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为缺少外部证据或注册表；请判断是否延后。
- 风险摘要：Session-aligned closed bars and versioned session alignment are required, but the
  authoritative instrument-calendar mapping and holiday, night-session, early-close, missing-bar,
  timezone, and unsupported-instrument behavior are unavailable.
- 建议修改：
- Define the calendar-registry schema and all exception behavior without inventing calendar
  contents.
- 恢复审核所需证据：
- Provide a content-addressed authoritative instrument-to-calendar registry containing the required
  session and exception rules.

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.SESSION.001
owner_decision: APPROVE_AS_IS
owner_rationale: Approve the session-aligned closed-bar and versioned-calendar principles; concrete authoritative
  calendars and instrument mappings may be supplied later.
required_changes: []
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:28:13+08:00'
```

---

<a id="item-07"></a>

## 07. `MG1A.COMMON.ROLL.001`

**要确认的问题**：是否同意冻结主力合约、换月边界、价格调整及合约存续期语义？

<details><summary>查看英文原始问题</summary>

Freeze active-contract, roll-boundary, adjustment, and contract-age semantics.

</details>

**当前定义定位**：
- `resolved_contract (absence of roll_contract)`

> **当前定义缺失。** 这一项不能按现有 v2 文本直接批准；请要求补充、延后或拒绝。

<details><summary>查看当前 v2 选定字段原文（MISSING）</summary>

```yaml
missing_contract: resolved_contract.roll_contract
meaning: Current v2 contains no roll contract to approve.
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：V2 has no roll contract and therefore does not preserve active-contract identity or encode
  roll event/version, contract age, TERMINATE_AT_ROLL, and the cross-roll anchor prohibition.
- 建议修改：
- Add a roll contract with active_contract_id, roll_event_id, mapping version, contract age,
  terminate-at-roll semantics, and explicit prohibition of cross-roll anchors.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.ROLL.001
owner_decision: REQUEST_REVISION
owner_rationale: The data source must switch discretely at the roll boundary; all pre-roll bars use the
  outgoing contract and all bars from the boundary onward use the incoming contract.
required_changes:
- Define a roll event with an exact effective closed-bar boundary, outgoing contract ID, and incoming
  contract ID.
- Require every bar before the boundary to come only from the outgoing contract and every bar at or after
  the boundary to come only from the incoming contract.
- Prohibit blending two contracts within one bar and persist active_contract_id and roll_event_id with
  the data.
- Terminate every active pattern instance at the roll boundary, prohibit cross-roll anchors, and start
  pattern recognition anew on the incoming contract.
- Use each active contract's raw OHLC prices for pattern recognition; do not use an adjusted continuous
  series.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:33:59+08:00'
```

---

<a id="item-08"></a>

## 08. `MG1A.COMMON.PROVENANCE.001`

**要确认的问题**：是否同意冻结每种基准用途可接受的真值来源及审核溯源要求？

<details><summary>查看英文原始问题</summary>

Freeze admissible truth sources and review provenance for each benchmark use.

</details>

**当前定义定位**：
- `resolved_contract.relationship_graph.required_fields`
- `resolved_contract.review_protocol`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
anchor_required_fields:
- anchor_id
- anchor_lineage_id
- role
- event_eob
- available_as_of_eob
- as_of_eob
- source_bar_ref
- source_bar_closed
- anchor_state
- anchor_revision
- supersedes_anchor_id
- revision_reason
- normalized_price
- price_basis
- confirmation_basis
- scale_spec_ref
- source_method
- source_method_version
- assumption_ids
relation_required_fields:
- relation_id
- relation_type
- source_ref
- target_ref
- relation_event_id
- source_event_ids
- relation_ordinal
- event_eob
- available_as_of_eob
- as_of_eob
- relation_provenance
- assumption_ids
review_required_fields:
- review_mode
- review_id
- sample_id
- namespace
- reviewer_ids
- reviewer_decisions
- adjudicator_id
- adjudication_status
- ambiguity_status
- future_visibility_policy
- evidence_snapshot_sha256
- reviewed_at
- provenance
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Relation and review records carry provenance and model-assisted independent truth is
  forbidden, but benchmark eligibility by rule, human, synthetic, and model-assisted lineage plus
  immutable correction lineage is not defined.
- 建议修改：
- Add a closed provenance enum, benchmark-eligibility matrix, and append-only
  source/reviewer/adjudication/correction lineage bound to artifact hashes.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.PROVENANCE.001
owner_decision: REQUEST_REVISION
owner_rationale: All official shape labels for training and evaluation must be generated by one unified,
  versioned recognition algorithm; human confirmation is optional and evaluated models cannot generate
  their own targets.
required_changes:
- Define one canonical pattern-recognition algorithm as the sole generator of official training and evaluation
  shape labels.
- Bind every official label set to the detector identity, version, configuration hash, algorithm artifact
  hash, and evidence snapshot.
- Keep AI, model, program, or human proposals in a separate advisory namespace unless the canonical algorithm
  produces the official label.
- Record human confirmation as optional metadata without changing the canonical algorithm provenance.
- Prohibit any trained or evaluated model from generating its own official target labels.
- Publish an immutable new label-set version when the canonical algorithm changes; do not overwrite prior
  labels.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:38:32+08:00'
```

---

<a id="item-09"></a>

## 09. `MG1A.COMMON.MIRROR.001`

**要确认的问题**：是否同意明确哪些镜像形态族仅共享 schema，哪些还可以共享参数？

<details><summary>查看英文原始问题</summary>

Decide which mirrored families share schema only and which may share parameters.

</details>

**当前定义定位**：
- `resolved_contract.object_model.class_by_family`
- `resolved_contract.families (independent mirrored entries)`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family_orientation_partition:
- family: trend_up
  object_class: STRUCTURAL_STATE
  structural_orientation: UPWARD
  orientation_binding: FIXED
  allowed_instance_orientations:
  - UPWARD
  semantic_roles:
  - TREND_STATE
- family: trend_down
  object_class: STRUCTURAL_STATE
  structural_orientation: DOWNWARD
  orientation_binding: FIXED
  allowed_instance_orientations:
  - DOWNWARD
  semantic_roles:
  - TREND_STATE
- family: trend_transition
  object_class: LEVEL_EVENT
  structural_orientation: null
  orientation_binding: INSTANCE_BOUND
  allowed_instance_orientations:
  - UPWARD
  - DOWNWARD
  semantic_roles:
  - TRANSITION_EVENT
- family: double_top
  object_class: MORPHOLOGY
  structural_orientation: UPPER
  orientation_binding: FIXED
  allowed_instance_orientations:
  - UPPER
  semantic_roles:
  - REVERSAL_CANDIDATE
  - RANGE_BOUNDARY_REJECTION
  - CONTINUATION_STRUCTURE
  - UNCLASSIFIED
- family: double_bottom
  object_class: MORPHOLOGY
  structural_orientation: LOWER
  orientation_binding: FIXED
  allowed_instance_orientations:
  - LOWER
  semantic_roles:
  - REVERSAL_CANDIDATE
  - RANGE_BOUNDARY_REJECTION
  - CONTINUATION_STRUCTURE
  - UNCLASSIFIED
- family: head_shoulders_top
  object_class: MORPHOLOGY
  structural_orientation: UPPER
  orientation_binding: FIXED
  allowed_instance_orientations:
  - UPPER
  semantic_roles:
  - REVERSAL_CANDIDATE
  - RANGE_BOUNDARY_REJECTION
  - CONTINUATION_STRUCTURE
  - UNCLASSIFIED
- family: inverse_head_shoulders
  object_class: MORPHOLOGY
  structural_orientation: LOWER
  orientation_binding: FIXED
  allowed_instance_orientations:
  - LOWER
  semantic_roles:
  - REVERSAL_CANDIDATE
  - RANGE_BOUNDARY_REJECTION
  - CONTINUATION_STRUCTURE
  - UNCLASSIFIED
- family: breakout
  object_class: LEVEL_EVENT
  structural_orientation: null
  orientation_binding: INSTANCE_BOUND
  allowed_instance_orientations:
  - UPWARD
  - DOWNWARD
  semantic_roles:
  - BREAK_EVENT
- family: failed_breakout
  object_class: LEVEL_EVENT
  structural_orientation: null
  orientation_binding: INSTANCE_BOUND
  allowed_instance_orientations:
  - UPWARD
  - DOWNWARD
  semantic_roles:
  - FAILURE_EVENT
- family: retest
  object_class: LEVEL_EVENT
  structural_orientation: null
  orientation_binding: INSTANCE_BOUND
  allowed_instance_orientations:
  - UPWARD
  - DOWNWARD
  semantic_roles:
  - RETEST_EVENT
- family: range
  object_class: STRUCTURAL_STATE
  structural_orientation: NEUTRAL
  orientation_binding: FIXED
  allowed_instance_orientations:
  - NEUTRAL
  semantic_roles:
  - RANGE_STATE
- family: support_resistance_conversion
  object_class: STRUCTURAL_RELATION
  structural_orientation: null
  orientation_binding: INSTANCE_BOUND
  allowed_instance_orientations:
  - UPWARD
  - DOWNWARD
  semantic_roles:
  - ROLE_CONVERSION
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Mirrored families retain distinct contracts, but exact anchor, orientation, semantic-role,
  and relation transforms and the schema-only-sharing/independent-parameter rule are absent.
- 建议修改：
- Add a normative mirror transform table for every mirrored pair and state that schema may be shared
  while fitted parameters remain independent.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.MIRROR.001
owner_decision: REQUEST_REVISION
owner_rationale: Every mirrored pair must share both its structure and parameters exactly, with only the
  directional, high-low, and breakout-orientation transformations applied.
required_changes:
- Define a normative mirror-pair table for trend direction, double top and bottom, head-and-shoulders
  top and bottom, and directional level events.
- Define exact mirrored transforms for anchors, phase events, invalidations, orientations, semantic roles,
  formulas, and relations.
- Require both members of a mirror pair to reference the same parameter set within the same timeframe
  and registry version.
- Apply parameter changes atomically to both members and reject any mirror-pair parameter drift.
- Keep recognition independent across timeframes while enforcing mirror equality inside each timeframe.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:39:46+08:00'
```

---

<a id="item-10"></a>

## 10. `MG1A.COMMON.OVERLAP.001`

**要确认的问题**：是否同意冻结相关形态族之间的重叠、优先级、嵌套、失效及对象身份规则？

<details><summary>查看英文原始问题</summary>

Freeze overlap, precedence, nesting, expiry, and object identity across related families.

</details>

**当前定义定位**：
- `resolved_contract.relationship_graph`
- `resolved_contract.families[*].relationship_types`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
relationship_graph:
  assumption_ids:
  - MG1A.COMMON.OVERLAP.001
  - MG1A.COMMON.PROVENANCE.001
  - MG1A1.COMMON.RELATION_GRAPH.001
  relation_types:
  - NESTED_IN
  - CONTAINS
  - CONFIRMED_BY
  - INVALIDATED_BY
  - RETESTS
  - ORIGINATES_FROM
  - CONVERTS_ROLE_OF
  - SAME_STRUCTURE_DIFFERENT_SCALE
  required_fields:
  - relation_id
  - relation_type
  - source_ref
  - target_ref
  - relation_event_id
  - source_event_ids
  - relation_ordinal
  - event_eob
  - available_as_of_eob
  - as_of_eob
  - relation_provenance
  - assumption_ids
  cross_family_objects_remain_distinct: true
  cross_scale_objects_remain_distinct: true
  destructive_merge_allowed: false
  source_event_identity_required: true
  binary_edges_only: true
  derived_views:
  - source_pattern_ids
  - target_pattern_ids
  - related_level_ids
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Typed non-destructive relations preserve cross-family and cross-scale identity, but
  formation-epoch idempotency, endpoint compatibility, relation expiry, conflict precedence, and
  duplicate keys are undefined.
- 建议修改：
- Define an idempotency key and closed endpoint/cardinality, lifecycle, expiry, and conflict-
  precedence matrices for every relation type.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.OVERLAP.001
owner_decision: APPROVE_AS_IS
owner_rationale: Record every independently valid overlapping, nested, cross-family, or cross-timeframe
  pattern as a distinct object and connect the objects with explicit relationships rather than selecting
  only one.
required_changes: []
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:40:58+08:00'
```

---

<a id="item-11"></a>

## 11. `MG1A.COMMON.ZONE.001`

**要确认的问题**：是否同意冻结归一化价位区间的构造、漂移及触及语义？

<details><summary>查看英文原始问题</summary>

Freeze normalized level-zone construction, drift, and touch semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[*].anchors`
- `resolved_contract (absence of common zone_contract)`

> **当前定义缺失。** 这一项不能按现有 v2 文本直接批准；请要求补充、延后或拒绝。

<details><summary>查看当前 v2 选定字段原文（MISSING）</summary>

```yaml
missing_contract: resolved_contract.zone_contract
meaning: Current v2 contains no common zone contract to approve.
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Family contracts use derived versioned zones and zone-width formulas, but no common zone
  object defines aggregation, width, touch, cross, penetration, drift, merge, replacement, or
  expiry.
- 建议修改：
- Add a versioned causal zone contract covering construction, interaction predicates, material
  revision, merge, precedence, and expiry.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.ZONE.001
owner_decision: REQUEST_REVISION
owner_rationale: Key price levels use price zones except for an explicitly registered canonical-algorithm
  exception; the time-varying HEAD_SHOULDERS_TOP and INVERSE_HEAD_SHOULDERS neckline follows its selected
  detector formula.
required_changes:
- Add a common versioned zone contract used by every pattern family and key price level unless an owner-reviewed
  algorithm-specific exception is registered.
- Store zone identity, lower and upper bounds, reference center, source anchors, timeframe, construction
  method, and version.
- Base recognition decisions on zone bounds rather than treating the reference center as an exact tradable
  line, except for a registered algorithm-specific dynamic-level rule.
- Construct each zone independently per timeframe from related confirmed closed-bar pivots, using the
  lowest relevant pivot price as the lower bound and the highest as the upper bound.
- When only one relevant pivot is available, initialize the zone from the low to the high of that pivot's
  closed bar.
- Register the canonical dynamic scalar neckline from the shared head-and-shoulders algorithm as an explicit
  exception; do not expand it with pivot-bar Low-High values or replace it with a static zone.
- Define exact touch, cross, penetration, hold, rejection, reclaim, and full-recross predicates.
- Define zone drift, material revision, merge, replacement, precedence, and expiry behavior.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:46:59+08:00'
```

---

<a id="item-12"></a>

## 12. `MG1A.COMMON.FILTER_MISSINGNESS.001`

**要确认的问题**：是否同意冻结可选成交量和未平仓量过滤器的缺失值及基线语义？

<details><summary>查看英文原始问题</summary>

Freeze missingness and baseline semantics for optional volume/open-interest filters.

</details>

**当前定义定位**：
- `resolved_contract.geometry_formula_contract`
- `resolved_contract.families[*].confirmation_rules`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
geometry_formula_contract:
  assumption_ids:
  - MG1A.COMMON.NORMALIZATION.001
  - MG1A.COMMON.FILTER_MISSINGNESS.001
  - MG1A1.COMMON.FORMULA_CONTRACT.001
  required_fields:
  - name
  - numerator
  - denominator
  - sign
  - unit
  - missingness
  - clipping_policy
  - fit_boundary
  sign_values:
  - SIGNED
  - NONNEGATIVE
  - UNIT_INTERVAL_WHEN_DEFINED
  missingness_values:
  - UNKNOWN_WITH_MASK
  - NOT_APPLICABLE_WITH_MASK
  clipping_policy_values:
  - NO_CLIPPING
  - TRAIN_FROZEN_CLIP_POLICY
  fit_boundary_values:
  - NOT_APPLICABLE
  - TRAIN_PARTITION_ONLY
  absolute_price_denominator_allowed: false
  numeric_detector_thresholds_allowed: false
  causality: PREFIX_ONLY
  denominator_zero_policy: UNKNOWN_WITH_MASK
  insufficient_history_policy: UNKNOWN_WITH_MASK
  formula_dependencies_must_be_acyclic: true
  every_output_has_exactly_one_formula: true
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Formula outputs mask unknown and inapplicable values and optional filters do not define
  topology, but volume/OI input states, baselines, minimum history, venue/roll handling, and tri-
  state confirmation behavior are absent.
- 建议修改：
- Add optional-filter input enums and masks, causal baseline/minimum-history rules, venue/roll
  behavior, and a TRUE/FALSE/UNKNOWN confirmation truth table.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.FILTER_MISSINGNESS.001
owner_decision: REQUEST_REVISION
owner_rationale: Futures volume and open-interest data are complete and correct, but neither field participates
  in determining whether a technical pattern exists, is confirmed, or becomes invalid.
required_changes:
- Require valid volume and open-interest values for futures input data and treat their absence as a data-quality
  error rather than a normal unknown state.
- Remove volume and open-interest filters from canonical pattern existence, phase-transition, confirmation,
  and invalidation predicates.
- Permit volume and open interest to remain observational features without changing the official shape
  label.
- Do not define volume or open-interest baselines as requirements for canonical shape recognition.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:48:56+08:00'
```

---

<a id="item-13"></a>

## 13. `MG1A.COMMON.LABEL_TIMING.001`

**要确认的问题**：是否同意冻结目标时间字段，同时严格保持三个命名空间相互分离？

<details><summary>查看英文原始问题</summary>

Freeze target timing fields while preserving the hard three-namespace separation.

</details>

**当前定义定位**：
- `resolved_contract.label_namespaces`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
label_namespaces:
  assumption_ids:
  - MG1A.COMMON.LABEL_TIMING.001
  online_causal_phase:
    future_visible: false
    runtime_input_allowed: false
    fields:
    - phase_at_as_of
    - causal_evidence
    - ambiguity_status
  retrospective_completion:
    future_visible: true
    runtime_input_allowed: false
    fields:
    - completion_state
    - completion_event_eob
    - completion_censor_state
  future_outcome:
    future_visible: true
    runtime_input_allowed: false
    fields:
    - outcome_profile_ref
    - outcome_horizon_ref
    - outcome_censor_state
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Online, retrospective, and future namespaces are separated and all runtime-input flags are
  false, but completion predicates, horizon registry, censor transitions, correction lineage, and
  causal join keys are not defined.
- 建议修改：
- Add a version/hash-bound label timing contract with completion events, horizons, censor
  transitions, corrections, and causal join keys.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.COMMON.LABEL_TIMING.001
owner_decision: APPROVE_AS_IS
owner_rationale: Strictly separate information available at recognition time, retrospective pattern completion,
  and future market outcomes; exact completion and outcome windows may be defined later.
required_changes: []
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:50:31+08:00'
```

---

<a id="item-62"></a>

## 62. `MG1A1.COMMON.OBJECT_MODEL.001`

**要确认的问题**：是否同意冻结对象类别及其与各形态族的精确映射？

<details><summary>查看英文原始问题</summary>

Freeze object classes and exact family mapping

</details>

**当前定义定位**：
- `resolved_contract.object_model`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
object_model:
  assumption_ids:
  - MG1A1.COMMON.OBJECT_MODEL.001
  - MG1A1.COMMON.MORPHOLOGY_EFFECT.001
  object_classes:
  - STRUCTURAL_STATE
  - MORPHOLOGY
  - LEVEL_EVENT
  - STRUCTURAL_RELATION
  class_by_family:
    trend_up: STRUCTURAL_STATE
    trend_down: STRUCTURAL_STATE
    trend_transition: LEVEL_EVENT
    double_top: MORPHOLOGY
    double_bottom: MORPHOLOGY
    head_shoulders_top: MORPHOLOGY
    inverse_head_shoulders: MORPHOLOGY
    breakout: LEVEL_EVENT
    failed_breakout: LEVEL_EVENT
    retest: LEVEL_EVENT
    range: STRUCTURAL_STATE
    support_resistance_conversion: STRUCTURAL_RELATION
  structural_orientations:
  - UPPER
  - LOWER
  - UPWARD
  - DOWNWARD
  - NEUTRAL
  orientation_binding_values:
  - FIXED
  - INSTANCE_BOUND
  semantic_roles:
  - TREND_STATE
  - RANGE_STATE
  - REVERSAL_CANDIDATE
  - RANGE_BOUNDARY_REJECTION
  - CONTINUATION_STRUCTURE
  - TRANSITION_EVENT
  - BREAK_EVENT
  - FAILURE_EVENT
  - RETEST_EVENT
  - ROLE_CONVERSION
  - UNCLASSIFIED
  morphology_effect_separation:
    structural_orientation_is_future_effect: false
    semantic_role_is_future_outcome: false
    classic_reversal_role_requires_preceding_context: true
    future_effect_namespace: future_outcome
    future_effect_allowed_in_family_contract: false
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：Four exact object classes and an exhaustive 12-family mapping are explicit and validator-
  enforced; retest remains a LEVEL_EVENT and support-resistance conversion is a STRUCTURAL_RELATION.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A1.COMMON.OBJECT_MODEL.001
owner_decision: REQUEST_REVISION
owner_rationale: Keep pattern objects and append-only relationship edges as separate record kinds; relationship
  edges are not STRUCTURAL_RELATION pattern objects.
required_changes:
- Remove STRUCTURAL_RELATION from the pattern object-class enum.
- Retain STRUCTURAL_STATE for TREND_UP, TREND_DOWN, and RANGE; retain MORPHOLOGY for DOUBLE_TOP, DOUBLE_BOTTOM,
  HEAD_SHOULDERS_TOP, and INVERSE_HEAD_SHOULDERS; retain LEVEL_EVENT for BREAKOUT and FAILED_BREAKOUT.
- Represent BOX_RANGE and SIDEWAYS_RANGE as distinct closed range_type values under RANGE, not as merged
  identities.
- Remove TREND_TRANSITION, RETEST, and SUPPORT_RESISTANCE_CONVERSION and their transition, retest, and
  role-conversion semantic roles from the object model.
- Define relationship edges under their own append-only schema and never include them in pattern-family
  lifecycle counts.
- Preserve independent identities for every overlapping retained pattern object.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:34:05+08:00'
```

---

<a id="item-63"></a>

## 63. `MG1A1.COMMON.MORPHOLOGY_EFFECT.001`

**要确认的问题**：是否同意冻结方向和语义角色，并禁止未来结果信息泄漏？

<details><summary>查看英文原始问题</summary>

Freeze orientation and semantic role without future-effect leakage

</details>

**当前定义定位**：
- `resolved_contract.object_model.morphology_effect_separation`
- `resolved_contract.label_namespaces.future_outcome`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
morphology_effect_separation:
  structural_orientation_is_future_effect: false
  semantic_role_is_future_outcome: false
  classic_reversal_role_requires_preceding_context: true
  future_effect_namespace: future_outcome
  future_effect_allowed_in_family_contract: false
label_namespaces:
  assumption_ids:
  - MG1A.COMMON.LABEL_TIMING.001
  online_causal_phase:
    future_visible: false
    runtime_input_allowed: false
    fields:
    - phase_at_as_of
    - causal_evidence
    - ambiguity_status
  retrospective_completion:
    future_visible: true
    runtime_input_allowed: false
    fields:
    - completion_state
    - completion_event_eob
    - completion_censor_state
  future_outcome:
    future_visible: true
    runtime_input_allowed: false
    fields:
    - outcome_profile_ref
    - outcome_horizon_ref
    - outcome_censor_state
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：Structural orientation, semantic role, and future outcome are separate; future effect is
  forbidden in family contracts and classic reversal role requires preceding context plus causal
  confirmation.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A1.COMMON.MORPHOLOGY_EFFECT.001
owner_decision: APPROVE_AS_IS
owner_rationale: Structural orientation and semantic role must use only causally available morphology
  and preceding context; future performance remains isolated in future_outcome and cannot rewrite identity,
  role, or confirmation time.
required_changes: []
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:34:05+08:00'
```

---

<a id="item-64"></a>

## 64. `MG1A1.COMMON.LIFECYCLE_MAPPING.001`

**要确认的问题**：是否同意冻结各形态族可变的阶段定义及其到通用阶段的多对一映射？

<details><summary>查看英文原始问题</summary>

Freeze variable family phases and many-to-one common mapping

</details>

**当前定义定位**：
- `resolved_contract.common_lifecycle`
- `resolved_contract.families[*].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
common_lifecycle:
  assumption_ids:
  - MG1A.COMMON.LIFECYCLE.001
  - MG1A1.COMMON.LIFECYCLE_MAPPING.001
  statuses:
  - INACTIVE
  - ACTIVE
  - RESOLVED
  - INVALIDATED
  projected_edges:
  - INACTIVE_TO_ACTIVE
  - ACTIVE_TO_ACTIVE
  - ACTIVE_TO_RESOLVED
  - ACTIVE_TO_INVALIDATED
  mapping_policy: VARIABLE_LENGTH_MANY_TO_ONE
  branch_transitions_required: true
  retest_and_continuation_must_be_separate: true
  terminal_objects_are_immutable: true
  reactivation_requires_new_pattern_id: true
  every_invalidated_phase_requires_explicit_causal_in_edge: true
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：The four common statuses, exact projected edges, variable-length family vocabularies,
  disjoint many-to-one mappings, reachability, and terminality are explicit and validator-enforced;
  event semantics and path applicability remain scoped to inherited phase assumptions and the
  aggregate family gate.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A1.COMMON.LIFECYCLE_MAPPING.001
owner_decision: REQUEST_REVISION
owner_rationale: Use the owner-selected UNACTIVATED, ACTIVE, COMPLETED, and INVALIDATED names and map
  only transitions actually emitted by each canonical family algorithm.
required_changes:
- Replace INACTIVE with UNACTIVATED and RESOLVED with COMPLETED throughout schemas, validators, serializers,
  documents, and family contracts.
- Allow UNACTIVATED_TO_ACTIVE, ACTIVE_TO_ACTIVE, ACTIVE_TO_COMPLETED, ACTIVE_TO_INVALIDATED, and the event-object-specific
  UNACTIVATED_TO_COMPLETED transition.
- Instantiate FAILED_BREAKOUT directly as COMPLETED at its canonical reentry event without persisting
  an artificial ACTIVE phase.
- Keep confirmed BREAKOUT and its source RANGE ACTIVE until the versioned failure window closes, then
  complete both; inside the window invalidate the attempt and keep the source range active.
- Bind every family transition to its reviewed closed-bar predicate, threshold, time window, evidence
  priority, and canonical detector profile.
- Require each family to implement only its applicable edges; do not require every family to emit every
  common edge.
- Remove all lifecycle mappings for TREND_TRANSITION, RETEST, and SUPPORT_RESISTANCE_CONVERSION.
- Keep COMPLETED and INVALIDATED immutable; any later shape requires a new pattern identity, while data
  or algorithm corrections use append-only replacement provenance.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:38:36+08:00'
```

---

<a id="item-65"></a>

## 65. `MG1A1.COMMON.SCALE_SPEC.001`

**要确认的问题**：是否同意冻结 ScaleSpec 字段及版本化注册表？

<details><summary>查看英文原始问题</summary>

Freeze ScaleSpec fields and versioned registries

</details>

**当前定义定位**：
- `resolved_contract.scale_spec_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
scale_spec_contract:
  assumption_ids:
  - MG1A.COMMON.SCALE.001
  - MG1A.COMMON.SESSION.001
  - MG1A1.COMMON.SCALE_SPEC.001
  required_fields:
  - scale_id
  - scale_registry_version
  - scale_registry_sha256
  - bar_resolution
  - aggregation_policy
  - session_alignment
  - pivot_prominence_scale_ref
  - formation_span_bars
  - formation_span_session_bars
  - scale_octave
  - encoder_scale_ref
  bar_resolution: INSTANCE_PARAMETER
  aggregation_policy: SESSION_ALIGNED_CLOSED_BAR
  session_alignment: VERSIONED_INSTANCE_PARAMETER
  pivot_prominence_scale_ref: VERSIONED_CAUSAL_INSTANCE_PARAMETER
  formation_span_bars: INSTANCE_OBSERVATION
  formation_span_session_bars: INSTANCE_OBSERVATION
  scale_octave: OPTIONAL_INSTANCE_PARAMETER
  encoder_scale_ref: OPTIONAL_MODEL_METADATA_NOT_FAMILY_IDENTITY
  nullable_fields:
  - scale_octave
  - encoder_scale_ref
  scale_id_must_resolve_in_versioned_registry: true
  topology_scale_invariant: true
  scale_specific_parameter_policy: TRAIN_PARTITION_ONLY
  test_selection_allowed: false
```

</details>

### AI 参考意见（无审批权）

- 建议：`DEFER`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为缺少外部证据或注册表；请判断是否延后。
- 风险摘要：ScaleSpec fields, version/hash references, topology invariance, and train-only policy are
  coherent, but the actual scale, session, and prominence registries do not exist as reviewable
  artifacts.
- 建议修改：
- Finalize registry schemas and preserve the train-only selection boundary.
- 恢复审核所需证据：
- Provide content-addressed scale, session, and causal prominence registries matching the required
  fields.

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A1.COMMON.SCALE_SPEC.001
owner_decision: REQUEST_REVISION
owner_rationale: Freeze the versioned ScaleSpec interface now while deferring concrete calendars, per-timeframe
  parameter values, normalization, and train-fitted values.
required_changes:
- Require every formal object to reference a content-addressed scale registry entry containing timeframe
  identity, bar resolution, aggregation policy, session and calendar version, canonical detector version
  and hash, and effective parameter-profile version and hash.
- Detect every timeframe independently and assign independent pattern identities; never promote a lower-timeframe
  pattern to a higher-timeframe label.
- Keep family topology shared where the canonical algorithm is shared, while allowing separately versioned
  effective parameter profiles per timeframe.
- Keep formation-span observations on the object's own timeframe and do not use encoder scale metadata
  as family identity.
- Defer concrete trading calendars, timeframe parameter values, normalization, and train-fitted values
  without running training or GPU work in MG1-A.
- Block formal label generation for any timeframe whose required registry entry or effective detector
  profile is absent, unresolved, or hash-mismatched; do not block ontology schema freeze solely because
  those values are deferred.
- Prohibit test-set selection or tuning and require a new versioned profile for every later parameter
  change.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:40:11+08:00'
```

---

<a id="item-66"></a>

## 66. `MG1A1.COMMON.ANCHOR_REVISION.001`

**要确认的问题**：是否同意冻结锚点状态、修订转换及价格口径？

<details><summary>查看英文原始问题</summary>

Freeze anchor states revision transitions and price basis

</details>

**当前定义定位**：
- `resolved_contract.anchor_contract`
- `resolved_contract.families[*].anchors[*].state_requirement`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
anchor_contract:
  assumption_ids:
  - MG1A.COMMON.SWING_CAUSALITY.001
  - MG1A.COMMON.PARTIAL_BAR.001
  - MG1A1.COMMON.ANCHOR_REVISION.001
  states:
  - PROVISIONAL
  - CONFIRMED
  - REVISED
  - SUPERSEDED
  - REVOKED
  state_transitions:
    PROVISIONAL:
    - CONFIRMED
    - REVISED
    - REVOKED
    REVISED:
    - CONFIRMED
    - SUPERSEDED
    - REVOKED
    CONFIRMED:
    - SUPERSEDED
    SUPERSEDED: []
    REVOKED: []
  required_fields:
  - anchor_id
  - anchor_lineage_id
  - role
  - event_eob
  - available_as_of_eob
  - as_of_eob
  - source_bar_ref
  - source_bar_closed
  - anchor_state
  - anchor_revision
  - supersedes_anchor_id
  - revision_reason
  - normalized_price
  - price_basis
  - confirmation_basis
  - scale_spec_ref
  - source_method
  - source_method_version
  - assumption_ids
  price_basis_values:
  - HIGH
  - LOW
  - CLOSE
  - BODY_HIGH
  - BODY_LOW
  - SETTLEMENT
  - VWAP
  confirmation_basis_values:
  - PIVOT_SOURCE_RULE
  - CLOSED_BAR_CROSS
  - CLOSED_BAR_HOLD
  - ANCHOR_SEQUENCE
  - ZONE_ROLE_TEST
  time_order: EVENT_NOT_AFTER_AVAILABILITY_NOT_AFTER_OBSERVATION
  committed_anchor_requires_closed_bar: true
  partial_bar_namespace: PROVISIONAL_RUNTIME_ONLY
  committed_history_is_append_only: true
  revision_zero_requires_no_predecessor: true
  revision_successor_requires_same_lineage_role_and_scale: true
  confirmed_revision_reason: SOURCE_CORRECTION_ONLY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Five common states, transitions, lineage, time order, and price/confirmation bases are
  explicit, but family anchors use undeclared state_requirement predicates and an arbitrary
  replacement passes validation, so the family-to-state contract is not closed.
- 建议修改：
- Add a closed state-requirement predicate registry mapped to the five anchor states and enforce
  every family anchor reference in the validator.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A1.COMMON.ANCHOR_REVISION.001
owner_decision: REQUEST_REVISION
owner_rationale: Preserve causal closed-bar and append-only anchor lineage, but bind every retained anchor
  role and state requirement to its canonical detector.
required_changes:
- Keep PROVISIONAL, CONFIRMED, REVISED, SUPERSEDED, and REVOKED with append-only state transitions; provisional
  anchors are runtime-only and excluded from committed formal labels.
- Add a closed state-requirement registry for CONFIRMED, DERIVED_FROM_CONFIRMED, PROVISIONAL_ALLOWED_RUNTIME_ONLY,
  and CONFIRMED_AT_REENTRY and enforce every predicate in schema validation.
- Remove REVISED_UNTIL_RETURN because RETEST is no longer an official family.
- Bind every retained family anchor role to its canonical-detector price_basis and confirmation_basis
  and reject arbitrary substitutions.
- Use raw prices for recognition anchors and preserve event_eob, available_as_of_eob, and as_of_eob separately.
- Treat source-data corrections as same-lineage revisions; later market pivots are new lineages and never
  overwrite prior anchors.
- Preserve source method, algorithm version, effective parameter profile, scale, active contract, and
  assumption lineage on every committed anchor.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:34:05+08:00'
```

---

<a id="item-67"></a>

## 67. `MG1A1.COMMON.RELATION_GRAPH.001`

**要确认的问题**：是否同意冻结关系图中的身份、嵌套及重叠规则？

<details><summary>查看英文原始问题</summary>

Freeze relationship graph identity nesting and overlap rules

</details>

**当前定义定位**：
- `resolved_contract.relationship_graph`
- `resolved_contract.families[*].relationship_types`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
relationship_graph:
  assumption_ids:
  - MG1A.COMMON.OVERLAP.001
  - MG1A.COMMON.PROVENANCE.001
  - MG1A1.COMMON.RELATION_GRAPH.001
  relation_types:
  - NESTED_IN
  - CONTAINS
  - CONFIRMED_BY
  - INVALIDATED_BY
  - RETESTS
  - ORIGINATES_FROM
  - CONVERTS_ROLE_OF
  - SAME_STRUCTURE_DIFFERENT_SCALE
  required_fields:
  - relation_id
  - relation_type
  - source_ref
  - target_ref
  - relation_event_id
  - source_event_ids
  - relation_ordinal
  - event_eob
  - available_as_of_eob
  - as_of_eob
  - relation_provenance
  - assumption_ids
  cross_family_objects_remain_distinct: true
  cross_scale_objects_remain_distinct: true
  destructive_merge_allowed: false
  source_event_identity_required: true
  binary_edges_only: true
  derived_views:
  - source_pattern_ids
  - target_pattern_ids
  - related_level_ids
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Eight typed binary relations have identity, causal timestamps, source events, provenance, and
  non-destructive behavior, but endpoint types/cardinalities, mandatory links, idempotency, nesting
  constraints, expiry, and conflict precedence are absent.
- 建议修改：
- Add closed endpoint/cardinality and mandatory-edge matrices plus formation-epoch idempotency,
  nesting, lifecycle, expiry, and conflict rules.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A1.COMMON.RELATION_GRAPH.001
owner_decision: REQUEST_REVISION
owner_rationale: Store relationship edges as independent append-only records rather than pattern objects,
  with explicit endpoint and provenance contracts.
required_changes:
- Retain NESTED_IN, CONTAINS, CONFIRMED_BY, INVALIDATED_BY, ORIGINATES_FROM, and SAME_STRUCTURE_DIFFERENT_SCALE.
- Remove RETESTS and CONVERTS_ROLE_OF because their source families were removed.
- Add SUCCEEDS for a later confirmed trend related to a previously invalidated trend without recreating
  TREND_TRANSITION.
- Require BREAKOUT to originate from its source RANGE and FAILED_BREAKOUT to originate from its terminal
  BREAKOUT attempt and source RANGE through separate binary edges.
- Give every edge its own stable relation_id, event and availability times, source events, endpoint types,
  ordinal, provenance, and assumption lineage.
- Keep cross-family and cross-scale objects distinct; never destructively merge endpoints.
- Correct a relationship with a new relation ID and explicit superseding lineage rather than overwriting
  committed history.
- Validate allowed endpoint types, direction, cardinality, acyclicity where applicable, and source-event
  existence for every relation type.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:34:05+08:00'
```

---

<a id="item-68"></a>

## 68. `MG1A1.COMMON.FORMULA_CONTRACT.001`

**要确认的问题**：是否同意冻结公式的分子、分母、符号、缺失值及截断规则？

<details><summary>查看英文原始问题</summary>

Freeze formula numerator denominator sign missingness and clipping

</details>

**当前定义定位**：
- `resolved_contract.geometry_formula_contract`
- `resolved_contract.families[*].formula_contracts`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
geometry_formula_contract:
  assumption_ids:
  - MG1A.COMMON.NORMALIZATION.001
  - MG1A.COMMON.FILTER_MISSINGNESS.001
  - MG1A1.COMMON.FORMULA_CONTRACT.001
  required_fields:
  - name
  - numerator
  - denominator
  - sign
  - unit
  - missingness
  - clipping_policy
  - fit_boundary
  sign_values:
  - SIGNED
  - NONNEGATIVE
  - UNIT_INTERVAL_WHEN_DEFINED
  missingness_values:
  - UNKNOWN_WITH_MASK
  - NOT_APPLICABLE_WITH_MASK
  clipping_policy_values:
  - NO_CLIPPING
  - TRAIN_FROZEN_CLIP_POLICY
  fit_boundary_values:
  - NOT_APPLICABLE
  - TRAIN_PARTITION_ONLY
  absolute_price_denominator_allowed: false
  numeric_detector_thresholds_allowed: false
  causality: PREFIX_ONLY
  denominator_zero_policy: UNKNOWN_WITH_MASK
  insufficient_history_policy: UNKNOWN_WITH_MASK
  formula_dependencies_must_be_acyclic: true
  every_output_has_exactly_one_formula: true
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Every listed formula names numerator, denominator, sign, unit, masked missingness, clipping,
  and fit boundary and numeric thresholds are forbidden, but operands lack typed definitions and
  causal_price_scale remains unresolved.
- 建议修改：
- Add a typed operand/expression and normalization-reference registry while preserving train-only
  fitting and the numeric-threshold prohibition.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A1.COMMON.FORMULA_CONTRACT.001
owner_decision: REQUEST_REVISION
owner_rationale: Allow deterministic numeric thresholds required by the selected canonical detectors,
  while keeping every effective value versioned, content-addressed, scale-specific, and isolated from
  test-data tuning.
required_changes:
- Permit deterministic canonical-detector numeric thresholds and time windows only through the effective
  versioned parameter profile bound to the object's timeframe, detector version, and content hash.
- Use raw prices for all pattern-recognition predicates and keep every comparison operator, boundary convention,
  equality behavior, and closed-bar timing explicit.
- Require each derived geometry feature formula to declare typed numerator, denominator, sign, unit, missingness,
  clipping policy, fit boundary, and assumption lineage separately from recognition predicates.
- Defer normalization, train-fitted clipping values, and other training-derived transforms until the training-data
  and normalization scheme is reviewed; none may alter canonical formal labels.
- Prohibit fitting, selecting, or tuning detector thresholds, normalization, or clipping rules on test
  or evaluation data.
- Treat missing futures volume or open interest as a source-data error rather than a valid market observation.
- Permit explicit validity masks for genuinely inapplicable or insufficient-history derived features,
  without converting those masks into missing raw-market-data semantics.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:42:17+08:00'
```

---

<a id="item-69"></a>

## 69. `MG1A1.COMMON.REVIEW_PROTOCOL.001`

**要确认的问题**：是否同意冻结盲审、回顾性结果评估及裁决流程？

<details><summary>查看英文原始问题</summary>

Freeze blinded retrospective outcome and adjudication protocol

</details>

**当前定义定位**：
- `resolved_contract.review_protocol`

> **范围说明：** 摘录中的 TWO_REVIEWERS 是当前 v2 的待审核规则；它不为本次 ontology owner review 增加第二审核者，且负责人已要求在修订版中取消。

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
review_protocol:
  assumption_ids:
  - MG1A1.COMMON.REVIEW_PROTOCOL.001
  review_modes:
  - ONLINE_BLINDED
  - RETROSPECTIVE
  - OUTCOME
  required_fields:
  - review_mode
  - review_id
  - sample_id
  - namespace
  - reviewer_ids
  - reviewer_decisions
  - adjudicator_id
  - adjudication_status
  - ambiguity_status
  - future_visibility_policy
  - evidence_snapshot_sha256
  - reviewed_at
  - provenance
  adjudication_statuses:
  - NOT_REQUIRED
  - PENDING
  - ADJUDICATED
  ambiguity_statuses:
  - UNAMBIGUOUS
  - AMBIGUOUS
  - INSUFFICIENT_EVIDENCE
  future_visibility_by_mode:
    ONLINE_BLINDED: PREFIX_ONLY
    RETROSPECTIVE: COMPLETION_WINDOW_ONLY
    OUTCOME: DECLARED_OUTCOME_WINDOW_ONLY
  reviewer_identity_policy: PSEUDONYMOUS_IDS_NO_PERSONAL_DATA
  independent_review_policy: TWO_REVIEWERS_THEN_ADJUDICATE_DISAGREEMENT
  model_assisted_independent_truth_allowed: false
  minimum_independent_reviewers: TWO
  ambiguity_blocks_gold_eligibility: true
  ontology_freeze_requires_separate_v2_hash_bound_review: true
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Review modes, visibility, two-reviewer disagreement adjudication, ambiguity, evidence hash,
  and model-assisted-truth exclusion are explicit, but reviewer/adjudicator independence controls,
  amendment workflow, and hash-bound decision-record schema are incomplete.
- 建议修改：
- Define reviewer assignment and adjudicator independence attestations plus append-only, hash-bound
  decision, correction, and amendment records.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A1.COMMON.REVIEW_PROTOCOL.001
owner_decision: REQUEST_REVISION
owner_rationale: Remove mandatory dual review and adjudication because ontology approval has one project
  owner and formal training and evaluation labels are generated by the canonical versioned recognition
  algorithm.
required_changes:
- Replace TWO_REVIEWERS_THEN_ADJUDICATE_DISAGREEMENT and the minimum-two-reviewer requirement with single
  PROJECT_OWNER approval for ontology definitions.
- Remove mandatory adjudicator identity and adjudication-status fields from the ontology owner-review
  contract.
- Define the canonical algorithm version, effective parameter-profile version, scale entry, source-data
  snapshot, and their content hashes as the provenance of every formal training and evaluation label.
- Keep human inspection optional and append-only; record its reviewer, mode, evidence snapshot, finding,
  ambiguity, timestamp, and provenance without treating the finding as an alternate standard answer.
- Prohibit a human finding from directly overwriting an algorithm-generated formal label.
- Route a confirmed discrepancy to an algorithm, ontology, or parameter-profile change proposal; after
  approval, issue a new version and regenerate affected labels with explicit lineage.
- Retain ONLINE_BLINDED, RETROSPECTIVE, and OUTCOME visibility boundaries for optional audits so future
  information cannot leak into online-shape findings.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:43:57+08:00'
```

---
