# 突破、回踩与角色转换

[返回审核总览](README.md)

本章共 16 条。AI 内容仅作参考，最终决定由 `PROJECT_OWNER` 作出。

<a id="item-42"></a>

## 42. `MG1A.BREAKOUT.STRUCTURE.001`

**要确认的问题**：是否同意冻结 breakout 的方向依据、锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze breakout side basis, anchors, topology, geometry, duration, price ratios, and prior context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=breakout].anchors`
- `resolved_contract.families[family=breakout].topology_rules`
- `resolved_contract.families[family=breakout].formula_contracts`
- `resolved_contract.families[family=breakout].instance_fields`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: breakout
object_class: LEVEL_EVENT
structural_orientation: null
orientation_binding: INSTANCE_BOUND
allowed_instance_orientations:
- UPWARD
- DOWNWARD
semantic_roles:
- BREAK_EVENT
persistent_state: false
scale_spec: INSTANCE_PARAMETER
instance_fields:
- source_pattern_ids
- source_zone_id
- boundary_geometry
- cross_direction
anchors:
- role: reference_level_anchor
  cardinality: ONE_OR_MORE
  state_requirement: CONFIRMED
- role: reference_zone
  cardinality: ONE
  state_requirement: DERIVED_FROM_CONFIRMED
- role: approach_leg_start
  cardinality: ONE
  state_requirement: CONFIRMED
- role: boundary_cross_attempt
  cardinality: ONE
  state_requirement: CONFIRMED
- role: outside_close
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: post_cross_hold
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: retest_touch
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
topology_rules:
- VERSIONED_REFERENCE_ZONE_REQUIRED
- OUTWARD_CROSS_ATTEMPT_SEPARATE_FROM_ACCEPTED_BREAK
- CLOSED_BAR_CONFIRMATION_ONLY
- BOUNDARY_GEOMETRY_EXPLICIT
formula_contracts:
- name: cross_displacement_to_zone_width
  numerator: cross_price_minus_crossed_boundary_value
  denominator: reference_zone_width
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: outside_hold_displacement_to_zone_width
  numerator: causal_outside_hold_distance
  denominator: reference_zone_width
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: approach_duration_to_zone_age
  numerator: approach_session_bars
  denominator: causal_zone_age_session_bars
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
relationship_types:
- ORIGINATES_FROM
- CONFIRMED_BY
- INVALIDATED_BY
- RETESTS
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Source pattern and zone IDs, boundary geometry, and outward direction are required, but the
  common zone construction and exact boundary-cross geometry policies are undefined.
- 建议修改：
- Bind source identity to the common zone contract and define closed-bar boundary evaluation for
  horizontal, dynamic, and versioned boundaries.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.BREAKOUT.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Represent every outward range-break attempt as an independent object while preserving
  the distinct BOX_RANGE and SIDEWAYS_RANGE detector semantics.
required_changes:
- Give each breakout attempt a new stable identity and preserve source_range_type, source_range_id, direction,
  reference boundary evidence, first-cross time, and detector provenance.
- Preserve pending and accepted evidence separately for SIDEWAYS_RANGE and immediate closed-bar confirmation
  evidence for BOX_RANGE.
- Keep wick probes as diagnostic events; they must not create breakout objects without the canonical closed-bar
  cross.
- Require source range, attempt evidence, timeframe, and active contract to agree; never merge attempts
  from different ranges, types, timeframes, or contracts.
- Add explicit typed relations from every attempt to its source range and, when applicable, its FAILED_BREAKOUT
  object.
- Preserve overlapping attempts independently and never overwrite or reuse a terminal attempt identity.
- Volume and open interest remain observational and must not determine breakout structure.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:47:10+08:00'
```

---

<a id="item-43"></a>

## 43. `MG1A.BREAKOUT.PHASE.001`

**要确认的问题**：是否同意冻结 breakout 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze breakout phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=breakout].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - INACTIVE
  - APPROACHING_BOUNDARY
  - CROSS_ATTEMPT
  - OUTSIDE_CLOSE
  - ACCEPTANCE_PENDING
  - ACCEPTED_BREAK
  - RETEST
  - CONTINUATION
  - REENTRY_FAILURE
  - EXPIRED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - INACTIVE
    ACTIVE:
    - APPROACHING_BOUNDARY
    - CROSS_ATTEMPT
    - OUTSIDE_CLOSE
    - ACCEPTANCE_PENDING
    - ACCEPTED_BREAK
    - RETEST
    RESOLVED:
    - CONTINUATION
    INVALIDATED:
    - REENTRY_FAILURE
    - EXPIRED
    - INVALIDATED
  branch_transitions:
  - from_phase: INACTIVE
    to_phase: APPROACHING_BOUNDARY
    event: CAUSAL_APPROACH_LEG
  - from_phase: APPROACHING_BOUNDARY
    to_phase: CROSS_ATTEMPT
    event: OUTWARD_BOUNDARY_CROSS_ATTEMPT
  - from_phase: CROSS_ATTEMPT
    to_phase: OUTSIDE_CLOSE
    event: CLOSED_BAR_OUTSIDE_ZONE
  - from_phase: OUTSIDE_CLOSE
    to_phase: ACCEPTANCE_PENDING
    event: HOLD_POLICY_STARTED
  - from_phase: ACCEPTANCE_PENDING
    to_phase: ACCEPTED_BREAK
    event: CAUSAL_OUTSIDE_HOLD_CONFIRMED
  - from_phase: ACCEPTED_BREAK
    to_phase: RETEST
    event: LINKED_RETEST_TOUCH
  - from_phase: ACCEPTED_BREAK
    to_phase: CONTINUATION
    event: NO_RETEST_CONTINUATION
  - from_phase: RETEST
    to_phase: CONTINUATION
    event: RETEST_HOLD_OR_RECLAIM
  - from_phase: CROSS_ATTEMPT
    to_phase: REENTRY_FAILURE
    event: SHARED_BOUNDARY_REENTRY
  - from_phase: ACCEPTANCE_PENDING
    to_phase: REENTRY_FAILURE
    event: SHARED_BOUNDARY_REENTRY
  - from_phase: CROSS_ATTEMPT
    to_phase: EXPIRED
    event: SCALE_RELATIVE_ACCEPTANCE_EXPIRY
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - OPPOSITE_BOUNDARY_CROSS
    - OUTSIDE_HOLD_FAILURE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Attempt, outside close, acceptance, retest, continuation, and reentry are separated, but
  shared reentry is absent from OUTSIDE_CLOSE, ACCEPTED_BREAK, and RETEST and expiry is only
  reachable from CROSS_ATTEMPT.
- 建议修改：
- Add shared reentry and expiry edges from every applicable active phase and bind all linked failed-
  breakout/retest event IDs.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.BREAKOUT.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: Keep a confirmed breakout lifecycle-active until its source detector's failure window
  closes so no terminal state is later rewritten.
required_changes:
- UNACTIVATED means no qualifying closed-bar outward cross exists for the source range.
- ACTIVE begins on the first qualifying outward cross; BOX_RANGE immediate confirmation and SIDEWAYS_RANGE
  pending or accepted states are evidence substates inside ACTIVE.
- Keep the breakout ACTIVE after algorithmic confirmation while its versioned failed-breakout window remains
  open.
- COMPLETED begins when the failure window closes without a qualifying return into the source range.
- INVALIDATED begins when the canonical return predicate is met inside the failure window; link the terminal
  attempt to a separately completed FAILED_BREAKOUT object.
- A return after COMPLETED is a separately timed LATE_RECLAIM event and must not reactivate or rewrite
  the breakout.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:47:10+08:00'
```

---

<a id="item-44"></a>

## 44. `MG1A.BREAKOUT.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 breakout 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal breakout confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=breakout].confirmation_rules`
- `resolved_contract.families[family=breakout].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- CROSS_ATTEMPT_IS_NOT_ACCEPTED_BREAK
- CLOSED_BAR_OUTSIDE_ZONE_REQUIRED
- CAUSAL_OUTSIDE_HOLD_POLICY_REQUIRED
- RETEST_IS_POST_CONFIRMATION
branch_transitions:
- from_phase: INACTIVE
  to_phase: APPROACHING_BOUNDARY
  event: CAUSAL_APPROACH_LEG
- from_phase: APPROACHING_BOUNDARY
  to_phase: CROSS_ATTEMPT
  event: OUTWARD_BOUNDARY_CROSS_ATTEMPT
- from_phase: CROSS_ATTEMPT
  to_phase: OUTSIDE_CLOSE
  event: CLOSED_BAR_OUTSIDE_ZONE
- from_phase: OUTSIDE_CLOSE
  to_phase: ACCEPTANCE_PENDING
  event: HOLD_POLICY_STARTED
- from_phase: ACCEPTANCE_PENDING
  to_phase: ACCEPTED_BREAK
  event: CAUSAL_OUTSIDE_HOLD_CONFIRMED
- from_phase: ACCEPTED_BREAK
  to_phase: RETEST
  event: LINKED_RETEST_TOUCH
- from_phase: ACCEPTED_BREAK
  to_phase: CONTINUATION
  event: NO_RETEST_CONTINUATION
- from_phase: RETEST
  to_phase: CONTINUATION
  event: RETEST_HOLD_OR_RECLAIM
- from_phase: CROSS_ATTEMPT
  to_phase: REENTRY_FAILURE
  event: SHARED_BOUNDARY_REENTRY
- from_phase: ACCEPTANCE_PENDING
  to_phase: REENTRY_FAILURE
  event: SHARED_BOUNDARY_REENTRY
- from_phase: CROSS_ATTEMPT
  to_phase: EXPIRED
  event: SCALE_RELATIVE_ACCEPTANCE_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Closed outside bar and a separate causal hold are required, but
  CAUSAL_OUTSIDE_HOLD_POLICY_REQUIRED is an undefined token; this is a symbolic policy gap before
  any train-fitted value selection.
- 建议修改：
- Define the outside-hold observation schema, tri-state predicate, scale-policy reference, and
  optional-filter behavior; leave fitted duration/distance values train-only.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.BREAKOUT.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Breakout confirmation remains source-pattern-specific because BOX_RANGE and SIDEWAYS_RANGE
  are two independent shapes with different canonical detectors.
required_changes:
- For BOX_RANGE, confirm a breakout on the first closed bar whose Close is strictly outside the versioned
  buffered boundary selected by box_breakout_detector.
- For SIDEWAYS_RANGE, the first closed-bar outside close starts a pending breakout attempt; confirm only
  when the versioned sideways_detector acceptance count and window are satisfied. Preserve the current
  profile defaults of three outside closes within five bars until a later profile revision is reviewed.
- A wick-only boundary crossing must not confirm either breakout type.
- Preserve source_range_type, source_range_id, first-cross time, confirmation time, detector version,
  parameter profile, and acceptance evidence on every breakout object.
- Do not collapse the two confirmation policies into one common range-breakout predicate.
- Volume and open interest remain observational features and must not gate breakout confirmation.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:21:50+08:00'
```

---

<a id="item-45"></a>

## 45. `MG1A.BREAKOUT.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 breakout 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal breakout invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=breakout].invalidation_rules`
- `resolved_contract.families[family=breakout].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=breakout].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- SHARED_BOUNDARY_REENTRY
- OPPOSITE_BOUNDARY_CROSS
- OUTSIDE_HOLD_FAILURE
- SCALE_RELATIVE_ACCEPTANCE_EXPIRY
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - OPPOSITE_BOUNDARY_CROSS
  - OUTSIDE_HOLD_FAILURE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Shared reentry, opposite cross, hold failure, and expiry are named, but reentry coverage,
  opposite-event identity, expiry coverage, and terminal precedence are incomplete.
- 建议修改：
- Complete invalidation edges for every applicable phase and define shared event identity, expiry,
  and precedence.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.BREAKOUT.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: A failed breakout terminates the breakout attempt but does not terminate or reactivate
  its source range.
required_changes:
- Give every breakout attempt its own stable identity and make a failed attempt terminal.
- Never reactivate or reuse a terminal breakout-attempt identity.
- Preserve an explicit relation from the terminal breakout attempt to its separate FAILED_BREAKOUT object.
- Keep source-range lifecycle evaluation independent from breakout-attempt invalidation.
- For SIDEWAYS_RANGE, allow a breakout that already satisfied its acceptance rule to terminate as FAILED_BREAKOUT
  when the selected detector's reentry predicate is met inside its versioned failure window.
- Once the failure window closes, terminalize the breakout as COMPLETED; any later return is LATE_RECLAIM
  and cannot invalidate or reactivate that terminal object.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:23:36+08:00'
```

---

<a id="item-46"></a>

## 46. `MG1A.FAILED_BREAKOUT.STRUCTURE.001`

**要确认的问题**：是否同意冻结 failed_breakout 的方向依据、锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze failed_breakout side basis, anchors, topology, geometry, duration, price ratios, and prior
context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=failed_breakout].anchors`
- `resolved_contract.families[family=failed_breakout].topology_rules`
- `resolved_contract.families[family=failed_breakout].formula_contracts`
- `resolved_contract.families[family=failed_breakout].instance_fields`
- `resolved_contract.families[family=failed_breakout].direction_contract`
- `resolved_contract.families[family=failed_breakout].object_creation_event`
- `resolved_contract.families[family=failed_breakout].historical_attempt_relationship`
- `resolved_contract.families[family=failed_breakout].historical_source_evidence_available_at_creation_only`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: failed_breakout
object_class: LEVEL_EVENT
structural_orientation: null
orientation_binding: INSTANCE_BOUND
allowed_instance_orientations:
- UPWARD
- DOWNWARD
semantic_roles:
- FAILURE_EVENT
persistent_state: false
scale_spec: INSTANCE_PARAMETER
instance_fields:
- source_breakout_attempt_id
- source_zone_id
- attempt_direction
- reentry_direction
- resolution_direction
direction_contract:
  attempt_direction:
  - UPWARD
  - DOWNWARD
  reentry_direction:
  - UPWARD
  - DOWNWARD
  resolution_direction:
  - UPWARD
  - DOWNWARD
  - NEUTRAL
  - UNRESOLVED
  structural_orientation_basis: ATTEMPT_DIRECTION
  reentry_direction_is_opposite_attempt: true
  directions_must_not_be_collapsed: true
object_creation_event: SHARED_BOUNDARY_REENTRY
historical_attempt_relationship: ORIGINATES_FROM
historical_source_evidence_available_at_creation_only: true
anchors:
- role: reference_level_anchor
  cardinality: ONE_OR_MORE
  state_requirement: CONFIRMED
- role: reference_zone
  cardinality: ONE
  state_requirement: DERIVED_FROM_CONFIRMED
- role: attempt_cross
  cardinality: ONE
  state_requirement: CONFIRMED
- role: max_outside_excursion
  cardinality: ONE
  state_requirement: CONFIRMED_AT_REENTRY
- role: boundary_reentry
  cardinality: ONE
  state_requirement: CONFIRMED
- role: inside_rejection
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
topology_rules:
- SOURCE_BREAKOUT_ATTEMPT_ID_REQUIRED
- REENTRY_CANNOT_BE_BACKFILLED
- MAX_OUTSIDE_EXCURSION_REVISED_UNTIL_REENTRY
- ATTEMPT_AND_RESOLUTION_DIRECTIONS_SEPARATE
formula_contracts:
- name: outside_excursion_to_zone_width
  numerator: maximum_outside_excursion_distance_at_reentry
  denominator: source_zone_width
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: reentry_depth_to_zone_width
  numerator: causal_reentry_depth
  denominator: source_zone_width
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: attempt_to_reentry_duration_ratio
  numerator: attempt_to_reentry_session_bars
  denominator: source_zone_age_session_bars
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
relationship_types:
- ORIGINATES_FROM
- INVALIDATED_BY
- SAME_STRUCTURE_DIFFERENT_SCALE
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：The object is created at shared boundary reentry, requires source attempt and zone identity,
  preserves ORIGINATES_FROM history, separates attempt/reentry/resolution directions, and supplies
  causal normalized geometry forms; this fully discharges the v1 structure question without
  selecting numeric thresholds.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.FAILED_BREAKOUT.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Record each failed breakout as an independent pattern object rather than a reactivated
  state of its source range.
required_changes:
- Create a new stable FAILED_BREAKOUT identity for every failed breakout attempt.
- Preserve source_range_type, source_range_id, breakout_attempt_id, attempt direction, outward-cross evidence,
  and reentry evidence.
- Add explicit typed relations among the FAILED_BREAKOUT object, its source range, and its terminal breakout
  attempt.
- Do not overwrite, alias, or reactivate the source range or breakout-attempt object to represent FAILED_BREAKOUT.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:23:36+08:00'
```

---

<a id="item-47"></a>

## 47. `MG1A.FAILED_BREAKOUT.PHASE.001`

**要确认的问题**：是否同意冻结 failed_breakout 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze failed_breakout phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=failed_breakout].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - INACTIVE
  - REENTRY_OBSERVED
  - INSIDE_REJECTION
  - RESOLUTION_OBSERVED
  - EXPIRED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - INACTIVE
    ACTIVE:
    - REENTRY_OBSERVED
    - INSIDE_REJECTION
    RESOLVED:
    - RESOLUTION_OBSERVED
    INVALIDATED:
    - EXPIRED
    - INVALIDATED
  branch_transitions:
  - from_phase: INACTIVE
    to_phase: REENTRY_OBSERVED
    event: SHARED_BOUNDARY_REENTRY
  - from_phase: REENTRY_OBSERVED
    to_phase: INSIDE_REJECTION
    event: CAUSAL_INSIDE_REJECTION
  - from_phase: REENTRY_OBSERVED
    to_phase: RESOLUTION_OBSERVED
    event: REENTRY_ONLY_RESOLUTION_POLICY
  - from_phase: INSIDE_REJECTION
    to_phase: RESOLUTION_OBSERVED
    event: RESOLUTION_DIRECTION_AVAILABLE
  - from_phase: REENTRY_OBSERVED
    to_phase: EXPIRED
    event: SCALE_RELATIVE_REENTRY_EXPIRY
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - RENEWED_ACCEPTED_BREAK_LINK
    - SOURCE_ZONE_REDEFINITION
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The object starts at shared reentry without backfill, but REENTRY_ONLY_RESOLUTION_POLICY and
  the inside-rejection branch coexist without selecting a contract or instance policy.
- 建议修改：
- Add an explicit resolution-policy field/registry and bind each legal branch and expiry edge to the
  selected policy.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.FAILED_BREAKOUT.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: FAILED_BREAKOUT is an event-derived result object created only when the canonical closed-bar
  reentry predicate is observed inside the failure window.
required_changes:
- UNACTIVATED means no qualifying failed-breakout reentry has been observed for the related attempt.
- Instantiate the FAILED_BREAKOUT directly as COMPLETED at the qualifying closed-bar reentry timestamp;
  do not persist an artificial ACTIVE candidate state.
- Preserve the related breakout attempt's separate transition to INVALIDATED at the same closed-bar timestamp.
- Do not use a FAILED_BREAKOUT object to reactivate or overwrite its source range.
- Keep later target, reclaim, and ambiguity observations as outcomes or events without rewriting the completed
  lifecycle.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:47:10+08:00'
```

---

<a id="item-48"></a>

## 48. `MG1A.FAILED_BREAKOUT.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 failed_breakout 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal failed_breakout confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=failed_breakout].confirmation_rules`
- `resolved_contract.families[family=failed_breakout].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- REENTRY_ESTABLISHES_FAILURE_TO_ACCEPT
- OPTIONAL_INSIDE_REJECTION_IS_SEPARATE
- RESOLUTION_DIRECTION_REQUIRES_CAUSAL_EVIDENCE
branch_transitions:
- from_phase: INACTIVE
  to_phase: REENTRY_OBSERVED
  event: SHARED_BOUNDARY_REENTRY
- from_phase: REENTRY_OBSERVED
  to_phase: INSIDE_REJECTION
  event: CAUSAL_INSIDE_REJECTION
- from_phase: REENTRY_OBSERVED
  to_phase: RESOLUTION_OBSERVED
  event: REENTRY_ONLY_RESOLUTION_POLICY
- from_phase: INSIDE_REJECTION
  to_phase: RESOLUTION_OBSERVED
  event: RESOLUTION_DIRECTION_AVAILABLE
- from_phase: REENTRY_OBSERVED
  to_phase: EXPIRED
  event: SCALE_RELATIVE_REENTRY_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Reentry and inside rejection are separate, but the contract does not choose reentry-only
  versus reentry-plus-rejection confirmation or define UNKNOWN behavior.
- 建议修改：
- Select the confirmation policy explicitly and encode its tri-state truth table and optional-filter
  behavior.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.FAILED_BREAKOUT.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Use each source range detector's versioned failed-breakout rule, including the current
  SIDEWAYS_RANGE rule that may reclassify an already accepted breakout inside the failure window.
required_changes:
- For BOX_RANGE, confirm FAILED_BREAKOUT when Close returns to or through the original box boundary inside
  the selected box_breakout_detector failure window after its immediately confirmed breakout. Preserve
  the current default window of three bars until a later profile revision is reviewed.
- For SIDEWAYS_RANGE, confirm FAILED_BREAKOUT when Close returns to or through the original sideways core
  boundary inside the selected sideways_detector failure window after the first buffered outward cross.
  Apply this rule to both pending and already accepted breakouts, preserving the current default window
  of three bars.
- A retreat behind only the buffered break level is insufficient; confirmation requires a closed-bar return
  to the original source-range boundary.
- Classify a return after the failure window as late reclaim rather than FAILED_BREAKOUT.
- A wick-only return must not confirm FAILED_BREAKOUT.
- Preserve ambiguous target-touch and boundary-reentry evidence from the same bar without inventing an
  intrabar order.
- Version failure-window and reentry parameters independently for BOX_RANGE and SIDEWAYS_RANGE profiles.
- Volume and open interest remain observational features and must not gate FAILED_BREAKOUT confirmation.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:25:54+08:00'
```

---

<a id="item-49"></a>

## 49. `MG1A.FAILED_BREAKOUT.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 failed_breakout 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal failed_breakout invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=failed_breakout].invalidation_rules`
- `resolved_contract.families[family=failed_breakout].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=failed_breakout].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- RENEWED_ACCEPTED_BREAK_LINK
- SOURCE_ZONE_REDEFINITION
- SCALE_RELATIVE_EXPIRY
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - RENEWED_ACCEPTED_BREAK_LINK
  - SOURCE_ZONE_REDEFINITION
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Renewed accepted break, zone redefinition, and expiry are named, but renewed-event lineage,
  zone precedence, resolution meaning, and later-phase expiry are incomplete.
- 建议修改：
- Define renewed-break identity/lineage, source-zone redefinition precedence, reversal meaning,
  expiry coverage, and event priority.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.FAILED_BREAKOUT.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: A confirmed FAILED_BREAKOUT is immutable and has no later invalidation or expiry transition.
required_changes:
- If an outward attempt survives its failure window, do not create and then invalidate a FAILED_BREAKOUT
  object; no such object exists.
- Once created at canonical reentry, keep FAILED_BREAKOUT COMPLETED and preserve later evidence under
  separate timestamps.
- Corrections caused by data or algorithm-version changes must use provenance and replacement records
  rather than a market lifecycle invalidation event.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:47:10+08:00'
```

---

<a id="item-50"></a>

## 50. `MG1A.RETEST.STRUCTURE.001`

**要确认的问题**：是否同意冻结 retest 的方向依据、锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze retest side basis, anchors, topology, geometry, duration, price ratios, and prior context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=retest].anchors`
- `resolved_contract.families[family=retest].topology_rules`
- `resolved_contract.families[family=retest].formula_contracts`
- `resolved_contract.families[family=retest].instance_fields`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: retest
object_class: LEVEL_EVENT
structural_orientation: null
orientation_binding: INSTANCE_BOUND
allowed_instance_orientations:
- UPWARD
- DOWNWARD
semantic_roles:
- RETEST_EVENT
persistent_state: false
scale_spec: INSTANCE_PARAMETER
instance_fields:
- source_event_id
- source_pattern_id
- source_zone_id
- origin_break_direction
- retest_ordinal
anchors:
- role: source_event_ref
  cardinality: ONE
  state_requirement: CONFIRMED
- role: source_zone_ref
  cardinality: ONE
  state_requirement: CONFIRMED
- role: origin_break
  cardinality: ONE
  state_requirement: CONFIRMED
- role: post_break_extreme
  cardinality: ONE
  state_requirement: REVISED_UNTIL_RETURN
- role: return_touch
  cardinality: ONE
  state_requirement: CONFIRMED
- role: hold_or_rejection
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: reclaim
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: continuation_anchor
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
topology_rules:
- SOURCE_EVENT_PATTERN_AND_ZONE_IDS_REQUIRED
- ORIENTATION_INHERITS_ORIGIN_BREAK_DIRECTION
- RETEST_ORDINAL_REQUIRED
- HOLD_RECLAIM_AND_FULL_RECROSS_DISTINCT
formula_contracts:
- name: penetration_to_zone_width
  numerator: maximum_return_penetration_distance
  denominator: source_zone_width
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: return_duration_to_break_leg_duration
  numerator: break_to_touch_session_bars
  denominator: source_break_leg_session_bars
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
- name: continuation_to_break_displacement
  numerator: continuation_displacement
  denominator: source_break_displacement
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
relationship_types:
- ORIGINATES_FROM
- RETESTS
- CONFIRMED_BY
- INVALIDATED_BY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：V2 resolves retest as an independent LEVEL_EVENT and requires source event, pattern, zone,
  direction, and ordinal identity, but no legal source-family/source-event compatibility registry
  exists.
- 建议修改：
- Add a closed legal source-family/event/zone compatibility registry and validate ORIGINATES_FROM
  and RETESTS endpoints against it.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.RETEST.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Remove RETEST as an independent pattern family because no retained canonical detector
  emits a generic retest object and the only partial implementation was explicitly excluded.
required_changes:
- Remove RETEST from official pattern-family enums, object schemas, detector outputs, and training or
  evaluation label spaces.
- Do not create a retest identity, anchor topology, geometry record, or relation object.
- Keep source-detector boundary tests as diagnostic events only; they must not emit an official RETEST
  label.
- Preserve FAILED_BREAKOUT and LATE_RECLAIM as their already reviewed independent result or event semantics
  and never alias either one to RETEST.
- Do not use the ignored m_top_neckline_projection.py as retest evidence or an auxiliary label source.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:52:45+08:00'
```

---

<a id="item-51"></a>

## 51. `MG1A.RETEST.PHASE.001`

**要确认的问题**：是否同意冻结 retest 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze retest phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=retest].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - INACTIVE
  - SOURCE_EVENT_LINKED
  - RETURNING
  - TOUCH_OBSERVED
  - HOLD_CONFIRMED
  - PENETRATE_RECLAIMED
  - FULL_RECROSS_FAILURE
  - CONTINUATION
  - EXPIRED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - INACTIVE
    ACTIVE:
    - SOURCE_EVENT_LINKED
    - RETURNING
    - TOUCH_OBSERVED
    - HOLD_CONFIRMED
    - PENETRATE_RECLAIMED
    RESOLVED:
    - CONTINUATION
    INVALIDATED:
    - FULL_RECROSS_FAILURE
    - EXPIRED
    - INVALIDATED
  branch_transitions:
  - from_phase: INACTIVE
    to_phase: SOURCE_EVENT_LINKED
    event: SOURCE_BREAK_AVAILABLE
  - from_phase: SOURCE_EVENT_LINKED
    to_phase: RETURNING
    event: RETURN_LEG_OBSERVED
  - from_phase: RETURNING
    to_phase: TOUCH_OBSERVED
    event: SOURCE_ZONE_TOUCH
  - from_phase: TOUCH_OBSERVED
    to_phase: HOLD_CONFIRMED
    event: NEW_SIDE_HOLD_OR_REJECTION
  - from_phase: TOUCH_OBSERVED
    to_phase: PENETRATE_RECLAIMED
    event: PENETRATION_THEN_CAUSAL_RECLAIM
  - from_phase: TOUCH_OBSERVED
    to_phase: FULL_RECROSS_FAILURE
    event: FULL_RECROSS_TO_ORIGINAL_SIDE
  - from_phase: HOLD_CONFIRMED
    to_phase: CONTINUATION
    event: CONTINUATION_ANCHOR
  - from_phase: PENETRATE_RECLAIMED
    to_phase: CONTINUATION
    event: CONTINUATION_ANCHOR
  - from_phase: SOURCE_EVENT_LINKED
    to_phase: EXPIRED
    event: SCALE_RELATIVE_RETURN_EXPIRY
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - SOURCE_BREAK_INVALIDATED
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Return, touch, hold, penetrate-reclaim, full recross, and continuation are distinct, but
  expiry is reachable only from SOURCE_EVENT_LINKED rather than later return/touch/hold states.
- 建议修改：
- Add scale-relative expiry transitions from every applicable active state and bind them to defined
  clock semantics.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.RETEST.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: An eliminated RETEST family has no independent four-state lifecycle or phase contract.
required_changes:
- Remove every RETEST lifecycle, phase, timeout, and terminal-state definition.
- Do not infer a retest lifecycle from breakout boundary-test events.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:52:45+08:00'
```

---

<a id="item-52"></a>

## 52. `MG1A.RETEST.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 retest 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal retest confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=retest].confirmation_rules`
- `resolved_contract.families[family=retest].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- SOURCE_BREAK_AND_TOUCH_ARE_PREREQUISITES
- NEW_SIDE_HOLD_OR_RECLAIM_CONFIRMS
- CONTINUATION_IS_LATER_RESOLUTION
branch_transitions:
- from_phase: INACTIVE
  to_phase: SOURCE_EVENT_LINKED
  event: SOURCE_BREAK_AVAILABLE
- from_phase: SOURCE_EVENT_LINKED
  to_phase: RETURNING
  event: RETURN_LEG_OBSERVED
- from_phase: RETURNING
  to_phase: TOUCH_OBSERVED
  event: SOURCE_ZONE_TOUCH
- from_phase: TOUCH_OBSERVED
  to_phase: HOLD_CONFIRMED
  event: NEW_SIDE_HOLD_OR_REJECTION
- from_phase: TOUCH_OBSERVED
  to_phase: PENETRATE_RECLAIMED
  event: PENETRATION_THEN_CAUSAL_RECLAIM
- from_phase: TOUCH_OBSERVED
  to_phase: FULL_RECROSS_FAILURE
  event: FULL_RECROSS_TO_ORIGINAL_SIDE
- from_phase: HOLD_CONFIRMED
  to_phase: CONTINUATION
  event: CONTINUATION_ANCHOR
- from_phase: PENETRATE_RECLAIMED
  to_phase: CONTINUATION
  event: CONTINUATION_ANCHOR
- from_phase: SOURCE_EVENT_LINKED
  to_phase: EXPIRED
  event: SCALE_RELATIVE_RETURN_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Source break and touch are prerequisites and continuation is later, but hold/rejection and
  penetrate-reclaim predicates plus optional-filter UNKNOWN handling are undefined.
- 建议修改：
- Define causal hold, rejection, penetration, and reclaim boundaries with a tri-state confirmation
  truth table.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.RETEST.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: RETEST has no confirmation predicate after removal as an independent family.
required_changes:
- Remove retest confirmation fields, parameters, events, and labels.
- Do not treat an approach, boundary touch, rejection, hold, or later continuation as RETEST confirmation.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:52:45+08:00'
```

---

<a id="item-53"></a>

## 53. `MG1A.RETEST.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 retest 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal retest invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=retest].invalidation_rules`
- `resolved_contract.families[family=retest].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=retest].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- FULL_RECROSS_TO_ORIGINAL_SIDE
- SOURCE_BREAK_INVALIDATED
- SCALE_RELATIVE_TOUCH_OR_HOLD_EXPIRY
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - SOURCE_BREAK_INVALIDATED
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Full recross and source invalidation are separated, but graph coverage disagrees with
  touch/hold expiry rules, source-zone version changes and simultaneous-event precedence are absent.
- 建议修改：
- Complete recross/expiry edges, define source-zone revision behavior, and add deterministic
  invalidation precedence.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.RETEST.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: RETEST has no invalidation or expiry predicate after removal as an independent family.
required_changes:
- Remove retest invalidation, expiry, and terminal-reason fields and parameters.
- Continue to classify returns into the source range only under the reviewed FAILED_BREAKOUT or LATE_RECLAIM
  rules.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:52:45+08:00'
```

---

<a id="item-58"></a>

## 58. `MG1A.SUPPORT_RESISTANCE_CONVERSION.STRUCTURE.001`

**要确认的问题**：是否同意冻结 support_resistance_conversion 的方向依据、锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze support_resistance_conversion side basis, anchors, topology, geometry, duration, price
ratios, and prior context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=support_resistance_conversion].anchors`
- `resolved_contract.families[family=support_resistance_conversion].topology_rules`
- `resolved_contract.families[family=support_resistance_conversion].formula_contracts`
- `resolved_contract.families[family=support_resistance_conversion].instance_fields`
- `resolved_contract.families[family=support_resistance_conversion].object_class`
- `resolved_contract.families[family=support_resistance_conversion].role_conversion_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: support_resistance_conversion
object_class: STRUCTURAL_RELATION
structural_orientation: null
orientation_binding: INSTANCE_BOUND
allowed_instance_orientations:
- UPWARD
- DOWNWARD
semantic_roles:
- ROLE_CONVERSION
persistent_state: false
scale_spec: INSTANCE_PARAMETER
instance_fields:
- source_breakout_id
- source_retest_id
- source_zone_id
- original_role
- converted_role
role_conversion_contract:
  upward_break: RESISTANCE_TO_SUPPORT
  downward_break: SUPPORT_TO_RESISTANCE
  source_breakout_and_retest_required: true
  independent_morphology_label: false
anchors:
- role: original_level_anchor
  cardinality: ONE_OR_MORE
  state_requirement: CONFIRMED
- role: original_level_zone
  cardinality: ONE
  state_requirement: DERIVED_FROM_CONFIRMED
- role: role_break
  cardinality: ONE
  state_requirement: CONFIRMED
- role: opposite_side_return
  cardinality: ONE
  state_requirement: CONFIRMED
- role: converted_role_hold_or_rejection
  cardinality: ONE
  state_requirement: CONFIRMED
topology_rules:
- SOURCE_BREAKOUT_RETEST_AND_ZONE_IDS_REQUIRED
- ORIGINAL_AND_CONVERTED_ROLES_EXPLICIT
- ROLE_CONVERSION_IS_RELATION_NOT_FUTURE_EFFECT
formula_contracts:
- name: break_displacement_to_zone_width
  numerator: source_break_displacement
  denominator: source_zone_width
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: return_depth_to_break_leg
  numerator: opposite_side_return_depth
  denominator: source_break_displacement
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: penetration_to_zone_width
  numerator: converted_role_test_penetration
  denominator: source_zone_width
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
relationship_types:
- ORIGINATES_FROM
- RETESTS
- CONVERTS_ROLE_OF
- INVALIDATED_BY
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：The object-class choice, source breakout/retest/zone identity, exact upward and downward role
  transforms, causal anchors, and normalized geometry forms fully discharge the v1 structure and
  identity question at symbolic scope.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.SUPPORT_RESISTANCE_CONVERSION.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Remove SUPPORT_RESISTANCE_CONVERSION as an independent pattern family because no retained
  canonical detector emits a role-conversion object.
required_changes:
- Remove SUPPORT_RESISTANCE_CONVERSION from official pattern-family enums, object schemas, detector outputs,
  and training or evaluation label spaces.
- Do not create a conversion identity, anchor topology, geometry record, or lifecycle.
- Preserve support and resistance zones only as contextual level evidence for retained canonical families.
- Do not infer role conversion from breakout, boundary test, FAILED_BREAKOUT, or LATE_RECLAIM events.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:27:48+08:00'
```

---

<a id="item-59"></a>

## 59. `MG1A.SUPPORT_RESISTANCE_CONVERSION.PHASE.001`

**要确认的问题**：是否同意冻结 support_resistance_conversion 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze support_resistance_conversion phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=support_resistance_conversion].phase_contract`
- `resolved_contract.families[family=support_resistance_conversion].role_conversion_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - INACTIVE
  - SOURCE_BREAK_LINKED
  - RETURN_OBSERVED
  - CONVERTED_ROLE_TEST
  - CONVERSION_CONFIRMED
  - ROLE_RESTORED
  - EXPIRED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - INACTIVE
    ACTIVE:
    - SOURCE_BREAK_LINKED
    - RETURN_OBSERVED
    - CONVERTED_ROLE_TEST
    RESOLVED:
    - CONVERSION_CONFIRMED
    INVALIDATED:
    - ROLE_RESTORED
    - EXPIRED
    - INVALIDATED
  branch_transitions:
  - from_phase: INACTIVE
    to_phase: SOURCE_BREAK_LINKED
    event: ACCEPTED_BREAKOUT_AVAILABLE
  - from_phase: SOURCE_BREAK_LINKED
    to_phase: RETURN_OBSERVED
    event: OPPOSITE_SIDE_RETURN
  - from_phase: RETURN_OBSERVED
    to_phase: CONVERTED_ROLE_TEST
    event: CONVERTED_ZONE_INTERACTION
  - from_phase: CONVERTED_ROLE_TEST
    to_phase: CONVERSION_CONFIRMED
    event: CONVERTED_ROLE_HOLD_OR_REJECTION
  - from_phase: CONVERTED_ROLE_TEST
    to_phase: ROLE_RESTORED
    event: FULL_RECROSS_AND_ORIGINAL_ROLE_RESTORATION
  - from_phase: SOURCE_BREAK_LINKED
    to_phase: EXPIRED
    event: SCALE_RELATIVE_RETURN_EXPIRY
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - SOURCE_BREAK_INVALIDATED
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：Accepted breakout creates the relation candidate, return and converted-zone test are separate
  causal stages, hold/rejection confirms, and role restoration, expiry, and source invalidation are
  explicit terminals.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.SUPPORT_RESISTANCE_CONVERSION.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: An eliminated SUPPORT_RESISTANCE_CONVERSION family has no independent four-state lifecycle
  or phase contract.
required_changes:
- Remove every role-conversion lifecycle, phase, timeout, and terminal-state definition.
- Do not derive a conversion phase from ordinary support, resistance, or breakout context.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:27:48+08:00'
```

---

<a id="item-60"></a>

## 60. `MG1A.SUPPORT_RESISTANCE_CONVERSION.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 support_resistance_conversion 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal support_resistance_conversion confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=support_resistance_conversion].confirmation_rules`
- `resolved_contract.families[family=support_resistance_conversion].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- SOURCE_BREAK_STARTS_RELATION_CANDIDATE
- RETURN_AND_CONVERTED_ROLE_TEST_REQUIRED
- CONVERTED_ROLE_HOLD_OR_REJECTION_CONFIRMS
branch_transitions:
- from_phase: INACTIVE
  to_phase: SOURCE_BREAK_LINKED
  event: ACCEPTED_BREAKOUT_AVAILABLE
- from_phase: SOURCE_BREAK_LINKED
  to_phase: RETURN_OBSERVED
  event: OPPOSITE_SIDE_RETURN
- from_phase: RETURN_OBSERVED
  to_phase: CONVERTED_ROLE_TEST
  event: CONVERTED_ZONE_INTERACTION
- from_phase: CONVERTED_ROLE_TEST
  to_phase: CONVERSION_CONFIRMED
  event: CONVERTED_ROLE_HOLD_OR_REJECTION
- from_phase: CONVERTED_ROLE_TEST
  to_phase: ROLE_RESTORED
  event: FULL_RECROSS_AND_ORIGINAL_ROLE_RESTORATION
- from_phase: SOURCE_BREAK_LINKED
  to_phase: EXPIRED
  event: SCALE_RELATIVE_RETURN_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Break, return, test, and converted-role hold/rejection are ordered, but penetration, hold,
  rejection, and optional-filter UNKNOWN truth rules are not defined.
- 建议修改：
- Define zone-version-specific penetration/hold/rejection predicates and a tri-state confirmation
  truth table.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.SUPPORT_RESISTANCE_CONVERSION.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: SUPPORT_RESISTANCE_CONVERSION has no confirmation predicate after removal as an independent
  family.
required_changes:
- Remove role-conversion confirmation fields, parameters, events, and labels.
- Do not treat a break, opposite-side return, hold, or rejection as role-conversion confirmation.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:27:48+08:00'
```

---

<a id="item-61"></a>

## 61. `MG1A.SUPPORT_RESISTANCE_CONVERSION.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 support_resistance_conversion 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal support_resistance_conversion invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=support_resistance_conversion].invalidation_rules`
- `resolved_contract.families[family=support_resistance_conversion].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=support_resistance_conversion].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- SOURCE_BREAK_INVALIDATED
- FULL_RECROSS
- ORIGINAL_ROLE_RESTORED
- SCALE_RELATIVE_EXPIRY
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - SOURCE_BREAK_INVALIDATED
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Source failure, full recross, original-role restoration, and expiry are listed, but
  penetration-versus-recross boundaries, restoration identity, later-phase expiry, and precedence
  are incomplete.
- 建议修改：
- Define penetration/full-recross boundaries, role-restoration event identity, expiry coverage, and
  deterministic precedence.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.SUPPORT_RESISTANCE_CONVERSION.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: SUPPORT_RESISTANCE_CONVERSION has no invalidation or expiry predicate after removal as
  an independent family.
required_changes:
- Remove role-conversion invalidation, expiry, and terminal-reason fields and parameters.
- Keep later level behavior only as separately timed context or events owned by retained canonical families.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:27:48+08:00'
```

---
