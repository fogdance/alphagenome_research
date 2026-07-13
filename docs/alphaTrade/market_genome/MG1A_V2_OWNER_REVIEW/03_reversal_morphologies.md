# 反转形态

[返回审核总览](README.md)

本章共 16 条。AI 内容仅作参考，最终决定由 `PROJECT_OWNER` 作出。

<a id="item-26"></a>

## 26. `MG1A.DOUBLE_TOP.STRUCTURE.001`

**要确认的问题**：是否同意冻结 double_top 的锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze double_top anchors, topology, geometry, duration, price ratios, and prior context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=double_top].anchors`
- `resolved_contract.families[family=double_top].topology_rules`
- `resolved_contract.families[family=double_top].formula_contracts`
- `resolved_contract.families[family=double_top].instance_fields`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: double_top
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
persistent_state: false
scale_spec: INSTANCE_PARAMETER
anchors:
- role: left_peak
  cardinality: ONE
  state_requirement: CONFIRMED
- role: intervening_trough
  cardinality: ONE
  state_requirement: CONFIRMED
- role: right_peak
  cardinality: ONE
  state_requirement: CONFIRMED
- role: neckline_zone
  cardinality: DERIVED
  state_requirement: DERIVED_FROM_CONFIRMED
- role: neckline_break
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: neckline_retest
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: continuation_anchor
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
topology_rules:
- TWO_CONFIRMED_UPPER_EXTREMA_WITH_INTERVENING_LOWER_EXTREMUM
- SHAPE_IDENTITY_INDEPENDENT_OF_FUTURE_EFFECT
- NECKLINE_BREAK_IS_LINKED_EVENT
formula_contracts:
- name: peak_height_difference_to_formation_amplitude
  numerator: absolute_left_peak_minus_right_peak
  denominator: compatible_peak_reference_minus_intervening_trough
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: time_symmetry_ratio
  numerator: absolute_left_peak_to_trough_duration_minus_trough_to_right_peak_duration
  denominator: total_formation_session_bars
  sign: UNIT_INTERVAL_WHEN_DEFINED
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
- name: trough_depth_to_formation_amplitude
  numerator: compatible_peak_reference_minus_intervening_trough
  denominator: causal_price_scale
  sign: NONNEGATIVE
  unit: NORMALIZED_PRICE_MULTIPLE
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
relationship_types:
- NESTED_IN
- CONFIRMED_BY
- RETESTS
- SAME_STRUCTURE_DIFFERENT_SCALE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The three anchors, horizontal neckline, normalized formula forms, and linked break/retest
  identity are explicit, but peak compatibility and geometry/duration acceptance semantics remain
  undefined.
- 建议修改：
- Define peak compatibility, reference selection, and a versioned geometry/duration policy schema;
  keep fitted values train-only.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.DOUBLE_TOP.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Use only the strictly mirrored M/W detector as the formal DOUBLE_TOP and DOUBLE_BOTTOM
  definition source.
required_changes:
- Register tools/patterns/m_w_pattern_detector.py analyze_m_w_patterns as the only DOUBLE_TOP detector,
  pinned to stock_peter commit 57ee01baf8a6859d89e4c2e8ce83f41cc29c05b5 and file SHA256 f5c2e40baf3d3aebb3545ed5a646daeac633d8e0c5710c8dc3f76e745972190a.
- Completely ignore m_top_detector.py and m_top_neckline_projection.py for Market Genome labels, evidence,
  derived features, targets, and confirmation.
- Use the canonical detector's causal H-L-H structure, extreme similarity, minimum retracement, extreme
  separation, and leg-duration predicates under a versioned timeframe-specific MWPatternConfig profile.
- Share the same detector, state machine, and parameter profile with DOUBLE_BOTTOM by using pattern_scope
  both.
- Preserve the current geometry-only default require_pretrend false; record a canonical prior uptrend
  as context rather than a DOUBLE_TOP existence gate.
- Replace scalar neckline and outer-boundary fields with the owner-approved key-zone representation before
  formal use.
- Volume and open interest remain observational and must not gate DOUBLE_TOP structure.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:30:35+08:00'
```

---

<a id="item-27"></a>

## 27. `MG1A.DOUBLE_TOP.PHASE.001`

**要确认的问题**：是否同意冻结 double_top 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze double_top phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=double_top].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - NO_PATTERN
  - LEFT_PEAK_AVAILABLE
  - NECKLINE_PIVOT_AVAILABLE
  - RIGHT_PEAK_CANDIDATE
  - RIGHT_PEAK_CONFIRMED
  - SHAPE_COMPLETE
  - NECKLINE_BREAK_CONFIRMED
  - RETEST
  - CONTINUATION
  - EXPIRED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - NO_PATTERN
    ACTIVE:
    - LEFT_PEAK_AVAILABLE
    - NECKLINE_PIVOT_AVAILABLE
    - RIGHT_PEAK_CANDIDATE
    - RIGHT_PEAK_CONFIRMED
    - SHAPE_COMPLETE
    - NECKLINE_BREAK_CONFIRMED
    - RETEST
    RESOLVED:
    - CONTINUATION
    INVALIDATED:
    - EXPIRED
    - INVALIDATED
  branch_transitions:
  - from_phase: NO_PATTERN
    to_phase: LEFT_PEAK_AVAILABLE
    event: CONFIRMED_LEFT_PEAK
  - from_phase: LEFT_PEAK_AVAILABLE
    to_phase: NECKLINE_PIVOT_AVAILABLE
    event: CONFIRMED_INTERVENING_TROUGH
  - from_phase: NECKLINE_PIVOT_AVAILABLE
    to_phase: RIGHT_PEAK_CANDIDATE
    event: PROVISIONAL_RIGHT_PEAK
  - from_phase: RIGHT_PEAK_CANDIDATE
    to_phase: RIGHT_PEAK_CONFIRMED
    event: CONFIRMED_RIGHT_PEAK
  - from_phase: RIGHT_PEAK_CONFIRMED
    to_phase: SHAPE_COMPLETE
    event: GEOMETRY_POLICY_SATISFIED
  - from_phase: SHAPE_COMPLETE
    to_phase: NECKLINE_BREAK_CONFIRMED
    event: CAUSAL_DOWNWARD_NECKLINE_CROSS
  - from_phase: NECKLINE_BREAK_CONFIRMED
    to_phase: RETEST
    event: LINKED_NECKLINE_RETEST
  - from_phase: NECKLINE_BREAK_CONFIRMED
    to_phase: CONTINUATION
    event: NO_RETEST_CONTINUATION
  - from_phase: RETEST
    to_phase: CONTINUATION
    event: RETEST_HOLD_CONFIRMED
  - from_phase: SHAPE_COMPLETE
    to_phase: EXPIRED
    event: SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - PEAK_ZONE_UPSIDE_BREACH
    - ANCHOR_ORDER_FAILURE
    - POST_BREAK_NECKLINE_RECROSS
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：The graph causally enumerates both peaks and neckline pivot, shape completion, downward
  neckline break, separate retest/no-retest continuation, expiry, and invalidation.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.DOUBLE_TOP.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: Map the canonical detector's causal candidate state into the common four-state lifecycle
  without inventing intermediate phases that the algorithm does not emit.
required_changes:
- UNACTIVATED means no causally available H-L-H candidate exists; partial anchors may be retained only
  as evidence.
- ACTIVE begins when the right peak is confirmed and the complete H-L-H candidate passes the canonical
  geometry rules.
- COMPLETED begins on the closed-bar neckline-break confirmation event.
- INVALIDATED begins when an active candidate is invalidated, expires, or is superseded; a replacement
  candidate receives a new stable identity.
- Keep post-confirmation failed-breakdown, target-hit, full-failure, and ambiguous results as separately
  timed outcomes; they must not rewrite the completed pattern lifecycle or confirmation time.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:30:35+08:00'
```

---

<a id="item-28"></a>

## 28. `MG1A.DOUBLE_TOP.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 double_top 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal double_top confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=double_top].confirmation_rules`
- `resolved_contract.families[family=double_top].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- SHAPE_COMPLETION_IS_NOT_BEARISH_EFFECT
- CAUSAL_DOWNWARD_NECKLINE_CROSS_CONFIRMS_CLASSIC_REVERSAL_ROLE
- PRECEDING_UPTREND_REQUIRED_FOR_CLASSIC_REVERSAL_ROLE
branch_transitions:
- from_phase: NO_PATTERN
  to_phase: LEFT_PEAK_AVAILABLE
  event: CONFIRMED_LEFT_PEAK
- from_phase: LEFT_PEAK_AVAILABLE
  to_phase: NECKLINE_PIVOT_AVAILABLE
  event: CONFIRMED_INTERVENING_TROUGH
- from_phase: NECKLINE_PIVOT_AVAILABLE
  to_phase: RIGHT_PEAK_CANDIDATE
  event: PROVISIONAL_RIGHT_PEAK
- from_phase: RIGHT_PEAK_CANDIDATE
  to_phase: RIGHT_PEAK_CONFIRMED
  event: CONFIRMED_RIGHT_PEAK
- from_phase: RIGHT_PEAK_CONFIRMED
  to_phase: SHAPE_COMPLETE
  event: GEOMETRY_POLICY_SATISFIED
- from_phase: SHAPE_COMPLETE
  to_phase: NECKLINE_BREAK_CONFIRMED
  event: CAUSAL_DOWNWARD_NECKLINE_CROSS
- from_phase: NECKLINE_BREAK_CONFIRMED
  to_phase: RETEST
  event: LINKED_NECKLINE_RETEST
- from_phase: NECKLINE_BREAK_CONFIRMED
  to_phase: CONTINUATION
  event: NO_RETEST_CONTINUATION
- from_phase: RETEST
  to_phase: CONTINUATION
  event: RETEST_HOLD_CONFIRMED
- from_phase: SHAPE_COMPLETE
  to_phase: EXPIRED
  event: SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：A causal downward neckline cross correctly confirms classic reversal role and retest is post-
  confirmation, but the required optional-filter UNKNOWN handling and full truth combination are
  absent.
- 建议修改：
- Bind the exact neckline version and cross predicate, then add tri-state optional-filter behavior
  to the confirmation rule.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.DOUBLE_TOP.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Preserve the canonical M/W detector's strict closed-bar neckline break while applying
  its buffer from the owner-approved neckline-zone outer edge.
required_changes:
- After the right peak is causally confirmed, confirm DOUBLE_TOP only when Close is strictly below neckline_zone_low
  multiplied by one minus the versioned break_buffer_pct.
- Preserve the current canonical break_buffer_pct default of 0.003 and max_breakout_wait_bars default
  of 20 until their timeframe-specific profile is versioned or reviewed.
- Equality with the threshold and wick-only penetration must not confirm DOUBLE_TOP.
- Preserve require_pretrend false in the current profile; canonical prior uptrend is contextual evidence
  rather than a confirmation gate.
- Volume and open interest remain observational and must not gate confirmation.
- Record the closed-bar confirmation time separately from anchor event and availability times.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:32:01+08:00'
```

---

<a id="item-29"></a>

## 29. `MG1A.DOUBLE_TOP.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 double_top 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal double_top invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=double_top].invalidation_rules`
- `resolved_contract.families[family=double_top].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=double_top].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- PEAK_ZONE_UPSIDE_BREACH
- ANCHOR_ORDER_FAILURE
- SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
- POST_BREAK_NECKLINE_RECROSS
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - PEAK_ZONE_UPSIDE_BREACH
  - ANCHOR_ORDER_FAILURE
  - POST_BREAK_NECKLINE_RECROSS
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Peak breach, anchor-order failure, break-absence expiry, and post-break recross are distinct
  labels, but observation boundaries, expiry clock, and simultaneous-event precedence are not
  frozen.
- 建议修改：
- Define normalized breach/order observations, expiry timing, recross boundary, and deterministic
  event precedence.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.DOUBLE_TOP.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Apply the canonical pre-confirmation invalidation buffer outside the complete top zone
  and preserve detector expiry and supersession as terminal candidate events.
required_changes:
- Before confirmation, invalidate DOUBLE_TOP only when Close is strictly above top_zone_high multiplied
  by one plus the versioned invalidation_buffer_pct.
- Preserve the current invalidation_buffer_pct default of 0.005; equality and wick-only penetration must
  not invalidate.
- Expire an unconfirmed candidate when its canonical maximum wait is exceeded, and invalidate it as superseded
  when a new same-side structural extreme replaces it; every replacement receives a new stable identity.
- Keep post-confirmation failed breakdown, full failure, target hit, and same-bar ambiguity as separately
  timed outcomes rather than rewriting the completed DOUBLE_TOP lifecycle.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:32:01+08:00'
```

---

<a id="item-30"></a>

## 30. `MG1A.DOUBLE_BOTTOM.STRUCTURE.001`

**要确认的问题**：是否同意冻结 double_bottom 的锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze double_bottom anchors, topology, geometry, duration, price ratios, and prior context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=double_bottom].anchors`
- `resolved_contract.families[family=double_bottom].topology_rules`
- `resolved_contract.families[family=double_bottom].formula_contracts`
- `resolved_contract.families[family=double_bottom].instance_fields`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: double_bottom
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
persistent_state: false
scale_spec: INSTANCE_PARAMETER
anchors:
- role: left_trough
  cardinality: ONE
  state_requirement: CONFIRMED
- role: intervening_peak
  cardinality: ONE
  state_requirement: CONFIRMED
- role: right_trough
  cardinality: ONE
  state_requirement: CONFIRMED
- role: neckline_zone
  cardinality: DERIVED
  state_requirement: DERIVED_FROM_CONFIRMED
- role: neckline_break
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: neckline_retest
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: continuation_anchor
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
topology_rules:
- TWO_CONFIRMED_LOWER_EXTREMA_WITH_INTERVENING_UPPER_EXTREMUM
- SHAPE_IDENTITY_INDEPENDENT_OF_FUTURE_EFFECT
- NECKLINE_BREAK_IS_LINKED_EVENT
formula_contracts:
- name: trough_height_difference_to_formation_amplitude
  numerator: absolute_left_trough_minus_right_trough
  denominator: intervening_peak_minus_compatible_trough_reference
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: time_symmetry_ratio
  numerator: absolute_left_trough_to_peak_duration_minus_peak_to_right_trough_duration
  denominator: total_formation_session_bars
  sign: UNIT_INTERVAL_WHEN_DEFINED
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
- name: intervening_peak_height_to_formation_amplitude
  numerator: intervening_peak_minus_compatible_trough_reference
  denominator: causal_price_scale
  sign: NONNEGATIVE
  unit: NORMALIZED_PRICE_MULTIPLE
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
relationship_types:
- NESTED_IN
- CONFIRMED_BY
- RETESTS
- SAME_STRUCTURE_DIFFERENT_SCALE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The mirrored anchors, horizontal neckline, formula forms, and event links are explicit, but
  trough compatibility and geometry/duration acceptance semantics are undefined.
- 建议修改：
- Define trough compatibility, reference selection, and an independently versioned geometry/duration
  policy schema.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.DOUBLE_BOTTOM.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Use the DOUBLE_BOTTOM side of the same canonical M/W detector and shared parameter profile
  used by DOUBLE_TOP.
required_changes:
- Register tools/patterns/m_w_pattern_detector.py analyze_m_w_patterns as the only DOUBLE_BOTTOM detector
  under the same pinned commit, file hash, callable, and MWPatternConfig profile as DOUBLE_TOP.
- Completely ignore m_top_detector.py and m_top_neckline_projection.py for Market Genome processing.
- Use the canonical detector's causal L-H-L structure, extreme similarity, minimum retracement, extreme
  separation, and leg-duration predicates as the strict mirror of DOUBLE_TOP.
- Preserve the current geometry-only default require_pretrend false; record a canonical prior downtrend
  as context rather than a DOUBLE_BOTTOM existence gate.
- Replace scalar neckline and outer-boundary fields with the owner-approved key-zone representation before
  formal use.
- Volume and open interest remain observational and must not gate DOUBLE_BOTTOM structure.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:30:35+08:00'
```

---

<a id="item-31"></a>

## 31. `MG1A.DOUBLE_BOTTOM.PHASE.001`

**要确认的问题**：是否同意冻结 double_bottom 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze double_bottom phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=double_bottom].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - NO_PATTERN
  - LEFT_TROUGH_AVAILABLE
  - NECKLINE_PIVOT_AVAILABLE
  - RIGHT_TROUGH_CANDIDATE
  - RIGHT_TROUGH_CONFIRMED
  - SHAPE_COMPLETE
  - NECKLINE_BREAK_CONFIRMED
  - RETEST
  - CONTINUATION
  - EXPIRED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - NO_PATTERN
    ACTIVE:
    - LEFT_TROUGH_AVAILABLE
    - NECKLINE_PIVOT_AVAILABLE
    - RIGHT_TROUGH_CANDIDATE
    - RIGHT_TROUGH_CONFIRMED
    - SHAPE_COMPLETE
    - NECKLINE_BREAK_CONFIRMED
    - RETEST
    RESOLVED:
    - CONTINUATION
    INVALIDATED:
    - EXPIRED
    - INVALIDATED
  branch_transitions:
  - from_phase: NO_PATTERN
    to_phase: LEFT_TROUGH_AVAILABLE
    event: CONFIRMED_LEFT_TROUGH
  - from_phase: LEFT_TROUGH_AVAILABLE
    to_phase: NECKLINE_PIVOT_AVAILABLE
    event: CONFIRMED_INTERVENING_PEAK
  - from_phase: NECKLINE_PIVOT_AVAILABLE
    to_phase: RIGHT_TROUGH_CANDIDATE
    event: PROVISIONAL_RIGHT_TROUGH
  - from_phase: RIGHT_TROUGH_CANDIDATE
    to_phase: RIGHT_TROUGH_CONFIRMED
    event: CONFIRMED_RIGHT_TROUGH
  - from_phase: RIGHT_TROUGH_CONFIRMED
    to_phase: SHAPE_COMPLETE
    event: GEOMETRY_POLICY_SATISFIED
  - from_phase: SHAPE_COMPLETE
    to_phase: NECKLINE_BREAK_CONFIRMED
    event: CAUSAL_UPWARD_NECKLINE_CROSS
  - from_phase: NECKLINE_BREAK_CONFIRMED
    to_phase: RETEST
    event: LINKED_NECKLINE_RETEST
  - from_phase: NECKLINE_BREAK_CONFIRMED
    to_phase: CONTINUATION
    event: NO_RETEST_CONTINUATION
  - from_phase: RETEST
    to_phase: CONTINUATION
    event: RETEST_HOLD_CONFIRMED
  - from_phase: SHAPE_COMPLETE
    to_phase: EXPIRED
    event: SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - TROUGH_ZONE_DOWNSIDE_BREACH
    - ANCHOR_ORDER_FAILURE
    - POST_BREAK_NECKLINE_RECROSS
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：The graph causally enumerates both troughs and neckline pivot, shape completion, upward
  break, separate retest/no-retest continuation, expiry, and invalidation.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.DOUBLE_BOTTOM.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: Apply the exact mirrored lifecycle mapping used by DOUBLE_TOP to canonical L-H-L candidates.
required_changes:
- UNACTIVATED means no causally available L-H-L candidate exists; partial anchors may be retained only
  as evidence.
- ACTIVE begins when the right trough is confirmed and the complete L-H-L candidate passes the canonical
  geometry rules.
- COMPLETED begins on the closed-bar neckline-break confirmation event.
- INVALIDATED begins when an active candidate is invalidated, expires, or is superseded; a replacement
  candidate receives a new stable identity.
- Keep post-confirmation failed-breakout, target-hit, full-failure, and ambiguous results as separately
  timed outcomes; they must not rewrite the completed pattern lifecycle or confirmation time.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:30:35+08:00'
```

---

<a id="item-32"></a>

## 32. `MG1A.DOUBLE_BOTTOM.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 double_bottom 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal double_bottom confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=double_bottom].confirmation_rules`
- `resolved_contract.families[family=double_bottom].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- SHAPE_COMPLETION_IS_NOT_BULLISH_EFFECT
- CAUSAL_UPWARD_NECKLINE_CROSS_CONFIRMS_CLASSIC_REVERSAL_ROLE
- PRECEDING_DOWNTREND_REQUIRED_FOR_CLASSIC_REVERSAL_ROLE
branch_transitions:
- from_phase: NO_PATTERN
  to_phase: LEFT_TROUGH_AVAILABLE
  event: CONFIRMED_LEFT_TROUGH
- from_phase: LEFT_TROUGH_AVAILABLE
  to_phase: NECKLINE_PIVOT_AVAILABLE
  event: CONFIRMED_INTERVENING_PEAK
- from_phase: NECKLINE_PIVOT_AVAILABLE
  to_phase: RIGHT_TROUGH_CANDIDATE
  event: PROVISIONAL_RIGHT_TROUGH
- from_phase: RIGHT_TROUGH_CANDIDATE
  to_phase: RIGHT_TROUGH_CONFIRMED
  event: CONFIRMED_RIGHT_TROUGH
- from_phase: RIGHT_TROUGH_CONFIRMED
  to_phase: SHAPE_COMPLETE
  event: GEOMETRY_POLICY_SATISFIED
- from_phase: SHAPE_COMPLETE
  to_phase: NECKLINE_BREAK_CONFIRMED
  event: CAUSAL_UPWARD_NECKLINE_CROSS
- from_phase: NECKLINE_BREAK_CONFIRMED
  to_phase: RETEST
  event: LINKED_NECKLINE_RETEST
- from_phase: NECKLINE_BREAK_CONFIRMED
  to_phase: CONTINUATION
  event: NO_RETEST_CONTINUATION
- from_phase: RETEST
  to_phase: CONTINUATION
  event: RETEST_HOLD_CONFIRMED
- from_phase: SHAPE_COMPLETE
  to_phase: EXPIRED
  event: SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：A causal upward neckline cross correctly confirms classic reversal role and retest is post-
  confirmation, but optional-filter UNKNOWN handling and the truth combination are absent.
- 建议修改：
- Bind the exact neckline version and upward cross predicate, then add tri-state optional-filter
  behavior.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.DOUBLE_BOTTOM.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Strictly mirror DOUBLE_TOP confirmation by applying the same canonical buffer outside
  the upper edge of the neckline zone.
required_changes:
- After the right trough is causally confirmed, confirm DOUBLE_BOTTOM only when Close is strictly above
  neckline_zone_high multiplied by one plus the shared versioned break_buffer_pct.
- Preserve the current canonical break_buffer_pct default of 0.003 and max_breakout_wait_bars default
  of 20 until their shared timeframe-specific profile is versioned or reviewed.
- Equality with the threshold and wick-only penetration must not confirm DOUBLE_BOTTOM.
- Preserve require_pretrend false in the current profile; canonical prior downtrend is contextual evidence
  rather than a confirmation gate.
- Volume and open interest remain observational and must not gate confirmation.
- Record the closed-bar confirmation time separately from anchor event and availability times.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:32:01+08:00'
```

---

<a id="item-33"></a>

## 33. `MG1A.DOUBLE_BOTTOM.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 double_bottom 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal double_bottom invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=double_bottom].invalidation_rules`
- `resolved_contract.families[family=double_bottom].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=double_bottom].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- TROUGH_ZONE_DOWNSIDE_BREACH
- ANCHOR_ORDER_FAILURE
- SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
- POST_BREAK_NECKLINE_RECROSS
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - TROUGH_ZONE_DOWNSIDE_BREACH
  - ANCHOR_ORDER_FAILURE
  - POST_BREAK_NECKLINE_RECROSS
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Trough breach, order failure, expiry, and post-break recross are separated, but observation
  boundaries, timing, and precedence are not frozen.
- 建议修改：
- Define normalized trough/order observations, expiry clock, recross boundary, and event precedence.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.DOUBLE_BOTTOM.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Strictly mirror DOUBLE_TOP invalidation by applying the shared canonical buffer outside
  the complete bottom zone.
required_changes:
- Before confirmation, invalidate DOUBLE_BOTTOM only when Close is strictly below bottom_zone_low multiplied
  by one minus the shared versioned invalidation_buffer_pct.
- Preserve the current invalidation_buffer_pct default of 0.005; equality and wick-only penetration must
  not invalidate.
- Expire an unconfirmed candidate when its canonical maximum wait is exceeded, and invalidate it as superseded
  when a new same-side structural extreme replaces it; every replacement receives a new stable identity.
- Keep post-confirmation failed breakout, full failure, target hit, and same-bar ambiguity as separately
  timed outcomes rather than rewriting the completed DOUBLE_BOTTOM lifecycle.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:32:01+08:00'
```

---

<a id="item-34"></a>

## 34. `MG1A.HEAD_SHOULDERS_TOP.STRUCTURE.001`

**要确认的问题**：是否同意冻结 head_shoulders_top 的锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze head_shoulders_top anchors, topology, geometry, duration, price ratios, and prior context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=head_shoulders_top].anchors`
- `resolved_contract.families[family=head_shoulders_top].topology_rules`
- `resolved_contract.families[family=head_shoulders_top].formula_contracts`
- `resolved_contract.families[family=head_shoulders_top].instance_fields`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: head_shoulders_top
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
persistent_state: false
scale_spec: INSTANCE_PARAMETER
anchors:
- role: left_shoulder_peak
  cardinality: ONE
  state_requirement: CONFIRMED
- role: left_neck_trough
  cardinality: ONE
  state_requirement: CONFIRMED
- role: head_peak
  cardinality: ONE
  state_requirement: CONFIRMED
- role: right_neck_trough
  cardinality: ONE
  state_requirement: CONFIRMED
- role: right_shoulder_peak
  cardinality: ONE
  state_requirement: CONFIRMED
- role: neckline_segment
  cardinality: DERIVED
  state_requirement: DERIVED_FROM_CONFIRMED
- role: neckline_break
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: neckline_retest
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: continuation_anchor
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
topology_rules:
- FIVE_CONFIRMED_ALTERNATING_EXTREMA
- HEAD_ABOVE_BOTH_SHOULDERS
- SHOULDERS_GEOMETRICALLY_COMPATIBLE
- TWO_NECK_ANCHORS_DEFINE_DYNAMIC_NECKLINE
formula_contracts:
- name: head_prominence_to_formation_amplitude
  numerator: head_peak_minus_shoulder_reference
  denominator: head_peak_minus_neckline_at_head_eob
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: shoulder_amplitude_symmetry
  numerator: absolute_left_shoulder_minus_right_shoulder
  denominator: head_peak_minus_neckline_at_head_eob
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: shoulder_time_symmetry
  numerator: absolute_left_shoulder_to_head_duration_minus_head_to_right_shoulder_duration
  denominator: total_formation_session_bars
  sign: UNIT_INTERVAL_WHEN_DEFINED
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
- name: dynamic_neckline_slope
  numerator: right_neck_trough_minus_left_neck_trough
  denominator: right_neck_eob_minus_left_neck_eob_in_session_bars
  sign: SIGNED
  unit: NORMALIZED_PRICE_PER_SESSION_BAR
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
relationship_types:
- NESTED_IN
- CONFIRMED_BY
- RETESTS
- SAME_STRUCTURE_DIFFERENT_SCALE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Five anchors, a two-anchor dynamic neckline, break/retest/continuation links, and normalized
  forms are explicit, but shoulder/head compatibility, neckline evaluation, and geometry/duration
  acceptance policies remain undefined.
- 建议修改：
- Define compatibility/reference selection, neckline-at-event evaluation, and a versioned
  geometry/duration policy schema.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.HEAD_SHOULDERS_TOP.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: HEAD_SHOULDERS_TOP and INVERSE_HEAD_SHOULDERS must be direction modes of one shared canonical
  algorithm, not independently evolving detector copies.
required_changes:
- Replace independent recognition logic in head_shoulders_top_detector.py and head_shoulders_bottom_detector.py
  with one versioned shared core; retain the existing modules only as compatibility entry points.
- Pin both audited source snapshots as migration evidence, with top SHA256 ec3499752310112fa5826cbfb6bb5708ae6fcb8a7a5d8f3e38a9aab638933fbc
  and bottom SHA256 ebc39225a63db436ec0bebba43f440ad35045fccffb7e7e41d34c9328e800556 at stock_peter commit
  57ee01baf8a6859d89e4c2e8ce83f41cc29c05b5.
- Use the audited causal H-L-H-L-H topology and its shared shoulder symmetry, head excursion, arm excursion,
  leg duration, total duration, time asymmetry, and neckline-slope predicates.
- Replace direction-specific parameter names with one HeadShouldersConfig schema and a TOP or BOTTOM direction
  parameter.
- Preserve the current geometry-only default require_pretrend false; record canonical prior uptrend as
  context rather than a structure gate.
- Apply one direction-independent positive-price-domain validity rule to every derived level throughout
  its observation window.
- Preserve the canonical time-varying scalar neckline formula as the owner-reviewed exception to the common
  zone contract.
- Volume and open interest remain observational and must not gate structure.
- Add cross-direction generated mirror tests and an effective-config equality test.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:37:01+08:00'
```

---

<a id="item-35"></a>

## 35. `MG1A.HEAD_SHOULDERS_TOP.PHASE.001`

**要确认的问题**：是否同意冻结 head_shoulders_top 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze head_shoulders_top phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=head_shoulders_top].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - NO_PATTERN
  - LEFT_SHOULDER_AVAILABLE
  - LEFT_NECKLINE_AVAILABLE
  - HEAD_AVAILABLE
  - RIGHT_NECKLINE_AVAILABLE
  - RIGHT_SHOULDER_CANDIDATE
  - SHAPE_COMPLETE
  - NECKLINE_BREAK_CONFIRMED
  - RETEST
  - CONTINUATION
  - EXPIRED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - NO_PATTERN
    ACTIVE:
    - LEFT_SHOULDER_AVAILABLE
    - LEFT_NECKLINE_AVAILABLE
    - HEAD_AVAILABLE
    - RIGHT_NECKLINE_AVAILABLE
    - RIGHT_SHOULDER_CANDIDATE
    - SHAPE_COMPLETE
    - NECKLINE_BREAK_CONFIRMED
    - RETEST
    RESOLVED:
    - CONTINUATION
    INVALIDATED:
    - EXPIRED
    - INVALIDATED
  branch_transitions:
  - from_phase: NO_PATTERN
    to_phase: LEFT_SHOULDER_AVAILABLE
    event: CONFIRMED_LEFT_SHOULDER
  - from_phase: LEFT_SHOULDER_AVAILABLE
    to_phase: LEFT_NECKLINE_AVAILABLE
    event: CONFIRMED_LEFT_NECK_TROUGH
  - from_phase: LEFT_NECKLINE_AVAILABLE
    to_phase: HEAD_AVAILABLE
    event: CONFIRMED_HEAD_PEAK
  - from_phase: HEAD_AVAILABLE
    to_phase: RIGHT_NECKLINE_AVAILABLE
    event: CONFIRMED_RIGHT_NECK_TROUGH
  - from_phase: RIGHT_NECKLINE_AVAILABLE
    to_phase: RIGHT_SHOULDER_CANDIDATE
    event: PROVISIONAL_RIGHT_SHOULDER
  - from_phase: RIGHT_SHOULDER_CANDIDATE
    to_phase: SHAPE_COMPLETE
    event: CONFIRMED_RIGHT_SHOULDER_AND_GEOMETRY
  - from_phase: SHAPE_COMPLETE
    to_phase: NECKLINE_BREAK_CONFIRMED
    event: CAUSAL_DOWNWARD_DYNAMIC_NECKLINE_CROSS
  - from_phase: NECKLINE_BREAK_CONFIRMED
    to_phase: RETEST
    event: LINKED_NECKLINE_RETEST
  - from_phase: NECKLINE_BREAK_CONFIRMED
    to_phase: CONTINUATION
    event: NO_RETEST_CONTINUATION
  - from_phase: RETEST
    to_phase: CONTINUATION
    event: RETEST_HOLD_CONFIRMED
  - from_phase: SHAPE_COMPLETE
    to_phase: EXPIRED
    event: SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - HEAD_ZONE_UPSIDE_BREACH
    - RIGHT_SHOULDER_TOPOLOGY_FAILURE
    - ANCHOR_ORDER_FAILURE
    - POST_BREAK_NECKLINE_RECROSS
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：Every anchor-availability stage, shape completion, dynamic neckline break, separate
  retest/no-retest continuation, expiry, and invalidation is explicitly represented in a valid
  graph.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.HEAD_SHOULDERS_TOP.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: Map only the shared canonical algorithm's emitted candidate events into the common four-state
  lifecycle.
required_changes:
- UNACTIVATED means no causally available H-L-H-L-H candidate exists; partial anchors may be retained
  only as evidence.
- ACTIVE begins when the right shoulder is confirmed and the complete five-pivot candidate passes all
  canonical geometry and dynamic-neckline validity rules.
- COMPLETED begins on the closed-bar neckline-break confirmation event.
- INVALIDATED begins when an active candidate is invalidated, expires, or is superseded; a replacement
  receives a new stable identity.
- Keep failed breakdown, target hit, full failure, and same-bar ambiguity as separately timed post-completion
  outcomes; they must not rewrite lifecycle or confirmation time.
- Do not invent separate lifecycle phases for intermediate anchors, retest, or continuation when the canonical
  detector does not emit them.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:37:01+08:00'
```

---

<a id="item-36"></a>

## 36. `MG1A.HEAD_SHOULDERS_TOP.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 head_shoulders_top 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal head_shoulders_top confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=head_shoulders_top].confirmation_rules`
- `resolved_contract.families[family=head_shoulders_top].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- SHAPE_COMPLETION_IS_NOT_BEARISH_EFFECT
- DYNAMIC_NECKLINE_VALUE_EVALUATED_AT_CROSS_EOB
- CAUSAL_DOWNWARD_NECKLINE_CROSS_CONFIRMS_CLASSIC_REVERSAL_ROLE
branch_transitions:
- from_phase: NO_PATTERN
  to_phase: LEFT_SHOULDER_AVAILABLE
  event: CONFIRMED_LEFT_SHOULDER
- from_phase: LEFT_SHOULDER_AVAILABLE
  to_phase: LEFT_NECKLINE_AVAILABLE
  event: CONFIRMED_LEFT_NECK_TROUGH
- from_phase: LEFT_NECKLINE_AVAILABLE
  to_phase: HEAD_AVAILABLE
  event: CONFIRMED_HEAD_PEAK
- from_phase: HEAD_AVAILABLE
  to_phase: RIGHT_NECKLINE_AVAILABLE
  event: CONFIRMED_RIGHT_NECK_TROUGH
- from_phase: RIGHT_NECKLINE_AVAILABLE
  to_phase: RIGHT_SHOULDER_CANDIDATE
  event: PROVISIONAL_RIGHT_SHOULDER
- from_phase: RIGHT_SHOULDER_CANDIDATE
  to_phase: SHAPE_COMPLETE
  event: CONFIRMED_RIGHT_SHOULDER_AND_GEOMETRY
- from_phase: SHAPE_COMPLETE
  to_phase: NECKLINE_BREAK_CONFIRMED
  event: CAUSAL_DOWNWARD_DYNAMIC_NECKLINE_CROSS
- from_phase: NECKLINE_BREAK_CONFIRMED
  to_phase: RETEST
  event: LINKED_NECKLINE_RETEST
- from_phase: NECKLINE_BREAK_CONFIRMED
  to_phase: CONTINUATION
  event: NO_RETEST_CONTINUATION
- from_phase: RETEST
  to_phase: CONTINUATION
  event: RETEST_HOLD_CONFIRMED
- from_phase: SHAPE_COMPLETE
  to_phase: EXPIRED
  event: SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The dynamic downward neckline cross is causal and follows the right shoulder, but exact
  preceding-context linkage and optional-filter UNKNOWN behavior are not defined at family level.
- 建议修改：
- Require a specific preceding-context pattern reference, freeze neckline-at-cross evaluation, and
  add a tri-state optional-filter truth table.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.HEAD_SHOULDERS_TOP.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Use the shared canonical algorithm's time-varying scalar neckline and strict closed-bar
  breakdown rule without expanding the neckline into a price zone.
required_changes:
- Compute neckline N(t) by linearly extending the two confirmed neckline pivots under the shared core's
  slope and positive-price validity constraints.
- After the right shoulder is causally confirmed, confirm HEAD_SHOULDERS_TOP only when Close is strictly
  below N(t) multiplied by one minus the shared versioned break_buffer_pct.
- Preserve the current break_buffer_pct default of 0.003 and max_breakout_wait_bars default of 20 until
  their shared timeframe-specific profile is versioned or reviewed.
- Equality with the threshold and wick-only penetration must not confirm the pattern.
- Preserve require_pretrend false in the current profile; prior uptrend is contextual evidence rather
  than a confirmation gate.
- Volume and open interest remain observational and must not gate confirmation.
- Record N(t), the buffered threshold, and the closed-bar confirmation time as versioned evidence.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:39:42+08:00'
```

---

<a id="item-37"></a>

## 37. `MG1A.HEAD_SHOULDERS_TOP.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 head_shoulders_top 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal head_shoulders_top invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=head_shoulders_top].invalidation_rules`
- `resolved_contract.families[family=head_shoulders_top].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=head_shoulders_top].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- HEAD_ZONE_UPSIDE_BREACH
- RIGHT_SHOULDER_TOPOLOGY_FAILURE
- ANCHOR_ORDER_FAILURE
- SCALE_RELATIVE_EXPIRY
- POST_BREAK_NECKLINE_RECROSS
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - HEAD_ZONE_UPSIDE_BREACH
  - RIGHT_SHOULDER_TOPOLOGY_FAILURE
  - ANCHOR_ORDER_FAILURE
  - POST_BREAK_NECKLINE_RECROSS
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Head breach, shoulder/order failure, two expiry contexts, and recross are named, but
  normalized boundaries, separate pre/post-confirmation clocks, and event precedence are incomplete.
- 建议修改：
- Define normalized failure boundaries, distinct expiry clocks, recross observation, and
  deterministic precedence.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.HEAD_SHOULDERS_TOP.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Preserve the canonical head-based invalidation, dynamic-neckline expiry, maximum wait,
  and supersession rules in the shared core.
required_changes:
- Before confirmation, invalidate only when Close is strictly above head_price multiplied by one plus
  the shared versioned invalidation_buffer_pct; preserve the current default of 0.005.
- Expire the candidate if N(t) becomes non-positive, reaches or exceeds the right shoulder, or the canonical
  maximum wait is exceeded.
- Invalidate a candidate as superseded when a new structural pivot replaces it; every replacement receives
  a new stable identity.
- Equality and wick-only penetration must not trigger head-based invalidation.
- Preserve the canonical three-bar failed-breakdown window and later full-failure, target-hit, and same-bar
  ambiguity as separately timed post-completion outcomes rather than lifecycle rewrites.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:39:42+08:00'
```

---

<a id="item-38"></a>

## 38. `MG1A.INVERSE_HEAD_SHOULDERS.STRUCTURE.001`

**要确认的问题**：是否同意冻结 inverse_head_shoulders 的锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze inverse_head_shoulders anchors, topology, geometry, duration, price ratios, and prior
context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=inverse_head_shoulders].anchors`
- `resolved_contract.families[family=inverse_head_shoulders].topology_rules`
- `resolved_contract.families[family=inverse_head_shoulders].formula_contracts`
- `resolved_contract.families[family=inverse_head_shoulders].instance_fields`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: inverse_head_shoulders
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
persistent_state: false
scale_spec: INSTANCE_PARAMETER
anchors:
- role: left_shoulder_trough
  cardinality: ONE
  state_requirement: CONFIRMED
- role: left_neck_peak
  cardinality: ONE
  state_requirement: CONFIRMED
- role: head_trough
  cardinality: ONE
  state_requirement: CONFIRMED
- role: right_neck_peak
  cardinality: ONE
  state_requirement: CONFIRMED
- role: right_shoulder_trough
  cardinality: ONE
  state_requirement: CONFIRMED
- role: neckline_segment
  cardinality: DERIVED
  state_requirement: DERIVED_FROM_CONFIRMED
- role: neckline_break
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: neckline_retest
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: continuation_anchor
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
topology_rules:
- FIVE_CONFIRMED_ALTERNATING_EXTREMA
- HEAD_BELOW_BOTH_SHOULDERS
- SHOULDERS_GEOMETRICALLY_COMPATIBLE
- TWO_NECK_ANCHORS_DEFINE_DYNAMIC_NECKLINE
formula_contracts:
- name: head_prominence_to_formation_amplitude
  numerator: shoulder_reference_minus_head_trough
  denominator: neckline_at_head_eob_minus_head_trough
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: shoulder_amplitude_symmetry
  numerator: absolute_left_shoulder_minus_right_shoulder
  denominator: neckline_at_head_eob_minus_head_trough
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: shoulder_time_symmetry
  numerator: absolute_left_shoulder_to_head_duration_minus_head_to_right_shoulder_duration
  denominator: total_formation_session_bars
  sign: UNIT_INTERVAL_WHEN_DEFINED
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
- name: dynamic_neckline_slope
  numerator: right_neck_peak_minus_left_neck_peak
  denominator: right_neck_eob_minus_left_neck_eob_in_session_bars
  sign: SIGNED
  unit: NORMALIZED_PRICE_PER_SESSION_BAR
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
relationship_types:
- NESTED_IN
- CONFIRMED_BY
- RETESTS
- SAME_STRUCTURE_DIFFERENT_SCALE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The inverse family has independent anchors, dynamic neckline, event links, and formula forms,
  but shoulder/head compatibility and geometry/duration acceptance semantics are undefined.
- 建议修改：
- Define inverse compatibility/reference selection, neckline evaluation, and independently versioned
  geometry/duration policy references.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.INVERSE_HEAD_SHOULDERS.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Define INVERSE_HEAD_SHOULDERS as the strict BOTTOM-direction mirror of the same shared
  canonical core.
required_changes:
- Use the same pinned migration evidence, shared implementation, HeadShouldersConfig, output schema, and
  lifecycle engine as HEAD_SHOULDERS_TOP.
- Use the audited causal L-H-L-H-L topology with direction-mirrored shoulder, head, arm, duration, asymmetry,
  and neckline predicates.
- Preserve the current geometry-only default require_pretrend false; record canonical prior downtrend
  as context rather than a structure gate.
- Apply the same direction-independent positive-price-domain validity rule and preserve the canonical
  time-varying scalar neckline formula as the owner-reviewed exception to the common zone contract.
- Volume and open interest remain observational and must not gate structure.
- Require cross-direction generated mirror tests and prohibit independent bottom-only recognition logic.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:37:01+08:00'
```

---

<a id="item-39"></a>

## 39. `MG1A.INVERSE_HEAD_SHOULDERS.PHASE.001`

**要确认的问题**：是否同意冻结 inverse_head_shoulders 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze inverse_head_shoulders phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=inverse_head_shoulders].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - NO_PATTERN
  - LEFT_SHOULDER_AVAILABLE
  - LEFT_NECKLINE_AVAILABLE
  - HEAD_AVAILABLE
  - RIGHT_NECKLINE_AVAILABLE
  - RIGHT_SHOULDER_CANDIDATE
  - SHAPE_COMPLETE
  - NECKLINE_BREAK_CONFIRMED
  - RETEST
  - CONTINUATION
  - EXPIRED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - NO_PATTERN
    ACTIVE:
    - LEFT_SHOULDER_AVAILABLE
    - LEFT_NECKLINE_AVAILABLE
    - HEAD_AVAILABLE
    - RIGHT_NECKLINE_AVAILABLE
    - RIGHT_SHOULDER_CANDIDATE
    - SHAPE_COMPLETE
    - NECKLINE_BREAK_CONFIRMED
    - RETEST
    RESOLVED:
    - CONTINUATION
    INVALIDATED:
    - EXPIRED
    - INVALIDATED
  branch_transitions:
  - from_phase: NO_PATTERN
    to_phase: LEFT_SHOULDER_AVAILABLE
    event: CONFIRMED_LEFT_SHOULDER
  - from_phase: LEFT_SHOULDER_AVAILABLE
    to_phase: LEFT_NECKLINE_AVAILABLE
    event: CONFIRMED_LEFT_NECK_PEAK
  - from_phase: LEFT_NECKLINE_AVAILABLE
    to_phase: HEAD_AVAILABLE
    event: CONFIRMED_HEAD_TROUGH
  - from_phase: HEAD_AVAILABLE
    to_phase: RIGHT_NECKLINE_AVAILABLE
    event: CONFIRMED_RIGHT_NECK_PEAK
  - from_phase: RIGHT_NECKLINE_AVAILABLE
    to_phase: RIGHT_SHOULDER_CANDIDATE
    event: PROVISIONAL_RIGHT_SHOULDER
  - from_phase: RIGHT_SHOULDER_CANDIDATE
    to_phase: SHAPE_COMPLETE
    event: CONFIRMED_RIGHT_SHOULDER_AND_GEOMETRY
  - from_phase: SHAPE_COMPLETE
    to_phase: NECKLINE_BREAK_CONFIRMED
    event: CAUSAL_UPWARD_DYNAMIC_NECKLINE_CROSS
  - from_phase: NECKLINE_BREAK_CONFIRMED
    to_phase: RETEST
    event: LINKED_NECKLINE_RETEST
  - from_phase: NECKLINE_BREAK_CONFIRMED
    to_phase: CONTINUATION
    event: NO_RETEST_CONTINUATION
  - from_phase: RETEST
    to_phase: CONTINUATION
    event: RETEST_HOLD_CONFIRMED
  - from_phase: SHAPE_COMPLETE
    to_phase: EXPIRED
    event: SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - HEAD_ZONE_DOWNSIDE_BREACH
    - RIGHT_SHOULDER_TOPOLOGY_FAILURE
    - ANCHOR_ORDER_FAILURE
    - POST_BREAK_NECKLINE_RECROSS
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：The independent graph explicitly covers all formation anchors, shape completion, upward
  dynamic break, separate retest/no-retest continuation, expiry, and invalidation.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.INVERSE_HEAD_SHOULDERS.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: Strictly mirror the HEAD_SHOULDERS_TOP lifecycle using the shared core's L-H-L-H-L evidence.
required_changes:
- UNACTIVATED means no causally available L-H-L-H-L candidate exists; partial anchors may be retained
  only as evidence.
- ACTIVE begins when the right shoulder is confirmed and the complete five-pivot candidate passes all
  canonical geometry and dynamic-neckline validity rules.
- COMPLETED begins on the closed-bar neckline-break confirmation event.
- INVALIDATED begins when an active candidate is invalidated, expires, or is superseded; a replacement
  receives a new stable identity.
- Keep failed breakout, target hit, full failure, and same-bar ambiguity as separately timed post-completion
  outcomes; they must not rewrite lifecycle or confirmation time.
- Do not invent separate lifecycle phases for intermediate anchors, retest, or continuation when the canonical
  detector does not emit them.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:37:01+08:00'
```

---

<a id="item-40"></a>

## 40. `MG1A.INVERSE_HEAD_SHOULDERS.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 inverse_head_shoulders 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal inverse_head_shoulders confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=inverse_head_shoulders].confirmation_rules`
- `resolved_contract.families[family=inverse_head_shoulders].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- SHAPE_COMPLETION_IS_NOT_BULLISH_EFFECT
- DYNAMIC_NECKLINE_VALUE_EVALUATED_AT_CROSS_EOB
- CAUSAL_UPWARD_NECKLINE_CROSS_CONFIRMS_CLASSIC_REVERSAL_ROLE
branch_transitions:
- from_phase: NO_PATTERN
  to_phase: LEFT_SHOULDER_AVAILABLE
  event: CONFIRMED_LEFT_SHOULDER
- from_phase: LEFT_SHOULDER_AVAILABLE
  to_phase: LEFT_NECKLINE_AVAILABLE
  event: CONFIRMED_LEFT_NECK_PEAK
- from_phase: LEFT_NECKLINE_AVAILABLE
  to_phase: HEAD_AVAILABLE
  event: CONFIRMED_HEAD_TROUGH
- from_phase: HEAD_AVAILABLE
  to_phase: RIGHT_NECKLINE_AVAILABLE
  event: CONFIRMED_RIGHT_NECK_PEAK
- from_phase: RIGHT_NECKLINE_AVAILABLE
  to_phase: RIGHT_SHOULDER_CANDIDATE
  event: PROVISIONAL_RIGHT_SHOULDER
- from_phase: RIGHT_SHOULDER_CANDIDATE
  to_phase: SHAPE_COMPLETE
  event: CONFIRMED_RIGHT_SHOULDER_AND_GEOMETRY
- from_phase: SHAPE_COMPLETE
  to_phase: NECKLINE_BREAK_CONFIRMED
  event: CAUSAL_UPWARD_DYNAMIC_NECKLINE_CROSS
- from_phase: NECKLINE_BREAK_CONFIRMED
  to_phase: RETEST
  event: LINKED_NECKLINE_RETEST
- from_phase: NECKLINE_BREAK_CONFIRMED
  to_phase: CONTINUATION
  event: NO_RETEST_CONTINUATION
- from_phase: RETEST
  to_phase: CONTINUATION
  event: RETEST_HOLD_CONFIRMED
- from_phase: SHAPE_COMPLETE
  to_phase: EXPIRED
  event: SCALE_RELATIVE_BREAK_ABSENCE_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The causal upward dynamic-neckline cross follows the right shoulder and is separated from
  future effect, but preceding-context linkage and optional-filter UNKNOWN behavior are absent.
- 建议修改：
- Require a specific preceding-context reference, freeze neckline-at-cross evaluation, and add tri-
  state optional-filter behavior.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.INVERSE_HEAD_SHOULDERS.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Strictly mirror top confirmation by using the same shared dynamic-neckline formula and
  buffered closed-bar rule in the BOTTOM direction.
required_changes:
- Compute neckline N(t) by linearly extending the two confirmed neckline pivots under the shared core's
  slope and positive-price validity constraints.
- After the right shoulder is causally confirmed, confirm INVERSE_HEAD_SHOULDERS only when Close is strictly
  above N(t) multiplied by one plus the shared versioned break_buffer_pct.
- Preserve the current break_buffer_pct default of 0.003 and max_breakout_wait_bars default of 20 under
  the same shared timeframe-specific profile as HEAD_SHOULDERS_TOP.
- Equality with the threshold and wick-only penetration must not confirm the pattern.
- Preserve require_pretrend false in the current profile; prior downtrend is contextual evidence rather
  than a confirmation gate.
- Volume and open interest remain observational and must not gate confirmation.
- Record N(t), the buffered threshold, and the closed-bar confirmation time as versioned evidence.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:39:42+08:00'
```

---

<a id="item-41"></a>

## 41. `MG1A.INVERSE_HEAD_SHOULDERS.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 inverse_head_shoulders 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal inverse_head_shoulders invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=inverse_head_shoulders].invalidation_rules`
- `resolved_contract.families[family=inverse_head_shoulders].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=inverse_head_shoulders].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- HEAD_ZONE_DOWNSIDE_BREACH
- RIGHT_SHOULDER_TOPOLOGY_FAILURE
- ANCHOR_ORDER_FAILURE
- SCALE_RELATIVE_EXPIRY
- POST_BREAK_NECKLINE_RECROSS
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - HEAD_ZONE_DOWNSIDE_BREACH
  - RIGHT_SHOULDER_TOPOLOGY_FAILURE
  - ANCHOR_ORDER_FAILURE
  - POST_BREAK_NECKLINE_RECROSS
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Head breach, topology/order failure, expiry, and downward recross are represented, but
  normalized boundaries, separate pre/post-confirmation expiry, and precedence are not frozen.
- 建议修改：
- Define normalized failure/recross boundaries, both expiry clocks, and deterministic event
  precedence.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.INVERSE_HEAD_SHOULDERS.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Strictly mirror top invalidation, dynamic-neckline expiry, maximum wait, and supersession
  in the shared BOTTOM-direction core.
required_changes:
- Before confirmation, invalidate only when Close is strictly below head_price multiplied by one minus
  the shared versioned invalidation_buffer_pct; preserve the current default of 0.005.
- Expire the candidate if N(t) reaches or falls below the right shoulder or the canonical maximum wait
  is exceeded, while enforcing the common positive-price-domain rule on every derived level.
- Invalidate a candidate as superseded when a new structural pivot replaces it; every replacement receives
  a new stable identity.
- Equality and wick-only penetration must not trigger head-based invalidation.
- Preserve the canonical three-bar failed-breakout window and later full-failure, target-hit, and same-bar
  ambiguity as separately timed post-completion outcomes rather than lifecycle rewrites.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:39:42+08:00'
```

---
