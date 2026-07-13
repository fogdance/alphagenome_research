# 趋势与区间结构

[返回审核总览](README.md)

本章共 16 条。AI 内容仅作参考，最终决定由 `PROJECT_OWNER` 作出。

<a id="item-14"></a>

## 14. `MG1A.TREND_UP.STRUCTURE.001`

**要确认的问题**：是否同意冻结 trend_up 的锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze trend_up anchors, topology, geometry, duration, price ratios, and prior context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_up].anchors`
- `resolved_contract.families[family=trend_up].topology_rules`
- `resolved_contract.families[family=trend_up].formula_contracts`
- `resolved_contract.families[family=trend_up].instance_fields`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: trend_up
object_class: STRUCTURAL_STATE
structural_orientation: UPWARD
orientation_binding: FIXED
allowed_instance_orientations:
- UPWARD
semantic_roles:
- TREND_STATE
persistent_state: true
scale_spec: INSTANCE_PARAMETER
anchors:
- role: trend_origin_low
  cardinality: ONE
  state_requirement: CONFIRMED
- role: swing_high
  cardinality: ONE_OR_MORE
  state_requirement: CONFIRMED
- role: pullback_low
  cardinality: ONE_OR_MORE
  state_requirement: CONFIRMED
- role: current_frontier
  cardinality: ONE
  state_requirement: PROVISIONAL_ALLOWED_RUNTIME_ONLY
topology_rules:
- TEMPORALLY_ALTERNATING_CONFIRMED_SWINGS
- SUCCESSIVE_HIGHS_AND_LOWS_ASCEND
- SEQUENCE_READINESS_EXPLICIT
formula_contracts:
- name: high_envelope_slope_ratio
  numerator: high_envelope_normalized_price_change
  denominator: high_envelope_elapsed_session_bars
  sign: SIGNED
  unit: NORMALIZED_PRICE_PER_SESSION_BAR
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
- name: pullback_depth_to_prior_impulse_ratio
  numerator: prior_swing_high_minus_pullback_low
  denominator: prior_impulse_price_change
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: path_efficiency_ratio
  numerator: absolute_frontier_minus_origin
  denominator: cumulative_absolute_closed_bar_price_change
  sign: UNIT_INTERVAL_WHEN_DEFINED
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
relationship_types:
- CONTAINS
- INVALIDATED_BY
- SAME_STRUCTURE_DIFFERENT_SCALE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Confirmed alternating anchors and normalized formula forms exist, but
  SEQUENCE_READINESS_EXPLICIT has no minimum comparisons, order tolerance, or versioned
  geometry/duration policy binding.
- 建议修改：
- Define causal readiness and comparison semantics plus versioned geometry, duration, and ratio
  policy references; leave fitted values train-only.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_UP.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Adopt the canonical market_structure semantics in which local structure may report confirmed
  HH and HL, while the official uptrend activates earlier after a confirmed HL and a closed-bar break
  above the prior high.
required_changes:
- Bind the trend structure definition to the selected version of stock_peter tools/patterns/market_structure.py.
- Keep confirmed HH and HL as the local up-structure descriptor.
- Define official uptrend readiness as a confirmed L-H-HL setup followed by a closed-bar close strictly
  above the prior high under the registered break-buffer policy.
- Require all compared anchors to belong to the same timeframe and active contract.
- Make each pivot usable only on its algorithmic confirmation bar and apply the versioned key-zone and
  equality-tolerance rules.
- Apply the exact mirrored readiness rule to the downtrend family.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:16:42+08:00'
```

---

<a id="item-15"></a>

## 15. `MG1A.TREND_UP.PHASE.001`

**要确认的问题**：是否同意冻结 trend_up 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze trend_up phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_up].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - INACTIVE
  - IMPULSE_CANDIDATE
  - SEQUENCE_FORMING
  - ESTABLISHED
  - PULLBACK
  - RESUMING
  - WEAKENING
  - STRUCTURE_BREAK
  - TERMINATED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - INACTIVE
    ACTIVE:
    - IMPULSE_CANDIDATE
    - SEQUENCE_FORMING
    - ESTABLISHED
    - PULLBACK
    - RESUMING
    - WEAKENING
    - STRUCTURE_BREAK
    RESOLVED:
    - TERMINATED
    INVALIDATED:
    - INVALIDATED
  branch_transitions:
  - from_phase: INACTIVE
    to_phase: IMPULSE_CANDIDATE
    event: CAUSAL_DIRECTIONAL_IMPULSE
  - from_phase: IMPULSE_CANDIDATE
    to_phase: SEQUENCE_FORMING
    event: FIRST_COMPARABLE_SWING_PAIR
  - from_phase: SEQUENCE_FORMING
    to_phase: ESTABLISHED
    event: ASCENDING_SEQUENCE_READY
  - from_phase: ESTABLISHED
    to_phase: PULLBACK
    event: CAUSAL_PULLBACK_OBSERVED
  - from_phase: PULLBACK
    to_phase: RESUMING
    event: PRIOR_STRUCTURE_HOLDS
  - from_phase: RESUMING
    to_phase: ESTABLISHED
    event: NEW_ASCENDING_SEQUENCE_EVIDENCE
  - from_phase: ESTABLISHED
    to_phase: WEAKENING
    event: ORDER_OR_MOMENTUM_WEAKENS
  - from_phase: WEAKENING
    to_phase: STRUCTURE_BREAK
    event: DEFENDING_PIVOT_BREAK
  - from_phase: STRUCTURE_BREAK
    to_phase: TERMINATED
    event: TERMINATION_RECORDED
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - CONFIRMED_PULLBACK_LOW_BREACH
    - ASCENDING_ORDER_BREAK
    - SCALE_RELATIVE_FORMATION_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The graph separates impulse, sequence, pullback, resumption, weakening, break, and
  termination, but CAUSAL_DIRECTIONAL_IMPULSE and TERMINATION_RECORDED are undefined predicates and
  do not close persistent completion.
- 建议修改：
- Bind every trend-up phase edge to the common causal event registry, including an observable
  persistent-state termination predicate.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_UP.PHASE.001
owner_decision: APPROVE_AS_IS
owner_rationale: Retain the proposed uptrend phases; the exact causal predicate for every phase transition
  must be supplied under the common lifecycle revision requirement.
required_changes: []
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T20:53:11+08:00'
```

---

<a id="item-16"></a>

## 16. `MG1A.TREND_UP.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 trend_up 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal trend_up confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_up].confirmation_rules`
- `resolved_contract.families[family=trend_up].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- ASCENDING_SEQUENCE_READY
- LATEST_PULLBACK_HOLDS
- OPTIONAL_FILTERS_DO_NOT_DEFINE_TOPOLOGY
branch_transitions:
- from_phase: INACTIVE
  to_phase: IMPULSE_CANDIDATE
  event: CAUSAL_DIRECTIONAL_IMPULSE
- from_phase: IMPULSE_CANDIDATE
  to_phase: SEQUENCE_FORMING
  event: FIRST_COMPARABLE_SWING_PAIR
- from_phase: SEQUENCE_FORMING
  to_phase: ESTABLISHED
  event: ASCENDING_SEQUENCE_READY
- from_phase: ESTABLISHED
  to_phase: PULLBACK
  event: CAUSAL_PULLBACK_OBSERVED
- from_phase: PULLBACK
  to_phase: RESUMING
  event: PRIOR_STRUCTURE_HOLDS
- from_phase: RESUMING
  to_phase: ESTABLISHED
  event: NEW_ASCENDING_SEQUENCE_EVIDENCE
- from_phase: ESTABLISHED
  to_phase: WEAKENING
  event: ORDER_OR_MOMENTUM_WEAKENS
- from_phase: WEAKENING
  to_phase: STRUCTURE_BREAK
  event: DEFENDING_PIVOT_BREAK
- from_phase: STRUCTURE_BREAK
  to_phase: TERMINATED
  event: TERMINATION_RECORDED
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Ascending readiness and pullback hold are named but their conjunction, sequence completeness,
  and optional-filter UNKNOWN propagation are not normative.
- 建议修改：
- Add a tri-state confirmation truth table covering readiness, pullback hold, completeness, and
  missing optional filters.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_UP.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Use the existing canonical algorithm rule in which a confirmed higher low followed by
  a closed-bar break above the prior high confirms the uptrend without waiting for the new high to become
  a confirmed pivot.
required_changes:
- Reference the selected market_structure algorithm, source commit, file hash, callable, and versioned
  parameter profile.
- Confirm an uptrend only after the L-H-HL pivots are causally available and a later closed bar closes
  strictly above the prior high under the configured break buffer.
- Build the prior-high zone from its confirmed pivot evidence and require Close to be strictly above prior_high_zone_high
  multiplied by one plus the versioned break_buffer_pct.
- Equality with the threshold and wick-only penetration must not confirm TREND_UP.
- Do not use volume or open interest as confirmation conditions.
- Apply the exact mirrored rule to downtrend confirmation.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:16:42+08:00'
```

---

<a id="item-17"></a>

## 17. `MG1A.TREND_UP.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 trend_up 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal trend_up invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_up].invalidation_rules`
- `resolved_contract.families[family=trend_up].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=trend_up].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- CONFIRMED_PULLBACK_LOW_BREACH
- ASCENDING_ORDER_BREAK
- SCALE_RELATIVE_FORMATION_EXPIRY
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - CONFIRMED_PULLBACK_LOW_BREACH
  - ASCENDING_ORDER_BREAK
  - SCALE_RELATIVE_FORMATION_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Breach, order break, and scale-relative expiry are enumerated, but defending-pivot identity,
  observation basis, expiry clock, and simultaneous-event precedence remain undefined.
- 建议修改：
- Define pivot selection, normalized breach observation, expiry clock, and deterministic
  invalidation/transition precedence.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_UP.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Preserve the canonical protected-HL invalidation event while applying the break buffer
  outside the owner-approved protected-low zone.
required_changes:
- Build the protected-HL zone from the causally confirmed protected-low pivot and its closed bar.
- Invalidate TREND_UP only when Close is strictly below protected_hl_zone_low multiplied by one minus
  the shared versioned break_buffer_pct.
- Equality with the threshold and wick-only penetration must not invalidate TREND_UP.
- At the invalidating bar close, terminalize the active trend object as INVALIDATED and return the detector
  runtime state to neutral without rewriting prior trend history.
- Preserve the canonical continuation rule that replaces the protected HL only after a new external high,
  a later confirmed HL, and a closed-bar break above that external high.
- Do not use volume or open interest as invalidation conditions.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:42:33+08:00'
```

---

<a id="item-18"></a>

## 18. `MG1A.TREND_DOWN.STRUCTURE.001`

**要确认的问题**：是否同意冻结 trend_down 的锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze trend_down anchors, topology, geometry, duration, price ratios, and prior context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_down].anchors`
- `resolved_contract.families[family=trend_down].topology_rules`
- `resolved_contract.families[family=trend_down].formula_contracts`
- `resolved_contract.families[family=trend_down].instance_fields`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: trend_down
object_class: STRUCTURAL_STATE
structural_orientation: DOWNWARD
orientation_binding: FIXED
allowed_instance_orientations:
- DOWNWARD
semantic_roles:
- TREND_STATE
persistent_state: true
scale_spec: INSTANCE_PARAMETER
anchors:
- role: trend_origin_high
  cardinality: ONE
  state_requirement: CONFIRMED
- role: swing_low
  cardinality: ONE_OR_MORE
  state_requirement: CONFIRMED
- role: reaction_high
  cardinality: ONE_OR_MORE
  state_requirement: CONFIRMED
- role: current_frontier
  cardinality: ONE
  state_requirement: PROVISIONAL_ALLOWED_RUNTIME_ONLY
topology_rules:
- TEMPORALLY_ALTERNATING_CONFIRMED_SWINGS
- SUCCESSIVE_LOWS_AND_HIGHS_DESCEND
- SEQUENCE_READINESS_EXPLICIT
formula_contracts:
- name: low_envelope_slope_ratio
  numerator: low_envelope_normalized_price_change
  denominator: low_envelope_elapsed_session_bars
  sign: SIGNED
  unit: NORMALIZED_PRICE_PER_SESSION_BAR
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
- name: reaction_depth_to_prior_impulse_ratio
  numerator: reaction_high_minus_prior_swing_low
  denominator: prior_impulse_absolute_price_change
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: path_efficiency_ratio
  numerator: absolute_frontier_minus_origin
  denominator: cumulative_absolute_closed_bar_price_change
  sign: UNIT_INTERVAL_WHEN_DEFINED
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: NO_CLIPPING
  fit_boundary: NOT_APPLICABLE
relationship_types:
- CONTAINS
- INVALIDATED_BY
- SAME_STRUCTURE_DIFFERENT_SCALE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The independent descending topology and formula forms exist, but readiness, order tolerance,
  exact mirror transform, and independent geometry/duration policy bindings are absent.
- 建议修改：
- Define causal readiness/tolerance, the exact up-to-down transform, and independently versioned
  geometry/duration policy references.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_DOWN.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Strictly mirror the selected canonical market_structure uptrend rule in the DOWN direction.
required_changes:
- Bind both trend directions to tools/patterns/market_structure.py analyze_market_structure at stock_peter
  commit 57ee01baf8a6859d89e4c2e8ce83f41cc29c05b5 and file SHA256 cc87add75331f4a31359bb31f2120e6df6c302326f028260d504f27fc1bbd247.
- Keep confirmed LH and LL as the local down-structure descriptor.
- Define official downtrend readiness as a confirmed H-L-LH setup followed by a closed-bar close strictly
  below the prior low under the shared registered break-buffer policy.
- Require all anchors to belong to the same timeframe and active contract and to be usable only on their
  algorithmic confirmation bars.
- Apply the common key-zone contract and the exact same effective parameter profile used by TREND_UP.
- Add generated mirror tests proving that price-inverted inputs produce direction-inverted structures
  and events.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:41:13+08:00'
```

---

<a id="item-19"></a>

## 19. `MG1A.TREND_DOWN.PHASE.001`

**要确认的问题**：是否同意冻结 trend_down 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze trend_down phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_down].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - INACTIVE
  - IMPULSE_CANDIDATE
  - SEQUENCE_FORMING
  - ESTABLISHED
  - REACTION
  - RESUMING
  - WEAKENING
  - STRUCTURE_BREAK
  - TERMINATED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - INACTIVE
    ACTIVE:
    - IMPULSE_CANDIDATE
    - SEQUENCE_FORMING
    - ESTABLISHED
    - REACTION
    - RESUMING
    - WEAKENING
    - STRUCTURE_BREAK
    RESOLVED:
    - TERMINATED
    INVALIDATED:
    - INVALIDATED
  branch_transitions:
  - from_phase: INACTIVE
    to_phase: IMPULSE_CANDIDATE
    event: CAUSAL_DIRECTIONAL_IMPULSE
  - from_phase: IMPULSE_CANDIDATE
    to_phase: SEQUENCE_FORMING
    event: FIRST_COMPARABLE_SWING_PAIR
  - from_phase: SEQUENCE_FORMING
    to_phase: ESTABLISHED
    event: DESCENDING_SEQUENCE_READY
  - from_phase: ESTABLISHED
    to_phase: REACTION
    event: CAUSAL_REACTION_OBSERVED
  - from_phase: REACTION
    to_phase: RESUMING
    event: PRIOR_STRUCTURE_HOLDS
  - from_phase: RESUMING
    to_phase: ESTABLISHED
    event: NEW_DESCENDING_SEQUENCE_EVIDENCE
  - from_phase: ESTABLISHED
    to_phase: WEAKENING
    event: ORDER_OR_MOMENTUM_WEAKENS
  - from_phase: WEAKENING
    to_phase: STRUCTURE_BREAK
    event: DEFENDING_PIVOT_BREAK
  - from_phase: STRUCTURE_BREAK
    to_phase: TERMINATED
    event: TERMINATION_RECORDED
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - CONFIRMED_REACTION_HIGH_BREACH
    - DESCENDING_ORDER_BREAK
    - SCALE_RELATIVE_FORMATION_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The down-trend graph separates reaction, resumption, weakening, break, and termination, but
  its impulse and termination event tokens lack frozen causal definitions.
- 建议修改：
- Bind every edge to defined causal event predicates, including observable persistent-state
  termination.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_DOWN.PHASE.001
owner_decision: APPROVE_AS_IS
owner_rationale: Retain the proposed downtrend phases as the strict mirror of TREND_UP; exact causal predicates
  for the common lifecycle remain governed by the existing lifecycle revision requirement.
required_changes: []
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:41:13+08:00'
```

---

<a id="item-20"></a>

## 20. `MG1A.TREND_DOWN.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 trend_down 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal trend_down confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_down].confirmation_rules`
- `resolved_contract.families[family=trend_down].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- DESCENDING_SEQUENCE_READY
- LATEST_REACTION_HOLDS
- OPTIONAL_FILTERS_DO_NOT_DEFINE_TOPOLOGY
branch_transitions:
- from_phase: INACTIVE
  to_phase: IMPULSE_CANDIDATE
  event: CAUSAL_DIRECTIONAL_IMPULSE
- from_phase: IMPULSE_CANDIDATE
  to_phase: SEQUENCE_FORMING
  event: FIRST_COMPARABLE_SWING_PAIR
- from_phase: SEQUENCE_FORMING
  to_phase: ESTABLISHED
  event: DESCENDING_SEQUENCE_READY
- from_phase: ESTABLISHED
  to_phase: REACTION
  event: CAUSAL_REACTION_OBSERVED
- from_phase: REACTION
  to_phase: RESUMING
  event: PRIOR_STRUCTURE_HOLDS
- from_phase: RESUMING
  to_phase: ESTABLISHED
  event: NEW_DESCENDING_SEQUENCE_EVIDENCE
- from_phase: ESTABLISHED
  to_phase: WEAKENING
  event: ORDER_OR_MOMENTUM_WEAKENS
- from_phase: WEAKENING
  to_phase: STRUCTURE_BREAK
  event: DEFENDING_PIVOT_BREAK
- from_phase: STRUCTURE_BREAK
  to_phase: TERMINATED
  event: TERMINATION_RECORDED
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Descending readiness and reaction hold are named, but their conjunction, completeness
  requirement, and optional-filter UNKNOWN behavior are not defined.
- 建议修改：
- Add a tri-state confirmation truth table for descending readiness, reaction hold, completeness,
  and missing filters.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_DOWN.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Use the canonical mirrored rule in which a confirmed lower high followed by a closed-bar
  break below the prior low confirms the downtrend without waiting for the new low to become a confirmed
  pivot.
required_changes:
- Reference the same pinned market_structure callable and versioned effective parameter profile used by
  TREND_UP.
- Confirm a downtrend only after the H-L-LH pivots are causally available and a later closed bar closes
  strictly below the prior low under the shared configured break buffer.
- Build the prior-low zone from its confirmed pivot evidence and require Close to be strictly below prior_low_zone_low
  multiplied by one minus the shared versioned break_buffer_pct.
- Equality with the threshold and wick-only penetration must not confirm TREND_DOWN.
- Do not use volume or open interest as confirmation conditions.
- Preserve exact formula, equality, timing, and parameter mirror symmetry with TREND_UP.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:41:13+08:00'
```

---

<a id="item-21"></a>

## 21. `MG1A.TREND_DOWN.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 trend_down 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal trend_down invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_down].invalidation_rules`
- `resolved_contract.families[family=trend_down].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=trend_down].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- CONFIRMED_REACTION_HIGH_BREACH
- DESCENDING_ORDER_BREAK
- SCALE_RELATIVE_FORMATION_EXPIRY
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - CONFIRMED_REACTION_HIGH_BREACH
  - DESCENDING_ORDER_BREAK
  - SCALE_RELATIVE_FORMATION_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Reaction-high breach, order break, and expiry are listed, but pivot identity, normalized
  observation, expiry clock, precedence, and mirror-parameter independence are incomplete.
- 建议修改：
- Define pivot/breach/expiry semantics and event precedence with independently versioned downward
  parameters.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_DOWN.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Strictly mirror protected-HL invalidation by applying the shared break buffer outside
  the protected-LH zone.
required_changes:
- Build the protected-LH zone from the causally confirmed protected-high pivot and its closed bar.
- Invalidate TREND_DOWN only when Close is strictly above protected_lh_zone_high multiplied by one plus
  the shared versioned break_buffer_pct.
- Equality with the threshold and wick-only penetration must not invalidate TREND_DOWN.
- At the invalidating bar close, terminalize the active trend object as INVALIDATED and return the detector
  runtime state to neutral without rewriting prior trend history.
- Preserve the canonical continuation rule that replaces the protected LH only after a new external low,
  a later confirmed LH, and a closed-bar break below that external low.
- Do not use volume or open interest as invalidation conditions.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:42:33+08:00'
```

---

<a id="item-22"></a>

## 22. `MG1A.TREND_TRANSITION.STRUCTURE.001`

**要确认的问题**：是否同意冻结 trend_transition 的方向依据、锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze trend_transition side basis, anchors, topology, geometry, duration, price ratios, and prior
context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_transition].anchors`
- `resolved_contract.families[family=trend_transition].topology_rules`
- `resolved_contract.families[family=trend_transition].formula_contracts`
- `resolved_contract.families[family=trend_transition].instance_fields`
- `resolved_contract.families[family=trend_transition].object_class`
- `resolved_contract.families[family=trend_transition].orientation_binding`
- `resolved_contract.families[family=trend_transition].allowed_instance_orientations`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: trend_transition
object_class: LEVEL_EVENT
structural_orientation: null
orientation_binding: INSTANCE_BOUND
allowed_instance_orientations:
- UPWARD
- DOWNWARD
semantic_roles:
- TRANSITION_EVENT
persistent_state: false
scale_spec: INSTANCE_PARAMETER
anchors:
- role: prior_pattern_ref
  cardinality: ONE
  state_requirement: CONFIRMED
- role: prior_sequence_terminal_extreme
  cardinality: ONE
  state_requirement: CONFIRMED
- role: last_defending_pivot
  cardinality: ONE
  state_requirement: CONFIRMED
- role: structure_break
  cardinality: ONE
  state_requirement: CONFIRMED
- role: first_counter_pivot
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
- role: transition_retest
  cardinality: OPTIONAL
  state_requirement: CONFIRMED
topology_rules:
- PRIOR_ORDERED_TREND_REFERENCE_REQUIRED
- SHARED_DEFENDING_PIVOT_BREAK_EVENT
- COUNTER_DIRECTION_AVAILABLE_BEFORE_ORIENTATION_BIND
formula_contracts:
- name: break_displacement_to_prior_swing_ratio
  numerator: structure_break_displacement
  denominator: prior_swing_absolute_price_change
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: counter_swing_to_prior_impulse_ratio
  numerator: counter_swing_absolute_price_change
  denominator: prior_impulse_absolute_price_change
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
relationship_types:
- ORIGINATES_FROM
- CONFIRMED_BY
- INVALIDATED_BY
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：The LEVEL_EVENT requires prior trend identity and a shared defending-pivot break, preserves
  the prior object, and delays instance orientation until causal counter-direction evidence is
  available.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_TRANSITION.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Remove TREND_TRANSITION as an independent pattern family because the selected canonical
  market-structure algorithm defines only uptrend, downtrend, neutral, and their events.
required_changes:
- Remove TREND_TRANSITION from official pattern-family enums, object schemas, detector outputs, and training
  or evaluation label spaces.
- Do not create a transition pattern object, stable transition identity, or transition anchors.
- Preserve market_structure local mixed-structure descriptors only as diagnostic evidence; they must not
  emit an official pattern label.
- Represent the interval after trend invalidation using the detector's neutral runtime state.
- Preserve separately timed relations between the invalidated trend and any subsequently confirmed trend.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:44:51+08:00'
```

---

<a id="item-23"></a>

## 23. `MG1A.TREND_TRANSITION.PHASE.001`

**要确认的问题**：是否同意冻结 trend_transition 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze trend_transition phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_transition].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - INACTIVE
  - STRUCTURE_BREAK_OBSERVED
  - COUNTER_PIVOT_AVAILABLE
  - COUNTER_SEQUENCE_FORMING
  - REVERSE_TREND_CONFIRMED
  - EXPIRED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - INACTIVE
    ACTIVE:
    - STRUCTURE_BREAK_OBSERVED
    - COUNTER_PIVOT_AVAILABLE
    - COUNTER_SEQUENCE_FORMING
    RESOLVED:
    - REVERSE_TREND_CONFIRMED
    INVALIDATED:
    - EXPIRED
    - INVALIDATED
  branch_transitions:
  - from_phase: INACTIVE
    to_phase: STRUCTURE_BREAK_OBSERVED
    event: SHARED_DEFENDING_PIVOT_BREAK
  - from_phase: STRUCTURE_BREAK_OBSERVED
    to_phase: COUNTER_PIVOT_AVAILABLE
    event: CAUSAL_COUNTER_PIVOT
  - from_phase: COUNTER_PIVOT_AVAILABLE
    to_phase: COUNTER_SEQUENCE_FORMING
    event: COUNTER_SEQUENCE_EVIDENCE
  - from_phase: COUNTER_SEQUENCE_FORMING
    to_phase: REVERSE_TREND_CONFIRMED
    event: LINKED_REVERSE_TREND_ESTABLISHED
  - from_phase: STRUCTURE_BREAK_OBSERVED
    to_phase: EXPIRED
    event: SCALE_RELATIVE_EXPIRY
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - PRIOR_TREND_RESUMPTION
    - DEFENDING_PIVOT_RECOVERY
    - COUNTER_SEQUENCE_FAILURE
```

</details>

### AI 参考意见（无审批权）

- 建议：`ACCEPT`
- 置信度：`HIGH`
- 中文提示：AI 技术预审未提出阻断项；仍需负责人确认领域语义。
- 风险摘要：The object starts at the shared defending-pivot break, advances through counter pivot and
  sequence evidence, and resolves only through a separately linked reverse trend; the graph is
  reachable and terminal.
- 建议修改：
- 无
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_TRANSITION.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: An eliminated pattern family has no independent four-state lifecycle or phase contract.
required_changes:
- Remove every TREND_TRANSITION lifecycle and phase definition.
- Keep neutral as a market_structure runtime state rather than mapping it to an independent pattern lifecycle.
- Do not infer transition phases from elapsed neutral bars.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:44:51+08:00'
```

---

<a id="item-24"></a>

## 24. `MG1A.TREND_TRANSITION.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 trend_transition 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal trend_transition confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_transition].confirmation_rules`
- `resolved_contract.families[family=trend_transition].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- DEFENDING_PIVOT_BREAK_RECORDED
- COUNTER_SEQUENCE_EXPLICIT
- REVERSE_TREND_CONFIRMATION_IS_SEPARATE
branch_transitions:
- from_phase: INACTIVE
  to_phase: STRUCTURE_BREAK_OBSERVED
  event: SHARED_DEFENDING_PIVOT_BREAK
- from_phase: STRUCTURE_BREAK_OBSERVED
  to_phase: COUNTER_PIVOT_AVAILABLE
  event: CAUSAL_COUNTER_PIVOT
- from_phase: COUNTER_PIVOT_AVAILABLE
  to_phase: COUNTER_SEQUENCE_FORMING
  event: COUNTER_SEQUENCE_EVIDENCE
- from_phase: COUNTER_SEQUENCE_FORMING
  to_phase: REVERSE_TREND_CONFIRMED
  event: LINKED_REVERSE_TREND_ESTABLISHED
- from_phase: STRUCTURE_BREAK_OBSERVED
  to_phase: EXPIRED
  event: SCALE_RELATIVE_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Break, counter sequence, and linked reverse trend are separated, but the minimum confirmation
  stage and tri-state confirmation predicate are not selected.
- 建议修改：
- Select the minimum causal confirmation event and encode its TRUE/FALSE/UNKNOWN truth table without
  collapsing reverse-trend resolution into initial transition detection.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_TRANSITION.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: TREND_TRANSITION has no confirmation predicate after removal as an independent family.
required_changes:
- Remove transition confirmation fields, parameters, events, and labels.
- Record trend invalidation and later trend confirmation as separate canonical events with their own closed-bar
  timestamps and provenance.
- Do not treat neutral or mixed local structure as transition confirmation.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:44:51+08:00'
```

---

<a id="item-25"></a>

## 25. `MG1A.TREND_TRANSITION.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 trend_transition 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal trend_transition invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=trend_transition].invalidation_rules`
- `resolved_contract.families[family=trend_transition].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=trend_transition].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- PRIOR_TREND_RESUMPTION
- DEFENDING_PIVOT_RECOVERY
- COUNTER_SEQUENCE_FAILURE
- SCALE_RELATIVE_EXPIRY
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - PRIOR_TREND_RESUMPTION
  - DEFENDING_PIVOT_RECOVERY
  - COUNTER_SEQUENCE_FAILURE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Resumption, pivot recovery, counter-sequence failure, and expiry are present, but
  recovery/resumption identity and precedence are undefined and expiry is reachable only from the
  first active phase.
- 建议修改：
- Define event identity/precedence and add scale-relative expiry transitions from every applicable
  active phase.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.TREND_TRANSITION.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: TREND_TRANSITION has no invalidation or expiry predicate after removal as an independent
  family.
required_changes:
- Remove transition invalidation, expiry, timeout, and terminal-reason fields and parameters.
- A later trend confirmation ends the neutral runtime interval but does not complete or invalidate a transition
  object.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:44:51+08:00'
```

---

<a id="item-54"></a>

## 54. `MG1A.RANGE.STRUCTURE.001`

**要确认的问题**：是否同意冻结 range 的锚点、拓扑、几何、持续时间、价格比率及先验背景？

<details><summary>查看英文原始问题</summary>

Freeze range anchors, topology, geometry, duration, price ratios, and prior context.

</details>

**当前定义定位**：
- `resolved_contract.families[family=range].anchors`
- `resolved_contract.families[family=range].topology_rules`
- `resolved_contract.families[family=range].formula_contracts`
- `resolved_contract.families[family=range].instance_fields`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family: range
object_class: STRUCTURAL_STATE
structural_orientation: NEUTRAL
orientation_binding: FIXED
allowed_instance_orientations:
- NEUTRAL
semantic_roles:
- RANGE_STATE
persistent_state: true
range_geometry: HORIZONTAL_ZONE_PAIR
scale_spec: INSTANCE_PARAMETER
anchors:
- role: upper_boundary_touch
  cardinality: ORDERED_MANY
  state_requirement: CONFIRMED
- role: lower_boundary_touch
  cardinality: ORDERED_MANY
  state_requirement: CONFIRMED
- role: upper_boundary_zone
  cardinality: ONE
  state_requirement: DERIVED_FROM_CONFIRMED
- role: lower_boundary_zone
  cardinality: ONE
  state_requirement: DERIVED_FROM_CONFIRMED
- role: range_midline
  cardinality: DERIVED
  state_requirement: DERIVED_FROM_CONFIRMED
- role: current_frontier
  cardinality: ONE
  state_requirement: PROVISIONAL_ALLOWED_RUNTIME_ONLY
topology_rules:
- HORIZONTAL_UPPER_AND_LOWER_ZONES
- ALTERNATING_CONFIRMED_TOUCH_SEQUENCE
- CONTAINMENT_IS_PREFIX_ONLY
- DIRECTION_CONFLICT_ALONE_IS_INSUFFICIENT
formula_contracts:
- name: range_width_to_causal_price_scale
  numerator: upper_zone_reference_minus_lower_zone_reference
  denominator: causal_price_scale
  sign: NONNEGATIVE
  unit: NORMALIZED_PRICE_MULTIPLE
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: upper_anchor_dispersion_to_range_width
  numerator: upper_anchor_dispersion
  denominator: upper_zone_reference_minus_lower_zone_reference
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
- name: lower_anchor_dispersion_to_range_width
  numerator: lower_anchor_dispersion
  denominator: upper_zone_reference_minus_lower_zone_reference
  sign: NONNEGATIVE
  unit: DIMENSIONLESS_RATIO
  missingness: UNKNOWN_WITH_MASK
  clipping_policy: TRAIN_FROZEN_CLIP_POLICY
  fit_boundary: TRAIN_PARTITION_ONLY
relationship_types:
- CONTAINS
- CONFIRMED_BY
- INVALIDATED_BY
- SAME_STRUCTURE_DIFFERENT_SCALE
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：A persistent neutral horizontal zone pair with alternating touches is encoded, but readiness,
  stable-boundary evidence, and zone tolerance/version policy are not defined.
- 建议修改：
- Define alternating-sequence readiness, stable-boundary evidence, and versioned horizontal-
  zone/tolerance policy references.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.RANGE.STRUCTURE.001
owner_decision: REQUEST_REVISION
owner_rationale: Preserve both box ranges and long sideways ranges as distinct official range types, each
  generated by its own selected versioned detector.
required_changes:
- Add BOX_RANGE and SIDEWAYS_RANGE as closed range_type values under the range family.
- Bind BOX_RANGE to the selected box_breakout_detector version, callable, file hash, and timeframe-specific
  parameter profile.
- Bind SIDEWAYS_RANGE to the selected sideways_detector version, callable, file hash, and timeframe-specific
  parameter profile.
- Give each detected range its own stable identity and never merge or overwrite overlapping BOX_RANGE
  and SIDEWAYS_RANGE objects.
- Normalize both detector outputs into the common range object while preserving detector-specific structure
  evidence.
- Apply the common closed-bar, active-contract, key-zone, provenance, and lifecycle contracts to both
  types.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:18:58+08:00'
```

---

<a id="item-55"></a>

## 55. `MG1A.RANGE.PHASE.001`

**要确认的问题**：是否同意冻结 range 的阶段证据及完成语义？

<details><summary>查看英文原始问题</summary>

Freeze range phase evidence and completion semantics.

</details>

**当前定义定位**：
- `resolved_contract.families[family=range].phase_contract`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
phase_contract:
  family_phases:
  - INACTIVE
  - FORMING
  - ESTABLISHED
  - BOUNDARY_TEST
  - FALSE_BREAK
  - ACCEPTED_BREAK
  - TERMINATED
  - INVALIDATED
  common_mapping:
    INACTIVE:
    - INACTIVE
    ACTIVE:
    - FORMING
    - ESTABLISHED
    - BOUNDARY_TEST
    - FALSE_BREAK
    - ACCEPTED_BREAK
    RESOLVED:
    - TERMINATED
    INVALIDATED:
    - INVALIDATED
  branch_transitions:
  - from_phase: INACTIVE
    to_phase: FORMING
    event: BOTH_BOUNDARY_CANDIDATES_AVAILABLE
  - from_phase: FORMING
    to_phase: ESTABLISHED
    event: ALTERNATING_BOUNDARY_SEQUENCE_READY
  - from_phase: ESTABLISHED
    to_phase: BOUNDARY_TEST
    event: CAUSAL_BOUNDARY_INTERACTION
  - from_phase: BOUNDARY_TEST
    to_phase: FALSE_BREAK
    event: LINKED_FAILED_BREAKOUT_REENTRY
  - from_phase: FALSE_BREAK
    to_phase: ESTABLISHED
    event: CONTAINMENT_REESTABLISHED
  - from_phase: BOUNDARY_TEST
    to_phase: ACCEPTED_BREAK
    event: LINKED_ACCEPTED_BREAKOUT
  - from_phase: ACCEPTED_BREAK
    to_phase: TERMINATED
    event: RANGE_TERMINATION_RECORDED
  invalidation_transitions:
  - from_phases: ALL_ACTIVE_FAMILY_PHASES
    to_phase: INVALIDATED
    events:
    - MATERIAL_BOUNDARY_DRIFT
    - SCALE_RELATIVE_DISSOLUTION_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The graph keeps ACCEPTED_BREAK active and defers resolution to TERMINATED, but
  RANGE_TERMINATION_RECORDED is administrative rather than a defined causal market predicate, so
  phase evidence and completion are not fully frozen.
- 建议修改：
- Define the causal termination/dissolution predicate, timestamp, shared accepted-break identity,
  and precedence with invalidation.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.RANGE.PHASE.001
owner_decision: REQUEST_REVISION
owner_rationale: Preserve one stable source-range identity through failed attempts, but terminalize it
  after a successful breakout survives its failure window.
required_changes:
- UNACTIVATED means the selected BOX_RANGE or SIDEWAYS_RANGE detector has not causally confirmed the range.
- ACTIVE begins on detector-specific range confirmation and remains active during boundary tests, outward
  attempts, and the full failed-breakout window.
- If the attempt fails inside its window, keep the source range ACTIVE under the same identity.
- COMPLETED begins when a confirmed breakout's failure window closes without canonical reentry.
- INVALIDATED is reserved for contract roll or data and algorithm replacement; the selected detectors
  currently define no market-driven structural expiry for an already confirmed range.
- Do not infer range expiry from SIDEWAYS_RANGE maximum formation-window or post-breakout tracking parameters.
- LATE_RECLAIM must not reactivate a completed range; detecting a later range requires a new identity.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:47:10+08:00'
```

---

<a id="item-56"></a>

## 56. `MG1A.RANGE.CONFIRMATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 range 确认规则及可选过滤条件？

<details><summary>查看英文原始问题</summary>

Freeze causal range confirmation and optional filters.

</details>

**当前定义定位**：
- `resolved_contract.families[family=range].confirmation_rules`
- `resolved_contract.families[family=range].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
confirmation_rules:
- BOTH_ZONES_VERSIONED
- ALTERNATING_SEQUENCE_READY
- CONTAINED_PREFIX_REQUIRED
- OPTIONAL_FILTERS_DO_NOT_DEFINE_RANGE
branch_transitions:
- from_phase: INACTIVE
  to_phase: FORMING
  event: BOTH_BOUNDARY_CANDIDATES_AVAILABLE
- from_phase: FORMING
  to_phase: ESTABLISHED
  event: ALTERNATING_BOUNDARY_SEQUENCE_READY
- from_phase: ESTABLISHED
  to_phase: BOUNDARY_TEST
  event: CAUSAL_BOUNDARY_INTERACTION
- from_phase: BOUNDARY_TEST
  to_phase: FALSE_BREAK
  event: LINKED_FAILED_BREAKOUT_REENTRY
- from_phase: FALSE_BREAK
  to_phase: ESTABLISHED
  event: CONTAINMENT_REESTABLISHED
- from_phase: BOUNDARY_TEST
  to_phase: ACCEPTED_BREAK
  event: LINKED_ACCEPTED_BREAKOUT
- from_phase: ACCEPTED_BREAK
  to_phase: TERMINATED
  event: RANGE_TERMINATION_RECORDED
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`MEDIUM`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The current confirmation rules repeat zone availability, alternating topology, and
  containment, contrary to the v1 requirement for independent causal confirmation; the defect is
  repairable by a narrow contract revision.
- 建议修改：
- Select an independent minimum causal confirmation predicate and encode its tri-state truth table
  without allowing optional filters to define the range.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.RANGE.CONFIRMATION.001
owner_decision: REQUEST_REVISION
owner_rationale: Confirm BOX_RANGE and SIDEWAYS_RANGE with their separate selected detectors and versioned
  timeframe profiles; never mix their formation predicates.
required_changes:
- Confirm BOX_RANGE only when four causally available normalized pivots form H-L-H-L or L-H-L-H and pass
  the selected detector's high-touch, low-touch, boundary-separation, minimum-width, minimum-duration,
  and pre-confirmation containment predicates.
- Preserve the current BOX_RANGE baseline defaults of left and right 2, touch tolerance 0.005, minimum
  width 0.02, and minimum duration 10 bars until each timeframe profile is versioned or reviewed.
- Confirm SIDEWAYS_RANGE only when the current closed-bar prefix contains qualifying upper and lower confirmed-pivot
  clusters and passes its selected detector's duration, width, close-containment, consecutive-outside-close,
  drift, and net-change predicates.
- Preserve the current SIDEWAYS_RANGE daily baseline defaults of 120 to 600 bars, three touches per side,
  cluster tolerance 0.02, width 0.08 to 0.60, close containment 0.90, maximum three consecutive outside
  closes, drift-to-width 0.35, and net-change-to-width 0.50; do not silently reuse the daily profile on
  another timeframe.
- Use only pivots causally confirmed by the current EOB and record range_type, detector version and hash,
  effective timeframe profile, anchor availability, and confirmation time.
- Volume and open interest remain observational and must not gate range confirmation.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:50:31+08:00'
```

---

<a id="item-57"></a>

## 57. `MG1A.RANGE.INVALIDATION.001`

**要确认的问题**：是否同意冻结满足因果约束的 range 失效及到期规则？

<details><summary>查看英文原始问题</summary>

Freeze causal range invalidation and expiry.

</details>

**当前定义定位**：
- `resolved_contract.families[family=range].invalidation_rules`
- `resolved_contract.families[family=range].phase_contract.invalidation_transitions`
- `resolved_contract.families[family=range].phase_contract.branch_transitions`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
invalidation_rules:
- MATERIAL_BOUNDARY_DRIFT
- SCALE_RELATIVE_DISSOLUTION_EXPIRY
invalidation_transitions:
- from_phases: ALL_ACTIVE_FAMILY_PHASES
  to_phase: INVALIDATED
  events:
  - MATERIAL_BOUNDARY_DRIFT
  - SCALE_RELATIVE_DISSOLUTION_EXPIRY
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：Failed/accepted boundary events and drift/dissolution are named, but shared event identity,
  drift-versus-break precedence, and dissolution/expiry semantics are incomplete.
- 建议修改：
- Bind one shared break event across range and breakout objects, then define drift, dissolution,
  expiry, and event precedence.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A.RANGE.INVALIDATION.001
owner_decision: REQUEST_REVISION
owner_rationale: A failed breakout attempt does not by itself invalidate the source BOX_RANGE or SIDEWAYS_RANGE.
required_changes:
- Keep a still-valid source range active under the same stable identity after a failed breakout attempt.
- Remove reactivated from the range lifecycle; the range was never terminated by the failed attempt.
- Do not invent a market-driven structural invalidation or expiry rule; neither selected detector currently
  defines one for an already confirmed range.
- Do not treat SIDEWAYS_RANGE max_sideways_bars or max_post_breakout_tracking_bars as range expiry parameters.
- Preserve each failed attempt as a related object without merging it into the source range history.
- Complete the source range when a confirmed breakout survives its failure window.
- Record any later return as LATE_RECLAIM without reactivating the completed source range; a later range
  must receive a new identity after canonical detection.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T21:23:36+08:00'
```

---
