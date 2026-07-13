# 整体一致性确认

[返回审核总览](README.md)

本章共 1 条。AI 内容仅作参考，最终决定由 `PROJECT_OWNER` 作出。

<a id="item-70"></a>

## 70. `MG1A1.COMMON.FAMILY_REVISIONS.001`

**要确认的问题**：是否同意冻结 MG1-A.1 针对各形态族的修订？

<details><summary>查看英文原始问题</summary>

Freeze the family-specific MG1-A.1 revisions

</details>

**当前定义定位**：
- `resolved_contract.families[*]`
- `resolved_contract.object_model`
- `resolved_contract.common_lifecycle`

<details><summary>查看当前 v2 选定字段原文（EXACT）</summary>

```yaml
family_review_index:
- family: trend_up
  object_class: STRUCTURAL_STATE
  structural_orientation: UPWARD
  assumption_ids:
  - MG1A.TREND_UP.STRUCTURE.001
  - MG1A.TREND_UP.PHASE.001
  - MG1A.TREND_UP.CONFIRMATION.001
  - MG1A.TREND_UP.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: trend_down
  object_class: STRUCTURAL_STATE
  structural_orientation: DOWNWARD
  assumption_ids:
  - MG1A.TREND_DOWN.STRUCTURE.001
  - MG1A.TREND_DOWN.PHASE.001
  - MG1A.TREND_DOWN.CONFIRMATION.001
  - MG1A.TREND_DOWN.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: trend_transition
  object_class: LEVEL_EVENT
  structural_orientation: null
  assumption_ids:
  - MG1A.TREND_TRANSITION.STRUCTURE.001
  - MG1A.TREND_TRANSITION.PHASE.001
  - MG1A.TREND_TRANSITION.CONFIRMATION.001
  - MG1A.TREND_TRANSITION.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: double_top
  object_class: MORPHOLOGY
  structural_orientation: UPPER
  assumption_ids:
  - MG1A.DOUBLE_TOP.STRUCTURE.001
  - MG1A.DOUBLE_TOP.PHASE.001
  - MG1A.DOUBLE_TOP.CONFIRMATION.001
  - MG1A.DOUBLE_TOP.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: double_bottom
  object_class: MORPHOLOGY
  structural_orientation: LOWER
  assumption_ids:
  - MG1A.DOUBLE_BOTTOM.STRUCTURE.001
  - MG1A.DOUBLE_BOTTOM.PHASE.001
  - MG1A.DOUBLE_BOTTOM.CONFIRMATION.001
  - MG1A.DOUBLE_BOTTOM.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: head_shoulders_top
  object_class: MORPHOLOGY
  structural_orientation: UPPER
  assumption_ids:
  - MG1A.HEAD_SHOULDERS_TOP.STRUCTURE.001
  - MG1A.HEAD_SHOULDERS_TOP.PHASE.001
  - MG1A.HEAD_SHOULDERS_TOP.CONFIRMATION.001
  - MG1A.HEAD_SHOULDERS_TOP.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: inverse_head_shoulders
  object_class: MORPHOLOGY
  structural_orientation: LOWER
  assumption_ids:
  - MG1A.INVERSE_HEAD_SHOULDERS.STRUCTURE.001
  - MG1A.INVERSE_HEAD_SHOULDERS.PHASE.001
  - MG1A.INVERSE_HEAD_SHOULDERS.CONFIRMATION.001
  - MG1A.INVERSE_HEAD_SHOULDERS.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: breakout
  object_class: LEVEL_EVENT
  structural_orientation: null
  assumption_ids:
  - MG1A.BREAKOUT.STRUCTURE.001
  - MG1A.BREAKOUT.PHASE.001
  - MG1A.BREAKOUT.CONFIRMATION.001
  - MG1A.BREAKOUT.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: failed_breakout
  object_class: LEVEL_EVENT
  structural_orientation: null
  assumption_ids:
  - MG1A.FAILED_BREAKOUT.STRUCTURE.001
  - MG1A.FAILED_BREAKOUT.PHASE.001
  - MG1A.FAILED_BREAKOUT.CONFIRMATION.001
  - MG1A.FAILED_BREAKOUT.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: retest
  object_class: LEVEL_EVENT
  structural_orientation: null
  assumption_ids:
  - MG1A.RETEST.STRUCTURE.001
  - MG1A.RETEST.PHASE.001
  - MG1A.RETEST.CONFIRMATION.001
  - MG1A.RETEST.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: range
  object_class: STRUCTURAL_STATE
  structural_orientation: NEUTRAL
  assumption_ids:
  - MG1A.RANGE.STRUCTURE.001
  - MG1A.RANGE.PHASE.001
  - MG1A.RANGE.CONFIRMATION.001
  - MG1A.RANGE.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
- family: support_resistance_conversion
  object_class: STRUCTURAL_RELATION
  structural_orientation: null
  assumption_ids:
  - MG1A.SUPPORT_RESISTANCE_CONVERSION.STRUCTURE.001
  - MG1A.SUPPORT_RESISTANCE_CONVERSION.PHASE.001
  - MG1A.SUPPORT_RESISTANCE_CONVERSION.CONFIRMATION.001
  - MG1A.SUPPORT_RESISTANCE_CONVERSION.INVALIDATION.001
  - MG1A1.COMMON.FAMILY_REVISIONS.001
```

</details>

### AI 参考意见（无审批权）

- 建议：`REVISE`
- 置信度：`HIGH`
- 中文提示：AI 技术预审认为当前定义仍有缺口；请重点检查英文风险摘要和建议修改。
- 风险摘要：The P0 object, identity, morphology/effect, and major family distinctions are encoded, but
  family structure policies, graph coverage, confirmation truth tables, invalidation precedence, and
  range confirmation remain incomplete.
- 建议修改：
- Apply the minimal per-family changes recorded by the inherited assumptions, then re-run structural
  validation and re-review this aggregate assumption.
- 恢复审核所需证据：
- 无

### 项目负责人决定

以下块来自机器台账；正式结果以 `review.yaml` 为准。

```yaml
assumption_id: MG1A1.COMMON.FAMILY_REVISIONS.001
owner_decision: REQUEST_REVISION
owner_rationale: The current immutable v2 still contains removed families and superseded contracts, so
  all approved owner decisions must be implemented in a new ontology version before final freeze review.
required_changes:
- Generate a new immutable ontology version from all reviewed MG1-A owner decisions; preserve v2 and record
  explicit supersession and assumption lineage.
- Remove TREND_TRANSITION, RETEST, and SUPPORT_RESISTANCE_CONVERSION as independent formal families.
- Retain exactly trend_up, trend_down, range, double_top, double_bottom, head_shoulders_top, inverse_head_shoulders,
  breakout, and failed_breakout with their reviewed object-class mappings.
- Bind every retained family to its selected canonical algorithm version and source hash, effective scale-specific
  parameter-profile version and hash, raw-price basis, closed-bar predicates, lifecycle events, anchors,
  zones, and relationships.
- Apply all reviewed mirror, range subtype, failed-breakout, roll-boundary, label provenance, formula,
  scale, review, and append-only identity decisions consistently across contracts.
- Update schemas, registries, validators, serializers, generated review documents, and CPU-only tests;
  reject stale removed-family references and hash mismatches.
- Present the validated exact revised contract and content hash to PROJECT_OWNER for a new final review
  and signoff before MG1-B begins.
- Do not run training, GPU workloads, normalization fitting, or test-set parameter tuning during this
  revision.
evidence_needed_to_resume: []
replacement_proposition: []
reviewed_at: '2026-07-13T22:44:42+08:00'
```

---
