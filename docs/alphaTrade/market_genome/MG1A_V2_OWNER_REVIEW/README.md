# MG1-A.1 项目负责人审核工作簿

本目录是人工审核入口。正式机器记录位于
`configs/market_genome/reviews/mg1a_v2_owner/review.yaml`。

## 当前状态

- 状态：`AWAITING_OWNER_SIGNOFF`
- 已处理：`70/70`
- 通过当前文本：`8`
- 要求修改：`61`
- 延后：`1`
- 否决：`0`
- 缺失合同误批准：`0`
- 可进入 MG1-B freeze：`false`

## 审核权限

- 本次范围仅为 `MG1-A ontology draft approval`。
- `PROJECT_OWNER` 是唯一终审人，不要求第二审核者，不设置分歧裁决者。
- AI 双审仅提供风险提示，没有审批权；负责人可以覆盖其意见。
- 修订版必须取消后续标签治理中的强制双审；正式标签由版本化算法生成，人工抽查可选。
- 审核通过只表示可以启动 MG1-B freeze；当前 v2 文件仍是未冻结草案。
- AI/Agent 只能转录负责人明确给出的决定，不得代替负责人审核或签署。

## 决定含义

| 决定 | 含义 | 必填内容 |
|---|---|---|
| `APPROVE_AS_IS` | 批准当前精确 v2 hash 对应文本 | 覆盖 AI 非 ACCEPT 时填写理由 |
| `REQUEST_REVISION` | 修改后再审 | 理由、`required_changes` |
| `DEFER` | 等待外部证据 | 理由、`evidence_needed_to_resume` |
| `REJECT` | 否决当前命题 | 理由、`replacement_proposition` |
| `PENDING` | 尚未审核 | 保持其他字段为空 |

只有 70 条全部 `APPROVE_AS_IS`，并由同一 `PROJECT_OWNER` 另行最终签署，才会得到
`APPROVED_FOR_MG1B_FREEZE`。任何其他决定都保持 gate 阻塞。

## 先处理的硬阻塞

- [`MG1A.COMMON.ROLL.001`](01_common_contracts.md#item-07)：v2 缺少 `roll_contract`。
- [`MG1A.COMMON.ZONE.001`](01_common_contracts.md#item-11)：v2 缺少通用 `zone_contract`。

这两条不能选择 `APPROVE_AS_IS`；必须要求修改、延后或拒绝。

## 审核操作

1. 按章节阅读中文问题、当前 v2 摘录和 AI 风险提示。
2. 直接在对话中按 `assumption_id + 决定 + 理由/动作` 给出结论；工具只转录显式决定。
3. 也可以直接编辑机器台账；非 `PENDING` 项必须填写 RFC3339 `reviewed_at`。
4. Markdown 是生成视图，不要直接编辑；台账更新后运行下列 CPU-only 命令刷新。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m alphatrade.market_genome.ontology.owner_review --refresh --render --strict
```

5. 70 条全部处理后，由同一 `PROJECT_OWNER` 填写 `owner_signoff`。

若更新进程异常中断并遗留 `.review.yaml.lock`，先确认原进程已结束，再运行
`python -m alphatrade.market_genome.ontology.owner_review --recover-stale-lock`。

## 目标绑定

- Git commit：`0d378954d49b6467d05fdef152c522e94b660978`
- v2 file SHA-256：`cd569c19a73768b19b8f9ded2b570dc3fc93796a87b97ecd27becdd32ba5978e`
- resolved contract SHA-256：`ad3228a608998cc0823c440af0013b12658c9f8dcda4b858ea719be879389ebd`
- AI decisions SHA-256：`01a9c889c94ac47a9707bee1cc49d9b09b0d412eb5632f140fa39bfeeabc15d1`

## 分组

- [公共合同](01_common_contracts.md)：21 条
- [趋势与区间结构](02_trend_range.md)：16 条
- [反转形态](03_reversal_morphologies.md)：16 条
- [突破、回踩与角色转换](04_break_retest_conversion.md)：16 条
- [整体一致性确认](05_final_integration.md)：1 条

## AI 参考分布

- `ACCEPT`：13
- `REVISE`：53
- `DEFER`：4
- `REJECT`：0

## 70 条索引

| # | Assumption | 分组 | AI | Owner | 标记 |
|---:|---|---|---|---|---|
| 01 | [`MG1A.COMMON.LIFECYCLE.001`](01_common_contracts.md#item-01) | COMMON | REVISE | REQUEST_REVISION | - |
| 02 | [`MG1A.COMMON.SCALE.001`](01_common_contracts.md#item-02) | COMMON | DEFER | REQUEST_REVISION | - |
| 03 | [`MG1A.COMMON.NORMALIZATION.001`](01_common_contracts.md#item-03) | COMMON | DEFER | DEFER | - |
| 04 | [`MG1A.COMMON.SWING_CAUSALITY.001`](01_common_contracts.md#item-04) | COMMON | REVISE | APPROVE_AS_IS | - |
| 05 | [`MG1A.COMMON.PARTIAL_BAR.001`](01_common_contracts.md#item-05) | COMMON | ACCEPT | APPROVE_AS_IS | - |
| 06 | [`MG1A.COMMON.SESSION.001`](01_common_contracts.md#item-06) | COMMON | DEFER | APPROVE_AS_IS | - |
| 07 | [`MG1A.COMMON.ROLL.001`](01_common_contracts.md#item-07) | COMMON | REVISE | REQUEST_REVISION | MISSING_CONTRACT |
| 08 | [`MG1A.COMMON.PROVENANCE.001`](01_common_contracts.md#item-08) | COMMON | REVISE | REQUEST_REVISION | - |
| 09 | [`MG1A.COMMON.MIRROR.001`](01_common_contracts.md#item-09) | COMMON | REVISE | REQUEST_REVISION | - |
| 10 | [`MG1A.COMMON.OVERLAP.001`](01_common_contracts.md#item-10) | COMMON | REVISE | APPROVE_AS_IS | - |
| 11 | [`MG1A.COMMON.ZONE.001`](01_common_contracts.md#item-11) | COMMON | REVISE | REQUEST_REVISION | MISSING_CONTRACT |
| 12 | [`MG1A.COMMON.FILTER_MISSINGNESS.001`](01_common_contracts.md#item-12) | COMMON | REVISE | REQUEST_REVISION | - |
| 13 | [`MG1A.COMMON.LABEL_TIMING.001`](01_common_contracts.md#item-13) | COMMON | REVISE | APPROVE_AS_IS | - |
| 14 | [`MG1A.TREND_UP.STRUCTURE.001`](02_trend_range.md#item-14) | TREND_UP | REVISE | REQUEST_REVISION | - |
| 15 | [`MG1A.TREND_UP.PHASE.001`](02_trend_range.md#item-15) | TREND_UP | REVISE | APPROVE_AS_IS | - |
| 16 | [`MG1A.TREND_UP.CONFIRMATION.001`](02_trend_range.md#item-16) | TREND_UP | REVISE | REQUEST_REVISION | - |
| 17 | [`MG1A.TREND_UP.INVALIDATION.001`](02_trend_range.md#item-17) | TREND_UP | REVISE | REQUEST_REVISION | - |
| 18 | [`MG1A.TREND_DOWN.STRUCTURE.001`](02_trend_range.md#item-18) | TREND_DOWN | REVISE | REQUEST_REVISION | - |
| 19 | [`MG1A.TREND_DOWN.PHASE.001`](02_trend_range.md#item-19) | TREND_DOWN | REVISE | APPROVE_AS_IS | - |
| 20 | [`MG1A.TREND_DOWN.CONFIRMATION.001`](02_trend_range.md#item-20) | TREND_DOWN | REVISE | REQUEST_REVISION | - |
| 21 | [`MG1A.TREND_DOWN.INVALIDATION.001`](02_trend_range.md#item-21) | TREND_DOWN | REVISE | REQUEST_REVISION | - |
| 22 | [`MG1A.TREND_TRANSITION.STRUCTURE.001`](02_trend_range.md#item-22) | TREND_TRANSITION | ACCEPT | REQUEST_REVISION | - |
| 23 | [`MG1A.TREND_TRANSITION.PHASE.001`](02_trend_range.md#item-23) | TREND_TRANSITION | ACCEPT | REQUEST_REVISION | - |
| 24 | [`MG1A.TREND_TRANSITION.CONFIRMATION.001`](02_trend_range.md#item-24) | TREND_TRANSITION | REVISE | REQUEST_REVISION | - |
| 25 | [`MG1A.TREND_TRANSITION.INVALIDATION.001`](02_trend_range.md#item-25) | TREND_TRANSITION | REVISE | REQUEST_REVISION | - |
| 26 | [`MG1A.DOUBLE_TOP.STRUCTURE.001`](03_reversal_morphologies.md#item-26) | DOUBLE_TOP | REVISE | REQUEST_REVISION | - |
| 27 | [`MG1A.DOUBLE_TOP.PHASE.001`](03_reversal_morphologies.md#item-27) | DOUBLE_TOP | ACCEPT | REQUEST_REVISION | - |
| 28 | [`MG1A.DOUBLE_TOP.CONFIRMATION.001`](03_reversal_morphologies.md#item-28) | DOUBLE_TOP | REVISE | REQUEST_REVISION | - |
| 29 | [`MG1A.DOUBLE_TOP.INVALIDATION.001`](03_reversal_morphologies.md#item-29) | DOUBLE_TOP | REVISE | REQUEST_REVISION | - |
| 30 | [`MG1A.DOUBLE_BOTTOM.STRUCTURE.001`](03_reversal_morphologies.md#item-30) | DOUBLE_BOTTOM | REVISE | REQUEST_REVISION | - |
| 31 | [`MG1A.DOUBLE_BOTTOM.PHASE.001`](03_reversal_morphologies.md#item-31) | DOUBLE_BOTTOM | ACCEPT | REQUEST_REVISION | - |
| 32 | [`MG1A.DOUBLE_BOTTOM.CONFIRMATION.001`](03_reversal_morphologies.md#item-32) | DOUBLE_BOTTOM | REVISE | REQUEST_REVISION | - |
| 33 | [`MG1A.DOUBLE_BOTTOM.INVALIDATION.001`](03_reversal_morphologies.md#item-33) | DOUBLE_BOTTOM | REVISE | REQUEST_REVISION | - |
| 34 | [`MG1A.HEAD_SHOULDERS_TOP.STRUCTURE.001`](03_reversal_morphologies.md#item-34) | HEAD_SHOULDERS_TOP | REVISE | REQUEST_REVISION | - |
| 35 | [`MG1A.HEAD_SHOULDERS_TOP.PHASE.001`](03_reversal_morphologies.md#item-35) | HEAD_SHOULDERS_TOP | ACCEPT | REQUEST_REVISION | - |
| 36 | [`MG1A.HEAD_SHOULDERS_TOP.CONFIRMATION.001`](03_reversal_morphologies.md#item-36) | HEAD_SHOULDERS_TOP | REVISE | REQUEST_REVISION | - |
| 37 | [`MG1A.HEAD_SHOULDERS_TOP.INVALIDATION.001`](03_reversal_morphologies.md#item-37) | HEAD_SHOULDERS_TOP | REVISE | REQUEST_REVISION | - |
| 38 | [`MG1A.INVERSE_HEAD_SHOULDERS.STRUCTURE.001`](03_reversal_morphologies.md#item-38) | INVERSE_HEAD_SHOULDERS | REVISE | REQUEST_REVISION | - |
| 39 | [`MG1A.INVERSE_HEAD_SHOULDERS.PHASE.001`](03_reversal_morphologies.md#item-39) | INVERSE_HEAD_SHOULDERS | ACCEPT | REQUEST_REVISION | - |
| 40 | [`MG1A.INVERSE_HEAD_SHOULDERS.CONFIRMATION.001`](03_reversal_morphologies.md#item-40) | INVERSE_HEAD_SHOULDERS | REVISE | REQUEST_REVISION | - |
| 41 | [`MG1A.INVERSE_HEAD_SHOULDERS.INVALIDATION.001`](03_reversal_morphologies.md#item-41) | INVERSE_HEAD_SHOULDERS | REVISE | REQUEST_REVISION | - |
| 42 | [`MG1A.BREAKOUT.STRUCTURE.001`](04_break_retest_conversion.md#item-42) | BREAKOUT | REVISE | REQUEST_REVISION | - |
| 43 | [`MG1A.BREAKOUT.PHASE.001`](04_break_retest_conversion.md#item-43) | BREAKOUT | REVISE | REQUEST_REVISION | - |
| 44 | [`MG1A.BREAKOUT.CONFIRMATION.001`](04_break_retest_conversion.md#item-44) | BREAKOUT | REVISE | REQUEST_REVISION | - |
| 45 | [`MG1A.BREAKOUT.INVALIDATION.001`](04_break_retest_conversion.md#item-45) | BREAKOUT | REVISE | REQUEST_REVISION | - |
| 46 | [`MG1A.FAILED_BREAKOUT.STRUCTURE.001`](04_break_retest_conversion.md#item-46) | FAILED_BREAKOUT | ACCEPT | REQUEST_REVISION | - |
| 47 | [`MG1A.FAILED_BREAKOUT.PHASE.001`](04_break_retest_conversion.md#item-47) | FAILED_BREAKOUT | REVISE | REQUEST_REVISION | - |
| 48 | [`MG1A.FAILED_BREAKOUT.CONFIRMATION.001`](04_break_retest_conversion.md#item-48) | FAILED_BREAKOUT | REVISE | REQUEST_REVISION | - |
| 49 | [`MG1A.FAILED_BREAKOUT.INVALIDATION.001`](04_break_retest_conversion.md#item-49) | FAILED_BREAKOUT | REVISE | REQUEST_REVISION | - |
| 50 | [`MG1A.RETEST.STRUCTURE.001`](04_break_retest_conversion.md#item-50) | RETEST | REVISE | REQUEST_REVISION | - |
| 51 | [`MG1A.RETEST.PHASE.001`](04_break_retest_conversion.md#item-51) | RETEST | REVISE | REQUEST_REVISION | - |
| 52 | [`MG1A.RETEST.CONFIRMATION.001`](04_break_retest_conversion.md#item-52) | RETEST | REVISE | REQUEST_REVISION | - |
| 53 | [`MG1A.RETEST.INVALIDATION.001`](04_break_retest_conversion.md#item-53) | RETEST | REVISE | REQUEST_REVISION | - |
| 54 | [`MG1A.RANGE.STRUCTURE.001`](02_trend_range.md#item-54) | RANGE | REVISE | REQUEST_REVISION | - |
| 55 | [`MG1A.RANGE.PHASE.001`](02_trend_range.md#item-55) | RANGE | REVISE | REQUEST_REVISION | - |
| 56 | [`MG1A.RANGE.CONFIRMATION.001`](02_trend_range.md#item-56) | RANGE | REVISE | REQUEST_REVISION | - |
| 57 | [`MG1A.RANGE.INVALIDATION.001`](02_trend_range.md#item-57) | RANGE | REVISE | REQUEST_REVISION | - |
| 58 | [`MG1A.SUPPORT_RESISTANCE_CONVERSION.STRUCTURE.001`](04_break_retest_conversion.md#item-58) | SUPPORT_RESISTANCE_CONVERSION | ACCEPT | REQUEST_REVISION | - |
| 59 | [`MG1A.SUPPORT_RESISTANCE_CONVERSION.PHASE.001`](04_break_retest_conversion.md#item-59) | SUPPORT_RESISTANCE_CONVERSION | ACCEPT | REQUEST_REVISION | - |
| 60 | [`MG1A.SUPPORT_RESISTANCE_CONVERSION.CONFIRMATION.001`](04_break_retest_conversion.md#item-60) | SUPPORT_RESISTANCE_CONVERSION | REVISE | REQUEST_REVISION | - |
| 61 | [`MG1A.SUPPORT_RESISTANCE_CONVERSION.INVALIDATION.001`](04_break_retest_conversion.md#item-61) | SUPPORT_RESISTANCE_CONVERSION | REVISE | REQUEST_REVISION | - |
| 62 | [`MG1A1.COMMON.OBJECT_MODEL.001`](01_common_contracts.md#item-62) | COMMON | ACCEPT | REQUEST_REVISION | - |
| 63 | [`MG1A1.COMMON.MORPHOLOGY_EFFECT.001`](01_common_contracts.md#item-63) | COMMON | ACCEPT | APPROVE_AS_IS | - |
| 64 | [`MG1A1.COMMON.LIFECYCLE_MAPPING.001`](01_common_contracts.md#item-64) | COMMON | ACCEPT | REQUEST_REVISION | - |
| 65 | [`MG1A1.COMMON.SCALE_SPEC.001`](01_common_contracts.md#item-65) | COMMON | DEFER | REQUEST_REVISION | - |
| 66 | [`MG1A1.COMMON.ANCHOR_REVISION.001`](01_common_contracts.md#item-66) | COMMON | REVISE | REQUEST_REVISION | - |
| 67 | [`MG1A1.COMMON.RELATION_GRAPH.001`](01_common_contracts.md#item-67) | COMMON | REVISE | REQUEST_REVISION | - |
| 68 | [`MG1A1.COMMON.FORMULA_CONTRACT.001`](01_common_contracts.md#item-68) | COMMON | REVISE | REQUEST_REVISION | - |
| 69 | [`MG1A1.COMMON.REVIEW_PROTOCOL.001`](01_common_contracts.md#item-69) | COMMON | REVISE | REQUEST_REVISION | - |
| 70 | [`MG1A1.COMMON.FAMILY_REVISIONS.001`](05_final_integration.md#item-70) | COMMON | REVISE | REQUEST_REVISION | - |

## 最终签署

逐条审核完成后再填写 `review.yaml` 的 `owner_signoff`。签署不会原地修改或
伪装冻结当前 v2；冻结文件和最终 hash 由后续 MG1-B 步骤生成。
签署声明使用工具内置的受控文本，不能自由改写为已冻结或已解锁。
