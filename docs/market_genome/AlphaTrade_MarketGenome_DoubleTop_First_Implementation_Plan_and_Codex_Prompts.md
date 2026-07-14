# AlphaTrade MarketGenome：Double-Top-First 可执行实施方案与 Codex 提示词

**版本**：v0.2  
**日期**：2026-07-13  
**依赖文档**：`AlphaTrade_MarketGenome_DoubleTop_First_Technical_Report.md`  
**起始代码**：`fogdance/alphagenome_research` / `feature/market-genome-mg0`  
**建议开发分支**：`feature/market-genome-double-top-mvp`  
**总目标**：用双顶（M 头）完成一条严格因果、跨周期、dense-track、可解释 effect-scoring 的 MarketGenome 端到端最小链路

---

# 0. 如何使用本实施方案

本计划不是一次性“大任务”。每个子任务必须按以下节奏执行：

```text
下发一个 Codex prompt
→ Codex 检查真实代码与依赖
→ 生成代码、测试和 contract reports
→ 人工/领域 review gate
→ 决定 PASS、返工或停止
→ 才能下发下一 prompt
```

不得把 DT0-DT9 一次交给 Codex 自动完成。这样会重新出现原 MG0-MG9 方案的问题：过度抽象、未冻结领域定义、失败不可归因。

## 0.1 分支策略

建议：

```bash
git checkout feature/market-genome-mg0
git pull --ff-only
git checkout -b feature/market-genome-double-top-mvp
```

必须保留现有 broad MG1-A ontology 草稿。不得删除或伪造其 human freeze。

## 0.2 环境

```bash
export RUN="conda run --no-capture-output -n alphatrade_cuda12 env -u LD_LIBRARY_PATH"
export ALPHATRADE_RUNS_ROOT="$(pwd)/../alphatrade_runs/market_genome_dt_<milestone>_$(date +%Y%m%d_%H%M%S)"
export REPORTS="$ALPHATRADE_RUNS_ROOT/reports"
export CHECKPOINTS="$ALPHATRADE_RUNS_ROOT/checkpoints"
export ARTIFACTS="$ALPHATRADE_RUNS_ROOT/artifacts"
```

正式 GPU 任务禁止静默 CPU fallback。早期 ontology/data/test 任务可以 CPU-only。

## 0.3 全局硬约束

所有任务共同遵守：

1. AlphaTrade 仍处开发阶段，不得声称可交易。
2. 第一阶段唯一 pattern 是 `DOUBLE_TOP`。
3. 不实现 double bottom、head-and-shoulders、trend detector、breakout family registry。
4. trend 只作为 double-top 的 context track，不作为独立 pattern 主线。
5. morphology、confirmation、future outcome 必须分 namespace。
6. 禁止 centered swing、未来 ZigZag、未来 pivot 回填。
7. 所有高周期数据只在 bar 完全关闭后可见。
8. test 不参与 ontology、labeler threshold、loss weight、architecture 或 model selection。
9. 原 M0-M12 contracts 不得破坏。
10. 新代码优先放在：

```text
src/alphatrade/market_genome/double_top/
src/alphatrade/market_genome/common/
configs/market_genome/double_top/
docs/alphaTrade/market_genome/double_top/
tests/market_genome/double_top/
```

11. 不得为了未来多 pattern 提前实现大型通用框架。
12. 所有 reports 增加独立 profile；validator/schema 是 source of truth。
13. 每个 task 开始前执行：

```bash
git status --short
git rev-parse HEAD
```

14. Codex 最终回复统一包含：

```text
files inspected
files changed
commands run
tests and results
reports/artifacts generated
gate status
blockers/uncertainties
next recommended task
```

---

# 1. 总体路线与 Stop Gates

| 里程碑 | 交付 | 进入条件 | 退出 Gate |
|---|---|---|---|
| DT0 | Double Top ontology + detector contract | MG0/MG1-A branch available | 领域专家冻结拓扑与阶段 |
| DT1 | Causal multi-scale bars + pivots | DT0 pass | future-mutation/closed-bar pass |
| DT2 | Synthetic multi-scale generator | DT0/DT1 pass | synthetic contract pass |
| DT3 | Real-data causal labeler + gold-set tooling | DT1/DT2 pass | label quality human review pass |
| DT4 | Dense-track long-sequence dataset | DT3 approved | schema/causality/distribution pass |
| DT5 | Baseline ladder + learnability | DT2/DT4 pass | synthetic/gold-set learnability pass |
| DT6 | Mini MarketGenome backbone + heads | DT5 pass | overfit/invariance/synthetic pass |
| DT7 | Formal real-data training/evaluation | DT6 pass | representation/effect/alpha decision |
| DT8 | Market ISM/effect scoring | DT7 representation pass | structure-sensitivity pass |
| DT9 | MVP closeout + controlled abstraction | DT7/DT8 complete | expand/redesign/pause |

## 1.1 立即停止条件

任一发生时不得继续增加复杂度：

- ontology 未获得领域专家批准；
- causal pivot 无法通过 future mutation；
- synthetic held-out-scale 无法学习；
- gold-set reviewer agreement 太低；
- simple TCN/GRU 和 MarketGenome 均无 learnability；
- 双顶可识别，但 outcome 与 matched controls 无任何增量；
- 所有真实结果高度依赖单一 symbol/period。

---

# DT0：Double Top Ontology 与 Detector Contract

## DT0 目标

把现有 broad MG1-A ontology 保留为 draft，同时新增一份只针对 double top 的、可以人工冻结的最小 ontology。DT0 不实现 detector、dataset 或模型。

## DT0 产物

```text
configs/market_genome/double_top/double_top_ontology_v1.yaml
configs/market_genome/double_top/double_top_detector_search_space_v1.yaml
src/alphatrade/market_genome/double_top/schema.py
src/alphatrade/market_genome/double_top/validation.py
docs/alphaTrade/market_genome/double_top/DT0_DOUBLE_TOP_ONTOLOGY.md
docs/alphaTrade/market_genome/double_top/DT0_HUMAN_FREEZE_CHECKLIST.md
tests/market_genome/double_top/test_dt0_ontology.py
$REPORTS/dt0_double_top_ontology_validation.json
$REPORTS/dt0_double_top_ontology_validation.md
```

## DT0 人工审核点

必须由领域专家批准：

- H1/N/H2 拓扑；
- morphology 与 bearish effect 分离；
- online phases；
- confirmation、invalidation、expiry 语义；
- price basis；
- bar close policy；
- scale contract；
- detector search space 的边界。

## 发给 Codex 的提示词：DT0

```text
You are Codex working inside:
- repository: fogdance/alphagenome_research
- starting branch: feature/market-genome-mg0
- target branch: feature/market-genome-double-top-mvp

Sprint name:
DT0 Double Top Ontology + Detector Contract

Context:
The previous MarketGenome plan was too broad. We are now building one complete
vertical slice for DOUBLE_TOP only. Preserve the existing broad MG1-A ontology
as a research draft. Do not delete it, freeze it, or rewrite all pattern
families.

Primary objective:
Create a minimal, domain-reviewable, strictly causal ontology and detector
configuration contract for double top. Do not implement a detector, labeler,
synthetic generator, dataset, model, or trading logic in this task.

Hard constraints:
1. DOUBLE_TOP is the only morphology.
2. Do not add DOUBLE_BOTTOM, HEAD_SHOULDERS, TREND, RANGE, or generic pattern
   registries.
3. Prior trend may be represented only as independent context metadata.
4. Morphology must not be named or encoded as guaranteed bearish future effect.
5. Separate namespaces:
   - online_causal
   - retrospective_structure
   - future_outcome
6. No numeric detector threshold may live inside immutable topology definitions.
7. Numeric threshold search space must be a separate config and must be marked
   TRAIN_VAL_ONLY.
8. No centered-window pivot or future ZigZag semantics.
9. Preserve event_eob and available_as_of_eob.
10. Keep the artifact DRAFT_NEEDS_HUMAN_FREEZE until a real human review occurs.

Inspect first:
- configs/market_genome/pattern_ontology_v1.yaml
- docs/alphaTrade/market_genome/MG1_PATTERN_ONTOLOGY.md
- src/alphatrade/market_genome/ontology/schema.py
- src/alphatrade/market_genome/ontology/validation.py
- tests/market_genome/test_mg1a_ontology.py
- existing contracts_manifest and validator patterns

Required ontology:

A. Object identity
- ontology_id
- version
- family = DOUBLE_TOP
- object_class = MORPHOLOGY
- structural_orientation = UPPER
- semantic_role must not imply future outcome

B. ScaleSpec
- bar_resolution
- aggregation_policy
- session_alignment
- pivot_prominence_scale_ref
- formation_span_bars_min/max
- optional scale_octave
- label_close_alignment = CLOSED_BAR_ONLY

C. Anchor contract
- H1, N, H2, optional B and R
- pivot_id
- pivot_type
- anchor_state:
  PROVISIONAL, CONFIRMED, REVISED, SUPERSEDED, REVOKED
- event_eob
- available_as_of_eob
- price_basis
- price_value
- revision
- supersedes_anchor_id
- source_bar_ref

D. Geometry formula contracts
Define exact formula metadata for:
- formation_height
- peak_similarity
- neckline_depth_atr
- time_left
- time_right
- time_symmetry
- second_peak_relative_height
For each formula define numerator, denominator, sign, missingness, clipping,
price basis, and scale reference. Do not assign optimized threshold values.

E. Family phases
- NONE
- FIRST_PEAK_AVAILABLE
- NECKLINE_AVAILABLE
- SECOND_PEAK_FORMING
- SHAPE_COMPLETE
- BREAK_ATTEMPT
- BREAK_CONFIRMED
- RETESTING
- CONTINUATION
- INVALIDATED
- EXPIRED

For every phase define:
- first visible causal condition
- allowed previous phases
- allowed next phases
- required anchors
- whether future data is forbidden

F. Confirmation/invalidation/expiry semantic contracts
- distinguish SHAPE_COMPLETE from BREAK_CONFIRMED
- define accepted-break semantics abstractly
- define invalidation categories
- define expiry categories
- keep numerical values outside ontology

G. Detector search-space config
Create a small, explicit search space with parameter names only and bounded
candidate grids/ranges for:
- pivot prominence
- peak similarity
- minimum neckline depth
- min/max formation span
- break buffer
- break hold bars
- invalidation buffer
- expiry bars
Mark every field TRAIN_VAL_ONLY and forbid test-based selection.
Do not claim the grid is human-approved.

Validation requirements:
- reject future_outcome fields in runtime input schema
- reject trading fields: buy, sell, entry, action, pnl, stop, take_profit,
  teacher, grade
- enforce event_eob <= available_as_of_eob <= as_of_eob
- reject unknown phases and invalid transitions
- ensure formula contracts are complete
- ensure ontology and detector config are separate
- ensure existing MG1-A v1 remains valid and unchanged

Reports/profile:
Add an independent dt0 profile requiring JSON and markdown validation reports.
Do not modify existing M0-M12 or MG1-A profile behavior.

Acceptance:
- CPU-only tests pass
- dt0 strict validation passes
- no detector/labeler/model code exists
- no human approval is fabricated
- final status is DRAFT_NEEDS_HUMAN_FREEZE

Stop after DT0 artifacts. Final response must list every unresolved human
question in a compact freeze checklist.
```

## DT0 Gate

状态只能是：

```text
HUMAN_FREEZE_REQUIRED
ONTOLOGY_FROZEN
BLOCKED
```

只有真实领域审核后才能进入 DT1。

---

# DT1：因果多周期 Bar 与 Pivot Primitives

## DT1 目标

实现双顶链路所需的最小公共 primitives：closed-bar 多周期聚合和可修订的因果 pivot tracker。不得实现完整 double-top detector。

## DT1 产物

```text
src/alphatrade/market_genome/common/timeframes.py
src/alphatrade/market_genome/double_top/pivots.py
src/alphatrade/market_genome/double_top/pivot_state.py
configs/market_genome/double_top/dt1_pivot_default_v1.yaml
tests/market_genome/double_top/test_dt1_timeframes.py
tests/market_genome/double_top/test_dt1_pivots.py
$REPORTS/dt1_causality_validation.json/md
$REPORTS/dt1_pivot_distribution.json/md
```

## 发给 Codex 的提示词：DT1

```text
You are Codex working on branch feature/market-genome-double-top-mvp.

Sprint name:
DT1 Causal Multi-Scale Bars + Pivot Primitives

Prerequisite:
DT0 ontology is human-frozen. Read the frozen DT0 ontology and detector config
before coding. Do not silently alter the ontology.

Primary objective:
Implement closed-bar aligned multi-timeframe aggregation and a strictly causal,
revision-aware pivot tracker. Do not implement double-top detection yet.

Required timeframes for MVP:
- 1m
- 5m
- 15m
- 60m

Hard constraints:
1. A higher-timeframe bar is invisible until fully closed.
2. No centered rolling window.
3. No future ZigZag confirmation.
4. A pivot candidate may be revised by a later extreme before confirmation.
5. event_eob is the extreme bar; available_as_of_eob is the causal confirmation
   bar.
6. Future mutation after as_of must not change any output at or before as_of.
7. Session boundaries and exchange calendars must be respected.
8. Contract roll discontinuities must be recorded; do not bridge invalid gaps.
9. Do not add pattern-family abstractions.

Implement:

A. Closed-bar aggregation
- deterministic session-aligned resampling from 1m
- explicit timezone
- bar_end timestamps
- source row provenance
- no partial 5m/15m/60m values before close

B. Pivot state machine
- HIGH and LOW candidates
- PROVISIONAL, CONFIRMED, REVISED, SUPERSEDED, REVOKED
- configurable prominence using frozen scale reference
- stable pivot_id and revision lineage
- event_eob and available_as_of_eob
- source_bar_ref and price_basis

C. Streaming and batch parity
- streaming update API
- offline batch API
- identical outputs for the same prefix

Tests:
1. future-row mutation invariance
2. truncated-prefix parity
3. partial higher-timeframe bar invisibility
4. candidate revision after a new higher high/lower low
5. confirmation delay semantics
6. session boundary
7. missing bars/gaps
8. contract roll boundary
9. batch vs streaming parity
10. deterministic repeated runs

Reports:
- causality counts and max differences
- pivot count by symbol/timeframe/split
- confirmation delay distribution
- revision/supersede rates
- gap/roll exclusions

Use a one-symbol smoke first, then the existing M12 formal5 symbols for runtime
validation. Do not use future outcome data.

Acceptance:
- all CPU tests pass
- causality max_abs_diff = 0 for exact features, or documented numerical
  tolerance for floating calculations
- no pivot visible before available_as_of_eob
- dt1 schema/semantic profile passes

Stop after primitives and reports. Do not generate double-top labels.
```

## DT1 Gate

```text
CAUSAL_PRIMITIVES_PASS
REVISE
BLOCKED
```

---

# DT2：Synthetic Multi-Scale Double Top Generator

## DT2 目标

创建具有真实 ground truth 的跨尺度双顶 synthetic 数据，用于验证 ontology、label timing、模型 learnability 和 Market ISM。

## DT2 产物

```text
src/alphatrade/market_genome/double_top/synthetic/generator.py
src/alphatrade/market_genome/double_top/synthetic/paths.py
src/alphatrade/market_genome/double_top/synthetic/hard_negatives.py
src/alphatrade/market_genome/double_top/synthetic/contracts.py
configs/market_genome/double_top/dt2_synthetic_v1.yaml
$ARTIFACTS/dt2_synthetic/*.parquet
$REPORTS/dt2_synthetic_contract.json/md
$REPORTS/dt2_synthetic_distribution.json/md
```

## 发给 Codex 的提示词：DT2

```text
You are Codex working on branch feature/market-genome-double-top-mvp.

Sprint name:
DT2 Synthetic Multi-Scale Double Top Generator

Prerequisites:
- DT0 ontology frozen
- DT1 causal primitives pass

Primary objective:
Generate synthetic causal market sequences with exact double-top phase, anchor,
geometry, and future-outcome ground truth across multiple time scales. Do not
train the MarketGenome model in this task.

Required sequence families:
1. Valid double top with no neckline break.
2. Valid double top with accepted neckline break.
3. Break followed by retest and continuation.
4. Break followed by full recross failure.
5. Invalidated before shape completion.
6. Expired candidate.
7. Hard negatives.

Base process components:
- prior trend or range
- stochastic volatility
- microstructure noise
- optional volume/OI process
- session boundaries
- optional gaps

Randomized morphology parameters:
- H1/H2 relative height
- neckline depth
- peak spacing
- left/right duration
- formation span
- time dilation
- break strength
- break delay
- retest depth
- invalidation path

Required target scales:
- 1m-equivalent short pattern
- 5m-equivalent
- 15m-equivalent
- 60m-equivalent
Represent scale by time dilation and/or causal aggregation while preserving the
same topology.

Hard negatives must include:
- peaks too unequal
- trough too shallow
- random two-local-high coincidence
- triple top
- repeated range-boundary tests
- upward continuation after H2
- wick-only neckline touch without accepted close
- candidate superseded by a later high
- formation span outside contract
- head-and-shoulders-like sequence that is not a valid double top under DT0

Ground truth outputs:
- raw OHLCV/OI-like sequence
- H1/N/H2/B/R anchors
- anchor event_eob and available_as_of_eob
- online phase track
- retrospective structure track
- future outcome track
- geometry values
- generator parameters
- positive/negative subtype

Invariance tests:
1. affine price shift/scale preserves morphology labels
2. time dilation preserves topology and expected phase ordering
3. future mutation after as_of preserves online tracks
4. invalid OHLC is never produced
5. deterministic seed replay
6. hard negatives remain negative under allowed transforms

Distribution requirements:
- balanced enough for learnability experiments
- separate train/val/test generator seeds
- held-out parameter ranges or time-dilation bucket reserved for test
- no identical parameter seed across splits

Reports/profile:
- event counts by subtype and scale
- phase occupancy
- formation span distribution
- geometry distribution
- hard-negative distribution
- held-out-scale definition
- exact generator config hash

Acceptance:
- synthetic contract and tests pass
- ground truth follows frozen ontology
- no real-market threshold is inferred from synthetic test data
- no model training in this task

Stop after generating a small smoke dataset and a formal synthetic dataset.
```

## DT2 Gate

```text
SYNTHETIC_DATA_READY
REVISE_GENERATOR
BLOCKED
```

---

# DT3：真实数据 Causal Labeler 与 Human Gold Set

## DT3 目标

在真实 AlphaTrade 数据上生成可复现 weak labels，并建立领域专家审核工具。这个任务必须在人工冻结点停止，不能直接进入训练。

## DT3 产物

```text
src/alphatrade/market_genome/double_top/labeler.py
src/alphatrade/market_genome/double_top/state_machine.py
src/alphatrade/market_genome/double_top/geometry.py
src/alphatrade/market_genome/double_top/review_bundle.py
configs/market_genome/double_top/dt3_labeler_v1.yaml
$ARTIFACTS/dt3_labels/*.parquet
$ARTIFACTS/dt3_review_bundle/
$REPORTS/dt3_label_contract.json/md
$REPORTS/dt3_human_review_summary.json/md
```

## 发给 Codex 的提示词：DT3

```text
You are Codex working on branch feature/market-genome-double-top-mvp.

Sprint name:
DT3 Real-Data Causal Double-Top Labeler + Human Gold-Set Workflow

Prerequisites:
- DT0 ontology frozen
- DT1 causal primitives pass
- DT2 synthetic contract pass

Primary objective:
Implement a deterministic causal weak labeler for DOUBLE_TOP only and generate a
human-review bundle. Do not train a neural model in this task.

Hard constraints:
1. Use DT1 pivots and DT0 state transitions; do not invent another pivot method.
2. Online phase at as_of may use only anchors with available_as_of_eob <= as_of.
3. Separate online_causal, retrospective_structure, and future_outcome columns.
4. Future outcome must never affect labeler phase transitions.
5. Prior trend is context metadata, not a mandatory morphology definition unless
   explicitly approved in DT0.
6. Do not output buy/sell/entry/action/grade fields.
7. Do not tune on test.

Implement double-top state machine:
- NONE
- FIRST_PEAK_AVAILABLE
- NECKLINE_AVAILABLE
- SECOND_PEAK_FORMING
- SHAPE_COMPLETE
- BREAK_ATTEMPT
- BREAK_CONFIRMED
- RETESTING
- CONTINUATION
- INVALIDATED
- EXPIRED

Implement:
- H1/N/H2 assignment
- candidate revision behavior
- geometry formulas from DT0
- threshold evaluation using versioned labeler config
- accepted-break contract
- retest state
- invalidation and expiry reason codes
- relation back to source pivots

Data scope:
1. one-symbol smoke
2. M12 formal5 for runtime validation
3. generate a proposed full-universe plan, but do not run expensive full build
   until review passes

Human review bundle:
For stratified examples, generate:
- static chart image with only information allowed by review mode
- machine-readable event JSON
- anchor/phase timeline
- scale, symbol, as_of, config hash
- reviewer form/template

Review modes:
- ONLINE_BLINDED: hide all future bars after as_of
- RETROSPECTIVE_STRUCTURE: show full formation but no future-outcome labels
- OUTCOME: separate review artifact

Sampling must include:
- positives at every phase
- shape complete without break
- break confirmed
- invalidated/expired
- each hard-negative subtype
- each timeframe
- short/medium/long formation span
- each symbol and volatility bucket

Human fields:
- reviewer_id
- review_mode
- morphology_decision: YES | NO | AMBIGUOUS | INSUFFICIENT_EVIDENCE
- phase_decision
- anchor corrections
- confidence
- comments
- adjudication_status

Metrics after reviews are supplied:
- reviewer agreement
- labeler precision/recall against adjudicated gold set
- phase confusion
- anchor localization error
- error categories by scale

Reports:
- weak-label counts by split/scale/symbol/phase
- class imbalance
- invalidation/expiry reasons
- config sensitivity on train/val only
- no final test performance before human freeze

Acceptance before stop:
- labeler causality tests pass
- review bundle is generated
- schema/profile passes
- status = HUMAN_GOLD_SET_REVIEW_REQUIRED

Stop and wait for human review. Do not mark labels approved and do not build the
training dataset yet.
```

## DT3 人工 Gate

领域专家需要输出：

```text
LABELER_APPROVED
LABELER_APPROVED_WITH_CONFIG_REVISION
LABELER_REJECTED
```

只有批准后进入 DT4。

---

# DT4：Dense-Track Long-Sequence Dataset

## DT4 目标

把原始 1m sequence、各周期双顶 tracks 和未来效果 targets 构造成长序列训练数据；不训练模型。

## DT4 产物

```text
src/alphatrade/market_genome/double_top/dataset_builder.py
src/alphatrade/market_genome/double_top/track_spec.py
src/alphatrade/market_genome/common/chunking.py
configs/market_genome/double_top/dt4_dataset_v1.yaml
$ARTIFACTS/dt4_dataset/
$REPORTS/dt4_dataset_contract.json/md
$REPORTS/dt4_track_distribution.json/md
```

## 发给 Codex 的提示词：DT4

```text
You are Codex working on branch feature/market-genome-double-top-mvp.

Sprint name:
DT4 Dense-Track Long-Sequence Dataset

Prerequisite:
DT3 causal labeler and gold-set review are approved. Use the approved ontology,
labeler config, and review decision hashes.

Primary objective:
Build a long-sequence, dense-track training dataset for the double-top vertical
slice. Do not implement or train the MarketGenome model in this task.

Input sequence:
- base resolution: 1m
- initial context lengths supported: 1024, 4096, 8192
- raw/causal features only
- symbol/exchange/session/roll metadata

Required input channels:
- normalized OHLC geometry
- returns
- volume and OI causal statistics
- session position and boundaries
- contract/roll metadata
No pattern grade/action/future outcome is allowed as input.

Required target namespaces:

A. online_causal
- pivot high/low confirmed tracks by scale
- double_top_phase by scale
- double_top_active
- geometry tracks
- break/retest event tracks

B. retrospective_structure
- adjudicated/retrospective structure labels where available
- must have a mask and provenance

C. future_outcome
- future return targets
- rolling historical quantile baselines
- residual quantile targets
- future realized vol
- future high-low range
- MFE/MAE
- target-before-stop labels
- time-to-event

Required target scales:
- 1m
- 5m
- 15m
- 60m

Alignment:
- output tracks aligned to base 1m time axis
- higher-timeframe labels valid only at closed-bar aligned positions or via an
  explicitly documented hold-forward policy with masks
- no partial higher-timeframe leakage

Splits:
- reuse project time split principles
- purged train/val/test
- optional held-out symbol test
- optional held-out scale synthetic benchmark remains separate
- scaler fit on train only

Chunking:
- chunks must not cross invalid roll/session gaps unless explicitly supported
- preserve source row provenance
- include masks for warmup, missing outcomes, and unavailable anchors
- deterministic chunk IDs and content hashes

Data reports:
- number of chunks and valid positions
- target occupancy by phase/scale/symbol/split
- positive event counts
- class imbalance
- outcome sample counts
- context coverage
- roll/session exclusions
- train/val/test distribution shift
- exact feature/track schema

Tests:
1. no future input fields
2. prefix/future mutation parity
3. chunk boundary correctness
4. higher-timeframe close alignment
5. scaler train-only
6. no split overlap after purge/embargo
7. deterministic rebuild
8. masks consistent with label availability
9. synthetic and real dataset loaders share the same model-facing schema

Acceptance:
- dt4 strict schema/semantic profile passes
- all causal tests pass
- dataset loader can return [B,T,F] inputs and dense target/mask dictionaries
- no training is run

Stop after data contract validation.
```

## DT4 Gate

```text
DENSE_TRACK_DATA_READY
REVISE
BLOCKED
```

---

# DT5：Simple Baselines 与 Learnability Gate

## DT5 目标

在实现复杂 backbone 前，证明数据和标签可被简单模型学习，并确定长上下文/时序模型是否必要。

## DT5 产物

```text
src/alphatrade/market_genome/double_top/baselines/
configs/market_genome/double_top/dt5_baselines_v1.yaml
$REPORTS/dt5_learnability_report.json/md
$REPORTS/dt5_simple_model_leaderboard.json/md
```

## 发给 Codex 的提示词：DT5

```text
You are Codex working on branch feature/market-genome-double-top-mvp.

Sprint name:
DT5 Double-Top Learnability Gate + Simple Model Ladder

Prerequisites:
- DT2 synthetic data ready
- DT4 dense-track data ready

Primary objective:
Determine whether the double-top phase and effect targets are learnable before
building the MarketGenome backbone.

Do not implement the final MarketGenome model in this task.

Required experiments:

A. Sanity controls
1. one-batch overfit
2. shuffled phase labels
3. shuffled outcome labels
4. future-only forbidden feature injection test must be rejected by schema
5. zero/unconditional prediction

B. Pattern baselines
1. linear/logistic model on fixed local summary
2. 2-layer MLP on fixed window
3. small causal TCN
4. small GRU
5. optional current AlphaTrade short-window trunk as a reference only

C. Outcome baselines
1. event frequency baseline
2. zero return
3. rolling historical quantile
4. simple event-feature logistic/regression
5. optional LightGBM/HistGradientBoosting if dependencies permit

Datasets:
- synthetic train/val/test including held-out time-dilation bucket
- real weak-label dataset
- human gold set
- event outcome matched controls

Required metrics:
Pattern:
- per-scale phase AUPRC/macro-F1
- event precision/recall
- anchor localization if model supports it
- hard-negative false positive rate
- held-out-scale performance

Outcome:
- Brier/AUC for target-before-stop
- pinball vs rolling baseline
- calibration
- IC/rank IC within eligible event samples

Context ablation:
- 60 or 128 bars
- 512 bars
- 1024 bars
- 4096 bars where supported

Questions to answer:
1. Can simple models recover synthetic ground truth?
2. Can they generalize to held-out time dilation?
3. Is long context materially better?
4. Is phase learnable on human gold set?
5. Do real double-top events show any outcome separation vs matched controls?

Gate classification:
- LEARNABILITY_PASS
- PATTERN_ONLY_PASS_EFFECT_UNPROVEN
- LABELS_NOT_LEARNABLE
- OUTCOME_NO_INCREMENTAL_SIGNAL
- BLOCKED

Minimum requirements for LEARNABILITY_PASS:
- real phase metrics exceed shuffled and trivial baselines
- synthetic held-out-scale succeeds
- at least one temporal model improves over fixed-window linear/MLP
- no leakage control failure

Do not choose thresholds from test. Report confidence intervals and event counts.

Acceptance:
- all experiments reproducible
- simple-model leaderboard generated
- a clear go/no-go recommendation for DT6

Stop after the learnability decision.
```

## DT5 Gate

Only `LEARNABILITY_PASS` or a consciously accepted `PATTERN_ONLY_PASS_EFFECT_UNPROVEN` can enter DT6. If `LABELS_NOT_LEARNABLE`, return to DT0-DT4.

---

# DT6：MarketGenome Mini Backbone + Heads/Losses

## DT6 目标

实现单形态、单卡可运行的 causal multi-scale MarketGenome mini-backbone。先在 synthetic 和 one-batch 上验证，不跑正式真实训练。

## DT6 产物

```text
src/alphatrade/market_genome/common/causal_conv.py
src/alphatrade/market_genome/common/encoder.py
src/alphatrade/market_genome/common/transformer.py
src/alphatrade/market_genome/common/decoder.py
src/alphatrade/market_genome/double_top/model.py
src/alphatrade/market_genome/double_top/heads.py
src/alphatrade/market_genome/double_top/losses.py
configs/market_genome/double_top/dt6_model_small_v1.yaml
$REPORTS/dt6_architecture_validation.json/md
$REPORTS/dt6_synthetic_train_metrics.json/md
```

## 发给 Codex 的提示词：DT6

```text
You are Codex working on branch feature/market-genome-double-top-mvp.

Sprint name:
DT6 Causal MarketGenome Mini Backbone for Double Top

Prerequisite:
DT5 learnability gate passes. Read the DT5 report and do not build a model that
ignores its conclusions.

Primary objective:
Implement a small, single-GPU, strictly causal, multi-scale U-Net + Transformer
model for the double-top vertical slice. Train only smoke/one-batch/synthetic
experiments in this task.

Architecture constraints:
- input [B,T,F]
- initial context support: 1024 and 4096; 8192 optional after memory test
- causal stem
- 4 encoder scales approximately /1, /4, /16, /64
- small coarse Transformer, 4-6 blocks
- causal decoder with aligned skip connections
- base-resolution and coarse embeddings
- shared double-top motif head with scale embedding
- no pair stack
- no teacher/distillation
- parameter budget target: 5M-20M

Required heads:
1. pivot head
2. double-top phase head by scale
3. anchor/geometry regression head
4. break/retest event head
5. future realized vol/range heads
6. rolling-residual quantile head
7. target-before-stop head

Quantile requirement:
- use a strict causal rolling historical quantile baseline
- model predicts DeltaQ
- zero-initialize or near-zero-initialize DeltaQ path
- horizon-specific target scale
- initial output must be numerically close to rolling baseline
- preserve non-crossing quantiles

Loss requirements:
- class-balanced phase loss
- masked regression losses
- masked event/outcome losses
- residual pinball
- Brier/BCE for barrier outcome
- optional cross-scale consistency
- every weight from resolved config

Causality requirements:
1. all convolutions causal
2. all pooling/downsampling causal
3. Transformer causal mask
4. decoder/skip alignment causal
5. prefix inference parity
6. mutation after t cannot change outputs <= t

Tests:
- shape tests for multiple context lengths
- exact prefix/future mutation tests
- gradient finite
- JIT smoke
- save/restore checkpoint
- deterministic eval mode
- zero-residual initialization equals rolling baseline within tolerance
- one-batch overfit on synthetic
- held-out synthetic scale evaluation smoke

Training in this task:
- CPU unit tests
- GPU forward/backward smoke
- one-batch overfit
- small synthetic run only
- no formal real-data run

Reports:
- resolved architecture
- parameter count
- memory use
- throughput
- causality results
- one-batch curve
- synthetic metrics vs DT5 simple baselines

Gate:
- ARCHITECTURE_PASS
- ARCHITECTURE_REVISE
- SYNTHETIC_FAIL
- BLOCKED

Stop after the DT6 gate. Do not run full real-data training.
```

## DT6 Gate

必须同时：

- causal tests pass；
- one-batch overfit；
- synthetic 不劣于 DT5 最佳简单时序模型；
- rolling residual 初始化正确。

---

# DT7：正式端到端训练与评估

## DT7 目标

在真实数据上正式训练，分层给出 representation、effect、alpha 结论。不得用单一 leaderboard 或 backtest cell promotion。

## DT7 产物

```text
configs/market_genome/double_top/dt7_formal_sweep_v1.yaml
$REPORTS/dt7_sweep_manifest.json
$REPORTS/dt7_leaderboard.json/md
$REPORTS/dt7_pattern_benchmark.json/md
$REPORTS/dt7_effect_benchmark.json/md
$REPORTS/dt7_alpha_benchmark.json/md
$REPORTS/DT7_DOUBLE_TOP_DECISION_MEMO.md
```

## 发给 Codex 的提示词：DT7

```text
You are Codex working on branch feature/market-genome-double-top-mvp.

Sprint name:
DT7 Double-Top End-to-End Formal Training + Evaluation

Prerequisite:
DT6 architecture pass. Use a fresh run root and do not use --resume for the
first formal sweep.

Primary objective:
Train and evaluate the double-top MarketGenome vertical slice on real data.
Produce separate decisions for pattern representation, future effect, and alpha.
Do not claim tradability.

Formal experiment stages:

Stage A: formal5 validation
- use the M12 formal5 symbols or equivalent small universe
- 3 seeds
- verify runtime and evaluation contracts

Stage B: full existing AlphaTrade universe
- run only if Stage A has no engineering blocker and pattern metrics are above
  trivial baselines
- 3 seeds or time-fold plan documented

Required experiments:
1. DT5 best simple temporal baseline
2. MarketGenome short context
3. MarketGenome long context
4. no synthetic pretraining
5. synthetic pretraining
6. no scale sharing
7. shared scale motif head
8. no auxiliary vol/range heads
9. full MVP multi-task

Do not combine too many changes in one experiment. Keep a frozen baseline.

Required evaluation splits:
- purged time validation/test
- per-symbol metrics
- per-scale metrics
- formation-span buckets
- held-out time dilation synthetic benchmark
- optional held-out symbol benchmark

Pattern benchmark:
- phase AUPRC/macro-F1
- per-phase confusion
- event precision/recall
- anchor localization
- hard-negative false positives
- human gold-set agreement
- scale transfer

Effect benchmark:
Evaluate SHAPE_COMPLETE and BREAK_CONFIRMED separately:
- future return distribution
- matched-control comparison
- target-before-stop AUC/Brier
- MFE/MAE
- time-to-event
- event count and concentration

Alpha benchmark:
- rolling-residual pinball vs rolling historical quantile
- coverage and calibration
- Pearson IC/rank IC in eligible event samples
- direction hit
- lightweight cost matrix as diagnostic only
- concentration by symbol/month/event

Matched controls must match at least:
- symbol
- time/session bucket
- volatility regime
- prior trend context
- formation-time period where applicable

Gate classification:
A. representation_status:
   DOUBLE_TOP_REPRESENTATION_PASS | FAIL
B. effect_status:
   DOUBLE_TOP_EFFECT_PROMISING | UNPROVEN | FAIL
C. alpha_status:
   DOUBLE_TOP_ALPHA_CANDIDATE | NOT_PROMOTED

Promotion rules:
- schema/causality pass is not model-quality pass
- pattern detection pass is not effect pass
- effect pass is not tradability
- no promotion based on one seed, one scale, one symbol, or one backtest cell

Reports must include confidence intervals, seed variance, and event counts.

Final decision memo must recommend exactly one:
- PROCEED_TO_DT8
- REVISE_LABELS
- REVISE_MODEL
- EFFECT_UNPROVEN_CONTINUE_RESEARCH
- STOP_DOUBLE_TOP_SOURCE

Stop after the formal decision.
```

## DT7 Gate

进入 DT8 至少需要 `DOUBLE_TOP_REPRESENTATION_PASS`。如果 effect 完全无增量，DT8 仍可做解释验证，但不得扩展其他形态。

---

# DT8：Market ISM 与 Structure Effect Scoring

## DT8 目标

验证模型是否真正依赖双顶关键结构，而不是只看最后几根 bar、品种或波动水平。

## DT8 产物

```text
src/alphatrade/market_genome/double_top/perturbation.py
src/alphatrade/market_genome/double_top/effect_scoring.py
configs/market_genome/double_top/dt8_perturbation_v1.yaml
$REPORTS/dt8_market_ism.json/md
$ARTIFACTS/dt8_effect_examples/
```

## 发给 Codex 的提示词：DT8

```text
You are Codex working on branch feature/market-genome-double-top-mvp.

Sprint name:
DT8 Double-Top Market In-Silico Mutagenesis + Effect Scoring

Prerequisite:
DT7 representation passes. Use the frozen DT7 model bundle(s).

Primary objective:
Apply legal, localized K-line edits to synthetic and real double-top examples and
measure changes in pattern and outcome predictions.

Hard constraints:
1. Preserve valid OHLC relationships.
2. Preserve non-negative volume and valid OI.
3. Do not alter bars outside the declared edit interval.
4. Do not introduce future data into reference predictions.
5. Every edit must record exact provenance and parameters.
6. Real-market edits are counterfactual diagnostics, not claims about executable
   market interventions.

Required edits:
- raise/lower H2
- move H2 earlier/later via local time warp
- deepen/shallow neckline
- increase/decrease H1-H2 similarity
- convert close-confirmed break to wick-only attempt
- convert break to full-recross failure
- add/remove H2 volume expansion
- add/remove OI confirmation
- compress/expand formation duration

Synthetic evaluation:
- compare to known generator ground truth
- verify expected morphology phase transitions
- test monotonicity where the generator defines it

Real evaluation:
- report model sensitivity without assuming every edit has a known profit effect
- compare pattern probability, phase, break probability, residual quantiles,
  barrier probability, and embeddings

Controls:
- edit non-structural distant bars with matched magnitude
- random local edit
- last-bar-only edit
- symbol/volatility matched examples

Required outputs:
- per-edit prediction deltas
- structure vs distant-control effect ratio
- attribution around H1/N/H2/B/R
- examples where model behaves incorrectly
- sensitivity by scale and context length

Gate:
- STRUCTURE_SENSITIVITY_PASS
- LAST_BAR_SHORTCUT_DETECTED
- NONSTRUCTURAL_SHORTCUT_DETECTED
- EFFECT_INCONSISTENT
- BLOCKED

Do not retrain or tune the model on DT8 test examples.

Stop after the effect-scoring report.
```

---

# DT9：MVP 关闭与最小 Pattern Plugin 抽象

## DT9 目标

根据真实证据决定扩展、重构或暂停。只有在双顶链路成功后，才从实际代码抽取最小可复用接口。

## DT9 产物

```text
$REPORTS/DT9_DOUBLE_TOP_MVP_CLOSEOUT.md
$REPORTS/dt9_decision.json
可选：src/alphatrade/market_genome/pattern_api.py
```

## 发给 Codex 的提示词：DT9

```text
You are Codex working on branch feature/market-genome-double-top-mvp.

Sprint name:
DT9 Double-Top MVP Closeout + Controlled Pattern API Extraction

Prerequisites:
DT0-DT8 reports exist. This is a decision and cleanup task.

Primary objective:
Close the double-top vertical slice honestly and decide whether to add another
pattern. Do not add another pattern in this task.

Inspect:
- ontology and human freeze
- causal primitives
- synthetic benchmark
- gold-set review
- dense-track data contract
- simple baseline ladder
- DT6 architecture
- DT7 formal decisions
- DT8 ISM/effect scoring

Decision must be exactly one:
1. EXPAND_TO_DOUBLE_BOTTOM
2. REPRESENTATION_PASS_EFFECT_UNPROVEN
3. REDESIGN_DOUBLE_TOP
4. PAUSE_MARKETGENOME_PATTERN_SOURCE

EXPAND_TO_DOUBLE_BOTTOM requires:
- representation pass
- causality and gold-set pass
- held-out scale pass
- no severe shortcut in DT8
- evidence that the code interfaces are stable
It does not require a tradable strategy, but effect results must be at least
scientifically promising or justify multi-pattern representation pretraining.

If expansion is approved, extract only the interfaces already proven necessary:
- PatternOntology protocol
- CausalLabeler protocol
- SyntheticGenerator protocol
- DenseTrackSpec protocol
- PatternHead protocol
- EffectEvaluator protocol
- PerturbationSpec protocol

Do not build a generic registry with unused hooks. Do not implement double
bottom yet.

Closeout report must include:
- original hypothesis
- what was proven
- what failed
- model vs simple baselines
- scale generalization
- human gold-set quality
- outcome effect
- alpha status
- ISM shortcut findings
- compute and engineering cost
- exact recommendation

Schema/profile:
Add dt9 closeout profile. Preserve all earlier artifacts.

Stop after closeout and optional minimal API extraction.
```

---

# 2. 建议的人工审核节奏

| 时间点 | 审核人 | 重点 |
|---|---|---|
| DT0 后 | K 线领域专家 | 形态拓扑、阶段、失效、尺度合同 |
| DT2 后 | 领域专家 + ML | synthetic 是否覆盖真实变化和 hard negatives |
| DT3 后 | 至少两名领域审核者 | labeler precision、阶段、anchors、歧义 |
| DT5 后 | ML/量化 | 是否真正可学，长上下文是否必要 |
| DT7 后 | 项目负责人 + 量化 | representation/effect/alpha 三层决策 |
| DT8 后 | 领域专家 + 可解释性 | 模型是否关注 H1/N/H2/颈线等关键结构 |
| DT9 | 项目负责人 | 是否增加第二形态 |

---

# 3. 每阶段建议运行规模

| 阶段 | 数据/算力 |
|---|---|
| DT0 | CPU，schema/tests |
| DT1 | 1 symbol smoke → formal5 CPU/并行 |
| DT2 | CPU synthetic，百万级位置可后扩 |
| DT3 | 1 symbol → formal5；人工抽样数百事件 |
| DT4 | formal5 先建，合同通过后 full universe |
| DT5 | CPU/simple GPU，短训练 |
| DT6 | 单卡 16GB synthetic/one-batch |
| DT7 Stage A | formal5，3 seeds |
| DT7 Stage B | 24 symbols，仅 Stage A 通过后 |
| DT8 | 固定模型推理，CPU/GPU 混合 |
| DT9 | 报告与少量重构 |

---

# 4. 统一报告状态词

不得混用 schema pass 和模型 pass。使用：

```text
CONTRACT_PASS
CAUSALITY_PASS
ONTOLOGY_FROZEN
LABELER_APPROVED
SYNTHETIC_DATA_READY
DENSE_TRACK_DATA_READY
LEARNABILITY_PASS
ARCHITECTURE_PASS
DOUBLE_TOP_REPRESENTATION_PASS
DOUBLE_TOP_EFFECT_PROMISING
DOUBLE_TOP_ALPHA_CANDIDATE
NOT_PROMOTED
BLOCKED
```

示例：

```text
DT7 schema/causality = PASS
representation = PASS
effect = UNPROVEN
alpha = NOT_PROMOTED
```

这是一个合法、诚实的结论。

---

# 5. 第一周推荐执行清单

不要立即训练模型。第一周建议只完成：

1. 创建 `feature/market-genome-double-top-mvp`；
2. 执行 DT0；
3. 人工冻结 DT0；
4. 执行 DT1 one-symbol causality；
5. 执行 DT2 synthetic smoke；
6. 做一次项目 review，确认再进入 DT3。

第一周结束时应能回答：

- 什么是 double top morphology？
- 何时在线可见？
- 不同周期如何共享定义？
- causal pivot 是否可靠？
- synthetic 是否能生成多尺度 positives/negatives？

如果这些问题不能明确回答，不应写 backbone。

---

# 6. 最终开发原则

本方案的核心不是“把大计划拆小”这么简单，而是改变研究组织方式：

```text
旧路线：
先设计完整 foundation system，再期待某些形态有效。

新路线：
先证明一个 motif 的完整链路有效，再从成功代码中抽象 foundation system。
```

Double Top MVP 通过后，第二个形态优先选择 double bottom。它可以检验几何镜像是否成立，以及市场是否存在多空不对称。之后才加入头肩顶、突破和趋势状态。

最终目标仍然是完整 MarketGenome：原始长序列、多尺度 motif、dense tracks、teachers、distillation、合法 perturbation 和 alpha benchmark。但每一项扩展都必须建立在已经通过的端到端证据上。
