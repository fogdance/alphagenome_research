# AlphaTrade MarketGenome 增量垂直切片实施方案与 Codex 提示词

**版本**：v1.0
**日期**：2026-07-13
**状态**：执行顺序方案，不是 ontology freeze 或 MG5 正式决议
**依赖文档**：

- `AlphaTrade_MarketGenome_Implementation_Plan_and_Codex_Prompts.md`
- `AlphaTrade_MarketGenome_Technical_Report.md`
- `configs/market_genome/reviews/mg1a_v2_owner/review.yaml`

依据优先级：PROJECT_OWNER 已记录的最新决定约束领域语义；本文约束增量开发顺序；原实施计划继续约束正式 MG1-MG5 验收。原计划中与最新 owner 决定冲突的双人审核要求不再适用。

---

## 1. 结论

采用“先完成一个形态的端到端垂直切片，再逐个增加形态”的路线，比一次性建设完整 ontology、生成器、数据集、模型和治理平台更可行。

第一条参考链路固定为 `DOUBLE_TOP`：

```text
DOUBLE_TOP contract / canonical adapter
-> provisional label record
-> deterministic synthetic generator
-> dense tracks and causal dataset
-> cheap baseline learnability
-> minimal causal model
-> DOUBLE_TOP learnability gate
```

双顶链路通过后才提取已经被真实使用证明的共享接口，并按依赖顺序逐个增加其余形态。全部九个保留形态完成后，再执行原实施计划定义的正式 MG1-MG5 集成门禁。

这个方案改变的是**开发顺序和反馈粒度**，不降低正式验收标准。

在获得训练恢复授权后，单独使用 `DOUBLE_TOP` 最多可以走到 `MG5-DT`，其机器状态是 `SLICE_LEARNABILITY_EVIDENCE_ACCEPTED`。这表示双顶的合同、数据、模型和可学习性链路已局部跑通；它不是正式 MG5，不能产生 `PASS_TO_REAL_PRETRAINING`，也不能解锁 MG6。

---

## 2. 为什么更可行

1. **最短反馈回路**：每一步都有可运行输入、机器可验收输出和停止条件。
2. **先验证真实算法接口**：先证明现有 detector 能映射为因果 anchor、zone、lifecycle 和 provenance，再设计共享抽象。
3. **减少无效平台建设**：前三个独立算法源完成前，不建设插件系统、通用关系图引擎或分布式标注平台。
4. **提前发现数据和标签问题**：在模型建设前先用 ontology oracle、hard negatives、prefix invariance 和简单模型验证数据合同。
5. **控制回归范围**：新增一个形态时必须保持已接入形态的合同 hash、标签和 learnability 回归不退化。
6. **正式门禁不被稀释**：单形态专项 PASS 不能冒充原计划的 MG2、MG3、MG4 或 MG5 正式 PASS。

主要风险是模型和接口过度适配双顶。缓解方法是：双顶跑通后只提取最小共享接口；随后尽早加入镜像 morphology、persistent structural state 和 dependent level event 三类不同对象。

---

## 3. 状态与权限

### 3.1 专项状态

每个专项 decision manifest 必须分别记录成熟度与执行状态，不能用“训练延后”覆盖已经完成的合同验证。

`slice_status` 只能使用：

- `SLICE_DRAFT`
- `SLICE_READY_FOR_OWNER_REVIEW`
- `SLICE_VERIFIED`
- `SLICE_LEARNABILITY_EVIDENCE_ACCEPTED`
- `SLICE_REDESIGN_REQUIRED`
- `SLICE_FAIL_STOP`

`execution_status` 只能使用：

- `NOT_REQUESTED`
- `TRAINING_DEFERRED`
- `GPU_DEFERRED`
- `BLOCKED_NOT_RUN`
- `EXECUTED`

canonical pilot record 和 bundle 必须固定：

```yaml
formal_label_generation_allowed: false
eligible_for_training: false
eligible_for_evaluation: false
mg1b_allowed: false
mg6_allowed: false
```

隔离式 synthetic slice dataset 和 experiment manifest 必须固定：

```yaml
eligible_for_formal_training: false
eligible_for_formal_evaluation: false
formal_label_generation_allowed: false
mg1b_allowed: false
experimental_slice_fit_allowed: false
mg6_allowed: false
```

只有独立、run-specific、记录当前线程用户授权的 experiment authorization 才能把 `experimental_slice_fit_allowed` 设为 `true`；它不能改变两个 formal eligibility 字段。是否实际运行任何训练仍由当前线程的训练/GPU 授权控制。

专项 schema 禁止把 `GOLDSET_WORKFLOW_READY`、`TRACK_SPEC_READY_FOR_DATASET_BUILD`、`PASS_TO_REAL_PRETRAINING`、`PASS_WITH_REDESIGN` 或 `FAIL_STOP` 等完整正式值写入 status/decision 字段。

### 3.2 正式状态

只有独立的全量、hash-bound、PROJECT_OWNER 签署的正式 artifact 才能使用：

- `GOLDSET_WORKFLOW_READY`
- `TRACK_SPEC_READY_FOR_DATASET_BUILD`
- `PASS_TO_REAL_PRETRAINING`
- `PASS_WITH_REDESIGN`
- `FAIL_STOP`

MG6 入口只能读取正式 MG5 manifest，不得解析 Markdown、专项状态或人工口头描述。resolver 必须使用 enum 精确相等，禁止 substring 或文件名匹配。

### 3.3 审核规则

- ontology 最终定义由 `PROJECT_OWNER` 单人负责。
- 正式训练和评测标签由版本化 canonical algorithm + adapter 生成。
- 人工抽查是可选、append-only 的 audit metadata，不能直接覆盖算法标签。
- 发现标签问题时修订算法、adapter、profile 或 ontology 版本并重新生成，不原地改标签。

### 3.4 当前审核基线

截至 2026-07-13，owner review 已记录 `70/70` 项决定，其中 `8` 项 `APPROVE_AS_IS`、`61` 项 `REQUEST_REVISION`、`1` 项 `DEFER`。当前状态仍是 `AWAITING_OWNER_SIGNOFF`，`ready_for_mg1b_freeze=false`；`review.yaml` SHA256 是 `734b812ff82b7594149090c71eac4fff9792554b23dbbfd5d4d31bcb08eca445`。因此这些决定是增量实现的输入，但现有 ontology 还不是正式冻结版本，任何专项结果也不能跳过后续 exact-hash owner signoff。若该文件 hash 改变，当前提示词必须停止并先更新本方案中的输入基线。

### 3.5 人工 Gate artifact

需要 PROJECT_OWNER 决定的阶段只使用一个轻量 JSON 文件，不开发审批系统。约定路径为 `configs/market_genome/gates/<stage>_owner_signoff.json`，最少字段为：

```json
{
  "schema_version": "market_genome.owner_signoff.v1",
  "stage": "DT-0",
  "artifact_path": "repo-relative path",
  "artifact_sha256": "64 lowercase hex",
  "decision": "PENDING",
  "reviewer_role": "PROJECT_OWNER",
  "reviewed_at": null,
  "signoff_sha256": "64 lowercase hex or null"
}
```

`decision` 的闭集是 `PENDING | ACCEPT | REQUEST_REVISION`。canonical hash 必须复用 `alphatrade.market_genome.ontology.revision.canonical_sha256`；计算 signoff 时先把 `signoff_sha256` 置为 `null`，避免自引用。产生待审 artifact 的任务同时生成 `PENDING` 模板并停止。只有 PROJECT_OWNER 明确决定后才可写 `ACCEPT` 或 `REQUEST_REVISION`。后继任务必须通过 `slice_gates.py` 验证 signoff 自身 hash、目标 artifact hash 和 `decision=ACCEPT`；缺失、陈旧或未接受时设置 `execution_status=BLOCKED_NOT_RUN`，且不得编辑实现文件。

训练授权使用 `$REPORTS/market_genome/authorizations/<run_id>.json`，必须由同一个 validator 校验并绑定 `run_id`、stage、config SHA256、command SHA256、`training_authorized`、`gpu_authorized`、device policy、PROJECT_OWNER、authorized/expires timestamps 和 authorization SHA256。Agent 只能转录当前线程中用户明确给出的授权；过期、command/config 不匹配或缺少授权时 fail closed。CPU scaler/baseline 的 `gpu_authorized` 必须为 `false`；需要 GPU 的 run 必须为 `true` 且不得回退 CPU。

---

## 4. 总体路线

### 4.1 第一阶段：DOUBLE_TOP 参考切片

| 顺序 | 子任务 | 原里程碑的专项映射 | 退出 Gate | 是否训练 |
|---|---|---|---|---|
| DT-0 | 合同、真实 detector 绑定、zone/lifecycle adapter | MG1-DT contract | `SLICE_READY_FOR_OWNER_REVIEW` | 否 |
| DT-1 | JSONL record、hash、provenance、tamper validation | MG1-DT bundle | `SLICE_VERIFIED` | 否 |
| DT-2 | deterministic synthetic、partial/invalid/hard negatives | MG2-DT | `SLICE_VERIFIED` | 否 |
| DT-3 | dense track specification | MG3-A-DT | `SLICE_READY_FOR_OWNER_REVIEW` | 否 |
| DT-4 | causal dataset、split、mask、fingerprint | MG3-B-DT | `SLICE_VERIFIED` | 否 |
| DT-5P | baseline/scaler preregistration | MG5-DT 前置合同 | `SLICE_READY_FOR_OWNER_REVIEW` | 否 |
| DT-5S | train-only scaler 与 dataset fingerprint rebuild | MG3-B-DT finalization | `SLICE_VERIFIED` | 是，当前延后 |
| DT-5R | cheap baseline 与 leakage gate | MG5-DT 前置数据检查 | `SLICE_VERIFIED` | 是，当前延后 |
| DT-6 | minimal causal model 与三个必要 heads | MG4-DT | `SLICE_VERIFIED` | 静态实现可做；forward/backward 延后 |
| DT-7P | learnability preregistration | MG5-DT experiment contract | `SLICE_READY_FOR_OWNER_REVIEW` | 否 |
| DT-7R | one-batch、held-out scale、shuffle、model ladder | MG5-DT | `SLICE_LEARNABILITY_EVIDENCE_ACCEPTED` | 是，需再次授权 |

`MG1-DT` 到 `MG5-DT` 只是本文的专项映射名，不是原计划的正式 milestone 或 promotion token。

### 4.2 第二阶段：逐形态接入

建议顺序：

1. `DOUBLE_BOTTOM`：验证 M/W 同一 state machine 和严格镜像。
2. `TREND_UP` + `TREND_DOWN`：验证 persistent structural state 和方向镜像。
3. `RANGE/BOX_RANGE`：验证持续状态和独立 identity。
4. `RANGE/SIDEWAYS_RANGE`：验证同 family 的不同算法和 breakout acceptance 规则。
5. `BREAKOUT`：依赖已冻结的 RANGE source identity。
6. `FAILED_BREAKOUT`：依赖 RANGE + BREAKOUT，并验证直接创建为 COMPLETED。
7. `HEAD_SHOULDERS_TOP` + `INVERSE_HEAD_SHOULDERS`：最后处理 shared-core 重构和动态 neckline 例外。

镜像对需要共享同一个 core/profile，但仍应分别输出和验收 family slice。

每新增一个 family，依次执行：

```text
contract/adapter
-> synthetic/hard negatives
-> tracks/dataset
-> cheap baseline
-> shared model extension if needed
-> combined regression gate
```

不得自动进入下一个 family。

### 4.3 第三阶段：正式集成

九个正式 family 全部完成专项接入后（其中 `RANGE` 内含 `BOX_RANGE` 和 `SIDEWAYS_RANGE` 两种独立 `range_type`）：

1. 生成新的统一 ontology 版本并由 PROJECT_OWNER 审核 exact hash。
2. 执行正式 MG1-B label governance。
3. 执行全 family MG2 synthetic acceptance。
4. 执行完整 MG3 track/data acceptance。
5. 补齐正式 MG4 backbone、全部 approved heads/losses。
6. 实现 fail-closed promotion resolver，预注册并执行正式 MG5。
7. PROJECT_OWNER 签署正式结果 hash 后运行 resolver；只有正式决议为 `PASS_TO_REAL_PRETRAINING` 且 resolver 输出 `mg6_allowed=true` 才允许进入 MG6。

---

## 5. 全局执行规则

所有后续提示词共同遵守：

执行前采用原实施计划的路径合同，并显式固定 CPU 解释器：

```bash
export ALPHATRADE_ROOT="$(git rev-parse --show-toplevel)"
export ALPHATRADE_RUNS_ROOT="${ALPHATRADE_RUNS_ROOT:-$HOME/alphatrade_runs}"
export REPORTS="$ALPHATRADE_RUNS_ROOT/reports"
export MG_CPU_PYTHON="${MG_CPU_PYTHON:-/home/v/miniconda3/envs/alphatrade/bin/python}"
test -x "$MG_CPU_PYTHON"
"$MG_CPU_PYTHON" -m pytest --version
```

当前主机已用该解释器完成 `tests/market_genome` CPU 验证。先验证运行目录可写；若受限环境不能创建仓库外目录，应请求对应写权限，不得改写到 Git 仓库内。解释器或路径不可用时设置 `execution_status=BLOCKED_NOT_RUN`。

1. 先读取 `AGENTS.md`、真实代码、真实输出和当前 `git status`，不得猜测实现。
2. 一次只执行一个提示词；达到当前 Gate 后停止，不自动进入下一任务。
3. 当前 GPU 和训练禁令持续有效。没有用户在当前线程的明确恢复授权时：
   - 使用 `CUDA_VISIBLE_DEVICES=''`；
   - 不导入或运行 JAX/model runtime，不运行 forward、backward、optimizer、训练、sweep 或 GPU smoke；
   - 不停止或影响其他进程；
   - 模型任务只实现代码、schema、manifest 和不导入模型 runtime 的静态校验，然后设置 `execution_status=TRAINING_DEFERRED`。
   - 所有 CPU 检查固定 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、无 multiprocessing、fixture-only/bounded rows，不执行批量真实数据构建。
4. 不使用 test/held-out data 选择 ontology threshold、profile、loss weight、architecture 或 decision threshold。
5. 所有价格识别使用已收盘 raw OHLC；Volume/OI 必须有效但不参与形态存在、确认或失效。
6. 每个 timeframe 独立识别和保存 identity；不得把低周期形态提升为高周期标签。
7. 单个 run 不能跨 active-contract roll；roll 后终止旧 candidate 并重新识别。
8. online causal phase、retrospective completion 和 future outcome 严格分开。
9. 所有 source、contract、profile、dataset、split、scaler、model 和 report 都必须有内容 hash。
10. 任何 causal 输出必须通过 prefix/future-mutation invariance。
11. 每个阶段生成一个小型 machine-readable decision manifest，至少记录 input/git/artifact hashes、命令与 exit code、passed/failed/skipped tests、skip reason、两轴专项状态、gate flags 和唯一允许的下一阶段；不建设通用 workflow 服务。
12. `PREDECESSOR_MANIFESTS` 必须列出每个动态前置 manifest 的绝对路径或 run-root-relative 路径及 SHA256；禁止解析“latest”。前置缺失、hash stale 或 Gate 不符时不编辑实现，只写 stop manifest，设置 `execution_status=BLOCKED_NOT_RUN`，且不生成当前阶段 decision。
13. `execution_status` 优先级固定：前置失败为 `BLOCKED_NOT_RUN`；前置齐全但无训练授权为 `TRAINING_DEFERRED`；已有训练授权但所需 GPU 未授权或不可用为 `GPU_DEFERRED`；当前阶段全部执行完才是 `EXECUTED`。`NOT_REQUESTED` 只用于尚未启动的阶段。
14. 最终回复必须列出：观察文件、修改文件、命令、测试结果、artifact、Gate、未验证项和下一步。

---

## 6. 子任务提示词

以下提示词应逐个复制执行，不能一次性下发多个。

### 6.1 DT-0：DOUBLE_TOP 合同与 canonical adapter

```text
You are Codex working inside the AlphaTrade repository.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-0 DOUBLE_TOP contract and canonical adapter.

Goal:
Finish the smallest CPU-only, non-formal DOUBLE_TOP slice. Do not build a
global ontology platform or support another family.

Required evidence:
- Read configs/market_genome/reviews/mg1a_v2_owner/review.yaml at SHA256
  734b812ff82b7594149090c71eac4fff9792554b23dbbfd5d4d31bcb08eca445.
  REQUEST_REVISION entries are implementation requirements, not freeze
  approval. Inspect all COMMON and MG1A.DOUBLE_TOP.* records.
- Inspect the real source
  /home/v/Documents/work/stock_peter/tools/patterns/m_w_pattern_detector.py
  at commit 57ee01baf8a6859d89e4c2e8ce83f41cc29c05b5 and file SHA256
  f5c2e40baf3d3aebb3545ed5a646daeac633d8e0c5710c8dc3f76e745972190a.
- Inspect current MarketGenome extension points. Do not restore the deleted
  zero-hash pilot drafts.

Implement only:
- one versioned DOUBLE_TOP pilot contract;
- one adapter that verifies source/config hashes, validates closed raw
  OHLCV/OI, runs analyze_m_w_patterns with pattern_scope=both, selects m_top,
  constructs anchors/zones, maps the four-state lifecycle, and emits stable
  content-addressed records;
- focused CPU tests.

Contract rules:
- H-L-H anchors become available only on causal pivot confirmation;
- top_zone spans the two peak pivot bars from
  min(left_peak_bar.Low, right_peak_bar.Low) to
  max(left_peak_bar.High, right_peak_bar.High); neckline_zone spans the
  neckline pivot bar Low-High;
- completion requires strict Close below neckline_zone.low*(1-buffer);
- invalidation requires strict Close above top_zone.high*(1+buffer);
- equality and wick-only penetration do not trigger;
- expiry and supersession run before same-EOB close tests;
- COMPLETED/INVALIDATED are immutable;
- outcomes cannot rewrite lifecycle;
- require a single active contract and reject rolls;
- stable IDs cannot depend on detector sequential IDs or input start.

Hard gates:
- formal_label_generation_allowed=false;
- eligible_for_training=false; eligible_for_evaluation=false;
- mg1b_allowed=false; mg6_allowed=false;
- source-default geometry parameters remain PROVISIONAL until owner-approved
  for an explicit ScaleSpec/calendar profile.

Tests must include the real 13-bar confirmed fixture, source/config/contract
hash tampering, missing OI, partial-bar rejection, candidate/completed/
invalidated/expired/superseded, equality/wick/epsilon boundaries, threshold
agreement between scalar detector output and zone edges, stable replay ID,
prefix invariance, future mutation, and no JAX/GPU imports.

Required outputs:
- configs/market_genome/double_top_contract_v1.yaml
- src/alphatrade/market_genome/double_top.py
- tests/market_genome/test_double_top.py
- src/alphatrade/market_genome/slice_gates.py
- src/alphatrade/market_genome/slice_decision.schema.json
- src/alphatrade/market_genome/owner_signoff.schema.json
- src/alphatrade/market_genome/experiment_authorization.schema.json
- tests/market_genome/test_slice_gates.py with stale/tamper/unknown-field tests
- configs/market_genome/gates/dt0_owner_signoff.json with decision=PENDING
- $REPORTS/market_genome/slices/double_top/dt0_decision.json

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_slice_gates.py tests/market_genome/test_double_top.py
Expected exit code: 0.

Use CUDA_VISIBLE_DEVICES=''. Do not train or generate formal labels.
Decision: SLICE_READY_FOR_OWNER_REVIEW or SLICE_FAIL_STOP.
Stop after tests, decision manifest and owner-signoff template. The only
permitted next prompt after owner ACCEPT is DT-1; do not execute it now.
```

### 6.2 DT-1：最小 label bundle 与 tamper validation

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-1 DOUBLE_TOP provisional record bundle.

Prerequisite:
DT-0 tests and decision manifest pass, and
configs/market_genome/gates/dt0_owner_signoff.json validates with
decision=ACCEPT and the exact current contract hash. If not, set
execution_status=BLOCKED_NOT_RUN and do not edit files.

Goal:
Prove record serialization, provenance, replay, and tamper detection for the
DOUBLE_TOP slice. This is not MG1-B and does not create gold/formal labels.

Implement the minimum needed to:
- write duplicate-safe JSONL records atomically;
- write one small manifest containing contract, detector, profile, source
  snapshot, scale, active-contract, record-set and schema hashes;
- reload and semantically validate the bundle;
- keep online phase, retrospective completion and future outcome separate;
- record optional human audit metadata without allowing it to overwrite the
  canonical record.

Do not add reviewer comparison, adjudication services, Parquet, a plugin
registry, or generic annotation infrastructure.

Tests:
- deterministic replay produces identical IDs and hashes;
- stale contract/source/profile/record/manifest hashes fail closed;
- unknown fields and duplicate keys fail;
- pilot records cannot set formal/gold/training/evaluation/MG6 flags;
- detector batch-local IDs are never used as canonical IDs;
- future outcome edits cannot change online/completion hashes.

Required outputs:
- src/alphatrade/market_genome/double_top_bundle.py
- tests/market_genome/test_double_top_bundle.py
- $REPORTS/market_genome/slices/double_top/dt1_fixture_bundle/manifest.json
- $REPORTS/market_genome/slices/double_top/dt1_decision.json

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_double_top.py tests/market_genome/test_double_top_bundle.py
Expected exit code: 0.

CPU only; no training or real-data bulk labeling.
Decision: SLICE_VERIFIED or SLICE_FAIL_STOP. Stop after the bundle test and
decision manifest. The only permitted next prompt is DT-2; do not execute it.
```

### 6.3 DT-2：DOUBLE_TOP deterministic synthetic 与 hard negatives

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-2 reproducible DOUBLE_TOP synthetic generator.

Prerequisite:
The accepted DT-0 signoff and passing DT-1 bundle decision are present and
hash-valid. If not, set execution_status=BLOCKED_NOT_RUN and do not edit files.

Goal:
Generate legal deterministic OHLCV/OI sequences with exactly recoverable
DOUBLE_TOP anchors and per-EOB lifecycle truth. Synthetic data tests pipeline
learnability, not market alpha.

Implement only DOUBLE_TOP cases:
- ACTIVE, COMPLETED, top-zone INVALIDATED, EXPIRED, SUPERSEDED;
- partial H-L and H-L-H not yet causally available;
- no-pattern and overlap fixtures;
- threshold-bracketing hard negatives for peak similarity, retracement,
  separation, leg duration, neckline equality, wick-only crossing, near break,
  early break, and near invalidation.

Requirements:
- deterministic seed/config replay;
- legal positive OHLC and finite Volume/OI;
- multiple explicitly versioned pilot-only synthetic durations, amplitudes
  and scales without claiming they are approved real-market profiles;
- price scaling, volatility/noise, time dilation and causal crop tests;
- canonical adapter is the oracle;
- generator metadata, oracle fields and labels never enter runtime inputs;
- holding OHLC fixed while changing Volume/OI cannot change shape labels;
- future mutation cannot change earlier labels.

Produce a small config, schema, manifest and machine-readable fixtures. Avoid
a generic all-family generator until another family is onboarded.

Required outputs:
- configs/market_genome/synthetic_double_top_v1.yaml
- src/alphatrade/market_genome/synthetic/double_top.py
- tests/market_genome/test_double_top_synthetic.py
- $REPORTS/market_genome/slices/double_top/dt2_decision.json

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_double_top.py tests/market_genome/test_double_top_bundle.py tests/market_genome/test_double_top_synthetic.py
Expected exit code: 0.

CPU only; no model fitting.
Decision: SLICE_VERIFIED or SLICE_FAIL_STOP.
Stop after the generator contract and decision manifest. The only permitted
next prompt is DT-3; do not execute it.
```

### 6.4 DT-3：DOUBLE_TOP dense track specification

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-3 DOUBLE_TOP dense track specification.

Prerequisites:
DT-2 synthetic contract passes. No model training is allowed.

Goal:
Define only the tracks required to measure DOUBLE_TOP learnability:
- four-state online lifecycle at each EOB;
- candidate/completion/invalidation event impulses;
- left-peak, neckline-trough and right-peak heatmaps plus masks;
- top-zone and neckline-zone bounds when causally available;
- retrospective completion in a separate namespace.

For every track define formula, unit, resolution, shape, dtype, mask, causal
interpretation, scaler policy, loss family, metric and leakage risk. Do not add
future return, MFE/MAE, barrier, flow, reconstruction or trading tracks in this
slice unless a separate owner decision explicitly requires them.

Runtime inputs must exclude generator metadata, oracle fields, anchor IDs,
labels, future masks and future outcomes. Any train-fitted scaler remains
unresolved until training resumes.

Implement schema validation and missing/invalid definition tests.

Required outputs:
- configs/market_genome/double_top_track_spec_v1.yaml
- src/alphatrade/market_genome/datasets/double_top_tracks.py
- tests/market_genome/test_double_top_tracks.py
- configs/market_genome/gates/dt3_owner_signoff.json with decision=PENDING
- $REPORTS/market_genome/slices/double_top/dt3_decision.json

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_double_top_tracks.py
Expected exit code: 0.

Decision: SLICE_READY_FOR_OWNER_REVIEW or SLICE_FAIL_STOP.
Stop after the owner-signoff template. The only permitted next prompt after
owner ACCEPT is DT-4; do not execute it now.
```

### 6.5 DT-4：DOUBLE_TOP causal dataset

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-4 DOUBLE_TOP causal dataset builder.

Prerequisites:
configs/market_genome/gates/dt3_owner_signoff.json validates with
decision=ACCEPT and the exact track-spec hash, and DT-2 passes. Otherwise set
execution_status=BLOCKED_NOT_RUN and do not edit files.

Goal:
Build the smallest synthetic-only long-sequence dataset needed for DOUBLE_TOP
learnability. Do not build the full MG3 storage platform.

Requirements:
- configurable contexts beginning with 512 and 1024; add 4096 only when needed;
- dense inputs/tracks/masks and source provenance;
- grouped split by generator seed and scenario identity;
- frozen held-out seeds and held-out synthetic scales;
- no overlapping source scenario across splits;
- train-only scaler artifacts if and only if both the normalization/scaler
  policy is owner-approved and fitting has been authorized;
  otherwise keep the scaler unresolved and validate that no fitted values are
  present;
- dataset fingerprint includes all upstream hashes and splits plus either the
  fitted scaler hash or scaler_spec_hash with fit_status=UNRESOLVED;
- document a measured storage choice instead of generalizing prematurely.

Causality tests:
- prefix invariance and future-source mutation;
- event boundary and chunk overlap;
- held-out group disjointness;
- high-timeframe closed-bar alignment where present;
- purge sufficiency;
- train-only scaler when present; otherwise absence of fitted statistics;
- no generator/oracle/label metadata in runtime inputs.

Required outputs:
- configs/market_genome/double_top_dataset_v1.yaml
- src/alphatrade/market_genome/datasets/double_top_dataset.py
- tests/market_genome/test_double_top_dataset.py
- $REPORTS/market_genome/slices/double_top/dt4_decision.json

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_double_top_synthetic.py tests/market_genome/test_double_top_tracks.py tests/market_genome/test_double_top_dataset.py
Expected exit code: 0.

CPU only; do not fit a model.
Decision: SLICE_VERIFIED or SLICE_FAIL_STOP. Record execution_status as
EXECUTED when the build and tests complete.
Stop after the dataset decision manifest. The only permitted next prompt is
DT-5P; do not execute it now.
```

### 6.6.1 DT-5P：baseline/scaler preregistration

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-5P freeze the DOUBLE_TOP baseline and scaler experiment contract.

Prerequisite:
Bind the exact DT-4 decision manifest path/hash. DT-4 may still declare its
scaler fit_status=UNRESOLVED.

Create one hash-bound preregistration bundle containing the normalization and
train-only scaler policy, folds, metrics, thresholds, seeds, held-out scales,
baseline list, grouped shuffle, leakage controls, commands, compute limits and
stop rules. Do not fit a scaler or model and do not inspect held-out results.

Required outputs:
- configs/market_genome/double_top_baseline_prereg_v1.yaml
- tests/market_genome/test_double_top_baseline_prereg.py
- configs/market_genome/gates/dt5p_owner_signoff.json with decision=PENDING
- $REPORTS/market_genome/slices/double_top/dt5p_decision.json

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_double_top_baseline_prereg.py
Expected exit code: 0. Set slice_status=SLICE_READY_FOR_OWNER_REVIEW and
execution_status=EXECUTED.

Stop for owner review. The only permitted next prompt after owner ACCEPT is
DT-5S; do not execute it now.
```

### 6.6.2 DT-5S：train-only scaler 与 fingerprint rebuild

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-5S fit only the preregistered train-only scaler and rebuild DT data.

Prerequisites:
dt5p_owner_signoff.json is ACCEPT for the exact preregistration, DT-4 inputs
match their path/hash, and a non-expired run authorization binds the frozen CPU
scaler command with training_authorized=true and gpu_authorized=false.

Without training authorization, write only the stop manifest with
execution_status=TRAINING_DEFERRED. After authorization, fit on train groups
only, rebuild the dataset fingerprint, prove validation/test values cannot
change the scaler, and do not fit any baseline or neural model.

Required outputs:
- one scaler artifact and manifest at the paths frozen by DT-5P
- rebuilt DT dataset manifest/fingerprint
- tests/market_genome/test_double_top_scaler.py
- $REPORTS/market_genome/slices/double_top/dt5s_decision.json

Acceptance command:
Run the exact bounded CPU command frozen in DT-5P; expected exit code is 0.
Set slice_status=SLICE_VERIFIED only when train-only and fingerprint tests pass.

Stop after the decision manifest. The only permitted next prompt is DT-5R;
do not execute it now.
```

### 6.6.3 DT-5R：低成本 baseline 与 leakage gate

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-5R execute the preregistered DOUBLE_TOP baseline gate.

Prerequisites:
DT-5P owner signoff and DT-5S decision/fingerprint validate by path and hash;
a non-expired run authorization binds the exact CPU command with
training_authorized=true and gpu_authorized=false.

Run identical folds/masks/features for majority/constant, ontology oracle,
logistic/linear, small MLP and small causal TCN. Evaluate lifecycle,
completion/invalidation events and anchor localization. Report every seed,
macro-F1/AUPRC, support, timing error, hard-negative FPR and held-out scales.
Run grouped label shuffle and metadata/padding/position/scale-ID leakage audits.
No best-seed selection or test tuning; suspiciously perfect results fail.

Required outputs:
- src/alphatrade/market_genome/training/double_top_baselines.py
- tests/market_genome/test_double_top_baselines.py
- $REPORTS/market_genome/slices/double_top/dt5r_leaderboard.json
- $REPORTS/market_genome/slices/double_top/dt5r_decision.json

Acceptance command:
Run the exact command frozen in the accepted DT-5P bundle; expected exit code
is 0. Result is SLICE_VERIFIED, SLICE_REDESIGN_REQUIRED or SLICE_FAIL_STOP.

Stop after the decision manifest. Only SLICE_VERIFIED permits DT-6; this does
not authorize formal MG4, and DT-6 must not run now.
```

### 6.7 DT-6：最小 causal model 与必要 heads

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-6 minimal causal model for DOUBLE_TOP.

Prerequisite:
DT-5R is SLICE_VERIFIED and its manifest path/hash validates. Otherwise follow
the common BLOCKED_NOT_RUN contract without editing.

Goal:
Implement the smallest shared causal dense model that can be compared with the
small TCN. Do not implement the full MarketGenome architecture.

Initial scope:
- a configurable causal convolutional encoder, optionally one small coarse
  attention block only if evidence requires it;
- explicit scale embedding for approved synthetic scales;
- heads only for four-state lifecycle, completion/invalidation events and
  three anchor heatmaps;
- mask-aware/class-balanced losses and per-head diagnostics.

Exclude return/path/barrier/flow/reconstruction heads, pair stack, teachers,
distillation and trading outputs.

With the current training deferral, implement code/config plus AST/schema tests
that do not import JAX or the model runtime. Do not run forward, backward,
optimizer, JIT or a sweep. Set execution_status=TRAINING_DEFERRED and do not
claim SLICE_VERIFIED.

After explicit training restoration, require a command/config-bound run
authorization, then run shapes, dense not-last-only outputs, prefix/future
mutation, variable context, deterministic eval, masks, class weights,
finite-gradient backward and loss-behavior tests. All config values must be
resolved and reported. Only then set SLICE_VERIFIED or SLICE_FAIL_STOP.

Required outputs:
- configs/market_genome/double_top_model_v1.yaml
- src/alphatrade/market_genome/model_double_top.py
- tests/market_genome/test_double_top_model_static.py
- tests/market_genome/test_double_top_model_runtime.py
- $REPORTS/market_genome/slices/double_top/dt6_decision.json

Acceptance command before training restoration:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_double_top_model_static.py
Expected exit code: 0. The runtime command, backend/device policy and tolerances
must be hash-frozen before authorization; a static pass is not DT-6 acceptance.

Stop after the current verification scope and decision manifest. The only
permitted next prompt after SLICE_VERIFIED is DT-7P; do not execute it now.
```

### 6.8.1 DT-7P：DOUBLE_TOP learnability preregistration

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-7P freeze the DOUBLE_TOP slice learnability experiment.

Prerequisites:
Bind exact paths/hashes for accepted DT-0 through DT-6 manifests. Do not read
held-out results.

Freeze metrics, thresholds, seeds, contexts, seen/held-out scales, ablations,
compute budget, device policy, commands, stop rules, shuffle-collapse rule and
the exact value-beyond-TCN criterion.

Required outputs:
- configs/market_genome/double_top_slice_gate_v1.yaml
- tests/market_genome/test_double_top_slice_gate_prereg.py
- configs/market_genome/gates/dt7p_owner_signoff.json with decision=PENDING
- $REPORTS/market_genome/slices/double_top/dt7p_decision.json

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_double_top_slice_gate_prereg.py
Expected exit code: 0. Set slice_status=SLICE_READY_FOR_OWNER_REVIEW and
execution_status=EXECUTED.

Stop for owner review. The only permitted next prompt after owner ACCEPT is
DT-7R; do not execute it now.
```

### 6.8.2 DT-7R：DOUBLE_TOP 专项 learnability gate

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: DT-7R execute the accepted DOUBLE_TOP slice learnability gate.

Run prerequisites:
DT-7P owner signoff validates with decision=ACCEPT and the exact config hash;
all DT-0 through DT-6 predecessor paths/hashes still validate; and a
non-expired authorization binds the exact command/config and training=true.

Safety gate:
Without training authorization set TRAINING_DEFERRED. If the accepted device
policy requires GPU and training is authorized but GPU permission/resource is
absent, set GPU_DEFERRED. Never stop another process or silently change the
device policy.

After authorization run:
- one-batch overfit;
- phase/event/anchor synthetic recovery;
- held-out seed and held-out scale;
- grouped shuffled-label control;
- constant, logistic, MLP, small TCN and minimal DT model comparison;
- only preregistered context/downsampling ablations;
- post-training prefix/future invariance.

PASS requires every preregistered threshold, shuffled collapse, causal tests,
held-out-scale performance and meaningful value beyond the small TCN on at
least one multi-scale DT task.

Allowed decisions:
- SLICE_LEARNABILITY_EVIDENCE_ACCEPTED
- SLICE_REDESIGN_REQUIRED
- SLICE_FAIL_STOP

Execution status is recorded separately as EXECUTED, TRAINING_DEFERRED or
GPU_DEFERRED.

Required outputs after execution:
- $REPORTS/market_genome/slices/double_top/dt7r_manifest.json
- $REPORTS/market_genome/slices/double_top/dt7r_leaderboard.json
- $REPORTS/market_genome/slices/double_top/dt7r_decision.json

Acceptance command:
Run only the exact command and environment frozen in DT-7P; expected exit code
is 0. A blocked/deferred run writes only a stop manifest, not a leaderboard.

SLICE_LEARNABILITY_EVIDENCE_ACCEPTED only permits onboarding the next family. It is not
official MG5, cannot use PASS_TO_REAL_PRETRAINING, and cannot unlock MG6.
Stop after the decision manifest. The only permitted next prompt after
SLICE_LEARNABILITY_EVIDENCE_ACCEPTED is CORE-1; do not execute it now.
```

### 6.9 CORE-1：双顶通过后提取最小共享接口

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: CORE-1 evidence-based extraction after DOUBLE_TOP slice pass.

Prerequisite:
DT-7R is SLICE_LEARNABILITY_EVIDENCE_ACCEPTED with an exact manifest path/hash,
or its accepted redesign has been implemented and the full DT regression has
subsequently passed. Otherwise follow the common BLOCKED_NOT_RUN contract.

Goal:
Extract only interfaces proven necessary by the completed DT path: canonical
source binding, contract/profile hash validation, closed-bar input validation,
stable identity, anchor/zone serialization, lifecycle namespaces, JSONL bundle
validation and dataset provenance.

Rules:
- preserve exact DT outputs and hashes or publish an explicit new version;
- no detector discovery/plugin framework;
- no generic relation graph engine;
- no support for a family not yet onboarded;
- prefer explicit adapter code over speculative abstraction;
- require golden replay and full DT regression before/after extraction.

Required outputs:
- only the minimal shared modules justified by the diff;
- tests/market_genome/test_market_genome_slice_core.py
- $REPORTS/market_genome/slices/double_top/core1_decision.json

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_double_top.py tests/market_genome/test_double_top_bundle.py tests/market_genome/test_double_top_synthetic.py tests/market_genome/test_double_top_tracks.py tests/market_genome/test_double_top_dataset.py tests/market_genome/test_market_genome_slice_core.py
Expected exit code: 0 and identical golden DOUBLE_TOP outputs before/after.

Decision: SLICE_VERIFIED or SLICE_FAIL_STOP. Stop before adding a family.
```

### 6.10 FAMILY-N：逐个形态接入模板

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: onboard exactly one ONBOARDING_UNIT through exactly one STAGE.

Required parameters, which must be concretely filled before execution:
- ONBOARDING_UNIT = one family, one range subtype, or one required mirror pair
- OUTPUT_FAMILIES = exact closed list of formal family outputs
- VARIANT = NONE | BOX_RANGE | SIDEWAYS_RANGE
- STAGE = CONTRACT | SYNTHETIC | TRACKS | DATASET | BASELINE | MODEL | SLICE_GATE
- APPROVED_PREDECESSOR_HASHES = explicit immutable hashes
- SOURCE_REPO_COMMIT, SOURCE_FILE, SOURCE_FILE_SHA256, SOURCE_CALLABLE
- PROFILE_SHA256, SCALE_SPEC_SHA256, CALENDAR_SHA256; each is either
  owner-approved formal or explicitly marked pilot-only
- OUTPUT_NAMESPACE
- TARGET_OUTPUT_PATHS = exact repo/report paths for this STAGE
- ACCEPTANCE_COMMANDS = exact commands and expected exit codes
- PREREGISTRATION_SHA256 for BASELINE, MODEL or SLICE_GATE

Unset placeholders are a blocker. Never advance to the next STAGE or ONBOARDING_UNIT
automatically. Use execution_status=BLOCKED_NOT_RUN without editing when a
required value is absent or stale.

Read AGENTS.md, the 70/70 decision-complete but non-freeze-ready owner-review
COMMON records, all records for OUTPUT_FAMILIES, the pinned detector source,
and current integrated tests.

Stage behavior:
- CONTRACT: bind one canonical source/profile, implement family-specific
  anchors/zones/lifecycle and tests, then request owner review.
- SYNTHETIC: add deterministic positives, partials, invalidations, overlaps and
  threshold-bracketing hard negatives. No model fitting.
- TRACKS: add only required dense tracks/masks and leakage rules.
- DATASET: extend grouped splits/fingerprints without breaking prior families.
- BASELINE: run only after preregistration and training authorization.
- MODEL: extend the shared trunk/heads minimally; no family-specific backbone
  without evidence.
- SLICE_GATE: run the combined prior-family + new-family regression and
  preregistered learnability subset.

Allowed stage results:
- CONTRACT or TRACKS: SLICE_READY_FOR_OWNER_REVIEW | SLICE_FAIL_STOP
- SYNTHETIC or DATASET: SLICE_VERIFIED | SLICE_FAIL_STOP
- BASELINE: SLICE_VERIFIED | SLICE_REDESIGN_REQUIRED | SLICE_FAIL_STOP
- MODEL: SLICE_VERIFIED | SLICE_FAIL_STOP, or TRAINING_DEFERRED as execution
- SLICE_GATE: SLICE_LEARNABILITY_EVIDENCE_ACCEPTED |
  SLICE_REDESIGN_REQUIRED | SLICE_FAIL_STOP

Family-specific constraints:
- DOUBLE_BOTTOM: exact M/W mirror, same callable and profile as DOUBLE_TOP.
- TREND_UP/TREND_DOWN: same market_structure source/profile and generated
  direction-mirror tests; onboard the pair atomically at core level.
- RANGE: BOX_RANGE and SIDEWAYS_RANGE remain distinct range_type values and
  identities; do not reuse daily defaults on another timeframe.
- BREAKOUT: requires frozen RANGE source identity; no duplicate detector.
- FAILED_BREAKOUT: requires RANGE + BREAKOUT; create directly COMPLETED at
  canonical reentry and keep source range ACTIVE.
- HEAD_SHOULDERS_TOP/INVERSE_HEAD_SHOULDERS: one shared TOP/BOTTOM core/profile;
  existing separate files are migration evidence, not two final cores; preserve
  the dynamic neckline exception.

Global constraints:
- raw closed bars; Volume/OI observational;
- independent timeframe identities and no cross-roll anchors;
- append-only provenance and terminal identity;
- separate online/completion/outcome namespaces;
- no test tuning, trading claims, formal promotion or GPU work without current
  authorization.

Every STAGE must rerun prior-family golden, causal and bundle tests. A new
family cannot silently change existing labels or contract hashes.

Required outputs:
- every filled TARGET_OUTPUT_PATH for the named STAGE;
- configs/market_genome/gates/<onboarding_unit>_<stage>_owner_signoff.json when
  human review is required;
- $REPORTS/market_genome/slices/<onboarding_unit>/<stage>_decision.json.

Acceptance command:
Run only the filled ACCEPTANCE_COMMANDS, which must include new targeted tests
and every prior-family golden/causal/bundle test with CUDA hidden where
applicable. BASELINE, MODEL runtime and SLICE_GATE additionally require the
accepted preregistration and current training/GPU authorization.

Return the current slice Gate and stop. Record the only permitted next STAGE
in the decision manifest; do not execute it or another ONBOARDING_UNIT now.
```

### 6.11 FINAL-1A：统一 ontology freeze

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-1A generate the next unified formal ontology candidate.

Entry requirements:
- exactly nine retained families have immutable slice artifacts;
- removed families are absent;
- the ScaleSpec/registry interface and unresolved-state rules are owner-approved
  and hash-bound; unresolved concrete calendars/profiles block only the
  affected timeframe's formal labels, not ontology schema freeze;
- all slice regression manifests pass and no recognition blocker remains.

Generate one new immutable ontology version, schema, registry, exact semantic
diff, supersession lineage, CPU validation tests, and
configs/market_genome/gates/final_1a_owner_signoff.json with decision=PENDING.
Do not mutate v2 and do not run MG1-B.

Required outputs:
- the next versioned configs/market_genome/pattern_ontology_<version>.yaml;
- its closed schema, registry, exact-diff report and validation tests;
- configs/market_genome/gates/final_1a_owner_signoff.json;
- $REPORTS/market_genome/formal/final_1a_decision.json.

Acceptance:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome
must exit 0. Write $REPORTS/market_genome/formal/final_1a_decision.json with
all nine families and hashes.

Stop for PROJECT_OWNER exact-hash review. Only owner ACCEPT permits FINAL-1B;
do not execute it now.
```

### 6.12 FINAL-1B：正式 MG1-B

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-1B formal ontology freeze and algorithm-label governance.

Binding source:
Read MG1-B in the original Implementation Plan; its required repository and
$REPORTS outputs remain mandatory except for the superseded dual-review rule.

Prerequisite:
final_1a_owner_signoff.json validates with decision=ACCEPT and the exact
ontology hash. Otherwise use execution_status=BLOCKED_NOT_RUN and do not edit.

Implement the original MG1-B schema/manifest outputs, replacing its obsolete
mandatory dual-review/adjudication mechanics with the owner-approved rule:
canonical versioned algorithms generate formal labels; PROJECT_OWNER alone
approves ontology; optional human audits are append-only and cannot overwrite
labels. Preserve online/completion/outcome visibility separation.

Acceptance:
Run the targeted MG1-B tests and full tests/market_genome with
CUDA_VISIBLE_DEVICES=''; both must exit 0. Generate the frozen ontology and
$REPORTS/mg1_goldset_manifest.json/md with exact provenance hashes.

On success set formal_decision=GOLDSET_WORKFLOW_READY. If blocked, set only
execution_status=BLOCKED_NOT_RUN and emit no formal decision. Stop before
annotation, MG2 or training. Only GOLDSET_WORKFLOW_READY permits FINAL-1C.
```

### 6.13 FINAL-1C：正式 MG2

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-1C official all-family MG2 synthetic acceptance.

Binding source:
Read MG2 in the original Implementation Plan; its exact package/config/report
outputs and acceptance scope are mandatory.

Prerequisite:
The exact frozen MG1 ontology and GOLDSET_WORKFLOW_READY manifest validate.

Implement and verify the original MG2 contract for all nine families, every
applicable phase, partials, invalidations, overlaps, threshold-bracketing hard
negatives and several approved scales. Preserve legal OHLCV/OI, deterministic
seeds, oracle recovery and transformation/future-mutation invariance. Do not
fit a model.

Outputs:
- configs/market_genome/synthetic_v1.yaml and the MG2 package/tests;
- $REPORTS/mg2_synthetic_contract.json/md;
- $REPORTS/market_genome/formal/final_1c_decision.json.

Acceptance: all targeted MG2 and prior MarketGenome tests run CPU-only and
exit 0; the decision manifest records acceptance_passed=true. Stop before
MG3. Only that exact manifest permits FINAL-1D.
```

### 6.14 FINAL-1D：正式 MG3-A

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-1D official MG3-A dense track specification.

Binding source:
Read MG3-A in the original Implementation Plan; its exact outputs are
mandatory.

Prerequisite:
Frozen ontology and formal MG2 manifests validate with exact hashes.

Implement the complete original MG3-A modality contract, not only pattern
tracks. Every target needs formula, unit, resolution, horizon, mask, scaler,
loss, baseline, metric and leakage audit; future-derived values are targets
only. Do not build the dataset or fit scalers.

Outputs:
- configs/market_genome/track_spec_v1.yaml and validation code/tests;
- $REPORTS/mg3_track_spec.json/md;
- $REPORTS/market_genome/formal/final_1d_decision.json.

Acceptance: CPU targeted and prior tests exit 0; decision is
TRACK_SPEC_READY_FOR_DATASET_BUILD. Stop before MG3-B. Only that exact manifest
permits FINAL-1E.
```

### 6.15 FINAL-1E：正式 MG3-B

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-1E official MG3-B causal dataset contract.

Binding source:
Read MG3-B in the original Implementation Plan; its builder/config/report
outputs and acceptance scope are mandatory.

Prerequisite:
Frozen ontology, MG2 and TRACK_SPEC_READY_FOR_DATASET_BUILD hashes validate;
calendar, roll and normalization policies are owner-approved.

Build the original MG3-B synthetic and small-real contracts with dense tracks,
four reproducible purged folds, event-boundary isolation, closed higher bars,
train-only scalers and complete fingerprints. Do not train a neural model.

Outputs are the original MG3-B builders/manifests plus
$REPORTS/mg3_dataset_contract.json/md and
$REPORTS/market_genome/formal/final_1e_decision.json.

Acceptance: all causality, leakage, scaler and prior tests exit 0 and the
decision records acceptance_passed=true. Stop before MG4-A. Only that exact
manifest permits FINAL-1F.
```

### 6.16 FINAL-1F：正式 MG4-A

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-1F official MG4-A mg_tiny causal backbone.

Binding source:
Read MG4-A in the original Implementation Plan; every listed module, test and
report output is mandatory.

Prerequisite:
Formal MG2 and MG3 decision manifests validate with exact hashes.

Implement the original configurable causal multi-scale mg_tiny backbone and
its shape, dense-output, prefix/future-mutation, variable-context and resolved
config tests. Do not start long training or implement MG4-B.

Current deferral:
Without explicit training restoration, run only source/config/static checks
that do not import JAX or model runtime, set TRAINING_DEFERRED, and do not claim
MG4-A pass. After authorization, the frozen forward and finite-gradient
backward smoke must pass under the approved device policy.

Required outputs:
- every MG4-A source/config/test path listed in the original plan;
- $REPORTS/mg4_backbone_contract.json/md;
- $REPORTS/market_genome/formal/final_1f_decision.json.

Acceptance:
Before authorization run only a dedicated non-runtime static test command.
After authorization run the exact frozen forward/backward command; exit 0 and
all causal tests are required for acceptance_passed=true. Stop before MG4-B.
Only acceptance_passed=true permits FINAL-1G.
```

### 6.17 FINAL-1G：正式 MG4-B

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-1G official MG4-B heads and losses.

Binding source:
Read MG4-B in the original Implementation Plan; every listed head/loss module,
test and report output is mandatory.

Prerequisite:
Formal MG3-A and MG4-A decision manifests validate with exact hashes.

Implement every original approved head/loss, including the past-only rolling
baseline plus zero-initialized residual return head, full masks, effective
weights and diagnostics. Do not run the MG5 experiment.

Current deferral:
Without explicit training restoration, run only static checks that do not
import JAX or model runtime, set TRAINING_DEFERRED, and do not claim MG4-B
pass. Authorized forward/loss/backward tests must pass before
acceptance_passed=true.

Required outputs:
- every MG4-B source/config/test path listed in the original plan;
- $REPORTS/mg4_heads_losses_contract.json/md;
- $REPORTS/market_genome/formal/final_1g_decision.json.

Acceptance:
Before authorization run only dedicated non-runtime static tests. After
authorization run the frozen head/loss/backward tests; exit 0 is required for
acceptance_passed=true. Stop before MG5. Only the exact accepted manifest
permits FINAL-1H.
```

### 6.18 FINAL-1H：MG6 promotion resolver

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-1H implement the fail-closed formal MG5 promotion resolver.

Prerequisite:
FINAL-1A through FINAL-1G manifests validate by explicit path and hash.

Required outputs:
- src/alphatrade/market_genome/mg5_promotion.py
- src/alphatrade/market_genome/mg5_promotion.schema.json
- tests/market_genome/test_mg5_promotion.py

The resolver accepts only the fixed formal result path
$REPORTS/market_genome/formal/mg5_result_manifest.json plus
configs/market_genome/gates/final_2_result_owner_signoff.json. It must reject
every SLICE_*, deferred, blocked, redesign and failure value; accept only exact
PASS_TO_REAL_PRETRAINING before validating all prerequisite hashes, nine-family
coverage, seeds, thresholds, no-NaN results and result-owner signoff. Any byte
change invalidates promotion.

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_mg5_promotion.py
Expected exit code: 0. Write
$REPORTS/market_genome/formal/final_1h_decision.json.

Stop before MG5. Only acceptance_passed=true permits FINAL-2; do not execute it.
```

### 6.19 FINAL-2：正式 MG5

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-2 official MG5 MarketGenome learnability gate.

Binding source:
Read MG5 in the original Implementation Plan; every listed experiment,
ablation, output and PASS condition remains mandatory.

First-run behavior:
If the exact formal sweep/preregistration or
configs/market_genome/gates/final_2_prereg_owner_signoff.json is absent, create
those artifacts with decision=PENDING plus
$REPORTS/market_genome/formal/mg5_prereg_stop_manifest.json, then stop before
test access with execution_status=EXECUTED and no formal MG5 decision.

Mandatory entry gates:
1. PROJECT_OWNER-signed ontology contains exactly the nine retained families.
2. Formal MG1-B, MG2, MG3 and MG4 manifests pass with immutable hashes.
3. Integrated MG2 covers every family, phase, mirror, dependency, overlap,
   invalidation, hard negative and the frozen scale set.
4. Integrated MG3 provides the complete approved modalities and train-only
   scalers, not only pattern tracks.
5. Integrated MG4 provides the approved causal backbone and all required
   heads/losses, including any approved rolling-residual return head.
6. Metrics, thresholds, seeds, held-out scales, ablations, compute budget and
   stop rules were owner-approved before test access.
7. FINAL-1H promotion resolver and its decision manifest pass.

If any requirement is missing, return BLOCKED_NOT_RUN. Do not shrink scope and
call it official MG5.

Safety gate:
Start no training or GPU work without explicit current-thread authorization
that training is resumed, the GPU is free and this exact run is approved.
Never kill another process; never silently fall back to CPU.

Run the frozen official ladder:
- one-batch overfit for every head;
- nine-family per-family/per-phase/anchor recovery;
- completion/invalidation event recovery with separate preregistered metrics;
- approved future-path and barrier-outcome synthetic recovery;
- multi-scale and held-out-scale generalization;
- mirror, overlap and dependency consistency;
- grouped shuffled-label controls;
- majority, ontology oracle, linear/logistic, MLP, small TCN, GRU and mg_tiny
  comparisons under equal folds/masks/budgets;
- context 512/1024/4096 ablations;
- learned strided convolution versus mean pooling versus max pooling;
- independent scale blocks versus shared scale core;
- identity conditioning on/off;
- residual return head versus absolute return head;
- prefix/future mutation and train-only normalization rechecks.

Report all seeds, failures, masks, supports, effective weights and baselines.
Synthetic success proves learnability, not alpha.

Formal MG5 decision classes, available only after the frozen run executes:
- PASS_TO_REAL_PRETRAINING
- PASS_WITH_REDESIGN
- FAIL_STOP

Non-decision stop outcomes:
- TRAINING_DEFERRED
- GPU_DEFERRED
- BLOCKED_NOT_RUN

Required outputs only after execution_status=EXECUTED:
- configs/market_genome/sweep_mg5_learnability.yaml
- configs/market_genome/gates/final_2_prereg_owner_signoff.json
- $REPORTS/mg5_sweep_manifest.json
- $REPORTS/mg5_leaderboard.json/md
- $REPORTS/mg5_learnability_report.json/md
- $REPORTS/MG5_LEARNABILITY_DECISION.md
- $REPORTS/market_genome/formal/mg5_result_manifest.json
- configs/market_genome/gates/final_2_result_owner_signoff.json with
  decision=PENDING and the exact result-manifest hash.

If blocked or deferred, write only
$REPORTS/market_genome/formal/mg5_stop_manifest.json; do not create a formal
decision, leaderboard, result manifest or result-owner signoff.

Acceptance:
Run only the exact command/environment in the accepted formal preregistration.
It must exit 0, all required results must be finite and every preregistered
criterion must resolve without post-test edits.

PASS requires all preregistered criteria, several-scale recovery, held-out-scale
superiority to simple baselines, shuffled-label collapse, causal invariance and
demonstrated value beyond a small TCN on multi-scale tasks.

PASS_WITH_REDESIGN, TRAINING_DEFERRED, GPU_DEFERRED, BLOCKED_NOT_RUN and
FAIL_STOP do not unlock MG6. After an executed run, generate the formal outputs
and stop for PROJECT_OWNER review of the result hash. Do not run FINAL-3 or MG6.
```

### 6.20 FINAL-3：正式 promotion resolution

```text
You are Codex working inside AlphaTrade MarketGenome.

Mandatory setup:
Read AGENTS.md plus sections 3 and 5 of
docs/market_genome/AlphaTrade_MarketGenome_Incremental_Vertical_Slice_Plan_and_Codex_Prompts.md,
inspect git status, preserve unrelated work, and treat those rules as part of
this prompt.

Task: FINAL-3 resolve the formal MG5 result for MG6 promotion.

Prerequisites:
The fixed mg5_result_manifest path/hash validates, its formal decision is
PASS_TO_REAL_PRETRAINING, final_2_result_owner_signoff.json is ACCEPT for that
exact hash, and FINAL-1H code/tests are unchanged. Otherwise write only a stop
manifest and set BLOCKED_NOT_RUN.

Run the FINAL-1H resolver without training or GPU access. Write only
$REPORTS/market_genome/formal/mg5_promotion_manifest.json, including every
verified prerequisite hash and mg6_allowed boolean.

Acceptance command:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "$MG_CPU_PYTHON" -m pytest -q tests/market_genome/test_mg5_promotion.py
Expected exit code: 0. MG6 is permitted only when the resolver exits 0 and the
new manifest contains mg6_allowed=true. Stop before implementing MG6.
```

---

## 7. 每阶段停止条件

出现以下任一情况必须停止，不得用增加代码量掩盖问题：

- canonical detector/adapter 与 owner contract 状态或事件不一致；
- source、profile、contract、dataset 或 report hash 不匹配；
- ontology oracle 不能恢复 generator truth；
- future mutation 改变 prefix label、track 或 model output；
- generator metadata、padding、固定位置、mask 或 scale ID 成为捷径；
- shuffled labels 仍有显著预测能力；
- held-out seed/scale 被用于调参；
- one-batch overfit 在确认数据、mask 和 loss 正确后仍失败；
- 新 family 改变已有 family 的 golden labels 或 stable IDs；
- 当前阶段需要训练或 GPU，但未获得对应恢复授权；
- 专项结果被尝试用于解锁 MG6。

---

## 8. 当前立即执行范围

当前只执行 `DT-0`。此前零 hash、无测试的 `double_top_pilot_v1.yaml` 和 `double_top_pilot.py` 草稿已删除，不能作为实现证据或恢复起点。DT-0 应从已提交的 owner ledger、真实 detector 和仓库现有 extension points 开始，先完成最小合同、adapter、定向测试和 `PENDING` owner-signoff artifact。

在 DT-0 通过并完成负责人审核前，不创建 MG2、MG3、MG4 或 MG5 代码。

DT-0 完成后的唯一下一步是 DT-1，不是全量 ontology 或第二个 family。
