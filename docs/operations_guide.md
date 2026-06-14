# AlphaTrade 全流程操作指南

**日期**: 2026-03-03
**适用版本**: M0-M9 基础设施

---

## 目录

1. [流程全景](#1-流程全景)
2. [前置条件](#2-前置条件)
3. [第一阶段：初始训练 → 建立基准](#3-第一阶段初始训练--建立基准)
4. [第二阶段：设计 ablation → 迭代训练](#4-第二阶段设计-ablation--迭代训练)
5. [第三阶段：升级基准 → 重复迭代](#5-第三阶段升级基准--重复迭代)
6. [第四阶段：导出最终模型 → 批量推理](#6-第四阶段导出最终模型--批量推理)
7. [完整示例：从零到交付](#7-完整示例从零到交付)
8. [可调参数一览](#8-可调参数一览)
9. [FAQ](#9-faq)

---

## 1. 流程全景

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ① 初始训练          ② 冻结基准          ③ 设计 ablation        │
│  (sweep baseline)    (freeze baseline)   (编辑 YAML)           │
│       │                    │                    │               │
│       ▼                    ▼                    ▼               │
│  ┌─────────┐        ┌───────────┐        ┌───────────┐         │
│  │ 3-seed  │───────▶│  基准冻结  │───────▶│  迭代循环  │         │
│  │ 训练+评估│        │  报告生成  │        │  sweep →   │         │
│  └─────────┘        └───────────┘        │  回归检测  │         │
│                                          │  → 门禁    │         │
│                                          └─────┬─────┘         │
│                                                │               │
│                           ┌────────────────────┤               │
│                           │                    │               │
│                    ④ 赢家出现？          ⑤ 未改善？            │
│                    (leaderboard         (继续设计              │
│                     排名第一)            新 ablation)          │
│                           │                    │               │
│                           ▼                    └──────▶ ③      │
│                    ┌───────────┐                                │
│                    │ 升级基准   │────────────────────▶ ③        │
│                    │ (bump ver) │                               │
│                    └───────────┘                                │
│                           │                                    │
│                    ⑥ 收敛？（无法再改善）                        │
│                           │                                    │
│                           ▼                                    │
│                    ┌───────────┐                                │
│                    │ 导出 champion │                             │
│                    │ 批量推理     │                              │
│                    └───────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**核心循环**: ③ → ④ → ⑤（或升级基准回到 ③）

当前基础设施 **完全支持** 上述流程。每一步都有对应脚本，只需编辑 YAML 配置即可驱动。

---

## 2. 前置条件

### 环境

```bash
# 所有命令都在此 prefix 下运行
export RUN="conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH"

# 训练/评估/推理产物默认写到仓库同级目录，避免污染 alphagenome_research
export ALPHATRADE_RUNS_ROOT="$(pwd)/../alphatrade_runs/default"
export REPORTS="$ALPHATRADE_RUNS_ROOT/reports"
export CHECKPOINTS="$ALPHATRADE_RUNS_ROOT/checkpoints"
export ARTIFACTS="$ALPHATRADE_RUNS_ROOT/artifacts"
```

### 数据

确保数据已处理完毕：

```bash
ls data/processed/m1_f8/DCE.JM/bars.parquet  # 应存在
```

### 目录结构与产物隔离

```
alphagenome_research/
├── configs/
│   ├── dataset/m2.yaml              # 数据集配置（固定）
│   ├── sweep/                       # sweep 配置（每轮迭代一个）
│   └── universe/m1_selected.yaml    # 品种列表
└── src/alphatrade/scripts/          # 所有脚本

alphatrade_runs/
└── default/
    ├── reports/                     # metrics、leaderboard、schema validation
    ├── checkpoints/                 # 训练 checkpoint
    └── artifacts/model_bundle/      # 最终模型 bundle
```

脚本默认读取 `ALPHATRADE_RUNS_ROOT`；未设置时使用仓库同级的 `../alphatrade_runs/default`。
需要分轮隔离时，直接改 `ALPHATRADE_RUNS_ROOT`，例如：

```bash
export ALPHATRADE_RUNS_ROOT="$(pwd)/../alphatrade_runs/round1"
```

---

## 3. 第一阶段：初始训练 → 建立基准

**目标**: 用默认超参训练 baseline，冻结为不可变的基准。

### 3.1 创建 sweep 配置

创建 `configs/sweep/round1.yaml`：

```yaml
version: round1_v1
baseline_exp_id: baseline
expected_seeds: [42, 43, 44]
primary_metric: "pinball_loss.overall"
dataset_config: "configs/dataset/m2.yaml"
universe: "cta_top20"
dataset: "m2"

defaults:
  max_steps: 500       # 初始训练步数（正式建议 1000-5000）
  batch_size: 128
  jit: 1
  clip_norm: 1.0
  save_every: 100
  keep_last: 3
  eval_split: val
  ckpt_step: best

experiments:
  - exp_id: baseline
    description: "初始基准"
    overrides: {}
```

### 3.2 Smoke Test（验证流程通畅）

```bash
$RUN python src/alphatrade/scripts/run_m5_sweep.py \
  --sweep-config configs/sweep/round1.yaml --smoke --resume
```

- `--smoke`：只用 3 个品种、3 步训练，约 2 分钟完成
- 确认无报错后再跑全量

### 3.3 全量训练 baseline

```bash
$RUN python src/alphatrade/scripts/run_m5_sweep.py \
  --sweep-config configs/sweep/round1.yaml
```

输出：
- `$REPORTS/m5_baseline_seed{42,43,44}_{train,eval}_metrics.json`
- `$REPORTS/m5_sweep_manifest.json`
- `$REPORTS/m5_leaderboard.json`

### 3.4 冻结基准

```bash
$RUN python src/alphatrade/scripts/gen_m6_baseline_report.py
```

输出：`$REPORTS/m6_baseline_run.md`（记录基准口径：git_sha、config_hash、run_id、指标）

### 3.5 验证

```bash
$RUN python src/alphatrade/scripts/validate_reports_schema.py --profile m6 --strict
```

**检查点**：记录 baseline 的 `primary_mean`，这是后续所有比较的参照。

---

## 4. 第二阶段：设计 ablation → 迭代训练

**目标**: 在基准之上尝试不同超参，通过回归报告判定是否有改善。

### 4.1 设计 ablation 实验

编辑 sweep 配置，添加 ablation 实验。创建 `configs/sweep/round1_ablation.yaml`：

```yaml
version: round1_ablation_v1
baseline_exp_id: baseline
expected_seeds: [42, 43, 44]
primary_metric: "pinball_loss.overall"
dataset_config: "configs/dataset/m2.yaml"
universe: "cta_top20"
dataset: "m2"

defaults:
  max_steps: 500
  batch_size: 128
  jit: 1
  clip_norm: 1.0
  save_every: 100
  keep_last: 3
  eval_split: val
  ckpt_step: best

experiments:
  - exp_id: baseline
    description: "初始基准（不变）"
    overrides: {}

  # --- 以下是你要尝试的变体 ---
  - exp_id: batch_256
    description: "加大 batch size"
    overrides: {batch_size: 256}

  - exp_id: steps_2000
    description: "增加训练步数"
    overrides: {max_steps: 2000}

  - exp_id: lr_3e4
    description: "提高学习率"
    overrides: {}
    # 注意：learning_rate 当前不在 sweep overrides 支持列表中
    # 仅限 run_m5_sweep.py 传递的参数，见第 8 节
```

> **重要**: `experiments` 列表中必须包含 `baseline`（overrides 为空），否则回归报告无法找到基准。

### 4.2 一键迭代

```bash
$RUN python src/alphatrade/scripts/run_iteration.py \
  --sweep-config configs/sweep/round1_ablation.yaml \
  --profile m7 \
  --resume --strict
```

此命令自动执行：
1. **Sweep**: 训练+评估所有 (exp_id × seed)，生成 leaderboard
2. **Regression report**: 每个 ablation 与 baseline 对比，输出 verdict
3. **Gate**: schema + semantic 验证

`--resume` 会跳过 baseline 已有的 3 个 run（从上一轮结果复用），只训练新增的 ablation。

### 4.3 查看结果

```bash
cat "$REPORTS/m5_leaderboard.md"
```

示例输出：
```
| Rank | Experiment | Mean   | Std    | Best   | Best Run | Seeds |
|------|-----------|--------|--------|--------|----------|-------|
| 1    | batch_256 | 0.1334 | 0.0117 | 0.1231 | 9169f783 | 3/3   |
| 2    | baseline  | 0.1342 | 0.0106 | 0.1254 | 38a265ee | 3/3   |
| 3    | steps_2000| 0.1350 | 0.0120 | 0.1240 | abcd1234 | 3/3   |
```

```bash
cat "$REPORTS/m7_regression_report.md"
```

示例输出：
```
| Experiment | Mean   | Delta    | Delta % | Verdict      |
|------------|--------|----------|---------|--------------|
| batch_256  | 0.1334 | -0.0008  | -0.59%  | neutral      |
| steps_2000 | 0.1350 | +0.0008  | +0.60%  | neutral      |
```

### 4.4 决策

根据 leaderboard 和 regression report：

| 情况 | 行动 |
|------|------|
| 有实验 verdict=`improved`（下降 >= 1%） | → 进入第三阶段，升级基准 |
| 排名第一但 verdict=`neutral` | → 可选择升级，或继续设计新 ablation |
| 全部 `neutral` 或 `regressed` | → 设计新的 ablation 方向，回到 4.1 |

---

## 5. 第三阶段：升级基准 → 重复迭代

**目标**: 把赢家的配置吸收为新基准，基于新基准继续探索。

### 5.1 确认赢家

查看 leaderboard 排名第一的实验：

```bash
# 用 jq 提取冠军
cat "$REPORTS/m5_leaderboard.json" | python -c "
import json, sys
lb = json.load(sys.stdin)
winner = lb['experiments'][0]
print(f\"Winner: {winner['exp_id']}\")
print(f\"  primary_mean: {winner['metrics']['primary_mean']:.6f}\")
print(f\"  primary_std:  {winner['metrics']['primary_std']:.6f}\")
for run in winner['artifacts']['runs']:
    print(f\"  seed={run['seed']} run_id={run['run_id']}\")
"
```

### 5.2 创建新一轮 sweep 配置

假设赢家是 `batch_256`（overrides: `{batch_size: 256}`），创建 `configs/sweep/round2.yaml`：

```yaml
version: round2_v1             # ← 版本号 bump
baseline_exp_id: baseline
expected_seeds: [42, 43, 44]
primary_metric: "pinball_loss.overall"
dataset_config: "configs/dataset/m2.yaml"
universe: "cta_top20"
dataset: "m2"

defaults:
  max_steps: 500
  batch_size: 256              # ← 吸收上轮赢家的配置
  jit: 1
  clip_norm: 1.0
  save_every: 100
  keep_last: 3
  eval_split: val
  ckpt_step: best

experiments:
  - exp_id: baseline
    description: "Round 2 基准（batch_size=256）"
    overrides: {}              # ← 新的 baseline 就是上轮赢家

  # --- 基于新基准设计新的 ablation ---
  - exp_id: steps_2000
    description: "在新基准上增加训练步数"
    overrides: {max_steps: 2000}

  - exp_id: no_clip
    description: "在新基准上去掉梯度裁剪"
    overrides: {clip_norm: 1000000000.0}

  - exp_id: batch_512
    description: "继续加大 batch size"
    overrides: {batch_size: 512}
```

### 5.3 切换产物目录 + 全量重跑

> **关键**: 升级基准后 defaults 变了，config_hash 会变化。必须不带 `--resume` 全量重跑，否则会复用旧的 baseline 结果（config_hash 不匹配）。

```bash
# 每轮使用独立 runs root，旧报告自然保留在 round1/default 中
export ALPHATRADE_RUNS_ROOT="$(pwd)/../alphatrade_runs/round2"
export REPORTS="$ALPHATRADE_RUNS_ROOT/reports"
export CHECKPOINTS="$ALPHATRADE_RUNS_ROOT/checkpoints"
export ARTIFACTS="$ALPHATRADE_RUNS_ROOT/artifacts"

# 全量重跑（不带 --resume）
$RUN python src/alphatrade/scripts/run_iteration.py \
  --sweep-config configs/sweep/round2.yaml \
  --profile m7 \
  --strict
```

### 5.4 重新冻结基准

```bash
$RUN python src/alphatrade/scripts/gen_m6_baseline_report.py
```

### 5.5 查看结果 → 决策 → 继续迭代

重复第 4.3 和 4.4 的流程。如果排名第一的实验再次优于新基准，回到 5.2 继续升级。

**迭代终止条件**:
- 连续 2-3 轮所有 ablation 均为 `neutral`
- primary_mean 改善幅度 < 0.1%
- 训练预算用完

---

## 6. 第四阶段：导出最终模型 → 批量推理

**目标**: 将 leaderboard 排名第一的实验导出为可部署的 model bundle，运行批量推理。

### 6.1 导出 champion bundle

```bash
$RUN python src/alphatrade/scripts/export_model_bundle.py
```

自动选择 leaderboard 排名第一的实验的 best seed。如需手动指定：

```bash
$RUN python src/alphatrade/scripts/export_model_bundle.py \
  --exp-id batch_256 --run-id 9169f783
```

输出：
- `$ARTIFACTS/model_bundle/alphatrade_v0.2_<exp_id>_<run_id>/` — 模型 bundle
- `$REPORTS/m9_model_bundle_manifest.json` — 溯源记录

### 6.2 批量推理

`batch_infer_offline.py` 默认清理 `LD_LIBRARY_PATH`，避免 JAX CUDA 插件加载到错误版本的 cuSPARSE/cuDNN。确实需要保留该变量时，设置 `ALPHATRADE_KEEP_LD_LIBRARY_PATH=1`。

```bash
# Smoke test（快速验证）
$RUN python src/alphatrade/scripts/batch_infer_offline.py \
  --bundle "$ARTIFACTS/model_bundle/alphatrade_v0.2_batch_256_9169f783" \
  --data-dir data/processed/m1_f8 \
  --symbols DCE.JM,SHFE.AG \
  --start 2024-01-02 --end 2024-01-04 \
  --smoke

# 全量推理（所有品种 × 完整时间范围）
$RUN python src/alphatrade/scripts/batch_infer_offline.py \
  --bundle "$ARTIFACTS/model_bundle/alphatrade_v0.2_batch_256_9169f783" \
  --data-dir data/processed/m1_f8 \
  --symbols DCE.JM,SHFE.AG,CZCE.MA,DCE.PP,SHFE.CU,... \
  --start 2024-01-02 --end 2024-12-31
```

输出：
- `$REPORTS/m9_predictions.parquet` — 宽表格式预测（symbol × eob × 20 prediction columns）
- `$REPORTS/m9_infer_metrics.json` + `.md` — 推理统计

### 6.3 最终门禁

```bash
$RUN python src/alphatrade/scripts/validate_reports_schema.py --profile m9 --strict
```

---

## 7. 完整示例：从零到交付

以下是一个 3 轮迭代的完整操作序列。

```bash
export RUN="conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH"
export ALPHATRADE_RUNS_ROOT="$(pwd)/../alphatrade_runs/round1"
export REPORTS="$ALPHATRADE_RUNS_ROOT/reports"
export CHECKPOINTS="$ALPHATRADE_RUNS_ROOT/checkpoints"
export ARTIFACTS="$ALPHATRADE_RUNS_ROOT/artifacts"

# ========================================
# Round 1: 建立初始基准
# ========================================

# 创建 configs/sweep/round1.yaml（内容见第 3.1 节）

# Smoke test
$RUN python src/alphatrade/scripts/run_m5_sweep.py \
  --sweep-config configs/sweep/round1.yaml --smoke

# 全量训练 baseline（3 seeds × 500 steps ≈ 30-60 分钟）
$RUN python src/alphatrade/scripts/run_m5_sweep.py \
  --sweep-config configs/sweep/round1.yaml

# 冻结基准
$RUN python src/alphatrade/scripts/gen_m6_baseline_report.py

# 验证
$RUN python src/alphatrade/scripts/validate_reports_schema.py --profile m6 --strict

# ✅ Round 1 完成，记录 baseline primary_mean


# ========================================
# Round 1 Ablation: 挑战基准
# ========================================

# 创建 configs/sweep/round1_ablation.yaml（内容见第 4.1 节）
# 添加 3 个 ablation: batch_256, steps_2000, no_clip

# 一键迭代（复用 baseline，只训练新增的 3×3=9 个 run）
$RUN python src/alphatrade/scripts/run_iteration.py \
  --sweep-config configs/sweep/round1_ablation.yaml \
  --profile m7 --resume --strict

# 查看结果
cat "$REPORTS/m5_leaderboard.md"
cat "$REPORTS/m7_regression_report.md"

# 假设结果：batch_256 排名第一，delta=-0.59%，verdict=neutral
# 决策：虽然未达 improved 阈值，但确实排名第一 → 升级基准


# ========================================
# Round 2: 升级基准 + 新 ablation
# ========================================

# 切换到新一轮独立 runs root，旧报告保留在 ../alphatrade_runs/round1
export ALPHATRADE_RUNS_ROOT="$(pwd)/../alphatrade_runs/round2"
export REPORTS="$ALPHATRADE_RUNS_ROOT/reports"
export CHECKPOINTS="$ALPHATRADE_RUNS_ROOT/checkpoints"
export ARTIFACTS="$ALPHATRADE_RUNS_ROOT/artifacts"

# 创建 configs/sweep/round2.yaml
# defaults.batch_size: 256（吸收赢家）
# 新 ablation: steps_2000, no_clip, batch_512

# 全量重跑（不带 --resume，因为 defaults 变了）
$RUN python src/alphatrade/scripts/run_iteration.py \
  --sweep-config configs/sweep/round2.yaml \
  --profile m7 --strict

# 重新冻结基准
$RUN python src/alphatrade/scripts/gen_m6_baseline_report.py

# 查看结果
cat "$REPORTS/m5_leaderboard.md"
cat "$REPORTS/m7_regression_report.md"

# 假设结果：steps_2000 排名第一，delta=-1.5%，verdict=improved! ✅


# ========================================
# Round 3: 再次升级基准
# ========================================

# 切换到 round3 产物目录
export ALPHATRADE_RUNS_ROOT="$(pwd)/../alphatrade_runs/round3"
export REPORTS="$ALPHATRADE_RUNS_ROOT/reports"
export CHECKPOINTS="$ALPHATRADE_RUNS_ROOT/checkpoints"
export ARTIFACTS="$ALPHATRADE_RUNS_ROOT/artifacts"

# 创建 configs/sweep/round3.yaml
# defaults: batch_size=256, max_steps=2000（吸收两轮赢家）
# 新 ablation: steps_5000, lr_5e5, batch_512

$RUN python src/alphatrade/scripts/run_iteration.py \
  --sweep-config configs/sweep/round3.yaml \
  --profile m7 --strict

$RUN python src/alphatrade/scripts/gen_m6_baseline_report.py

# 假设结果：所有 ablation 都是 neutral → 收敛，停止迭代


# ========================================
# 最终交付：导出 champion + 批量推理
# ========================================

$RUN python src/alphatrade/scripts/export_model_bundle.py

$RUN python src/alphatrade/scripts/batch_infer_offline.py \
  --bundle "$ARTIFACTS/model_bundle/alphatrade_v0.2_baseline_<run_id>" \
  --data-dir data/processed/m1_f8 \
  --symbols DCE.JM,SHFE.AG,CZCE.MA \
  --start 2024-01-02 --end 2024-12-31

$RUN python src/alphatrade/scripts/validate_reports_schema.py --profile m9 --strict

# ✅ 全流程完成
```

---

## 8. 可调参数一览

### sweep 配置中 defaults / overrides 支持的参数

以下参数通过 `run_m5_sweep.py` 传递给 `train_m4_alphatrade.py` / `eval_m4_fast.py`：

**训练参数（`_TRAIN_PARAM_MAP`）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_steps` | 500 | 训练步数 |
| `batch_size` | 128 | Batch size |
| `jit` | 1 | JIT 编译（0/1） |
| `clip_norm` | 1.0 | 梯度裁剪范数 |
| `save_every` | 100 | Checkpoint 保存间隔 |
| `keep_last` | 3 | 保留最近 N 个 checkpoint |
| `learning_rate` | *config YAML* | 学习率（仅在 overrides 中设置时传递，覆盖 config YAML） |
| `weight_decay` | *config YAML* | 权重衰减（同上） |
| `val_every` | *config YAML* | 验证间隔（同上） |

**评估参数（`_EVAL_PARAM_MAP`）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 128 | 评估 Batch size |
| `eval_split` | val | 评估数据集划分 |
| `ckpt_step` | best | 评估用的 checkpoint（best/last） |

> **添加新参数**：只需两步——
> 1. 在 `train_m4_alphatrade.py` 的 `parse_args()` 中添加 CLI 参数
> 2. 在 `run_m5_sweep.py` 的 `_TRAIN_PARAM_MAP`（或 `_EVAL_PARAM_MAP`）中添加一行 `(yaml_key, cli_flag, default_or_None)`
>
> 如果 `default_or_None` 设为 `None`，则该参数仅在 sweep overrides 中显式设置时才传递，训练脚本使用自身或 config YAML 的默认值。

### 回归判定阈值

在 `build_m7_regression_report.py` 中硬编码（也可通过参数覆盖）：

| 阈值 | 值 | 说明 |
|------|-----|------|
| `improvement_pct` | 1.0% | pinball_mean 下降 >= 此值 → improved |
| `regression_pct` | 5.0% | pinball_mean 上升 >= 此值 → regressed |

### 训练规模参考

| 场景 | max_steps | 预计时间（RTX 4060 Ti） |
|------|-----------|----------------------|
| Smoke test | 3 | ~1 分钟 |
| 快速验证 | 50-100 | ~5 分钟 |
| 标准训练 | 500 | ~15-30 分钟 |
| 深度训练 | 2000-5000 | ~1-4 小时 |
| 全量训练 | 10000+ | ~8 小时以上 |

每轮迭代（4 experiments × 3 seeds = 12 runs）总时间 ≈ 单 run 时间 × 12。

---

## 9. FAQ

### Q: 升级基准时为什么要切换 runs root？

`run_m5_sweep.py` 用固定的文件名模式（`m5_{exp_id}_seed{seed}_*.json`），如果旧的 exp_id 与新的相同但配置不同，`--resume` 会错误地复用旧结果。
现在推荐每轮使用独立的 `ALPHATRADE_RUNS_ROOT`，这样无需移动文件，也不会把训练产物写进源码仓库。

### Q: 可以同时跑多轮吗？

不建议。每轮迭代都依赖上一轮的 leaderboard 结果做决策。串行执行能确保每一步的决策基于最新数据。

### Q: 如何回滚到某一轮的状态？

每轮对应一个独立目录，例如 `../alphatrade_runs/round2`。回滚时把 `ALPHATRADE_RUNS_ROOT` 指回对应目录即可。

### Q: `--resume` 什么时候安全？

- **安全**: 在同一轮内中断后恢复（配置未变）
- **安全**: 添加新 experiment 但不修改 defaults 和已有 experiment 的 overrides
- **不安全**: 修改了 defaults 或已有 experiment 的 overrides → 必须不带 `--resume`

### Q: 如何添加新的可调参数（如 learning_rate）？

1. 确认 `train_m4_alphatrade.py` 支持对应 CLI 参数
2. 编辑 `run_m5_sweep.py` 的 `run_single_train()` 函数，添加参数传递
3. 在 sweep config 的 overrides 中使用新参数

### Q: 多少轮迭代是合理的？

通常 3-5 轮即可收敛。标志是连续 1-2 轮所有 ablation 均为 neutral 且 leaderboard 排名无变化。

### Q: predictions.parquet 的格式是什么？

宽表，每行一个 (symbol, eob) 时间点：

```
symbol | eob | model_version | h1_q10 | h1_q30 | h1_q50 | h1_q70 | h1_q90 | h5_q10 | ... | h60_q90
```

共 23 列：3 元数据列 + 20 预测列（4 horizons × 5 quantiles）。

---

## 脚本速查

| 步骤 | 脚本 | 主要输出 |
|------|------|----------|
| 训练+评估 sweep | `run_m5_sweep.py` | m5_sweep_manifest.json, m5_leaderboard.json |
| 冻结基准 | `gen_m6_baseline_report.py` | m6_baseline_run.md |
| 一键迭代 | `run_iteration.py` | sweep + regression + gate |
| 回归报告 | `build_m7_regression_report.py` | m7_regression_report.json |
| Schema 验证 | `validate_reports_schema.py` | {profile}_schema_validation.json |
| 导出 champion | `export_model_bundle.py` | artifacts/model_bundle/ under `ALPHATRADE_RUNS_ROOT` |
| 批量推理 | `batch_infer_offline.py` | m9_predictions.parquet |
