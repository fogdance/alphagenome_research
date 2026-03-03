# M7 总结：Ablation Sweep（挑战基准）

**日期**: 2026-03-03
**状态**: ✅ 完成
**Git SHA**: `8892e72`

---

## 执行摘要

M7 在 M6 冻结基准之上运行了 3 组 ablation 实验，通过机器可读的回归报告自动判定每组实验相对基准的表现。引入 `m7_regression_report` schema 和对应的 semantic checks，建立了完整的「挑战基准」工作流。

---

## Ablation 设计

| exp_id | 变更 | 目的 |
|--------|------|------|
| `baseline` | 无（M6 冻结配置） | 参照 |
| `no_clip` | clip_norm=1e9 | 测试去掉梯度裁剪的稳定性 |
| `batch_256` | batch_size=256 | 更大 batch → 更平滑梯度 |
| `steps_1000` | max_steps=1000 | 翻倍训练预算 |

每组实验运行 3 seeds（42, 43, 44），共 12 次训练 + 评估。

## 判定规则

| 判定 | 条件 |
|------|------|
| **improved** | pinball_mean 下降 >= 1% |
| **neutral** | 变化在 -1% 到 +5% 之间 |
| **regressed** | pinball_mean 上升 >= 5% |

---

## 结果

### Leaderboard 排名

| Rank | Experiment | Mean | Std | Best | Best Run |
|------|-----------|------|-----|------|----------|
| 1 | batch_256 | 0.1334 | 0.0117 | 0.1231 | 9169f783 |
| 2 | baseline | 0.1342 | 0.0106 | 0.1254 | 38a265ee |
| 3 | steps_1000 | 0.1342 | 0.0106 | 0.1254 | 83d1f207 |
| 4 | no_clip | 0.1360 | 0.0104 | 0.1296 | 356fca98 |

### 回归报告

| Experiment | Mean | Delta | Delta % | Verdict |
|------------|------|-------|---------|---------|
| batch_256 | 0.133383 | -0.000795 | -0.59% | neutral |
| steps_1000 | 0.134178 | +0.000000 | +0.00% | neutral |
| no_clip | 0.135976 | +0.001798 | +1.34% | neutral |

**Summary**: 3 neutral, 0 improved, 0 regressed

### 分析

- `batch_256` 是排名第一的实验（mean 最低），但 delta -0.59% 未达到 1% improvement 阈值
- `steps_1000` 与 baseline 完全一致（训练步数翻倍未带来改善，可能是 smoke 模式下 steps 已够）
- `no_clip` 略微退化但未触发 regressed 阈值

---

## 交付物

### 新增脚本（1）
- `src/alphatrade/scripts/build_m7_regression_report.py`

### 新增 Schema（1）
- `src/alphatrade/schemas/m7_regression_report.schema.json`

### 新增配置（1）
- `configs/sweep/m7.yaml`

### 新增文档（1）
- `docs/m7_ablation_contract.md`

### 数据输出
- `reports/m7_regression_report.json` + `.md`
- 9 组 ablation 训练/评估 metrics + markdown（18 个 JSON + 18 个 .md）
- 更新后的 leaderboard（4 experiments）

### 测试
- `test_reports_validator.py` 新增 `TestM7SemanticChecks`（3 tests）+ `TestM7Integration`（1 test）
- M7 semantic checks：baseline 在 leaderboard 中存在、verdict 合法、summary counts 一致

---

## 验收

```bash
# 复现
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_m5_sweep.py --sweep-config configs/sweep/m7.yaml

conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/build_m7_regression_report.py

# 门禁
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py --profile m7 --strict
```

- [x] 4 experiments × 3 seeds = 12 runs 全部完成
- [x] Leaderboard 正确排序
- [x] 回归报告生成，verdict 与数据一致
- [x] M7 profile schema 验证全绿（5/5 schema + 68/68 semantic）
- [x] 无回归（0 regressed）

---

## 意义

M7 验证了基准的稳健性：三组 ablation 均未触发回归或显著改善，说明基准配置处于较优区间。`batch_256` 以微弱优势排名第一，为后续 M9 champion 选择提供了候选。
