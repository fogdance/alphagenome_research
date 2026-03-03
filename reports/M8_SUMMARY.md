# M8 总结：Iteration Loop（可重复迭代机制 + 回归门禁）

**日期**: 2026-03-03
**状态**: ✅ 完成
**Git SHA**: `22866ff`

---

## 执行摘要

M8 将 M5-M7 的三阶段流程（sweep → leaderboard → regression → gate）封装为一条命令驱动的迭代循环。新增 `run_iteration.py` 脚本，支持 `--resume` 断点续跑和 `--strict` 门禁模式。同时制定了基准升级策略（Versioned Upgrade）和实验分支工作流。

---

## 核心成果

### 1. `run_iteration.py` — 一条命令跑完全部

```bash
python src/alphatrade/scripts/run_iteration.py \
  --sweep-config configs/sweep/m7.yaml \
  --profile m7 \
  --resume --strict
```

自动链式调用：

1. `run_m5_sweep.py` — 训练 + 评估所有 (exp_id, seed)
2. `build_m7_regression_report.py` — 对比基准，生成回归报告
3. `validate_reports_schema.py` — schema + semantic 门禁

任一步骤失败则管线中止，返回非零退出码。

### 2. `--resume` 断点续跑

- 跳过已存在输出文件的 run
- 适用于长时间 sweep 中断后恢复
- 安全边界：配置变更后必须不带 `--resume` 重跑

### 3. `--dry-run` 预览

打印 sweep 计划后退出，不执行训练。

### 4. 基准升级策略

**规则**：改变 `defaults:` 会改变 `config_hash`，必须 bump `version` 并重建全部报告。

升级流程：
1. Bump sweep config `version`
2. 更新 `defaults:` 为新基准配置
3. 不带 `--resume` 全量重跑
4. 重新生成 baseline freeze record
5. 一次 commit 提交配置 + 报告

### 5. CI 门禁

`.github/workflows/presubmit_checks.yml` 新增 M5 和 M7 门禁 step，在 PR 中按 `REPORTS_PROFILE` 环境变量触发。

---

## 交付物

### 新增脚本（1）
- `src/alphatrade/scripts/run_iteration.py`

### 新增文档（1）
- `docs/m8_iteration_loop.md` — 完整的操作手册，包含 Quick Start、CLI Reference、分支策略

### 编辑
- `.github/workflows/presubmit_checks.yml` — 新增 M5、M7 CI gate
- `configs/sweep/m5.yaml` + `m7.yaml` — 微调

---

## 迭代循环全景

```
                    ┌──────────────┐
                    │  sweep config │
                    │  (YAML)       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ run_m5_sweep │  训练 + 评估
                    │ (per exp×seed)│
                    └──────┬───────┘
                           │
                    ┌──────▼────────────┐
                    │ build_m5_leaderboard│  排名
                    └──────┬────────────┘
                           │
                    ┌──────▼──────────────────┐
                    │ build_m7_regression_report│  回归检测
                    └──────┬──────────────────┘
                           │
                    ┌──────▼────────────────────┐
                    │ validate_reports_schema.py │  门禁
                    │ --profile m7 --strict     │
                    └──────┬────────────────────┘
                           │
                     ✅ PASS / ❌ FAIL
```

---

## 验收

```bash
# Smoke 验证
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_iteration.py \
  --sweep-config configs/sweep/m7.yaml --profile m7 --smoke --resume --strict

# 全量运行
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_iteration.py \
  --sweep-config configs/sweep/m7.yaml --profile m7 --resume --strict
```

- [x] `run_iteration.py` 一条命令跑通全部流程
- [x] `--resume` 正确跳过已完成 run
- [x] `--dry-run` 正确打印计划
- [x] `--strict` 门禁模式正确退出码
- [x] CI 门禁 step 已加入 workflow
- [x] 文档完整（操作手册 + 基准升级策略 + 分支工作流）

---

## 意义

M8 完成了从「手动分步执行」到「一键迭代循环」的转变。此后添加新实验只需编辑 YAML 配置，无需改动代码。迭代循环 + 回归门禁确保了实验质量的可持续性。
