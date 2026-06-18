# M5 Reports Contract

生成时间: 2026-03-03

---

## Required 文件列表（5 个）

| # | name | path | schema | required |
|---|------|------|--------|----------|
| 1 | m5_sweep_manifest | `$ALPHATRADE_RUNS_ROOT/reports/m5_sweep_manifest.json` | `m5_sweep_manifest.schema.json` | yes |
| 2 | m5_leaderboard | `$ALPHATRADE_RUNS_ROOT/reports/m5_leaderboard.json` | `m5_leaderboard.schema.json` | yes |
| 3 | m5_leaderboard_md | `$ALPHATRADE_RUNS_ROOT/reports/m5_leaderboard.md` | (existence only) | yes |
| 4 | m5_schema_validation | `$ALPHATRADE_RUNS_ROOT/reports/m5_schema_validation.json` | (existence only) | yes |
| 5 | m5_schema_validation_md | `$ALPHATRADE_RUNS_ROOT/reports/m5_schema_validation.md` | (existence only) | yes |

---

## Schema 复用策略

- **train metrics**: 复用 `m2_train_metrics_v1` schema（与 M2/M3/M4 共用）
- **eval metrics**: 复用 `m4_eval_metrics_v1` schema
- **新增 schemas**:
  - `m5_sweep_manifest_v1` — sweep manifest 顶层结构 + runs 数组
  - `m5_leaderboard_v1` — 聚合 leaderboard 结构

---

## 命名规范

M5 不再硬编码 seed 文件名。Run 产物路径由 `m5_sweep_manifest.json` 中的 `runs[].train_metrics_path` 和 `runs[].eval_metrics_path` 引用。

示例 run 产物路径:
```
$ALPHATRADE_RUNS_ROOT/reports/m5_baseline_seed42_train_metrics.json
$ALPHATRADE_RUNS_ROOT/reports/m5_baseline_seed42_eval_metrics.json
```

validator 通过 sweep_manifest 中声明的路径动态发现并校验每个 run。

---

## 严格门禁定义

### Phase 1: Schema Validation

5 个 required 文件全部存在且过 schema（有 schema 的过 schema，无 schema 的检查存在性）。

### Phase 2: Semantic Checks（sweep 内 run 产物）

对 `m5_sweep_manifest.json` 中声明的每个 run：

1. **文件存在**: `train_metrics_path` 和 `eval_metrics_path` 指向的文件必须存在
2. **Schema 合规**: train 文件过 `m2_train_metrics.schema.json`，eval 文件过 `m4_eval_metrics.schema.json`
3. **Seed 覆盖**: 每个 `exp_id` 的 runs 必须覆盖 `expected_seeds` 中的所有 seed（缺 seed = **strict fail**）
4. **Config hash 一致**: 同一 `exp_id` 下所有 run 的 `config_hash` 必须相同（不一致 = **strict fail**）
5. **Run ID 唯一**: 所有 `run_id` 全局唯一
6. **Artifact metadata 一致**: train/eval JSON 内嵌的 sweep metadata 必须与 manifest 中的
   `exp_id`, `seed`, `config_hash`, `dataset_config`, `universe`, `eval_split`, `ckpt_step`
   完全一致；不允许用旧配置产物替换当前 manifest。

---

## 与 M4 的关系

- M4 profile 不变，M5 profile 独立
- M5 eval metrics 复用 M4 schema，保持向后兼容
- CI 中 M4 和 M5 门禁各自独立运行
