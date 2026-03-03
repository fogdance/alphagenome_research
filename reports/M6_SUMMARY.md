# M6 总结：Baseline Freeze（冻结基准 & 基准口径）

**日期**: 2026-03-03
**状态**: ✅ 完成
**Git SHA**: `acd0e21`

---

## 执行摘要

M6 建立了可复现的基准口径：将 M5 sweep 基础设施扩展为 3-seed 基准实验，冻结训练配置、数据集划分和评估方法，为后续 ablation（M7）和迭代（M8）提供不可变的参考线。

---

## 核心成果

### 1. 基准冻结

| 项目 | 值 |
|------|-----|
| 模型 | AlphaTrade v0.2 |
| max_steps | 500 |
| batch_size | 128 |
| clip_norm | 1.0 |
| Seeds | [42, 43, 44] |
| Primary metric | `pinball_loss.overall`（越低越好） |
| **Mean** | **0.134178** |
| **Std** | **0.010631** |
| **Best** | **0.125448**（seed=42, run_id=38a265ee） |

### 2. Per-Seed 结果

| Seed | Run ID | Pinball Loss | IC | Rank IC | Crossing Rate |
|------|--------|--------------|-----|---------|---------------|
| 42 | 38a265ee | 0.125448 | -0.009325 | -0.007650 | 0.0000 |
| 43 | e2eb17a4 | 0.131068 | 0.016028 | 0.008861 | 0.0000 |
| 44 | b6c00273 | 0.146016 | -0.007615 | -0.008672 | 0.0000 |

### 3. Sweep 基础设施

M6 复用了 M5 sweep 基础设施，引入了关键脚本：

- `src/alphatrade/scripts/run_m5_sweep.py` — 多实验 × 多 seed 批量训练 + 评估
- `src/alphatrade/scripts/gen_m6_baseline_report.py` — 生成基准冻结报告
- `configs/sweep/m5.yaml` — 基准 sweep 配置

### 4. 契约治理

- 新增 `m5`、`m6` profile 到 `contracts_manifest.yaml`
- M5 semantic checks：run_id 唯一性、config_hash 一致性、seed 覆盖、文件存在性
- M6 profile 继承 M5 全部检查，额外要求 `m6_baseline_run.md` 存在

---

## 交付物

### 新增脚本（2）
- `src/alphatrade/scripts/run_m5_sweep.py`
- `src/alphatrade/scripts/gen_m6_baseline_report.py`

### 新增配置（1）
- `configs/sweep/m5.yaml`

### 新增文档（2）
- `docs/m6_baseline_contract.md`
- `reports/m6_baseline_run.md`

### 数据输出（14）
- `reports/m5_sweep_manifest.json`
- `reports/m5_leaderboard.json` + `.md`
- `reports/m5_baseline_seed{42,43,44}_{train,eval}_metrics.json`（6 个）
- `reports/m5_baseline_seed{42,43,44}_{train,eval}_run.md`（6 个）

### Schema 验证输出（4）
- `reports/m5_schema_validation.json` + `.md`
- `reports/m6_schema_validation.json` + `.md`

---

## 验收

```bash
# 复现
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_m5_sweep.py --sweep-config configs/sweep/m5.yaml

conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/gen_m6_baseline_report.py

# 门禁
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py --profile m6 --strict
```

- [x] 3-seed 基准训练 + 评估完成
- [x] 基准口径冻结（m6_baseline_run.md 生成）
- [x] M5 profile schema 验证全绿
- [x] M6 profile schema 验证全绿
- [x] Leaderboard 正确排序

---

## 意义

M6 确立了「不可变基准」原则：后续任何实验改进都必须与这个冻结基准对比，确保回归检测有稳定的参照。
