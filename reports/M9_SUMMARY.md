# M9 总结：Champion Bundle + Offline Batch Inference

**日期**: 2026-03-03
**状态**: ✅ 完成
**Git SHA**: `22866ff`

---

## 执行摘要

M9 是最终交付里程碑：从 M5 leaderboard 自动选出冠军模型，打包为可复现的 bundle，并提供离线批量推理输出 `predictions.parquet`。整条链路通过 M9 profile 的 schema + semantic 门禁验证。

---

## 核心成果

### 1. Champion 选择

| 项目 | 值 |
|------|-----|
| 选择方法 | Leaderboard `primary_mean` 最低 |
| 冠军实验 | `batch_256` |
| 冠军 Run | `9169f783`（seed=42） |
| Primary Mean | 0.1334 |
| Primary Std | 0.0117 |
| Primary Best | 0.1231 |
| Config Hash | `f1d8a097` |
| Model Version | `alphatrade_v0.2_batch_256_9169f783` |

### 2. Model Bundle

自包含目录 `artifacts/model_bundle/alphatrade_v0.2_batch_256_9169f783/`：

| 文件 | 说明 |
|------|------|
| `best/` | Flax checkpoint（从 checkpoints/m5/ 复制） |
| `artifacts.json` | 训练元数据 |
| `model_config.json` | AlphaTradeConfig（含 stem_channels, num_encoder_stages 等完整字段） |
| `bundle_manifest.json` | 全量溯源记录（champion 选择方法、指标、源路径、git sha） |

### 3. 离线批量推理

Smoke 测试结果（DCE.JM, 2024-01-02 ~ 2024-01-04）：

| 指标 | 值 |
|------|-----|
| 样本数 | 690 |
| 预测数 | 13,800（690 × 4 horizons × 5 quantiles） |
| 耗时 | 26.1 秒 |
| 吞吐 | 26 samples/sec |
| 输出列数 | 23（symbol + eob + model_version + 20 prediction columns） |

### 4. predictions.parquet 格式

Wide format — 每行一个 (symbol, eob)：

```
symbol | eob                 | model_version | h1_q10 | h1_q30 | ... | h60_q90
-------+---------------------+---------------+--------+--------+-----+--------
DCE.JM | 2024-01-02 09:01:00 | alphatrade_.. | -0.001 | -0.000 | ... | 0.003
```

20 prediction columns：4 horizons (1, 5, 20, 60) × 5 quantiles (10, 30, 50, 70, 90)

### 5. M9 Semantic Validation

8/8 semantic checks 全部通过：

| Check | 状态 |
|-------|------|
| manifest_exists | ✅ |
| bundle_path_exists | ✅ |
| model_version_non_empty | ✅ |
| predictions_exists | ✅ |
| predictions_parquet_load | ✅ |
| predictions_rows_gt_0 | ✅ |
| predictions_columns_complete | ✅ |
| predictions_column_types | ✅ |

---

## 交付物

### 新增脚本（2）
- `src/alphatrade/scripts/export_model_bundle.py` — champion 导出
- `src/alphatrade/scripts/batch_infer_offline.py` — 离线批量推理

### 新增 Schema（3）
- `src/alphatrade/schemas/m9_model_bundle_manifest.schema.json` — bundle 溯源 schema
- `src/alphatrade/schemas/m9_predictions.schema.json` — predictions 逻辑 schema（列名 + 类型）
- `src/alphatrade/schemas/m9_infer_metrics.schema.json` — 推理指标 schema

### 新增文档（1）
- `docs/m9_champion_and_infer_contract.md` — 契约文档

### 编辑
- `src/alphatrade/schemas/contracts_manifest.yaml` — 新增 `m9` profile（4 items）
- `src/alphatrade/scripts/validate_reports_schema.py` — 新增 `semantic_check_m9()`
- `src/alphatrade/tests/test_reports_validator.py` — 新增 `TestM9SemanticChecks`（3 tests）
- `.github/workflows/presubmit_checks.yml` — 新增 M9 CI gate

### 输出
- `artifacts/model_bundle/alphatrade_v0.2_batch_256_9169f783/` — 模型 bundle
- `reports/m9_model_bundle_manifest.json` — bundle 溯源记录
- `reports/m9_predictions.parquet` — 预测输出
- `reports/m9_infer_metrics.json` + `.md` — 推理指标
- `reports/m9_schema_validation.json` + `.md` — 验证结果

---

## 验收

```bash
# 1. 导出 champion bundle
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/export_model_bundle.py

# 2. 离线推理（smoke）
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/batch_infer_offline.py \
  --bundle artifacts/model_bundle/alphatrade_v0.2_batch_256_9169f783 \
  --data-dir data/processed/m1_f8 \
  --symbols DCE.JM,SHFE.AG \
  --start 2024-01-02 --end 2024-01-04 \
  --output reports/m9_predictions.parquet --smoke

# 3. 门禁验证
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py --profile m9 --strict

# 4. 回归检查（M7 不受影响）
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py --profile m7 --strict

# 5. 单元测试
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python -m pytest src/alphatrade/tests/test_reports_validator.py
```

- [x] Champion 自动选择正确（batch_256, primary_mean 最低）
- [x] Bundle 目录完整（best/ + artifacts.json + model_config.json + bundle_manifest.json）
- [x] 离线推理成功输出 predictions.parquet（690 rows × 23 cols）
- [x] M9 profile schema 验证全绿（4/4 schema + 8/8 semantic）
- [x] M7 profile 不受影响（68/68 semantic）
- [x] 26/26 单元测试通过

---

## 实现中修复的问题

1. **model_config.json 不完整**：artifacts.json 缺少 `stem_channels` 和 `num_encoder_stages`，导致模型初始化 shape 不匹配。修复：从 `train_metrics.model.config` 取完整配置。
2. **Checkpoint 路径必须为绝对路径**：orbax 库要求绝对路径。修复：`Path.resolve()` 转换。

---

## 意义

M9 完成了从「实验循环」到「可交付模型」的最后一步。Champion bundle 包含完整溯源信息（选择方法、指标、源 checkpoint、git sha），predictions.parquet 提供了标准化的离线推理输出格式。

至此，M0-M9 全部里程碑完成，覆盖了从数据处理、模型训练、评估、超参搜索、基准冻结、ablation 测试、迭代循环到最终模型交付的完整链路。
