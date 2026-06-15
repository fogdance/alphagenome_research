# AlphaTrade M10 Evaluation + Calibration Hardening

**日期**: 2026-06-15  
**Sprint**: M10 Evaluation + Calibration Hardening  
**Run root**: `../alphatrade_runs/m10_evalcal_full_20260615_065041`  
**环境**: `alphatrade_cuda12`  
**状态**: 开发阶段评估，不代表 trading-ready  

M9/M10 backtest 仅作为产品化和评估 handoff，不是完整交易执行模拟器。

---

## 1. 目标

M10 的目标不是扩大模型，而是补齐评估层，让后续实验能够可靠回答：

1. 预测分位是否校准。
2. 当前 champion 是否打过简单 naive baselines。
3. 扣成本后是否存在稳定方向价值。
4. 哪些 horizon、symbol、quantile 明显异常。
5. 后续 champion selection 是否能超越单一 `pinball_loss.overall`。

---

## 2. 执行范围

本次不是 M0-M10 全量重训。执行范围是：

- 复跑 M9 productization outputs，因为 M10 依赖 full-universe M9 predictions。
- 新增 M10 realized-target evaluation。
- 新增 naive baseline comparison。
- 新增 M10 schema/profile。
- 增强 M9/M10 进度日志，便于定位长耗时阶段。
- 跑通 full-universe M10 评估和全量本地测试。

M0-M8 作为已有训练历史和 champion 来源，没有在本轮重跑。

---

## 3. M9 Productization 结果

M9 full-universe inference 使用 GPU 跑通。

| 项 | 值 |
|---|---:|
| model_version | `alphatrade_m10_evalcal_full_20260615_065041` |
| symbols | `24` |
| prediction rows | `4,442,327` |
| prediction columns | `23` |
| batch_size | `2048` |
| jax_backend | `gpu` |
| elapsed_seconds | `498.62` |
| samples_per_second | `8909.3` |

M9 predictions 仍保持宽表格式：

```text
symbol, eob, model_version,
h1/h5/h20/h60 x q10/q30/q50/q70/q90
```

M9 strict validation 结果：

- Schema: `6/6 passed`
- Semantic: `14/14 passed`
- 检查项包括 required columns、重复键、数值列、NaN/Inf、quantile non-crossing。

主要产物：

- `reports/m9_predictions.parquet`
- `reports/m9_infer_metrics.json`
- `reports/m9_infer_metrics.md`
- `reports/m9_infer_progress.jsonl`
- `reports/m9_backtest_metrics.json`
- `reports/m9_backtest_metrics.md`
- `reports/m9_schema_validation.json`
- `reports/m9_schema_validation.md`
- `reports/m9_model_bundle_manifest.json`

---

## 4. M10 评估结果

M10 将 M9 predictions 与 `data/processed/m1_f8/{symbol}/bars.parquet` 对齐，realized target 使用：

```text
realized_h = log(close[t+h] / close[t])
```

对齐结果：

| 项 | 值 |
|---|---:|
| aligned rows | `4,440,887` |
| dropped unavailable target rows | `1,440` |
| missing symbols | `0` |
| missing eob rows | `0` |

核心指标：

| metric | value |
|---|---:|
| model pinball overall | `0.003051934663220208` |
| model coverage MAE | `0.21442328300630034` |
| quantile crossing rate | `0.0` |
| max abs Pearson IC | `0.0027180243603707158` |
| max abs rank IC | `0.0024624567278215357` |
| IC materiality proxy | `false` |

By horizon:

| Horizon | Pinball | Coverage MAE | Pearson IC | Rank IC | Direction Hit |
|---|---:|---:|---:|---:|---:|
| h1 | `0.0029405374` | `0.2225656316` | `-0.0004113692` | `-0.0005097428` | `0.3892697562` |
| h5 | `0.0030748887` | `0.2076418562` | `-0.0027180244` | `-0.0017024508` | `0.4411238115` |
| h20 | `0.0032225114` | `0.2374563955` | `-0.0017259411` | `0.0013183885` | `0.4675903710` |
| h60 | `0.0029698012` | `0.1900292487` | `0.0023215685` | `0.0024624567` | `0.4787577347` |

Worst calibration:

| Scope | Horizon | Symbol | Quantile | Expected | Observed | Abs Error |
|---|---|---|---|---:|---:|---:|
| overall | h20 | n/a | q50 | `0.5` | `0.0151780489` | `0.4848219511` |
| symbol/horizon | h20 | `CFFEX.T` | q50 | `0.5` | `0.0003270512` | `0.4996729488` |

结论：分位非交叉检查通过，但绝对尺度和校准明显失败。IC/rank IC 接近 0，当前 median prediction 没有表现出稳定方向排序能力。

---

## 5. Baseline Comparison

M10 新增三个 naive baselines：

1. `zero_return_quantile`
2. `rolling_historical_quantile`
3. `rolling_median_direction`

Pinball 和 coverage 对比：

| Source | Pinball | Coverage MAE |
|---|---:|---:|
| model | `0.0030519347` | `0.2144233` |
| zero_return_quantile | `0.0007933594` | `0.2504408` |
| rolling_historical_quantile | `0.0006104896` | `0.0124525` |
| rolling_median_direction | `0.0007947781` | `0.2504203` |

模型没有打过 naive pinball baselines。尤其 `rolling_historical_quantile` 同时显著优于模型的 pinball 和 coverage calibration。

Baseline report 中的摘要：

| Baseline | Common Rows | Beats Pinball | Pinball Improvement | Better Coverage MAE | Coverage MAE Delta |
|---|---:|---|---:|---|---:|
| zero_return_quantile | `4,440,887` | no | `-0.0022585752` | yes | `-0.0360174690` |
| rolling_historical_quantile | `4,440,887` | no | `-0.0024414451` | no | `0.2019707594` |
| rolling_median_direction | `4,440,887` | no | `-0.0022571566` | yes | `-0.0359970227` |

说明：`Pinball Improvement` 定义为 baseline minus model，正数才表示模型更好。

---

## 6. Backtest Matrix

M10 backtest matrix 覆盖：

- horizons: `1, 5, 20, 60`
- signal variants:
  - `q50_sign`
  - `q30_q70_confirmed`
  - `uncertainty_filtered_q50`
- costs: `0, 1, 2, 5 bps`
- thresholds: `0, 0.5sigma, 1sigma`

Best net return row:

| Horizon | Variant | Cost bps | Threshold | Net | Drawdown | Turnover | Hit Rate | Trades |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 60 | q50_sign | 0 | 1sigma | `11.3748053412` | `-112.7680711270` | `1,540,334` | `0.4778408990` | `2,544,485` |

该结果不能解释为可交易收益。原因：

- M9/M10 backtest 是 lightweight handoff，不是完整执行模拟器。
- annualized return 未实现完整交易日历，字段为 null。
- 成本敏感性很强，多数组合在成本后表现不稳。
- 当前 IC 和 calibration 均未通过质量判断。

---

## 7. Schema 与测试

M10 strict validation：

- Profile: `m10`
- Schema: `6/6 passed`
- Semantic: `0/0 passed`

全量测试：

```bash
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python -m pytest src/alphatrade/tests -q
```

结果：

```text
139 passed in 340.66s
```

M9 strict validation：

```bash
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py \
  --profile m9 --strict \
  --output-root ../alphatrade_runs/m10_evalcal_full_20260615_065041
```

M10 strict validation：

```bash
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py \
  --profile m10 --strict \
  --output-root ../alphatrade_runs/m10_evalcal_full_20260615_065041
```

---

## 8. Code Changes

主要工程改动：

- M9 inference 改为按 symbol/batch 流式推理，避免一次性 materialize 全量窗口。
- M9 inference metrics 增加 `batch_size`、`jax_backend`、吞吐、进度日志等字段。
- M9 predictions semantic checks 补齐重复键、NaN/Inf、数值列、分位非交叉。
- 新增 `build_m10_prediction_eval.py`。
- 新增 M10 prediction eval、backtest matrix、baseline comparison schema。
- 新增 `m10` contracts manifest profile。
- 增强 M10 progress logging。
- 增强 resume guard，避免配置变化后复用 stale outputs。
- 增加 target alignment、rolling baseline no-leakage、coverage、crossing、schema、resume mismatch 等测试。

主要文件：

- `src/alphatrade/scripts/batch_infer_offline.py`
- `src/alphatrade/scripts/build_m10_prediction_eval.py`
- `src/alphatrade/scripts/validate_reports_schema.py`
- `src/alphatrade/scripts/run_m5_sweep.py`
- `src/alphatrade/prediction_schema.py`
- `src/alphatrade/schemas/contracts_manifest.yaml`
- `src/alphatrade/schemas/m9_infer_metrics.schema.json`
- `src/alphatrade/schemas/m10_prediction_eval_metrics.schema.json`
- `src/alphatrade/schemas/m10_backtest_matrix.schema.json`
- `src/alphatrade/schemas/m10_baseline_comparison.schema.json`
- `src/alphatrade/tests/test_m10_prediction_eval.py`
- `src/alphatrade/tests/test_batch_infer_offline.py`
- `src/alphatrade/tests/test_reports_validator.py`
- `src/alphatrade/tests/test_prediction_schema.py`

---

## 9. 结论

M10 成功把评估、baseline、schema 和 full-universe productization gate 补齐。工程链路上，M9 GPU inference 和 M10 full-universe evaluation 都已经跑通。

但模型质量结论是负面的：

1. 当前 champion 没有打过 zero 和 rolling historical quantile baselines。
2. 分位 coverage calibration 明显失败。
3. IC/rank IC 不显著。
4. post-cost backtest 不稳，不能作为交易结论。

因此当前 champion 应继续保持 rejected / development-stage 状态。下一步应进入 M11，优先审计并修复 target/output scale 与 quantile calibration，而不是扩大模型。

---

## 10. 后续 M11 起点

M11 应优先验证：

- target 是 raw log return、percent return、bps，还是 normalized value。
- 训练阶段是否使用 label scaling/normalization。
- M9 inference 输出是否缺少 inverse transform。
- M10 realized target 是否和训练 target 单位一致。
- h1 target std 约 `0.000742`，但 h1 prediction q10/q90 均值约 `-0.032/+0.034` 的尺度差异来自哪里。
- post-hoc calibration 是否能显著降低 coverage MAE。

M11 acceptance 不应只看 pinball，必须继续与 `rolling_historical_quantile` 对比。
