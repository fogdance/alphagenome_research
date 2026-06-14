# M4 Contract: Scale + Evaluate

**版本**: m4_v1
**日期**: 2026-03-01
**状态**: Active

---

## 目标

M4 任务目标：
1. **Scale**: 长跑训练（1000 steps）+ 多配置 matrix runner
2. **Evaluate**: 完整评估指标（pinball loss, quantile coverage, IC, 分 symbol 分析）

---

## 报告类型定义

M4 产出两类报告：

### 1. Training Reports（训练报告）

**文件名**: `$ALPHATRADE_RUNS_ROOT/reports/m4_train_metrics.json`

**Schema**: 复用 `m2_train_metrics.schema.json`

**字段要求**:
- 顶层字段固定：`run`, `dataset`, `model`, `training`, `loss`, `stability`
- `model.type` = `"alphatrade_v0.2"`
- `model.backend` = `"jax"`
- `run.run_id` 必须唯一且贯穿整个 M4 流程

### 2. Evaluation Reports（评估报告）

**文件名**: `$ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics.json`

**Schema**: 新增 `m4_eval_metrics.schema.json`

**字段要求**:
- 必须包含 `run_id`（关联到训练报告）
- 必须包含完整评估指标（见下文）

---

## run_id 贯穿原则

**关键约定**: `run_id` 必须在 train → eval 全流程保持一致

**流程**:
```
train_m4_alphatrade.py
  └─> 生成 run_id (UUID)
      └─> 写入 m4_train_metrics.json
          └─> eval_m4.py 读取 run_id
              └─> 写入 m4_eval_metrics.json
```

**验证**:
```bash
# 训练 run_id
train_run_id=$(jq -r '.run.run_id' $ALPHATRADE_RUNS_ROOT/reports/m4_train_metrics.json)

# 评估 run_id
eval_run_id=$(jq -r '.run_id' $ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics.json)

# 必须相等
[ "$train_run_id" = "$eval_run_id" ] && echo "✅ run_id 一致"
```

---

## M4 Training Metrics Schema

**复用**: `m2_train_metrics.schema.json`

**新增可选字段** (在 `stability` 下):
```json
{
  "stability": {
    "nan_steps": 0,
    "inf_steps": 0,
    "max_grad_norm": 6.45,           // 保留（向后兼容）
    "grad_norm_pre_clip_max": 6.45,  // 新增：裁剪前最大值
    "grad_norm_post_clip_max": 1.0,  // 新增：裁剪后最大值
    "oom_count": 0
  }
}
```

**说明**:
- `max_grad_norm` 保留以兼容 M2/M3
- 新增 `grad_norm_pre_clip_max` 和 `grad_norm_post_clip_max` 以明确语义

---

## M4 Evaluation Metrics Schema

**文件**: `src/alphatrade/schemas/m4_eval_metrics.schema.json`

**必须字段**:

```json
{
  "run_id": "string",              // 关联到训练 run_id
  "eval_timestamp": "string",      // ISO 8601 格式
  "dataset": {
    "split": "val|test",           // 评估数据集
    "symbols": 3,
    "samples": 11313
  },
  "pinball_loss": {
    "overall": 0.1069,
    "by_horizon": {
      "h1": 0.0253,
      "h5": 0.0251,
      "h20": 0.0263,
      "h60": 0.0302
    }
  },
  "quantile_coverage": {
    "q10": 0.12,                   // 实际覆盖率（期望 0.1）
    "q30": 0.31,
    "q50": 0.49,
    "q70": 0.69,
    "q90": 0.88
  },
  "quantile_crossing": {
    "rate": 0.02,                  // 越界比例
    "count": 226                   // 越界次数
  },
  "ic_metrics": {                  // 可选但强烈建议
    "ic": 0.045,                   // 信息系数（用 q50）
    "rank_ic": 0.052,              // 排序 IC
    "ic_by_horizon": {
      "h1": 0.038,
      "h5": 0.042,
      "h20": 0.048,
      "h60": 0.051
    }
  },
  "by_symbol": [                   // 分 symbol 统计
    {
      "symbol": "DCE.JM",
      "samples": 726,
      "pinball_loss": 0.0245,
      "ic": 0.042
    },
    ...
  ]
}
```

---

## 命令模板

### Training

```bash
# 单次训练
python src/alphatrade/scripts/train_m4_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 1000 \
  --batch-size 128 \
  --jit 1 \
  --clip-norm 1.0 \
  --seed 42

# Matrix runner（多配置）
python src/alphatrade/scripts/run_m4_matrix.py \
  --config configs/dataset/m2.yaml \
  --seeds 42,43,44 \
  --batch-sizes 64,128 \
  --max-steps 1000
```

### Evaluation

```bash
# 评估指定 run_id
python src/alphatrade/scripts/eval_m4.py \
  --train-metrics $ALPHATRADE_RUNS_ROOT/reports/m4_train_metrics.json \
  --dataset-config configs/dataset/m2.yaml \
  --split val

# 或直接指定 run_id
python src/alphatrade/scripts/eval_m4.py \
  --run-id 60f040b9 \
  --checkpoint checkpoints/m4_60f040b9_best.pkl \
  --dataset-config configs/dataset/m2.yaml \
  --split val
```

### Validation

```bash
# 验证所有 M4 报告
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir "$ALPHATRADE_RUNS_ROOT/reports" \
  --schemas-dir src/alphatrade/schemas
```

---

## 验收清单

### M4-T0 (Contract)

- [ ] `docs/m4_contract.md` 创建完成
- [ ] `src/alphatrade/schemas/m4_eval_metrics.schema.json` 创建完成
- [ ] `validate_reports_schema.py` 支持 M4 报告验证

### M4-T1 (Training)

- [ ] `train_m4_alphatrade.py` 创建完成
- [ ] `run_m4_matrix.py` 创建完成
- [ ] `m4_train_metrics.json` 生成且通过 schema 验证
- [ ] `m4_train_run.md` 生成
- [ ] `m4_matrix_summary.md` 生成（如果运行 matrix）

### M4-T2 (Evaluation)

- [ ] `eval_m4.py` 创建完成
- [ ] `m4_eval_metrics.json` 生成且通过 schema 验证
- [ ] `m4_eval_run.md` 生成
- [ ] 所有评估指标计算正确

### M4-T3 (Fixes)

- [ ] M3 run_id 不一致问题已修复
- [ ] stability 指标拆分为 pre/post clip

---

## 字段名固定原则

**训练报告** (m4_train_metrics.json):
- 顶层字段：`run`, `dataset`, `model`, `training`, `loss`, `stability`
- 不允许新增顶层字段
- 子字段可扩展（如 `stability.grad_norm_pre_clip_max`）

**评估报告** (m4_eval_metrics.json):
- 顶层字段：`run_id`, `eval_timestamp`, `dataset`, `pinball_loss`, `quantile_coverage`, `quantile_crossing`, `ic_metrics`, `by_symbol`
- 字段名固定，不允许随意修改

---

## 输出文件清单

### 必须产出

1. **训练报告**:
   - `$ALPHATRADE_RUNS_ROOT/reports/m4_train_metrics.json`
   - `$ALPHATRADE_RUNS_ROOT/reports/m4_train_run.md`

2. **评估报告**:
   - `$ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics.json`
   - `$ALPHATRADE_RUNS_ROOT/reports/m4_eval_run.md`

3. **验证报告**:
   - `$ALPHATRADE_RUNS_ROOT/reports/m4_schema_validation.json`
   - `$ALPHATRADE_RUNS_ROOT/reports/m4_schema_validation.md`

### 可选产出

4. **Matrix 汇总** (如果运行 matrix):
   - `$ALPHATRADE_RUNS_ROOT/reports/m4_matrix_summary.md`

5. **任务总结**:
   - `$ALPHATRADE_RUNS_ROOT/reports/M4_T0_SUMMARY.md`
   - `$ALPHATRADE_RUNS_ROOT/reports/M4_T1_SUMMARY.md`
   - `$ALPHATRADE_RUNS_ROOT/reports/M4_T2_SUMMARY.md`
   - `$ALPHATRADE_RUNS_ROOT/reports/M4_T3_SUMMARY.md`

---

## 评估指标说明

### 1. Pinball Loss

**定义**: Quantile regression loss

**计算**:
```python
error = y_true - y_pred
loss = np.where(error >= 0, q * error, (q - 1) * error)
```

**输出**:
- Overall: 所有 horizon 平均
- By horizon: h1, h5, h20, h60

### 2. Quantile Coverage

**定义**: 实际覆盖率 vs 期望覆盖率

**计算**:
```python
coverage_q10 = (y_true < y_pred_q10).mean()  # 期望 0.1
coverage_q90 = (y_true < y_pred_q90).mean()  # 期望 0.9
```

**输出**: q10, q30, q50, q70, q90 的实际覆盖率

### 3. Quantile Crossing

**定义**: 分位数越界（非单调）

**计算**:
```python
crossings = (y_pred[:, 1:] < y_pred[:, :-1]).sum()
rate = crossings / total_predictions
```

**输出**:
- Rate: 越界比例
- Count: 越界次数

### 4. IC Metrics

**定义**: 信息系数（预测与实际的相关性）

**计算**:
```python
# 使用 q50 作为预测值
ic = np.corrcoef(y_true, y_pred_q50)[0, 1]

# Rank IC
rank_ic = spearmanr(y_true, y_pred_q50).correlation
```

**输出**:
- IC: Pearson 相关系数
- Rank IC: Spearman 相关系数
- By horizon: 分 horizon 的 IC

### 5. By Symbol

**定义**: 分 symbol 的主要指标

**输出**: 每个 symbol 的样本数、pinball loss、IC

---

## 验证流程

### 1. Schema 验证

```bash
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir "$ALPHATRADE_RUNS_ROOT/reports" \
  --schemas-dir src/alphatrade/schemas
```

**预期输出**:
```
Validating m4_train_metrics... ✅ pass
Validating m4_eval_metrics... ✅ pass
```

### 2. run_id 一致性验证

```bash
# 提取 run_id
train_run_id=$(jq -r '.run.run_id' $ALPHATRADE_RUNS_ROOT/reports/m4_train_metrics.json)
eval_run_id=$(jq -r '.run_id' $ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics.json)

# 验证
if [ "$train_run_id" = "$eval_run_id" ]; then
  echo "✅ run_id 一致: $train_run_id"
else
  echo "❌ run_id 不一致: train=$train_run_id, eval=$eval_run_id"
  exit 1
fi
```

### 3. 字段完整性验证

```bash
# 检查必须字段
jq -e '.pinball_loss.overall' $ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics.json
jq -e '.quantile_coverage.q50' $ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics.json
jq -e '.by_symbol | length > 0' $ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics.json
```

---

## 故障排查

### 问题 1: Schema 验证失败

**症状**: `ValidationError: 'xxx' is a required property`

**解决**: 检查 JSON 文件是否包含所有必须字段

### 问题 2: run_id 不一致

**症状**: train 和 eval 的 run_id 不匹配

**解决**: 确保 eval_m4.py 从 train_metrics.json 读取 run_id

### 问题 3: 评估指标异常

**症状**: IC > 1 或 coverage 超出 [0, 1]

**解决**: 检查计算逻辑和数据预处理

---

**Contract 状态**: ✅ 已定义

**下一步**: M4-T1 - 实现训练脚本和 matrix runner
