# M4-T0 总结报告：Contract 定义

**日期**: 2026-03-01
**状态**: ✅ 完成

---

## 任务目标

定义 M4 的验收标准和契约，明确：
1. Training 和 Evaluation 两类报告的 schema
2. run_id 贯穿原则
3. 字段名固定规则
4. 命令模板和验收清单

---

## 完成内容

### 1. M4 Contract 文档

**文件**: `docs/m4_contract.md`

**内容**:
- ✅ 报告类型定义（Training + Evaluation）
- ✅ run_id 贯穿原则
- ✅ Training Metrics Schema（复用 m2_train_metrics.schema.json）
- ✅ Evaluation Metrics Schema（新增 m4_eval_metrics.schema.json）
- ✅ 命令模板（train, eval, validate）
- ✅ 验收清单（T0/T1/T2/T3）
- ✅ 评估指标说明（pinball loss, coverage, crossing, IC, by_symbol）
- ✅ 验证流程和故障排查

### 2. M4 Evaluation Metrics Schema

**文件**: `src/alphatrade/schemas/m4_eval_metrics.schema.json`

**必须字段**:
```json
{
  "run_id": "string",
  "eval_timestamp": "string",
  "dataset": {...},
  "pinball_loss": {
    "overall": number,
    "by_horizon": {...}
  },
  "quantile_coverage": {
    "q10": number,
    "q30": number,
    "q50": number,
    "q70": number,
    "q90": number
  },
  "quantile_crossing": {
    "rate": number,
    "count": integer
  },
  "ic_metrics": {...},      // 可选但强烈建议
  "by_symbol": [...]
}
```

### 3. Schema 验证器更新

**文件**: `src/alphatrade/scripts/validate_reports_schema.py`

**更新内容**:
- ✅ 添加 m3_train_metrics 验证（使用 m2 schema）
- ✅ 添加 m4_train_metrics 验证（使用 m2 schema）
- ✅ 添加 m4_eval_metrics 验证（使用 m4 schema）
- ✅ 输出文件名更新为 m4_schema_validation.json/md

---

## 关键约定

### 1. Schema 复用策略

**Training Metrics**:
- M2/M3/M4 都使用 `m2_train_metrics.schema.json`
- 保持字段名固定，避免 schema 漂移
- 新增字段放在子对象下（如 `stability.grad_norm_pre_clip_max`）

**Evaluation Metrics**:
- M4 新增 `m4_eval_metrics.schema.json`
- 独立定义评估指标结构

### 2. run_id 贯穿原则

**流程**:
```
train_m4_alphatrade.py
  └─> 生成 run_id (UUID)
      └─> 写入 m4_train_metrics.json (run.run_id)
          └─> eval_m4.py 读取 run_id
              └─> 写入 m4_eval_metrics.json (run_id)
```

**验证**:
```bash
train_run_id=$(jq -r '.run.run_id' reports/m4_train_metrics.json)
eval_run_id=$(jq -r '.run_id' reports/m4_eval_metrics.json)
[ "$train_run_id" = "$eval_run_id" ] && echo "✅ run_id 一致"
```

### 3. 字段名固定

**Training 报告**:
- 顶层字段：`run`, `dataset`, `model`, `training`, `loss`, `stability`
- 不允许新增顶层字段
- 子字段可扩展

**Evaluation 报告**:
- 顶层字段：`run_id`, `eval_timestamp`, `dataset`, `pinball_loss`, `quantile_coverage`, `quantile_crossing`, `ic_metrics`, `by_symbol`
- 字段名固定

---

## 评估指标定义

### 1. Pinball Loss

**定义**: Quantile regression loss

**公式**:
```python
error = y_true - y_pred
loss = np.where(error >= 0, q * error, (q - 1) * error)
```

**输出**: Overall + by_horizon (h1, h5, h20, h60)

### 2. Quantile Coverage

**定义**: 实际覆盖率 vs 期望覆盖率

**公式**:
```python
coverage_q10 = (y_true < y_pred_q10).mean()  # 期望 0.1
```

**输出**: q10, q30, q50, q70, q90

### 3. Quantile Crossing

**定义**: 分位数越界（非单调）

**公式**:
```python
crossings = (y_pred[:, 1:] < y_pred[:, :-1]).sum()
rate = crossings / total_predictions
```

**输出**: rate, count

### 4. IC Metrics

**定义**: 信息系数（预测与实际的相关性）

**公式**:
```python
ic = np.corrcoef(y_true, y_pred_q50)[0, 1]
rank_ic = spearmanr(y_true, y_pred_q50).correlation
```

**输出**: ic, rank_ic, ic_by_horizon

### 5. By Symbol

**定义**: 分 symbol 的主要指标

**输出**: symbol, samples, pinball_loss, ic

---

## 验收标准

### M4-T0 验收清单

- [x] ✅ `docs/m4_contract.md` 创建完成
- [x] ✅ `src/alphatrade/schemas/m4_eval_metrics.schema.json` 创建完成
- [x] ✅ `validate_reports_schema.py` 支持 M4 报告验证
- [x] ✅ Contract 文档包含所有必要信息

---

## 输出文件清单

### 新增文件

```
docs/
└── m4_contract.md                              # M4 契约文档

src/alphatrade/schemas/
└── m4_eval_metrics.schema.json                 # M4 评估 schema

src/alphatrade/scripts/
└── validate_reports_schema.py                  # 已更新（支持 M4）

reports/
└── M4_T0_SUMMARY.md                            # 本文件
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
  --seed 42
```

### Evaluation

```bash
# 评估
python src/alphatrade/scripts/eval_m4.py \
  --train-metrics reports/m4_train_metrics.json \
  --dataset-config configs/dataset/m2.yaml \
  --split val
```

### Validation

```bash
# Schema 验证
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir reports \
  --schemas-dir src/alphatrade/schemas
```

---

## 下一步：M4-T1

**任务**: 实现训练脚本和 matrix runner

**产物**:
1. `src/alphatrade/scripts/train_m4_alphatrade.py`
2. `src/alphatrade/scripts/run_m4_matrix.py`
3. `reports/m4_train_metrics.json`
4. `reports/m4_train_run.md`
5. `reports/m4_matrix_summary.md`（如果运行 matrix）

**预计时间**: 1-2 小时

---

## 总结

✅ **M4-T0 完成**

**核心成果**:
- M4 Contract 文档完整
- Evaluation Schema 定义清晰
- Schema 验证器已更新
- run_id 贯穿原则明确

**关键约定**:
- Training 复用 M2 schema
- Evaluation 使用新 M4 schema
- 字段名固定，不允许随意修改
- run_id 必须贯穿 train → eval

**Contract 状态**: ✅ 已定义且完整

---

**完成时间**: 2026-03-01
**下一步**: M4-T1 - 实现训练脚本
