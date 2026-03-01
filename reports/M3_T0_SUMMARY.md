# M3-T0 总结：Contract Freeze

生成时间: 2026-03-01

## 任务目标

明确 M3 的验收标准和契约，定义"怎么验收"。

---

## ✅ 完成内容

### 1. M3 Training Contract 文档

**文件**: `docs/m3_training_contract.md`

**内容**:
- Schema 复用说明（使用 M2 的 schema）
- M3 Metrics 字段填充规则
- 输出文件定义
- 数据复用说明
- 训练稳定性配置
- 验收标准
- 故障排查

---

## 核心约定

### ✅ Schema 复用

**M3 使用 M2 的 Schema**: `m2_train_metrics.schema.json`

**原因**:
- 保持字段名固定
- 避免 schema 漂移
- 便于版本管理

**约束**:
- 顶层字段固定: run, dataset, model, training, loss, stability
- `additionalProperties: false` (不可新增顶层字段)

### ✅ M3 Metrics 字段填充规则

**model 字段**:
```json
{
  "type": "alphatrade_v0.2",
  "backend": "jax",
  "total_params": <从 params tree 统计>,
  "trainable_params": <同上>
}
```

**training 字段**:
```json
{
  "compile_jit": true/false
}
```

**stability 字段**:
```json
{
  "nan_steps": <NaN 出现次数>,
  "inf_steps": <Inf 出现次数>,
  "max_grad_norm": <全程最大梯度范数>,
  "oom_count": <OOM 次数>
}
```

### ✅ 输出文件

**Metrics JSON**:
- 文件名: `reports/m3_train_metrics.json`
- Schema: `src/alphatrade/schemas/m2_train_metrics.schema.json`
- 验证: `validate_reports_schema.py`

**Run Markdown**:
- 文件名: `reports/m3_train_run.md`
- 内容: 命令 + 配置 + 数据量 + loss + stability

### ✅ 数据复用

**Dataset**: `data/processed/m1_f8/{symbol}/`

**特征顺序** (固定):
1. ret_1m
2. hl_range
3. co_change
4. vol_log1p
5. pos_log1p
6. minute_sin
7. minute_cos
8. is_session_open

**约束**:
- Feature dim: 8
- Dtype: float32
- Lookback: 60
- Horizons: [1, 5, 20, 60]
- Quantiles: [0.1, 0.3, 0.5, 0.7, 0.9]

### ✅ 训练稳定性配置

**必须开启**:
1. Gradient Clipping: `optax.clip_by_global_norm(1.0)`
2. NaN/Inf 检测: 每个 step 检查
3. OOM 捕获: 捕获异常并记录

**推荐配置**:
- Learning rate: 1e-4
- Batch size: 128
- JIT 编译: 开启

---

## 验收标准

### ✅ 必须项

1. 一条命令稳定运行（200-1000 steps）
2. 产出符合 schema 的报告
3. 训练稳定（nan_steps = 0 或很少）
4. 字段名固定（不新增顶层字段）

### 可选项

- Loss 收敛
- 性能优化
- 多 GPU 训练

---

## 与 M2 的差异

| 项目 | M2 | M3 |
|------|----|----|
| 模型 | SimpleQuantileModel (PyTorch) | AlphaTrade v0.2 (JAX) |
| Backend | PyTorch | JAX |
| 参数量 | 165K | ~1-10M |
| JIT 编译 | 否 | 是 |
| Schema | m2_train_metrics_v1 | 复用 m2_train_metrics_v1 |
| 输出文件 | m2_train_metrics.json | m3_train_metrics.json |

---

## 下一步：M3-T1

### 任务

模型入口确认 + Adapter：
- 确认 AlphaTrade v0.2 的 import 路径
- 写 adapter（输入 [B, L, 8] → 输出 quantiles）
- 写 forward smoke test

### 验收

- 一条命令能跑通 forward
- 输出 shape 合理

---

## 总结

✅ **M3-T0 完成**

**核心成果**:
- M3 Training Contract 文档完成
- Schema 复用规则明确
- 字段填充规则定义
- 验收标准清晰

**关键约定**:
- 复用 M2 schema
- 字段名固定
- 训练稳定性配置
- 输出文件定义

**Contract 状态**: ✅ 已定义（Defined）
