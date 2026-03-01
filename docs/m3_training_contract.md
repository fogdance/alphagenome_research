# M3 Training Contract

生成时间: 2026-03-01

---

## 目标

M3 使用真实的 AlphaTrade v0.2 (JAX) 模型进行训练，替换 M2 的 SimpleQuantileModel placeholder。

**核心原则**:
- 复用 M2 的 dataset/dataloader/index 产物
- 复用 M2 的 reports schema（字段名固定）
- 训练稳定（无 NaN/Inf 扩散）
- 产出符合 schema 的报告

---

## Schema 复用

### M3 使用 M2 的 Schema

**Schema 文件**: `src/alphatrade/schemas/m2_train_metrics.schema.json`

**版本**: m2_train_metrics_v1

**说明**: M3 不创建新的 schema，直接复用 M2 的 schema，保证字段名固定。

### 顶层字段（固定，不可修改）

```json
{
  "run": {...},
  "dataset": {...},
  "model": {...},
  "training": {...},
  "loss": {...},
  "stability": {...}
}
```

**约束**: `additionalProperties: false` (顶层不可新增字段)

---

## M3 Metrics 字段填充规则

### 1. model 字段

**必须字段**:
```json
{
  "model": {
    "type": "alphatrade_v0.2",           // 固定值
    "backend": "jax",                     // 固定值
    "total_params": 1234567,              // 从 params tree 统计（int）
    "trainable_params": 1234567,          // 同上
    "config": {
      // 模型配置（可选，不限制）
      "num_stages": 6,
      "channel_increment": 64,
      "num_quantiles": 5,
      ...
    }
  }
}
```

**填充规则**:
- `type`: 必须是 `"alphatrade_v0.2"`
- `backend`: 必须是 `"jax"`
- `total_params`: 从 JAX params tree 统计，使用 `jax.tree_util.tree_map` 计算
- `trainable_params`: 同 `total_params`（AlphaTrade v0.2 所有参数都可训练）
- `config`: 可选，包含模型配置信息

### 2. training 字段

**必须字段**:
```json
{
  "training": {
    "max_steps": 1000,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "optimizer": "adamw",
    "lr_schedule": {...},
    "compile_jit": true                   // 新增：是否使用 JIT 编译
  }
}
```

**填充规则**:
- `compile_jit`: boolean，表示是否使用 `jax.jit` 编译训练步

### 3. stability 字段

**必须字段**:
```json
{
  "stability": {
    "nan_steps": 0,                       // NaN/Inf 出现的 step 次数
    "inf_steps": 0,                       // 同上（或合并到 nan_steps）
    "max_grad_norm": 1.234,               // 全程最大梯度范数
    "oom_count": 0                        // OOM 次数（可选）
  }
}
```

**填充规则**:
- `nan_steps`: 统计训练过程中出现 NaN 的 step 次数
- `inf_steps`: 统计训练过程中出现 Inf 的 step 次数
- `max_grad_norm`: 记录全程最大的梯度范数（clip 前）
- `oom_count`: 可选，记录 OOM 异常次数

### 4. 其他字段

**run, dataset, loss**: 与 M2 相同，无变化

---

## M3 输出文件

### 1. Metrics JSON

**文件名**: `reports/m3_train_metrics.json`

**Schema**: `src/alphatrade/schemas/m2_train_metrics.schema.json`

**验证命令**:
```bash
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir reports \
  --schemas-dir src/alphatrade/schemas
```

**注意**: 文件名是 `m3_train_metrics.json`，但使用 `m2_train_metrics.schema.json` 验证。

### 2. Run Markdown

**文件名**: `reports/m3_train_run.md`

**内容**:
- 运行命令（可复制）
- 配置摘要（model, dataset, training）
- 数据量统计（symbols, samples）
- Loss 表格（train/val, by_horizon）
- Stability 指标（nan_steps, max_grad_norm, oom_count）

---

## 数据复用

### Dataset/Dataloader

**数据路径**: `data/processed/m1_f8/{symbol}/`

**文件**:
- `bars.parquet`: 8D features
- `index_train.parquet`: 训练索引
- `index_val.parquet`: 验证索引
- `index_test.parquet`: 测试索引

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

---

## 训练稳定性配置

### 必须开启

1. **Gradient Clipping**
   - 方法: `optax.clip_by_global_norm(1.0)`
   - 默认值: 1.0
   - 记录: `stability.max_grad_norm`

2. **NaN/Inf 检测**
   - 每个 step 检查 loss 和 gradients
   - 记录: `stability.nan_steps`, `stability.inf_steps`
   - 出现 NaN/Inf 时：记录并跳过该 step（或终止训练）

3. **OOM 捕获**
   - 捕获 `jax.errors.OutOfMemoryError`
   - 记录: `stability.oom_count`

### 推荐配置

- Learning rate: 1e-4 (保守)
- Batch size: 128 (根据显存调整)
- JIT 编译: 开启（提速）
- Mixed precision: 可选（节省显存）

---

## 验收标准

### 必须项

1. ✅ **一条命令稳定运行**
   - 至少 200-1000 steps
   - 无 crash

2. ✅ **产出符合 schema 的报告**
   - `m3_train_metrics.json` 通过 schema 验证
   - `m3_train_run.md` 包含所有必要信息

3. ✅ **训练稳定**
   - `stability.nan_steps` = 0（或很少）
   - Loss 不发散

4. ✅ **字段名固定**
   - 顶层字段: run, dataset, model, training, loss, stability
   - 不新增顶层字段

### 可选项

- Loss 收敛（不强制要求，因为可能需要更多 steps）
- 性能优化（JIT 编译、混合精度）
- 多 GPU 训练

---

## 与 M2 的差异

| 项目 | M2 | M3 |
|------|----|----|
| 模型 | SimpleQuantileModel (PyTorch) | AlphaTrade v0.2 (JAX) |
| Backend | PyTorch | JAX |
| 参数量 | 165K | ~1-10M (取决于配置) |
| JIT 编译 | 否 | 是 |
| Schema | m2_train_metrics_v1 | 复用 m2_train_metrics_v1 |
| 输出文件 | m2_train_metrics.json | m3_train_metrics.json |

---

## 故障排查

### NaN/Inf 问题

**症状**: `stability.nan_steps > 0`

**可能原因**:
1. 学习率过高
2. 梯度爆炸
3. 数据中有 NaN

**解决**:
1. 降低学习率（1e-4 → 1e-5）
2. 增加梯度裁剪（1.0 → 0.5）
3. 检查数据（使用 `np.nan_to_num`）

### OOM 问题

**症状**: `stability.oom_count > 0`

**可能原因**:
1. Batch size 过大
2. 模型过大
3. 显存不足

**解决**:
1. 减小 batch size（128 → 64）
2. 使用混合精度训练
3. 使用梯度累积

### JIT 编译慢

**症状**: 第一个 step 很慢

**原因**: JAX JIT 编译需要时间

**解决**:
1. 正常现象，后续 step 会快
2. 可以预热（warmup）几个 step

---

## 参考

### 相关文档

- `docs/m2_reports_contract.md` - M2 reports schema 契约
- `docs/m2_training_data_spec.md` - M2 训练数据规范
- `reports/M2_FINAL_ACCEPTANCE.md` - M2 最终验收报告

### Schema 文件

- `src/alphatrade/schemas/m2_train_metrics.schema.json`
- `src/alphatrade/schemas/m2_universe_sweep.schema.json`
- `src/alphatrade/schemas/m2_t1_dataloader_check.schema.json`

### 验证脚本

- `src/alphatrade/scripts/validate_reports_schema.py`

---

## 变更历史

| 日期 | 版本 | 变更 | 作者 |
|------|------|------|------|
| 2026-03-01 | v1 | 初始版本，定义 M3 training contract | - |

---

**Contract 状态**: ✅ **已定义（Defined）**

**下一步**: T1 - 模型入口确认 + Adapter
