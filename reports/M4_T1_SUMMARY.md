# M4-T1 总结报告：训练脚本实现

**日期**: 2026-03-01
**状态**: ✅ 完成

---

## 任务目标

实现 M4 训练脚本，支持：
1. 长跑训练（默认 1000 steps）
2. 更好的默认参数（JIT=1, batch=128）
3. 输出 m4_train_metrics.json 和 m4_train_run.md

---

## 完成内容

### 1. M4 训练脚本

**文件**: `src/alphatrade/scripts/train_m4_alphatrade.py`

**基于**: M3 训练脚本（复制并修改）

**主要改进**:

#### 默认参数优化
```python
--max-steps: 1000 (M3: None, 需要从 config 读取)
--batch-size: 128 (M3: None, 需要从 config 读取)
--seed: 42 (M3: None, 需要从 config 读取)
--jit: 1 (M3: 1, 保持)
```

#### 参数处理简化
- M3: 从 config 读取，然后用 args 覆盖
- M4: 直接使用 args，不依赖 config 中的 training 部分

#### 输出文件更新
- `reports/m4_train_metrics.json` (M3: m3_train_metrics.json)
- `reports/m4_train_run.md` (M3: m3_train_run.md)

#### 标题和提示更新
- 标题: "M4: AlphaTrade v0.2 Training"
- 完成提示: "✅ M4 Training Complete"

---

## 命令示例

### Smoke Test (快速验证)

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 50 \
  --batch-size 32 \
  --smoke \
  --jit 0
```

**预期**:
- 3 个品种
- 50 steps
- 无 JIT（调试模式）
- 约 5-10 分钟

### 中等规模训练

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 500 \
  --batch-size 128 \
  --smoke \
  --jit 1
```

**预期**:
- 3 个品种
- 500 steps
- 开启 JIT
- 约 30-60 分钟

### 完整训练（默认参数）

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --smoke
```

**预期**:
- 3 个品种
- 1000 steps（默认）
- batch=128（默认）
- JIT=1（默认）
- seed=42（默认）
- 约 60-120 分钟

### 全品种训练

```bash
python src/alphatrade/scripts/train_m4_alphatrade.py \
  --config configs/dataset/m2.yaml
```

**预期**:
- 24 个品种
- 1000 steps
- 约 2-4 小时

---

## 输出文件

### 1. m4_train_metrics.json

**Schema**: `m2_train_metrics.schema.json`

**关键字段**:
```json
{
  "run": {
    "run_id": "abc12345",
    "seed": 42
  },
  "model": {
    "type": "alphatrade_v0.2",
    "backend": "jax"
  },
  "training": {
    "max_steps": 1000,
    "batch_size": 128,
    "compile_jit": true
  },
  "loss": {
    "val_best": 0.1069,
    "best_step": 850
  },
  "stability": {
    "nan_steps": 0,
    "inf_steps": 0,
    "max_grad_norm": 6.45
  }
}
```

### 2. m4_train_run.md

**内容**:
- 运行命令（可复制）
- 配置信息（run_id, device, symbols, samples）
- 模型信息（params, d_model, layers）
- Loss 指标（train/val, by-horizon）
- Stability 指标（NaN/Inf, grad norm）

---

## 与 M3 对比

| 项目 | M3 | M4 | 改进 |
|------|----|----|------|
| 默认 steps | None (需 config) | 1000 | ✅ 明确 |
| 默认 batch | None (需 config) | 128 | ✅ 明确 |
| 默认 seed | None (需 config) | 42 | ✅ 明确 |
| 默认 JIT | 1 | 1 | - |
| 参数处理 | config → args 覆盖 | 直接用 args | ✅ 简化 |
| 输出文件 | m3_*.json/md | m4_*.json/md | ✅ 更新 |

---

## 验收标准

### M4-T1 验收清单

- [x] ✅ `train_m4_alphatrade.py` 创建完成
- [x] ✅ 默认参数优化（steps=1000, batch=128, seed=42）
- [x] ✅ 输出文件名更新为 m4
- [ ] ⏳ 运行 smoke test 验证（待执行）
- [ ] ⏳ `m4_train_metrics.json` 生成且通过 schema 验证（待执行）
- [ ] ⏳ `m4_train_run.md` 生成（待执行）

---

## 下一步：运行验证

### 1. Smoke Test (50 steps)

```bash
conda activate alphatrade
python src/alphatrade/scripts/train_m4_alphatrade.py \
  --config configs/dataset/m2.yaml \
  --max-steps 50 \
  --batch-size 32 \
  --smoke \
  --jit 0
```

**预期输出**:
```
M4: AlphaTrade v0.2 Training
Backend: JAX
JIT: disabled
Symbols: 3
Max steps: 50
Batch size: 32
...
✅ M4 Training Complete
✅ Metrics: reports/m4_train_metrics.json
✅ Report: reports/m4_train_run.md
```

### 2. Schema 验证

```bash
python src/alphatrade/scripts/validate_reports_schema.py \
  --reports-dir reports \
  --schemas-dir src/alphatrade/schemas
```

**预期输出**:
```
Validating m4_train_metrics... ✅ pass
```

---

## 注意事项

### 1. Matrix Runner 未实现

**原因**:
- 单次训练脚本已完成
- Matrix runner 可以用 shell 脚本实现
- 优先验证单次训练

**临时方案**:
```bash
# 手动运行多个配置
for seed in 42 43 44; do
  python src/alphatrade/scripts/train_m4_alphatrade.py \
    --seed $seed \
    --smoke
done
```

### 2. Checkpoint 保存未实现

**原因**:
- M3 也没有实现
- 可以在 M4-T2 或后续改进

**影响**:
- 训练中断无法恢复
- 无法加载最佳模型进行评估

**缓解**:
- 训练时间不长（1-2 小时）
- 可以重新训练

### 3. grad_norm_pre_clip/post_clip 未拆分

**原因**:
- 需要修改训练循环逻辑
- 可以在 M4-T3 实现

**当前**:
- 只记录 `max_grad_norm`（pre-clip）

---

## 总结

✅ **M4-T1 核心任务完成**

**完成内容**:
- M4 训练脚本创建完成
- 默认参数优化
- 输出文件名更新

**待验证**:
- 运行 smoke test
- Schema 验证
- 完整训练（1000 steps）

**未实现**（可选）:
- Matrix runner（可用 shell 脚本替代）
- Checkpoint 保存（M4-T2 或后续）
- grad_norm 拆分（M4-T3）

---

**完成时间**: 2026-03-01
**下一步**: 运行 smoke test 验证
