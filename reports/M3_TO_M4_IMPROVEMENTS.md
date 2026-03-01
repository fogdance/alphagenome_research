# M3 遗留问题与 M4 改进清单

**日期**: 2026-03-01
**状态**: M3 已完成，M4 待开始

---

## 🔧 M3 遗留的小瑕疵（已修复）

### ✅ 1. Run ID 不一致（已修复）

**问题**:
- `m3_train_metrics.json` 中 Run ID 是 `60f040b9`
- `M3_FINAL_ACCEPTANCE.md` 表格中写的是 `fa719c4a`

**原因**: 多次运行导致的不一致

**修复**: 已统一为 `60f040b9`（以 metrics.json 为准）

**影响**: 无，已修复

---

### ✅ 2. 梯度范数语义不清（已说明）

**问题**:
- `max_grad_norm=6.45` 与 `clip_norm=1.0` 的关系不清楚
- 容易误解为裁剪后的值

**说明**:
- `max_grad_norm` 是**裁剪前**的最大梯度范数
- `clip_norm=1.0` 是裁剪阈值
- 裁剪后的梯度范数 ≤ 1.0

**修复**: 已在文档中明确标注 "pre-clip"

**影响**: 文档已清晰，但代码层面需要 M4 改进

---

## 🚀 M4 优先改进项

### 1. 梯度范数指标拆分（优先级：高）

**当前实现**:
```json
"stability": {
  "max_grad_norm": 6.45  // 裁剪前的值
}
```

**问题**:
- 只记录裁剪前的值
- 无法看到裁剪后的实际梯度范数
- 定位训练炸点时信息不足

**建议实现**:
```json
"stability": {
  "grad_norm_pre_clip_max": 6.45,   // 裁剪前的最大值
  "grad_norm_post_clip_max": 1.0,   // 裁剪后的最大值
  "grad_norm_pre_clip_mean": 3.2,   // 裁剪前的平均值（可选）
  "clip_threshold": 1.0,             // 裁剪阈值
  "clipped_steps": 45                // 触发裁剪的步数（可选）
}
```

**优点**:
- ✅ 清晰区分裁剪前后
- ✅ 便于定位训练炸点
- ✅ 更好的可解释性
- ✅ 可以看到裁剪是否生效

**实现位置**: `src/alphatrade/scripts/train_m3_alphatrade.py`

**代码修改**:
```python
# 当前实现
grad_norm = optax.global_norm(grads)
updates, new_opt_state = optimizer.update(grads, opt_state, params)

# 建议实现
grad_norm_pre_clip = optax.global_norm(grads)
updates, new_opt_state = optimizer.update(grads, opt_state, params)
grad_norm_post_clip = optax.global_norm(updates)  # 或从 optimizer 获取

# 记录
max_grad_norm_pre_clip = max(max_grad_norm_pre_clip, grad_norm_pre_clip)
max_grad_norm_post_clip = max(max_grad_norm_post_clip, grad_norm_post_clip)
if grad_norm_pre_clip > clip_threshold:
    clipped_steps += 1
```

**预计工作量**: 30 分钟

---

### 2. Schema 扩展（优先级：中）

**当前 Schema**: `m2_train_metrics_v1`

**问题**:
- M3 新增字段（`compile_jit`, `oom_count`）未在 schema 中定义
- 虽然验证通过（因为 schema 允许额外字段），但不够规范

**建议**: 创建 `m3_train_metrics_v1` schema

**新增字段**:
```json
{
  "training": {
    "compile_jit": {"type": "boolean"}  // 新增
  },
  "stability": {
    "oom_count": {"type": "integer"},   // 新增
    "grad_norm_pre_clip_max": {"type": "number"},   // 新增
    "grad_norm_post_clip_max": {"type": "number"},  // 新增
    "clip_threshold": {"type": "number"},           // 新增
    "clipped_steps": {"type": "integer"}            // 新增（可选）
  }
}
```

**权衡**:
- ✅ 优点: 更规范，字段定义清晰
- ❌ 缺点: 需要维护新 schema，增加复杂度

**建议**: M4 评估是否值得创建新 schema

**预计工作量**: 1 小时

---

### 3. 训练日志增强（优先级：中）

**当前实现**: 只在验证时打印 loss

**建议**: 增加更详细的日志

**新增日志**:
```python
# 每 N 步打印详细信息
if step % log_every == 0:
    print(f"Step {step}/{max_steps}")
    print(f"  Loss: {loss:.6f}")
    print(f"  Grad norm (pre-clip): {grad_norm_pre:.4f}")
    print(f"  Grad norm (post-clip): {grad_norm_post:.4f}")
    print(f"  Learning rate: {lr:.6e}")
    print(f"  By-horizon loss:")
    for h in [1, 5, 20, 60]:
        print(f"    h{h}: {metrics[f'h{h}_total']:.6f}")
```

**优点**:
- ✅ 更好的训练监控
- ✅ 便于调试
- ✅ 可以看到训练动态

**预计工作量**: 30 分钟

---

### 4. Checkpoint 保存（优先级：中）

**当前实现**: 不保存 checkpoint

**问题**:
- 训练中断后无法恢复
- 无法加载最佳模型

**建议**: 添加 checkpoint 保存

**实现**:
```python
# 保存最佳模型
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_step = step

    # 保存 checkpoint
    checkpoint = {
        'params': params,
        'state': state,
        'opt_state': opt_state,
        'step': step,
        'best_val_loss': best_val_loss
    }
    save_checkpoint(checkpoint, f'checkpoints/m3_best.pkl')
```

**优点**:
- ✅ 可以恢复训练
- ✅ 可以加载最佳模型
- ✅ 便于模型评估

**预计工作量**: 1 小时

---

### 5. 学习率调度（优先级：低）

**当前实现**: 固定学习率 1e-4

**建议**: 添加学习率调度

**实现**:
```python
# Cosine annealing with warmup
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=1e-4,
    warmup_steps=100,
    decay_steps=max_steps,
    end_value=1e-5
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule)
)
```

**优点**:
- ✅ 更好的收敛
- ✅ 避免过拟合

**预计工作量**: 30 分钟

---

## 📊 M4 改进优先级总结

| 改进项 | 优先级 | 工作量 | 影响 |
|--------|--------|--------|------|
| 梯度范数指标拆分 | 🔴 高 | 30 分钟 | 提升可观测性 |
| 训练日志增强 | 🟡 中 | 30 分钟 | 提升调试效率 |
| Checkpoint 保存 | 🟡 中 | 1 小时 | 提升鲁棒性 |
| Schema 扩展 | 🟡 中 | 1 小时 | 提升规范性 |
| 学习率调度 | 🟢 低 | 30 分钟 | 提升性能 |

**总工作量**: 3.5 小时

---

## 🎯 M4 建议任务列表

### Phase 1: 快速改进（1 小时）

1. ✅ 梯度范数指标拆分（30 分钟）
2. ✅ 训练日志增强（30 分钟）

### Phase 2: 功能增强（2 小时）

3. ✅ Checkpoint 保存（1 小时）
4. ✅ Schema 扩展（1 小时）

### Phase 3: 性能优化（0.5 小时）

5. ✅ 学习率调度（30 分钟）

### Phase 4: 完整训练（2-3 小时）

6. ✅ 500-1000 steps 训练
7. ✅ 全品种训练（24 个品种）
8. ✅ 评估指标扩展

---

## 📝 实施建议

### 立即执行（Phase 1）

**梯度范数指标拆分** 是最重要的改进，建议立即实施：

1. 修改 `train_m3_alphatrade.py`
2. 记录 `grad_norm_pre_clip` 和 `grad_norm_post_clip`
3. 更新 metrics 输出
4. 运行 smoke test 验证

### 后续执行（Phase 2-4）

根据 M4 的具体目标和时间安排，逐步实施其他改进。

---

## ✅ 验收标准

M4 改进完成后，应满足：

- [x] 梯度范数指标清晰（pre-clip 和 post-clip）
- [x] 训练日志详细（每 N 步打印）
- [x] 支持 checkpoint 保存和加载
- [x] Schema 规范（如果创建新 schema）
- [x] 学习率调度生效（如果实施）

---

**文档创建日期**: 2026-03-01
**维护者**: Claude (Opus 4.6)
