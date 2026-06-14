# M4 评估脚本对比：eval_m4.py vs eval_m4_fast.py

**日期**: 2026-03-01

---

## 概述

| 版本 | 文件 | 推理方式 | 预计时间 | GPU 利用率 | 提速 |
|------|------|----------|----------|------------|------|
| 原版 | eval_m4.py | 逐样本 (batch=1) | 30-40 分钟 | 5% | 1x |
| 优化版 | eval_m4_fast.py | 批处理 (batch=128) + JIT | 1-2 分钟 | 60-80% | 15-25x |

---

## 核心差异

### 1. 推理方式

**原版 (eval_m4.py)**:
```python
# 逐样本循环
for sample in dataset:  # 11,313 次循环
    X_batch = jnp.expand_dims(jnp.array(sample['X']), 0)  # [1, 60, 8]
    rng = jax.random.PRNGKey(0)
    output, _ = model_apply_fn(params, state, rng, X_batch)
    # 处理单个样本的输出
```

**优化版 (eval_m4_fast.py)**:
```python
# 预加载所有数据
all_X, all_targets, all_symbols = dataset.get_all_data()
X_array = jnp.array(all_X)  # [11313, 60, 8]

# JIT 编译推理函数
@jax.jit
def inference_fn(params, state, X_batch):
    rng = jax.random.PRNGKey(0)
    return model_apply_fn(params, state, rng, X_batch)

# 批处理推理
for i in range(0, N, batch_size):  # 89 次循环 (batch_size=128)
    X_batch = X_array[i:i+batch_size]  # [128, 60, 8]
    output, _ = inference_fn(params, state, X_batch)
    # 处理批量输出
```

### 2. 性能优化点

| 优化点 | 原版 | 优化版 | 提速 |
|--------|------|--------|------|
| 批处理 | ❌ batch=1 | ✅ batch=128 | 10-15x |
| JIT 编译 | ❌ 无 | ✅ @jax.jit | 1.5-2x |
| 数据预加载 | ❌ 逐个加载 | ✅ 一次性加载 | 1.2-1.5x |
| **总提速** | - | - | **15-25x** |

### 3. 内存使用

**原版**:
- 逐样本处理，内存占用低
- GPU 内存: ~12 GB (模型参数 + 单样本)

**优化版**:
- 预加载所有数据到内存
- GPU 内存: ~12 GB (模型参数 + 批量数据)
- CPU 内存: 额外 ~500 MB (11,313 samples × 60 × 8 × 4 bytes ≈ 22 MB)

### 4. 代码差异

**新增功能 (eval_m4_fast.py)**:
1. `--batch-size` 参数 (默认 128)
2. `get_all_data()` 方法 (预加载数据)
3. `@jax.jit` 装饰器 (JIT 编译)
4. 进度显示 (每 10 个 batch 打印一次)
5. 输出文件名: `m4_eval_metrics_fast.json` / `m4_eval_run_fast.md`

---

## 使用方法

### 原版 (慢速，验证流程)

```bash
python src/alphatrade/scripts/eval_m4.py \
  --train-metrics $ALPHATRADE_RUNS_ROOT/reports/m4_train_metrics.json \
  --dataset-config configs/dataset/m2.yaml \
  --split val
```

**预计时间**: 30-40 分钟

### 优化版 (快速，生产使用)

```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics $ALPHATRADE_RUNS_ROOT/reports/m4_train_metrics.json \
  --dataset-config configs/dataset/m2.yaml \
  --split val \
  --batch-size 128
```

**预计时间**: 1-2 分钟

### 调整 batch size

```bash
# 更大的 batch (更快，但需要更多 GPU 内存)
python src/alphatrade/scripts/eval_m4_fast.py \
  --batch-size 256 \
  ...

# 更小的 batch (更慢，但内存占用少)
python src/alphatrade/scripts/eval_m4_fast.py \
  --batch-size 64 \
  ...
```

---

## 验证一致性

两个版本应该产生相同的评估结果（数值上可能有微小差异，因为浮点运算顺序不同）。

### 对比输出

```bash
# 对比 JSON 输出
diff $ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics.json $ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics_fast.json

# 对比关键指标
python -c "
import json
slow = json.load(open('$ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics.json'))
fast = json.load(open('$ALPHATRADE_RUNS_ROOT/reports/m4_eval_metrics_fast.json'))

print('Pinball Loss:')
print(f\"  Slow: {slow['pinball_loss']['overall']:.6f}\")
print(f\"  Fast: {fast['pinball_loss']['overall']:.6f}\")
print(f\"  Diff: {abs(slow['pinball_loss']['overall'] - fast['pinball_loss']['overall']):.6e}\")

print('\\nIC:')
print(f\"  Slow: {slow['ic_metrics']['ic']:.6f}\")
print(f\"  Fast: {fast['ic_metrics']['ic']:.6f}\")
print(f\"  Diff: {abs(slow['ic_metrics']['ic'] - fast['ic_metrics']['ic']):.6e}\")
"
```

**预期**: 差异应该 < 1e-5 (浮点精度范围内)

---

## 性能基准测试

### 测试环境

- GPU: NVIDIA GeForce RTX 4060 Ti (16 GB)
- Samples: 11,313 (val split)
- Model: AlphaTrade v0.2 (6.3M params)

### 预期结果

| Batch Size | 迭代次数 | 预计时间 | GPU 利用率 | GPU 内存 |
|------------|----------|----------|------------|----------|
| 1 (原版) | 11,313 | 30-40 分钟 | 5% | 12 GB |
| 64 | 177 | 3-4 分钟 | 40-50% | 12 GB |
| 128 | 89 | 1-2 分钟 | 60-80% | 12 GB |
| 256 | 45 | 1 分钟 | 80-95% | 13 GB |
| 512 | 23 | 45 秒 | 95%+ | 14 GB |

**推荐**: batch_size=128 或 256 (平衡速度和内存)

---

## 何时使用哪个版本

### 使用原版 (eval_m4.py)

- ✅ 首次验证评估流程
- ✅ 调试评估逻辑
- ✅ 内存受限环境
- ✅ 需要逐样本分析

### 使用优化版 (eval_m4_fast.py)

- ✅ 生产环境评估
- ✅ 大规模实验
- ✅ 需要快速迭代
- ✅ GPU 资源充足

---

## 后续优化空间

### 1. 多 GPU 支持

```python
# 使用 jax.pmap 进行数据并行
@jax.pmap
def inference_fn(params, state, X_batch):
    ...
```

**预期提速**: 2-4x (取决于 GPU 数量)

### 2. 混合精度推理

```python
# 使用 float16 减少内存和计算
X_batch = X_batch.astype(jnp.float16)
```

**预期提速**: 1.5-2x
**内存节省**: 50%

### 3. 模型量化

```python
# 量化模型参数到 int8
params_quantized = quantize_params(params)
```

**预期提速**: 2-3x
**内存节省**: 75%

---

## 总结

| 指标 | 原版 | 优化版 | 改进 |
|------|------|--------|------|
| 时间 | 30-40 分钟 | 1-2 分钟 | **15-25x** |
| GPU 利用率 | 5% | 60-80% | **12-16x** |
| 迭代次数 | 11,313 | 89 | **127x** |
| 代码复杂度 | 简单 | 中等 | - |
| 内存占用 | 低 | 中 | - |

**推荐策略**:
1. 首次运行: 使用原版验证流程
2. 生产使用: 切换到优化版
3. 对比结果: 确保一致性

---

**创建时间**: 2026-03-01
**下次更新**: 性能测试完成后
