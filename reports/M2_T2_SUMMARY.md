# M2-T2 总结报告：训练目标与 Loss 验证

生成时间: 2026-03-01

## 任务目标

验证训练 loss 的正确性和收敛性：
- Pinball loss 计算正确
- Quantile crossing penalty 生效
- By-horizon loss 统计
- 200 steps 内 loss 下降

---

## ✅ 完成内容

### 1. Loss 验证脚本

**文件**: `src/alphatrade/scripts/test_m2_loss.py`

**功能**:
- ✅ Pinball loss 计算（带 per-horizon 详情）
- ✅ Quantile crossing penalty 计算
- ✅ By-horizon loss 统计
- ✅ 收敛性检查
- ✅ 产出详细报告

---

## 验收结果

### 测试命令

```bash
python src/alphatrade/scripts/test_m2_loss.py \
  --config configs/dataset/m2.yaml \
  --max-steps 200 \
  --smoke
```

### 测试结果

✅ **所有检查通过**

| 指标 | 值 |
|------|-----|
| Max steps | 200 |
| Symbols | 3 |
| Loss trend | Decreasing ✅ |
| No NaN | ✅ |
| No Inf | ✅ |
| Stable | ✅ |

---

## Loss 统计

### Total Loss

| 指标 | 值 |
|------|-----|
| First | 0.106073 |
| Last | 0.003094 |
| Min | 0.003094 |
| Mean | 0.010719 |
| **Decrease** | **97.1%** ✅ |

**趋势**: Decreasing (从 0.106 降到 0.003)

### Pinball Loss

| 指标 | 值 |
|------|-----|
| First | 0.089192 |
| Last | 0.002877 |
| Min | 0.002877 |
| **Decrease** | **96.8%** ✅ |

### Crossing Penalty

| 指标 | 值 |
|------|-----|
| First | 0.168808 |
| Last | 0.002170 |
| Min | 0.002079 |
| Active | ✅ |
| **Decrease** | **98.7%** ✅ |

**观察**: Crossing penalty 从 0.169 降到 0.002，说明模型学会了保持 quantile 顺序

---

## By-Horizon Loss

### 详细统计

| Horizon | First | Last | Min | Mean | Decrease |
|---------|-------|------|-----|------|----------|
| h1 (1 bar) | 0.032885 | 0.002524 | 0.002524 | 0.007839 | 92.3% |
| h5 (5 bars) | 0.118390 | 0.003210 | 0.003210 | 0.010687 | 97.3% |
| h20 (20 bars) | 0.099407 | 0.002949 | 0.002891 | 0.009893 | 97.0% |
| h60 (60 bars) | 0.106086 | 0.002825 | 0.002825 | 0.010454 | 97.3% |

### 观察

1. **所有 horizon 都在下降** ✅
   - h1: 92.3% decrease
   - h5: 97.3% decrease
   - h20: 97.0% decrease
   - h60: 97.3% decrease

2. **Loss 大小合理**
   - 最终 loss 在 0.0025-0.0032 范围
   - 符合 log return 的量级

3. **Horizon 之间平衡**
   - 使用 equal weights [1.0, 1.0, 1.0, 1.0]
   - 各 horizon loss 相近，说明训练平衡

---

## 收敛性验证

### ✅ 所有检查通过

| 检查项 | 结果 | 说明 |
|--------|------|------|
| Loss decreased | ✅ | 从 0.106 降到 0.003 (97.1%) |
| No NaN | ✅ | 所有 200 steps 无 NaN |
| No Inf | ✅ | 所有 200 steps 无 Inf |
| Stable | ✅ | 最后 20 steps std < 0.01 |

### Loss 曲线

```
Step 50:  Loss=0.011334, Pinball=0.010498, Crossing=0.008353
Step 100: Loss=0.006375, Pinball=0.005942, Crossing=0.004327
Step 150: Loss=0.004310, Pinball=0.004011, Crossing=0.002990
Step 200: Loss=0.003094, Pinball=0.002877, Crossing=0.002170
```

**观察**: 持续下降，无震荡

---

## Loss 实现验证

### 1. Pinball Loss ✅

**公式**:
```python
loss = where(error >= 0, 
             quantile * error,
             (quantile - 1) * error)
```

**验证**: 
- ✅ 正确实现
- ✅ Per-horizon 统计正常
- ✅ Horizon weights 生效

### 2. Quantile Crossing Penalty ✅

**公式**:
```python
diffs = pred[:, :, 1:] - pred[:, :, :-1]
penalty = relu(-diffs).mean()
```

**验证**:
- ✅ 正确实现
- ✅ Penalty 从 0.169 降到 0.002
- ✅ 模型学会保持 quantile 顺序

### 3. By-Horizon Loss ✅

**实现**:
```python
loss_per_horizon = loss.mean(dim=(0, 2))  # [H]
```

**验证**:
- ✅ 正确计算
- ✅ 4 个 horizon 都有统计
- ✅ 可用于 metrics 输出

---

## 配置验证

### Quantiles

- **Levels**: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Count**: 5
- **验证**: ✅ 正确使用

### Horizon Weights

- **Weights**: [1.0, 1.0, 1.0, 1.0]
- **Strategy**: Equal weights
- **验证**: ✅ 各 horizon loss 平衡

### Crossing Penalty

- **Enabled**: True
- **Weight**: 0.1
- **验证**: ✅ 生效且下降

---

## 生成的文件

### 报告文件

- `reports/m2_t2_loss_sanity.md` - Loss 验证报告
- `reports/M2_T2_SUMMARY.md` - 本文件

### 报告内容

**m2_t2_loss_sanity.md** 包含:
- 配置信息
- Total loss 统计
- Pinball loss 统计
- Crossing penalty 统计
- By-horizon loss 表格
- 收敛性检查结果

---

## 下一步：M2-T3

### 任务

接入 AlphaTrade v0.2 真实模型：
- 替换 SimpleQuantileModel
- 使用 JAX/Haiku 实现
- 保持 metrics schema 不变
- 稳定训练验证

### 验收

- 使用真实 AlphaTrade v0.2 模型
- 训练稳定（无 NaN/Inf）
- 产出 m2_train_metrics_v1 schema 报告

### 产物

- 更新后的 `train_m2_alphatrade_v0_2.py`
- `reports/m2_train_metrics.json`
- `reports/m2_train_run.md`

---

## 总结

✅ **M2-T2 验收通过**

**核心成果**:
- Loss 验证脚本完成
- 200 steps 训练成功
- 所有收敛性检查通过

**Loss 验证**:
- ✅ Pinball loss 正确实现
- ✅ Crossing penalty 生效（0.169 → 0.002）
- ✅ By-horizon loss 统计正常
- ✅ Loss 下降 97.1%

**收敛性**:
- ✅ Loss 持续下降
- ✅ 无 NaN/Inf
- ✅ 训练稳定

**可以继续 M2-T3**: Loss 实现验证完成
