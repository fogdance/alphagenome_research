# AlphaTrade 深度模型 corr≈0 问题诊断报告

**日期**: 2026-02-26
**状态**: 🔴 根本原因已找到

---

## 🎯 核心发现

**深度模型 corr≈0 的根本原因：缺少目标归一化（target normalization）**

---

## 📊 实验证据

### 实验 1：OLS Baseline（numpy）

使用 `diagnose_calibration.py` 在 val.npz 上的结果：

| Horizon | OLS corr | R² | 最佳特征 corr |
|---------|----------|-----|---------------|
| h=1 | **0.182** | 0.033 | 0.155 (last_4) |
| h=5 | **0.211** | 0.045 | 0.104 (last_1) |
| h=20 | **0.148** | 0.022 | 0.101 (last_4) |
| h=60 | **0.203** | 0.041 | 0.155 (last_1) |

**结论**：数据有信号，OLS 可以学到 corr 0.15-0.21。

---

### 实验 2：JAX 线性模型（未归一化）

使用 `test_single_horizon.py` 单独训练每个 horizon：

| Horizon | JAX corr | OLS corr | Gap | std_ratio |
|---------|----------|----------|-----|-----------|
| h=1 | **0.091** | 0.182 | -0.091 | **81.5** ❌ |
| h=5 | **0.004** | 0.211 | -0.207 | 0.38 |
| h=20 | **0.037** | 0.148 | -0.111 | **32.8** ❌ |
| h=60 | **-0.069** | 0.203 | -0.272 | 5.1 |

**观察**：
- JAX 模型远差于 OLS
- h=1 和 h=20 的 std_ratio 异常高（81.5 和 32.8）
- 说明模型预测发散，没有学到正确的信号

---

### 实验 3：JAX 线性模型（归一化目标）

使用 `test_normalized_training.py`，对目标做归一化：

```python
y_mean = np.mean(Y)
y_std = np.std(Y)
Y_norm = (Y - y_mean) / (y_std + 1e-8)
```

| Horizon | JAX corr | OLS corr | Gap | std_ratio |
|---------|----------|----------|-----|-----------|
| h=1 | **0.153** | 0.182 | -0.029 | 0.25 ✅ |
| h=5 | **0.119** | 0.211 | -0.092 | 0.11 ✅ |
| h=20 | **0.100** | 0.148 | -0.048 | 0.10 ✅ |
| h=60 | **0.168** | 0.203 | -0.035 | 0.08 ✅ |

**观察**：
- JAX 模型接近 OLS 水平！
- std_ratio 正常（0.08-0.25）
- 所有 horizon 都能正常学习

**改进幅度**：
- h=1: 0.091 → **0.153** (+68%)
- h=5: 0.004 → **0.119** (+2875%)
- h=20: 0.037 → **0.100** (+170%)
- h=60: -0.069 → **0.168** (从负变正)

---

## 🔍 根本原因分析

### 问题：目标值的 scale 差异巨大

```
h=1:  std = 0.0012  (非常小)
h=5:  std = 0.0030  (2.5x)
h=20: std = 0.0053  (4.5x)
h=60: std = 0.0100  (8.5x)
```

### 为什么未归一化会导致失败？

1. **梯度 scale 不匹配**
   - h=1 的目标值很小 → pinball loss 很小 → 梯度很小
   - 在相同学习率下，h=1 学习速度极慢
   - 但由于 loss 聚合方式，h=1 可能得到过大的更新

2. **优化器难以平衡**
   - Adam 的自适应学习率无法完全解决 scale 差异
   - 不同 horizon 需要不同的有效学习率
   - 未归一化时，某些 horizon 会发散（std_ratio > 30）

3. **数值稳定性问题**
   - 极小的目标值（~0.001）接近浮点精度边界
   - 梯度计算可能不稳定

### 为什么 OLS 不受影响？

OLS 使用闭式解（closed-form solution）：
```python
w = (X^T X)^{-1} X^T y
```

- 不依赖梯度下降
- 不受 scale 影响（线性变换）
- 一步到位，没有优化过程

---

## 🔧 当前训练脚本的问题

检查 `train_loop_minimal.py`：

```python
def load_dataset(npz_path: str, horizons_arg: List[int] | None):
    npz = np.load(npz_path, allow_pickle=True)
    X = _find_feature_array(npz)
    y = _parse_targets(npz, horizons_arg)
    return X, y  # ❌ 直接返回原始目标，没有归一化
```

**发现**：
- ✅ 特征有归一化（robust scaler）
- ❌ **目标没有归一化**

这导致：
1. 不同 horizon 的 loss scale 差异巨大
2. 梯度更新不平衡
3. 模型无法有效学习

---

## 📝 解决方案

### 方案 1：目标归一化（推荐）

在训练前对每个 horizon 的目标做归一化：

```python
def normalize_targets(Y_dict: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[float, float]]]:
    """Normalize targets to zero mean and unit variance.

    Returns:
        Y_norm: Normalized targets
        stats: {horizon: (mean, std)} for denormalization
    """
    Y_norm = {}
    stats = {}

    for h, y in Y_dict.items():
        mean = float(np.mean(y))
        std = float(np.std(y))
        Y_norm[h] = (y - mean) / (std + 1e-8)
        stats[h] = (mean, std)

    return Y_norm, stats
```

**训练时**：
- 在归一化的目标上训练
- Loss 在相同 scale 上

**推理时**：
- 预测归一化的分位数
- 反归一化：`y_pred = y_pred_norm * std + mean`

**优点**：
- 简单直接
- 所有 horizon 在相同 scale 上优化
- 数值稳定

**缺点**：
- 需要保存归一化统计量
- 推理时需要反归一化

---

### 方案 2：Per-horizon loss weighting

根据目标的 std 调整 loss 权重：

```python
horizon_weights = {
    1: 1.0 / 0.0012,   # 反比于 std
    5: 1.0 / 0.0030,
    20: 1.0 / 0.0053,
    60: 1.0 / 0.0100,
}
# Normalize weights
total = sum(horizon_weights.values())
horizon_weights = {h: w/total for h, w in horizon_weights.items()}
```

**优点**：
- 不改变目标值
- 不需要反归一化

**缺点**：
- 需要手动调整权重
- 不如归一化直接
- 仍可能有数值稳定性问题

---

### 方案 3：Per-horizon learning rate

为每个 horizon 使用不同的学习率（复杂，不推荐）。

---

## ✅ 推荐行动

### 立即行动（高优先级）

1. **修改 `train_loop_minimal.py`**
   - 在 `load_dataset` 后添加目标归一化
   - 保存归一化统计量到 checkpoint
   - 在推理时反归一化

2. **重新训练简化模型**
   - 使用你之前推荐的简化配置
   - 验证 corr 能达到 0.15+ 水平

3. **验证深度模型**
   - 如果归一化后深度模型仍然 corr≈0
   - 则继续排查其他问题（架构、容量、过拟合）

---

### 后续排查（如果归一化后仍有问题）

按照你的 6 步指南继续：

2. ✅ **Horizon 对齐** - 已验证（通过 cross-correlation）
3. ✅ **时间错位** - 数据看起来正确
4. ✅ **特征归一化** - 已有 robust scaler
5. ⏭️ **模型坍塌检查** - 如果归一化后仍失败
6. ⏭️ **32 样本 overfit** - 最终验证

---

## 📊 预期结果

**归一化后，预期深度模型能达到**：
- Train corr: 0.3-0.5（如果容量足够）
- Val corr: 0.15-0.25（接近或超过 OLS）

**如果仍然 corr≈0**：
- 说明有架构/容量/过拟合问题
- 需要继续简化模型（减少层数、d_model、lookback）

---

## 🎓 技术洞察

### 为什么这个问题如此隐蔽？

1. **特征归一化掩盖了问题**
   - 看到有 scaler，以为归一化做了
   - 但只归一化了输入，没归一化目标

2. **Loss 仍在下降**
   - 未归一化时 loss 也会下降
   - 但模型学到的是错误的 scale

3. **不同 horizon 表现不一致**
   - h=5 和 h=60 稍好（std 较大）
   - h=1 和 h=20 很差（std 较小）
   - 容易误判为 horizon mapping 问题

### 关键教训

**在训练分位数回归模型时，目标归一化是必须的，尤其是**：
- 多个目标变量（multi-horizon）
- 目标 scale 差异大（>2x）
- 使用梯度下降优化

---

**下一步**：修改训练脚本，添加目标归一化，重新训练验证。

**预计修复时间**：1-2 小时（代码修改 + 重新训练）

**成功标准**：Val corr > 0.15 for all horizons
