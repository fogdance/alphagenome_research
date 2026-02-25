# AlphaTrade 校准问题修复建议

## 问题总结

你的模型有严重的校准问题，根本原因是：**模型没有学到输入特征与目标的关系**。

### 诊断结果

1. **数据有信号**：线性模型可以达到 R²=2-4%，corr=0.15-0.21
2. **深度模型失败**：预测 corr≈0.00-0.04，比线性回归差 5 倍
3. **预测几乎是常数**：std 只有真实值的 1/4
4. **校准失败**：
   - h=1: coverage_q90=0.994 (应该 0.90) → 预测区间太宽
   - h=60: coverage_q90=0.881 (应该 0.90) → 预测区间太窄

---

## 修复方案

### 方案 A：简化模型（推荐）

**问题**：模型太复杂，数据太少，信号太弱

**修改配置**：

```python
config = schemas.AlphaTradeConfig(
    # 减少模型容量
    d_model=64,           # 从 128 → 64
    num_layers=2,         # 从 3 → 2
    num_heads=2,          # 从 4 → 2

    # 减少序列长度
    lookback_length=1024, # 从 4096 → 1024

    # 其他保持不变
    num_features=8,
    horizons=[1, 5, 20, 60],
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
)
```

**原因**：
- 更小的模型更容易学到弱信号
- 更短的序列减少过拟合风险
- 更少的参数需要更少的数据

---

### 方案 B：增加正则化

**修改训练**：

```python
# 1. 添加 weight decay
optimizer = optax.adamw(
    learning_rate=1e-4,
    weight_decay=1e-4,  # 添加 L2 正则化
)

# 2. 添加 dropout
config = schemas.AlphaTradeConfig(
    ...
    dropout_rate=0.1,  # 如果支持的话
)

# 3. 使用更小的学习率
learning_rate = 5e-5  # 从 1e-4 → 5e-5
```

---

### 方案 C：改进数据

**1. 移除常数特征**：

```python
# Feature 7 是常数，应该移除
X = X[:, :, :7]  # 只保留前 7 个特征
config.num_features = 7
```

**2. 检查数据质量**：

```python
# 检查是否有足够的训练数据
train = np.load('train.npz')
print(f"Training samples: {len(train['features'])}")

# 如果 < 5000，考虑：
# - 收集更多数据
# - 使用数据增强
# - 使用更简单的模型
```

---

### 方案 D：使用简单基线

**如果深度模型持续失败，使用线性模型**：

```python
from sklearn.linear_model import QuantileRegressor

# 对每个 horizon 和 quantile 训练一个线性模型
models = {}
for h in [1, 5, 20, 60]:
    models[h] = {}
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        model = QuantileRegressor(quantile=q, alpha=0.01)
        model.fit(X_train[:, -1, :7], y_train[h])
        models[h][q] = model

# 这个基线应该能达到 corr=0.15-0.21
# 如果深度模型不能超过它，说明模型设计有问题
```

---

## 实施步骤

### Step 1: 重新训练简化模型

```bash
# 修改配置文件，使用方案 A 的参数
# 然后重新训练
python train_loop_minimal.py --config config_simple.json
```

### Step 2: 监控训练

检查以下指标：

```python
# 每 500 steps 检查：
# 1. eval_loss 是否持续下降
# 2. 预测的 std 是否接近真实值的 std
# 3. 预测与真实值的相关性是否 > 0.1

# 如果训练 5000 steps 后 corr 仍然 < 0.1，停止训练
# 说明模型架构有问题
```

### Step 3: 验证校准

```bash
# 使用诊断脚本
python diagnose_calibration.py

# 期望结果：
# - corr > 0.15 (至少接近线性模型)
# - Scale ratio 在 0.8-1.2 之间
# - Coverage 偏差 < 0.1
```

---

## 预期结果

### 如果方案 A 成功：

```
h=1:  corr=0.15-0.20, Scale ratio=0.9-1.1, coverage_q90=0.85-0.95
h=5:  corr=0.18-0.23, Scale ratio=0.9-1.1, coverage_q90=0.85-0.95
h=20: corr=0.12-0.18, Scale ratio=0.9-1.1, coverage_q90=0.85-0.95
h=60: corr=0.15-0.22, Scale ratio=0.9-1.1, coverage_q90=0.85-0.95
```

### 如果仍然失败：

说明问题不在模型容量，而在：
1. **数据质量**：特征与目标没有真实关系
2. **实现 bug**：模型实现有问题（但我们已经修复了所有 P0 bug）
3. **任务太难**：金融预测本身就很难，R²=2-4% 可能是上限

---

## 最终建议

**如果你的目标是生产可用的模型**：

1. ✅ 先用方案 A（简化模型）
2. ✅ 如果失败，用方案 D（线性基线）
3. ✅ 如果线性基线也失败（corr < 0.1），说明数据本身没有足够信号

**如果你的目标是研究/学习**：

1. ✅ 当前的诊断工具已经很完善
2. ✅ 你已经理解了校准失败的原因
3. ✅ 可以尝试不同的架构和超参数

---

## 附录：为什么 Pinball Loss 小但校准差？

Pinball loss = 0.0014 看起来很好，但这是**误导性的**：

```python
# 如果模型预测常数（无条件分位数）：
q10_pred = np.quantile(y_train, 0.10)  # 常数
q50_pred = np.quantile(y_train, 0.50)  # 常数
q90_pred = np.quantile(y_train, 0.90)  # 常数

# Pinball loss 会很小（因为预测了合理的分位数）
# 但 calibration 会失败（因为没有条件信息）
```

**这就是你的模型现在的状态**：学到了无条件分位数，但没有学到条件关系。

---

**下一步**：修改配置，重新训练，然后运行 `diagnose_calibration.py` 验证。
