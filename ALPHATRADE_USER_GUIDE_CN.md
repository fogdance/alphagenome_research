# AlphaTrade 使用指南

## 简介

AlphaTrade 是一个基于 AlphaGenome 架构的多时间跨度金融时序预测模型，专门用于预测未来对数收益的分位数分布。

## 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate alphatrade

# 验证安装
python -c "import alphagenome_research.alphatrade; print('✓ AlphaTrade installed')"
```

### 2. 运行快速演示

```bash
# 快速演示（30秒内完成）
python -m alphagenome_research.alphatrade.quick_demo

# 完整演示（包含训练、推理、API）
python -m alphagenome_research.alphatrade.demo
```

### 3. 基础使用

```python
from alphagenome_research.alphatrade import (
    schemas, training, preprocessing
)
import jax
import jax.numpy as jnp

# 1. 配置模型
config = schemas.AlphaTradeConfig(
    lookback_length=4096,      # 回看窗口：4096分钟
    horizons=[1, 5, 20, 60],   # 预测时间跨度
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],  # 分位数
)

# 2. 准备数据
preprocessor = preprocessing.FeaturePreprocessor()
features = preprocessor.compute_features(
    open_prices=open_prices,
    high_prices=high_prices,
    low_prices=low_prices,
    close_prices=close_prices,
    volumes=volumes,
)

# 3. 初始化模型
rng = jax.random.PRNGKey(0)
train_state = training.create_train_state(config, rng)

# 4. 训练（示例）
for batch in train_loader:
    train_state, metrics = training.train_step(
        train_state, batch, rng, config
    )
    print(f"Loss: {metrics['loss']:.6f}")

# 5. 预测
features_batch = features[jnp.newaxis, ...]  # 添加batch维度
predictions = training.predict(train_state, features_batch, rng)

# 6. 查看结果
for horizon, quantiles in predictions.log_return_quantiles.items():
    print(f"Horizon {horizon}分钟:")
    for i, q_level in enumerate(config.quantiles):
        print(f"  q{int(q_level*100)} = {quantiles[0, i]:.6f}")
```

## 详细教程

### 数据准备

#### 从OHLCV计算特征

```python
import numpy as np
from alphagenome_research.alphatrade import preprocessing

# 准备OHLCV数据（示例）
L = 4096  # 回看长度
close_prices = np.array([...])  # 收盘价
open_prices = np.array([...])   # 开盘价
high_prices = np.array([...])   # 最高价
low_prices = np.array([...])    # 最低价
volumes = np.array([...])       # 成交量

# 计算8个特征
preprocessor = preprocessing.FeaturePreprocessor(
    epsilon=1e-9,
    session_period_minutes=1440,
)

features = preprocessor.compute_features(
    open_prices=jnp.array(open_prices),
    high_prices=jnp.array(high_prices),
    low_prices=jnp.array(low_prices),
    close_prices=jnp.array(close_prices),
    volumes=jnp.array(volumes),
)

print(f"特征形状: {features.shape}")  # (4096, 8)
```

#### 特征归一化

```python
# 在训练集上拟合scaler
scaler = preprocessing.RobustScaler()
train_features = np.array([...])  # [N, L, 8]
scaler.fit(train_features)

# 转换训练集和测试集
train_normalized = scaler.transform(train_features)
test_normalized = scaler.transform(test_features)

# 保存scaler参数
scaler_params = scaler.get_params()
# 稍后加载
scaler_new = preprocessing.RobustScaler()
scaler_new.set_params(scaler_params)
```

### 模型配置

#### 小型模型（快速实验）

```python
config = schemas.AlphaTradeConfig(
    lookback_length=512,
    stem_channels=64,
    num_encoder_stages=4,      # /16下采样
    channel_increment=32,
    d_model=256,
    num_transformer_layers=2,
    num_heads=4,
    horizons=[1, 5, 20],
)
```

#### 标准模型（推荐）

```python
config = schemas.AlphaTradeConfig(
    lookback_length=4096,
    stem_channels=128,
    num_encoder_stages=6,      # /128下采样
    channel_increment=64,
    d_model=512,
    num_transformer_layers=6,
    num_heads=8,
    horizons=[1, 5, 20, 60],
)
```

#### 大型模型（高精度）

```python
config = schemas.AlphaTradeConfig(
    lookback_length=8192,
    stem_channels=192,
    num_encoder_stages=7,      # /256下采样
    channel_increment=96,
    d_model=768,
    num_transformer_layers=10,
    num_heads=12,
    horizons=[1, 5, 20, 60, 120],
)
```

### 训练流程

#### 准备训练数据

```python
from alphagenome_research.alphatrade import schemas

# 创建训练批次
def create_batch(features, targets):
    return schemas.TrainingBatch(
        inputs=schemas.AlphaTradeInput(
            features=jnp.array(features)
        ),
        targets=schemas.AlphaTradeTargets(
            log_returns={
                1: jnp.array(targets['h1']),
                5: jnp.array(targets['h5']),
                20: jnp.array(targets['h20']),
                60: jnp.array(targets['h60']),
            }
        )
    )
```

#### 训练循环

```python
from alphagenome_research.alphatrade import training

# 初始化
rng = jax.random.PRNGKey(42)
train_state = training.create_train_state(
    config=config,
    rng=rng,
    learning_rate=1e-4,
)

# 训练
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch in train_loader:
        rng, step_rng = jax.random.split(rng)
        train_state, metrics = training.train_step(
            train_state, batch, step_rng, config
        )
        epoch_loss += float(metrics['loss'])

    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.6f}")

    # 验证
    if (epoch + 1) % 5 == 0:
        val_metrics = evaluate(train_state, val_loader, config)
        print(f"  Validation: {val_metrics}")
```

#### 评估

```python
def evaluate(train_state, val_loader, config):
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for batch in val_loader:
        rng = jax.random.PRNGKey(0)
        metrics = training.eval_step(
            train_state, batch, rng, config
        )
        total_loss += float(metrics['loss'])

        # 收集预测
        predictions = training.predict(
            train_state, batch.inputs.features, rng
        )
        all_predictions.append(predictions)
        all_targets.append(batch.targets)

    # 计算覆盖率
    coverage = training.compute_quantile_coverage(
        y_true=all_targets,
        y_pred_quantiles=all_predictions,
        quantile_levels=config.quantiles,
    )

    return {
        'loss': total_loss / len(val_loader),
        'coverage': coverage,
    }
```

### API服务部署

#### 创建服务

```python
from alphagenome_research.alphatrade import api

# 从训练状态创建服务
service = api.create_service_from_train_state(
    train_state=train_state,
    config=config,
    scaler=scaler,  # 可选
    model_version='alphatrade_v0.2_production',
)

# 健康检查
health = service.health_check()
print(health)
```

#### 处理预测请求

```python
# 创建请求
request = schemas.PredictionRequest(
    instrument_id='IF2403',
    asof_bar_end='2026-02-13T10:07:00+08:00',
    features=features,  # [L, 8]
)

# 预测
response = service.predict(request)

# 查看结果
print(f"延迟: {response.latency_ms:.2f}ms")
for horizon, quantiles in response.log_return_quantiles.items():
    print(f"Horizon {horizon}: {quantiles}")
```

#### HTTP服务（示例）

```python
from alphagenome_research.alphatrade import api
from flask import Flask, request, jsonify

app = Flask(__name__)
handler = api.AlphaTradeHTTPHandler(service)

@app.route('/v0.1/predict', methods=['POST'])
def predict():
    request_json = request.get_json()
    response = handler.handle_predict(request_json)
    return jsonify(response)

@app.route('/v0.1/health', methods=['GET'])
def health():
    response = handler.handle_health()
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## 常见问题

### Q1: 如何处理不同的交易时段？

```python
# 使用 is_session_open 标志
is_session_open = np.ones(L)
# 标记休市时段
is_session_open[lunch_break_indices] = 0
is_session_open[after_hours_indices] = 0

features = preprocessor.compute_features(
    ...,
    is_session_open=jnp.array(is_session_open),
)
```

### Q2: 如何调整预测时间跨度？

```python
# 修改配置
config = schemas.AlphaTradeConfig(
    horizons=[1, 3, 5, 10, 15, 30, 60],  # 自定义时间跨度
)
```

### Q3: 如何添加更多特征？

当前版本固定8个特征。如需添加特征：
1. 修改 `schemas.py` 中的 `num_features`
2. 修改 `preprocessing.py` 中的特征计算
3. 更新 `FeatureEmbedder` 的输入维度

### Q4: 梯度爆炸怎么办？

```python
# 在 training.py 中添加梯度裁剪
import optax

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # 梯度裁剪
    optax.adam(learning_rate),
)
```

### Q5: 如何保存和加载模型？

```python
# 保存（需要实现）
# TODO: 使用 Orbax 保存 train_state

# 加载（需要实现）
# TODO: 使用 Orbax 加载 train_state
```

## 性能优化建议

### 1. 使用GPU

```python
# 检查GPU可用性
import jax
print(jax.devices())  # 应该显示GPU设备

# JAX会自动使用GPU
```

### 2. 批处理

```python
# 增大batch size以提高GPU利用率
batch_size = 32  # 根据GPU内存调整
```

### 3. JIT编译

```python
# 使用JIT加速（需要修复静态参数问题）
# train_step_jit = jax.jit(train_step, static_argnums=(...))
```

### 4. 混合精度

```python
# 使用bfloat16加速训练
# TODO: 在模型中添加混合精度支持
```

## 最佳实践

1. **数据质量**
   - 确保OHLCV数据无缺失
   - 处理异常值和停牌
   - 使用is_session_open标记非交易时段

2. **特征工程**
   - 在训练集上拟合scaler
   - 保存scaler参数用于线上推理
   - 验证特征分布稳定性

3. **模型训练**
   - 使用walk-forward验证
   - 监控梯度范数
   - 定期评估覆盖率和校准误差

4. **生产部署**
   - 监控推理延迟
   - 记录预测和实际结果
   - 定期重训练模型

## 参考资料

- 模型规范: `docs/alphaTrade/mode.md`
- API规范: `docs/alphaTrade/api.md`
- 实现总结: `src/alphagenome_research/alphatrade/IMPLEMENTATION_SUMMARY.md`
- 完成报告: `ALPHATRADE_COMPLETION_REPORT.md`

## 支持

如有问题，请查看：
- README: `src/alphagenome_research/alphatrade/README.md`
- 测试用例: `src/alphagenome_research/alphatrade/*_test.py`
- Demo代码: `src/alphagenome_research/alphatrade/demo.py`