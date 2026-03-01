# M2 Training Data Specification

Version: m2_train_metrics_v1  
Date: 2026-03-01

---

## 输入特征 (8 维)

特征维度固定为 **F=8**，顺序如下（不可变）：

| Index | Feature | Description | Dtype |
|-------|---------|-------------|-------|
| 0 | `ret_1m` | 1-minute return: `(close_t - close_{t-1}) / close_{t-1}` | float32 |
| 1 | `hl_range` | High-low range: `(high - low) / close` | float32 |
| 2 | `co_change` | Close-open change: `(close - open) / open` | float32 |
| 3 | `vol_log1p` | Log volume: `log(1 + volume)` | float32 |
| 4 | `pos_log1p` | Log position: `log(1 + position)` | float32 |
| 5 | `minute_sin` | Minute phase (sin): `sin(2π * minute / 1440)` | float32 |
| 6 | `minute_cos` | Minute phase (cos): `cos(2π * minute / 1440)` | float32 |
| 7 | `is_session_open` | Session open indicator: `1.0` (constant) | float32 |

**输入形状**: `[Batch, Lookback, Features]` = `[B, 60, 8]`

---

## 标签定义

### Return Labels

对于每个 horizon `h ∈ {1, 5, 20, 60}`，标签定义为：

```
y_h = log(close_{t+h} / close_t)
```

其中：
- `close_t`: 当前时刻的收盘价
- `close_{t+h}`: h 个 bar 后的收盘价
- 使用自然对数

**标签形状**: `[Batch, Horizons]` = `[B, 4]`

### Quantile Prediction

模型输出 5 个分位数预测：

```
quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
```

**输出形状**: `[Batch, Horizons, Quantiles]` = `[B, 4, 5]`

---

## 数据约束

### 1. 不跨 Segment

- 每个样本的窗口 `[x_start, x_end]` 必须在同一个 segment 内
- 标签的目标时刻 `t+h` 也必须在同一个 segment 内
- **由 index 生成时保证**（`build_m1_sample_index_v2.py` 中的 `require_same_segment=True`）

### 2. 不跨合约切换

- Segment 的划分基于时间 gap（gap_seconds=1800）
- 合约切换会产生 gap，自动形成新的 segment
- **由 segment 机制保证**

### 3. NaN 处理

- `ret_1m` 的第一行可能是 NaN（pct_change 的结果）
- 使用 `np.nan_to_num(feature_data, nan=0.0)` 填充

---

## 数据分割

### 时间分割

| Split | Start | End | Purpose |
|-------|-------|-----|---------|
| Train | 2018-01-01 | 2023-01-01 | 训练 (5 years) |
| Val | 2023-01-01 | 2024-01-01 | 验证 (1 year) |
| Test | 2024-01-01 | 2026-01-01 | 测试 (2 years) |

### 采样参数

- **Lookback**: 60 bars
- **Stride**: 5 bars
- **Horizons**: [1, 5, 20, 60] bars

---

## Metrics Schema (m2_train_metrics_v1)

### 必须字段

```json
{
  "run": {
    "run_id": "string (8 chars)",
    "git_sha": "string (8 chars)",
    "created_at": "ISO 8601 timestamp",
    "device": "string (cuda/cpu)",
    "seed": "int"
  },
  "dataset": {
    "name": "string (m2)",
    "config_path": "string",
    "processed_dir": "string",
    "symbols": "int (number of symbols)",
    "train_samples": "int",
    "val_samples": "int",
    "test_samples": "int",
    "feature_dim": "int (8)",
    "feature_cols": ["list of 8 strings"],
    "lookback": "int (60)",
    "horizons": ["list of ints"],
    "quantiles": ["list of floats"]
  },
  "model": {
    "type": "string (alphatrade_v0_2)",
    "total_params": "int",
    "trainable_params": "int",
    "config": "dict (model-specific)"
  },
  "training": {
    "max_steps": "int",
    "batch_size": "int",
    "learning_rate": "float",
    "weight_decay": "float",
    "grad_clip": "float",
    "optimizer": "string",
    "lr_schedule": "dict"
  },
  "loss": {
    "train_last": "float",
    "train_best": "float",
    "val_last": "float",
    "val_best": "float",
    "best_step": "int",
    "by_horizon": {
      "h1": {"train": "float", "val": "float"},
      "h5": {"train": "float", "val": "float"},
      "h20": {"train": "float", "val": "float"},
      "h60": {"train": "float", "val": "float"}
    }
  },
  "stability": {
    "nan_steps": "int",
    "inf_steps": "int",
    "max_grad_norm": "float"
  }
}
```

### 字段说明

- **run**: 运行元信息
- **dataset**: 数据集信息（必须包含 feature_cols 和 quantiles）
- **model**: 模型信息
- **training**: 训练配置
- **loss**: Loss 统计（必须包含 by_horizon）
- **stability**: 训练稳定性指标

---

## 数据文件结构

```
data/processed/m1_f8/
└── {SYMBOL}/
    ├── bars.parquet
    │   └── Columns: [eob, open, high, low, close, volume, position,
    │                  symbol, segment_id,
    │                  ret_1m, hl_range, co_change, vol_log1p, pos_log1p,
    │                  minute_sin, minute_cos, is_session_open]
    │
    ├── index_train.parquet
    │   └── Columns: [t, x_start, x_end, eob, segment_id,
    │                  y_h1, y_h5, y_h20, y_h60, split]
    │
    ├── index_val.parquet
    └── index_test.parquet
```

---

## 使用示例

### 加载数据

```python
import pandas as pd
import numpy as np

# Load bars
bars_df = pd.read_parquet('data/processed/m1_f8/DCE.JM/bars.parquet')

# Load index
index_df = pd.read_parquet('data/processed/m1_f8/DCE.JM/index_train.parquet')

# Get a sample
sample = index_df.iloc[0]
x_start = int(sample['x_start'])
x_end = int(sample['x_end'])

# Extract features (8D)
feature_cols = [
    'ret_1m', 'hl_range', 'co_change', 'vol_log1p',
    'pos_log1p', 'minute_sin', 'minute_cos', 'is_session_open'
]
X = bars_df.iloc[x_start:x_end+1][feature_cols].values  # [60, 8]

# Handle NaN
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Extract labels
y = np.array([
    sample['y_h1'],
    sample['y_h5'],
    sample['y_h20'],
    sample['y_h60']
], dtype=np.float32)  # [4]
```

### 验证数据

```python
# Check shape
assert X.shape == (60, 8), f"Expected (60, 8), got {X.shape}"
assert y.shape == (4,), f"Expected (4,), got {y.shape}"

# Check no NaN/Inf
assert not np.isnan(X).any(), "X contains NaN"
assert not np.isinf(X).any(), "X contains Inf"
assert not np.isnan(y).any(), "y contains NaN"
assert not np.isinf(y).any(), "y contains Inf"

# Check segment constraint
assert sample['segment_id'] == bars_df.iloc[x_start]['segment_id']
assert sample['segment_id'] == bars_df.iloc[x_end]['segment_id']
```

---

## 版本历史

- **v1** (2026-03-01): 初始版本
  - 8 维特征
  - 4 个 horizons
  - 5 个 quantiles
  - m2_train_metrics_v1 schema
