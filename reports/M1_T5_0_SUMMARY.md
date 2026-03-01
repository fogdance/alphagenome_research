# M1-T5.0 数据契约核对总结

生成时间: 2026-03-01

## 任务目标

确认 M1 canonical 数据能正确喂给 AlphaTrade v0.2，验证特征和窗口定位字段。

---

## ✅ 核对结果

### 抽样品种

随机抽取 3 个品种进行检查：
- DCE.P
- CZCE.FG
- SHFE.SN

### 检查项目

#### 1. bars.parquet Schema ✅

**列数**: 16 列

**关键列**:
- 基础 OHLCV: `open`, `high`, `low`, `close`, `volume`, `position`
- 时间戳: `eob`
- 分段: `segment_id`
- 合约: `symbol`
- **特征 (7个)**: `ret_1m`, `hl_range`, `co_change`, `vol_log1p`, `pos_log1p`, `minute_sin`, `minute_cos`

**数据类型**:
- Float 特征: `float32` ✅
- Segment ID: `int32` ✅
- 时间戳: `datetime64[ns]` ✅

#### 2. index_train.parquet Schema ✅

**列数**: 10 列

**关键列**:
- **窗口定位**: `x_start`, `x_end` ✅
- 时间戳: `eob`
- 分段: `segment_id`
- 样本索引: `t`
- **标签 (4个)**: `y_h1`, `y_h5`, `y_h20`, `y_h60` ✅
- 数据集划分: `split`

**数据类型**:
- 索引字段: `int64` ✅
- 标签: `float64` ✅

---

## 契约验证结果

### ✅ 前 7 个特征 - 全部存在

```python
feature_cols = [
    'ret_1m',      # 1-minute return
    'hl_range',    # (high - low) / close
    'co_change',   # (close - open) / open
    'vol_log1p',   # log(1 + volume)
    'pos_log1p',   # log(1 + position)
    'minute_sin',  # sin(2π * minute / 1440)
    'minute_cos',  # cos(2π * minute / 1440)
]
```

### ❌ 第 8 个特征 - 缺失

**问题**: AlphaTrade v0.2 期望 F=8，但当前只有 7 个特征。

**解决方案**:

#### 方案 A: 补充 is_session_open (推荐)

在 canonical bars 中添加第 8 个特征：

```python
# 在 build_m1_canonical_bars.py 中添加
df['is_session_open'] = 1.0  # 简单版本：全 1
```

**优点**:
- 符合 AlphaTrade v0.2 规范
- 数据完整，不需要 dataloader 特殊处理
- 可扩展（未来可基于 segment_id 标记首 bar）

#### 方案 B: Dataloader 拼接常数 1.0

在 dataloader 中动态添加：

```python
# 在 __getitem__ 中
x = window_df[feature_cols].values  # [L, 7]
ones = np.ones((len(x), 1), dtype=np.float32)
x = np.concatenate([x, ones], axis=1)  # [L, 8]
```

**优点**:
- 不需要重新生成 canonical bars
- 快速实现

**缺点**:
- Dataloader 逻辑复杂
- 不符合数据规范

### ✅ 窗口定位字段 - 存在

```python
window_start_field = 'x_start'  # 窗口起始索引
window_end_field = 'x_end'      # 窗口结束索引
```

**用法**:
```python
# 从 index_train.parquet 读取
index_entry = index_df.iloc[i]
x_start = int(index_entry['x_start'])
x_end = int(index_entry['x_end'])

# 从 bars.parquet 切片窗口
window = bars_df.iloc[x_start:x_end+1]
```

---

## 最终特征列表

### 推荐配置 (方案 A)

```python
feature_cols = [
    'ret_1m',
    'hl_range',
    'co_change',
    'vol_log1p',
    'pos_log1p',
    'minute_sin',
    'minute_cos',
    'is_session_open',  # 第 8 个特征
]
```

**特征数**: F = 8 ✅

---

## 数据质量确认

### 抽样统计

| Symbol | Bars Rows | Index Rows | Features | Window Locators |
|--------|-----------|------------|----------|-----------------|
| DCE.P | 661,738 | 5,652 | 7/8 | ✅ |
| CZCE.FG | 665,733 | 7,905 | 7/8 | ✅ |
| SHFE.SN | 835,338 | 25,662 | 7/8 | ✅ |

### 总体状态

- **前 7 个特征**: ✅ 全部存在
- **第 8 个特征**: ❌ 需要补充
- **窗口定位字段**: ✅ 存在
- **数据类型**: ✅ 正确
- **总体状态**: ✅ PASS (可用，需补充第 8 个特征)

---

## 下一步行动

### 立即可做 (方案 B)

使用现有 7 个特征 + dataloader 拼接常数，直接开始训练：

```python
# 在 dataloader 中
x = window_df[feature_cols].values.astype(np.float32)  # [L, 7]
ones = np.ones((len(x), 1), dtype=np.float32)
x = np.concatenate([x, ones], axis=1)  # [L, 8]
```

### 推荐做法 (方案 A)

1. 修改 `build_m1_canonical_bars.py`，添加 `is_session_open` 列
2. 重新生成 24 个品种的 canonical bars
3. 使用完整的 8 个特征进行训练

---

## 生成的文件

- `reports/m1_t5_contract_check.md` - 详细报告
- `reports/m1_t5_contract_check.json` - 机器可读报告
- `reports/M1_T5_0_SUMMARY.md` - 本文件

---

## 总结

✅ **M1-T5.0 数据契约核对完成**

**核心发现**:
- 前 7 个特征全部存在且正确
- 窗口定位字段 (x_start, x_end) 存在且正确
- 缺少第 8 个特征，但有明确的解决方案

**建议**:
- 短期：使用方案 B (dataloader 拼接) 快速开始训练
- 长期：使用方案 A (补充 is_session_open) 符合规范

**可以继续 M1-T5.1**: 数据契约明确，可以开始训练
