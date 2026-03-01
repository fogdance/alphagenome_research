# M1-T5 Data Contract Check Report

生成时间: 2026-03-01 10:09:27

## 目的

确认 M1 canonical 数据能正确喂给 AlphaTrade v0.2，验证：
1. bars.parquet 包含所需的 8 个特征
2. index_train.parquet 包含窗口定位字段

## 抽样品种

- DCE.P
- CZCE.FG
- SHFE.SN

## 品种 1: DCE.P

### bars.parquet Schema

- **Path**: `data/processed/m1/DCE.P/bars.parquet`
- **Rows**: 661,738
- **Columns** (16):
  - `eob`: datetime64[ns]
  - `open`: float32
  - `high`: float32
  - `low`: float32
  - `close`: float32
  - `volume`: float32
  - `position`: float32
  - `symbol`: object
  - `segment_id`: int32
  - `ret_1m`: float32
  - `hl_range`: float32
  - `co_change`: float32
  - `vol_log1p`: float32
  - `pos_log1p`: float32
  - `minute_sin`: float32
  - `minute_cos`: float32

### index_train.parquet Schema

- **Path**: `data/processed/m1/DCE.P/index_train.parquet`
- **Rows**: 5,652
- **Columns** (10):
  - `t`: int64
  - `x_start`: int64
  - `x_end`: int64
  - `eob`: datetime64[ns]
  - `segment_id`: int64
  - `y_h1`: float64
  - `y_h5`: float64
  - `y_h20`: float64
  - `y_h60`: float64
  - `split`: object

### 契约验证

#### 特征检查 (前 7 个)

✅ **所有前 7 个特征都存在**

- ✅ `ret_1m`
- ✅ `hl_range`
- ✅ `co_change`
- ✅ `vol_log1p`
- ✅ `pos_log1p`
- ✅ `minute_sin`
- ✅ `minute_cos`

#### 第 8 个特征检查

❌ **缺少第 8 个特征**

#### 窗口定位字段检查

✅ **窗口定位字段存在**: `x_start`, `x_end`

#### 总体状态: **PASS**

## 品种 2: CZCE.FG

### bars.parquet Schema

- **Path**: `data/processed/m1/CZCE.FG/bars.parquet`
- **Rows**: 665,733
- **Columns** (16):
  - `eob`: datetime64[ns]
  - `open`: float32
  - `high`: float32
  - `low`: float32
  - `close`: float32
  - `volume`: float32
  - `position`: float32
  - `symbol`: object
  - `segment_id`: int32
  - `ret_1m`: float32
  - `hl_range`: float32
  - `co_change`: float32
  - `vol_log1p`: float32
  - `pos_log1p`: float32
  - `minute_sin`: float32
  - `minute_cos`: float32

### index_train.parquet Schema

- **Path**: `data/processed/m1/CZCE.FG/index_train.parquet`
- **Rows**: 7,905
- **Columns** (10):
  - `t`: int64
  - `x_start`: int64
  - `x_end`: int64
  - `eob`: datetime64[ns]
  - `segment_id`: int64
  - `y_h1`: float64
  - `y_h5`: float64
  - `y_h20`: float64
  - `y_h60`: float64
  - `split`: object

### 契约验证

#### 特征检查 (前 7 个)

✅ **所有前 7 个特征都存在**

- ✅ `ret_1m`
- ✅ `hl_range`
- ✅ `co_change`
- ✅ `vol_log1p`
- ✅ `pos_log1p`
- ✅ `minute_sin`
- ✅ `minute_cos`

#### 第 8 个特征检查

❌ **缺少第 8 个特征**

#### 窗口定位字段检查

✅ **窗口定位字段存在**: `x_start`, `x_end`

#### 总体状态: **PASS**

## 品种 3: SHFE.SN

### bars.parquet Schema

- **Path**: `data/processed/m1/SHFE.SN/bars.parquet`
- **Rows**: 835,338
- **Columns** (16):
  - `eob`: datetime64[ns]
  - `open`: float32
  - `high`: float32
  - `low`: float32
  - `close`: float32
  - `volume`: float32
  - `position`: float32
  - `symbol`: object
  - `segment_id`: int32
  - `ret_1m`: float32
  - `hl_range`: float32
  - `co_change`: float32
  - `vol_log1p`: float32
  - `pos_log1p`: float32
  - `minute_sin`: float32
  - `minute_cos`: float32

### index_train.parquet Schema

- **Path**: `data/processed/m1/SHFE.SN/index_train.parquet`
- **Rows**: 25,662
- **Columns** (10):
  - `t`: int64
  - `x_start`: int64
  - `x_end`: int64
  - `eob`: datetime64[ns]
  - `segment_id`: int64
  - `y_h1`: float64
  - `y_h5`: float64
  - `y_h20`: float64
  - `y_h60`: float64
  - `split`: object

### 契约验证

#### 特征检查 (前 7 个)

✅ **所有前 7 个特征都存在**

- ✅ `ret_1m`
- ✅ `hl_range`
- ✅ `co_change`
- ✅ `vol_log1p`
- ✅ `pos_log1p`
- ✅ `minute_sin`
- ✅ `minute_cos`

#### 第 8 个特征检查

❌ **缺少第 8 个特征**

#### 窗口定位字段检查

✅ **窗口定位字段存在**: `x_start`, `x_end`

#### 总体状态: **PASS**

## 总结

- **前 7 个特征**: ✅ 全部存在
- **第 8 个特征**: ❌ 缺失
- **窗口定位字段**: ✅ 存在
- **总体状态**: ✅ PASS

## 最终特征列表 (喂给模型)

### 方案 A: 补充 is_session_open (推荐)

```python
feature_cols = [
    'ret_1m',
    'hl_range',
    'co_change',
    'vol_log1p',
    'pos_log1p',
    'minute_sin',
    'minute_cos',
    'is_session_open',  # 补充：全 1 或基于 segment_id
]
```

### 方案 B: Dataloader 拼接常数 1.0

```python
feature_cols = [
    'ret_1m',
    'hl_range',
    'co_change',
    'vol_log1p',
    'pos_log1p',
    'minute_sin',
    'minute_cos',
]
# 在 dataloader 中拼接一维常数 1.0
```

## 窗口定位字段

```python
# index_train.parquet 中的窗口定位字段
window_start_field = 'x_start'
window_end_field = 'x_end'
```

## 建议

### 补充第 8 个特征

**推荐方案 A**: 在 canonical bars 中补充 `is_session_open` 列

```python
# 在 build_m1_canonical_bars.py 中添加
df['is_session_open'] = 1.0  # 简单版本：全 1
# 或基于 segment_id 的首 bar 标记
```

