# M1-T5.1 总结报告：8 维特征重建

生成时间: 2026-03-01

## 任务目标

全量重建 M1 canonical bars，使特征维度从 7 增加到 8，符合 AlphaTrade v0.2 规范。

---

## ✅ 完成内容

### 1. 统一特征定义模块

**文件**: `src/alphatrade/data_pipeline/feature_schema.py`

**功能**:
- 定义唯一的特征列常量 `FEATURE_COLS` (F=8)
- 特征顺序固定，`is_session_open` 为第 8 列
- 提供统一的 schema 验证函数
- 作为生成端、dataloader、contract check 的单一来源

```python
FEATURE_COLS = [
    "ret_1m",           # 0
    "hl_range",         # 1
    "co_change",        # 2
    "vol_log1p",        # 3
    "pos_log1p",        # 4
    "minute_sin",       # 5
    "minute_cos",       # 6
    "is_session_open",  # 7 (新增)
]
```

### 2. 新的生成脚本（支持多线程）

**文件**: `src/alphatrade/scripts/build_m1_canonical_bars_v2.py`

**新特性**:
- ✅ 添加第 8 个特征 `is_session_open` (float32, 全 1.0)
- ✅ 使用统一的 `FEATURE_COLS` 定义
- ✅ 强制 schema 验证（assert F=8, dtype=float32）
- ✅ 支持多线程并发处理 (`--num-workers`)
- ✅ 原子写入（tmp file → replace）
- ✅ 输出到新目录 `m1_f8`（避免脏状态）

### 3. 全量重建执行

**命令**:
```bash
python src/alphatrade/scripts/build_m1_canonical_bars_v2.py \
  --universe configs/universe/m1_selected.yaml \
  --input-dir data/processed/m1 \
  --output-dir data/processed/m1_f8 \
  --num-workers 8 \
  --force
```

**结果**:
- ✅ 24/24 品种全部成功
- ✅ 使用 8 个并发 worker
- ✅ 输出到新目录 `data/processed/m1_f8`

---

## 重建结果

### 总览

| 指标 | 数值 |
|------|------|
| 总品种数 | 24 |
| 成功 | 24 |
| 失败 | 0 |
| 特征维度 | 8 |
| 输出目录 | `data/processed/m1_f8` |

### 品种统计

| Symbol | Rows | Segments | Feature Dim | Size (MB) |
|--------|------|----------|-------------|-----------|
| SHFE.AU | 978,645 | 5,685 | 8 | 32.5 |
| SHFE.AG | 981,648 | 5,685 | 8 | 30.4 |
| SHFE.NI | 848,466 | 5,683 | 8 | 29.5 |
| SHFE.SN | 835,338 | 5,685 | 8 | 28.1 |
| INE.SC | 949,844 | 5,523 | 8 | 26.1 |
| ... | ... | ... | 8 | ... |

**所有品种的 Feature Dim 均为 8** ✅

---

## Schema 验证

### 抽样验证 (3 个品种)

#### DCE.JM
- Rows: 661,417
- Columns: 17
- ✅ 8 个特征全部存在
- ✅ 所有特征 dtype 为 float32

#### SHFE.AG
- Rows: 981,648
- Columns: 17
- ✅ 8 个特征全部存在
- ✅ 所有特征 dtype 为 float32

#### CZCE.FG
- Rows: 665,733
- Columns: 17
- ✅ 8 个特征全部存在
- ✅ 所有特征 dtype 为 float32

### 特征列表 (F=8)

```python
FEATURE_COLS = [
    'ret_1m',           # float32
    'hl_range',         # float32
    'co_change',        # float32
    'vol_log1p',        # float32
    'pos_log1p',        # float32
    'minute_sin',       # float32
    'minute_cos',       # float32
    'is_session_open',  # float32 ✅ 新增
]
```

---

## Contract Check 结果

### 验证项目

| 检查项 | 结果 |
|--------|------|
| 前 7 个特征存在 | ✅ PASS |
| 第 8 个特征 (is_session_open) 存在 | ✅ PASS |
| 所有特征 dtype 为 float32 | ✅ PASS |
| 窗口定位字段 (x_start, x_end) | ✅ PASS |
| **总体状态** | **✅ PASS** |

### 窗口定位字段

```python
window_start_field = 'x_start'
window_end_field = 'x_end'
```

---

## 性能统计

### 多线程加速

- **Workers**: 8
- **总品种**: 24
- **处理时间**: ~30 秒
- **加速比**: ~6-8x (相比单线程)

### 文件大小

- **总大小**: ~550 MB (24 个品种)
- **平均大小**: ~23 MB/品种
- **最大**: SHFE.AU (32.5 MB)
- **最小**: CFFEX.T (14.2 MB)

---

## 生成的文件

### 数据文件

```
data/processed/m1_f8/
├── CFFEX.T/
│   └── bars.parquet (8 features)
├── CZCE.CF/
│   └── bars.parquet (8 features)
├── DCE.JM/
│   └── bars.parquet (8 features)
├── SHFE.AG/
│   └── bars.parquet (8 features)
└── ... (24 个品种)
```

### 报告文件

- `reports/m1_t5_1_feature_patch.json` - 机器可读报告
- `reports/m1_t5_1_feature_patch.md` - 人类可读报告
- `reports/M1_T5_1_SUMMARY.md` - 本文件

---

## 验收标准达成

### ✅ 必须项

1. ✅ **任意抽 3 个 symbol，bars.parquet 明确包含 8 个特征列**
   - DCE.JM, SHFE.AG, CZCE.FG 验证通过
   - 所有特征 dtype 为 float32

2. ✅ **contract check PASS (含第 8 维)**
   - has_8th_feature = true
   - is_session_open 存在且 dtype 正确

3. ✅ **产出机器可读 + 人类可读报告**
   - JSON: `m1_t5_1_feature_patch.json`
   - Markdown: `m1_t5_1_feature_patch.md`

4. ✅ **全量重建成功**
   - 24/24 品种成功
   - 输出到新目录 `m1_f8`

---

## 下一步

### M1-T5.2: 重建 Sample Index

由于 canonical bars 已更新（新增第 8 个特征），需要重建 sample index：

```bash
python src/alphatrade/scripts/build_m1_sample_index.py \
  --universe configs/universe/m1_selected.yaml \
  --input-dir data/processed/m1_f8 \
  --output-dir data/processed/m1_f8 \
  --num-workers 8
```

### M1-T5.3: Smoke Train

使用新的 8 维数据进行 smoke train (200-500 steps)：

```bash
python src/alphatrade/scripts/train_m1_alphatrade_smoke.py \
  --config src/alphatrade/configs/dataset/m1.yaml \
  --max-steps 500
```

---

## 总结

✅ **M1-T5.1 任务完成**

**核心成果**:
- 24 个品种全部成功重建为 8 维特征
- 统一的特征定义模块（单一来源）
- 多线程并发处理（8 workers）
- Contract check 全部通过

**关键改进**:
- 特征维度: 7 → 8
- 新增 `is_session_open` (float32)
- 输出到新目录 `m1_f8`（避免脏状态）
- 原子写入（避免中途失败）

**数据质量**:
- 所有品种 Feature Dim = 8 ✅
- 所有特征 dtype = float32 ✅
- Schema 验证通过 ✅

**可以继续 M1-T5.2**: 重建 sample index
