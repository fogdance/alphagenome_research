# M1-T3 最终总结报告

生成时间: 2026-03-01

## 任务目标

基于 raw_continuous_bars 生成 canonical bars 和 sample index，为训练做准备。

---

## ✅ 完成内容

### 1. Canonical Bars 生成

**脚本**: `src/alphatrade/scripts/build_m1_canonical_bars.py`

**功能**:
- 加载 raw_continuous_bars.parquet
- 计算 gap-based segmentation (gap_seconds=1800)
- 计算基础特征 (ret_1m, hl_range, co_change, vol_log1p, pos_log1p)
- 计算 minute phase 特征 (minute_sin, minute_cos)
- 输出 bars.parquet

**结果**: 48/48 成功 (100%)

### 2. Sample Index 生成

**脚本**: `src/alphatrade/scripts/build_m1_sample_index.py`

**功能**:
- 加载 canonical bars
- 生成 sample indices (lookback=60, stride=5)
- 计算多 horizon 标签 (h1, h5, h20, h60)
- 时间分割 (train/val/test)
- 输出 index_{train,val,test}.parquet

**结果**: 48/48 成功 (100%)

---

## 数据统计

### Canonical Bars

**总览**:
- 成功品种: 48/48 (100%)
- 平均 segments: ~5,000-6,000 per symbol
- 平均文件大小: ~15-20 MB per symbol

**示例品种**:
- DCE.JM: 661,417 rows, 5,685 segments, 20.4 MB
- SHFE.AG: 981,648 rows, segments, 17.9 MB
- SHFE.RB: 652,991 rows, segments, 12.3 MB

### Sample Index

**总览**:
- 成功品种: 48/48 (100%)
- **Total train samples**: 461,849
- **Total val samples**: 90,396
- **Total test samples**: 178,960
- **Total samples**: 731,205

**时间分割**:
- Train: 2018-01-01 to 2023-01-01 (5 years)
- Val: 2023-01-01 to 2024-01-01 (1 year)
- Test: 2024-01-01 to 2026-01-01 (2 years)

**示例品种**:
- DCE.JM: 5,726 train, 726 val, 1,446 test
- 平均每品种: ~9,621 train, ~1,883 val, ~3,728 test

---

## 生成的文件

### 每个品种的输出

```
data/processed/m1/{csymbol}/
  ├── raw_continuous_bars.parquet  (M1-T2 输出)
  ├── bars.parquet                 (M1-T3 canonical)
  ├── index_train.parquet          (M1-T3 train index)
  ├── index_val.parquet            (M1-T3 val index)
  └── index_test.parquet           (M1-T3 test index)
```

### 报告文件

- `reports/m1_t3_canonical_profile.md` - Canonical bars 特征统计
- `reports/m1_t3_sample_index.md` - Sample index 统计
- `reports/M1_T3_SUMMARY.md` - 本文件

---

## 数据质量验证

### Canonical Bars 质量

✅ **Segmentation 正确**:
- Gap threshold: 1800 seconds (30 minutes)
- 平均 segment 长度: 100-120 bars
- Segment 数量合理: 5,000-6,000 per symbol

✅ **特征计算正确**:
- ret_1m: 1-minute return
- hl_range: (high - low) / close
- co_change: (close - open) / open
- vol_log1p: log(1 + volume)
- pos_log1p: log(1 + position)
- minute_sin/cos: minute phase encoding

✅ **数据类型优化**:
- Float columns: float32
- Int columns: int32
- 节省内存和加载时间

### Sample Index 质量

✅ **Lookback window 正确**:
- Lookback: 60 bars
- Stride: 5 bars
- 确保 window 和 target 在同一 segment

✅ **Label 计算正确**:
- y_h1: log(close[t+1] / close[t])
- y_h5: log(close[t+5] / close[t])
- y_h20: log(close[t+20] / close[t])
- y_h60: log(close[t+60] / close[t])

✅ **时间分割正确**:
- 无数据泄漏
- Train/val/test 时间不重叠
- 符合时间序列分割原则

---

## 使用示例

### 生成 Canonical Bars

```bash
# 单品种
python src/alphatrade/scripts/build_m1_canonical_bars.py --csymbol DCE.JM

# 批量
python src/alphatrade/scripts/build_m1_canonical_bars.py \
  --universe configs/universe/m1_candidates.yaml
```

### 生成 Sample Index

```bash
# 单品种
python src/alphatrade/scripts/build_m1_sample_index.py --csymbol DCE.JM

# 批量
python src/alphatrade/scripts/build_m1_sample_index.py \
  --universe configs/universe/m1_candidates.yaml
```

### 检查输出

```bash
# 查看 canonical bars
python -c "import pandas as pd; df = pd.read_parquet('data/processed/m1/DCE.JM/bars.parquet'); print(df.info())"

# 查看 sample index
python -c "import pandas as pd; df = pd.read_parquet('data/processed/m1/DCE.JM/index_train.parquet'); print(df.head())"
```

---

## 性能优化建议

### 当前性能

- Canonical bars: ~1-2 秒/品种
- Sample index: ~5-10 秒/品种
- 总耗时: ~10-15 分钟 (48 品种串行)

### 优化方案

**并行处理**:
- 使用 multiprocessing 并行处理多个品种
- 预期加速: 4-8x (取决于 CPU 核心数)
- 总耗时可降至: ~2-4 分钟

**实现方式**:
```python
from multiprocessing import Pool

def process_symbol_wrapper(args):
    return process_symbol(*args)

with Pool(processes=8) as pool:
    results = pool.map(process_symbol_wrapper, symbol_args)
```

---

## 下一步

### M1-T4: Universe 最终选择

**基于 T2 + T3 的实际质量筛选品种**:

**筛选标准**:
1. ✅ Coverage >= 95% (M1-T2)
2. ✅ Canonical 生成成功 (M1-T3)
3. ✅ Sample 数量 >= 阈值 (M1-T3)
   - Train samples >= 5,000
   - Val samples >= 500
   - Test samples >= 1,000

**预期结果**:
- 高质量品种: ~43 个 (基于 M1-T2 的 43 个高质量品种)
- 生成 `configs/universe/m1_selected.yaml`
- 生成 `reports/m1_t4_universe_selection.md`

### M1-T5: 多品种训练

**使用最终选择的品种进行训练**:
- 复用 M0 的 baseline 模型
- 支持多品种 dataloader
- 输出训练报告

---

## 总结

✅ **M1-T3 任务完成**

**核心成果**:
- 48 个品种全部成功生成 canonical bars 和 sample index
- 总计 731,205 个训练样本 (train: 461,849, val: 90,396, test: 178,960)
- 数据质量验证通过

**关键改进**:
- 基于 raw_continuous_bars 生成，确保数据正确性
- 特征计算完整，包含基础特征和 minute phase
- 时间分割合理，符合时间序列原则

**可以继续 M1-T4**: 现有数据足够进行 universe 最终选择
