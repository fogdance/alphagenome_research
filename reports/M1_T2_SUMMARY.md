# M1-T2 最终总结报告

生成时间: 2026-03-01

## 任务目标

构建 continuous bars：通过 `fut_continuous_map_v2` 拼接 real 合约，解决 archive 中现成 continuous symbol 的 stale 和数据不完整问题。

---

## ✅ 完成内容

### 1. 核心模块

**数据库连接器**: `src/alphatrade/data_pipeline/db_connector.py`
- 连接 MySQL 数据库
- 读取 `fut_continuous_map_v2` 映射表
- 支持日期范围过滤

**构建脚本**: `src/alphatrade/scripts/build_continuous_bars.py`
- 单品种模式: `--csymbol DCE.JM`
- 批量模式: `--universe configs/universe/m1_candidates.yaml`
- 输出: `data/processed/m1/{csymbol}/raw_continuous_bars.parquet`
- 审计报告: `reports/m1_t2_continuous_build.md`

### 2. 数据源改进

**问题**: 初始 DB 数据不完整（只有最近 43 天）  
**解决**: 使用掘金 API 批量获取完整历史映射并写入 DB  
**结果**: 所有品种都有完整历史数据（2014-2016 年开始到 2026-02-27）

---

## 执行结果

### 批量处理 (48 个候选品种)

**命令**:
```bash
python src/alphatrade/scripts/build_continuous_bars.py \
  --universe configs/universe/m1_candidates.yaml \
  --start 2018-01-01 \
  --end 2026-02-13
```

**结果**:
- ✅ 成功: 48/48 (100%)
- ✅ 失败: 0

---

## 数据质量分析

### 按 Coverage 分类

| Coverage | 品种数 | 占比 |
|----------|--------|------|
| >= 98% | 45 | 93.8% |
| >= 95% | 46 | 95.8% |
| >= 90% | 47 | 97.9% |
| < 90% | 1 | 2.1% |

### 按 Mapping Days 分类

| Mapping Days | 品种数 | 占比 |
|--------------|--------|------|
| >= 1900 天 (~7.6 年) | 45 | 93.8% |
| >= 1800 天 (~7.2 年) | 46 | 95.8% |
| >= 1700 天 (~6.8 年) | 48 | 100% |

### 高质量品种 (43 个)

**筛选标准**: mapping_days >= 1900 AND coverage >= 95%

**交易所分布**:
- SHFE: 12 个
- DCE: 14 个
- CZCE: 12 个
- CFFEX: 4 个
- INE: 1 个

**示例品种**:
- SHFE.AG: 1972 days, 98.3%, 981,648 bars
- DCE.JM: 1972 days, 98.3%, 661,417 bars
- SHFE.RB: 1972 days, 98.3%, 652,991 bars
- CZCE.SR: 1972 days, 98.3%, 666,257 bars
- CFFEX.IC: 1972 days, 98.2%, 464,974 bars

### 低质量品种 (2 个)

| Symbol | Mapping Days | Coverage | 原因 |
|--------|--------------|----------|------|
| CZCE.RS | 1972 | 74.4% | 数据缺失严重 |
| SHFE.FU | 1959 | 92.3% | Coverage 略低 |

---

## Stale 问题分析

### 问题现状

**Archive 数据限制**:
- 所有品种的 max_eob: 2025-12-24 到 2025-12-26
- 距离目标日期 2026-02-13: 落后 49-51 天

**原因**:
- Archive 数据源本身只到 2025-12-26
- 即使 DB mapping 最新（到 2026-02-27），也无法超越 archive 限制

### 结论

✅ **方法正确**: 通过 DB mapping 拼接 real 合约的方法是正确的  
⚠️ **数据源限制**: Stale 问题是 archive 数据源的限制，不是脚本问题  
✅ **相比直接用 archive continuous**:
- 映射逻辑正确（基于每日主力合约切换规则）
- 历史数据完整（1900+ 天）
- Coverage 高（98%+）

---

## 生成的文件

### 每个品种的输出

```
data/processed/m1/{csymbol}/
  └── raw_continuous_bars.parquet
```

**示例**:
- `data/processed/m1/SHFE.AG/raw_continuous_bars.parquet` (17.9 MB, 981,648 rows)
- `data/processed/m1/DCE.JM/raw_continuous_bars.parquet` (12.3 MB, 661,417 rows)
- `data/processed/m1/SHFE.RB/raw_continuous_bars.parquet` (12.3 MB, 652,991 rows)

### 审计报告

- `reports/m1_t2_continuous_build.md` - 完整的构建报告
- `reports/M1_T2_FINAL_SUMMARY.md` - 本文件

---

## 验收标准

### 必须满足 ✅

1. ✅ **脚本功能完整**
   - 单品种模式 ✅
   - 批量模式 ✅
   - DB mapping 读取 ✅
   - Archive 数据加载 ✅
   - 拼接逻辑正确 ✅

2. ✅ **输出文件正确**
   - `raw_continuous_bars.parquet` 生成 ✅
   - 包含所有必要列 ✅
   - 按 eob 排序 ✅

3. ✅ **审计报告完整**
   - 映射覆盖率 ✅
   - 缺失 trading_date 数 ✅
   - Real 合约切换次数 ✅
   - 缺失 bars 的天数 ✅
   - 最终 eob 范围 ✅

4. ✅ **Stale 分析明确**
   - 报告中明确指出每个品种的 stale 状态 ✅
   - 解释为什么未修复（Archive 数据限制）✅

### 数据质量 ✅

1. ✅ **历史数据完整**
   - 45 个品种有 1900+ 天历史
   - 平均覆盖 ~8 年

2. ✅ **Coverage 高**
   - 45 个品种 coverage >= 98%
   - 平均 coverage 97.8%

3. ✅ **合约切换正确**
   - 基于 DB mapping 的主力合约切换规则
   - 切换次数合理（24-96 次/品种）

---

## 使用示例

### 单品种
```bash
python src/alphatrade/scripts/build_continuous_bars.py \
  --csymbol DCE.JM \
  --start 2018-01-01 \
  --end 2026-02-13
```

### 批量处理
```bash
python src/alphatrade/scripts/build_continuous_bars.py \
  --universe configs/universe/m1_candidates.yaml \
  --start 2018-01-01 \
  --end 2026-02-13
```

### 检查输出
```bash
# 查看生成的文件
ls -lh data/processed/m1/*/raw_continuous_bars.parquet

# 查看审计报告
cat reports/m1_t2_continuous_build.md

# 检查单个品种
python -c "import pandas as pd; df = pd.read_parquet('data/processed/m1/DCE.JM/raw_continuous_bars.parquet'); print(df.info())"
```

---

## 下一步

### M1-T3: 生成 Canonical + Index

**输入**: `data/processed/m1/{csymbol}/raw_continuous_bars.parquet`  
**输出**:
- `data/processed/m1/{csymbol}/bars.parquet` (canonical)
- `data/processed/m1/{csymbol}/index_train.parquet`
- `data/processed/m1/{csymbol}/index_val.parquet`
- `data/processed/m1/{csymbol}/index_test.parquet`

**复用 M0 流程**:
- 使用现有的 `build_canonical_bars.py`
- 使用现有的 `build_sample_index.py`
- 只需修改输入路径

### M1-T4: Universe 最终选择

**基于 T2 + T3 的实际质量**:
- 过滤 coverage < 95% 的品种
- 过滤 canonical 生成失败的品种
- 过滤 sample 数量 < 阈值的品种
- 生成 `configs/universe/m1_selected.yaml`

### M1-T5: 多品种训练

**使用最终选择的品种**:
- 复用 M0 的 baseline 模型
- 支持多品种 dataloader
- 输出训练报告

---

## 总结

✅ **M1-T2 任务完成**

**核心成果**:
- 48 个品种全部成功构建 continuous bars
- 43 个高质量品种（mapping_days >= 1900, coverage >= 95%）
- 数据完整性和正确性验证通过

**关键改进**:
- 使用 DB mapping 替代 archive 现成 continuous
- 确保主力合约切换逻辑正确
- 历史数据完整（~8 年）

**已知限制**:
- Archive 数据 stale（只到 2025-12-26）
- 这是数据源限制，不影响训练使用

**可以继续 M1-T3**: 现有数据足够进行 canonical + index 生成
