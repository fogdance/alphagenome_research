# M1-T4 最终总结报告

生成时间: 2026-03-01

## 任务目标

基于 M1-T2 和 M1-T3 的实际数据质量，筛选出最终的高质量品种 universe。

---

## ✅ 完成内容

### 筛选脚本

**脚本**: `src/alphatrade/scripts/select_m1_universe.py`

**功能**:
- 解析 M1-T2 continuous bars 报告
- 解析 M1-T3 sample index 报告
- 应用质量过滤器
- 生成最终 universe YAML
- 生成选择报告

---

## 筛选标准

### 质量阈值

1. **Coverage >= 95%** (来自 M1-T2)
   - 确保 continuous bars 的交易日覆盖率足够高
   
2. **Train samples >= 5,000** (来自 M1-T3)
   - 确保训练集有足够的样本
   
3. **Val samples >= 500** (来自 M1-T3)
   - 确保验证集有足够的样本
   
4. **Test samples >= 1,000** (来自 M1-T3)
   - 确保测试集有足够的样本

---

## 筛选结果

### 总览

- **候选品种**: 48
- **选中品种**: 24
- **拒绝品种**: 24
- **选中率**: 50.0%

### 选中品种统计

**总样本数**:
- Train: 387,673
- Val: 74,459
- Test: 147,523
- Total: 609,655

**平均质量**:
- Coverage: 98.3%
- Train samples: 16,153
- Val samples: 3,102
- Test samples: 6,147

### 交易所分布

| 交易所 | 品种数 | 占比 |
|--------|--------|------|
| SHFE | 8 | 33.3% |
| DCE | 7 | 29.2% |
| CZCE | 7 | 29.2% |
| INE | 1 | 4.2% |
| CFFEX | 1 | 4.2% |

---

## 选中品种列表

### 高样本量品种 (>40,000 samples)

这些品种有非常丰富的训练数据：

| Symbol | Total Samples | Train | Val | Test |
|--------|---------------|-------|-----|------|
| SHFE.AG | 72,240 | 44,334 | 9,383 | 18,523 |
| SHFE.AU | 71,616 | 43,751 | 9,349 | 18,516 |
| INE.SC | 69,715 | 41,924 | 9,340 | 18,451 |
| SHFE.NI | 45,784 | 28,114 | 5,936 | 11,734 |
| SHFE.CU | 45,662 | 28,072 | 5,927 | 11,663 |
| SHFE.ZN | 45,636 | 28,121 | 5,892 | 11,623 |
| SHFE.AL | 44,637 | 27,232 | 5,836 | 11,569 |
| SHFE.SN | 43,310 | 25,662 | 5,928 | 11,720 |
| SHFE.PB | 41,268 | 25,830 | 4,699 | 10,739 |

### 中等样本量品种 (8,000-12,000 samples)

| Symbol | Total Samples | Train | Val | Test |
|--------|---------------|-------|-----|------|
| CZCE.MA | 10,948 | 7,358 | 1,204 | 2,386 |
| CZCE.FG | 10,071 | 7,905 | 726 | 1,440 |
| CZCE.OI | 9,055 | 6,886 | 726 | 1,443 |
| CZCE.CF | 8,805 | 6,639 | 726 | 1,440 |
| CZCE.SR | 8,738 | 6,569 | 726 | 1,443 |
| CZCE.RM | 8,698 | 6,529 | 726 | 1,443 |
| CZCE.TA | 8,591 | 6,423 | 726 | 1,442 |
| CFFEX.T | 8,437 | 5,965 | 864 | 1,608 |
| DCE.Y | 8,405 | 6,233 | 726 | 1,446 |
| DCE.A | 8,136 | 5,541 | 1,010 | 1,585 |
| DCE.J | 8,079 | 5,643 | 911 | 1,525 |
| DCE.M | 8,032 | 5,860 | 726 | 1,446 |
| DCE.P | 8,018 | 5,652 | 920 | 1,446 |

### 标准样本量品种 (7,000-8,000 samples)

| Symbol | Total Samples | Train | Val | Test |
|--------|---------------|-------|-----|------|
| DCE.JM | 7,898 | 5,726 | 726 | 1,446 |
| DCE.I | 7,876 | 5,704 | 726 | 1,446 |

---

## 拒绝品种分析

### 拒绝原因分类

**Coverage 不足** (2 个):
- CZCE.RS: 74.4% (数据缺失严重)
- SHFE.FU: 92.3%

**Train samples 不足** (24 个):
- 大部分拒绝品种都是因为 train samples < 5,000
- 这些品种的 segment 较少或 stride 采样后样本不足

**多重原因** (部分品种):
- CFFEX.IC, CFFEX.IF, CFFEX.IH: 股指期货，样本量严重不足
- CZCE.RS: Coverage 和样本量都不足

---

## 生成的文件

### Universe 配置

**文件**: `configs/universe/m1_selected.yaml`

```yaml
version: m1_selected
description: M1 final selected universe based on data quality
candidates: [24 symbols]
selection_criteria:
  min_coverage: 95.0
  min_train_samples: 5000
  min_val_samples: 500
  min_test_samples: 1000
```

### 选择报告

**文件**: `reports/m1_t4_universe_selection.md`

包含:
- 选中品种详情表
- 拒绝品种详情表
- 统计摘要
- 交易所分布

---

## 数据质量验证

### 选中品种质量

✅ **Coverage 优秀**:
- 所有品种 >= 95%
- 平均 98.3%
- 数据完整性高

✅ **样本量充足**:
- Train: 387,673 (平均 16,153/品种)
- Val: 74,459 (平均 3,102/品种)
- Test: 147,523 (平均 6,147/品种)
- 足够支持多品种训练

✅ **交易所分布均衡**:
- SHFE: 8 个 (33.3%)
- DCE: 7 个 (29.2%)
- CZCE: 7 个 (29.2%)
- 覆盖主要交易所

✅ **品种多样性**:
- 贵金属: SHFE.AG, SHFE.AU
- 有色金属: SHFE.AL, SHFE.CU, SHFE.NI, SHFE.ZN, SHFE.PB, SHFE.SN
- 能源化工: INE.SC
- 农产品: CZCE.MA, CZCE.SR, CZCE.RM, CZCE.OI, CZCE.TA, CZCE.CF, DCE.A, DCE.M, DCE.Y, DCE.P
- 工业品: DCE.I, DCE.J, DCE.JM, CZCE.FG
- 金融: CFFEX.T

---

## 使用示例

### 查看选中品种

```bash
cat configs/universe/m1_selected.yaml
```

### 查看选择报告

```bash
cat reports/m1_t4_universe_selection.md
```

### 验证数据完整性

```bash
# 检查所有选中品种的数据文件
for symbol in $(grep "^-" configs/universe/m1_selected.yaml | cut -d' ' -f2); do
  echo "Checking $symbol..."
  ls -lh data/processed/m1/$symbol/
done
```

---

## 下一步

### M1-T5: 多品种训练

**使用 24 个选中品种进行训练**:

**输入**:
- Universe: `configs/universe/m1_selected.yaml`
- 数据: `data/processed/m1/{csymbol}/`
  - bars.parquet
  - index_train.parquet
  - index_val.parquet
  - index_test.parquet

**任务**:
1. 创建多品种 dataloader
2. 复用 M0 的 baseline 模型
3. 训练并评估
4. 生成训练报告

**预期输出**:
- 训练好的模型
- `reports/m1_train_metrics.json`
- `reports/m1_train_run.md`

---

## 总结

✅ **M1-T4 任务完成**

**核心成果**:
- 从 48 个候选品种中筛选出 24 个高质量品种
- 选中率 50%，质量标准严格
- 总计 609,655 个训练样本
- 交易所和品种分布均衡

**关键指标**:
- 平均 Coverage: 98.3%
- 平均 Train samples: 16,153
- 平均 Val samples: 3,102
- 平均 Test samples: 6,147

**可以继续 M1-T5**: 数据质量优秀，准备就绪
