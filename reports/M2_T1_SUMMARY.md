# M2-T1 总结报告：Dataset/Dataloader 验证

生成时间: 2026-03-01

## 任务目标

验证 Dataset/Dataloader 的数据质量，确保：
- 样本 shape 正确
- 无 NaN/Inf
- 不跨 segment
- 多品种采样正常

---

## ✅ 完成内容

### 1. Dataloader 检查脚本

**文件**: `src/alphatrade/scripts/check_m2_dataloader.py`

**功能**:
- ✅ 随机抽样验证（10 个样本）
- ✅ 按品种抽样验证（每个品种 3 个样本）
- ✅ Shape 检查（X: [60, 8], Y: [4]）
- ✅ NaN/Inf 检查
- ✅ Segment 一致性检查
- ✅ 产出 JSON + MD 报告

---

## 验收结果

### 测试命令

```bash
python src/alphatrade/scripts/check_m2_dataloader.py \
  --config configs/dataset/m2.yaml \
  --smoke \
  --num-samples 10
```

### 测试结果

✅ **19/19 样本全部通过 (100%)**

| 指标 | 值 |
|------|-----|
| Total samples | 57,418 |
| Symbols | 3 (DCE.JM, SHFE.AG, CZCE.MA) |
| Random checks | 10 |
| Per-symbol checks | 9 (3 per symbol) |
| Valid checks | 19/19 (100%) |
| Invalid checks | 0 |

### Symbol Sample Counts

| Symbol | Samples | 占比 |
|--------|---------|------|
| SHFE.AG | 44,334 | 77.2% |
| CZCE.MA | 7,358 | 12.8% |
| DCE.JM | 5,726 | 10.0% |

---

## 验证项目

### 1. Shape 验证 ✅

**期望**:
- X: `(60, 8)` - [Lookback, Features]
- Y: `(4,)` - [Horizons]

**结果**: ✅ 所有样本 shape 正确

### 2. NaN/Inf 检查 ✅

**检查项**:
- X 是否包含 NaN
- X 是否包含 Inf
- Y 是否包含 NaN
- Y 是否包含 Inf

**结果**: ✅ 所有样本无 NaN/Inf

### 3. Segment 一致性 ✅

**验证逻辑**:
```python
# 窗口内所有 bar 的 segment_id 必须相同
segment_ids = bars['segment_id'][x_start:x_end+1]
all_same = np.all(segment_ids == expected_segment_id)
```

**结果**: ✅ 所有样本 segment 一致，无跨 segment

### 4. 多品种采样 ✅

**采样策略**: 当前使用简单的 shuffle（所有品种混合）

**结果**: 
- 3 个品种都能正常采样
- 样本分布符合预期（SHFE.AG 占比最高）

---

## 抽样验证详情

### Random Samples (10 个)

所有样本验证通过：
- DCE.JM: 1 个样本 ✅
- CZCE.MA: 2 个样本 ✅
- SHFE.AG: 7 个样本 ✅

### Per-Symbol Samples (9 个)

每个品种 3 个样本，全部通过：
- DCE.JM: 3/3 ✅
- SHFE.AG: 3/3 ✅
- CZCE.MA: 3/3 ✅

**示例**:
```
DCE.JM:
  Sample 1066: ✅ shape=(60, 8), segment=363
  Sample 2762: ✅ shape=(60, 8), segment=1008
  Sample 4241: ✅ shape=(60, 8), segment=2161

SHFE.AG:
  Sample 14068: ✅ shape=(60, 8), segment=655
  Sample 45661: ✅ shape=(60, 8), segment=3203
  Sample 30187: ✅ shape=(60, 8), segment=2012

CZCE.MA:
  Sample 53084: ✅ shape=(60, 8), segment=966
  Sample 54895: ✅ shape=(60, 8), segment=1707
  Sample 53586: ✅ shape=(60, 8), segment=1137
```

---

## 数据质量确认

### ✅ 特征维度

- **Feature dim**: 8 (固定)
- **Feature cols**: 
  1. ret_1m
  2. hl_range
  3. co_change
  4. vol_log1p
  5. pos_log1p
  6. minute_sin
  7. minute_cos
  8. is_session_open

### ✅ 标签维度

- **Horizons**: [1, 5, 20, 60]
- **Label type**: log returns

### ✅ 数据约束

- **不跨 segment**: ✅ 验证通过
- **不跨合约**: ✅ 由 segment 机制保证
- **无 NaN/Inf**: ✅ 验证通过

---

## 生成的文件

### 报告文件

- `reports/m2_t1_dataloader_check.json` - 机器可读报告
- `reports/m2_t1_dataloader_check.md` - 人类可读报告
- `reports/M2_T1_SUMMARY.md` - 本文件

### 报告内容

**JSON 报告包含**:
- Config 信息
- Dataset 统计
- Validation 结果
- Random samples 详情
- Symbol samples 详情

**Markdown 报告包含**:
- 配置摘要
- Dataset 统计表
- 验证结果表
- 错误详情（如有）

---

## 采样策略说明

### 当前策略：Simple Shuffle

**实现**:
```python
# 所有品种的 index 混合在一起
self.indices = []
for symbol in symbols:
    for row in index_df.iterrows():
        self.indices.append({...})

# DataLoader 使用 shuffle=True
train_loader = DataLoader(dataset, shuffle=True, ...)
```

**特点**:
- ✅ 简单直接
- ✅ 每个 batch 可能包含多个品种
- ⚠️ 样本多的品种（SHFE.AG）会占主导

### 可选策略（未实现）

**方案 A: Weighted Sampling**
```python
# 按品种样本数的平方根加权
weights = sqrt(sample_counts)
sampler = WeightedRandomSampler(weights, ...)
```

**方案 B: Round-Robin**
```python
# 轮流从每个品种采样
for epoch:
    for symbol in symbols:
        sample from symbol
```

**建议**: 当前 simple shuffle 已足够，后续可根据训练效果调整

---

## 下一步：M2-T2

### 任务

训练目标与 Loss 实现：
- Pinball loss（已实现，需验证）
- Quantile crossing penalty（已实现，需验证）
- By-horizon loss 统计（需实现）

### 验收

- 200 steps 内 train loss 有下降趋势
- 不发散/NaN
- Crossing penalty 生效

### 产物

- `reports/m2_t2_loss_sanity.md`

---

## 总结

✅ **M2-T1 验收通过**

**核心成果**:
- Dataloader 检查脚本完成
- 19/19 样本验证通过 (100%)
- 所有验证项目通过

**验证项目**:
- ✅ Shape 正确 (X: [60, 8], Y: [4])
- ✅ 无 NaN/Inf
- ✅ Segment 一致性
- ✅ 多品种采样正常

**数据质量**:
- 57,418 训练样本
- 3 个品种
- 特征维度 8
- 标签维度 4

**可以继续 M2-T2**: Dataloader 验证完成
