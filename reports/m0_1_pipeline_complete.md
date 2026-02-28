# M0.1 数据处理与训练流程完成总结

生成时间: 2026-02-28

## 概览

成功实现了从 Archive 到训练的完整流程，包括数据加工、样本索引生成和 baseline 训练。

## 代码组织

### Archive 管理（juejin 工程）
位置: `/home/v/Documents/work/juejin/ed27929a-30a5-11f0-a3a6-366f24a5ee2d`

- 负责下载期货数据并保存为 archive
- 提供 archive 检查和验证工具
- 生成 manifest 文件

### 数据加工与训练（alphatrade 工程）
位置: `/home/v/Documents/work/1_open_source/alphagenome_research/src/alphatrade`

**数据处理模块** (`data_pipeline/`)
- `archive_reader.py` - 从 archive 读取数据

**脚本** (`scripts/`)
- `build_canonical_bars.py` - 生成标准化 bars
- `build_sample_index.py` - 生成训练样本索引
- `train_m0_1_baseline.py` - Baseline 训练

**配置** (`configs/`)
- `m0_1_dataset.yaml` - M0.1 数据集配置

## 实现的功能

### T2: Archive 读取器 ✅
- 基于 manifest 驱动的 parquet 文件读取
- 自动拼接多月数据
- EOB 去重（保留 last）
- 数据校验（单调性、缺失值、数据类型）

### T3: Canonical Bars 生成 ✅
- Gap-based 分段（segment_id，gap_seconds=1800）
- 基础特征计算：ret_1m, hl_range, co_change, vol_log1p, pos_log1p
- 分钟相位特征：minute_sin, minute_cos, is_open
- 数据类型优化（float32, int32）

### T4: 训练样本索引生成 ✅
- 滑动窗口采样（lookback=60, stride=5）
- 多 horizon 标签计算（1, 5, 20, 60 分钟）
- 时间切分（train: 2018-2024, val: 2024-2025, test: 2025-2026）
- 同 segment 约束（避免跨 gap 采样）

### T5: Baseline 训练 ✅
- 简单 MLP 模型（flatten + 2 hidden layers）
- 多任务回归（4 个 horizon）
- 在线窗口切片（不提前物化）
- 生成标准化 metrics JSON 和 markdown 报告

## 数据统计

### DCE.JM (焦煤)
- **原始数据**: 663,184 bars (2017-12 至 2025-12)
- **Segments**: 5,699 个（gap=1800s）
- **训练样本**:
  - Train: 6,333 samples
  - Val: 726 samples
  - Test: 720 samples

### SHFE.RB (螺纹钢)
- **原始数据**: 654,596 bars (2017-12 至 2025-12)
- **Segments**: 5,699 个（gap=1800s）
- **训练样本**:
  - Train: 4,537 samples
  - Val: 726 samples
  - Test: 720 samples

### 合并数据集
- **Train total**: 10,870 samples
- **Val total**: 1,452 samples
- **Test total**: 1,440 samples

## 训练结果（Smoke Test）

**配置**:
- Max steps: 200
- Batch size: 128
- Learning rate: 0.001
- Device: CUDA

**Loss**:
- Train: 0.000155
- Val: 0.000134

✅ 训练成功收敛，验证集 loss 低于训练集（正常现象，因为样本量小）

## 关键配置参数

```yaml
canonical:
  gap_seconds: 1800  # 30分钟，创建更长的 segment
  eps: 1.0e-12
  cast_float: float32

sampling:
  lookback: 60  # 1小时历史窗口
  stride: 5     # 每5分钟采样一次
  horizons: [1, 5, 20, 60]  # 预测未来 1/5/20/60 分钟
  require_same_segment: true

time_split:
  train: 2018-01-01 to 2024-01-01
  val:   2024-01-01 to 2025-01-01
  test:  2025-01-01 to 2026-01-01
```

## 生成的报告

1. **Canonical Profile** (`reports/m0_1_canonical_profile.md`)
   - Segment 统计（数量、长度分布）
   - 特征统计（均值、标准差、分位数）

2. **Index Stats** (`reports/m0_1_index_stats.md`)
   - 样本数量分布
   - Label 分布统计（每个 horizon）
   - 跨 segment 验证

3. **Training Metrics** (`reports/m0_1_train_metrics.json`)
   - 机器可读的训练指标
   - 包含 run_id, git_sha, 数据集信息, loss 等

4. **Training Run** (`reports/m0_1_train_run.md`)
   - 人类可读的训练报告
   - 运行命令、配置摘要、训练结果

## 数据质量验证

✅ **Archive 读取**
- 无缺失值
- EOB 单调递增
- 重复 EOB 已去重

✅ **Canonical Bars**
- Segment 分段正常（5,699 segments）
- 特征计算正确（ret_1m 均值≈0）
- 数据类型优化完成

✅ **Sample Index**
- 所有样本在同一 segment 内
- Train/Val/Test 无时间泄漏
- Label 分布合理（均值≈0，标准差随 horizon 增大）

✅ **Training**
- Dataloader 正常工作（在线切片）
- 模型收敛
- Metrics JSON schema 符合要求

## 使用示例

```bash
# 1. 生成 canonical bars
python src/alphatrade/scripts/build_canonical_bars.py --symbol DCE.JM
python src/alphatrade/scripts/build_canonical_bars.py --symbol SHFE.RB

# 2. 生成训练样本索引
python src/alphatrade/scripts/build_sample_index.py --symbol DCE.JM
python src/alphatrade/scripts/build_sample_index.py --symbol SHFE.RB

# 3. 训练 baseline 模型
python src/alphatrade/scripts/train_m0_1_baseline.py \
  --max-steps 500 \
  --batch-size 256 \
  --num-workers 4

# 快速 smoke test
python src/alphatrade/scripts/train_m0_1_baseline.py \
  --max-steps 200 \
  --batch-size 128 \
  --limit-train-samples 1000 \
  --limit-val-samples 200
```

## 下一步

数据处理和训练流程已完成，可以进行：
1. 特征工程优化
2. 模型架构改进（LSTM, Transformer）
3. 超参数调优
4. 扩展到更多品种
5. 回测和评估

## 文件结构

```
alphagenome_research/
├── src/alphatrade/
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   └── archive_reader.py
│   ├── scripts/
│   │   ├── build_canonical_bars.py
│   │   ├── build_sample_index.py
│   │   └── train_m0_1_baseline.py
│   └── configs/
│       └── m0_1_dataset.yaml
├── data/processed/m0_1/
│   ├── DCE.JM/
│   │   ├── bars.parquet (21MB)
│   │   ├── index_train.parquet
│   │   ├── index_val.parquet
│   │   └── index_test.parquet
│   └── SHFE.RB/
│       ├── bars.parquet (22MB)
│       ├── index_train.parquet
│       ├── index_val.parquet
│       └── index_test.parquet
└── reports/
    ├── m0_1_canonical_profile.md
    ├── m0_1_index_stats.md
    ├── m0_1_train_metrics.json
    └── m0_1_train_run.md
```
