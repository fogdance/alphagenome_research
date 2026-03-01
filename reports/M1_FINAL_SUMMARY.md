# M1 任务最终总结：关键文件和命令

生成时间: 2026-03-01

---

## 📁 关键文件结构

### 1. 配置文件

```
configs/
├── universe/
│   ├── m1_candidates.yaml          # 48 个候选品种
│   └── m1_selected.yaml            # 24 个最终选中品种
└── dataset/
    └── m1.yaml                     # M1 数据集配置
```

### 2. 核心脚本

```
src/alphatrade/
├── data_pipeline/
│   └── feature_schema.py           # 统一特征定义（F=8）
│
└── scripts/
    ├── scan_universe.py            # T0: Universe 扫描
    ├── build_continuous_bars.py    # T2: 构建 continuous bars
    ├── build_m1_canonical_bars_v2.py   # T5.1: 8D canonical bars
    ├── build_m1_sample_index_v2.py     # T5.2: Sample index
    ├── select_m1_universe.py       # T4: Universe 选择
    ├── check_m1_data_contract.py   # T5.0: 数据契约检查
    └── train_m1_smoke.py           # T5.3: Smoke training
```

### 3. 数据文件

```
data/
├── archive/                        # 原始数据（按月）
│   └── {EXCHANGE}.{SYMBOL}/
│       └── {YEAR}-{MONTH}.parquet
│
└── processed/
    └── m1_f8/                      # 最终处理数据（8D）
        └── {SYMBOL}/
            ├── raw_continuous_bars.parquet  # T2 输出
            ├── bars.parquet                 # T5.1 输出（8D）
            ├── index_train.parquet          # T5.2 输出
            ├── index_val.parquet
            └── index_test.parquet
```

### 4. 报告文件

```
reports/
├── m1_t0_universe_inventory.md
├── m1_t2_continuous_build.md
├── M1_T2_FINAL_SUMMARY.md
├── m1_t3_canonical_profile.md
├── m1_t3_sample_index.md
├── M1_T3_SUMMARY.md
├── m1_t4_universe_selection.md
├── M1_T4_SUMMARY.md
├── m1_t5_contract_check.md/json
├── M1_T5_0_SUMMARY.md
├── m1_t5_1_feature_patch.md/json
├── M1_T5_1_SUMMARY.md
├── m1_t5_2_index_rebuild.md/json
├── M1_T5_2_SUMMARY.md
├── m1_train_metrics.json
├── m1_train_run.md
├── M1_T5_3_SUMMARY.md
└── M1_FINAL_SUMMARY.md             # 本文件
```

---

## 🚀 关键命令

### M1-T0: Universe Scanner

```bash
# 扫描 archive 识别连续合约
python src/alphatrade/scripts/scan_universe.py \
  --archive-dir data/archive \
  --output reports/m1_t0_universe_inventory.md

# 生成候选品种配置
# 手动创建 configs/universe/m1_candidates.yaml (48 个品种)
```

### M1-T2: 构建 Continuous Bars

```bash
# 批量构建（48 个候选品种）
python src/alphatrade/scripts/build_continuous_bars.py \
  --universe configs/universe/m1_candidates.yaml \
  --archive-dir data/archive \
  --output-dir data/processed/m1 \
  --num-workers 8
```

### M1-T4: Universe 最终选择

```bash
# 基于 T2+T3 质量筛选最终品种
python src/alphatrade/scripts/select_m1_universe.py

# 输出: configs/universe/m1_selected.yaml (24 个品种)
```

### M1-T5.0: 数据契约核对

```bash
# 检查数据 schema
python src/alphatrade/scripts/check_m1_data_contract.py

# 输出: reports/m1_t5_contract_check.md/json
```

### M1-T5.1: 8 维特征重建

```bash
# 重建 canonical bars（添加第 8 个特征）
python src/alphatrade/scripts/build_m1_canonical_bars_v2.py \
  --universe configs/universe/m1_selected.yaml \
  --input-dir data/processed/m1 \
  --output-dir data/processed/m1_f8 \
  --num-workers 8 \
  --force

# 输出: data/processed/m1_f8/{SYMBOL}/bars.parquet (8D)
```

### M1-T5.2: Sample Index 重建

```bash
# 重建 sample index
python src/alphatrade/scripts/build_m1_sample_index_v2.py \
  --universe configs/universe/m1_selected.yaml \
  --input-dir data/processed/m1_f8 \
  --output-dir data/processed/m1_f8 \
  --num-workers 8 \
  --force

# 输出: data/processed/m1_f8/{SYMBOL}/index_{train,val,test}.parquet
```

### M1-T5.3: Smoke Training

```bash
# 训练验证（500 steps）
python src/alphatrade/scripts/train_m1_smoke.py \
  --universe configs/universe/m1_selected.yaml \
  --processed-dir data/processed/m1_f8 \
  --max-steps 500 \
  --batch-size 256 \
  --lr 1e-4 \
  --seed 42

# 输出: reports/m1_train_metrics.json, reports/m1_train_run.md
```

---

## 🔄 完整流程（一键执行）

### 方案 A: 从头开始（假设已有 archive）

```bash
# 1. 扫描 universe（手动创建 candidates.yaml）
python src/alphatrade/scripts/scan_universe.py \
  --archive-dir data/archive \
  --output reports/m1_t0_universe_inventory.md

# 2. 构建 continuous bars
python src/alphatrade/scripts/build_continuous_bars.py \
  --universe configs/universe/m1_candidates.yaml \
  --archive-dir data/archive \
  --output-dir data/processed/m1 \
  --num-workers 8

# 3. 生成 canonical + index（7D，旧版）
python src/alphatrade/scripts/build_m1_canonical_bars.py \
  --universe configs/universe/m1_candidates.yaml \
  --input-dir data/processed/m1 \
  --output-dir data/processed/m1

python src/alphatrade/scripts/build_m1_sample_index.py \
  --universe configs/universe/m1_candidates.yaml \
  --input-dir data/processed/m1 \
  --output-dir data/processed/m1

# 4. Universe 选择
python src/alphatrade/scripts/select_m1_universe.py

# 5. 数据契约核对
python src/alphatrade/scripts/check_m1_data_contract.py

# 6. 重建 8D 数据
python src/alphatrade/scripts/build_m1_canonical_bars_v2.py \
  --universe configs/universe/m1_selected.yaml \
  --input-dir data/processed/m1 \
  --output-dir data/processed/m1_f8 \
  --num-workers 8 \
  --force

python src/alphatrade/scripts/build_m1_sample_index_v2.py \
  --universe configs/universe/m1_selected.yaml \
  --input-dir data/processed/m1_f8 \
  --output-dir data/processed/m1_f8 \
  --num-workers 8 \
  --force

# 7. Smoke training
python src/alphatrade/scripts/train_m1_smoke.py \
  --universe configs/universe/m1_selected.yaml \
  --processed-dir data/processed/m1_f8 \
  --max-steps 500 \
  --batch-size 256 \
  --lr 1e-4 \
  --seed 42
```

### 方案 B: 快速重建（假设已有 m1_selected.yaml）

```bash
# 直接从 T5.1 开始（假设已有 raw_continuous_bars）
python src/alphatrade/scripts/build_m1_canonical_bars_v2.py \
  --universe configs/universe/m1_selected.yaml \
  --input-dir data/processed/m1 \
  --output-dir data/processed/m1_f8 \
  --num-workers 8 \
  --force

python src/alphatrade/scripts/build_m1_sample_index_v2.py \
  --universe configs/universe/m1_selected.yaml \
  --input-dir data/processed/m1_f8 \
  --output-dir data/processed/m1_f8 \
  --num-workers 8 \
  --force

python src/alphatrade/scripts/train_m1_smoke.py \
  --universe configs/universe/m1_selected.yaml \
  --processed-dir data/processed/m1_f8 \
  --max-steps 500 \
  --batch-size 256 \
  --lr 1e-4 \
  --seed 42
```

---

## 📊 关键数据统计

### Universe

| 阶段 | 品种数 | 说明 |
|------|--------|------|
| T0 扫描 | 88 | CONTINUOUS_LIKELY |
| T0 筛选 | 48 | Score >= 60 |
| T4 最终选择 | 24 | Coverage >= 95%, Samples >= 阈值 |

### 数据量

| 数据集 | Train | Val | Test | Total |
|--------|-------|-----|------|-------|
| 样本数 | 387,673 | 74,459 | 147,523 | 609,655 |

### 特征

| 特征 | 说明 |
|------|------|
| ret_1m | 1-minute return |
| hl_range | (high - low) / close |
| co_change | (close - open) / open |
| vol_log1p | log(1 + volume) |
| pos_log1p | log(1 + position) |
| minute_sin | sin(2π * minute / 1440) |
| minute_cos | cos(2π * minute / 1440) |
| is_session_open | Session open indicator (1.0) |

**Feature Dim**: F = 8

---

## ⚡ 性能参数

### 多线程处理

| 任务 | Workers | 品种数 | 时间 | 加速比 |
|------|---------|--------|------|--------|
| Canonical bars | 8 | 24 | ~30s | 6-8x |
| Sample index | 8 | 24 | ~20s | 6-8x |

### 训练

| 参数 | 值 |
|------|-----|
| Steps | 500 |
| Batch size | 256 |
| Learning rate | 1e-4 |
| Grad clip | 1.0 |
| Time | ~2 min |
| Speed | ~4 steps/sec |

---

## 🎓 关键经验

### 1. 数据契约

- ✅ 先验证 schema 再训练
- ✅ 使用统一的特征定义模块（`feature_schema.py`）
- ✅ 避免生成端和训练端不一致

### 2. NaN 处理

- ⚠️ `ret_1m` 第一行是 NaN（pct_change 的结果）
- ✅ 使用 `np.nan_to_num(feature_data, nan=0.0)`

### 3. 学习率调整

- ❌ 初始 3e-4 导致 NaN
- ✅ 降低到 1e-4 稳定
- 💡 建议使用 warmup

### 4. 多线程加速

- ✅ 8 workers 加速 6-8x
- ✅ 总处理时间 < 1 分钟
- 💡 根据 CPU 核心数调整

### 5. 原子写入

- ✅ 写 tmp file → replace
- ✅ 避免中途失败留下坏文件

---

## 🔍 验证检查清单

### 数据完整性

- [ ] 所有品种都有 5 个文件（raw_continuous_bars, bars, index_train/val/test）
- [ ] bars.parquet 有 8 个特征
- [ ] 所有特征 dtype = float32
- [ ] index 有窗口定位字段（x_start, x_end）
- [ ] index 有标签字段（y_h1, y_h5, y_h20, y_h60）

### 训练验证

- [ ] 训练能正常启动
- [ ] Loss 正常下降
- [ ] 无 NaN/Inf
- [ ] 产出 metrics 报告

### 快速验证命令

```bash
# 检查数据文件
ls -lh data/processed/m1_f8/DCE.JM/

# 检查特征维度
python -c "import pandas as pd; df = pd.read_parquet('data/processed/m1_f8/DCE.JM/bars.parquet'); print(df[['ret_1m', 'hl_range', 'co_change', 'vol_log1p', 'pos_log1p', 'minute_sin', 'minute_cos', 'is_session_open']].dtypes)"

# 检查样本数
python -c "import pandas as pd; print('Train:', len(pd.read_parquet('data/processed/m1_f8/DCE.JM/index_train.parquet')))"

# 检查训练报告
cat reports/m1_train_metrics.json | python -m json.tool | grep -A 5 '"loss"'
```

---

## 🚀 后续工作

### 短期

1. **增加训练步数**: 5,000-10,000 steps
2. **学习率调度**: Cosine annealing, warmup
3. **更多验证**: 不同 seed, 不同超参数

### 中期

1. **使用 AlphaTrade v0.2 完整模型**: Transformer-based
2. **添加更多特征**: 技术指标, 市场微观结构
3. **改进采样策略**: Symbol-balanced sampling

### 长期

1. **多 GPU 训练**: DDP, FSDP
2. **模型优化**: 架构搜索, 超参数优化
3. **生产部署**: 推理优化, 在线学习

---

## 📞 常见问题

### Q1: 如何添加新品种？

1. 在 `configs/universe/m1_selected.yaml` 添加品种
2. 运行 T5.1 和 T5.2 生成数据
3. 重新训练

### Q2: 如何修改特征？

1. 修改 `src/alphatrade/data_pipeline/feature_schema.py`
2. 重新运行 T5.1（canonical bars）
3. 重新运行 T5.2（sample index）
4. 重新训练

### Q3: 训练出现 NaN 怎么办？

1. 检查数据是否有 NaN/Inf
2. 降低学习率
3. 增加梯度裁剪
4. 检查 batch normalization

### Q4: 如何加速数据处理？

1. 增加 `--num-workers`（建议 8-16）
2. 使用 SSD 存储
3. 增加内存

---

## ✅ 总结

M1 任务完成了从数据扫描到训练验证的完整闭环：

- **24 个高质量品种**
- **609,655 个训练样本**
- **8 维特征（符合 AlphaTrade v0.2）**
- **稳定的训练流程**
- **完善的文档和报告**

所有关键文件和命令都已整理完毕，可以直接复用！🎉
