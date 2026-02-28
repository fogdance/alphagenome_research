# M0.1 验收清单

生成时间: 2026-02-28

## ✅ T2: Archive 读取器

**产物**:
- [x] `src/alphatrade/data_pipeline/archive_reader.py` - Archive 读取模块
- [x] 支持从 manifest 驱动读取 parquet
- [x] 自动拼接多月数据
- [x] EOB 去重（保留 last）
- [x] 数据校验（单调性、缺失值、dtype）

**验证**:
- DCE.JM: 663,184 rows, 无缺失值, EOB 单调递增 ✅
- SHFE.RB: 654,596 rows, 无缺失值, EOB 单调递增 ✅

---

## ✅ T3: Canonical Bars 生成

**产物**:
- [x] `src/alphatrade/scripts/build_canonical_bars.py`
- [x] `data/processed/m0_1/{symbol}/bars.parquet`
- [x] `reports/m0_1_canonical_profile.md`

**功能**:
- [x] Gap-based 分段（gap_seconds=1800）
- [x] 基础特征：ret_1m, hl_range, co_change, vol_log1p, pos_log1p
- [x] 分钟相位特征：minute_sin, minute_cos, is_open
- [x] 数据类型优化（float32, int32）

**验证**:
- DCE.JM: 663,184 bars, 5,699 segments ✅
- SHFE.RB: 654,596 bars, 5,699 segments ✅
- 特征统计正常（ret_1m 均值≈0）✅

---

## ✅ T4: 训练样本索引生成

**产物**:
- [x] `src/alphatrade/scripts/build_sample_index.py`
- [x] `data/processed/m0_1/{symbol}/index_{train,val,test}.parquet`
- [x] `reports/m0_1_index_stats.md`

**功能**:
- [x] 滑动窗口采样（lookback=60, stride=5）
- [x] 多 horizon 标签（1, 5, 20, 60 分钟）
- [x] 时间切分（train/val/test）
- [x] 同 segment 约束

**验证**:
- DCE.JM: train=6,333, val=726, test=720 ✅
- SHFE.RB: train=4,537, val=726, test=720 ✅
- 总计: train=10,870, val=1,452, test=1,440 ✅
- 无跨 segment 采样 ✅
- Label 分布合理（均值≈0）✅

---

## ✅ T5: Baseline 训练

**产物**:
- [x] `src/alphatrade/scripts/train_m0_1_baseline.py`
- [x] `reports/m0_1_train_metrics.json`
- [x] `reports/m0_1_train_run.md`

**功能**:
- [x] 简单 MLP 模型
- [x] 多任务回归（4 个 horizon）
- [x] 在线窗口切片（不提前物化）
- [x] 标准化 metrics JSON 输出

**验证**:
- [x] 一条命令跑完 ✅
- [x] Dataloader 在线切片 ✅
- [x] JSON schema 符合要求 ✅
- [x] 包含 train_last/train_best/val_last/val_best ✅
- [x] 训练收敛（200 steps, train_loss=0.000155, val_loss=0.000134）✅

**Metrics JSON 必填字段验证**:
```json
{
  "run": {
    "run_id": "90c310db",           ✅
    "git_sha": "5b056cc8",          ✅
    "created_at": "2026-02-28 ...", ✅
    "device": "cuda",               ✅
    "seed": 42                      ✅
  },
  "dataset": {
    "name": "m0_1",                 ✅
    "config_path": "...",           ✅
    "lookback": 60,                 ✅
    "stride": 5,                    ✅
    "horizons": [1,5,20,60],        ✅
    "features": [...],              ✅
    "symbols": ["DCE.JM","SHFE.RB"],✅
    "samples": {
      "train_total": 10870,         ✅
      "val_total": 1452,            ✅
      "by_symbol": {...}            ✅
    }
  },
  "train": {
    "max_steps": 200,               ✅
    "batch_size": 128,              ✅
    "lr": 0.001,                    ✅
    "loss": {
      "train_last": 0.000155,       ✅
      "train_best": 0.000155,       ✅
      "val_last": 0.000134,         ✅
      "val_best": 0.000134,         ✅
      "best_step": 200              ✅
    }
  }
}
```

---

## 数据质量门槛

### Archive Read
- [x] 每个 symbol 行数统计 ✅
- [x] 起止 eob ✅
- [x] 重复 eob 数量 ✅
- [x] 是否单调递增 ✅

### Canonical Profile
- [x] Segment 数量 ✅
- [x] Segment 长度统计（min/median/p95）✅
- [x] 特征统计（mean/std/p1/p99）✅

### Index Stats
- [x] Train/val/test 样本数 ✅
- [x] Label 分布统计 ✅
- [x] 跨 segment 样本=0 的证明 ✅

### Smoke Train
- [x] 训练命令 ✅
- [x] Train/val loss ✅
- [x] 最好 val ✅
- [x] Metrics.json 路径 ✅

---

## 代码组织验证

### 工程分离 ✅
- Archive 管理在 juejin 工程 ✅
- 数据加工和训练在 alphatrade 工程 ✅

### 模块结构 ✅
```
src/alphatrade/
├── data_pipeline/          ✅
│   ├── __init__.py
│   └── archive_reader.py
├── scripts/                ✅
│   ├── build_canonical_bars.py
│   ├── build_sample_index.py
│   └── train_m0_1_baseline.py
└── configs/                ✅
    └── dataset/
        └── m0_1.yaml
```

### 数据目录 ✅
```
data/processed/m0_1/
├── DCE.JM/                 ✅
│   ├── bars.parquet
│   ├── index_train.parquet
│   ├── index_val.parquet
│   └── index_test.parquet
└── SHFE.RB/                ✅
    ├── bars.parquet
    ├── index_train.parquet
    ├── index_val.parquet
    └── index_test.parquet
```

---

## 最终验收

### 必须满足的要求
1. ✅ 一条命令跑完训练
2. ✅ Dataloader 运行时切片（不提前 materialize）
3. ✅ JSON schema 满足要求（key 名固定）
4. ✅ 至少 train_loss 与 val_loss 都有 last/best

### 额外加分项
1. ✅ 支持 --limit-train-samples / --limit-val-samples
2. ✅ Git SHA 获取（失败时写 "unknown"）
3. ✅ 完整的报告生成（markdown + json）
4. ✅ 数据质量验证和统计

---

## 运行命令总结

```bash
# 完整流程
python src/alphatrade/scripts/build_canonical_bars.py --symbol DCE.JM
python src/alphatrade/scripts/build_canonical_bars.py --symbol SHFE.RB
python src/alphatrade/scripts/build_sample_index.py --symbol DCE.JM
python src/alphatrade/scripts/build_sample_index.py --symbol SHFE.RB
python src/alphatrade/scripts/train_m0_1_baseline.py --max-steps 500

# 快速 smoke test
python src/alphatrade/scripts/train_m0_1_baseline.py \
  --max-steps 200 \
  --batch-size 128 \
  --limit-train-samples 1000
```

---

## 结论

✅ **所有任务（T2-T5）已完成并通过验收**

- 数据处理流程完整且可复现
- 训练流程稳定且符合规范
- 报告格式标准化，便于后续迭代
- 代码组织清晰，职责分离明确

可以进入下一阶段的模型优化和扩展工作。
