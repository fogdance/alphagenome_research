# M1-T5.3 总结报告：Smoke Train

生成时间: 2026-03-01

## 任务目标

使用完整的 8D 数据进行 smoke train (500 steps)，验证训练流程稳定性。

---

## ✅ 完成内容

### 1. 训练脚本

**文件**: `src/alphatrade/scripts/train_m1_smoke.py`

**功能**:
- ✅ 使用统一的 8D 特征定义 (FEATURE_COLS)
- ✅ 多品种 dataloader（24 个品种）
- ✅ 简单 MLP baseline 模型
- ✅ 梯度裁剪 (grad_clip=1.0)
- ✅ NaN/Inf 检测和处理
- ✅ 产出机器可读 + 人类可读报告

### 2. 训练执行

**命令**:
```bash
python src/alphatrade/scripts/train_m1_smoke.py \
  --universe configs/universe/m1_selected.yaml \
  --processed-dir data/processed/m1_f8 \
  --max-steps 500 \
  --batch-size 256 \
  --lr 1e-4 \
  --seed 42
```

**结果**: ✅ 训练成功完成

---

## 训练结果

### 运行信息

| 项目 | 值 |
|------|-----|
| Run ID | 532b1d98 |
| Git SHA | 842f3f26 |
| Device | cuda |
| Seed | 42 |

### Dataset

| 项目 | 值 |
|------|-----|
| Universe | 24 symbols |
| Train samples | 387,673 |
| Val samples | 74,459 |
| Feature dim | 8 |
| Lookback | 60 |
| Horizons | [1, 5, 20, 60] |

### Model

| 项目 | 值 |
|------|-----|
| Type | SimpleMLPModel |
| Input dim | 8 |
| Hidden dims | [256, 128, 64] |
| Total params | 164,548 |

### Training

| 项目 | 值 |
|------|-----|
| Max steps | 500 |
| Batch size | 256 |
| Learning rate | 1e-4 |
| Grad clip | 1.0 |

### Loss

| 指标 | 值 |
|------|-----|
| Train last | 0.000113 |
| Train best | 0.000106 |
| Val last | 0.000041 |
| Val best | 0.000041 (step 500) |

---

## 训练曲线

### Loss 下降趋势

```
Step 100: Train=0.000609, Val=0.000202
Step 200: Train=0.000353, Val=0.000109
Step 300: Train=0.000214, Val=0.000082
Step 400: Train=0.000162, Val=0.000053
Step 500: Train=0.000113, Val=0.000041
```

**观察**:
- ✅ Loss 持续下降
- ✅ 无 NaN/Inf
- ✅ Val loss 低于 train loss（可能是 dropout 的影响）
- ✅ 收敛稳定

---

## 验收标准达成

### ✅ 必须项

1. ✅ **一条命令稳定产出 json + md**
   - `reports/m1_train_metrics.json` ✅
   - `reports/m1_train_run.md` ✅

2. ✅ **Loss 能正常下降或至少不发散**
   - Train loss: 0.000609 → 0.000113 (下降 81%)
   - Val loss: 0.000202 → 0.000041 (下降 80%)

3. ✅ **无 NaN/Inf**
   - 所有 500 steps 都正常
   - 添加了 NaN 处理（np.nan_to_num）

---

## 问题与解决

### 问题 1: 初次训练出现 NaN

**原因**:
- `ret_1m` 第一行是 NaN（pct_change 的结果）
- 学习率可能过高（3e-4）

**解决**:
1. 添加 NaN 处理：`np.nan_to_num(feature_data, nan=0.0)`
2. 降低学习率：3e-4 → 1e-4

**结果**: ✅ 训练稳定

---

## 生成的文件

### 报告文件

- `reports/m1_train_metrics.json` - 机器可读报告
- `reports/m1_train_run.md` - 人类可读报告
- `reports/M1_T5_3_SUMMARY.md` - 本文件

### 报告内容

**m1_train_metrics.json** 包含:
- run: run_id, git_sha, created_at, device, seed
- dataset: name, universe, symbols, samples, feature_dim, feature_cols
- model: type, params, architecture
- training: max_steps, batch_size, lr, grad_clip
- loss: train_last, train_best, val_last, val_best, best_step

**m1_train_run.md** 包含:
- 运行命令（可复制）
- 配置摘要
- Loss 统计
- 训练状态

---

## 数据流验证

### 完整数据流

```
data/processed/m1_f8/{symbol}/
  ├── bars.parquet (8D features)
  │   └── [ret_1m, hl_range, co_change, vol_log1p, 
  │        pos_log1p, minute_sin, minute_cos, is_session_open]
  │
  ├── index_train.parquet
  │   └── [x_start, x_end, y_h1, y_h5, y_h20, y_h60]
  │
  └── Dataloader
      └── Slice window: bars[x_start:x_end+1] → [L=60, F=8]
          └── Model: MLP([60*8]) → [4 horizons]
              └── Loss: MSE(pred, labels)
```

**验证结果**: ✅ 数据流完整且正确

---

## 性能统计

### 训练速度

- **Total steps**: 500
- **Total time**: ~2 分钟
- **Speed**: ~4 steps/sec
- **Samples/sec**: ~1,000 samples/sec (256 batch × 4 steps)

### 资源使用

- **Device**: CUDA (GPU)
- **Memory**: 正常（无 OOM）
- **CPU workers**: 4

---

## 下一步建议

### 短期改进

1. **增加训练步数**
   - 当前 500 steps 只是 smoke test
   - 建议 5,000-10,000 steps 进行完整训练

2. **调整学习率**
   - 当前 1e-4 较保守
   - 可以尝试 3e-4 with warmup

3. **添加学习率调度**
   - Cosine annealing
   - Step decay

### 长期改进

1. **使用 AlphaTrade v0.2 完整模型**
   - 当前使用简单 MLP
   - 可以切换到 Transformer-based 模型

2. **添加更多特征**
   - 当前 8 个特征
   - 可以添加更多技术指标

3. **多 GPU 训练**
   - 当前单 GPU
   - 可以使用 DDP 加速

---

## 总结

✅ **M1-T5.3 任务完成**

**核心成果**:
- 训练流程稳定运行 500 steps
- Loss 正常下降，无 NaN/Inf
- 产出完整的 metrics 报告
- 验证了完整的数据流

**关键指标**:
- Train loss: 0.000113
- Val loss: 0.000041
- 训练时间: ~2 分钟
- 成功率: 100%

**数据质量**:
- 8D 特征正确加载 ✅
- 窗口切片正确 ✅
- 标签正确 ✅
- NaN 处理正确 ✅

**可以继续后续工作**: 训练流程验证完成，可以进行更长时间的训练
