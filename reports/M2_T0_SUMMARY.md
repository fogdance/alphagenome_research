# M2-T0 总结报告：配置 + 训练入口骨架

生成时间: 2026-03-01

## 任务目标

搭建 M2 训练框架，确保一条命令能启动训练并产出固定 schema 的 metrics 报告。

---

## ✅ 完成内容

### 1. 配置文件

**文件**: `configs/dataset/m2.yaml`

**内容**:
- Universe 配置（smoke_symbols + candidates_file）
- 数据路径（复用 M1 数据：`data/processed/m1_f8`）
- 特征规范（8D，固定顺序）
- 模型配置（quantiles: 5 个分位数）
- 训练配置（max_steps, batch_size, lr, grad_clip）
- Loss 配置（pinball + crossing penalty）
- Metrics schema 版本（m2_train_metrics_v1）

### 2. 训练数据规范文档

**文件**: `docs/m2_training_data_spec.md`

**内容**:
- 输入特征（8 维，固定顺序）
- 标签定义（log returns for 4 horizons）
- 数据约束（不跨 segment/合约）
- 数据分割（train/val/test）
- **Metrics Schema (m2_train_metrics_v1)** - 固定字段定义

### 3. 训练脚本骨架

**文件**: `src/alphatrade/scripts/train_m2_alphatrade_v0_2.py`

**功能**:
- ✅ 解析 config + CLI override
- ✅ M2Dataset（多品种，on-the-fly 窗口切片）
- ✅ SimpleQuantileModel（placeholder，输出 [B, H, Q]）
- ✅ Pinball loss + quantile crossing penalty
- ✅ 训练 loop（train + val）
- ✅ 产出 m2_train_metrics_v1 schema 的 JSON
- ✅ 产出 Markdown 报告

---

## 验收结果

### 测试命令

```bash
python src/alphatrade/scripts/train_m2_alphatrade_v0_2.py \
  --config configs/dataset/m2.yaml \
  --max-steps 20 \
  --smoke
```

### 测试结果

✅ **20 steps 成功完成**

| 指标 | 值 |
|------|-----|
| Symbols | 3 (DCE.JM, SHFE.AG, CZCE.MA) |
| Train samples | 57,418 |
| Val samples | 11,313 |
| Train loss | 0.024043 |
| Val loss | 0.012035 |
| Max grad norm | 1.4290 |
| NaN steps | 0 |
| Inf steps | 0 |

### 生成的文件

✅ `reports/m2_train_metrics.json` - 字段齐全，符合 m2_train_metrics_v1 schema  
✅ `reports/m2_train_run.md` - 包含运行命令、配置摘要、loss 统计

---

## Metrics Schema 验证

### m2_train_metrics_v1 必须字段

```json
{
  "run": {
    "run_id": "3b889d5a",
    "git_sha": "f86d1c89",
    "created_at": "2026-03-01T16:10:23.898198",
    "device": "cuda",
    "seed": 42
  },
  "dataset": {
    "name": "m2",
    "config_path": "configs/dataset/m2.yaml",
    "processed_dir": "data/processed/m1_f8",
    "symbols": 3,
    "train_samples": 57418,
    "val_samples": 11313,
    "test_samples": 0,
    "feature_dim": 8,
    "feature_cols": [...],
    "lookback": 60,
    "horizons": [1, 5, 20, 60],
    "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9]
  },
  "model": {
    "type": "alphatrade_v0_2",
    "total_params": 165588,
    "trainable_params": 165588,
    "config": {...}
  },
  "training": {
    "max_steps": 20,
    "batch_size": 256,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "optimizer": "adamw",
    "lr_schedule": {...}
  },
  "loss": {
    "train_last": 0.024043,
    "train_best": 0.024043,
    "val_last": 0.012035,
    "val_best": 0.012035,
    "best_step": 20,
    "by_horizon": {
      "h1": {"train": 0.0, "val": 0.0},
      "h5": {"train": 0.0, "val": 0.0},
      "h20": {"train": 0.0, "val": 0.0},
      "h60": {"train": 0.0, "val": 0.0}
    }
  },
  "stability": {
    "nan_steps": 0,
    "inf_steps": 0,
    "max_grad_norm": 1.4290
  }
}
```

**验证结果**: ✅ 所有必须字段都存在，字段名固定

---

## 关键特性

### 1. 配置灵活性

- ✅ 支持 smoke test（2-5 个品种）
- ✅ 支持完整训练（24 个品种）
- ✅ CLI 可 override max_steps, batch_size, seed

### 2. 数据处理

- ✅ 复用 M1 数据（`data/processed/m1_f8`）
- ✅ 多品种混合采样
- ✅ On-the-fly 窗口切片（不 materialize）
- ✅ NaN 处理（`np.nan_to_num`）

### 3. 模型输出

- ✅ Quantile prediction: [B, H, Q] = [B, 4, 5]
- ✅ Pinball loss
- ✅ Quantile crossing penalty

### 4. 训练稳定性

- ✅ Gradient clipping (1.0)
- ✅ NaN/Inf 检测
- ✅ Max grad norm 记录

### 5. Metrics 输出

- ✅ 固定 schema (m2_train_metrics_v1)
- ✅ 字段名不变
- ✅ 包含 by_horizon loss（占位）
- ✅ 包含 stability 指标

---

## 下一步：M2-T1

### 任务

实现完整的 Dataset/Dataloader：
- 多品种采样策略（round-robin 或 weighted）
- 输出 dataloader 统计报告
- 验证数据质量（shape, NaN/Inf, segment 约束）

### 产物

- `reports/m2_t1_dataloader_check.json`
- `reports/m2_t1_dataloader_check.md`

---

## 总结

✅ **M2-T0 验收通过**

**核心成果**:
- 配置文件完整（m2.yaml）
- 训练数据规范文档（m2_training_data_spec.md）
- 训练脚本骨架可运行（20 steps 成功）
- Metrics schema 固定且完整（m2_train_metrics_v1）

**验收标准达成**:
- ✅ 一条命令能启动训练
- ✅ 生成 JSON + MD 报告
- ✅ 字段齐全（没有缺失 key）
- ✅ 无 NaN/Inf

**可以继续 M2-T1**: 框架已搭建完成
