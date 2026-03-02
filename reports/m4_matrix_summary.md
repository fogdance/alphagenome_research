# M4 Matrix Summary

**生成时间**: 2026-03-03T00:40:30.648696

---


## 配置

- Dataset config: `configs/dataset/m2.yaml`
- Seeds: [42, 43, 44]
- Max steps: 500
- Batch size: 128
- JIT: disabled
- Smoke: no
- GPU: yes
- Eval split: val

## Train + Eval 对比

| Seed | Train Loss | Val Loss | Best Step | NaN/Inf | Grad Max | Pinball | IC | Rank IC | Crossing |
|------|-----------|----------|-----------|---------|----------|---------|------|---------|----------|
| 42 | 0.021610 | 0.020110 | 500 | 0/0 | 7.6113 | 0.005037 | -0.0003 | -0.0046 | 0.0000 |
| 43 | 0.017460 | 0.012557 | 500 | 0/0 | 7.5788 | 0.003107 | -0.0016 | -0.0014 | 0.0000 |
| 44 | 0.016444 | 0.013364 | 500 | 0/0 | 7.1594 | 0.003388 | 0.0012 | 0.0017 | 0.0000 |

### By-Horizon 训练损失

| Seed | h1 | h5 | h20 | h60 |
|------|----|----|-----|-----|
| 42 | 0.003995 | 0.003863 | 0.005075 | 0.008242 |
| 43 | 0.003049 | 0.003652 | 0.002900 | 0.005579 |
| 44 | 0.004136 | 0.003699 | 0.003633 | 0.004484 |

### By-Horizon 评估损失

| Seed | h1 | h5 | h20 | h60 |
|------|----|----|-----|-----|
| 42 | 0.003542 | 0.005427 | 0.004849 | 0.006332 |
| 43 | 0.002811 | 0.003530 | 0.002902 | 0.003183 |
| 44 | 0.003100 | 0.003766 | 0.002625 | 0.004062 |

### Quantile Coverage

| Seed | q10 | q30 | q50 | q70 | q90 |
|------|-----|-----|-----|-----|-----|
| 42 | 0.0000 | 0.0000 | 0.0018 | 1.0000 | 1.0000 |
| 43 | 0.0000 | 0.0004 | 0.9912 | 1.0000 | 1.0000 |
| 44 | 0.0000 | 0.0000 | 0.6388 | 1.0000 | 1.0000 |

## 统计分析

### 训练

- Train loss: 0.018505 +/- 0.002235
- Val loss:   0.015344 +/- 0.003386
- Best val:   0.012557 (seed=43)

### 评估

- Pinball loss: 0.003844 +/- 0.000852
- IC:           -0.0002 +/- 0.0012
- Best IC:      0.0012 (seed=44)

## 文件追溯

| Seed | train_metrics | eval_metrics | ckpt_dir |
|------|---------------|--------------|----------|
| 42 | `reports/m4_train_metrics_seed42.json` | `reports/m4_eval_metrics_seed42.json` | `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed42` |
| 43 | `reports/m4_train_metrics_seed43.json` | `reports/m4_eval_metrics_seed43.json` | `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed43` |
| 44 | `reports/m4_train_metrics_seed44.json` | `reports/m4_eval_metrics_seed44.json` | `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed44` |

---

**总计**: 3/3 seeds 完成 train+eval
