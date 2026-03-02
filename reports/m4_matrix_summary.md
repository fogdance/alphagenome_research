# M4 Matrix Summary
**生成时间**: 2026-03-01 23:04:58
---

## 配置

- Dataset config: configs/dataset/m2.yaml
- Seeds: [42, 43, 44]
- Max steps: 10
- Batch size: 32
- JIT: enabled
- Smoke test: yes

## 训练结果对比

| Seed | Train Loss | Val Loss | Best Step | NaN Steps | Inf Steps | Grad Norm (pre/post) |
|------|------------|----------|-----------|-----------|-----------|----------------------|
| 42 | 0.327098 | 0.302739 | 10 | 0 | 0 | 6.3908/0.2496 |
| 43 | 0.279281 | 0.258886 | 10 | 0 | 0 | 4.7166/0.2498 |
| 44 | 0.334956 | 0.305501 | 10 | 0 | 0 | 6.9422/0.2496 |

### By-Horizon 训练损失

| Seed | h1 | h5 | h20 | h60 |
|------|----|----|-----|-----|
| 42 | 0.123812 | 0.122633 | 0.115851 | 0.128888 |
| 43 | 0.119000 | 0.113663 | 0.121047 | 0.090296 |
| 44 | 0.114861 | 0.174008 | 0.112178 | 0.125908 |

## 统计分析

### 训练损失统计

- Train loss: 0.313778 ± 0.024603
- Val loss: 0.289042 ± 0.021354
- Best val loss: 0.258886 (seed=43)

## 生成文件

### 训练输出

- `reports/m4_train_metrics_seed42.json`
- `reports/m4_train_run_seed42.md`
- `reports/m4_train_metrics_seed43.json`
- `reports/m4_train_run_seed43.md`
- `reports/m4_train_metrics_seed44.json`
- `reports/m4_train_run_seed44.md`

---

**总计**: 3 个训练实验
