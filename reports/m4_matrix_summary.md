# M4 Matrix Summary

**生成时间**: 2026-03-02T21:51:51.952088

---


## 配置

- Dataset config: `configs/dataset/m2.yaml`
- Seeds: [42, 43, 44]
- Max steps: 500
- Batch size: 128
- JIT: enabled
- Smoke: no
- GPU: yes
- Eval split: val

## Train + Eval 对比

| Seed | Train Loss | Val Loss | Best Step | NaN/Inf | Grad Max | Pinball | IC | Rank IC | Crossing |
|------|-----------|----------|-----------|---------|----------|---------|------|---------|----------|
| 42 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 43 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| 44 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

## 统计分析

## 文件追溯

| Seed | train_metrics | eval_metrics | ckpt_dir |
|------|---------------|--------------|----------|
| 42 | `N/A` | `N/A` | `N/A` |
| 43 | `N/A` | `N/A` | `N/A` |
| 44 | `N/A` | `N/A` | `N/A` |

---

**总计**: 0/3 seeds 完成 train+eval
