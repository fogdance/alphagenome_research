# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 11:11:37

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 3
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/baseline_seed44/best`
- Train run ID: b6c00273
- Git SHA: ec2d865
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m5_baseline_seed44_train_metrics.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: b6c00273
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 128

## Pinball Loss

- Overall: 0.146016

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.117588 |
| h5 | 0.190027 |
| h20 | 0.116925 |
| h60 | 0.159526 |

## Quantile Coverage

- q10: 0.0000 (expected: 0.10)
- q30: 0.0000 (expected: 0.30)
- q50: 1.0000 (expected: 0.50)
- q70: 1.0000 (expected: 0.70)
- q90: 1.0000 (expected: 0.90)

## Quantile Crossing

- Rate: 0.0000
- Count: 0

## IC Metrics

- IC: -0.0076
- Rank IC: -0.0087

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0076 |
| h5 | -0.0025 |
| h20 | 0.0148 |
| h60 | 0.0409 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.148310 | 0.0035 |
| SHFE.AG | 9,383 | 0.145970 | -0.0152 |
| CZCE.MA | 1,204 | 0.144993 | 0.0447 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 1.0000 > 0.98
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
