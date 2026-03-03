# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 13:37:45

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 3
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/batch_256_seed43/best`
- Train run ID: cd01fa6c
- Git SHA: acd0e21
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m5_batch_256_seed43_train_metrics.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: cd01fa6c
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 256

## Pinball Loss

- Overall: 0.130922

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.134051 |
| h5 | 0.162421 |
| h20 | 0.110917 |
| h60 | 0.116298 |

## Quantile Coverage

- q10: 0.0000 (expected: 0.10)
- q30: 0.0000 (expected: 0.30)
- q50: 0.0000 (expected: 0.50)
- q70: 1.0000 (expected: 0.70)
- q90: 1.0000 (expected: 0.90)

## Quantile Crossing

- Rate: 0.0000
- Count: 0

## IC Metrics

- IC: 0.0157
- Rank IC: 0.0085

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | 0.0157 |
| h5 | 0.0009 |
| h20 | 0.0122 |
| h60 | 0.0437 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.128344 | -0.0294 |
| SHFE.AG | 9,383 | 0.131119 | 0.0156 |
| CZCE.MA | 1,204 | 0.130938 | -0.0016 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 0.0000 < 0.02
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
