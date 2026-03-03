# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 13:35:09

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 3
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/batch_256_seed42/best`
- Train run ID: 9169f783
- Git SHA: acd0e21
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m5_batch_256_seed42_train_metrics.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: 9169f783
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 256

## Pinball Loss

- Overall: 0.123102

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.123493 |
| h5 | 0.120371 |
| h20 | 0.123440 |
| h60 | 0.125103 |

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

- IC: -0.0075
- Rank IC: -0.0068

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0075 |
| h5 | -0.0145 |
| h20 | -0.0215 |
| h60 | 0.0106 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.122967 | 0.0302 |
| SHFE.AG | 9,383 | 0.122953 | -0.0237 |
| CZCE.MA | 1,204 | 0.124344 | 0.0633 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 1.0000 > 0.98
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
