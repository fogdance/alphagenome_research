# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 13:47:34

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 3
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/steps_1000_seed43/best`
- Train run ID: 15de4235
- Git SHA: acd0e21
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m5_steps_1000_seed43_train_metrics.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: 15de4235
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 128

## Pinball Loss

- Overall: 0.131068

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.134166 |
| h5 | 0.163155 |
| h20 | 0.110429 |
| h60 | 0.116521 |

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

- IC: 0.0160
- Rank IC: 0.0089

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | 0.0160 |
| h5 | 0.0012 |
| h20 | 0.0120 |
| h60 | 0.0442 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.128981 | -0.0300 |
| SHFE.AG | 9,383 | 0.131218 | 0.0155 |
| CZCE.MA | 1,204 | 0.131153 | -0.0001 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 0.0000 < 0.02
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
