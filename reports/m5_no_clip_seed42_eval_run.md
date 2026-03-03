# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 13:27:01

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 3
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/no_clip_seed42/best`
- Train run ID: 356fca98
- Git SHA: acd0e21
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m5_no_clip_seed42_train_metrics.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: 356fca98
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 128

## Pinball Loss

- Overall: 0.129604

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.142762 |
| h5 | 0.128755 |
| h20 | 0.113709 |
| h60 | 0.133192 |

## Quantile Coverage

- q10: 0.0000 (expected: 0.10)
- q30: 0.0001 (expected: 0.30)
- q50: 1.0000 (expected: 0.50)
- q70: 1.0000 (expected: 0.70)
- q90: 1.0000 (expected: 0.90)

## Quantile Crossing

- Rate: 0.0000
- Count: 0

## IC Metrics

- IC: -0.0135
- Rank IC: -0.0101

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0135 |
| h5 | -0.0145 |
| h20 | -0.0223 |
| h60 | 0.0079 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.131565 | 0.0273 |
| SHFE.AG | 9,383 | 0.129060 | -0.0267 |
| CZCE.MA | 1,204 | 0.132668 | 0.0605 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0001 < 0.02
- quantile_coverage[q50] = 1.0000 > 0.98
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
