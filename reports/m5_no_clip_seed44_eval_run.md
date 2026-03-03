# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 13:32:31

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 3
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/no_clip_seed44/best`
- Train run ID: 27934f62
- Git SHA: acd0e21
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m5_no_clip_seed44_train_metrics.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: 27934f62
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 128

## Pinball Loss

- Overall: 0.148020

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.118175 |
| h5 | 0.191787 |
| h20 | 0.118889 |
| h60 | 0.163230 |

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
- Rank IC: -0.0086

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0076 |
| h5 | -0.0025 |
| h20 | 0.0145 |
| h60 | 0.0411 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.150324 | 0.0033 |
| SHFE.AG | 9,383 | 0.147972 | -0.0152 |
| CZCE.MA | 1,204 | 0.147006 | 0.0447 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 1.0000 > 0.98
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
