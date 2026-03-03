# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 11:06:13

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 3
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/baseline_seed42/best`
- Train run ID: 38a265ee
- Git SHA: ec2d865
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m5_baseline_seed42_train_metrics.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: 38a265ee
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 128

## Pinball Loss

- Overall: 0.125448

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.124095 |
| h5 | 0.126494 |
| h20 | 0.124915 |
| h60 | 0.126289 |

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

- IC: -0.0093
- Rank IC: -0.0076

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0093 |
| h5 | -0.0149 |
| h20 | -0.0215 |
| h60 | 0.0124 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.124783 | 0.0289 |
| SHFE.AG | 9,383 | 0.125632 | -0.0248 |
| CZCE.MA | 1,204 | 0.124420 | 0.0634 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 1.0000 > 0.98
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
