# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-02 20:58:10

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 20
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed43/best`
- Train run ID: 5a685bdc
- Git SHA: da3301b
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics_seed43.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: 5a685bdc
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 128

## Pinball Loss

- Overall: 0.049094

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.052304 |
| h5 | 0.051921 |
| h20 | 0.052920 |
| h60 | 0.039230 |

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

- IC: -0.0196
- Rank IC: -0.0090

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0196 |
| h5 | -0.0059 |
| h20 | 0.0155 |
| h60 | 0.0349 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.050449 | -0.0209 |
| SHFE.AG | 9,383 | 0.048985 | -0.0078 |
| CZCE.MA | 1,204 | 0.049120 | -0.0234 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 1.0000 > 0.98
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
