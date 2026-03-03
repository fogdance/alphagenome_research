# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 13:29:46

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 3
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m5/no_clip_seed43/best`
- Train run ID: b61aefa4
- Git SHA: acd0e21
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m5_no_clip_seed43_train_metrics.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: b61aefa4
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 128

## Pinball Loss

- Overall: 0.130303

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.126789 |
| h5 | 0.168252 |
| h20 | 0.117972 |
| h60 | 0.108197 |

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

- IC: 0.0133
- Rank IC: 0.0080

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | 0.0133 |
| h5 | 0.0020 |
| h20 | 0.0122 |
| h60 | 0.0453 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.127834 | -0.0291 |
| SHFE.AG | 9,383 | 0.130450 | 0.0150 |
| CZCE.MA | 1,204 | 0.130643 | 0.0022 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 0.0000 < 0.02
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
