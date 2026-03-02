# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-02 21:01:42

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 20
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed44/best`
- Train run ID: f587ecaa
- Git SHA: da3301b
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics_seed44.json \
  --ckpt-step best \
  --split val \
  --smoke
```

## 配置

- Run ID: f587ecaa
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 128

## Pinball Loss

- Overall: 0.049309

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.049649 |
| h5 | 0.056206 |
| h20 | 0.044694 |
| h60 | 0.046689 |

## Quantile Coverage

- q10: 0.0000 (expected: 0.10)
- q30: 0.0000 (expected: 0.30)
- q50: 0.0005 (expected: 0.50)
- q70: 1.0000 (expected: 0.70)
- q90: 1.0000 (expected: 0.90)

## Quantile Crossing

- Rate: 0.0000
- Count: 0

## IC Metrics

- IC: -0.0161
- Rank IC: -0.0070

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0161 |
| h5 | -0.0049 |
| h20 | 0.0017 |
| h60 | 0.0429 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.049865 | 0.0018 |
| SHFE.AG | 9,383 | 0.049236 | -0.0277 |
| CZCE.MA | 1,204 | 0.049550 | 0.0266 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 0.0005 < 0.02
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
