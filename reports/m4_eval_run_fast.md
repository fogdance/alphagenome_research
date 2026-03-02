# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-01 22:24:14

## 配置

- Run ID: a0a17f4b
- Split: val
- Symbols: 3
- Samples: 11,313
- Batch size: 128

## Pinball Loss

- Overall: 0.183285

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.152624 |
| h5 | 0.212585 |
| h20 | 0.185363 |
| h60 | 0.182566 |

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

- IC: -0.0127
- Rank IC: -0.0075

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0127 |
| h5 | -0.0080 |
| h20 | -0.0206 |
| h60 | 0.0077 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.185025 | 0.0044 |
| SHFE.AG | 9,383 | 0.183437 | -0.0100 |
| CZCE.MA | 1,204 | 0.181046 | 0.0557 |
