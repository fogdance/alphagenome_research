# M4 Evaluation Run - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-01 22:20:51

## 配置

- Run ID: a0a17f4b
- Split: val
- Symbols: 3
- Samples: 11,313

## Pinball Loss

- Overall: 0.183290

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.152613 |
| h5 | 0.212611 |
| h20 | 0.185367 |
| h60 | 0.182568 |

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
- Rank IC: -0.0076

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0127 |
| h5 | -0.0081 |
| h20 | -0.0208 |
| h60 | 0.0076 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| DCE.JM | 726 | 0.185028 | 0.0040 |
| SHFE.AG | 9,383 | 0.183443 | -0.0100 |
| CZCE.MA | 1,204 | 0.181051 | 0.0557 |
