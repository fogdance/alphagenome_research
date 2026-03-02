# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 00:16:54

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 500
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed43/best`
- Train run ID: aa70597e
- Git SHA: 6cbb4a1
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics_seed43.json \
  --ckpt-step best \
  --split val
```

## 配置

- Run ID: aa70597e
- Split: val
- Symbols: 24
- Samples: 74,459
- Batch size: 128

## Pinball Loss

- Overall: 0.003107

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.002811 |
| h5 | 0.003530 |
| h20 | 0.002902 |
| h60 | 0.003183 |

## Quantile Coverage

- q10: 0.0000 (expected: 0.10)
- q30: 0.0004 (expected: 0.30)
- q50: 0.9912 (expected: 0.50)
- q70: 1.0000 (expected: 0.70)
- q90: 1.0000 (expected: 0.90)

## Quantile Crossing

- Rate: 0.0000
- Count: 0

## IC Metrics

- IC: -0.0016
- Rank IC: -0.0014

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0016 |
| h5 | -0.0039 |
| h20 | -0.0019 |
| h60 | 0.0127 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| CFFEX.T | 864 | 0.003093 | 0.0655 |
| CZCE.CF | 726 | 0.003089 | 0.0640 |
| CZCE.FG | 726 | 0.003217 | -0.0907 |
| CZCE.MA | 1,204 | 0.003129 | -0.0016 |
| CZCE.OI | 726 | 0.003113 | -0.0511 |
| CZCE.RM | 726 | 0.003100 | -0.0380 |
| CZCE.SR | 726 | 0.003068 | -0.0103 |
| CZCE.TA | 726 | 0.003111 | 0.0410 |
| DCE.A | 1,010 | 0.003070 | -0.0244 |
| DCE.I | 726 | 0.003146 | 0.0609 |
| DCE.J | 911 | 0.003224 | 0.0421 |
| DCE.JM | 726 | 0.003279 | 0.0041 |
| DCE.M | 726 | 0.003070 | 0.0044 |
| DCE.P | 920 | 0.003151 | 0.0126 |
| DCE.Y | 726 | 0.003095 | -0.0130 |
| INE.SC | 9,340 | 0.003151 | -0.0059 |
| SHFE.AG | 9,383 | 0.003100 | 0.0022 |
| SHFE.AL | 5,836 | 0.003082 | 0.0104 |
| SHFE.AU | 9,349 | 0.003085 | -0.0121 |
| SHFE.CU | 5,927 | 0.003090 | -0.0123 |
| SHFE.NI | 5,936 | 0.003131 | 0.0205 |
| SHFE.PB | 4,699 | 0.003077 | -0.0110 |
| SHFE.SN | 5,928 | 0.003101 | 0.0061 |
| SHFE.ZN | 5,892 | 0.003078 | -0.0136 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0004 < 0.02
- quantile_coverage[q50] = 0.9912 > 0.98
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
