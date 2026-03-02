# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-02 21:38:30

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 500
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed42/best`
- Train run ID: 3c4db23f
- Git SHA: da3301b
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics_seed42.json \
  --ckpt-step best \
  --split val
```

## 配置

- Run ID: 3c4db23f
- Split: val
- Symbols: 24
- Samples: 74,459
- Batch size: 128

## Pinball Loss

- Overall: 0.005037

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.003542 |
| h5 | 0.005427 |
| h20 | 0.004849 |
| h60 | 0.006332 |

## Quantile Coverage

- q10: 0.0000 (expected: 0.10)
- q30: 0.0000 (expected: 0.30)
- q50: 0.0018 (expected: 0.50)
- q70: 1.0000 (expected: 0.70)
- q90: 1.0000 (expected: 0.90)

## Quantile Crossing

- Rate: 0.0000
- Count: 0

## IC Metrics

- IC: -0.0003
- Rank IC: -0.0046

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | -0.0003 |
| h5 | -0.0073 |
| h20 | 0.0006 |
| h60 | -0.0064 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| CFFEX.T | 864 | 0.005044 | -0.0061 |
| CZCE.CF | 726 | 0.004920 | 0.0172 |
| CZCE.FG | 726 | 0.005113 | 0.0380 |
| CZCE.MA | 1,204 | 0.005015 | 0.0443 |
| CZCE.OI | 726 | 0.005151 | -0.0371 |
| CZCE.RM | 726 | 0.005064 | -0.0353 |
| CZCE.SR | 726 | 0.004972 | 0.0013 |
| CZCE.TA | 726 | 0.004984 | -0.0148 |
| DCE.A | 1,010 | 0.004985 | -0.0256 |
| DCE.I | 726 | 0.005015 | 0.0245 |
| DCE.J | 911 | 0.005139 | -0.0066 |
| DCE.JM | 726 | 0.005170 | 0.0051 |
| DCE.M | 726 | 0.004909 | 0.0386 |
| DCE.P | 920 | 0.005133 | 0.0514 |
| DCE.Y | 726 | 0.005074 | -0.0046 |
| INE.SC | 9,340 | 0.005082 | 0.0137 |
| SHFE.AG | 9,383 | 0.005003 | -0.0236 |
| SHFE.AL | 5,836 | 0.004997 | 0.0008 |
| SHFE.AU | 9,349 | 0.004989 | -0.0060 |
| SHFE.CU | 5,927 | 0.005002 | -0.0083 |
| SHFE.NI | 5,936 | 0.005115 | -0.0029 |
| SHFE.PB | 4,699 | 0.005042 | -0.0182 |
| SHFE.SN | 5,928 | 0.005072 | -0.0016 |
| SHFE.ZN | 5,892 | 0.005035 | 0.0061 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q50] = 0.0018 < 0.02
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
