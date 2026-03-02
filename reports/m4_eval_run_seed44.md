# M4 Evaluation Run (FAST) - AlphaTrade v0.2 (JAX)

生成时间: 2026-03-03 00:40:28

## Checkpoint Info

- Source: checkpoint
- Checkpoint step: 500
- Checkpoint dir: `/home/v/Documents/work/1_open_source/alphagenome_research/checkpoints/m4/matrix_seed44/best`
- Train run ID: 66dec787
- Git SHA: 6cbb4a1
- Split: val

Reproduce:
```bash
python src/alphatrade/scripts/eval_m4_fast.py \
  --train-metrics reports/m4_train_metrics_seed44.json \
  --ckpt-step best \
  --split val
```

## 配置

- Run ID: 66dec787
- Split: val
- Symbols: 24
- Samples: 74,459
- Batch size: 128

## Pinball Loss

- Overall: 0.003388

### By-Horizon

| Horizon | Loss |
|---------|------|
| h1 | 0.003100 |
| h5 | 0.003766 |
| h20 | 0.002625 |
| h60 | 0.004062 |

## Quantile Coverage

- q10: 0.0000 (expected: 0.10)
- q30: 0.0000 (expected: 0.30)
- q50: 0.6388 (expected: 0.50)
- q70: 1.0000 (expected: 0.70)
- q90: 1.0000 (expected: 0.90)

## Quantile Crossing

- Rate: 0.0000
- Count: 0

## IC Metrics

- IC: 0.0012
- Rank IC: 0.0017

### By-Horizon IC

| Horizon | IC |
|---------|----|
| h1 | 0.0012 |
| h5 | -0.0019 |
| h20 | 0.0053 |
| h60 | -0.0055 |

## By-Symbol Metrics

| Symbol | Samples | Pinball Loss | IC |
|--------|---------|--------------|----|
| CFFEX.T | 864 | 0.003417 | -0.0123 |
| CZCE.CF | 726 | 0.003436 | 0.0454 |
| CZCE.FG | 726 | 0.003585 | -0.0089 |
| CZCE.MA | 1,204 | 0.003474 | 0.0016 |
| CZCE.OI | 726 | 0.003467 | -0.0211 |
| CZCE.RM | 726 | 0.003464 | -0.0835 |
| CZCE.SR | 726 | 0.003418 | -0.0020 |
| CZCE.TA | 726 | 0.003471 | -0.0129 |
| DCE.A | 1,010 | 0.003393 | 0.0264 |
| DCE.I | 726 | 0.003506 | 0.0712 |
| DCE.J | 911 | 0.003571 | 0.0154 |
| DCE.JM | 726 | 0.003622 | -0.0231 |
| DCE.M | 726 | 0.003431 | -0.0052 |
| DCE.P | 920 | 0.003501 | 0.0131 |
| DCE.Y | 726 | 0.003452 | -0.0212 |
| INE.SC | 9,340 | 0.003407 | 0.0173 |
| SHFE.AG | 9,383 | 0.003359 | -0.0181 |
| SHFE.AL | 5,836 | 0.003351 | 0.0010 |
| SHFE.AU | 9,349 | 0.003339 | -0.0135 |
| SHFE.CU | 5,927 | 0.003351 | -0.0068 |
| SHFE.NI | 5,936 | 0.003410 | 0.0066 |
| SHFE.PB | 4,699 | 0.003363 | -0.0214 |
| SHFE.SN | 5,928 | 0.003394 | 0.0015 |
| SHFE.ZN | 5,892 | 0.003358 | 0.0168 |

## Sanity Warnings

- quantile_coverage[q10] = 0.0000 < 0.02
- quantile_coverage[q30] = 0.0000 < 0.02
- quantile_coverage[q70] = 1.0000 > 0.98
- quantile_coverage[q90] = 1.0000 > 0.98
