# M6 Baseline Run Report

Generated: 2026-03-03 13:09:42
git_sha: `ec2d865`

## Reproduction

```bash
# Run 3-seed sweep
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_m5_sweep.py --sweep-config configs/sweep/m5.yaml

# Generate this report
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/gen_m6_baseline_report.py
```

## Data

- **Universe**: cta_top20 (24 symbols)
- **Dataset config**: `configs/dataset/m2.yaml`
- **Lookback**: 60
- **Horizons**: [1, 5, 20, 60]
- **Quantiles**: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Train**: 2018-01-01 ~ 2023-01-01
- **Val**: 2023-01-01 ~ 2024-01-01
- **Features**: 8D

## Training Config

| Parameter | Value |
|-----------|-------|
| max_steps | 500 |
| batch_size | 128 |
| jit | 1 |
| clip_norm | 1.0 |
| optimizer | adamw |
| learning_rate | 0.0001 |
| eval_split | val |
| ckpt_step | best |
| seeds | [42, 43, 44] |

## Metrics Summary

- **Primary metric**: `pinball_loss.overall` (lower is better)
- **Mean**: 0.134178
- **Std**: 0.010631
- **Best**: 0.125448 (run_id: 38a265ee)

## Per-Seed Results

| Seed | Run ID | Pinball Loss | IC | Rank IC | Crossing Rate |
|------|--------|--------------|-----|---------|---------------|
| 42 | 38a265ee | 0.125448 | -0.009325 | -0.007650 | 0.0000 |
| 43 | e2eb17a4 | 0.131068 | 0.016028 | 0.008861 | 0.0000 |
| 44 | b6c00273 | 0.146016 | -0.007615 | -0.008672 | 0.0000 |

## Artifacts

- Sweep manifest: `reports/m5_sweep_manifest.json`
- Leaderboard: `reports/m5_leaderboard.json`
- Seed 42 train: `reports/m5_baseline_seed42_train_metrics.json`
- Seed 42 eval: `reports/m5_baseline_seed42_eval_metrics.json`
- Seed 43 train: `reports/m5_baseline_seed43_train_metrics.json`
- Seed 43 eval: `reports/m5_baseline_seed43_eval_metrics.json`
- Seed 44 train: `reports/m5_baseline_seed44_train_metrics.json`
- Seed 44 eval: `reports/m5_baseline_seed44_eval_metrics.json`
