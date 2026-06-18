# M6 Baseline Contract

**Status**: Frozen
**Created**: 2026-03-03

---

## 1. Model

- **Model**: AlphaTrade v0.2
- **Architecture**: hidden_dim=256, num_layers=6, num_heads=8, dropout=0.1

## 2. Universe

- **Universe**: `cta_top20` — 24 symbols from `configs/universe/m1_selected.yaml`
- **Selection criteria**: min_coverage=95%, min_train_samples=5000, min_val_samples=500, min_test_samples=1000

## 3. Dataset

- **Config**: `configs/dataset/m2.yaml`
- **Features**: 8D (ret_1m, hl_range, co_change, vol_log1p, pos_log1p, minute_sin, minute_cos, is_session_open)
- **Lookback**: 60
- **Horizons**: [1, 5, 20, 60]
- **Quantiles**: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Train split**: 2018-01-01 ~ 2023-01-01
- **Val split**: 2023-01-01 ~ 2024-01-01

## 4. Training

| Parameter | Value |
|-----------|-------|
| max_steps | 500 |
| batch_size | 128 |
| jit | 1 |
| clip_norm | 1.0 |
| optimizer | adamw |
| learning_rate | 0.0001 |
| save_every | 100 |
| keep_last | 3 |

## 5. Seeds

- **Seeds**: [42, 43, 44]

## 6. Evaluation

- **Split**: val
- **Checkpoint**: best (by val pinball_loss)

## 7. Primary Metric

- **Metric**: `pinball_loss.overall` (lower is better)
- Secondary: IC, rank_IC, quantile_coverage, quantile_crossing_rate

## 8. Reproduction

```bash
# Full 3-seed sweep (uses configs/sweep/m5.yaml)
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_m5_sweep.py --sweep-config configs/sweep/m5.yaml

# Generate M6 baseline report
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/gen_m6_baseline_report.py --baseline-exp baseline

# Validate
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py --profile m6 --strict
```

## 9. Required Reports (M6 Profile)

| Report | Path | Schema |
|--------|------|--------|
| m5_sweep_manifest | `$ALPHATRADE_RUNS_ROOT/reports/m5_sweep_manifest.json` | `m5_sweep_manifest.schema.json` |
| m5_leaderboard | `$ALPHATRADE_RUNS_ROOT/reports/m5_leaderboard.json` | `m5_leaderboard.schema.json` |
| m5_leaderboard_md | `$ALPHATRADE_RUNS_ROOT/reports/m5_leaderboard.md` | existence-only |
| m5_schema_validation | `$ALPHATRADE_RUNS_ROOT/reports/m5_schema_validation.json` | existence-only |
| m5_schema_validation_md | `$ALPHATRADE_RUNS_ROOT/reports/m5_schema_validation.md` | existence-only |
| m6_baseline_run | `$ALPHATRADE_RUNS_ROOT/reports/m6_baseline_run.md` | existence-only |

All items are required. Profile defined in `src/alphatrade/schemas/contracts_manifest.yaml` under `m6`.

`gen_m6_baseline_report.py` must freeze the explicitly selected baseline experiment by `exp_id`;
it must not assume the first leaderboard row is the baseline, because the leaderboard is metric-sorted.
