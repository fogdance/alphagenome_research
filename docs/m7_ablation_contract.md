# M7 Ablation Sweep Contract

**Status**: Active
**Created**: 2026-03-03
**Depends on**: M6 (frozen baseline)

---

## 1. Objective

Run 3 ablation experiments against the M6 frozen baseline, using the same benchmark setup (AlphaTrade v0.2, val split, 3 seeds). Generate a machine-readable regression report to determine whether any ablation beats or regresses from baseline.

## 2. Baseline (from M6)

| Parameter | Value |
|-----------|-------|
| Model | AlphaTrade v0.2 |
| max_steps | 500 |
| batch_size | 128 |
| clip_norm | 1.0 |
| Seeds | [42, 43, 44] |
| Primary metric | `pinball_loss.overall` (lower is better) |

## 3. Ablations

| exp_id | Change | Rationale |
|--------|--------|-----------|
| `no_clip` | clip_norm=1e9 | Remove gradient clipping to test stability |
| `batch_256` | batch_size=256 | Double batch size for smoother gradients |
| `steps_1000` | max_steps=1000 | Double training budget |

All other hyperparameters remain identical to baseline. No changes to model architecture, dataset, or evaluation.

## 4. Judgment Criteria

| Verdict | Condition |
|---------|-----------|
| **improved** | pinball_mean decreases by >= 1% vs baseline |
| **neutral** | change between -1% and +5% |
| **regressed** | pinball_mean increases by >= 5% vs baseline |

Formula: `delta_pct = (exp_mean - baseline_mean) / baseline_mean * 100`

## 5. Infrastructure

Reuses M5 sweep infrastructure:
- **Sweep config**: `configs/sweep/m7.yaml`
- **Sweep runner**: `src/alphatrade/scripts/run_m5_sweep.py`
- **Leaderboard builder**: `src/alphatrade/scripts/build_m5_leaderboard.py`
- **Regression report**: `src/alphatrade/scripts/build_m7_regression_report.py`

## 6. Reproduction

```bash
# 1. Run full sweep (baseline + 3 ablations × 3 seeds = 12 runs)
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_m5_sweep.py --sweep-config configs/sweep/m7.yaml

# 2. Generate regression report
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/build_m7_regression_report.py

# 3. Validate M7 gate
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py --profile m7 --strict
```

## 7. Required Reports (M7 Profile)

| Report | Path | Schema |
|--------|------|--------|
| m5_sweep_manifest | `reports/m5_sweep_manifest.json` | `m5_sweep_manifest.schema.json` |
| m5_leaderboard | `reports/m5_leaderboard.json` | `m5_leaderboard.schema.json` |
| m5_leaderboard_md | `reports/m5_leaderboard.md` | existence-only |
| m7_regression_report | `reports/m7_regression_report.json` | `m7_regression_report.schema.json` |
| m7_regression_report_md | `reports/m7_regression_report.md` | existence-only |

All items are required. Profile defined in `src/alphatrade/schemas/contracts_manifest.yaml` under `m7`.
