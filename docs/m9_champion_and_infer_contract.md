# M9 Champion Bundle + Offline Inference + Backtest Contract

**Status**: Active
**Created**: 2026-03-03
**Depends on**: M7/M8 (iteration loop with leaderboard + regression gate)

---

## 1. Objective

Select the champion model from the M5 leaderboard, package it into a reproducible bundle, provide offline batch inference that outputs stable `predictions.parquet`, and hand predictions to a lightweight backtest report.

## 2. Champion Selection Rule

| Step | Rule |
|------|------|
| Primary | Lowest `primary_mean` on leaderboard (experiments[0], already sorted) |
| Tiebreaker | Lowest `primary_std` (stability) |
| Exported seed | `best_run_id` from the winning experiment (best individual seed) |

Override: `--exp-id` / `--run-id` flags on the export script.

## 3. Bundle Format

Self-contained directory under `$ALPHATRADE_RUNS_ROOT/artifacts/model_bundle/<model_version>/`:

| File | Description |
|------|-------------|
| `best/` | Flax checkpoint dir (copied from `$ALPHATRADE_RUNS_ROOT/checkpoints/m5/`) |
| `artifacts.json` | Training metadata (copied) |
| `model_config.json` | AlphaTradeConfig as JSON (extracted for readability) |
| `bundle_manifest.json` | Full provenance record |

Also writes `$ALPHATRADE_RUNS_ROOT/reports/m9_model_bundle_manifest.json` (same content, validated by m9 profile).

The manifest includes `bundle_format_version`, `prediction_schema_version`, `bundle_id`, and a `bundle_files` SHA256 manifest so a deployed bundle can be checked for accidental mutation.

## 4. predictions.parquet Format

Wide format — one row per (symbol, eob):

| Column | Type |
|--------|------|
| `symbol` | string |
| `eob` | datetime |
| `model_version` | string |
| `h{H}_q{Q}` (20 columns) | float64 |

4 horizons (1, 5, 20, 60) × 5 quantiles (10, 30, 50, 70, 90) = 20 prediction columns.

## 5. Batch Inference Data Loading

Directly slice `bars.parquet` by `eob` timestamp range (not relying on pre-built index files). Create sliding windows with lookback=60, stride=1. Only needs `data/processed/m1_f8/{symbol}/bars.parquet`.

## 6. Backtest Handoff

`backtest_predictions.py` consumes `m9_predictions.parquet`, aligns `(symbol, eob)` to `data/processed/m1_f8/{symbol}/bars.parquet`, uses one prediction column such as `h20_q50` as a deterministic direction signal, and writes:

- `$ALPHATRADE_RUNS_ROOT/reports/m9_backtest_metrics.json`
- `$ALPHATRADE_RUNS_ROOT/reports/m9_backtest_metrics.md`
- `$ALPHATRADE_RUNS_ROOT/reports/m9_backtest_trades.parquet`

This is intentionally a minimal integration contract, not a full execution simulator.

## 7. Reproduction

```bash
export ALPHATRADE_RUNS_ROOT="$(pwd)/../alphatrade_runs/default"

# 1. Export champion bundle
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/export_model_bundle.py

# 2. Run batch inference (smoke: 2 symbols × small range)
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/batch_infer_offline.py \
  --bundle "$ALPHATRADE_RUNS_ROOT/artifacts/model_bundle/alphatrade_v0.2_batch_256_9169f783" \
  --data-dir data/processed/m1_f8 \
  --symbols DCE.JM,SHFE.AG \
  --start 2024-01-02 --end 2024-01-04 \
  --smoke

# 3. Backtest predictions handoff
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/backtest_predictions.py \
  --predictions "$ALPHATRADE_RUNS_ROOT/reports/m9_predictions.parquet" \
  --data-dir data/processed/m1_f8 \
  --horizon 20 --quantile 0.5

# 4. Validate M9 gate
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py --profile m9 --strict
```

## 8. Required Reports (M9 Profile)

| Report | Path | Schema |
|--------|------|--------|
| m9_model_bundle_manifest | `$ALPHATRADE_RUNS_ROOT/reports/m9_model_bundle_manifest.json` | `m9_model_bundle_manifest.schema.json` |
| m9_infer_metrics | `$ALPHATRADE_RUNS_ROOT/reports/m9_infer_metrics.json` | `m9_infer_metrics.schema.json` |
| m9_infer_metrics_md | `$ALPHATRADE_RUNS_ROOT/reports/m9_infer_metrics.md` | existence-only |
| m9_predictions_parquet | `$ALPHATRADE_RUNS_ROOT/reports/m9_predictions.parquet` | existence-only + semantic check |
| m9_backtest_metrics | `$ALPHATRADE_RUNS_ROOT/reports/m9_backtest_metrics.json` | `m9_backtest_metrics.schema.json` |
| m9_backtest_metrics_md | `$ALPHATRADE_RUNS_ROOT/reports/m9_backtest_metrics.md` | existence-only |

All items are required. Profile defined in `src/alphatrade/schemas/contracts_manifest.yaml` under `m9`.

`m9_infer_metrics.json` records the actual `batch_size` and JAX backend used for inference (`jax_backend`) so GPU runs can be audited from artifacts.
