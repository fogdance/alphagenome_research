# M9 Champion Bundle + Offline Batch Inference Contract

**Status**: Active
**Created**: 2026-03-03
**Depends on**: M7/M8 (iteration loop with leaderboard + regression gate)

---

## 1. Objective

Select the champion model from the M5 leaderboard, package it into a reproducible bundle, and provide offline batch inference that outputs `predictions.parquet`.

## 2. Champion Selection Rule

| Step | Rule |
|------|------|
| Primary | Lowest `primary_mean` on leaderboard (experiments[0], already sorted) |
| Tiebreaker | Lowest `primary_std` (stability) |
| Exported seed | `best_run_id` from the winning experiment (best individual seed) |

Override: `--exp-id` / `--run-id` flags on the export script.

## 3. Bundle Format

Self-contained directory under `artifacts/model_bundle/<model_version>/`:

| File | Description |
|------|-------------|
| `best/` | Flax checkpoint dir (copied from checkpoints/m5/) |
| `artifacts.json` | Training metadata (copied) |
| `model_config.json` | AlphaTradeConfig as JSON (extracted for readability) |
| `bundle_manifest.json` | Full provenance record |

Also writes `reports/m9_model_bundle_manifest.json` (same content, validated by m9 profile).

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

## 6. Reproduction

```bash
# 1. Export champion bundle
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/export_model_bundle.py

# 2. Run batch inference (smoke: 2 symbols × small range)
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/batch_infer_offline.py \
  --bundle artifacts/model_bundle/alphatrade_v0.2_batch_256_9169f783 \
  --data-dir data/processed/m1_f8 \
  --symbols DCE.JM,SHFE.AG \
  --start 2024-01-02 --end 2024-01-04 \
  --output reports/m9_predictions.parquet \
  --smoke

# 3. Validate M9 gate
conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/validate_reports_schema.py --profile m9 --strict
```

## 7. Required Reports (M9 Profile)

| Report | Path | Schema |
|--------|------|--------|
| m9_model_bundle_manifest | `reports/m9_model_bundle_manifest.json` | `m9_model_bundle_manifest.schema.json` |
| m9_infer_metrics | `reports/m9_infer_metrics.json` | `m9_infer_metrics.schema.json` |
| m9_infer_metrics_md | `reports/m9_infer_metrics.md` | existence-only |
| m9_predictions_parquet | `reports/m9_predictions.parquet` | existence-only + semantic check |

All items are required. Profile defined in `src/alphatrade/schemas/contracts_manifest.yaml` under `m9`.
