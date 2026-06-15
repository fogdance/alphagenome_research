# M10 Evaluation + Calibration Contract

**Status**: Active  
**Created**: 2026-06-15  
**Depends on**: M9 champion bundle + offline inference  

---

## 1. Objective

M10 adds a reliable evaluation and calibration layer around the current AlphaTrade champion.

The goal is to answer, with reproducible reports:

1. Are predictions calibrated?
2. Do predictions beat simple naive baselines?
3. Is there any directional value after costs?
4. Which horizon, symbol, or quantile is broken?
5. Can future champion selection use more than `pinball_loss.overall`?

AlphaTrade is still development-stage. M10 reports are evaluation and productization gates only, not trading-readiness claims.

M9/M10 backtest outputs are lightweight handoff checks, not full execution simulators.

---

## 2. Inputs

M10 consumes M9 predictions and processed M1 bars.

| Input | Path | Contract |
|---|---|---|
| M9 predictions | `$ALPHATRADE_RUNS_ROOT/reports/m9_predictions.parquet` | Wide format from M9 |
| Processed bars | `data/processed/m1_f8/{symbol}/bars.parquet` | 8D M1 bars |
| Bundle manifest | `$ALPHATRADE_RUNS_ROOT/reports/m9_model_bundle_manifest.json` | M9 provenance |

M9 predictions must remain wide format:

```text
symbol, eob, model_version,
h1_q10, h1_q30, h1_q50, h1_q70, h1_q90,
h5_q10, h5_q30, h5_q50, h5_q70, h5_q90,
h20_q10, h20_q30, h20_q50, h20_q70, h20_q90,
h60_q10, h60_q30, h60_q50, h60_q70, h60_q90
```

---

## 3. Target Alignment

For every `(symbol, eob)` row in `m9_predictions.parquet`, M10 aligns to the matching row in:

```text
data/processed/m1_f8/{symbol}/bars.parquet
```

Realized target definition:

```text
realized_h = log(close[t+h] / close[t])
```

where `t` is the bars row matching `eob`.

Rows are dropped when the future target is unavailable, for example near the tail of each symbol's bar series.

M10 baselines must not use future data. Rolling historical quantiles may only use past h-step returns whose target window ended at or before the current `eob`.

---

## 4. Required Reports (M10 Profile)

All required reports are defined in `src/alphatrade/schemas/contracts_manifest.yaml` under profile `m10`.

| Report | Path | Schema |
|---|---|---|
| m10_prediction_eval_metrics | `$ALPHATRADE_RUNS_ROOT/reports/m10_prediction_eval_metrics.json` | `m10_prediction_eval_metrics.schema.json` |
| m10_prediction_eval_metrics_md | `$ALPHATRADE_RUNS_ROOT/reports/m10_prediction_eval_metrics.md` | existence-only |
| m10_backtest_matrix | `$ALPHATRADE_RUNS_ROOT/reports/m10_backtest_matrix.json` | `m10_backtest_matrix.schema.json` |
| m10_backtest_matrix_md | `$ALPHATRADE_RUNS_ROOT/reports/m10_backtest_matrix.md` | existence-only |
| m10_baseline_comparison | `$ALPHATRADE_RUNS_ROOT/reports/m10_baseline_comparison.json` | `m10_baseline_comparison.schema.json` |
| m10_baseline_comparison_md | `$ALPHATRADE_RUNS_ROOT/reports/m10_baseline_comparison.md` | existence-only |

Optional diagnostic artifact:

| Artifact | Path | Notes |
|---|---|---|
| m10_eval_rows | `$ALPHATRADE_RUNS_ROOT/reports/m10_eval_rows.parquet` | Aligned predictions + realized returns |
| m10_progress | `$ALPHATRADE_RUNS_ROOT/reports/m10_progress.jsonl` | Stage-level progress log |

---

## 5. Prediction Evaluation Metrics

`m10_prediction_eval_metrics.json` records model-vs-realized metrics overall, by horizon, by symbol, and by symbol/horizon where applicable.

Required metric groups:

### 5.1 Pinball Loss

Required levels:

- `overall`
- `by_horizon`
- `by_symbol`
- `by_symbol_horizon`

### 5.2 Quantile Coverage

For q10/q30/q50/q70/q90:

- expected coverage
- observed coverage
- absolute coverage error
- coverage calibration MAE by horizon
- worst overall calibration
- worst symbol/horizon calibration

### 5.3 Quantile Crossing

Required fields:

- crossing rate
- crossing count
- pair count
- by horizon

### 5.4 Distribution Diagnostics

For realized targets and predictions:

- mean
- std
- min
- max
- p01
- p05
- p10
- p50
- p90
- p95
- p99

Diagnostics should be available overall, by horizon, and by symbol where practical.

### 5.5 IC Metrics

Using q50:

- Pearson IC
- rank IC
- by horizon
- by symbol
- materiality proxy

### 5.6 Direction Metrics

Using sign(q50) vs sign(realized):

- hit rate
- active hit rate
- long count
- short count
- near-zero count
- near-zero fraction

---

## 6. Backtest Matrix

`m10_backtest_matrix.json` records lightweight directional handoff checks across horizons, signals, thresholds, and costs.

Dimensions:

| Dimension | Values |
|---|---|
| horizons | `1, 5, 20, 60` |
| signal variants | `q50_sign`, `q30_q70_confirmed`, `uncertainty_filtered_q50` |
| costs | `0, 1, 2, 5 bps` |
| thresholds | `0`, `0.5sigma`, `1sigma` |

Required metrics:

- gross_return
- net_return
- annualized_return_if_available
- max_drawdown
- turnover
- hit_rate
- avg_trade_return
- trade_count

`annualized_return_if_available` may be null until a full execution calendar is implemented.

Backtest outputs must clearly state that M9/M10 backtests are not full execution simulators.

---

## 7. Naive Baselines

`m10_baseline_comparison.json` compares the current model against naive baselines on common rows.

Required baselines:

| Baseline | Definition |
|---|---|
| `zero_return_quantile` | all quantile predictions equal `0.0` |
| `rolling_historical_quantile` | past-only rolling h-step return quantiles |
| `rolling_median_direction` | past-only rolling median used for direction/backtest |

The comparison must include:

- pinball overall and by horizon
- coverage calibration MAE
- IC/rank IC
- direction hit rate
- selected backtest net_return, max_drawdown, and turnover

Positive pinball improvement means:

```text
baseline_pinball - model_pinball > 0
```

---

## 8. Reproduction

Use the CUDA environment:

```bash
export RUN="conda run -n alphatrade_cuda12 env -u LD_LIBRARY_PATH"
export ALPHATRADE_RUNS_ROOT="$(pwd)/../alphatrade_runs/m10_evalcal_$(date +%Y%m%d_%H%M%S)"
export REPORTS="$ALPHATRADE_RUNS_ROOT/reports"
export CHECKPOINTS="$ALPHATRADE_RUNS_ROOT/checkpoints"
export ARTIFACTS="$ALPHATRADE_RUNS_ROOT/artifacts"
```

M9 export:

```bash
$RUN python src/alphatrade/scripts/export_model_bundle.py \
  --output-root "$ALPHATRADE_RUNS_ROOT"
```

M9 batch inference:

```bash
$RUN python src/alphatrade/scripts/batch_infer_offline.py \
  --bundle "$ARTIFACTS/model_bundle/<model_version>" \
  --data-dir data/processed/m1_f8 \
  --symbols <comma-separated-symbols> \
  --start 2024-01-01 \
  --end 2026-01-01 \
  --output-root "$ALPHATRADE_RUNS_ROOT"
```

M9 backtest handoff:

```bash
$RUN python src/alphatrade/scripts/backtest_predictions.py \
  --predictions "$REPORTS/m9_predictions.parquet" \
  --data-dir data/processed/m1_f8 \
  --horizon 20 \
  --quantile 0.5 \
  --cost-bps 0.0 \
  --output-root "$ALPHATRADE_RUNS_ROOT"
```

M10 evaluation:

```bash
$RUN python src/alphatrade/scripts/build_m10_prediction_eval.py \
  --predictions "$REPORTS/m9_predictions.parquet" \
  --data-dir data/processed/m1_f8 \
  --horizons 1,5,20,60 \
  --quantiles 0.1,0.3,0.5,0.7,0.9 \
  --out-prefix m10 \
  --cost-bps-list 0,1,2,5 \
  --thresholds 0,0.5sigma,1sigma \
  --output-root "$ALPHATRADE_RUNS_ROOT"
```

Validation:

```bash
$RUN python src/alphatrade/scripts/validate_reports_schema.py \
  --profile m9 --strict \
  --output-root "$ALPHATRADE_RUNS_ROOT"

$RUN python src/alphatrade/scripts/validate_reports_schema.py \
  --profile m10 --strict \
  --output-root "$ALPHATRADE_RUNS_ROOT"
```

Tests:

```bash
$RUN python -m pytest src/alphatrade/tests -q
```

---

## 9. Validation Rules

M10 strict validation currently enforces schema and required report existence through `validate_reports_schema.py`.

M9 semantic validation remains responsible for checking `m9_predictions.parquet`:

- required columns exist
- no duplicate `(symbol, eob, model_version)`
- prediction columns are numeric
- no NaN/Inf in prediction columns
- quantiles are non-crossing per horizon

M10 model-quality checks are diagnostic unless promoted by a later profile. Recommended diagnostic checks:

- model pinball worse than `zero_return_quantile`
- model pinball worse than `rolling_historical_quantile`
- coverage calibration MAE above `0.05`
- max abs IC below `0.005`
- prediction q90/q10 scale more than 10x realized p99/p01 scale
- any horizon has q50 coverage abs error above `0.20`

When these checks fail, reports should use severity labels such as:

- `PASS`
- `WARN`
- `FAIL_MODEL_QUALITY`

They should not be described as production trading gates.

---

## 10. Acceptance Checklist

M10 is accepted only when:

- [ ] M9 export produces `m9_model_bundle_manifest.json`.
- [ ] M9 inference produces wide-format `m9_predictions.parquet`.
- [ ] M9 inference records `batch_size` and `jax_backend`.
- [ ] M9 predictions pass schema and semantic validation.
- [ ] M9 backtest handoff report is generated.
- [ ] M10 prediction evaluation reports are generated.
- [ ] M10 baseline comparison reports are generated.
- [ ] M10 backtest matrix reports are generated.
- [ ] `validate_reports_schema.py --profile m9 --strict` passes.
- [ ] `validate_reports_schema.py --profile m10 --strict` passes.
- [ ] `python -m pytest src/alphatrade/tests -q` passes.
- [ ] M10 markdown reports state that AlphaTrade is development-stage.
- [ ] M10 markdown reports state that M9/M10 backtests are not full execution simulators.
- [ ] M10 reports state whether the champion beats zero and rolling baselines.
- [ ] M10 reports identify the worst horizon/symbol/quantile calibration.
- [ ] M10 reports state whether IC/rank IC is materially different from zero.

---

## 11. Known Result From Full M10 Run

The full M10 run at:

```text
../alphatrade_runs/m10_evalcal_full_20260615_065041
```

passed M9 and M10 schema gates and local tests, but rejected the current champion on substantive model quality.

Summary:

| Metric | Value |
|---|---:|
| model pinball | `0.0030519347` |
| zero_return_quantile pinball | `0.0007933594` |
| rolling_historical_quantile pinball | `0.0006104896` |
| model coverage MAE | `0.2144233` |
| rolling_historical_quantile coverage MAE | `0.0124525` |
| max abs Pearson IC | `0.002718` |
| max abs rank IC | `0.002462` |

Conclusion:

- The current champion remains rejected / development-stage.
- The next milestone should focus on target/output scale and quantile calibration.
- Do not prioritize increasing model size before fixing calibration and baseline gaps.
