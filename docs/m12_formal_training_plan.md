# M12 Formal Training Plan

**Status**: Active
**Sprint**: M12-CHG-FEATURES Formal Sweep
**Scope**: AlphaTrade development-stage research evaluation

AlphaTrade is not trading-ready. M12 formal training only evaluates whether
Chendage processed numeric market-state features add predictive value on common
rows. It must not be used as tradability evidence.

## Objective

M12 compares a common-row 8D AlphaTrade control against Chendage processed
feature candidates. The comparison must use M10/M11-style evaluation after
training; M5 validation pinball alone is not enough for promotion.

## Formal Datasets

The formal DB builder generates one control and six Chendage candidates:

| Experiment | Dataset Root | Purpose |
|---|---|---|
| `base8_control_common_rows` | `m12_common_base8_formal` | 8D control rebuilt on candidate common rows |
| `chg_core_common_rows` | `m12_chg_core_formal` | Base8 plus core Chendage processed features |
| `chg_core_no_daily` | `m12_chg_core_no_daily_formal` | Core ablation without daily group |
| `chg_core_no_h1` | `m12_chg_core_no_h1_formal` | Core ablation without H1 group |
| `chg_core_no_m5` | `m12_chg_core_no_m5_formal` | Core ablation without M5 group |
| `chg_core_no_minute_behavior` | `m12_chg_core_no_minute_behavior_formal` | Core ablation without minute-behavior group |
| `chg_full_diagnostic` | `m12_chg_full_diagnostic_formal` | Full processed numeric feature diagnostic only |

All candidates must use the Chendage processed numeric feature-vector boundary
only. Do not consume rating, score, grade, action, candidate, reason,
hard-block, human annotation, outcome review, entry plan, trade plan, teacher,
or label fields.

## Builder Contract

`src/alphatrade/scripts/build_m12_chendage_db_formal.py` owns:

- DB continuous-bar export by symbol.
- Chendage processed snapshot export by symbol.
- Train-only scaler fitting per formal dataset.
- Common-row base8 control rebuild.
- Truncated and mutated future causality checks.
- M12 feature contract reports.
- Generated M5 sweep config.

Parallelism is controlled by `--jobs` and applies at symbol granularity.

`--reuse-existing-processed` may be used only when the existing export exactly
matches the requested symbols, date ranges, Chendage commit, and recorded file
hashes. A mismatch must fail loudly instead of silently reusing stale inputs.

The generated formal sweep budget is explicit CLI state:

| Argument | Default |
|---|---:|
| `--sweep-max-steps` | 3000 |
| `--sweep-batch-size` | 256 |
| `--sweep-save-every` | 500 |
| `--sweep-keep-last` | 3 |
| `--sweep-seeds` | `42,43,44` |
| `--sweep-eval-split` | `val` |
| `--sweep-ckpt-step` | `best` |

The default 3000-step budget is a first formal ablation budget, not a claim of
model convergence. Longer budgets should be declared through CLI args and
recorded in the generated config/report.

## Required Commands

Build formal datasets:

```bash
python3 src/alphatrade/scripts/build_m12_chendage_db_formal.py \
  --symbols CZCE.FG,SHFE.SP,DCE.JM,SHFE.RB,CZCE.MA \
  --source-start 2017-12-01 \
  --source-end 2026-06-17 \
  --eval-start 2018-01-01 \
  --eval-end 2026-06-17 \
  --train-start 2018-01-01 \
  --train-end 2023-01-01 \
  --val-start 2023-01-01 \
  --val-end 2024-01-01 \
  --test-start 2024-01-01 \
  --test-end 2026-06-18 \
  --causality-samples-per-symbol 5 \
  --jobs 5 \
  --output-root "$RUN_ROOT" \
  --force
```

Run formal spotcheck:

```bash
python3 src/alphatrade/scripts/build_m12_formal_data_contract_spotcheck.py \
  --run-root "$RUN_ROOT" \
  --strict
```

Run the generated sweep on GPU:

```bash
conda run --no-capture-output -n alphatrade_cuda12 env -u LD_LIBRARY_PATH \
  python src/alphatrade/scripts/run_m5_sweep.py \
    --sweep-config "$RUN_ROOT/configs/m12_chendage_formal5.yaml" \
    --output-root "$FORMAL_ROOT" \
    --reports-dir "$FORMAL_ROOT/reports" \
    --checkpoints-dir "$FORMAL_ROOT/checkpoints" \
    --gpu
```

Do not silently fall back to CPU for official M12 training.

## Acceptance

Formal training is usable for model-quality review only when:

1. M12 feature contract is PASS.
2. Formal spotcheck is PASS for every dataset.
3. Truncated and mutated causality checks both PASS.
4. Candidate and control indices match by symbol, split, `eob`,
   `target_eob`, and `_m12_source_pos`.
5. M5 sweep completes all configured experiments and seeds.
6. M10/M11-style evaluation compares candidates against base8 control,
   zero-return baseline, and rolling historical quantile baseline.
7. The decision memo states PASS, PROMISING, or FAIL without claiming
   tradability.
