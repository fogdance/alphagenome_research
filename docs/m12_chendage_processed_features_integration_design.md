# M12-CHG-FEATURES Chendage Processed Features Integration Design

**Status**: Draft  
**Created**: 2026-06-18  
**Depends on**: M10/M11 evaluation gates and Chendage processed export boundary  

---

## 1. Objective

M12 evaluates how to use the processed market-state features from
`chendage-signal-engine` as an AlphaTrade input source.

This sprint is:

```text
M12-CHG-FEATURES
```

It is not:

```text
M12-CDG-TEACHER
```

The goal is not to make the model tradable. AlphaTrade is still in active
development. M12 should answer whether the added processed features improve
evaluation quality under the existing M10/M11 gates:

1. Does the augmented feature set beat the current 8D baseline?
2. Does calibration improve without post-hoc rescue?
3. Does the model beat zero and rolling historical baselines?
4. Which processed feature groups help or hurt?
5. Can the integration remain rule-free and reproducible?

Scope split:

| Scope | Status | Contract |
|---|---|---|
| `M12-CHG-FEATURES` | in scope | Use `chendage_signal.processed` numeric features for AlphaTrade feature ablation |
| `M12-CDG-TEACHER` | out of scope | Optional future branch for OOS reports using validated A/D/EntryPlan teacher outputs |

Passing `M12-CHG-FEATURES` proves only that Chendage processed market-state
features may add predictive value. It does not prove that Chendage A-class
signals, ratings, entry plans, or trading rules are profitable.

---

## 2. Chendage Boundary Review

Reviewed source repository:

```text
/home/v/Documents/work/chendage-signal-engine
```

Reviewed commit:

```text
6eb7a8d fix: harden processed export boundaries
```

Current conclusion: the processed-data entrypoint is suitable for AlphaTrade
integration, subject to using only the new processed API.

Accepted entrypoints:

```text
python -m chendage_signal.processed.export
```

```python
from chendage_signal.processed import (
    ProcessedExportRequest,
    export_processed_features,
)
```

Rejected entrypoint for AlphaTrade:

```text
chendage-signal
```

The legacy mixed CLI remains a rule/review orchestration CLI. It must not be
used by AlphaTrade data generation.

Validation performed:

| Check | Result |
|---|---|
| Chendage tests | `173 passed` |
| Real CSV processed export | 345 rows exported |
| Runtime import boundary | `import chendage_signal.processed` did not load mixed rule/review modules |
| Rule-only output fields | none found in top-level rows or `feature_vector` |

The three prior review issues are fixed:

| Prior issue | Current status |
|---|---|
| `timestamps` mode allowed out-of-range timestamps | fixed with evaluation-range validation |
| Legacy mixed CLI exposed processed export | fixed; old `export-processed-features` removed |
| CSV multi-symbol inference was too permissive | fixed; multi-symbol CSV requires explicit contract symbol |

---

## 3. Source Data Contract

Chendage processed export produces point-in-time snapshots using only bars at or
before `as_of`.

Current source schema identifiers:

```text
processed_market_snapshot.v1
processed_feature_vector.v1
```

AlphaTrade should consume only:

1. identity/provenance fields needed for alignment
2. raw OHLCV fields needed for target construction and validation
3. deterministic numeric feature columns from `feature_vector`

AlphaTrade must not consume:

| Field class | Examples |
|---|---|
| rule ratings | `rating`, `score`, `action` |
| hard blocks | `hard_block_codes`, `is_prohibited`, `blocks` |
| candidate labels | `candidate_reasons`, setup direction labels |
| explanatory reasons | `reasons`, `positive_reasons`, `negative_reasons` |
| human annotation | `human_annotation`, `review_status`, `human_*` |
| outcome review | `outcome`, `mfe_points`, `mae_points` |

---

## 4. AlphaTrade Current Data Contract

AlphaTrade currently trains and infers from:

```text
data/processed/m1_f8/{symbol}/bars.parquet
data/processed/m1_f8/{symbol}/index_train.parquet
data/processed/m1_f8/{symbol}/index_val.parquet
data/processed/m1_f8/{symbol}/index_test.parquet
```

The current canonical feature profile is fixed at 8 features:

```text
ret_1m
hl_range
co_change
vol_log1p
pos_log1p
minute_sin
minute_cos
is_session_open
```

These are defined in:

```text
src/alphatrade/data_pipeline/feature_schema.py
```

Training and inference currently import this global `FEATURE_COLS`, and
`train_m4_alphatrade.py` sets `num_features=8`. Therefore, Chendage integration
must not overwrite `m1_f8` in place.

M12 requires AlphaTrade to support dynamic feature profiles before training.
The profile must propagate through data loading, window cache, training, bundle
metadata, and offline inference.

Required propagation points:

| Component | Required M12 behavior |
|---|---|
| dataset config | declares `feature_profile_id`, `feature_cols`, `feature_dim`, and `processed_root` |
| feature schema | resolves columns by profile, with `m1_f8` as the default control profile |
| window cache | fingerprint includes `feature_profile_id`, `feature_cols`, `processed_root`, and scaler hash |
| training | `num_features` comes from dataset/profile, not a hardcoded `8` |
| model metadata | records `feature_profile_id`, `feature_cols`, and `feature_dim` |
| bundle metadata | records feature profile and scaler metadata |
| M9 inference | uses bundle feature metadata and the selected data-dir profile |
| M10/M11 evaluation | prediction schema remains unchanged; `--data-dir` points to the matching processed root |

---

## 5. Recommended Integration Shape

Use a new dataset variant instead of modifying the existing baseline.

Recommended output root:

```text
data/processed/m12_chendage_f{N}/{symbol}/
```

Per-symbol files:

```text
bars.parquet
index_train.parquet
index_val.parquet
index_test.parquet
feature_manifest.json
source_manifest.json
```

`bars.parquet` should keep the AlphaTrade-required columns:

```text
eob, open, high, low, close, volume, position, symbol, segment_id
```

and add a versioned feature set:

```text
ret_1m
hl_range
co_change
vol_log1p
pos_log1p
minute_sin
minute_cos
is_session_open
chg.<processed feature columns>
```

Use `chg.` as a namespace prefix to make external features auditable and to
avoid collisions with native AlphaTrade columns.

Do not change label construction:

```text
y_h = log(close[t+h] / close[t])
```

The index files should keep the current AlphaTrade window loading structure:

```text
t, x_start, x_end, eob, segment_id, y_h1, y_h5, y_h20, y_h60, split
```

`m1_f8` remains immutable. M12 builders must write only to new `m12_*` roots.

---

## 6. Feature Profile Strategy

Do not feed all 100 Chendage numeric features directly in the first integration.
The current model has already failed M10/M11 quality gates, and increasing input
width without controlled baselines will make attribution harder.

Use staged profiles:

| Profile | Purpose | Feature set |
|---|---|---|
| `m12_base8` | frozen control | current 8D `m1_f8` |
| `m12_chg_core` | first integration | base8 + compact Chendage groups |
| `m12_chg_full` | diagnostic upper bound | base8 + all stable Chendage numeric features |
| `m12_chg_ablation_*` | attribution | one group removed at a time |

Recommended first compact groups:

| Group | Examples | Reason |
|---|---|---|
| daily trend | `daily_trend_*`, `daily_trend_confidence`, swing counts | higher-timeframe context |
| H1 distances/flags | active support/resistance distances, near flags | structure context |
| M5 MACD facts | `m5_dif`, `m5_dea`, `m5_hist`, cross one-hots | medium-horizon momentum |
| minute behavior numerics | volume ratio, recent price/OI change, recent high/low | short-horizon microstructure |

Feature exclusions for first pass:

| Exclude | Reason |
|---|---|
| raw external price levels without normalization | scale varies by symbol and contract |
| duplicate current OHLCV from Chendage | already present in AlphaTrade bars |
| human-readable string state columns | use one-hot numeric `feature_vector` only |
| rule/candidate/review outputs | violates processed-data boundary |

---

## 7. Symbol And Contract Mapping

AlphaTrade and Chendage may use different symbol conventions. This must be
explicit before any feature rows are merged.

AlphaTrade currently uses continuous symbols such as:

```text
DCE.JM
SHFE.AG
```

Chendage processed export may use concrete contract symbols or a caller-provided
continuous-symbol CSV. M12 must not infer this silently.

Required manifest fields:

```text
source_symbol
alphatrade_symbol
chendage_export_symbol
symbol_type: continuous | contract
contract_map
roll_policy
timezone
input_files
input_hashes
chendage_commit
source_schema_versions
```

Mapping rules:

| Rule | Contract |
|---|---|
| one AlphaTrade symbol to one feature source per timestamp | required |
| contract rollover policy | recorded in `source_manifest.json` |
| main-contract switch date | recorded when applicable |
| mismatched symbol at same `as_of` | fail |
| implicit multi-symbol CSV inference | prohibited |
| feature from different contract than label source | fail unless explicitly documented as a controlled experiment |

The safest first implementation is a one-symbol smoke run where AlphaTrade bars
and Chendage export are both generated from the same concrete or continuous
source file.

---

## 8. Normalization Policy

AlphaTrade should normalize Chendage-derived features inside AlphaTrade, not in
the Chendage export.

Recommended policy:

| Feature type | Transform |
|---|---|
| one-hot and boolean flags | keep as 0/1 float32 |
| ratios and changes | winsorize on train split, then z-score |
| price distances in points | divide by current close or rolling ATR-like scale before z-score |
| raw prices | drop initially, or convert to relative distance |
| count features | `log1p`, then train-split z-score |

Scaler fitting rules:

1. fit on train split only
2. persist scaler metadata under dataset root
3. include feature profile and scaler fingerprint in reports
4. fail if val/test transformation requires fitting new state

---

## 9. Implementation Plan

### Phase 0: Freeze Contracts

Add an AlphaTrade-side feature profile layer:

```text
src/alphatrade/data_pipeline/feature_profiles.py
```

It should define:

```text
profile_id
feature_cols
feature_dim
source_schema_versions
normalization_policy
```

The existing `FEATURE_COLS` remains the default `m1_f8` control profile.

Required code changes:

| Area | Required change |
|---|---|
| dataset config | add `feature_profile_id`, `feature_cols`, `feature_dim`, `processed_root` |
| dataloader/window cache | load feature columns from profile, not global-only `FEATURE_COLS` |
| cache fingerprint | include profile id, feature columns, processed root, scaler hash |
| training | set `AlphaTradeConfig.num_features` from profile |
| train metrics | record profile id, feature columns, feature dim, scaler hash |
| model bundle | persist profile id, columns, dim, scaler metadata |
| inference | verify bundle feature profile matches data-dir profile before prediction |

### Phase 1: Export Chendage Features

Generate Chendage processed feature rows per contract symbol or continuous
symbol source.

Example command:

```bash
PYTHONPATH=/home/v/Documents/work/chendage-signal-engine/src \
conda run -n chen python -m chendage_signal.processed.export \
  --source csv \
  --csv <input.csv> \
  --symbol <CONTRACT_SYMBOL> \
  --from <YYYY-MM-DD> \
  --to <YYYY-MM-DD> \
  --out <staging>/processed_features.jsonl
```

For sparse exports, AlphaTrade must provide timestamps:

```bash
python -m chendage_signal.processed.export \
  --mode timestamps \
  --timestamps-file <as_of_timestamps.txt>
```

Do not use candidate generation.

### Phase 2: Build AlphaTrade M12 Bars

Add a builder script:

```text
src/alphatrade/scripts/build_m12_chendage_features.py
```

Responsibilities:

1. load current `m1_f8/{symbol}/bars.parquet`
2. load Chendage processed JSONL/CSV/parquet
3. align rows by `(symbol, eob/as_of)`
4. derive selected `chg.*` feature columns
5. apply AlphaTrade-side train-only normalization to Chendage numeric features
6. write `data/processed/m12_chendage_f{N}/{symbol}/bars.parquet`
7. reuse or rebuild sample indexes with unchanged label definitions
8. write manifests, generated dataset configs, generated universe files, and validation reports

Alignment rules:

| Rule | Contract |
|---|---|
| timestamp key | `bars.eob == chendage.as_of` after timezone normalization |
| duplicate key | fail |
| missing Chendage row | configurable: fail for strict run, fill only for smoke |
| extra Chendage row | allowed but reported |
| future leakage | no Chendage feature may be computed from bars after `as_of` |

Common-row control is mandatory:

```text
base8_control_common_rows and chg_core must use identical symbols,
timestamps, split rows, and evaluation rows.
```

M12 comparisons must not compare the old `m1_f8` baseline against a Chendage
dataset with different rows. The builder should materialize a `common_rows`
index and produce both:

```text
data/processed/m12_common_base8/{symbol}/
data/processed/m12_chendage_f{N}/{symbol}/
```

from the same row set.

Implemented entrypoint:

```bash
python src/alphatrade/scripts/build_m12_chendage_features.py \
  --base-dir data/processed/m1_f8 \
  --chendage-input <processed_features.jsonl> \
  --truncated-chendage-input <processed_features_truncated.jsonl> \
  --mutated-chendage-input <processed_features_mutated.jsonl> \
  --symbol-map <symbol_map.json> \
  --symbols DCE.JM \
  --exclude-feature-groups h1 \
  --output-dir data/processed/m12_chendage_fN \
  --control-output-dir data/processed/m12_common_base8
```

The generated candidate and control roots each contain `dataset_config.yaml`
and `universe.yaml`, so they can be passed directly to
`check_m2_dataloader.py`, `train_m4_alphatrade.py`, and `eval_m4_fast.py`.
Group ablations such as `chg_core_no_h1`, `chg_core_no_m5`, and
`chg_core_no_minute_behavior` are generated by passing
`--exclude-feature-groups h1`, `--exclude-feature-groups m5`, or
`--exclude-feature-groups minute_behavior` with a matching
`--feature-profile-id`.

### Phase 3: Train as Ablation, Not Promotion

Use the builder-generated dataset configs:

```text
data/processed/m12_common_base8/dataset_config.yaml
data/processed/m12_chendage_f{N}/dataset_config.yaml
```

Committed sweep template:

```text
configs/sweep/m12_chendage.yaml
```

It uses per-experiment `dataset_config` entries, so `base8_control_common_rows`,
`chg_core`, ablations, and `chg_full_diagnostic` can be evaluated inside one
M5-style sweep after the corresponding M12 data roots have been generated.

Minimum experiments:

| exp_id | Description |
|---|---|
| `base8_control_common_rows` | current 8D features on the exact M12 common rows |
| `chg_core` | compact Chendage features |
| `chg_core_no_h1` | remove H1 structure group |
| `chg_core_no_m5` | remove M5 MACD group |
| `chg_core_no_minute_behavior` | remove minute behavior group |
| `chg_full` | all stable numeric Chendage features, diagnostic only |

Champion selection must still go through M10/M11-style evaluation. A lower train
loss alone is not acceptable.

---

## 10. Required Validation

Add a new M12 data-quality report before training:

```text
$REPORTS/m12_chendage_feature_contract.json
$REPORTS/m12_chendage_feature_contract.md
```

Required checks:

| Check | Severity |
|---|---|
| source schema versions match expected values | fail |
| no rule-only fields in source rows | fail |
| no duplicate `(symbol, as_of)` | fail |
| all selected features numeric after transformation | fail |
| no NaN/Inf after fill and normalization | fail |
| feature rows align to bars rows | fail |
| missing feature rate by symbol | fail if above threshold |
| feature profile fingerprint recorded | fail |
| train-only scaler fit recorded | fail |
| feature distribution by split reported | warn/fail depending on drift |
| common-row control row set recorded | fail |
| truncated-input causality test | fail |

Truncated-input causality test is a hard validation requirement:

1. choose representative `(symbol, as_of)` samples from train, val, and test
2. export `snapshot_full(as_of=t)` from the full available input
3. export `snapshot_truncated(as_of=t)` after truncating source input to `<= t`
4. require both `feature_vector` outputs to match exactly or within configured floating tolerance
5. mutate input rows after `t` and require the `t` feature vector to remain unchanged

Any mismatch fails M12 data validation because it indicates hidden future
dependency or non-deterministic feature generation.

Existing validation should also run:

```text
check_m1_data_contract.py
check_m2_dataloader.py
validate_reports_schema.py
```

M12 schema/profile validation is:

```bash
python src/alphatrade/scripts/validate_reports_schema.py --profile m12 --strict
```

The M12 semantic gate checks contract `overall_status`, failed checks,
feature profile width, train-only scaler metadata, rule-only field absence,
truncated/mutated causality, symbol build status, common rows/samples, feature
coverage, and candidate/control root existence.

M10/M11 evaluation remains mandatory for any trained candidate.

---

## 11. Experiment Gates

M12 should be considered useful only if it improves evaluation quality, not just
training loss.

M12 outcomes use three severity labels:

| Status | Meaning |
|---|---|
| `PASS` | candidate beats both base8 control and rolling historical baseline under common-row evaluation |
| `PROMISING` | candidate clearly improves over base8 control but still loses to rolling baseline |
| `FAIL` | candidate does not improve over base8 control, calibration/IC is not better, or only one backtest cell improves |

Minimum `PASS` requirements:

| Gate | Requirement |
|---|---|
| schema and data validation | all pass |
| M10 pinball | better than `base8_control_common_rows` on common rows |
| rolling baseline | beats `rolling_historical_quantile` pinball on common rows |
| coverage MAE | `<= 0.05` or materially better than base8 control |
| IC/rank IC | `max_abs_rank_ic >= 0.005` and not driven by one symbol only |
| post-cost backtest | not sufficient alone; only a secondary diagnostic |
| ablation | at least one feature group shows attributable value |

`PROMISING` requirements:

| Gate | Requirement |
|---|---|
| base8 control | clearly improved on common rows |
| rolling baseline | still worse than rolling baseline |
| calibration/IC | at least one material improvement vs base8 control |
| next step | allowed to continue to M12.1, not promotion |

`FAIL` triggers:

| Trigger | Meaning |
|---|---|
| worse than base8 control | reject |
| no calibration or IC improvement | reject |
| only one backtest cell improves | reject |
| schema/data validation fails | stop before training or reject trained run |

If the model still loses to rolling historical baselines, the correct conclusion
is that the candidate remains rejected or at most `PROMISING`, not promoted.

---

## 12. Open Design Choices

| Topic | Recommendation |
|---|---|
| input frequency | keep AlphaTrade on 1m first; use Chendage features aligned to 1m `as_of` |
| non-1m inputs | defer until the 1m integration has a clean baseline |
| feature count | start compact; full 100-feature profile only as diagnostic |
| dependency model | call Chendage exporter out-of-process first; avoid vendoring |
| data storage | store generated processed rows under `../alphatrade_runs` or staging, final AlphaTrade bars under `data/processed/m12_*` |
| rule labels | do not ingest |
| candidate timestamps | only caller-supplied timestamps; no rule-generated candidates |

---

## 13. Risks

| Risk | Mitigation |
|---|---|
| accidental rule leakage | static field denylist plus runtime output checks |
| hidden future leakage | compare Chendage snapshots with truncated input data in tests |
| feature scale instability | train-only scaler plus split distribution report |
| overfitting from wide feature set | staged compact/full/ablation profiles |
| inference profile mismatch | bundle `feature_profile_id` and `feature_cols` into model metadata; fail rather than silently using old data roots |
| stale window cache | include feature profile and processed root in cache fingerprint |
| provenance ambiguity | persist Chendage commit, schema versions, command, input hashes |

---

## 14. Recommended Execution Order

M12 should execute in this order:

1. implement `feature_profiles.py`
2. make AlphaTrade support `m1_f8` and `m12_chendage_fN` profiles end to end
3. implement `build_m12_chendage_features.py`
4. run one-symbol `chg_core` dataset generation only, without training
5. generate `m12_chendage_feature_contract.json/md`
6. add schema and semantic validation, including truncated-input causality test
7. run `check_m1_data_contract.py`
8. run `check_m2_dataloader.py`
9. verify the dataloader really reads `F > 8`
10. run smoke training for `base8_control_common_rows` and `chg_core`
11. run full 3-seed sweep for control, core, ablations, and diagnostic full profile
12. evaluate with M10/M11-style reports and do not promote directly

---

## 15. Recommended Next Step

Implement M12 as a data/feature experiment, not as a model promotion.

First concrete patch:

1. add `feature_profiles.py` - implemented
2. add `build_m12_chendage_features.py` - implemented
3. propagate feature profiles through dataloader, window cache, training, bundle metadata, and inference checks - implemented for M2 check, M4 train/eval, M5 sweep planning, M9 bundle/inference
4. add `m12_chendage_feature_contract` report/schema - implemented
5. add common-row and truncated-input causality tests - implemented with synthetic regression coverage
6. generate one-symbol M12 dataset and run `check_m2_dataloader.py` - implemented with synthetic smoke data

Only after that should training start.

Current local validation:

```text
/home/v/miniconda3/envs/alphatrade_cuda12/bin/python -m pytest src/alphatrade/tests -q
163 passed
```

Synthetic one-symbol M12 smoke validation:

```text
root: /tmp/alphatrade_m12_smoke_20260618_160256

build_m12_chendage_features.py:
  overall_status: PASS

validate_reports_schema.py --profile m12 --strict:
  schema: 2/2
  semantic: 15/15

check_m2_dataloader.py candidate:
  feature profile: m12_chg_core
  observed shape: (20, 12)
  valid: 13/13

check_m2_dataloader.py base8 control:
  feature profile: m12_base8_control_common_rows
  observed shape: (20, 8)
  valid: 13/13
```
