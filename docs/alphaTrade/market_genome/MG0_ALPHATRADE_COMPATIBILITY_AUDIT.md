# MG0-B AlphaTrade Compatibility Audit and MarketGenome Scaffold

**Status:** `PASS_WITH_DOCUMENTED_BOUNDARIES`

**Scope:** `SCAFFOLD_ONLY`

**Audited baseline:** `feature/market-genome-mg0` at
`72df1cc1e340f85447f2d4a0c488a15a20c0b2a1`

**Training/GPU:** `DEFERRED_BY_USER`; none executed in MG0-B

**JAX boundary:** one legacy control test imported JAX with a forced CPU
backend; no JAX computation or model runtime was executed

The machine-readable source for this audit is
`MG0_ALPHATRADE_COMPATIBILITY_AUDIT.json`. Its resolved scaffold config hash is
`927c4775b7a857b2346dbfbe44c101cac9d8471b795a8f0733a02201dab9d451`.

## Decision

MarketGenome is a separate model line. The current AlphaTrade v0.2 fixed-window
model, training launcher, inference loader, and historical M11 closure must not
be patched in place. Existing ingestion/provenance, causal primitives,
feature-profile contracts, M8 resume discipline, bundle hashing, M10 metric
kernels, and report governance contain reusable parts, but most require an
explicit adapter and a new versioned contract.

This milestone adds only an import-safe namespace, status/config contract,
independent `mg0b` schema/profile, strict semantic checks, tests, and this audit.
It does not add a backbone, ontology, long-sequence store, tensor model schema,
head, loss, trainer, checkpoint loader, inference pipeline, or evaluator.

## Evidence Boundary

- Current behavior is grounded in inspected repository code and exact locators.
- M12 numerical conclusions are read from the authoritative historical run;
  they were not recomputed.
- The MarketGenome package import test loads no JAX/model module. One legacy
  productization-control test imported JAX with `CUDA_VISIBLE_DEVICES` empty;
  it did not execute a JAX computation or model.
- No training, inference, checkpoint, or GPU command was run.
- A compatibility classification is a design boundary, not an implemented
  MarketGenome capability.
- User-provided source documents under `docs/market_genome/` remain untracked
  and are not modified by this milestone.

## M12 Formal Conclusion

Authoritative run root:
`/data/alphatrade/runs/m12_chg_formal5_20260626_parallel`.

| Dimension | Observed status |
|---|---|
| Engineering feature contract | `PASS` |
| Formal data-contract spotcheck | `PASS` |
| M12 schema validation | `PASS` (`2/2` schema, `15/15` semantic) |
| Formal sweep completion | `PASS` (7 experiments x seeds 42/43/44, 21 historical CUDA runs, 3000 steps each) |
| Model quality | `FAIL_MODEL_QUALITY` |
| Formal decision | `FAIL` |
| Promotion | `REJECTED` |

These statuses are intentionally not collapsed into one `overall` field. The
M12 validator reports `model_quality_status: null`; it is a data-contract gate,
not a model-quality gate. The M11 closure for the evaluated M12 models is
`9/9` schema but `8/9` semantic. The failed semantic is
`output_calibration_problem_confirmed`; the absence of the old >10x scale
symptom does not make the model pass quality.

Three-seed validation means:

| Experiment | Primary mean |
|---|---:|
| `base8_control_common_rows` | `0.0012521361622341064` |
| `chg_core_no_m5` (best processed candidate) | `0.0013347068852399642` |
| `chg_core_common_rows` | `0.0015268211131357744` |

Every processed-feature candidate is worse than the 8D control. For the base8
model, raw pinball is `0.0011936268421304148`, the rolling historical baseline
is `0.0008602891155530023`, and calibrated pinball is
`0.0008975829278504696`. Calibration improves coverage but still loses on
pinball. Therefore M12 trusts the engineering/data-contract path while rejecting
processed-feature promotion. It neither proves model quality nor disproves the
separate MarketGenome hypothesis.

Authoritative artifact SHA256 values:

| Artifact | SHA256 |
|---|---|
| `reports/M12_CHG_FORMAL_DECISION_MEMO.md` | `04e5df55d8e74c983a5bb838cf7040af772a5215fbb53401222688a8fdff3afc` |
| `reports/m12_formal_decision_summary.json` | `9e3cc22a3646a9a1e8945631628c51dbfbce96b6feaea8795f90e9f61ce9cfb5` |
| `reports/m12_schema_validation.json` | `98d9b73f8ad877622f21fbcefcfcdc631e54cf88b9b5d9280387701cfc8eec0c` |
| `reports/m12_formal_data_contract_spotcheck.json` | `4c6465e8d3f4124301f0a467b8ec4c484cb7c13b8f5a1c72d0d84b951a0e0462` |
| `reports/formal_sweep/m5_sweep_manifest.json` | `b220773b76b9ef75bc0d546d464affcbe3a8767670a6706c02a5b5f87d407b77` |

## Compatibility Matrix

### Data ingestion: `REUSE_WITH_ADAPTER`

`archive_reader.py:64-145` validates base columns, filters dates, sorts by EOB,
and de-duplicates. Reuse those operations. A new adapter must add explicit
identity, session, time-slot, source-version, resolution, and feature
availability metadata. The returned frame is not a complete long-sequence
contract.

### Continuous contract mapping: `REUSE_WITH_ADAPTER`

`build_continuous_bars.py:149-176` joins each trading date to the selected real
contract. Reuse the authoritative map and stitch order. Add per-position active
contract, contract age, roll boundary, gap/session masks, and provenance in the
new dataset path. Aggregate switch counts are insufficient model inputs.

### Feature profiles: `REUSE_WITH_ADAPTER`

`feature_profiles.py:31-54,82-162` already records ordered columns, dimension,
source schema, scaler, normalization, and fingerprints. Reuse those contract
ideas and checks. MarketGenome profiles additionally need modality, resolution,
identity, track availability, and mask semantics. M12 dynamically supplies
`feature_dim`; it is incorrect to describe the current trainer as functionally
fixed to 8D solely because stale type annotations say `B L 8`.

### Window cache: `REPLACE_FOR_MARKETGENOME`

`window_cache.py:18-20,150-199` fixes four horizons, converts non-finite values
to zero, and materializes `[N, lookback, F]`, `[N, 4]`, and symbol arrays. The
fingerprint/atomic-publication concepts can be reapplied, but the layout and
loader cannot. Existing M0-M12 cache behavior remains unchanged.

### Long-sequence storage: `REPLACE_FOR_MARKETGENOME`

`window_cache.py:181-200` demonstrates that the only current training store is
fixed-window NPY. No current long-chunk schema/storage exists. MG3 must define a
separately versioned store with timestamps, valid-position masks, per-track
masks, identity, fold, roll, overlap, and source metadata after the ontology and
track contracts are frozen.

### Model config: `REPLACE_FOR_MARKETGENOME`

`core/schemas.py:23-57` contains AlphaTrade-family defaults. The core default is
4096/8 features/6 stages, while `train_m4_alphatrade.py:383,429-441` resolves a
60-bar/two-stage M2/M12 path. MarketGenome must not inherit either set
implicitly. A future config must be fully resolved and its canonical payload
and hash must appear in reports, artifacts, checkpoints, resume checks, and
bundles. MG0-B's config records scaffold state only.

### Training launcher: `ISOLATE`

`train_m4_alphatrade.py:429-441` constructs horizons, quantiles, encoder stages,
channels, `d_model`, and transformer depth in code. Its batches and losses are
terminal-horizon AlphaTrade contracts. `run_iteration.py:81-139` is additionally
wired to the M5 sweep and M7 regression command chain. Reuse the
plan-run-report-strict-gate workflow, not these scripts. A separate launcher and
runner follow MG4 and require resolved config/data/ontology/track hashes and an
explicit backend.

### Checkpointing: `REUSE_WITH_ADAPTER`

`train_m4_alphatrade.py:658-685` persists parameters, model state, optimizer
state, step, and best metric. `run_m5_sweep.py:183-220` provides the stronger
fail-closed identity check. Reuse the persistence/provenance principles, not the
checkpoint format. A future MarketGenome checkpoint needs its own version plus
config, dataset, ontology, track, precision, and sharding fingerprints.

### Bundle format: `REUSE_WITH_ADAPTER`

`export_model_bundle.py:70-103` supplies useful file hashing and deterministic
bundle identity. `export_model_bundle.py:258-311` fixes the payload to
`alphatrade_model_bundle_v1`, v0.2, M5 champion selection, and M9 prediction
schemas. Preserve v1. A future MarketGenome bundle must use a new format and
loader identity.

### Offline inference: `ISOLATE`

`inference.py:51-74,125-182` constructs `AlphaTradeConfig/AlphaTrade`, restores
a fixed-window model, and returns only terminal `log_return_quantiles`. The
future MarketGenome predictor requires a separate chunk iterator, restore
contract, dense writer, and masks. An optional terminal-alpha handoff can adapt
a declared output to M9 later; it cannot make the existing predictor generic.

### M10/M11 evaluation: `REUSE_WITH_ADAPTER`

`build_m10_prediction_eval.py:444-610` implements useful terminal-quantile
pinball, coverage, and crossing kernels. `build_m11_scale_calibration_report.py:
590-642` uses validation-only additive calibration. Reuse only after a versioned
adapter produces semantically identical canonical rows. Do not reuse M9 wide
alignment, q50 signal assumptions, historical thresholds, or the fixed M11
closure conclusion for dense tasks. The current M10 strict route has no semantic
checks (`0/0`), so its schema PASS is not a model-quality gate.

### Report governance: `DIRECT_REUSE`

`docs/reports_governance.md:7-40` and `runtime_paths.py:21-37` establish
source-of-truth order, external run roots, schemas, and profile isolation. These
rules apply unchanged. MG0-B uses an independent manifest because MG0-A's freeze
checker byte-anchors the current legacy validator, manifest, and validator tests
to tag `mg0-a-alphagenome-parity-v1`.

### Resume safety: `REUSE_WITH_ADAPTER`

`run_m5_sweep.py:70-86,183-271` hashes canonical config and rejects mismatched or
stale resume artifacts. Reuse the fail-closed principle. MarketGenome must add
ontology, track schema, storage, model, precision, and distributed-layout
identity. Do not reuse the M5 runner itself.

## Architecture Boundaries

| Component | Classification | Exact boundary |
|---|---|---|
| Causal standardized convolution (`causal_layers.py:71-94`) | `REUSE_WITH_ADAPTER` | Re-prove availability/valid-position masks and prefix/future invariance in the new namespace. |
| Pool/decoder alignment (`causal_layers.py:121-143,200-205`) | `REUSE_WITH_ADAPTER` | Test every length remainder and dense-position alignment before reuse. |
| Soft-capped causal attention (`causal_attention.py:134-170`) | `REUSE_WITH_ADAPTER` | Add padding/session/roll/gap masks and actual time positions; current RoPE uses implicit positions. |
| Encoder-transformer-decoder trunk (`model.py:238-269`) | `REPLACE_FOR_MARKETGENOME` | No scale-shared motif, identity conditioning, or contracted dense multi-resolution output exists. |
| Last-timestep readout (`model.py:271-315`) | `REPLACE_FOR_MARKETGENOME` | It directly conflicts with dense per-position supervision. |
| Median/cumulative-delta quantile head (`model.py:118-171`) | `REPLACE_FOR_MARKETGENOME` | It is a terminal quantile head, not a general multimodal dense-head contract. |
| Pinball/crossing primitives (`losses.py:23-132`) | `REUSE_WITH_ADAPTER` | Reuse math only for compatible quantile tracks; add masks and per-track aggregation. |

Pair stack, teachers, distillation, synthetic generation, perturbation, and
effect scoring have no current AlphaTrade implementation to reuse. They remain
`DEFER` and must not be represented by pseudo-implementations.

## Confirmed Incompatible Assumptions

1. **Fixed short lookback.** The operational M12 path uses one 60-bar window
   (`train_m4_alphatrade.py:379-403`). This fact is distinct from the 4096 core
   default.
2. **Last-timestep-only readout.** Every required head consumes
   `x[:, -1, :]` (`model.py:271-280`).
3. **Terminal-return-only labels.** Each sample stores four scalar horizon
   returns (`window_cache.py:158-194`).
4. **Fixed-window cache layout.** Arrays are `[N,L,F]` and `[N,4]`, with a
   hard-coded `(0,60,F)` empty case (`window_cache.py:181-200`).
5. **Current quantile head parameterization.** Median plus cumulative signed
   deltas requires odd centered quantiles (`model.py:149-169`,
   `core/schemas.py:77-102`).
6. **Hard-coded architecture behavior.** The launcher fixes task and topology
   fields, while public annotations retain stale `B L 8` assumptions
   (`train_m4_alphatrade.py:429-441`, `core/schemas.py:113-159`).

All six assumptions are isolated from the MarketGenome namespace. None is
silently retained as a default.

## M10/M11 Reuse Boundary

Reusable after a canonical-row adapter, and only where units/tasks match:

- pinball loss;
- quantile coverage and coverage MAE;
- quantile crossing;
- continuous output distributions;
- Pearson/rank IC and direction hit for a declared terminal scalar;
- zero and past-only rolling baselines;
- validation-only quantile residual calibration;
- common-row, split isolation, and schema-versus-quality governance.

Not reusable as a general MarketGenome gate:

- q10/q30/q50/q70/q90 lightweight backtest signals;
- current M11 thresholds and historical closure conclusions;
- fixed M9 wide prediction rows and all-horizon availability rules;
- current target-scale audit conclusions;
- claims of execution realism, annualized performance, or trade readiness.

No implementation exists yet for CRPS, dense path/profile divergence,
volatility/range track scoring, barrier AUC/AUPRC/Brier, MFE/MAE, pattern phase
and timing, scale transfer, matched controls, or perturbation effects. These are
future contracts, not inferred reuse.

## Scaffold and Profile

The new package root imports without JAX, Haiku, model, training, inference, or
GPU initialization. Placeholder packages expose only
`IMPLEMENTATION_STATUS = "NOT_IMPLEMENTED"`.

The independent profile is
`configs/market_genome/contracts_manifest.yaml`. It requires:

- `mg0_alphatrade_compatibility_audit.json` against
  `src/alphatrade/market_genome/mg0b_compatibility_audit.schema.json`;
- `mg0_alphatrade_compatibility_audit.md` as a required nonempty companion.

`python -m alphatrade.market_genome.schemas --profile mg0b --reports-dir
<reports> --strict` additionally checks the exact 13-subsystem set, enum values,
repository evidence line ranges, six incompatible assumptions, M12 PASS/FAIL
separation, artifact hashes, exact scaffold inventory, legacy preservation,
resolved config hash, deferred execution status, gate consistency, and open
uncertainties.

The legacy `src/alphatrade/schemas/contracts_manifest.yaml`, validator, tests,
model files, reports, and result files are unchanged. This preserves the MG0-A
freeze and every M0-M12 profile rather than weakening or moving the freeze tag.

## Verification

- MG0-B tests: `19 passed`.
- Non-GPU AlphaTrade ingestion/profile/M8-M12 report/evaluation regression set:
  `108 passed`.
- Independent strict validator: schema `1/1`, semantic `14/14`.
- Independent manifest through the legacy schema phase: required artifacts
  `2/2`; its generic legacy semantic dispatcher reports `0/0`, so the
  authoritative MG0-B semantics are the new fail-closed `14/14` checker.
- The canonical JSON payload and Markdown companion are SHA256-bound; all five
  locally available M12 source artifacts matched their recorded content hashes.
- Frozen MG0-A strict regression: schema `2/2`, semantic `13/13`.
- Existing report-validator tests: `54 passed` (also included in the 108-test
  regression set).
- Canonical JSON files parse and `git diff --check` passes.

All project test commands used `CUDA_VISIBLE_DEVICES` empty. The full model/JAX
training test set was not run because GPU/model work is explicitly deferred;
this limitation does not affect the static scaffold/schema/import acceptance
checks.

## Uncertainties

The following remain `NEEDS_VERIFICATION`:

- runtime behavior around dynamic feature dimensions versus stale `B L 8`
  annotations, because MG0-B intentionally did not execute JAX;
- long-chunk length, overlap, masks, and storage format until MG1/MG3;
- per-task MarketGenome metrics and promotion thresholds;
- checkpoint, precision, optimizer, and distributed resume identity.

## Gate

`PASS_WITH_DOCUMENTED_BOUNDARIES` means:

- all 13 required subsystem boundaries and all six incompatible assumptions are
  explicit and source-grounded;
- M12 contract/schema success is not presented as model-quality success;
- the package is scaffold-only and import-safe;
- current M0-M12 files and result contracts remain unchanged;
- an independent strict `mg0b` schema/profile exists;
- training, JAX computation, inference, checkpoints, and GPU execution remain
  deferred; the recorded legacy JAX import was CPU-only.

The exact next task is `MG1-A_PATTERN_ONTOLOGY_DRAFT_REQUIRES_HUMAN_FREEZE`.
MG1 may produce a draft workflow, but domain ontology cannot be declared frozen
without explicit human review.
