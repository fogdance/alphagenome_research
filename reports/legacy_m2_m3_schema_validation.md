# Reports Schema & Semantic Validation (profile: legacy_m2_m3)

Generated: 2026-03-03 08:09:42
Strict: False

## Phase 1: Schema Validation

- Total: 4
- Passed: 2
- Failed/Missing: 2

| Report | Required | Status | Error |
|--------|----------|--------|-------|
| m2_train_metrics | - | ❌ fail | 'grad_norm_pre_clip_max' is a required property |
| m2_universe_sweep | - | ✅ pass | - |
| m2_t1_dataloader_check | - | ✅ pass | - |
| m3_train_metrics | - | ❌ fail | 'grad_norm_pre_clip_max' is a required property |

## Phase 2: M4 Semantic Checks

### Seed 42

- eval: `reports/m4_eval_metrics_seed42.json`
- train: `reports/m4_train_metrics_seed42.json`

| Check | Status | Detail |
|-------|--------|--------|
| model.source == 'checkpoint' | ✅ | - |
| model.train_run_id == train.run.run_id | ✅ | - |
| model.checkpoint_step >= 1 | ✅ | - |
| model.checkpoint_dir non-empty | ✅ | - |

### Seed 43

- eval: `reports/m4_eval_metrics_seed43.json`
- train: `reports/m4_train_metrics_seed43.json`

| Check | Status | Detail |
|-------|--------|--------|
| model.source == 'checkpoint' | ✅ | - |
| model.train_run_id == train.run.run_id | ✅ | - |
| model.checkpoint_step >= 1 | ✅ | - |
| model.checkpoint_dir non-empty | ✅ | - |

### Seed 44

- eval: `reports/m4_eval_metrics_seed44.json`
- train: `reports/m4_train_metrics_seed44.json`

| Check | Status | Detail |
|-------|--------|--------|
| model.source == 'checkpoint' | ✅ | - |
| model.train_run_id == train.run.run_id | ✅ | - |
| model.checkpoint_step >= 1 | ✅ | - |
| model.checkpoint_dir non-empty | ✅ | - |

**Semantic summary**: 12/12 checks passed

## Overall

- Schema (required): ✅ pass
- Semantic: ✅ pass
- **Overall: ✅ ALL PASS**
