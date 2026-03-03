# Reports Schema & Semantic Validation (profile: m4)

Generated: 2026-03-03 13:09:53
Strict: True

## Phase 1: Schema Validation

- Total: 6
- Passed: 6
- Failed/Missing: 0

| Report | Required | Status | Error |
|--------|----------|--------|-------|
| m4_train_metrics_seed42 | ✅ | ✅ pass | - |
| m4_train_metrics_seed43 | ✅ | ✅ pass | - |
| m4_train_metrics_seed44 | ✅ | ✅ pass | - |
| m4_eval_metrics_seed42 | ✅ | ✅ pass | - |
| m4_eval_metrics_seed43 | ✅ | ✅ pass | - |
| m4_eval_metrics_seed44 | ✅ | ✅ pass | - |

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
