# Reports Schema & Semantic Validation

生成时间: 2026-03-02 21:11:04

## Phase 1: Schema Validation

- Total: 12
- Passed: 8
- Failed/Missing: 4

| Report | Status | Error |
|--------|--------|-------|
| m2_train_metrics | ❌ fail | 'grad_norm_pre_clip_max' is a required property |
| m2_universe_sweep | ✅ pass | - |
| m2_t1_dataloader_check | ✅ pass | - |
| m3_train_metrics | ❌ fail | 'grad_norm_pre_clip_max' is a required property |
| m4_train_metrics | ⚠️ missing_report | Report file not found: reports/m4_train_metrics.json |
| m4_eval_metrics | ⚠️ missing_report | Report file not found: reports/m4_eval_metrics.json |
| m4_eval_metrics_seed42 | ✅ pass | - |
| m4_train_metrics_seed42 | ✅ pass | - |
| m4_eval_metrics_seed43 | ✅ pass | - |
| m4_train_metrics_seed43 | ✅ pass | - |
| m4_eval_metrics_seed44 | ✅ pass | - |
| m4_train_metrics_seed44 | ✅ pass | - |

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

## 总体状态

- Schema: ❌ fail
- Semantic: ✅ pass
- **Overall: ❌ FAIL**
