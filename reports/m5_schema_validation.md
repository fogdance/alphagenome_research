# Reports Schema & Semantic Validation (profile: m5)

Generated: 2026-03-03 13:50:19
Strict: True

## Phase 1: Schema Validation

- Total: 5
- Passed: 5
- Failed/Missing: 0

| Report | Required | Status | Error |
|--------|----------|--------|-------|
| m5_sweep_manifest | ✅ | ✅ pass | - |
| m5_leaderboard | ✅ | ✅ pass | - |
| m5_leaderboard_md | ✅ | ✅ pass | - |
| m5_schema_validation | ✅ | ✅ pass | - |
| m5_schema_validation_md | ✅ | ✅ pass | - |

## Phase 2: M5 Sweep Semantic Checks

### Experiment: baseline

| Check | Status | Detail |
|-------|--------|--------|
| [baseline] config_hash_consistent | ✅ | - |
| [baseline] seeds_complete | ✅ | - |
| [baseline/seed42] train_exists | ✅ | - |
| [baseline/seed42] train_schema | ✅ | - |
| [baseline/seed42] eval_exists | ✅ | - |
| [baseline/seed42] eval_schema | ✅ | - |
| [baseline/seed43] train_exists | ✅ | - |
| [baseline/seed43] train_schema | ✅ | - |
| [baseline/seed43] eval_exists | ✅ | - |
| [baseline/seed43] eval_schema | ✅ | - |
| [baseline/seed44] train_exists | ✅ | - |
| [baseline/seed44] train_schema | ✅ | - |
| [baseline/seed44] eval_exists | ✅ | - |
| [baseline/seed44] eval_schema | ✅ | - |

### Experiment: no_clip

| Check | Status | Detail |
|-------|--------|--------|
| [no_clip] config_hash_consistent | ✅ | - |
| [no_clip] seeds_complete | ✅ | - |
| [no_clip/seed42] train_exists | ✅ | - |
| [no_clip/seed42] train_schema | ✅ | - |
| [no_clip/seed42] eval_exists | ✅ | - |
| [no_clip/seed42] eval_schema | ✅ | - |
| [no_clip/seed43] train_exists | ✅ | - |
| [no_clip/seed43] train_schema | ✅ | - |
| [no_clip/seed43] eval_exists | ✅ | - |
| [no_clip/seed43] eval_schema | ✅ | - |
| [no_clip/seed44] train_exists | ✅ | - |
| [no_clip/seed44] train_schema | ✅ | - |
| [no_clip/seed44] eval_exists | ✅ | - |
| [no_clip/seed44] eval_schema | ✅ | - |

### Experiment: batch_256

| Check | Status | Detail |
|-------|--------|--------|
| [batch_256] config_hash_consistent | ✅ | - |
| [batch_256] seeds_complete | ✅ | - |
| [batch_256/seed42] train_exists | ✅ | - |
| [batch_256/seed42] train_schema | ✅ | - |
| [batch_256/seed42] eval_exists | ✅ | - |
| [batch_256/seed42] eval_schema | ✅ | - |
| [batch_256/seed43] train_exists | ✅ | - |
| [batch_256/seed43] train_schema | ✅ | - |
| [batch_256/seed43] eval_exists | ✅ | - |
| [batch_256/seed43] eval_schema | ✅ | - |
| [batch_256/seed44] train_exists | ✅ | - |
| [batch_256/seed44] train_schema | ✅ | - |
| [batch_256/seed44] eval_exists | ✅ | - |
| [batch_256/seed44] eval_schema | ✅ | - |

### Experiment: steps_1000

| Check | Status | Detail |
|-------|--------|--------|
| [steps_1000] config_hash_consistent | ✅ | - |
| [steps_1000] seeds_complete | ✅ | - |
| [steps_1000/seed42] train_exists | ✅ | - |
| [steps_1000/seed42] train_schema | ✅ | - |
| [steps_1000/seed42] eval_exists | ✅ | - |
| [steps_1000/seed42] eval_schema | ✅ | - |
| [steps_1000/seed43] train_exists | ✅ | - |
| [steps_1000/seed43] train_schema | ✅ | - |
| [steps_1000/seed43] eval_exists | ✅ | - |
| [steps_1000/seed43] eval_schema | ✅ | - |
| [steps_1000/seed44] train_exists | ✅ | - |
| [steps_1000/seed44] train_schema | ✅ | - |
| [steps_1000/seed44] eval_exists | ✅ | - |
| [steps_1000/seed44] eval_schema | ✅ | - |

**Semantic summary**: 58/58 checks passed

## Overall

- Schema (required): ✅ pass
- Semantic: ✅ pass
- **Overall: ✅ ALL PASS**
