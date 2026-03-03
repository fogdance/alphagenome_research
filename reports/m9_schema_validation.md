# Reports Schema & Semantic Validation (profile: m9)

Generated: 2026-03-03 18:32:26
Strict: True

## Phase 1: Schema Validation

- Total: 4
- Passed: 4
- Failed/Missing: 0

| Report | Required | Status | Error |
|--------|----------|--------|-------|
| m9_model_bundle_manifest | ✅ | ✅ pass | - |
| m9_infer_metrics | ✅ | ✅ pass | - |
| m9_infer_metrics_md | ✅ | ✅ pass | - |
| m9_predictions_parquet | ✅ | ✅ pass | - |

## Phase 2: M9 Bundle + Predictions Checks

| Check | Status | Detail |
|-------|--------|--------|
| manifest_exists | ✅ | - |
| bundle_path_exists | ✅ | - |
| model_version_non_empty | ✅ | - |
| predictions_exists | ✅ | - |
| predictions_parquet_load | ✅ | - |
| predictions_rows_gt_0 | ✅ | - |
| predictions_columns_complete | ✅ | - |
| predictions_column_types | ✅ | - |

**M9 checks**: 8/8 passed

## Overall

- Schema (required): ✅ pass
- Semantic: ✅ pass
- **Overall: ✅ ALL PASS**
