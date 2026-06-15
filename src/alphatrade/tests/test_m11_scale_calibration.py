from __future__ import annotations

import json

import numpy as np
import pandas as pd

from alphatrade.scripts import build_m10_prediction_eval as m10
from alphatrade.scripts import build_m11_closure
from alphatrade.scripts import build_m11_scale_calibration_report as m11
from alphatrade.scripts.validate_reports_schema import semantic_check_m11, validate_report


def _rows_with_bad_constant_quantiles(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    y = rng.normal(loc=0.0, scale=0.001, size=n)
    return pd.DataFrame(
        {
            "symbol": ["DCE.JM"] * n,
            "eob": pd.date_range("2023-01-01 09:00:00", periods=n, freq="1min"),
            "model_version": ["m"] * n,
            "realized_h1": y,
            "h1_q10": np.full(n, -0.03),
            "h1_q30": np.full(n, -0.01),
            "h1_q50": np.full(n, 0.01),
            "h1_q70": np.full(n, 0.03),
            "h1_q90": np.full(n, 0.05),
        }
    )


def test_quantile_shift_calibrator_reduces_coverage_error():
    rows = _rows_with_bad_constant_quantiles()
    raw = m10.compute_quantile_coverage(rows, [1], [0.1, 0.3, 0.5, 0.7, 0.9])

    calibrator = m11.fit_quantile_shift_calibrator(
        rows,
        horizons=[1],
        quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    calibrated = m11.apply_quantile_shift_calibrator(
        rows,
        calibrator,
        horizons=[1],
        quantiles=[0.1, 0.3, 0.5, 0.7, 0.9],
    )
    calibrated_cov = m10.compute_quantile_coverage(
        calibrated,
        [1],
        [0.1, 0.3, 0.5, 0.7, 0.9],
    )
    crossing = m10.compute_quantile_crossing(
        calibrated,
        [1],
        [0.1, 0.3, 0.5, 0.7, 0.9],
    )

    assert calibrated_cov["overall_mae"] < raw["overall_mae"]
    assert calibrated_cov["overall_mae"] < 0.01
    assert crossing["count"] == 0


def test_quality_validation_flags_model_quality_failures():
    m10_metrics = {
        "pinball_loss": {"overall": 0.003},
        "quantile_coverage": {
            "overall_mae": 0.21,
            "by_horizon": {
                "h1": {"coverage": {"q50": {"abs_error": 0.25}}},
            },
        },
        "ic_metrics": {
            "materiality_proxy": {
                "max_abs_pearson_ic": 0.002,
                "max_abs_rank_ic": 0.003,
            }
        },
    }
    baseline = {
        "baselines": {
            "zero_return_quantile": {"pinball_loss": {"overall": 0.001}},
            "rolling_historical_quantile": {"pinball_loss": {"overall": 0.0008}},
        }
    }
    scale_ratios = {
        "h1": {
            "prediction_width_to_realized_p99_p01_ratio": 20.0,
        }
    }

    report = m11.build_quality_validation(
        m10_metrics=m10_metrics,
        m10_baseline=baseline,
        scale_ratios=scale_ratios,
    )

    assert report["overall_status"] == "FAIL_MODEL_QUALITY"
    failed = {c["name"] for c in report["checks"] if c["status"] == "FAIL_MODEL_QUALITY"}
    assert "model_pinball_vs_zero_return_quantile" in failed
    assert "model_pinball_vs_rolling_historical_quantile" in failed
    assert "coverage_calibration_mae" in failed
    assert "max_abs_ic" in failed
    assert "prediction_q90_q10_scale" in failed
    assert "q50_coverage_abs_error" in failed


def test_filter_rows_to_split_index_uses_only_requested_split(tmp_path):
    data_dir = tmp_path / "data"
    symbol_dir = data_dir / "DCE.JM"
    symbol_dir.mkdir(parents=True)
    eobs = pd.date_range("2023-01-01 09:00:00", periods=3, freq="1min")
    pd.DataFrame({"eob": [eobs[0], eobs[2]]}).to_parquet(
        symbol_dir / "index_val.parquet",
        index=False,
    )
    rows = pd.DataFrame(
        {
            "symbol": ["DCE.JM"] * 3,
            "eob": eobs,
            "realized_h1": [0.0, 0.0, 0.0],
        }
    )

    filtered, meta = m11.filter_rows_to_split_index(rows, data_dir=data_dir, split="val")

    assert meta["rows_before"] == 3
    assert meta["rows_after"] == 2
    assert filtered["eob"].tolist() == [eobs[0], eobs[2]]


def test_m11_minimal_reports_validate_against_schemas(tmp_path):
    schemas_dir = m11._SRC_ROOT / "alphatrade" / "schemas"
    audit = {
        "schema_version": "m11_target_scale_audit_v1",
        "generated_at": "2026-01-01T00:00:00",
        "inputs": {},
        "target_unit_conclusion": {
            "target_unit": "raw_log_return",
            "target_formula": "log(close[t+h] / close[t])",
            "training_target_normalization": "none_found",
            "inference_inverse_transform_required": False,
            "engineering_scale_mismatch_found": False,
            "model_output_scale_mismatch_found": True,
        },
        "code_path_audit": {},
        "target_statistics": {},
        "scale_diagnostics": {},
    }
    quality = {
        "schema_version": "m11_model_quality_validation_v1",
        "generated_at": "2026-01-01T00:00:00",
        "inputs": {},
        "severity_levels": ["PASS", "WARN", "FAIL_MODEL_QUALITY"],
        "overall_status": "FAIL_MODEL_QUALITY",
        "checks": [
            {
                "name": "coverage_calibration_mae",
                "status": "FAIL_MODEL_QUALITY",
                "threshold": "<= 0.05",
                "detail": "bad coverage",
            }
        ],
    }
    calibration = {
        "schema_version": "m11_calibration_comparison_v1",
        "generated_at": "2026-01-01T00:00:00",
        "calibrator": {},
        "fit_diagnostics": {},
        "raw": {},
        "calibrated": {},
        "raw_vs_calibrated": {},
        "rolling_historical_recomparison": {},
        "conclusion": {
            "coverage_mae_reduced_materially": True,
            "calibrated_model_can_be_retested_as_champion": False,
            "champion_status": "rejected_pending_retrain_or_stronger_fix",
        },
    }
    metadata = {
        "schema_version": "m11_calibrated_prediction_metadata_v1",
        "generated_at": "2026-01-01T00:00:00",
        "source_reports": {},
        "target_scale_conclusion": {},
        "calibrated_prediction_identity": {
            "raw_model_version": "m",
            "calibrated_candidate_version": "m_m11_posthoc_calibrated",
            "prediction_schema_version": "m9_predictions_v1",
            "calibrated_predictions_written": False,
            "calibrated_predictions_path": None,
        },
        "calibration": {},
        "metrics_summary": {},
        "promotion": {
            "promoted": False,
            "promotion_status": "not_promoted",
            "champion_status": "rejected_pending_retrain_or_stronger_fix",
            "reason": "diagnostic only",
        },
    }
    reports = [
        ("audit.json", audit, "m11_target_scale_audit.schema.json"),
        ("quality.json", quality, "m11_model_quality_validation.schema.json"),
        ("calibration.json", calibration, "m11_calibration_comparison.schema.json"),
        ("metadata.json", metadata, "m11_calibrated_prediction_metadata.schema.json"),
    ]
    for filename, payload, schema_name in reports:
        report_path = tmp_path / filename
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        result = validate_report(str(report_path), str(schemas_dir / schema_name))
        assert result["status"] == "pass", result


def test_build_m11_closure_metadata_does_not_promote_calibrated_candidate(tmp_path):
    audit = {
        "target_unit_conclusion": {
            "target_unit": "raw_log_return",
            "engineering_scale_mismatch_found": False,
            "model_output_scale_mismatch_found": True,
        }
    }
    quality = {"overall_status": "FAIL_MODEL_QUALITY"}
    calibration = {
        "calibrator": {
            "schema_version": "m11_quantile_shift_calibrator_v1",
            "method": "shift",
            "fit_rows": 10,
        },
        "raw": {
            "pinball_loss": {"overall": 0.003},
            "quantile_coverage": {"overall_mae": 0.2},
            "ic_metrics": {
                "materiality_proxy": {
                    "max_abs_pearson_ic": 0.001,
                    "max_abs_rank_ic": 0.001,
                }
            },
        },
        "calibrated": {
            "pinball_loss": {"overall": 0.0008},
            "quantile_coverage": {"overall_mae": 0.04},
            "ic_metrics": {
                "materiality_proxy": {
                    "max_abs_pearson_ic": 0.001,
                    "max_abs_rank_ic": 0.001,
                }
            },
        },
        "raw_vs_calibrated": {
            "coverage_mae_delta_raw_minus_calibrated": 0.16,
        },
        "rolling_historical_recomparison": {
            "rolling_historical_quantile": {
                "pinball_loss": {"overall": 0.0006},
            },
            "pinball_delta_rolling_minus_calibrated": -0.0002,
        },
        "conclusion": {
            "champion_status": "rejected_pending_retrain_or_stronger_fix",
        },
    }

    metadata = build_m11_closure.build_calibrated_prediction_metadata(
        audit=audit,
        quality=quality,
        calibration=calibration,
        val_infer_metrics={"model_version": "model_v1"},
        reports_dir=tmp_path,
    )

    assert metadata["promotion"]["promoted"] is False
    assert metadata["promotion"]["promotion_status"] == "not_promoted"
    assert metadata["calibrated_prediction_identity"]["calibrated_predictions_written"] is False


def test_m11_semantic_checks_separate_quality_fail_from_closure_pass(tmp_path):
    reports_dir = tmp_path
    audit = {
        "schema_version": "m11_target_scale_audit_v1",
        "target_unit_conclusion": {
            "target_unit": "raw_log_return",
            "engineering_scale_mismatch_found": False,
            "inference_inverse_transform_required": False,
            "model_output_scale_mismatch_found": True,
        },
    }
    quality = {
        "schema_version": "m11_model_quality_validation_v1",
        "overall_status": "FAIL_MODEL_QUALITY",
    }
    calibration = {
        "schema_version": "m11_calibration_comparison_v1",
        "raw": {
            "quantile_coverage": {"overall_mae": 0.214},
        },
        "calibrated": {
            "quantile_coverage": {"overall_mae": 0.041},
        },
        "raw_vs_calibrated": {
            "coverage_mae_delta_raw_minus_calibrated": 0.173,
        },
        "rolling_historical_recomparison": {
            "rolling_historical_quantile": {
                "pinball_loss": {"overall": 0.00061},
            },
            "calibrated_model_on_common": {
                "pinball_loss": {"overall": 0.00083},
            },
            "pinball_delta_rolling_minus_calibrated": -0.00022,
        },
        "conclusion": {
            "champion_status": "rejected_pending_retrain_or_stronger_fix",
        },
    }
    metadata = {
        "schema_version": "m11_calibrated_prediction_metadata_v1",
        "calibrated_prediction_identity": {
            "calibrated_predictions_written": False,
        },
        "promotion": {
            "promoted": False,
            "promotion_status": "not_promoted",
            "champion_status": "rejected_pending_retrain_or_stronger_fix",
        },
    }
    (reports_dir / "m11_target_scale_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (reports_dir / "m11_model_quality_validation.json").write_text(json.dumps(quality), encoding="utf-8")
    (reports_dir / "m11_calibration_comparison.json").write_text(json.dumps(calibration), encoding="utf-8")
    (reports_dir / "m11_calibrated_prediction_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (reports_dir / "M11_DECISION_MEMO.md").write_text(
        "Champion remains rejected. Next milestone is M12 feature/label/baseline expansion.",
        encoding="utf-8",
    )

    result = semantic_check_m11(str(reports_dir))

    assert result["all_pass"] is True
    assert result["model_quality_status"] == "FAIL_MODEL_QUALITY"
    assert all(check["status"] == "pass" for check in result["checks"])
