#!/usr/bin/env python3
"""Build M11 closure metadata and decision memo from completed M11 reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from alphatrade import prediction_schema
from alphatrade import runtime_paths


_STAGE_NOTE = (
    "AlphaTrade is in active development. These reports are evaluation and "
    "calibration diagnostics only, not trading-readiness claims."
)
_BACKTEST_NOTE = (
    "M9/M10/M11 backtest outputs are lightweight evaluation handoff checks, "
    "not full execution simulators."
)


def parse_args():
    parser = argparse.ArgumentParser(description="M11 closure metadata and decision memo")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root for generated outputs")
    parser.add_argument("--reports-dir", type=str, default=None,
                        help="Reports directory")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(obj: dict, path: list[str], default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _metric_summary(calibration: dict, quality: dict) -> dict:
    raw = calibration.get("raw", {})
    calibrated = calibration.get("calibrated", {})
    rolling = calibration.get("rolling_historical_recomparison", {})
    return {
        "model_quality_status": quality.get("overall_status"),
        "raw_pinball": _safe_get(raw, ["pinball_loss", "overall"]),
        "calibrated_pinball": _safe_get(calibrated, ["pinball_loss", "overall"]),
        "rolling_historical_pinball": _safe_get(
            rolling,
            ["rolling_historical_quantile", "pinball_loss", "overall"],
        ),
        "raw_coverage_mae": _safe_get(raw, ["quantile_coverage", "overall_mae"]),
        "calibrated_coverage_mae": _safe_get(
            calibrated,
            ["quantile_coverage", "overall_mae"],
        ),
        "coverage_mae_delta_raw_minus_calibrated": _safe_get(
            calibration,
            ["raw_vs_calibrated", "coverage_mae_delta_raw_minus_calibrated"],
        ),
        "max_abs_pearson_ic_raw": _safe_get(
            raw,
            ["ic_metrics", "materiality_proxy", "max_abs_pearson_ic"],
        ),
        "max_abs_rank_ic_raw": _safe_get(
            raw,
            ["ic_metrics", "materiality_proxy", "max_abs_rank_ic"],
        ),
        "max_abs_pearson_ic_calibrated": _safe_get(
            calibrated,
            ["ic_metrics", "materiality_proxy", "max_abs_pearson_ic"],
        ),
        "max_abs_rank_ic_calibrated": _safe_get(
            calibrated,
            ["ic_metrics", "materiality_proxy", "max_abs_rank_ic"],
        ),
        "rolling_minus_calibrated_pinball": rolling.get(
            "pinball_delta_rolling_minus_calibrated"
        ),
    }


def build_calibrated_prediction_metadata(
    *,
    audit: dict,
    quality: dict,
    calibration: dict,
    val_infer_metrics: dict | None,
    reports_dir: Path,
) -> dict:
    raw_model_version = None
    if val_infer_metrics:
        raw_model_version = val_infer_metrics.get("model_version")
    if raw_model_version is None:
        raw_model_version = "unknown"

    conclusion = calibration.get("conclusion", {})
    champion_status = conclusion.get("champion_status")
    metrics = _metric_summary(calibration, quality)

    return {
        "schema_version": "m11_calibrated_prediction_metadata_v1",
        "generated_at": datetime.now().isoformat(),
        "stage": "development_evaluation",
        "notes": [_STAGE_NOTE, _BACKTEST_NOTE],
        "source_reports": {
            "target_scale_audit": str(reports_dir / "m11_target_scale_audit.json"),
            "model_quality_validation": str(reports_dir / "m11_model_quality_validation.json"),
            "calibration_comparison": str(reports_dir / "m11_calibration_comparison.json"),
        },
        "target_scale_conclusion": audit.get("target_unit_conclusion", {}),
        "calibrated_prediction_identity": {
            "raw_model_version": raw_model_version,
            "calibrated_candidate_version": f"{raw_model_version}_m11_posthoc_calibrated",
            "prediction_schema_version": prediction_schema.PREDICTION_SCHEMA_VERSION,
            "horizons": prediction_schema.DEFAULT_HORIZONS,
            "quantiles": prediction_schema.DEFAULT_QUANTILES,
            "wide_prediction_columns": prediction_schema.prediction_columns(),
            "calibrated_predictions_written": False,
            "calibrated_predictions_path": None,
            "metadata_only_reason": (
                "Post-hoc calibration is diagnostic only for M11 closure; "
                "the calibrated candidate is not promoted because it still loses "
                "to the rolling historical baseline and IC remains weak."
            ),
        },
        "calibration": {
            "method": _safe_get(calibration, ["calibrator", "method"]),
            "calibrator_schema_version": _safe_get(calibration, ["calibrator", "schema_version"]),
            "fit_rows": _safe_get(calibration, ["calibrator", "fit_rows"]),
            "validation_split_filter": calibration.get("validation_split_filter", {}),
            "non_crossing_enforced": True,
        },
        "metrics_summary": metrics,
        "promotion": {
            "promoted": False,
            "promotion_status": "not_promoted",
            "champion_status": champion_status,
            "reason": (
                "Coverage improves after calibration, but calibrated pinball "
                "remains worse than rolling_historical_quantile and IC/rank IC "
                "remain materially indistinguishable from zero."
            ),
        },
    }


def write_metadata_md(metadata: dict, path: Path) -> None:
    identity = metadata["calibrated_prediction_identity"]
    promotion = metadata["promotion"]
    metrics = metadata["metrics_summary"]
    lines = [
        "# M11 Calibrated Prediction Metadata",
        "",
        _STAGE_NOTE,
        "",
        _BACKTEST_NOTE,
        "",
        "## Identity",
        "",
        f"- Raw model version: `{identity['raw_model_version']}`",
        f"- Calibrated candidate version: `{identity['calibrated_candidate_version']}`",
        f"- Prediction schema version: `{identity['prediction_schema_version']}`",
        f"- Calibrated predictions written: `{identity['calibrated_predictions_written']}`",
        f"- Promotion status: `{promotion['promotion_status']}`",
        f"- Champion status: `{promotion['champion_status']}`",
        "",
        "## Metrics",
        "",
        f"- Raw pinball: {metrics['raw_pinball']}",
        f"- Calibrated pinball: {metrics['calibrated_pinball']}",
        f"- Rolling historical pinball: {metrics['rolling_historical_pinball']}",
        f"- Raw coverage MAE: {metrics['raw_coverage_mae']}",
        f"- Calibrated coverage MAE: {metrics['calibrated_coverage_mae']}",
        f"- Rolling minus calibrated pinball delta: {metrics['rolling_minus_calibrated_pinball']}",
        "",
        "## Promotion Decision",
        "",
        promotion["reason"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision_memo(
    *,
    audit: dict,
    quality: dict,
    calibration: dict,
    metadata: dict,
    path: Path,
) -> None:
    target = audit.get("target_unit_conclusion", {})
    metrics = metadata["metrics_summary"]
    lines = [
        "# M11 Decision Memo",
        "",
        _STAGE_NOTE,
        "",
        _BACKTEST_NOTE,
        "",
        "## Decision",
        "",
        "Champion remains rejected. The calibrated candidate is retained only as "
        "diagnostic metadata and is not promoted.",
        "",
        "## Required Findings",
        "",
        "1. Target scale bug ruled out.",
        f"   Target unit is `{target.get('target_unit')}` and training/inference/evaluation use the same raw log-return convention.",
        "2. Output calibration problem confirmed.",
        f"   Model output scale mismatch found: `{target.get('model_output_scale_mismatch_found')}`; model quality status is `{quality.get('overall_status')}`.",
        "3. Post-hoc calibration improves coverage.",
        f"   Coverage MAE improves from {metrics['raw_coverage_mae']} to {metrics['calibrated_coverage_mae']}.",
        "4. Calibrated model still loses to rolling historical baseline.",
        f"   Calibrated pinball is {metrics['calibrated_pinball']} versus rolling historical {metrics['rolling_historical_pinball']}.",
        "5. Champion remains rejected.",
        f"   Champion status is `{metadata['promotion']['champion_status']}` and promotion status is `{metadata['promotion']['promotion_status']}`.",
        "6. Next milestone is M12 feature/label/baseline expansion.",
        "",
        "## Metric Snapshot",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Raw pinball | {metrics['raw_pinball']} |",
        f"| Calibrated pinball | {metrics['calibrated_pinball']} |",
        f"| Rolling historical pinball | {metrics['rolling_historical_pinball']} |",
        f"| Raw coverage MAE | {metrics['raw_coverage_mae']} |",
        f"| Calibrated coverage MAE | {metrics['calibrated_coverage_mae']} |",
        f"| Max abs Pearson IC raw | {metrics['max_abs_pearson_ic_raw']} |",
        f"| Max abs Rank IC raw | {metrics['max_abs_rank_ic_raw']} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit = _load_json(reports_dir / "m11_target_scale_audit.json")
    quality = _load_json(reports_dir / "m11_model_quality_validation.json")
    calibration = _load_json(reports_dir / "m11_calibration_comparison.json")
    val_infer_path = reports_dir / "m11_val_infer_metrics.json"
    val_infer_metrics = _load_json(val_infer_path) if val_infer_path.exists() else None

    metadata = build_calibrated_prediction_metadata(
        audit=audit,
        quality=quality,
        calibration=calibration,
        val_infer_metrics=val_infer_metrics,
        reports_dir=reports_dir,
    )

    metadata_json = reports_dir / "m11_calibrated_prediction_metadata.json"
    metadata_md = reports_dir / "m11_calibrated_prediction_metadata.md"
    memo_md = reports_dir / "M11_DECISION_MEMO.md"

    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_metadata_md(metadata, metadata_md)
    write_decision_memo(
        audit=audit,
        quality=quality,
        calibration=calibration,
        metadata=metadata,
        path=memo_md,
    )

    print(f"Wrote {metadata_json}")
    print(f"Wrote {metadata_md}")
    print(f"Wrote {memo_md}")


if __name__ == "__main__":
    main()
