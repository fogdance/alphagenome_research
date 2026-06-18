from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alphatrade.scripts import build_m12_chendage_features as m12
from alphatrade.scripts import validate_reports_schema


BASE8 = [
    "ret_1m",
    "hl_range",
    "co_change",
    "vol_log1p",
    "pos_log1p",
    "minute_sin",
    "minute_cos",
    "is_session_open",
]


def _write_base_bars(root, symbol="DCE.JM", n_rows=24):
    symbol_dir = root / symbol
    symbol_dir.mkdir(parents=True)
    eobs = pd.date_range("2024-01-02 09:00:00", periods=n_rows, freq="1min")
    data = {
        "eob": eobs,
        "open": np.linspace(100.0, 101.0, n_rows),
        "high": np.linspace(100.1, 101.1, n_rows),
        "low": np.linspace(99.9, 100.9, n_rows),
        "close": np.linspace(100.0, 102.0, n_rows),
        "volume": np.arange(n_rows, dtype=np.float64) + 1,
        "amount": np.arange(n_rows, dtype=np.float64) + 10,
        "vwap": np.linspace(100.0, 102.0, n_rows),
        "position": np.arange(n_rows, dtype=np.float64) + 100,
        "segment_id": np.ones(n_rows, dtype=np.int32),
    }
    for col in BASE8:
        data[col] = np.zeros(n_rows, dtype=np.float32)
    bars = pd.DataFrame(data)
    bars.to_parquet(symbol_dir / "bars.parquet", index=False)
    return bars


def _write_processed_jsonl(path, eobs, export_symbol="CDG.JM"):
    rows = []
    with path.open("w", encoding="utf-8") as f:
        for i, eob in enumerate(eobs):
            row = {
                "symbol": export_symbol,
                "as_of": str(eob),
                "schema_version": "processed_market_snapshot.v1",
                "feature_vector_version": "processed_feature_vector.v1",
                "feature_vector": {
                    "daily_trend_strength": float(i) / 100.0,
                    "h1_distance_to_structure": float(i) / 200.0,
                    "m5_macd_hist": float(i) / 300.0,
                    "minute_behavior_count": float(i % 3),
                },
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
    return rows


def _write_processed_jsonl_with_as_of(path, as_of_values, export_symbol="CDG.JM"):
    with path.open("w", encoding="utf-8") as f:
        for i, as_of in enumerate(as_of_values):
            row = {
                "symbol": export_symbol,
                "as_of": str(as_of),
                "schema_version": "processed_market_snapshot.v1",
                "feature_vector_version": "processed_feature_vector.v1",
                "feature_vector": {
                    "daily_trend_strength": float(i) / 100.0,
                    "h1_distance_to_structure": float(i) / 200.0,
                    "m5_macd_hist": float(i) / 300.0,
                    "minute_behavior_count": float(i % 3),
                },
            }
            f.write(json.dumps(row) + "\n")


def test_m12_builder_contract_passes_on_synthetic_processed_export(tmp_path):
    base_dir = tmp_path / "m1_f8"
    bars = _write_base_bars(base_dir)
    processed = tmp_path / "processed.jsonl"
    _write_processed_jsonl(processed, bars["eob"])
    symbol_map = tmp_path / "symbol_map.json"
    symbol_map.write_text(
        json.dumps(
            {
                "DCE.JM": {
                    "chendage_export_symbol": "CDG.JM",
                    "source_symbol": "jm2409",
                    "symbol_type": "explicit_contract_snapshot",
                    "roll_policy": "external_continuous_mapping_recorded",
                    "contract_map": {"2024-01-02": "jm2409"},
                }
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "m12_chendage_f12"
    control_dir = tmp_path / "m12_common_base8"
    reports_dir = tmp_path / "reports"

    argv = [
        "--base-dir", str(base_dir),
        "--chendage-input", str(processed),
        "--truncated-chendage-input", str(processed),
        "--mutated-chendage-input", str(processed),
        "--symbol-map", str(symbol_map),
        "--symbols", "DCE.JM",
        "--output-dir", str(output_dir),
        "--control-output-dir", str(control_dir),
        "--reports-dir", str(reports_dir),
        "--feature-profile-id", "m12_chg_core",
        "--lookback", "3",
        "--stride", "1",
        "--horizons", "1",
        "--train-start", "2024-01-02 09:00:00",
        "--train-end", "2024-01-02 09:12:00",
        "--val-start", "2024-01-02 09:12:00",
        "--val-end", "2024-01-02 09:18:00",
        "--test-start", "2024-01-02 09:18:00",
        "--test-end", "2024-01-02 10:00:00",
    ]
    old_argv = m12.sys.argv
    try:
        m12.sys.argv = ["build_m12_chendage_features.py"] + argv
        m12.main()
    finally:
        m12.sys.argv = old_argv

    candidate_bars = pd.read_parquet(output_dir / "DCE.JM" / "bars.parquet")
    control_bars = pd.read_parquet(control_dir / "DCE.JM" / "bars.parquet")
    assert len(candidate_bars) == len(control_bars) == len(bars)
    assert "chg.h1_distance_to_structure" in candidate_bars.columns
    assert "chg.h1_distance_to_structure" not in control_bars.columns
    assert (output_dir / "dataset_config.yaml").exists()
    assert (control_dir / "dataset_config.yaml").exists()

    report_path = reports_dir / "m12_chendage_feature_contract.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "PASS"
    assert report["feature_profile"]["feature_dim"] > 8

    schema_path = "src/alphatrade/schemas/m12_chendage_feature_contract.schema.json"
    schema_result = validate_reports_schema.validate_report(str(report_path), schema_path)
    assert schema_result["status"] == "pass", schema_result.get("error")

    semantic = validate_reports_schema.semantic_check_m12(str(reports_dir))
    assert semantic["all_pass"], [c for c in semantic["checks"] if c["status"] != "pass"]


def test_m12_builder_aligns_timezone_aware_processed_export(tmp_path):
    base_dir = tmp_path / "m1_f8"
    bars = _write_base_bars(base_dir)
    processed = tmp_path / "processed_utc.jsonl"
    utc_as_of = (
        pd.to_datetime(bars["eob"])
        .dt.tz_localize("Asia/Shanghai")
        .dt.tz_convert("UTC")
    )
    _write_processed_jsonl_with_as_of(processed, utc_as_of)
    symbol_map = tmp_path / "symbol_map.json"
    symbol_map.write_text(
        json.dumps({"DCE.JM": {"chendage_export_symbol": "CDG.JM", "timezone": "Asia/Shanghai"}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "m12_chendage_f12"
    control_dir = tmp_path / "m12_common_base8"
    reports_dir = tmp_path / "reports"

    old_argv = m12.sys.argv
    try:
        m12.sys.argv = [
            "build_m12_chendage_features.py",
            "--base-dir", str(base_dir),
            "--chendage-input", str(processed),
            "--truncated-chendage-input", str(processed),
            "--mutated-chendage-input", str(processed),
            "--symbol-map", str(symbol_map),
            "--symbols", "DCE.JM",
            "--output-dir", str(output_dir),
            "--control-output-dir", str(control_dir),
            "--reports-dir", str(reports_dir),
            "--feature-profile-id", "m12_chg_core",
            "--lookback", "3",
            "--stride", "1",
            "--horizons", "1",
            "--train-start", "2024-01-02 09:00:00",
            "--train-end", "2024-01-02 09:12:00",
            "--val-start", "2024-01-02 09:12:00",
            "--val-end", "2024-01-02 09:18:00",
            "--test-start", "2024-01-02 09:18:00",
            "--test-end", "2024-01-02 10:00:00",
        ]
        m12.main()
    finally:
        m12.sys.argv = old_argv

    report = json.loads((reports_dir / "m12_chendage_feature_contract.json").read_text())
    assert report["overall_status"] == "PASS"
    assert report["symbols"][0]["common_rows"] == len(bars)


def test_m12_builder_rejects_sparse_processed_feature_coverage(tmp_path):
    base_dir = tmp_path / "m1_f8"
    bars = _write_base_bars(base_dir)
    processed = tmp_path / "processed_sparse.jsonl"
    _write_processed_jsonl(processed, bars["eob"].iloc[::2])
    symbol_map = tmp_path / "symbol_map.json"
    symbol_map.write_text(
        json.dumps({"DCE.JM": {"chendage_export_symbol": "CDG.JM"}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "m12_chendage_f12"
    control_dir = tmp_path / "m12_common_base8"
    reports_dir = tmp_path / "reports"

    old_argv = m12.sys.argv
    try:
        m12.sys.argv = [
            "build_m12_chendage_features.py",
            "--base-dir", str(base_dir),
            "--chendage-input", str(processed),
            "--truncated-chendage-input", str(processed),
            "--mutated-chendage-input", str(processed),
            "--symbol-map", str(symbol_map),
            "--symbols", "DCE.JM",
            "--output-dir", str(output_dir),
            "--control-output-dir", str(control_dir),
            "--reports-dir", str(reports_dir),
            "--feature-profile-id", "m12_chg_core",
            "--lookback", "3",
            "--stride", "1",
            "--horizons", "1",
            "--train-start", "2024-01-02 09:00:00",
            "--train-end", "2024-01-02 09:12:00",
            "--val-start", "2024-01-02 09:12:00",
            "--val-end", "2024-01-02 09:18:00",
            "--test-start", "2024-01-02 09:18:00",
            "--test-end", "2024-01-02 10:00:00",
        ]
        with pytest.raises(SystemExit):
            m12.main()
    finally:
        m12.sys.argv = old_argv

    report = json.loads((reports_dir / "m12_chendage_feature_contract.json").read_text())
    assert report["overall_status"] == "FAIL"
    result = report["symbols"][0]
    assert result["status"] == "FAIL"
    assert result["feature_coverage"]["status"] == "FAIL"
    assert result["feature_coverage"]["missing_feature_rate"] > 0


def test_m12_source_schema_versions_rejects_mixed_versions():
    observed = {
        "schema_version": ["legacy_snapshot.v0", "processed_market_snapshot.v1"],
        "feature_vector_version": ["processed_feature_vector.v1"],
    }

    assert not m12.source_schema_versions_match_expected(
        observed,
        expected_schema_version="processed_market_snapshot.v1",
        expected_feature_vector_version="processed_feature_vector.v1",
    )


def test_m12_flatten_processed_rows_detects_rule_only_fields():
    rows, meta = m12.flatten_processed_rows(
        [
            {
                "symbol": "CDG.JM",
                "as_of": "2024-01-02 09:00:00",
                "rating": "A",
                "feature_vector": {"h1_distance": 0.1, "candidate_label": 1},
            }
        ]
    )

    assert rows[0]["features"]["h1_distance"] == 0.1
    assert "rating" in meta["rule_only_fields_found"]
    assert "candidate_label" in meta["rule_only_fields_found"]


def test_m12_feature_group_ablation_excludes_group(tmp_path):
    base_dir = tmp_path / "m1_f8"
    bars = _write_base_bars(base_dir)
    processed = tmp_path / "processed.jsonl"
    _write_processed_jsonl(processed, bars["eob"])
    symbol_map = tmp_path / "symbol_map.json"
    symbol_map.write_text(
        json.dumps({"DCE.JM": {"chendage_export_symbol": "CDG.JM"}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "m12_chendage_no_h1"
    control_dir = tmp_path / "m12_common_base8"
    reports_dir = tmp_path / "reports"

    old_argv = m12.sys.argv
    try:
        m12.sys.argv = [
            "build_m12_chendage_features.py",
            "--base-dir", str(base_dir),
            "--chendage-input", str(processed),
            "--truncated-chendage-input", str(processed),
            "--mutated-chendage-input", str(processed),
            "--symbol-map", str(symbol_map),
            "--symbols", "DCE.JM",
            "--output-dir", str(output_dir),
            "--control-output-dir", str(control_dir),
            "--reports-dir", str(reports_dir),
            "--feature-profile-id", "m12_chg_core_no_h1",
            "--exclude-feature-groups", "h1",
            "--lookback", "3",
            "--stride", "1",
            "--horizons", "1",
            "--train-start", "2024-01-02 09:00:00",
            "--train-end", "2024-01-02 09:12:00",
            "--val-start", "2024-01-02 09:12:00",
            "--val-end", "2024-01-02 09:18:00",
            "--test-start", "2024-01-02 09:18:00",
            "--test-end", "2024-01-02 10:00:00",
        ]
        m12.main()
    finally:
        m12.sys.argv = old_argv

    candidate_bars = pd.read_parquet(output_dir / "DCE.JM" / "bars.parquet")
    assert "chg.h1_distance_to_structure" not in candidate_bars.columns
    assert "chg.daily_trend_strength" in candidate_bars.columns
    assert "chg.m5_macd_hist" in candidate_bars.columns
    report = json.loads((reports_dir / "m12_chendage_feature_contract.json").read_text())
    assert report["feature_profile"]["excluded_feature_groups"] == ["h1"]
    assert "h1" not in report["feature_profile"]["feature_groups"]
