import json

import pytest

from alphatrade.scripts import build_m1_canonical_bars_v2
from alphatrade.data_pipeline.feature_profiles import (
    assert_feature_profiles_match,
    feature_profile_from_mapping,
    resolve_feature_profile,
)


def test_resolve_feature_profile_from_dataset_config():
    profile = resolve_feature_profile(
        {
            "paths": {"processed_dir": "data/processed/m12_chendage_f12"},
            "feature_profile": {
                "profile_id": "m12_chg_core",
                "feature_dim": 3,
                "feature_cols": ["a", "b", "c"],
            },
        }
    )

    assert profile.profile_id == "m12_chg_core"
    assert profile.feature_cols == ("a", "b", "c")
    assert profile.feature_dim == 3
    assert profile.processed_root == "data/processed/m12_chendage_f12"
    assert profile.fingerprint


def test_resolve_feature_profile_requires_explicit_profile():
    with pytest.raises(ValueError, match="missing required feature_profile"):
        resolve_feature_profile({"paths": {"processed_dir": "data/processed/m1_f8"}})


def test_feature_profile_requires_profile_id():
    with pytest.raises(ValueError, match="missing profile_id"):
        feature_profile_from_mapping({"feature_dim": 1, "feature_cols": ["a"]})


def test_feature_profile_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="feature_dim mismatch"):
        feature_profile_from_mapping(
            {
                "profile_id": "bad",
                "feature_dim": 2,
                "feature_cols": ["a"],
            }
        )


def test_assert_feature_profiles_match_detects_column_drift():
    expected = feature_profile_from_mapping(
        {"profile_id": "p", "feature_dim": 2, "feature_cols": ["a", "b"]}
    )
    observed = feature_profile_from_mapping(
        {"profile_id": "p", "feature_dim": 2, "feature_cols": ["a", "c"]}
    )

    with pytest.raises(ValueError, match="feature_profile_mismatch"):
        assert_feature_profiles_match(expected, observed, context="unit")


def test_assert_feature_profiles_match_requires_scaler_hash_when_expected():
    expected = feature_profile_from_mapping(
        {
            "profile_id": "p",
            "feature_dim": 2,
            "feature_cols": ["a", "b"],
            "scaler_hash": "scale123",
        }
    )
    observed = feature_profile_from_mapping(
        {"profile_id": "p", "feature_dim": 2, "feature_cols": ["a", "b"]}
    )

    with pytest.raises(ValueError, match="scaler_hash"):
        assert_feature_profiles_match(expected, observed, context="unit")


def test_m1_root_manifests_record_base_feature_profile(tmp_path):
    output_dir = tmp_path / "m1_f8"
    reports_dir = tmp_path / "reports"
    build_m1_canonical_bars_v2.write_root_manifests(
        [{"csymbol": "DCE.JM", "status": "SKIP"}],
        input_dir="data/processed/m1",
        output_dir=str(output_dir),
        reports_dir=reports_dir,
    )

    manifest = json.loads((output_dir / "feature_manifest.json").read_text())
    assert manifest["feature_profile"]["profile_id"] == "m1_f8"
    assert manifest["feature_profile"]["feature_dim"] == 8
    assert manifest["symbols"] == ["DCE.JM"]
