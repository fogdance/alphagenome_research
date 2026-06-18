import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_m12_chendage_db_formal.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
sys.path.insert(0, str(SCRIPT_PATH.parents[1]))
spec = importlib.util.spec_from_file_location("build_m12_chendage_db_formal", SCRIPT_PATH)
m12_db = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m12_db)


def _arrow_schema(frame: pd.DataFrame) -> pa.Schema:
    normalized = m12_db.normalize_processed_frame_for_parquet(frame)
    return pa.Table.from_pandas(normalized, preserve_index=False).replace_schema_metadata(None).schema


def test_processed_chunk_schema_is_stable_when_nulls_later_become_values():
    first_chunk = pd.DataFrame(
        {
            "symbol": ["DCE.JM"],
            "as_of": ["2024-01-02 09:01:00"],
            "h1_active_support_role": [None],
            "h1_distance_to_active_support": [None],
            "m5_bars_since_cross": [None],
            "h1_is_near_support": [False],
            "count_1m": [1],
            "feature.h1_distance_to_active_support": [0.0],
        }
    )
    later_chunk = pd.DataFrame(
        {
            "symbol": ["DCE.JM"],
            "as_of": ["2024-01-02 09:02:00"],
            "h1_active_support_role": ["support"],
            "h1_distance_to_active_support": [1.25],
            "m5_bars_since_cross": [3],
            "h1_is_near_support": [True],
            "count_1m": [2],
            "feature.h1_distance_to_active_support": [1.25],
        }
    )

    assert _arrow_schema(first_chunk) == _arrow_schema(later_chunk)


def test_processed_chunk_schema_rejects_non_numeric_feature_values():
    frame = pd.DataFrame(
        {
            "symbol": ["DCE.JM"],
            "as_of": ["2024-01-02 09:01:00"],
            "feature.bad": ["not-a-number"],
        }
    )

    with pytest.raises(ValueError, match="Unable to parse string"):
        m12_db.normalize_processed_frame_for_parquet(frame)


def test_fit_scaler_from_parquet_fits_exact_per_feature_params(tmp_path):
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2024-01-01", "2024-01-02", "2025-01-01"]),
            "feature.x": [1.0, 3.0, 100.0],
            "feature.flag": [0.0, 1.0, 1.0],
        }
    ).to_parquet(path_a, index=False)
    pd.DataFrame(
        {
            "as_of": pd.to_datetime(["2024-01-01", "2024-01-02", "2025-01-01"]),
            "feature.x": [5.0, 7.0, 200.0],
            "feature.flag": [0.0, 1.0, 0.0],
        }
    ).to_parquet(path_b, index=False)

    scaler = m12_db.fit_scaler_from_parquet(
        processed_paths={"A": str(path_a), "B": str(path_b)},
        feature_keys=["x", "flag"],
        train_start="2024-01-01",
        train_end="2024-12-31",
    )

    assert scaler["policy"] == "train_split_robust_zscore"
    assert scaler["params"]["x"]["median"] == 4.0
    assert scaler["params"]["x"]["scale"] == 3.0
    assert scaler["params"]["flag"]["transform"] == "identity_binary"
