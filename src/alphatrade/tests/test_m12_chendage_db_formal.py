import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_m12_chendage_db_formal.py"
)
SPOTCHECK_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_m12_formal_data_contract_spotcheck.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
sys.path.insert(0, str(SCRIPT_PATH.parents[1]))
spec = importlib.util.spec_from_file_location("build_m12_chendage_db_formal", SCRIPT_PATH)
m12_db = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(m12_db)
spotcheck_spec = importlib.util.spec_from_file_location(
    "build_m12_formal_data_contract_spotcheck",
    SPOTCHECK_PATH,
)
m12_spotcheck = importlib.util.module_from_spec(spotcheck_spec)
assert spotcheck_spec.loader is not None
spotcheck_spec.loader.exec_module(m12_spotcheck)


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


@dataclass(frozen=True)
class _FakeProcessedSnapshot:
    index: int

    def to_dict(self, *, include_features: bool = False):
        as_of = datetime(2024, 1, 2, 9, 1) + timedelta(minutes=self.index)
        payload = {
            "symbol": "DCE.JM",
            "as_of": as_of.strftime("%Y-%m-%d %H:%M:%S"),
            "current_datetime": as_of.strftime("%Y-%m-%d %H:%M:%S"),
            "h1_active_support_role": None if self.index == 0 else "SUPPORT",
            "h1_distance_to_active_support": None if self.index == 0 else 1.25,
            "m5_bars_since_cross": None if self.index == 0 else self.index,
            "h1_is_near_support": self.index % 2 == 0,
            "count_1m": self.index + 1,
        }
        if include_features:
            payload["feature_vector"] = self.to_feature_dict()
        return payload

    def to_feature_dict(self):
        return {
            "h1_distance_to_active_support": 0.0 if self.index == 0 else 1.25,
            "m5_bars_since_cross": float(self.index),
        }


def test_write_snapshots_parquet_chunked_streams_generator(tmp_path):
    output_path = tmp_path / "processed.parquet"

    summary = m12_db.write_snapshots_parquet_chunked(
        (_FakeProcessedSnapshot(index) for index in range(5)),
        output_path=output_path,
        chunk_size=2,
    )
    frame = pd.read_parquet(output_path)

    assert summary["snapshots"] == 5
    assert summary["write_chunk_size"] == 2
    assert summary["columns"] == len(frame.columns)
    assert len(frame) == 5
    assert "feature.h1_distance_to_active_support" in frame.columns
    assert "feature_vector" not in frame.columns


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


def _export_request_args() -> m12_db.argparse.Namespace:
    return m12_db.argparse.Namespace(
        source_start="2017-12-01",
        source_end="2026-06-17",
        eval_start="2018-01-01",
        eval_end="2026-06-17",
        train_start="2018-01-01",
        train_end="2023-01-01",
        val_start="2023-01-01",
        val_end="2024-01-01",
        test_start="2024-01-01",
        test_end="2026-06-18",
        chendage_src=str(Path.cwd()),
    )


def test_reuse_existing_processed_requires_matching_fingerprint(tmp_path):
    processed_path = tmp_path / "processed.parquet"
    base_path = tmp_path / "bars.parquet"
    processed_path.write_bytes(b"processed")
    base_path.write_bytes(b"base")
    processed_paths = {"DCE.JM": str(processed_path)}
    base_paths = {"DCE.JM": str(base_path)}
    args = _export_request_args()
    expected = m12_db.export_request_payload(
        symbols=["DCE.JM"],
        args=args,
        chendage_commit="abc123",
    )
    summary = {
        "symbols": ["DCE.JM"],
        "date_ranges": dict(expected["date_ranges"]),
        "chendage_commit": "abc123",
        "file_hashes": m12_db.export_file_hashes(
            processed_paths=processed_paths,
            base_paths=base_paths,
        ),
    }

    m12_db.validate_reused_export_summary(
        export_summary=summary,
        expected_request=expected,
        processed_paths=processed_paths,
        base_paths=base_paths,
    )

    stale = {
        **summary,
        "date_ranges": {**summary["date_ranges"], "eval_end": "2025-01-01"},
    }
    with pytest.raises(ValueError, match="reuse_existing_processed_config_mismatch"):
        m12_db.validate_reused_export_summary(
            export_summary=stale,
            expected_request=expected,
            processed_paths=processed_paths,
            base_paths=base_paths,
        )


def test_write_formal_sweep_config_uses_explicit_training_budget(tmp_path):
    control_config = tmp_path / "control" / "dataset_config.yaml"
    control_config.parent.mkdir()
    control_config.write_text("dataset: control\n", encoding="utf-8")
    candidate_dir = tmp_path / "candidate"
    args = m12_db.argparse.Namespace(
        sweep_max_steps=777,
        sweep_batch_size=64,
        sweep_save_every=111,
        sweep_keep_last=9,
        sweep_eval_split="test",
        sweep_ckpt_step="latest",
    )

    path = m12_db.write_formal_sweep_config(
        tmp_path,
        dataset_artifacts=[
            {
                "spec": {
                    "exp_id": "chg_core_common_rows",
                    "description": "candidate",
                },
                "output_dir": candidate_dir,
            }
        ],
        control_config=control_config,
        args=args,
        seeds=[7, 8],
    )
    config = m12_db.yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["expected_seeds"] == [7, 8]
    assert config["defaults"]["max_steps"] == 777
    assert config["defaults"]["batch_size"] == 64
    assert config["defaults"]["save_every"] == 111
    assert config["defaults"]["keep_last"] == 9
    assert config["defaults"]["eval_split"] == "test"
    assert config["defaults"]["ckpt_step"] == "latest"


def test_formal_spotcheck_rejects_index_files_without_required_compare_columns(tmp_path):
    candidate_root = tmp_path / "candidate"
    control_root = tmp_path / "control"
    for root in (candidate_root, control_root):
        symbol_dir = root / "DCE.JM"
        symbol_dir.mkdir(parents=True)
        for split in ("train", "val", "test"):
            pd.DataFrame({"t": [1, 2, 3]}).to_parquet(
                symbol_dir / f"index_{split}.parquet",
                index=False,
            )

    result = m12_spotcheck.compare_indices(candidate_root, control_root, ["DCE.JM"])

    assert result["status"] == "FAIL"
    assert result["failures"][0]["reason"] == "missing_required_index_compare_columns"
