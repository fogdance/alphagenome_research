#!/usr/bin/env python3
"""Build M12 Chendage processed-feature datasets.

This script consumes only the processed numeric feature-vector export boundary
from chendage-signal-engine. It never consumes rule labels, ratings, actions,
candidate decisions, explanations, human annotations, or outcome-review fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime_paths
from data_pipeline.feature_profiles import (
    BASE_FEATURE_COLS,
    FeatureProfile,
    feature_profile_to_dict,
)


SCHEMA_VERSION = "m12_chendage_feature_contract_v1"
SOURCE_SCHEMA_VERSION = "m12_chendage_features_v1"
RULE_ONLY_PATTERNS = (
    "rating",
    "score",
    "grade",
    "action",
    "candidate",
    "reason",
    "hard_block",
    "human",
    "outcome",
    "review",
    "entry_plan",
    "entryplan",
    "trade_plan",
    "teacher",
    "label",
)
METADATA_KEYS = {
    "symbol",
    "contract_symbol",
    "continuous_symbol",
    "alphatrade_symbol",
    "chendage_symbol",
    "as_of",
    "eob",
    "timestamp",
    "datetime",
    "schema_version",
    "feature_vector_version",
    "processed_schema_version",
    "feature_vector",
    "processed_feature_vector",
    "features",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M12: build Chendage processed feature datasets")
    parser.add_argument("--base-dir", default="data/processed/m1_f8", help="Base AlphaTrade processed root")
    parser.add_argument("--chendage-input", required=True, help="Processed feature export file (.jsonl/.json/.csv/.parquet)")
    parser.add_argument("--truncated-chendage-input", default=None, help="Processed export from source truncated at as_of")
    parser.add_argument("--mutated-chendage-input", default=None, help="Processed export after mutating rows after as_of")
    parser.add_argument("--chendage-commit", default="unknown", help="Chendage source commit used for processed export")
    parser.add_argument("--expected-schema-version", default="processed_market_snapshot.v1")
    parser.add_argument("--expected-feature-vector-version", default="processed_feature_vector.v1")
    parser.add_argument("--symbol-map", required=True, help="JSON/YAML mapping from AlphaTrade symbol to Chendage export symbol")
    parser.add_argument("--symbols", default=None, help="Comma-separated AlphaTrade symbols")
    parser.add_argument("--universe", default=None, help="Universe YAML with candidates")
    parser.add_argument("--output-dir", default="data/processed/m12_chendage_fN", help="Candidate output root")
    parser.add_argument("--control-output-dir", default="data/processed/m12_common_base8", help="Common-row base8 control output root")
    parser.add_argument("--feature-profile-id", default="m12_chg_core")
    parser.add_argument("--feature-set", choices=["core", "full"], default="core")
    parser.add_argument("--feature-keys", default=None, help="Optional comma-separated raw Chendage feature keys")
    parser.add_argument(
        "--exclude-feature-groups",
        default="",
        help="Comma-separated Chendage feature groups to drop: daily,h1,m5,minute_behavior,other",
    )
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--horizons", default="1,5,20,60")
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-end", default="2023-01-01")
    parser.add_argument("--val-start", default="2023-01-01")
    parser.add_argument("--val-end", default="2024-01-01")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2026-01-01")
    parser.add_argument("--causality-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--max-missing-feature-rate",
        type=float,
        default=0.0,
        help="Maximum allowed missing Chendage feature row/sample rate over requested splits",
    )
    parser.add_argument(
        "--require-causality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require truncated and post-t mutation processed exports to match",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing symbol outputs")
    runtime_paths.add_output_args(parser)
    return parser.parse_args()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_list(value: str | None, *, cast=str) -> list:
    if not value:
        return []
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def normalize_timestamp(value: Any, timezone: str | None) -> pd.Timestamp:
    """Normalize timestamps to AlphaTrade's naive local-time convention."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        target_tz = timezone or "Asia/Shanghai"
        return ts.tz_convert(target_tz).tz_localize(None)
    return ts


def source_schema_versions_match_expected(
    observed_versions: Mapping[str, Any],
    *,
    expected_schema_version: str,
    expected_feature_vector_version: str,
) -> bool:
    """Require source exports to contain only the expected processed versions."""
    return (
        set(observed_versions.get("schema_version", [])) == {expected_schema_version}
        and set(observed_versions.get("feature_vector_version", []))
        == {expected_feature_vector_version}
    )


def load_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return csv_list(args.symbols)
    if args.universe:
        with open(args.universe, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return list(data.get("candidates", []))
    raise SystemExit("ERROR: either --symbols or --universe is required")


def load_symbol_map(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)
    mapping = raw.get("symbols", raw) if isinstance(raw, dict) else raw
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("symbol map must be a non-empty mapping")
    result = {}
    for alpha_symbol, spec in mapping.items():
        if isinstance(spec, str):
            spec = {"chendage_export_symbol": spec}
        if not isinstance(spec, dict):
            raise ValueError(f"invalid symbol map entry for {alpha_symbol}: {spec!r}")
        export_symbol = spec.get("chendage_export_symbol") or spec.get("source_symbol") or spec.get("symbol")
        if not export_symbol:
            raise ValueError(f"symbol map entry for {alpha_symbol} missing chendage_export_symbol")
        result[str(alpha_symbol)] = {
            "alphatrade_symbol": str(alpha_symbol),
            "chendage_export_symbol": str(export_symbol),
            "source_symbol": str(spec.get("source_symbol") or export_symbol),
            "symbol_type": str(spec.get("symbol_type") or "explicit"),
            "contract_map": spec.get("contract_map") or {},
            "roll_policy": spec.get("roll_policy") or "recorded_external_to_builder",
            "timezone": spec.get("timezone") or "Asia/Shanghai",
        }
    return result


def _is_rule_only_name(name: str) -> bool:
    lower = name.lower()
    return any(pattern in lower for pattern in RULE_ONLY_PATTERNS)


def _is_numeric(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return False


def _parse_feature_vector(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("feature_vector JSON must decode to object")
        return parsed
    return None


def read_processed_export(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        for key in ("rows", "records", "features", "data"):
            if isinstance(data, dict) and isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"JSON export {path} must be a list or contain rows/records/features/data")
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError(f"unsupported processed export suffix: {path.suffix}")


def flatten_processed_rows(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rule_only_found: set[str] = set()
    non_numeric_features: set[str] = set()
    source_schema_versions: dict[str, set[str]] = defaultdict(set)

    for record in records:
        top_rule_fields = [k for k in record.keys() if _is_rule_only_name(str(k))]
        rule_only_found.update(top_rule_fields)

        export_symbol = (
            record.get("symbol")
            or record.get("chendage_symbol")
            or record.get("contract_symbol")
            or record.get("continuous_symbol")
        )
        as_of = record.get("as_of") or record.get("eob") or record.get("timestamp") or record.get("datetime")
        if export_symbol is None or as_of is None:
            raise ValueError("processed export row missing symbol/as_of")

        raw_vector = (
            _parse_feature_vector(record.get("processed_feature_vector"))
            or _parse_feature_vector(record.get("feature_vector"))
            or _parse_feature_vector(record.get("features"))
        )
        if raw_vector is None:
            raw_vector = {
                str(k): v
                for k, v in record.items()
                if str(k) not in METADATA_KEYS and not _is_rule_only_name(str(k))
            }

        numeric_vector: dict[str, float] = {}
        for key, value in raw_vector.items():
            key = str(key)
            if _is_rule_only_name(key):
                rule_only_found.add(key)
                continue
            if _is_numeric(value):
                numeric_vector[key] = float(value)
            else:
                non_numeric_features.add(key)

        if not numeric_vector:
            raise ValueError(f"processed export row for {export_symbol}@{as_of} has no numeric features")

        for key in ("schema_version", "feature_vector_version", "processed_schema_version"):
            if record.get(key) is not None:
                source_schema_versions[key].add(str(record[key]))

        rows.append(
            {
                "export_symbol": str(export_symbol),
                "as_of": pd.to_datetime(as_of),
                "features": numeric_vector,
            }
        )

    metadata = {
        "rule_only_fields_found": sorted(rule_only_found),
        "non_numeric_features_ignored": sorted(non_numeric_features),
        "source_schema_versions": {
            key: sorted(values) for key, values in source_schema_versions.items()
        },
    }
    return rows, metadata


def _looks_like_raw_price_feature(key: str) -> bool:
    lower = key.lower()
    if any(token in lower for token in ("distance", "ratio", "pct", "return", "ret", "change", "slope")):
        return False
    if lower in {"open", "high", "low", "close", "price", "current_close", "current_price"}:
        return True
    return lower.endswith(("_open", "_high", "_low", "_close", "_price"))


def select_chendage_feature_keys(
    rows: list[dict[str, Any]],
    *,
    feature_set: str,
    exclude_groups: list[str] | None = None,
    explicit_keys: list[str] | None = None,
) -> list[str]:
    all_keys = sorted({key for row in rows for key in row["features"].keys()})
    if explicit_keys:
        missing = sorted(set(explicit_keys) - set(all_keys))
        if missing:
            raise ValueError(f"explicit feature keys missing from processed export: {missing}")
        selected = list(explicit_keys)
        excluded = set(exclude_groups or [])
        if excluded:
            selected = [key for key in selected if feature_group_for_key(key) not in excluded]
        if not selected:
            raise ValueError("feature selection produced zero features after group exclusions")
        return selected
    if feature_set == "full":
        selected = [key for key in all_keys if not _looks_like_raw_price_feature(key)]
        excluded = set(exclude_groups or [])
        if excluded:
            selected = [key for key in selected if feature_group_for_key(key) not in excluded]
        if not selected:
            raise ValueError("feature selection produced zero features after group exclusions")
        return selected

    core_markers = ("daily", "h1", "m5", "minute")
    selected = [
        key for key in all_keys
        if any(marker in key.lower() for marker in core_markers)
        and not _looks_like_raw_price_feature(key)
    ]
    excluded = set(exclude_groups or [])
    if excluded:
        selected = [key for key in selected if feature_group_for_key(key) not in excluded]
    if not selected:
        raise ValueError(
            "m12_chg_core selected zero features; pass --feature-set full or --feature-keys"
        )
    return selected


def feature_group_for_key(key: str) -> str:
    lower = key.lower()
    if "daily" in lower or lower.startswith("d1_"):
        return "daily"
    if "h1" in lower or lower.startswith("hour") or "hourly" in lower:
        return "h1"
    if "m5" in lower or "macd" in lower:
        return "m5"
    if "minute" in lower or lower.startswith("m1_"):
        return "minute_behavior"
    return "other"


def feature_groups_for_keys(keys: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        groups[feature_group_for_key(key)].append(key)
    return {group: sorted(values) for group, values in sorted(groups.items())}


def safe_feature_column(raw_key: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", raw_key.strip()).strip("_").lower()
    if not safe:
        raise ValueError(f"invalid empty feature key after sanitization: {raw_key!r}")
    return f"chg.{safe}"


def feature_key_column_map(keys: list[str]) -> dict[str, str]:
    mapping = {key: safe_feature_column(key) for key in keys}
    reverse: dict[str, list[str]] = defaultdict(list)
    for key, col in mapping.items():
        reverse[col].append(key)
    duplicates = {col: vals for col, vals in reverse.items() if len(vals) > 1}
    if duplicates:
        raise ValueError(f"sanitized Chendage feature column names collide: {duplicates}")
    return mapping


def rows_for_export_symbol(rows: list[dict[str, Any]], export_symbol: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["export_symbol"] == export_symbol]


def processed_rows_to_frame(
    rows: list[dict[str, Any]],
    *,
    feature_keys: list[str],
    key_to_col: dict[str, str],
    timezone: str | None,
) -> pd.DataFrame:
    records = []
    seen: set[pd.Timestamp] = set()
    for row in rows:
        as_of = normalize_timestamp(row["as_of"], timezone)
        if as_of in seen:
            raise ValueError(f"duplicate processed export row at as_of={as_of}")
        seen.add(as_of)
        record = {"eob": as_of}
        for key in feature_keys:
            value = row["features"].get(key, np.nan)
            record[key_to_col[key]] = np.float32(value)
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values("eob").reset_index(drop=True)


def compute_labels(df: pd.DataFrame, horizons: list[int], eps: float = 1e-12) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        future_close = df["close"].shift(-h)
        df[f"y_h{h}"] = np.log((future_close + eps) / (df["close"] + eps))
    return df


def split_by_time(
    index_df: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if index_df.empty:
        empty = index_df.copy()
        empty["split"] = pd.Series(dtype=str)
        return empty, empty.copy(), empty.copy()

    eob = pd.to_datetime(index_df["eob"])
    target_eob = pd.to_datetime(index_df["target_eob"])
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    val_start_dt = pd.to_datetime(val_start)
    val_end_dt = pd.to_datetime(val_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)

    train = index_df[
        (eob >= train_start_dt) & (eob < train_end_dt)
        & (target_eob >= train_start_dt) & (target_eob < train_end_dt)
    ].copy()
    val = index_df[
        (eob >= val_start_dt) & (eob < val_end_dt)
        & (target_eob >= val_start_dt) & (target_eob < val_end_dt)
    ].copy()
    test = index_df[
        (eob >= test_start_dt) & (eob < test_end_dt)
        & (target_eob >= test_start_dt) & (target_eob < test_end_dt)
    ].copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return train, val, test


def generate_common_indices(
    common_df: pd.DataFrame,
    full_base_df: pd.DataFrame,
    *,
    lookback: int,
    horizons: list[int],
    stride: int,
) -> pd.DataFrame:
    max_horizon = max(horizons)
    samples = []
    source_pos = common_df["_m12_source_pos"].to_numpy(dtype=np.int64)
    full_segments = full_base_df["segment_id"].to_numpy()

    for t in range(lookback - 1, len(common_df), stride):
        window_start = t - lookback + 1
        window_source_pos = source_pos[window_start:t + 1]
        if not np.all(np.diff(window_source_pos) == 1):
            continue
        source_t = int(source_pos[t])
        target_pos = source_t + max_horizon
        if target_pos >= len(full_base_df):
            continue
        if full_segments[int(window_source_pos[0])] != full_segments[source_t]:
            continue
        if full_segments[source_t] != full_segments[target_pos]:
            continue
        if any(pd.isna(common_df.iloc[t][f"y_h{h}"]) for h in horizons):
            continue

        sample = {
            "t": t,
            "x_start": window_start,
            "x_end": t,
            "eob": common_df.iloc[t]["eob"],
            "target_eob": full_base_df.iloc[target_pos]["eob"],
            "segment_id": int(common_df.iloc[t]["segment_id"]),
            "_m12_source_pos": source_t,
        }
        for h in horizons:
            sample[f"y_h{h}"] = float(common_df.iloc[t][f"y_h{h}"])
        samples.append(sample)

    columns = ["t", "x_start", "x_end", "eob", "target_eob", "segment_id", "_m12_source_pos"] + [
        f"y_h{h}" for h in horizons
    ]
    return pd.DataFrame(samples, columns=columns)


def compute_feature_coverage(
    *,
    base_labeled: pd.DataFrame,
    feature_frame: pd.DataFrame,
    full_index_df: pd.DataFrame,
    split_indices: dict[str, pd.DataFrame],
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
    max_missing_feature_rate: float,
) -> dict[str, Any]:
    """Measure how much of the requested base row/sample set has feature coverage."""
    eob = pd.to_datetime(base_labeled["eob"])
    split_window = (eob >= pd.to_datetime(train_start)) & (eob < pd.to_datetime(test_end))
    required_rows = base_labeled.loc[split_window, ["eob"]].copy()
    feature_eobs = set(pd.to_datetime(feature_frame["eob"]).tolist())
    missing_row_mask = ~required_rows["eob"].isin(feature_eobs)
    missing_rows = required_rows.loc[missing_row_mask, "eob"]
    required_row_count = int(len(required_rows))
    missing_row_count = int(len(missing_rows))
    row_missing_rate = (
        float(missing_row_count / required_row_count)
        if required_row_count
        else 0.0
    )

    full_train, full_val, full_test = split_by_time(
        full_index_df,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )
    full_splits = {
        "train": full_train,
        "val": full_val,
        "test": full_test,
    }
    possible_samples_by_split = {
        split: int(len(index_df))
        for split, index_df in full_splits.items()
    }
    common_samples_by_split = {
        split: int(len(index_df))
        for split, index_df in split_indices.items()
    }
    possible_samples = int(sum(possible_samples_by_split.values()))
    common_samples = int(sum(common_samples_by_split.values()))

    missing_sample_eobs: list[pd.Timestamp] = []
    missing_samples_by_split: dict[str, int] = {}
    for split, index_df in full_splits.items():
        if "eob" not in index_df.columns or index_df.empty:
            missing_samples_by_split[split] = 0
            continue
        sample_eobs = pd.to_datetime(index_df["eob"])
        missing_mask = ~sample_eobs.isin(feature_eobs)
        missing_values = sample_eobs.loc[missing_mask].tolist()
        missing_samples_by_split[split] = int(len(missing_values))
        missing_sample_eobs.extend(missing_values)

    missing_samples = int(len(missing_sample_eobs))
    sample_missing_rate = (
        float(missing_samples / possible_samples)
        if possible_samples
        else 0.0
    )
    passed = (
        row_missing_rate <= max_missing_feature_rate
        and sample_missing_rate <= max_missing_feature_rate
        and possible_samples > 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "max_missing_feature_rate": float(max_missing_feature_rate),
        "required_rows": required_row_count,
        "matched_rows": int(required_row_count - missing_row_count),
        "missing_rows": missing_row_count,
        "missing_feature_rate": row_missing_rate,
        "missing_eob_sample": [str(x) for x in missing_rows.head(10).tolist()],
        "possible_samples": possible_samples,
        "common_samples": common_samples,
        "missing_samples": int(missing_samples),
        "missing_sample_rate": sample_missing_rate,
        "missing_sample_eob_sample": [str(x) for x in missing_sample_eobs[:10]],
        "possible_samples_by_split": possible_samples_by_split,
        "common_samples_by_split": common_samples_by_split,
        "missing_samples_by_split": missing_samples_by_split,
        "sample_count_delta": int(common_samples - possible_samples),
    }


def check_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    numeric = all(pd.api.types.is_numeric_dtype(frame[col]) for col in feature_cols)
    values = frame[feature_cols].to_numpy(dtype=np.float64, copy=False) if feature_cols else np.empty((len(frame), 0))
    finite = bool(np.isfinite(values).all()) if values.size else True
    missing_rate = float(np.isnan(values).mean()) if values.size else 0.0
    return {"numeric": numeric, "finite": finite, "missing_rate": missing_rate}


def _series_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.50)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def feature_distribution_by_split(
    common_df: pd.DataFrame,
    *,
    split_indices: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> dict[str, Any]:
    """Summarize selected feature distributions on the actual sample eob rows."""
    distribution: dict[str, Any] = {}
    for split, index_df in split_indices.items():
        if index_df.empty:
            distribution[split] = {"samples": 0, "features": {}}
            continue
        t_positions = index_df["t"].astype(int).to_numpy()
        split_frame = common_df.iloc[t_positions]
        distribution[split] = {
            "samples": int(len(index_df)),
            "features": {
                col: _series_stats(split_frame[col].to_numpy())
                for col in feature_cols
            },
        }
    return distribution


def fit_chendage_scaler(
    rows: list[dict[str, Any]],
    *,
    symbol_map: dict[str, dict[str, Any]],
    symbols: list[str],
    feature_keys: list[str],
    train_start: str,
    train_end: str,
) -> dict[str, Any]:
    """Fit deterministic train-only scaling parameters for Chendage features."""
    train_start_ts = pd.to_datetime(train_start)
    train_end_ts = pd.to_datetime(train_end)
    timezone_by_export_symbol = {
        symbol_map[symbol]["chendage_export_symbol"]: symbol_map[symbol].get("timezone") or "Asia/Shanghai"
        for symbol in symbols
    }
    export_symbols = set(timezone_by_export_symbol)
    values_by_key: dict[str, list[float]] = {key: [] for key in feature_keys}
    for row in rows:
        if row["export_symbol"] not in export_symbols:
            continue
        as_of = normalize_timestamp(row["as_of"], timezone_by_export_symbol.get(row["export_symbol"]))
        if as_of < train_start_ts or as_of >= train_end_ts:
            continue
        for key in feature_keys:
            if key in row["features"]:
                values_by_key[key].append(float(row["features"][key]))

    params = {}
    for key, values in values_by_key.items():
        if not values:
            raise ValueError(f"no train-split values available to fit scaler for {key}")
        arr = np.asarray(values, dtype=np.float64)
        unique = np.unique(arr)
        if len(unique) <= 2 and set(unique.tolist()).issubset({0.0, 1.0}):
            params[key] = {"transform": "identity_binary", "median": 0.0, "scale": 1.0}
            continue
        median = float(np.nanmedian(arr))
        q75 = float(np.nanquantile(arr, 0.75))
        q25 = float(np.nanquantile(arr, 0.25))
        scale = q75 - q25
        if not np.isfinite(scale) or abs(scale) < 1e-12:
            scale = float(np.nanstd(arr))
        if not np.isfinite(scale) or abs(scale) < 1e-12:
            scale = 1.0
        params[key] = {"transform": "robust_zscore", "median": median, "scale": float(scale)}

    payload = {
        "policy": "train_split_robust_zscore",
        "train_start": train_start,
        "train_end": train_end,
        "feature_count": len(feature_keys),
        "params": params,
    }
    payload["scaler_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def apply_chendage_scaler(
    feature_frame: pd.DataFrame,
    *,
    feature_keys: list[str],
    key_to_col: dict[str, str],
    scaler: dict[str, Any],
) -> pd.DataFrame:
    frame = feature_frame.copy()
    params = scaler.get("params", {})
    for key in feature_keys:
        col = key_to_col[key]
        item = params[key]
        if item["transform"] == "identity_binary":
            frame[col] = frame[col].astype(np.float32)
        else:
            frame[col] = ((frame[col].astype(np.float64) - item["median"]) / item["scale"]).astype(np.float32)
    return frame


def write_symbol_manifests(
    *,
    symbol_dir: Path,
    source_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
) -> None:
    with (symbol_dir / "source_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(source_manifest, f, indent=2)
    with (symbol_dir / "feature_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, indent=2)


def compare_processed_exports(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    symbol_map: dict[str, dict[str, Any]],
    feature_keys: list[str],
    tolerance: float,
) -> dict[str, Any]:
    timezone_by_export_symbol = {
        spec["chendage_export_symbol"]: spec.get("timezone") or "Asia/Shanghai"
        for spec in symbol_map.values()
    }
    ref_lookup = {
        (
            row["export_symbol"],
            normalize_timestamp(row["as_of"], timezone_by_export_symbol.get(row["export_symbol"])),
        ): row["features"]
        for row in reference_rows
    }
    cand_lookup = {
        (
            row["export_symbol"],
            normalize_timestamp(row["as_of"], timezone_by_export_symbol.get(row["export_symbol"])),
        ): row["features"]
        for row in candidate_rows
    }
    compared = 0
    missing = []
    mismatches = []
    max_abs_diff = 0.0

    for spec in symbol_map.values():
        export_symbol = spec["chendage_export_symbol"]
        keys = sorted(k for k in ref_lookup if k[0] == export_symbol)
        for key in keys:
            if key not in cand_lookup:
                missing.append({"symbol": key[0], "as_of": str(key[1])})
                continue
            ref_features = ref_lookup[key]
            cand_features = cand_lookup[key]
            for feature_key in feature_keys:
                if feature_key not in ref_features or feature_key not in cand_features:
                    missing.append({"symbol": key[0], "as_of": str(key[1]), "feature": feature_key})
                    continue
                diff = abs(float(ref_features[feature_key]) - float(cand_features[feature_key]))
                compared += 1
                max_abs_diff = max(max_abs_diff, diff)
                if diff > tolerance:
                    mismatches.append(
                        {
                            "symbol": key[0],
                            "as_of": str(key[1]),
                            "feature": feature_key,
                            "abs_diff": diff,
                        }
                    )

    passed = not missing and not mismatches and compared > 0
    return {
        "status": "PASS" if passed else "FAIL",
        "compared_values": compared,
        "missing_count": len(missing),
        "mismatch_count": len(mismatches),
        "max_abs_diff": max_abs_diff,
        "missing_sample": missing[:10],
        "mismatch_sample": mismatches[:10],
    }


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str = "", observed: Any = None) -> None:
    checks.append(
        {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "severity": "ERROR" if not passed else "INFO",
            "detail": detail,
            "observed": observed,
        }
    )


def process_symbol(
    *,
    symbol: str,
    symbol_spec: dict[str, Any],
    base_dir: Path,
    output_dir: Path,
    control_output_dir: Path,
    rows: list[dict[str, Any]],
    feature_keys: list[str],
    key_to_col: dict[str, str],
    scaler: dict[str, Any],
    feature_profile: FeatureProfile,
    feature_manifest_root: dict[str, Any],
    horizons: list[int],
    lookback: int,
    stride: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    symbol_base_dir = base_dir / symbol
    bars_path = symbol_base_dir / "bars.parquet"
    if not bars_path.exists():
        return {"symbol": symbol, "status": "FAIL", "error": f"base bars not found: {bars_path}"}

    export_symbol = symbol_spec["chendage_export_symbol"]
    chg_rows = rows_for_export_symbol(rows, export_symbol)
    if not chg_rows:
        return {"symbol": symbol, "status": "FAIL", "error": f"no Chendage rows for {export_symbol}"}

    feature_frame = processed_rows_to_frame(
        chg_rows,
        feature_keys=feature_keys,
        key_to_col=key_to_col,
        timezone=symbol_spec.get("timezone"),
    )
    feature_frame = apply_chendage_scaler(
        feature_frame,
        feature_keys=feature_keys,
        key_to_col=key_to_col,
        scaler=scaler,
    )
    base_df = pd.read_parquet(bars_path).reset_index(drop=True)
    base_df["eob"] = base_df["eob"].map(lambda value: normalize_timestamp(value, symbol_spec.get("timezone")))
    base_df["_m12_source_pos"] = np.arange(len(base_df), dtype=np.int64)
    if "segment_id" not in base_df.columns:
        raise ValueError(f"{symbol}: base bars missing segment_id")

    base_labeled = compute_labels(base_df, horizons)
    full_index_df = generate_common_indices(
        base_labeled,
        base_df,
        lookback=lookback,
        horizons=horizons,
        stride=stride,
    )
    common_df = base_labeled.merge(feature_frame, on="eob", how="inner", validate="one_to_one")
    common_df = common_df.sort_values("eob").reset_index(drop=True)
    if common_df.empty:
        return {"symbol": symbol, "status": "FAIL", "error": "zero common rows after eob alignment"}

    chg_cols = [key_to_col[key] for key in feature_keys]
    feature_check = check_feature_frame(common_df, chg_cols)
    if not (feature_check["numeric"] and feature_check["finite"] and feature_check["missing_rate"] == 0.0):
        return {
            "symbol": symbol,
            "status": "FAIL",
            "error": "feature semantic check failed",
            "feature_check": feature_check,
        }

    index_df = generate_common_indices(
        common_df,
        base_df,
        lookback=lookback,
        horizons=horizons,
        stride=stride,
    )
    train_df, val_df, test_df = split_by_time(
        index_df,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    feature_coverage = compute_feature_coverage(
        base_labeled=base_labeled,
        feature_frame=feature_frame,
        full_index_df=full_index_df,
        split_indices={"train": train_df, "val": val_df, "test": test_df},
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
        max_missing_feature_rate=args.max_missing_feature_rate,
    )
    if feature_coverage["status"] != "PASS":
        return {
            "symbol": symbol,
            "status": "FAIL",
            "error": "feature coverage below required threshold",
            "feature_coverage": feature_coverage,
        }
    split_distribution = feature_distribution_by_split(
        common_df,
        split_indices={"train": train_df, "val": val_df, "test": test_df},
        feature_cols=chg_cols,
    )

    candidate_symbol_dir = output_dir / symbol
    control_symbol_dir = control_output_dir / symbol
    if args.force:
        for directory in (candidate_symbol_dir, control_symbol_dir):
            if directory.exists():
                for path in directory.iterdir():
                    if path.is_file():
                        path.unlink()
    candidate_symbol_dir.mkdir(parents=True, exist_ok=True)
    control_symbol_dir.mkdir(parents=True, exist_ok=True)

    label_cols = [f"y_h{h}" for h in horizons]
    base_bar_cols = [c for c in base_df.columns if c not in label_cols]
    candidate_bars = common_df[base_bar_cols + chg_cols].copy()
    control_bars = common_df[base_bar_cols].copy()
    candidate_bars.to_parquet(candidate_symbol_dir / "bars.parquet", index=False)
    control_bars.to_parquet(control_symbol_dir / "bars.parquet", index=False)
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        split_df.to_parquet(candidate_symbol_dir / f"index_{split_name}.parquet", index=False)
        split_df.to_parquet(control_symbol_dir / f"index_{split_name}.parquet", index=False)

    source_manifest = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "alphatrade_symbol": symbol,
        "chendage_export_symbol": export_symbol,
        "symbol_map": symbol_spec,
        "base_bars_path": str(bars_path),
        "common_rows": int(len(common_df)),
        "first_eob": str(common_df["eob"].min()),
        "last_eob": str(common_df["eob"].max()),
    }
    symbol_feature_manifest = {
        **feature_manifest_root,
        "symbol": symbol,
        "common_rows": int(len(common_df)),
        "split_samples": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "feature_coverage": feature_coverage,
        "feature_distribution_by_split": split_distribution,
    }
    write_symbol_manifests(
        symbol_dir=candidate_symbol_dir,
        source_manifest=source_manifest,
        feature_manifest=symbol_feature_manifest,
    )
    write_symbol_manifests(
        symbol_dir=control_symbol_dir,
        source_manifest={**source_manifest, "control": "base8_common_rows"},
        feature_manifest={
            **symbol_feature_manifest,
            "feature_profile": feature_profile_to_dict(
                FeatureProfile(
                    profile_id="m12_base8_control_common_rows",
                    feature_cols=BASE_FEATURE_COLS,
                    feature_dim=len(BASE_FEATURE_COLS),
                    processed_root=str(control_output_dir),
                    source_schema_versions={"alphatrade_feature_profile": "m1_f8_v1"},
                )
            ),
            "chendage_feature_cols": [],
        },
    )

    return {
        "symbol": symbol,
        "status": "SUCCESS",
        "alphatrade_symbol": symbol,
        "chendage_export_symbol": export_symbol,
        "base_rows": int(len(base_df)),
        "common_rows": int(len(common_df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "feature_check": feature_check,
        "feature_coverage": feature_coverage,
        "feature_distribution_by_split": split_distribution,
        "first_eob": str(common_df["eob"].min()),
        "last_eob": str(common_df["eob"].max()),
    }


def write_root_manifests(
    *,
    output_dir: Path,
    control_output_dir: Path,
    feature_manifest: dict[str, Any],
    source_manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    control_output_dir.mkdir(parents=True, exist_ok=True)
    for root, control_label in ((output_dir, None), (control_output_dir, "base8_common_rows")):
        fm = dict(feature_manifest)
        sm = dict(source_manifest)
        if control_label:
            fm["feature_profile"] = feature_profile_to_dict(
                FeatureProfile(
                    profile_id="m12_base8_control_common_rows",
                    feature_cols=BASE_FEATURE_COLS,
                    feature_dim=len(BASE_FEATURE_COLS),
                    processed_root=str(control_output_dir),
                    source_schema_versions={"alphatrade_feature_profile": "m1_f8_v1"},
                )
            )
            fm["chendage_feature_cols"] = []
            sm["control"] = control_label
        with (root / "feature_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(fm, f, indent=2)
        with (root / "source_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(sm, f, indent=2)


def write_dataset_config(
    *,
    root: Path,
    profile: FeatureProfile,
    symbols: list[str],
    args: argparse.Namespace,
) -> None:
    universe_path = root / "universe.yaml"
    config_path = root / "dataset_config.yaml"
    with universe_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"candidates": symbols}, f, sort_keys=False)
    config = {
        "version": "m12",
        "description": f"Generated M12 dataset config for {profile.profile_id}",
        "universe": {
            "smoke_symbols": symbols[: min(3, len(symbols))],
            "candidates_file": str(universe_path),
            "full_candidates_file": str(universe_path),
        },
        "paths": {"processed_dir": str(root)},
        "feature_profile": feature_profile_to_dict(profile),
        "features": {
            "dim": profile.feature_dim,
            "cols": list(profile.feature_cols),
        },
        "sample_index": {
            "lookback": args.lookback,
            "stride": args.stride,
            "horizons": csv_list(args.horizons, cast=int),
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_start": args.val_start,
            "val_end": args.val_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
        },
        "model": {
            "type": "alphatrade_v0_2",
            "quantiles": {"num": 5, "levels": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "hidden_dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "dropout": 0.1,
        },
        "training": {
            "max_steps": 1000,
            "batch_size": 256,
            "num_workers": 4,
            "seed": 42,
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.0001,
            "grad_clip": 1.0,
            "lr_schedule": {"type": "cosine", "warmup_steps": 100},
            "val_every": 100,
            "save_every": 500,
            "keep_last_n": 3,
        },
        "loss": {
            "pinball": {
                "enabled": True,
                "horizon_weights": [1.0, 1.0, 1.0, 1.0],
            },
            "crossing_penalty": {"enabled": True, "weight": 0.1},
        },
        "metrics_schema": "m2_train_metrics_v1",
    }
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def write_contract_reports(report: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "m12_chendage_feature_contract.json"
    md_path = reports_dir / "m12_chendage_feature_contract.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# M12 Chendage Processed Feature Contract\n\n")
        f.write("AlphaTrade is development-stage. This contract validates processed numeric ")
        f.write("Chendage market-state features only; it does not validate Chendage rules as tradable signals.\n\n")
        f.write(f"- Generated: {report['generated_at']}\n")
        f.write(f"- Overall status: `{report['overall_status']}`\n")
        f.write(f"- Candidate output: `{report['outputs']['candidate_processed_root']}`\n")
        f.write(f"- Control output: `{report['outputs']['control_processed_root']}`\n")
        f.write(f"- Feature profile: `{report['feature_profile']['profile_id']}` ")
        f.write(f"({report['feature_profile']['feature_dim']} cols)\n")
        f.write(f"- Chendage feature cols: {len(report['feature_profile']['chendage_feature_cols'])}\n\n")
        if report["feature_profile"].get("excluded_feature_groups"):
            f.write(
                "- Excluded feature groups: "
                f"{report['feature_profile']['excluded_feature_groups']}\n\n"
            )
        f.write("## Feature Groups\n\n")
        f.write("| Group | Feature Count |\n")
        f.write("|-------|---------------|\n")
        for group, keys in report["feature_profile"].get("feature_groups", {}).items():
            f.write(f"| {group} | {len(keys)} |\n")
        f.write("\n")
        f.write("## Checks\n\n")
        f.write("| Check | Status | Detail |\n")
        f.write("|-------|--------|--------|\n")
        for check in report["checks"]:
            f.write(f"| {check['name']} | {check['status']} | {check.get('detail') or '-'} |\n")
        f.write("\n## Split Distribution\n\n")
        f.write("| Symbol | Common Rows | Train | Val | Test |\n")
        f.write("|--------|-------------|-------|-----|------|\n")
        for result in report["symbols"]:
            if result["status"] != "SUCCESS":
                continue
            f.write(
                f"| {result['symbol']} | {result['common_rows']} | "
                f"{result['train_samples']} | {result['val_samples']} | {result['test_samples']} |\n"
            )
        f.write("\n## Boundary\n\n")
        f.write("- Accepted API: `chendage_signal.processed.export` / `export_processed_features`\n")
        f.write("- Rejected: legacy mixed CLI, rule ratings, hard blocks, candidate labels, explanations, human annotations, outcome review\n")
        f.write("- M12 candidates are not promoted by this contract.\n")


def main() -> None:
    args = parse_args()
    horizons = csv_list(args.horizons, cast=int)
    symbols = load_symbols(args)
    symbol_map = load_symbol_map(args.symbol_map)
    missing_map = sorted(set(symbols) - set(symbol_map))
    if missing_map:
        raise SystemExit(f"ERROR: symbols missing from --symbol-map: {missing_map}")

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    control_output_dir = Path(args.control_output_dir).expanduser().resolve()
    reports_dir = runtime_paths.reports_dir(args.output_root, args.reports_dir)

    print("\n" + "=" * 60)
    print("M12: Chendage Processed Features Integration")
    print("=" * 60)
    print(f"Symbols: {len(symbols)}")
    print(f"Base dir: {base_dir}")
    print(f"Candidate output: {output_dir}")
    print(f"Control output: {control_output_dir}")
    print(f"Feature profile id: {args.feature_profile_id}")
    print(f"Require causality: {args.require_causality}")
    print("=" * 60 + "\n")

    records = read_processed_export(args.chendage_input)
    rows, processed_meta = flatten_processed_rows(records)
    addl_rows = {}
    for name, path in (
        ("truncated", args.truncated_chendage_input),
        ("mutated", args.mutated_chendage_input),
    ):
        if path:
            addl_records = read_processed_export(path)
            addl_rows[name], addl_rows[f"{name}_meta"] = flatten_processed_rows(addl_records)

    exclude_groups = csv_list(args.exclude_feature_groups)
    valid_groups = {"daily", "h1", "m5", "minute_behavior", "other"}
    invalid_groups = sorted(set(exclude_groups) - valid_groups)
    if invalid_groups:
        raise SystemExit(f"ERROR: invalid --exclude-feature-groups: {invalid_groups}")

    selected_keys = select_chendage_feature_keys(
        rows,
        feature_set=args.feature_set,
        exclude_groups=exclude_groups,
        explicit_keys=csv_list(args.feature_keys) if args.feature_keys else None,
    )
    selected_feature_groups = feature_groups_for_keys(selected_keys)
    key_to_col = feature_key_column_map(selected_keys)
    active_symbol_map = {symbol: symbol_map[symbol] for symbol in symbols}
    scaler = fit_chendage_scaler(
        rows,
        symbol_map=active_symbol_map,
        symbols=symbols,
        feature_keys=selected_keys,
        train_start=args.train_start,
        train_end=args.train_end,
    )
    chg_cols = [key_to_col[key] for key in selected_keys]
    feature_cols = list(BASE_FEATURE_COLS) + chg_cols
    feature_profile = FeatureProfile(
        profile_id=args.feature_profile_id,
        feature_cols=tuple(feature_cols),
        feature_dim=len(feature_cols),
        processed_root=str(output_dir),
        scaler_hash=scaler["scaler_hash"],
        source_schema_versions={
            "alphatrade_feature_profile": "m12_chendage_processed_features_v1",
            **processed_meta.get("source_schema_versions", {}),
        },
        normalization={
            "policy": scaler["policy"],
            "train_only": True,
            "scaler_hash": scaler["scaler_hash"],
        },
    )
    feature_manifest = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "feature_profile": feature_profile_to_dict(feature_profile),
        "base_feature_cols": list(BASE_FEATURE_COLS),
        "chendage_feature_keys": selected_keys,
        "chendage_feature_cols": chg_cols,
        "feature_groups": selected_feature_groups,
        "excluded_feature_groups": exclude_groups,
        "feature_set": args.feature_set,
        "source_schema_versions": processed_meta.get("source_schema_versions", {}),
        "normalization": {
            "policy": scaler["policy"],
            "train_only": True,
            "scaler_hash": scaler["scaler_hash"],
            "train_start": scaler["train_start"],
            "train_end": scaler["train_end"],
            "params": scaler["params"],
        },
    }
    source_manifest = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "chendage_commit": args.chendage_commit,
        "chendage_input": str(Path(args.chendage_input).expanduser().resolve()),
        "truncated_chendage_input": str(Path(args.truncated_chendage_input).expanduser().resolve()) if args.truncated_chendage_input else None,
        "mutated_chendage_input": str(Path(args.mutated_chendage_input).expanduser().resolve()) if args.mutated_chendage_input else None,
        "input_files": {
            "chendage_input": str(Path(args.chendage_input).expanduser().resolve()),
            "truncated_chendage_input": str(Path(args.truncated_chendage_input).expanduser().resolve()) if args.truncated_chendage_input else None,
            "mutated_chendage_input": str(Path(args.mutated_chendage_input).expanduser().resolve()) if args.mutated_chendage_input else None,
            "symbol_map": str(Path(args.symbol_map).expanduser().resolve()),
        },
        "input_hashes": {
            "chendage_input": sha256_file(args.chendage_input),
            "truncated_chendage_input": sha256_file(args.truncated_chendage_input) if args.truncated_chendage_input else None,
            "mutated_chendage_input": sha256_file(args.mutated_chendage_input) if args.mutated_chendage_input else None,
            "symbol_map": sha256_file(args.symbol_map),
        },
        "expected_source_schema_versions": {
            "schema_version": args.expected_schema_version,
            "feature_vector_version": args.expected_feature_vector_version,
        },
        "observed_source_schema_versions": processed_meta.get("source_schema_versions", {}),
        "symbol_map": active_symbol_map,
    }

    checks: list[dict[str, Any]] = []
    observed_schema_versions = processed_meta.get("source_schema_versions", {})
    schema_versions_ok = source_schema_versions_match_expected(
        observed_schema_versions,
        expected_schema_version=args.expected_schema_version,
        expected_feature_vector_version=args.expected_feature_vector_version,
    )
    add_check(
        checks,
        "source_schema_versions_match_expected",
        schema_versions_ok,
        observed={
            "expected": source_manifest["expected_source_schema_versions"],
            "observed": observed_schema_versions,
        },
    )
    add_check(
        checks,
        "rule_only_fields_absent",
        not processed_meta["rule_only_fields_found"],
        observed=processed_meta["rule_only_fields_found"],
    )
    add_check(checks, "selected_feature_keys_non_empty", bool(selected_keys), observed=selected_keys[:10])
    add_check(checks, "symbol_map_explicit", not missing_map, observed={symbol: symbol_map[symbol] for symbol in symbols})
    add_check(
        checks,
        "train_only_scaler_fitted",
        bool(scaler.get("scaler_hash")) and scaler.get("policy") == "train_split_robust_zscore",
        observed={
            "scaler_hash": scaler.get("scaler_hash"),
            "train_start": scaler.get("train_start"),
            "train_end": scaler.get("train_end"),
            "feature_count": scaler.get("feature_count"),
        },
    )

    causality = {"required": bool(args.require_causality), "checks": {}}
    for name in ("truncated", "mutated"):
        if name in addl_rows:
            causality["checks"][name] = compare_processed_exports(
                rows,
                addl_rows[name],
                symbol_map={symbol: symbol_map[symbol] for symbol in symbols},
                feature_keys=selected_keys,
                tolerance=args.causality_tolerance,
            )
        else:
            causality["checks"][name] = {
                "status": "FAIL" if args.require_causality else "SKIPPED",
                "reason": f"--{name}-chendage-input not provided",
            }
    causality_pass = all(c.get("status") == "PASS" for c in causality["checks"].values())
    add_check(
        checks,
        "truncated_and_mutated_causality",
        causality_pass if args.require_causality else True,
        detail="processed feature exports must match when input is truncated at as_of and when post-as_of rows are mutated",
        observed=causality,
    )

    results = []
    for idx, symbol in enumerate(symbols, start=1):
        print(f"[{idx}/{len(symbols)}] {symbol}")
        try:
            result = process_symbol(
                symbol=symbol,
                symbol_spec=symbol_map[symbol],
                base_dir=base_dir,
                output_dir=output_dir,
                control_output_dir=control_output_dir,
                rows=rows,
                feature_keys=selected_keys,
                key_to_col=key_to_col,
                scaler=scaler,
                feature_profile=feature_profile,
                feature_manifest_root=feature_manifest,
                horizons=horizons,
                lookback=args.lookback,
                stride=args.stride,
                args=args,
            )
        except Exception as exc:
            result = {"symbol": symbol, "status": "FAIL", "error": str(exc)}
        results.append(result)
        print(f"  {result['status']}: {result.get('common_rows', result.get('error', ''))}")

    success_results = [r for r in results if r["status"] == "SUCCESS"]
    add_check(checks, "all_symbols_built", len(success_results) == len(symbols), observed=results)
    add_check(
        checks,
        "common_rows_positive",
        all(r.get("common_rows", 0) > 0 for r in success_results) and bool(success_results),
        observed={r["symbol"]: r.get("common_rows", 0) for r in results},
    )
    add_check(
        checks,
        "feature_coverage_within_threshold",
        bool(results)
        and all(r.get("feature_coverage", {}).get("status") == "PASS" for r in results),
        observed={
            r["symbol"]: r.get("feature_coverage")
            for r in results
        },
    )
    add_check(
        checks,
        "common_row_control_rebuilt",
        all((control_output_dir / r["symbol"] / "bars.parquet").exists() for r in success_results),
        observed=str(control_output_dir),
    )
    add_check(
        checks,
        "feature_distribution_by_split_recorded",
        bool(success_results)
        and all(
            all(split in r.get("feature_distribution_by_split", {}) for split in ("train", "val", "test"))
            for r in success_results
        ),
        observed={
            r["symbol"]: list(r.get("feature_distribution_by_split", {}).keys())
            for r in success_results
        },
    )

    source_manifest["symbols"] = results
    feature_manifest["symbols"] = results
    write_root_manifests(
        output_dir=output_dir,
        control_output_dir=control_output_dir,
        feature_manifest=feature_manifest,
        source_manifest=source_manifest,
    )
    control_profile = FeatureProfile(
        profile_id="m12_base8_control_common_rows",
        feature_cols=BASE_FEATURE_COLS,
        feature_dim=len(BASE_FEATURE_COLS),
        processed_root=str(control_output_dir),
        source_schema_versions={"alphatrade_feature_profile": "m1_f8_v1"},
        normalization={"policy": "precomputed_in_bars", "train_only": False},
    )
    write_dataset_config(root=output_dir, profile=feature_profile, symbols=symbols, args=args)
    write_dataset_config(root=control_output_dir, profile=control_profile, symbols=symbols, args=args)

    overall_status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "stage": "development",
        "overall_status": overall_status,
        "inputs": {
            "base_processed_root": str(base_dir),
            "chendage_input": str(Path(args.chendage_input).expanduser().resolve()),
            "symbols": symbols,
            "feature_set": args.feature_set,
            "lookback": args.lookback,
            "stride": args.stride,
            "horizons": horizons,
            "feature_set": args.feature_set,
            "excluded_feature_groups": exclude_groups,
            "max_missing_feature_rate": args.max_missing_feature_rate,
        },
        "outputs": {
            "candidate_processed_root": str(output_dir),
            "control_processed_root": str(control_output_dir),
            "candidate_dataset_config": str(output_dir / "dataset_config.yaml"),
            "control_dataset_config": str(control_output_dir / "dataset_config.yaml"),
            "contract_json": str(reports_dir / "m12_chendage_feature_contract.json"),
            "contract_md": str(reports_dir / "m12_chendage_feature_contract.md"),
        },
        "feature_profile": {
            **feature_profile_to_dict(feature_profile),
            "chendage_feature_keys": selected_keys,
            "chendage_feature_cols": chg_cols,
            "feature_groups": selected_feature_groups,
            "excluded_feature_groups": exclude_groups,
        },
        "normalization": {
            "policy": scaler["policy"],
            "train_only": True,
            "scaler_hash": scaler["scaler_hash"],
            "train_start": scaler["train_start"],
            "train_end": scaler["train_end"],
            "feature_count": scaler["feature_count"],
        },
        "source_boundary": {
            "accepted_api": [
                "python -m chendage_signal.processed.export",
                "export_processed_features",
            ],
            "legacy_cli_rejected": True,
            "rule_only_fields_checked": list(RULE_ONLY_PATTERNS),
            "rule_only_fields_found": processed_meta["rule_only_fields_found"],
            "non_numeric_features_ignored": processed_meta["non_numeric_features_ignored"],
            "expected_source_schema_versions": source_manifest["expected_source_schema_versions"],
            "observed_source_schema_versions": observed_schema_versions,
            "chendage_commit": args.chendage_commit,
            "input_files": source_manifest["input_files"],
            "input_hashes": source_manifest["input_hashes"],
        },
        "symbol_mapping": {symbol: symbol_map[symbol] for symbol in symbols},
        "causality_test": causality,
        "symbols": results,
        "checks": checks,
    }
    write_contract_reports(report, reports_dir)

    print("\n" + "=" * 60)
    print(f"M12 contract: {overall_status}")
    print(f"Report: {reports_dir / 'm12_chendage_feature_contract.json'}")
    print("=" * 60)
    if overall_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
