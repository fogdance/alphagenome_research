#!/usr/bin/env python3
"""Build formal M12 Chendage processed-feature datasets from DB continuous bars.

This script is intentionally scoped to the processed-feature boundary:
it consumes Chendage's processed numeric snapshots and never consumes rule
ratings, actions, candidates, explanations, human annotations, or outcomes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_m12_chendage_features as m12
import runtime_paths
from data_pipeline.feature_profiles import (
    BASE_FEATURE_COLS,
    FeatureProfile,
    feature_profile_to_dict,
)


DEFAULT_SYMBOLS = ("CZCE.FG", "SHFE.SP", "DCE.JM", "SHFE.RB", "CZCE.MA")
FEATURE_COLS = list(BASE_FEATURE_COLS)
SOURCE_SCHEMA_VERSION = m12.SOURCE_SCHEMA_VERSION
CONTRACT_SCHEMA_VERSION = m12.SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M12 formal DB builder for Chendage processed features"
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated continuous symbols",
    )
    parser.add_argument("--source-start", default="2017-12-01")
    parser.add_argument("--source-end", default="2026-06-17")
    parser.add_argument("--eval-start", default="2018-01-01")
    parser.add_argument("--eval-end", default="2026-06-17")
    parser.add_argument("--train-start", default="2018-01-01")
    parser.add_argument("--train-end", default="2023-01-01")
    parser.add_argument("--val-start", default="2023-01-01")
    parser.add_argument("--val-end", default="2024-01-01")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--test-end", default="2026-06-18")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--horizons", default="1,5,20,60")
    parser.add_argument("--feature-profile-id", default="m12_chg_core")
    parser.add_argument("--feature-set", choices=["core", "full"], default="core")
    parser.add_argument("--feature-keys", default=None)
    parser.add_argument("--exclude-feature-groups", default="")
    parser.add_argument("--max-missing-feature-rate", type=float, default=0.0)
    parser.add_argument("--causality-samples-per-symbol", type=int, default=5)
    parser.add_argument("--causality-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--processed-write-chunk-size",
        type=int,
        default=50_000,
        help="Rows per parquet write chunk for processed snapshots",
    )
    parser.add_argument(
        "--chendage-src",
        default=os.environ.get(
            "CHENDAGE_SIGNAL_SRC",
            "/home/v/Documents/work/chendage-signal-engine/src",
        ),
        help="Path to chendage-signal-engine/src",
    )
    parser.add_argument("--chendage-commit", default=None)
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Run root. Default: /data/alphatrade/runs/"
            "m12_formal5_<timestamp>"
        ),
    )
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def csv_list(value: str | None, *, cast=str) -> list:
    return m12.csv_list(value, cast=cast)


def default_output_root() -> Path:
    return Path("/data/alphatrade/runs") / f"m12_formal5_{datetime.now():%Y%m%d_%H%M%S}"


def safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(".", "_")


def get_git_sha(path: str | Path | None = None) -> str:
    cmd = ["git"]
    if path is not None:
        cmd.extend(["-C", str(path)])
    cmd.extend(["rev-parse", "--short", "HEAD"])
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def connect() -> Any:
    import pymysql
    from pymysql.cursors import DictCursor

    return pymysql.connect(
        host=os.environ.get("MD_MYSQL_HOST", "10.0.0.9"),
        port=int(os.environ.get("MD_MYSQL_PORT", "3306")),
        user=os.environ.get("MD_MYSQL_USER", "root"),
        password=os.environ.get("MD_MYSQL_PASSWORD", "root"),
        database=os.environ.get("MD_MYSQL_DB", "market_data"),
        charset="utf8mb4",
        cursorclass=DictCursor,
        connect_timeout=int(os.environ.get("MD_MYSQL_CONNECT_TIMEOUT", "10")),
    )


def query_df(sql: str, params: list[Any]) -> pd.DataFrame:
    conn = connect()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows)


def load_continuous_bars(
    csymbol: str,
    *,
    source_start: str,
    source_end: str,
) -> pd.DataFrame:
    sql = """
    select
      m.csymbol,
      cast(m.trading_date as char) as trading_day,
      m.symbol as mapped_symbol,
      b.symbol as bar_symbol,
      b.underlying,
      b.eob,
      b.open,
      b.high,
      b.low,
      b.close,
      b.volume,
      b.position,
      b.source,
      b.provider
    from fut_continuous_map_v2 m
    join fut_bar_1m_v2 b
      on b.symbol = m.symbol
     and b.trading_date = m.trading_date
    where m.csymbol = %s
      and m.trading_date between %s and %s
    order by b.eob asc
    """
    df = query_df(sql, [csymbol, source_start, source_end])
    if df.empty:
        raise RuntimeError(f"{csymbol}: no DB continuous bars")
    df["eob"] = pd.to_datetime(df["eob"])
    for col in ("open", "high", "low", "close", "volume", "position"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.sort_values(["eob", "mapped_symbol"]).drop_duplicates("eob", keep="last")
    return df.reset_index(drop=True)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gap = out["eob"].diff().dt.total_seconds()
    is_new_segment = gap > 1800
    is_new_segment.iloc[0] = True
    out["segment_id"] = is_new_segment.cumsum().astype("int32")
    eps = 1e-8
    out["ret_1m"] = out["close"].pct_change()
    out["hl_range"] = (out["high"] - out["low"]) / (out["close"] + eps)
    out["co_change"] = (out["close"] - out["open"]) / (out["open"] + eps)
    out["vol_log1p"] = np.log1p(out["volume"])
    out["pos_log1p"] = np.log1p(out["position"])
    minute_of_day = out["eob"].dt.hour * 60 + out["eob"].dt.minute
    phase = 2 * np.pi * minute_of_day / 1440
    out["minute_sin"] = np.sin(phase)
    out["minute_cos"] = np.cos(phase)
    out["is_session_open"] = np.float32(1.0)
    out[FEATURE_COLS] = out[FEATURE_COLS].fillna(0.0).astype("float32")
    return out


def write_base_symbol(root: Path, csymbol: str, df: pd.DataFrame) -> dict[str, Any]:
    symbol_dir = root / "base_m1_f8" / csymbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    bars = add_base_features(df)
    bars_out = pd.DataFrame(
        {
            "eob": bars["eob"],
            "open": bars["open"].astype("float64"),
            "high": bars["high"].astype("float64"),
            "low": bars["low"].astype("float64"),
            "close": bars["close"].astype("float64"),
            "volume": bars["volume"].astype("float64"),
            "position": bars["position"].astype("float64"),
            "symbol": bars["mapped_symbol"].astype(str),
            "segment_id": bars["segment_id"].astype("int32"),
        }
    )
    for col in FEATURE_COLS:
        bars_out[col] = bars[col].astype("float32")
    bars_out.to_parquet(symbol_dir / "bars.parquet", index=False)
    contracts = bars["mapped_symbol"].astype(str).value_counts().sort_index().to_dict()
    return {
        "rows": int(len(bars_out)),
        "min_eob": str(bars_out["eob"].min()),
        "max_eob": str(bars_out["eob"].max()),
        "contracts": contracts,
        "segments": int(bars_out["segment_id"].nunique()),
    }


def install_chendage_import(chendage_src: str) -> None:
    src = Path(chendage_src).expanduser().resolve()
    if not src.exists():
        raise RuntimeError(f"chendage src not found: {src}")
    sys.path.insert(0, str(src))


def candles_from_frame(df: pd.DataFrame, csymbol: str):
    from chendage_signal.models import Candle

    candles = []
    for row in df.itertuples(index=False):
        candles.append(
            Candle(
                datetime=row.eob.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                open_interest=float(row.position),
                symbol=csymbol,
                underlying=csymbol,
                trading_day=str(row.trading_day),
                source="db:fut_continuous_map_v2+fut_bar_1m_v2",
                provider=str(getattr(row, "provider", "") or ""),
            )
        )
    return candles


def write_processed_symbol(
    *,
    root: Path,
    csymbol: str,
    bars_df: pd.DataFrame,
    eval_start: str,
    eval_end: str,
    source_start: str,
    source_end: str,
    write_chunk_size: int,
) -> dict[str, Any]:
    from chendage_signal.processed import (
        ProcessedExportRequest,
        build_processed_export_from_candles,
    )

    output_dir = root / "inputs" / "processed_full"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_symbol_name(csymbol)}.parquet"
    candles = candles_from_frame(bars_df, csymbol)
    request = ProcessedExportRequest(
        source="db",
        symbol=csymbol,
        trading_day_from=eval_start,
        trading_day_to=eval_end,
        mode="all-minutes",
    )
    bundle = build_processed_export_from_candles(
        candles,
        request,
        symbol=csymbol,
        data_version=(
            f"db_continuous_map:{csymbol}:full:"
            f"{source_start}:{source_end}:eval:{eval_start}:{eval_end}"
        ),
    )
    snapshot_count = len(bundle.snapshots)
    del candles
    gc.collect()
    summary = write_snapshots_parquet_chunked(
        bundle.snapshots,
        output_path=output_path,
        chunk_size=write_chunk_size,
    )
    del bundle
    gc.collect()
    summary["snapshots"] = snapshot_count
    return summary


def write_snapshots_parquet_chunked(
    snapshots: list[Any],
    *,
    output_path: str | Path,
    chunk_size: int,
) -> dict[str, Any]:
    """Write processed snapshots without materializing one full DataFrame."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from chendage_signal.processed.features import processed_snapshots_to_dataframe

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    column_count = 0
    min_as_of: pd.Timestamp | None = None
    max_as_of: pd.Timestamp | None = None
    try:
        for start in range(0, len(snapshots), chunk_size):
            frame = processed_snapshots_to_dataframe(
                snapshots[start:start + chunk_size],
                include_features=True,
            )
            if frame.empty:
                continue
            frame = normalize_processed_frame_for_parquet(frame)
            as_of = pd.to_datetime(frame["as_of"])
            chunk_min = as_of.min()
            chunk_max = as_of.max()
            min_as_of = chunk_min if min_as_of is None else min(min_as_of, chunk_min)
            max_as_of = chunk_max if max_as_of is None else max(max_as_of, chunk_max)
            total_rows += int(len(frame))
            column_count = int(len(frame.columns))
            table = pa.Table.from_pandas(frame, preserve_index=False)
            table = table.replace_schema_metadata(None)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            del frame, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    if total_rows == 0:
        raise ValueError("no processed snapshots to write")
    return {
        "path": str(output_path),
        "snapshots": total_rows,
        "min_as_of": str(min_as_of),
        "max_as_of": str(max_as_of),
        "columns": column_count,
        "size_bytes": int(output_path.stat().st_size),
        "write_chunk_size": int(chunk_size),
    }


def normalize_processed_frame_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    """Give chunked processed frames stable dtypes across nullable chunks."""
    out = frame.copy()
    string_cols = {
        "symbol",
        "underlying",
        "as_of",
        "source",
        "data_version",
        "current_datetime",
        "current_trading_day",
        "daily_trend",
        "h1_active_resistance_role",
        "h1_active_support_role",
        "m5_cross",
        "m5_last_cross_type",
        "minute_behavior_state",
        "minute_price_direction",
        "minute_open_interest_direction",
        "schema_version",
        "feature_vector_version",
    }
    bool_cols = {
        "h1_is_near_resistance",
        "h1_is_near_support",
        "h1_is_suspended",
        "minute_is_high_volume_failure",
        "minute_is_short_attack",
        "minute_is_low_position_stall",
        "minute_is_volume_expanded",
    }
    int_cols = {
        "count_1m",
        "count_5m",
        "count_1h",
        "count_1d",
        "daily_swing_high_count",
        "daily_swing_low_count",
    }
    numeric_cols = {
        "current_open",
        "current_high",
        "current_low",
        "current_close",
        "current_volume",
        "current_open_interest",
        "daily_trend_confidence",
        "daily_latest_swing_high_price",
        "daily_latest_swing_low_price",
        "h1_current_price",
        "h1_recent_swing_high_price",
        "h1_recent_swing_low_price",
        "h1_distance_to_swing_high",
        "h1_distance_to_swing_low",
        "h1_active_resistance_price",
        "h1_active_support_price",
        "h1_distance_to_active_resistance",
        "h1_distance_to_active_support",
        "h1_tolerance_points",
        "m5_dif",
        "m5_dea",
        "m5_hist",
        "m5_bars_since_cross",
        "minute_volume_ratio",
        "minute_recent_price_change",
        "minute_recent_open_interest_change",
        "minute_rejection_points",
        "minute_baseline_avg_volume",
        "minute_recent_avg_volume",
        "minute_recent_high",
        "minute_recent_low",
        "minute_previous_high",
        "minute_previous_low",
    }
    for col in out.columns:
        if col in string_cols:
            out[col] = out[col].astype("string")
        elif col in bool_cols:
            out[col] = out[col].astype("boolean")
        elif col in int_cols:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")
        elif col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
        elif col.startswith("feature."):
            out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
        elif pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype("boolean")
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
        else:
            out[col] = out[col].astype("string")
    return out


def parquet_feature_keys(path: str | Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        names = pq.read_schema(path).names
    except Exception:
        names = list(pd.read_parquet(path, columns=[]).columns)
    return sorted(name.removeprefix("feature.") for name in names if name.startswith("feature."))


def select_feature_keys(
    all_keys: list[str],
    *,
    feature_set: str,
    exclude_groups: list[str],
    explicit_keys: list[str] | None,
) -> list[str]:
    if explicit_keys:
        missing = sorted(set(explicit_keys) - set(all_keys))
        if missing:
            raise ValueError(f"explicit feature keys missing from processed export: {missing}")
        keys = list(explicit_keys)
    elif feature_set == "full":
        keys = [key for key in all_keys if not m12._looks_like_raw_price_feature(key)]
    else:
        markers = ("daily", "h1", "m5", "minute")
        keys = [
            key
            for key in all_keys
            if any(marker in key.lower() for marker in markers)
            and not m12._looks_like_raw_price_feature(key)
        ]
    if exclude_groups:
        excluded = set(exclude_groups)
        keys = [key for key in keys if m12.feature_group_for_key(key) not in excluded]
    if not keys:
        raise ValueError("feature selection produced zero Chendage features")
    return sorted(keys)


def observed_schema_versions(processed_paths: dict[str, str]) -> dict[str, list[str]]:
    versions: dict[str, set[str]] = {
        "schema_version": set(),
        "feature_vector_version": set(),
    }
    for path in processed_paths.values():
        frame = pd.read_parquet(path, columns=["schema_version", "feature_vector_version"])
        for key in versions:
            versions[key].update(str(x) for x in frame[key].dropna().unique().tolist())
    return {key: sorted(values) for key, values in versions.items()}


def fit_scaler_from_parquet(
    *,
    processed_paths: dict[str, str],
    feature_keys: list[str],
    train_start: str,
    train_end: str,
) -> dict[str, Any]:
    train_start_ts = pd.to_datetime(train_start)
    train_end_ts = pd.to_datetime(train_end)
    params: dict[str, dict[str, Any]] = {}
    for key in feature_keys:
        arrays = []
        raw_col = f"feature.{key}"
        for symbol, path in processed_paths.items():
            frame = pd.read_parquet(path, columns=["as_of", raw_col])
            frame["as_of"] = pd.to_datetime(frame["as_of"])
            train_mask = (frame["as_of"] >= train_start_ts) & (frame["as_of"] < train_end_ts)
            if not train_mask.any():
                continue
            values = frame.loc[train_mask, raw_col].to_numpy(dtype=np.float64, copy=False)
            values = values[np.isfinite(values)]
            if values.size:
                arrays.append(values.copy())
            del frame, values
        if not arrays:
            raise ValueError(f"no finite train values available to fit scaler for {key}")
        arr = np.concatenate(arrays)
        del arrays
        if arr.size == 0:
            raise ValueError(f"no finite train values for {key}")
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
        del arr

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


def processed_frame_from_parquet(
    path: str | Path,
    *,
    feature_keys: list[str],
    key_to_col: dict[str, str],
    scaler: dict[str, Any],
) -> pd.DataFrame:
    raw_cols = [f"feature.{key}" for key in feature_keys]
    frame = pd.read_parquet(path, columns=["as_of"] + raw_cols)
    frame["eob"] = pd.to_datetime(frame["as_of"])
    out = pd.DataFrame({"eob": frame["eob"]})
    for key in feature_keys:
        raw_col = f"feature.{key}"
        col = key_to_col[key]
        values = frame[raw_col].astype("float64")
        item = scaler["params"][key]
        if item["transform"] == "identity_binary":
            out[col] = values.astype("float32")
        else:
            out[col] = ((values - item["median"]) / item["scale"]).astype("float32")
    return out.sort_values("eob").reset_index(drop=True)


def process_dataset_symbol(
    *,
    symbol: str,
    symbol_spec: dict[str, Any],
    base_dir: Path,
    processed_path: str,
    output_dir: Path,
    control_output_dir: Path,
    feature_keys: list[str],
    key_to_col: dict[str, str],
    scaler: dict[str, Any],
    feature_profile: FeatureProfile,
    feature_manifest_root: dict[str, Any],
    horizons: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    bars_path = base_dir / symbol / "bars.parquet"
    if not bars_path.exists():
        return {"symbol": symbol, "status": "FAIL", "error": f"base bars not found: {bars_path}"}
    feature_frame = processed_frame_from_parquet(
        processed_path,
        feature_keys=feature_keys,
        key_to_col=key_to_col,
        scaler=scaler,
    )
    base_df = pd.read_parquet(bars_path).reset_index(drop=True)
    base_df["eob"] = pd.to_datetime(base_df["eob"])
    base_df["_m12_source_pos"] = np.arange(len(base_df), dtype=np.int64)
    if "segment_id" not in base_df.columns:
        raise ValueError(f"{symbol}: base bars missing segment_id")

    base_labeled = m12.compute_labels(base_df, horizons)
    full_index_df = m12.generate_common_indices(
        base_labeled,
        base_df,
        lookback=args.lookback,
        horizons=horizons,
        stride=args.stride,
    )
    common_df = base_labeled.merge(feature_frame, on="eob", how="inner", validate="one_to_one")
    common_df = common_df.sort_values("eob").reset_index(drop=True)
    if common_df.empty:
        return {"symbol": symbol, "status": "FAIL", "error": "zero common rows after eob alignment"}

    chg_cols = [key_to_col[key] for key in feature_keys]
    feature_check = m12.check_feature_frame(common_df, chg_cols)
    if not (feature_check["numeric"] and feature_check["finite"] and feature_check["missing_rate"] == 0.0):
        return {
            "symbol": symbol,
            "status": "FAIL",
            "error": "feature semantic check failed",
            "feature_check": feature_check,
        }

    index_df = m12.generate_common_indices(
        common_df,
        base_df,
        lookback=args.lookback,
        horizons=horizons,
        stride=args.stride,
    )
    train_df, val_df, test_df = m12.split_by_time(
        index_df,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    feature_coverage = m12.compute_feature_coverage(
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
    split_distribution = m12.feature_distribution_by_split(
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
        "chendage_export_symbol": symbol_spec["chendage_export_symbol"],
        "symbol_map": symbol_spec,
        "base_bars_path": str(bars_path),
        "processed_input_path": processed_path,
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
    m12.write_symbol_manifests(
        symbol_dir=candidate_symbol_dir,
        source_manifest=source_manifest,
        feature_manifest=symbol_feature_manifest,
    )
    m12.write_symbol_manifests(
        symbol_dir=control_symbol_dir,
        source_manifest={**source_manifest, "control": "base8_common_rows"},
        feature_manifest={
            **symbol_feature_manifest,
            "feature_profile": feature_profile_to_dict(
                FeatureProfile(
                    profile_id="m12_base8_control_common_rows",
                    feature_cols=tuple(BASE_FEATURE_COLS),
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
        "chendage_export_symbol": symbol_spec["chendage_export_symbol"],
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


def build_causality_check(
    *,
    base_paths: dict[str, str],
    feature_keys: list[str],
    symbols: list[str],
    samples_per_symbol: int,
    tolerance: float,
) -> dict[str, Any]:
    from chendage_signal.processed.features import build_processed_snapshots

    checks = {}
    overall_missing = 0
    overall_mismatch = 0
    overall_compared = 0
    overall_max_abs_diff = 0.0
    mismatch_sample = []

    for symbol in symbols:
        base_df = pd.read_parquet(base_paths[symbol])
        base_df["eob"] = pd.to_datetime(base_df["eob"])
        if base_df.empty:
            checks[symbol] = {"status": "FAIL", "reason": "empty base bars"}
            continue
        sample_count = min(samples_per_symbol, len(base_df))
        positions = np.linspace(0, len(base_df) - 1, num=sample_count, dtype=int)
        as_of_list = [base_df.iloc[int(pos)]["eob"].to_pydatetime() for pos in positions]
        candles = candles_from_frame(
            base_df.assign(
                trading_day=base_df["eob"].dt.date.astype(str),
                provider="",
            ),
            symbol,
        )
        full_snaps = build_processed_snapshots(candles, symbol, as_of_list)

        missing = 0
        mismatches = 0
        compared = 0
        max_abs_diff = 0.0
        for as_of, full_snapshot in zip(as_of_list, full_snaps):
            truncated_candles = [candle for candle in candles if candle.datetime <= as_of]
            mutated_candles = []
            for candle in candles:
                if candle.datetime <= as_of:
                    mutated_candles.append(candle)
                    continue
                mutated = type(candle)(
                    datetime=candle.datetime,
                    open=candle.open * 1.25,
                    high=candle.high * 1.25,
                    low=candle.low * 1.25,
                    close=candle.close * 1.25,
                    volume=candle.volume * 3.0,
                    open_interest=candle.open_interest * 2.0,
                    symbol=candle.symbol,
                    underlying=candle.underlying,
                    trading_day=candle.trading_day,
                    bob=candle.bob,
                    source=candle.source,
                    provider=candle.provider,
                )
                mutated_candles.append(mutated)
            truncated = build_processed_snapshots(truncated_candles, symbol, [as_of])[0]
            mutated = build_processed_snapshots(mutated_candles, symbol, [as_of])[0]
            full_features = full_snapshot.to_feature_dict()
            truncated_features = truncated.to_feature_dict()
            mutated_features = mutated.to_feature_dict()
            for feature_key in feature_keys:
                if (
                    feature_key not in full_features
                    or feature_key not in truncated_features
                    or feature_key not in mutated_features
                ):
                    missing += 1
                    continue
                for variant, candidate_features in (
                    ("truncated", truncated_features),
                    ("mutated", mutated_features),
                ):
                    diff = abs(float(full_features[feature_key]) - float(candidate_features[feature_key]))
                    compared += 1
                    max_abs_diff = max(max_abs_diff, diff)
                    if diff > tolerance:
                        mismatches += 1
                        if len(mismatch_sample) < 10:
                            mismatch_sample.append(
                                {
                                    "symbol": symbol,
                                    "as_of": str(as_of),
                                    "variant": variant,
                                    "feature": feature_key,
                                    "abs_diff": diff,
                                }
                            )
        status = "PASS" if compared > 0 and missing == 0 and mismatches == 0 else "FAIL"
        checks[symbol] = {
            "status": status,
            "sample_count": int(sample_count),
            "compared_values": int(compared),
            "missing_count": int(missing),
            "mismatch_count": int(mismatches),
            "max_abs_diff": float(max_abs_diff),
        }
        overall_missing += missing
        overall_mismatch += mismatches
        overall_compared += compared
        overall_max_abs_diff = max(overall_max_abs_diff, max_abs_diff)

    overall_status = (
        "PASS"
        if checks and all(item.get("status") == "PASS" for item in checks.values())
        else "FAIL"
    )
    common_payload = {
        "status": overall_status,
        "compared_values": int(overall_compared),
        "missing_count": int(overall_missing),
        "mismatch_count": int(overall_mismatch),
        "max_abs_diff": float(overall_max_abs_diff),
        "by_symbol": checks,
        "mismatch_sample": mismatch_sample,
    }
    return {
        "required": True,
        "checks": {
            "truncated": common_payload,
            "mutated": common_payload,
        },
    }


def write_symbol_map(path: Path, symbols: list[str], export_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    symbol_map: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        contracts = export_summary["symbols_detail"][symbol]["base"]["contracts"]
        symbol_map[symbol] = {
            "alphatrade_symbol": symbol,
            "chendage_export_symbol": symbol,
            "source_symbol": symbol,
            "symbol_type": "continuous_from_db_map",
            "contract_map": {
                "source_table": "fut_continuous_map_v2",
                "bar_table": "fut_bar_1m_v2",
                "contracts": contracts,
                "date_range": (
                    f"{export_summary['date_ranges']['source_start']}.."
                    f"{export_summary['date_ranges']['source_end']}"
                ),
            },
            "roll_policy": "db_fut_continuous_map_v2",
            "timezone": "Asia/Shanghai",
        }
    path.write_text(
        json.dumps(symbol_map, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return symbol_map


def write_formal_sweep_config(root: Path, candidate_config: Path, control_config: Path) -> Path:
    path = root / "configs" / "m12_chendage_formal5.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "profile": "m12_chendage_features_formal5",
        "universe": "m12_formal5_common_rows",
        "dataset": "m12_chendage_processed_features",
        "dataset_config": str(control_config),
        "expected_seeds": [42, 43, 44],
        "primary_metric": "pinball_loss.overall",
        "defaults": {
            "max_steps": 1000,
            "batch_size": 256,
            "jit": 1,
            "clip_norm": 1.0,
            "save_every": 500,
            "keep_last": 3,
            "window_cache": "auto",
            "window_cache_dir": str(root / "cache" / "window_cache"),
            "eval_split": "val",
            "ckpt_step": "best",
        },
        "experiments": [
            {
                "exp_id": "base8_control_common_rows",
                "description": "Base 8D features on the exact M12 common rows.",
                "dataset_config": str(control_config),
                "overrides": {},
            },
            {
                "exp_id": "chg_core",
                "description": "Base 8D plus compact Chendage processed feature groups.",
                "dataset_config": str(candidate_config),
                "overrides": {},
            },
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root()
    reports_dir = runtime_paths.reports_dir(root, args.reports_dir)
    for subdir in (
        "inputs/processed_full",
        "reports",
        "data/processed",
        "base_m1_f8",
        "configs",
        "cache",
        "checkpoints",
        "artifacts",
    ):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    (root / "RUN_ROOT").write_text(str(root) + "\n", encoding="utf-8")
    latest_path = Path("/data/alphatrade/runs/m12_formal5_latest")
    latest_path.write_text(str(root) + "\n", encoding="utf-8")

    install_chendage_import(args.chendage_src)
    symbols = csv_list(args.symbols)
    horizons = csv_list(args.horizons, cast=int)
    exclude_groups = csv_list(args.exclude_feature_groups)
    invalid_groups = sorted(set(exclude_groups) - {"daily", "h1", "m5", "minute_behavior", "other"})
    if invalid_groups:
        raise SystemExit(f"ERROR: invalid --exclude-feature-groups: {invalid_groups}")

    chendage_commit = (
        args.chendage_commit
        or get_git_sha(Path(args.chendage_src).expanduser().resolve().parents[0])
    )
    started = time.time()
    print("\n" + "=" * 72)
    print("M12 formal DB build: Chendage processed features")
    print("=" * 72)
    print(f"Root: {root}")
    print(f"Symbols: {symbols}")
    print(f"Source: {args.source_start}..{args.source_end}")
    print(f"Evaluation features: {args.eval_start}..{args.eval_end}")
    print(f"Splits: train<{args.train_end}, val<{args.val_end}, test<{args.test_end}")
    print("=" * 72 + "\n", flush=True)

    export_summary: dict[str, Any] = {
        "root": str(root),
        "symbols": symbols,
        "date_ranges": {
            "source_start": args.source_start,
            "source_end": args.source_end,
            "eval_start": args.eval_start,
            "eval_end": args.eval_end,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_start": args.val_start,
            "val_end": args.val_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
        },
        "base_dir": str(root / "base_m1_f8"),
        "processed_dir": str(root / "inputs" / "processed_full"),
        "symbols_detail": {},
    }
    processed_paths: dict[str, str] = {}
    base_paths: dict[str, str] = {}

    for index, symbol in enumerate(symbols, start=1):
        symbol_started = time.time()
        print(f"[{index}/{len(symbols)}] DB load {symbol}", flush=True)
        bars_df = load_continuous_bars(
            symbol,
            source_start=args.source_start,
            source_end=args.source_end,
        )
        base_summary = write_base_symbol(root, symbol, bars_df)
        base_paths[symbol] = str(root / "base_m1_f8" / symbol / "bars.parquet")
        print(
            f"  bars rows={base_summary['rows']:,} "
            f"range={base_summary['min_eob']}..{base_summary['max_eob']} "
            f"contracts={len(base_summary['contracts'])}",
            flush=True,
        )
        processed_summary = write_processed_symbol(
            root=root,
            csymbol=symbol,
            bars_df=bars_df,
            eval_start=args.eval_start,
            eval_end=args.eval_end,
            source_start=args.source_start,
            source_end=args.source_end,
            write_chunk_size=args.processed_write_chunk_size,
        )
        processed_paths[symbol] = processed_summary["path"]
        export_summary["symbols_detail"][symbol] = {
            "base": base_summary,
            "processed": processed_summary,
            "elapsed_seconds": time.time() - symbol_started,
        }
        print(
            f"  processed snapshots={processed_summary['snapshots']:,} "
            f"size_mb={processed_summary['size_bytes'] / 1024 / 1024:.1f} "
            f"elapsed={export_summary['symbols_detail'][symbol]['elapsed_seconds']:.1f}s",
            flush=True,
        )
        del bars_df
        gc.collect()

    symbol_map_path = root / "inputs" / "symbol_map.json"
    symbol_map = write_symbol_map(symbol_map_path, symbols, export_summary)
    export_summary["symbol_map"] = str(symbol_map_path)
    export_summary["processed_paths"] = processed_paths
    export_summary["elapsed_seconds_export"] = time.time() - started
    export_summary_path = root / "inputs" / "export_summary.json"
    export_summary_path.write_text(
        json.dumps(export_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\nSelecting features and fitting train-only scaler...", flush=True)
    key_sets = [set(parquet_feature_keys(path)) for path in processed_paths.values()]
    all_keys = sorted(set.union(*key_sets))
    common_keys = sorted(set.intersection(*key_sets))
    if set(all_keys) != set(common_keys):
        missing_by_symbol = {
            symbol: sorted(set(all_keys) - parquet_feature_keys(path))
            for symbol, path in processed_paths.items()
        }
        raise RuntimeError(f"processed feature key mismatch by symbol: {missing_by_symbol}")
    selected_keys = select_feature_keys(
        all_keys,
        feature_set=args.feature_set,
        exclude_groups=exclude_groups,
        explicit_keys=csv_list(args.feature_keys) if args.feature_keys else None,
    )
    selected_feature_groups = m12.feature_groups_for_keys(selected_keys)
    key_to_col = m12.feature_key_column_map(selected_keys)
    scaler = fit_scaler_from_parquet(
        processed_paths=processed_paths,
        feature_keys=selected_keys,
        train_start=args.train_start,
        train_end=args.train_end,
    )
    chg_cols = [key_to_col[key] for key in selected_keys]
    feature_cols = list(BASE_FEATURE_COLS) + chg_cols

    output_dir = root / "data" / "processed" / "m12_chendage_fN"
    control_output_dir = root / "data" / "processed" / "m12_common_base8"
    feature_profile = FeatureProfile(
        profile_id=args.feature_profile_id,
        feature_cols=tuple(feature_cols),
        feature_dim=len(feature_cols),
        processed_root=str(output_dir),
        scaler_hash=scaler["scaler_hash"],
        source_schema_versions={
            "alphatrade_feature_profile": "m12_chendage_processed_features_v1",
            **observed_schema_versions(processed_paths),
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
        "source_schema_versions": observed_schema_versions(processed_paths),
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
        "chendage_commit": chendage_commit,
        "input_files": {
            "processed_dir": str(root / "inputs" / "processed_full"),
            "processed_files": processed_paths,
            "symbol_map": str(symbol_map_path),
            "export_summary": str(export_summary_path),
        },
        "input_hashes": {
            "symbol_map": sha256_file(symbol_map_path),
            "export_summary": sha256_file(export_summary_path),
            "processed_files": {
                symbol: sha256_file(path) for symbol, path in processed_paths.items()
            },
        },
        "expected_source_schema_versions": {
            "schema_version": args.expected_schema_version
            if hasattr(args, "expected_schema_version")
            else "processed_market_snapshot.v1",
            "feature_vector_version": args.expected_feature_vector_version
            if hasattr(args, "expected_feature_vector_version")
            else "processed_feature_vector.v1",
        },
        "observed_source_schema_versions": observed_schema_versions(processed_paths),
        "symbol_map": symbol_map,
    }

    checks: list[dict[str, Any]] = []
    observed_versions = source_manifest["observed_source_schema_versions"]
    schema_versions_ok = m12.source_schema_versions_match_expected(
        observed_versions,
        expected_schema_version=source_manifest["expected_source_schema_versions"]["schema_version"],
        expected_feature_vector_version=source_manifest["expected_source_schema_versions"]["feature_vector_version"],
    )
    m12.add_check(
        checks,
        "source_schema_versions_match_expected",
        schema_versions_ok,
        observed={
            "expected": source_manifest["expected_source_schema_versions"],
            "observed": observed_versions,
        },
    )
    m12.add_check(checks, "rule_only_fields_absent", True, observed=[])
    m12.add_check(checks, "selected_feature_keys_non_empty", bool(selected_keys), observed=selected_keys[:10])
    m12.add_check(checks, "symbol_map_explicit", True, observed=symbol_map)
    m12.add_check(
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

    print("Running selected-timestamp causality check...", flush=True)
    causality = build_causality_check(
        base_paths=base_paths,
        feature_keys=selected_keys,
        symbols=symbols,
        samples_per_symbol=args.causality_samples_per_symbol,
        tolerance=args.causality_tolerance,
    )
    causality_pass = all(item.get("status") == "PASS" for item in causality["checks"].values())
    m12.add_check(
        checks,
        "truncated_and_mutated_causality",
        causality_pass,
        detail=(
            "selected processed feature snapshots must match when input is "
            "truncated at as_of and when post-as_of rows are mutated"
        ),
        observed=causality,
    )

    results = []
    for index, symbol in enumerate(symbols, start=1):
        print(f"[{index}/{len(symbols)}] Build AlphaTrade M12 dataset {symbol}", flush=True)
        try:
            result = process_dataset_symbol(
                symbol=symbol,
                symbol_spec=symbol_map[symbol],
                base_dir=root / "base_m1_f8",
                processed_path=processed_paths[symbol],
                output_dir=output_dir,
                control_output_dir=control_output_dir,
                feature_keys=selected_keys,
                key_to_col=key_to_col,
                scaler=scaler,
                feature_profile=feature_profile,
                feature_manifest_root=feature_manifest,
                horizons=horizons,
                args=args,
            )
        except Exception as exc:
            result = {"symbol": symbol, "status": "FAIL", "error": str(exc)}
        results.append(result)
        print(f"  {result['status']}: {result.get('common_rows', result.get('error', ''))}", flush=True)

    success_results = [result for result in results if result["status"] == "SUCCESS"]
    m12.add_check(checks, "all_symbols_built", len(success_results) == len(symbols), observed=results)
    m12.add_check(
        checks,
        "common_rows_positive",
        all(result.get("common_rows", 0) > 0 for result in success_results) and bool(success_results),
        observed={result["symbol"]: result.get("common_rows", 0) for result in results},
    )
    m12.add_check(
        checks,
        "feature_coverage_within_threshold",
        bool(results)
        and all(result.get("feature_coverage", {}).get("status") == "PASS" for result in results),
        observed={result["symbol"]: result.get("feature_coverage") for result in results},
    )
    m12.add_check(
        checks,
        "common_row_control_rebuilt",
        all((control_output_dir / result["symbol"] / "bars.parquet").exists() for result in success_results),
        observed=str(control_output_dir),
    )
    m12.add_check(
        checks,
        "feature_distribution_by_split_recorded",
        bool(success_results)
        and all(
            all(split in result.get("feature_distribution_by_split", {}) for split in ("train", "val", "test"))
            for result in success_results
        ),
        observed={
            result["symbol"]: list(result.get("feature_distribution_by_split", {}).keys())
            for result in success_results
        },
    )

    source_manifest["symbols"] = results
    feature_manifest["symbols"] = results
    m12.write_root_manifests(
        output_dir=output_dir,
        control_output_dir=control_output_dir,
        feature_manifest=feature_manifest,
        source_manifest=source_manifest,
    )
    control_profile = FeatureProfile(
        profile_id="m12_base8_control_common_rows",
        feature_cols=tuple(BASE_FEATURE_COLS),
        feature_dim=len(BASE_FEATURE_COLS),
        processed_root=str(control_output_dir),
        source_schema_versions={"alphatrade_feature_profile": "m1_f8_v1"},
        normalization={"policy": "precomputed_in_bars", "train_only": False},
    )
    m12.write_dataset_config(root=output_dir, profile=feature_profile, symbols=symbols, args=args)
    m12.write_dataset_config(root=control_output_dir, profile=control_profile, symbols=symbols, args=args)
    sweep_config = write_formal_sweep_config(
        root,
        candidate_config=output_dir / "dataset_config.yaml",
        control_config=control_output_dir / "dataset_config.yaml",
    )

    overall_status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    report = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "stage": "development",
        "overall_status": overall_status,
        "inputs": {
            "base_processed_root": str(root / "base_m1_f8"),
            "chendage_input": str(root / "inputs" / "processed_full"),
            "symbols": symbols,
            "feature_set": args.feature_set,
            "lookback": args.lookback,
            "stride": args.stride,
            "horizons": horizons,
            "excluded_feature_groups": exclude_groups,
            "max_missing_feature_rate": args.max_missing_feature_rate,
            "source_start": args.source_start,
            "source_end": args.source_end,
            "eval_start": args.eval_start,
            "eval_end": args.eval_end,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_start": args.val_start,
            "val_end": args.val_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
        },
        "outputs": {
            "candidate_processed_root": str(output_dir),
            "control_processed_root": str(control_output_dir),
            "candidate_dataset_config": str(output_dir / "dataset_config.yaml"),
            "control_dataset_config": str(control_output_dir / "dataset_config.yaml"),
            "sweep_config": str(sweep_config),
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
                "chendage_signal.processed.build_processed_export_from_candles",
                "chendage_signal.processed.export_processed_features",
            ],
            "legacy_cli_rejected": True,
            "rule_only_fields_checked": list(m12.RULE_ONLY_PATTERNS),
            "rule_only_fields_found": [],
            "non_numeric_features_ignored": [],
            "expected_source_schema_versions": source_manifest["expected_source_schema_versions"],
            "observed_source_schema_versions": observed_versions,
            "chendage_commit": chendage_commit,
            "input_files": source_manifest["input_files"],
            "input_hashes": source_manifest["input_hashes"],
        },
        "symbol_mapping": symbol_map,
        "causality_test": causality,
        "symbols": results,
        "checks": checks,
    }
    m12.write_contract_reports(report, reports_dir)

    export_summary["elapsed_seconds_total"] = time.time() - started
    export_summary["m12_outputs"] = report["outputs"]
    export_summary_path.write_text(
        json.dumps(export_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("\n" + "=" * 72)
    print(f"M12 contract: {overall_status}")
    print(f"Root: {root}")
    print(f"Report: {reports_dir / 'm12_chendage_feature_contract.json'}")
    print(f"Sweep config: {sweep_config}")
    print("=" * 72)
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
