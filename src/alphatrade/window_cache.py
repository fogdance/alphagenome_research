"""Window materialization cache for AlphaTrade datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


HORIZONS = (1, 5, 20, 60)
CACHE_SCHEMA_VERSION = 1


@dataclass
class WindowDataset:
    x: np.ndarray
    y: np.ndarray
    symbols: np.ndarray
    metadata: dict
    cache_hit: bool

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def memory_mb(self) -> float:
        return float((self.x.nbytes + self.y.nbytes + self.symbols.nbytes) / (1024 * 1024))


def fingerprint(
    *,
    symbols: Sequence[str],
    processed_root: str | os.PathLike[str],
    split: str,
    features: Sequence[str],
) -> tuple[str, dict]:
    """Build a content-ish fingerprint from inputs and source file metadata."""
    root = Path(processed_root).expanduser().resolve()
    files = []
    missing = []

    for symbol in symbols:
        for name in ("bars.parquet", f"index_{split}.parquet"):
            path = root / symbol / name
            if not path.exists():
                missing.append(str(path))
                files.append({"path": str(path), "exists": False})
                continue
            st = path.stat()
            files.append(
                {
                    "path": str(path),
                    "exists": True,
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                }
            )

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "processed_root": str(root),
        "split": split,
        "symbols": list(symbols),
        "features": list(features),
        "horizons": list(HORIZONS),
        "files": files,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    payload["fingerprint"] = digest
    payload["missing_files"] = missing
    return digest, payload


def cache_path(cache_dir: str | os.PathLike[str], fp: str) -> Path:
    return Path(cache_dir).expanduser().resolve() / f"windows_{fp}"


def _load_cache(path: Path, *, mmap: bool) -> WindowDataset:
    mmap_mode = "r" if mmap else None
    with open(path / "metadata.json") as f:
        metadata = json.load(f)
    return WindowDataset(
        x=np.load(path / "x.npy", mmap_mode=mmap_mode),
        y=np.load(path / "y.npy", mmap_mode=mmap_mode),
        symbols=np.load(path / "symbols.npy", mmap_mode=mmap_mode),
        metadata=metadata,
        cache_hit=True,
    )


def _atomic_write_cache(path: Path, dataset: WindowDataset) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{int(time.time() * 1000)}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / "x.npy", dataset.x)
    np.save(tmp / "y.npy", dataset.y)
    np.save(tmp / "symbols.npy", dataset.symbols)
    with open(tmp / "metadata.json", "w") as f:
        json.dump(dataset.metadata, f, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    tmp.replace(path)


def materialize_windows(
    *,
    symbols: Sequence[str],
    processed_root: str | os.PathLike[str],
    split: str,
    features: Sequence[str],
    fingerprint_metadata: dict | None = None,
) -> WindowDataset:
    """Materialize indexed bars into contiguous window and target arrays."""
    root = Path(processed_root).expanduser().resolve()
    bars_dict: dict[str, np.ndarray] = {}
    raw_indices = []
    loaded_symbols = []
    skipped_symbols = []

    print(f"\nLoading {split} data...")
    for symbol in symbols:
        symbol_dir = root / symbol
        bars_path = symbol_dir / "bars.parquet"
        index_path = symbol_dir / f"index_{split}.parquet"

        if not bars_path.exists():
            print(f"  WARNING {symbol}: bars.parquet not found, skipping")
            skipped_symbols.append(symbol)
            continue
        if not index_path.exists():
            print(f"  WARNING {symbol}: index_{split}.parquet not found, skipping")
            skipped_symbols.append(symbol)
            continue

        bars_df = pd.read_parquet(bars_path)
        feature_data = bars_df[list(features)].values.astype(np.float32)
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        bars_dict[symbol] = feature_data

        index_df = pd.read_parquet(index_path)
        n_before = len(raw_indices)
        for row in index_df.itertuples(index=False):
            x_start = int(getattr(row, "x_start"))
            x_end = int(getattr(row, "x_end"))
            raw_indices.append(
                (
                    symbol,
                    x_start,
                    x_end,
                    float(getattr(row, "y_h1")),
                    float(getattr(row, "y_h5")),
                    float(getattr(row, "y_h20")),
                    float(getattr(row, "y_h60")),
                )
            )

        loaded_symbols.append(symbol)
        print(f"  {symbol}: {len(bars_df):,} bars, {len(raw_indices) - n_before:,} samples")

    n = len(raw_indices)
    print(f"Total {split} samples: {n:,}")

    if n > 0:
        lookback = raw_indices[0][2] - raw_indices[0][1] + 1
        n_features = len(features)
        print(f"Pre-materializing {n:,} windows ({lookback}x{n_features})...", end=" ", flush=True)
        x = np.empty((n, lookback, n_features), dtype=np.float32)
        y = np.empty((n, len(HORIZONS)), dtype=np.float32)
        sample_symbols = np.empty((n,), dtype="U32")

        for i, (symbol, x_start, x_end, yh1, yh5, yh20, yh60) in enumerate(raw_indices):
            x[i] = bars_dict[symbol][x_start : x_end + 1]
            y[i, 0] = yh1
            y[i, 1] = yh5
            y[i, 2] = yh20
            y[i, 3] = yh60
            sample_symbols[i] = symbol
        print(f"done ({(x.nbytes + y.nbytes + sample_symbols.nbytes) / (1024 * 1024):.1f} MB)")
    else:
        x = np.empty((0, 60, len(features)), dtype=np.float32)
        y = np.empty((0, len(HORIZONS)), dtype=np.float32)
        sample_symbols = np.empty((0,), dtype="U32")

    metadata = {
        **(fingerprint_metadata or {}),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "samples": int(n),
        "loaded_symbols": loaded_symbols,
        "skipped_symbols": skipped_symbols,
        "memory_mb": float((x.nbytes + y.nbytes + sample_symbols.nbytes) / (1024 * 1024)),
    }
    return WindowDataset(x=x, y=y, symbols=sample_symbols, metadata=metadata, cache_hit=False)


def load_or_build(
    *,
    symbols: Sequence[str],
    processed_root: str | os.PathLike[str],
    split: str,
    features: Sequence[str],
    cache_dir: str | os.PathLike[str] | None,
    mode: str = "auto",
    mmap: bool = False,
) -> WindowDataset:
    """Load a cached window dataset or materialize and cache it.

    mode:
      - "auto": use existing cache, otherwise build and save
      - "refresh": rebuild and replace cache
      - "off": build in memory without reading or writing cache
    """
    if mode not in {"auto", "refresh", "off"}:
        raise ValueError(f"unknown window cache mode: {mode}")

    fp, fp_meta = fingerprint(
        symbols=symbols,
        processed_root=processed_root,
        split=split,
        features=features,
    )

    path = cache_path(cache_dir, fp) if cache_dir else None
    if mode == "auto" and path and path.exists():
        ds = _load_cache(path, mmap=mmap)
        ds.metadata["cache_path"] = str(path)
        print(f"Window cache hit: {path} ({len(ds):,} samples)")
        return ds

    ds = materialize_windows(
        symbols=symbols,
        processed_root=processed_root,
        split=split,
        features=features,
        fingerprint_metadata=fp_meta,
    )
    if mode != "off" and path:
        ds.metadata["cache_path"] = str(path)
        _atomic_write_cache(path, ds)
        print(f"Window cache saved: {path}")
        if mmap:
            return _load_cache(path, mmap=True)
    return ds


def iter_batches(dataset: WindowDataset, batch_size: int, indices: Iterable[int] | np.ndarray):
    """Yield numpy mini-batches from a materialized window dataset."""
    idx = np.asarray(list(indices), dtype=np.int64) if not isinstance(indices, np.ndarray) else indices
    for i in range(0, len(idx), batch_size):
        batch_idx = idx[i : i + batch_size]
        yield dataset.x[batch_idx], dataset.y[batch_idx]
