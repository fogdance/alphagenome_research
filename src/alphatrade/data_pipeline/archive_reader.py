"""
Archive Reader for Juejin Futures Data

Reads parquet files from archive based on manifest, concatenates them,
and performs basic validation.
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import numpy as np


class ArchiveReader:
    """Read and validate futures data from archive."""

    def __init__(self, archive_dir: str, manifest_path: str):
        """
        Initialize archive reader.

        Args:
            archive_dir: Root directory of the archive
            manifest_path: Path to manifest JSONL file
        """
        self.archive_dir = archive_dir
        self.manifest_path = manifest_path
        self._manifest_cache = None

    def _load_manifest(self) -> Dict[str, List[dict]]:
        """Load and cache manifest, grouped by symbol."""
        if self._manifest_cache is not None:
            return self._manifest_cache

        symbol_records = {}
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                symbol = record.get("symbol")
                if symbol:
                    if symbol not in symbol_records:
                        symbol_records[symbol] = []
                    symbol_records[symbol].append(record)

        self._manifest_cache = symbol_records
        return symbol_records

    def get_symbol_files(self, symbol: str) -> List[dict]:
        """
        Get all file records for a symbol from manifest.

        Args:
            symbol: Symbol name (e.g., "DCE.JM")

        Returns:
            List of file records sorted by (year, month)
        """
        manifest = self._load_manifest()
        records = manifest.get(symbol, [])
        # Sort by year, month
        return sorted(records, key=lambda r: (r.get("year", 0), r.get("month", 0)))

    def load_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        dedup_eob: str = "last"
    ) -> pd.DataFrame:
        """
        Load all data for a symbol from archive.

        Args:
            symbol: Symbol name
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            dedup_eob: How to handle duplicate eob ('last', 'first', or None)

        Returns:
            DataFrame with columns: eob, open, high, low, close, volume, position, symbol
        """
        records = self.get_symbol_files(symbol)

        if not records:
            raise ValueError(f"No data found for symbol: {symbol}")

        # Load all parquet files
        dfs = []
        for record in records:
            # Convert Windows path to Unix path
            rel_path = record.get("path", "").replace("\\", "/")
            full_path = os.path.join(self.archive_dir, rel_path)

            if not os.path.exists(full_path):
                print(f"Warning: File not found: {full_path}")
                continue

            try:
                df = pd.read_parquet(full_path)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to read {full_path}: {e}")
                continue

        if not dfs:
            raise ValueError(f"No valid parquet files found for symbol: {symbol}")

        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)

        # Ensure required columns exist
        required_cols = ["eob", "open", "high", "low", "close", "volume", "position"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert eob to datetime
        df["eob"] = pd.to_datetime(df["eob"])

        # Filter by date range if specified
        if start_date:
            df = df[df["eob"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["eob"] <= pd.to_datetime(end_date)]

        # Sort by eob
        df = df.sort_values("eob").reset_index(drop=True)

        # Handle duplicate eob
        if dedup_eob:
            dup_count = df["eob"].duplicated().sum()
            if dup_count > 0:
                print(f"Warning: Found {dup_count} duplicate eob timestamps")
                df = df.drop_duplicates(subset=["eob"], keep=dedup_eob)
                df = df.reset_index(drop=True)

        # Keep only essential columns
        keep_cols = ["eob", "open", "high", "low", "close", "volume", "position"]
        if "symbol" in df.columns:
            keep_cols.append("symbol")

        df = df[keep_cols].copy()

        return df

    def validate_dataframe(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Validate loaded dataframe and return statistics.

        Args:
            df: DataFrame to validate
            symbol: Symbol name

        Returns:
            Dict with validation results and statistics
        """
        stats = {
            "symbol": symbol,
            "rows": len(df),
            "min_eob": df["eob"].min(),
            "max_eob": df["eob"].max(),
            "duplicate_eob": 0,
            "is_monotonic": df["eob"].is_monotonic_increasing,
            "missing_values": {},
            "dtype_check": {}
        }

        # Check for duplicates
        stats["duplicate_eob"] = df["eob"].duplicated().sum()

        # Check for missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                stats["missing_values"][col] = missing

        # Check dtypes
        for col in ["open", "high", "low", "close", "volume", "position"]:
            if col in df.columns:
                stats["dtype_check"][col] = str(df[col].dtype)

        return stats
