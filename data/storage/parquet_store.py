from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


class ParquetStore:
    """Quarterly-partitioned Parquet store for time-series weather data.

    Directory layout::

        base_dir/
            KLGA/
                2025-Q4.parquet
                2026-Q1.parquet
                ...
    """

    # Dedup key candidates (checked in order; first match wins)
    _HRRR_KEY = ["init_time", "fxx"]
    _OBS_KEY = ["valid_time"]

    # Columns that hold timestamps and must be normalised to UTC before
    # dedup key comparison (protects against string vs Timestamp repr mismatches)
    _TIMESTAMP_KEY_COLS: frozenset[str] = frozenset({"init_time", "valid_time"})

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, station_id: str, df: pd.DataFrame) -> int:
        """Deduplicate and append rows.  Returns number of newly written rows.

        HRRR enforcement (ADR-0003): rows with fxx != 1 are dropped before
        writing.  Only the first forecast hour (fxx=1) is stored.
        """
        if df.empty:
            return 0

        # ADR-0003: f01-only Parquet storage
        if "fxx" in df.columns:
            df = df[df["fxx"] == 1].copy()
            if df.empty:
                return 0

        dedup_key = self._pick_dedup_key(df)
        df = df.copy()

        # Collect all quarters touched by the new data
        ts_col = dedup_key[0]
        quarters: set[str] = set()
        for ts in pd.to_datetime(df[ts_col]):
            quarters.add(self._quarter_key(ts))

        new_total = 0
        for q in sorted(quarters):
            mask = df[ts_col].apply(lambda t, _q=q: self._quarter_key(pd.Timestamp(t)) == _q)
            chunk = df[mask]
            new_total += self._append_to_quarter(station_id, q, chunk, dedup_key)

        return new_total

    def read(
        self,
        station_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Read all rows with primary timestamp in [start, end]."""
        station_dir = self._station_path(station_id)
        if not station_dir.exists():
            return pd.DataFrame()

        q_start = self._quarter_key(start)
        q_end = self._quarter_key(end)

        parts: list[pd.DataFrame] = []
        for path in sorted(station_dir.glob("*.parquet")):
            q = path.stem
            if q < q_start or q > q_end:
                continue
            parts.append(pd.read_parquet(path))

        if not parts:
            return pd.DataFrame()

        combined = pd.concat(parts, ignore_index=True)
        ts_col = self._pick_dedup_key(combined)[0]
        ts = pd.to_datetime(combined[ts_col], utc=True)
        filtered = combined[(ts >= start) & (ts <= end)]
        return filtered.sort_values(ts_col).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quarter_key(self, ts: pd.Timestamp) -> str:
        """Return partition key like '2026-Q1'."""
        q = math.ceil(ts.month / 3)
        return f"{ts.year}-Q{q}"

    def _station_path(self, station_id: str) -> Path:
        """Return the station directory path without creating it."""
        return self._base / station_id

    def _ensure_station_dir(self, station_id: str) -> Path:
        """Return the station directory path, creating it if necessary."""
        d = self._station_path(station_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _pick_dedup_key(self, df: pd.DataFrame) -> list[str]:
        if all(c in df.columns for c in self._HRRR_KEY):
            return self._HRRR_KEY
        if self._OBS_KEY[0] in df.columns:
            return self._OBS_KEY
        raise ValueError(
            "ParquetStore requires either init_time+fxx (HRRR) or valid_time (obs)"
        )

    def _make_key_tuples(
        self, df: pd.DataFrame, dedup_key: list[str]
    ) -> pd.Series:
        """Build comparable key tuples, normalising timestamp columns to UTC."""
        norm: dict[str, pd.Series] = {}
        for k in dedup_key:
            col = df[k]
            if k in self._TIMESTAMP_KEY_COLS:
                col = pd.to_datetime(col, utc=True)
            norm[k] = col
        return pd.DataFrame(norm).apply(lambda r: tuple(r), axis=1)

    def _append_to_quarter(
        self,
        station_id: str,
        quarter: str,
        new_rows: pd.DataFrame,
        dedup_key: list[str],
    ) -> int:
        # TODO: _append_to_quarter is not safe for concurrent multi-process
        # access.  If backfill scripts and the live bot ever run in parallel,
        # replace to_parquet() with write-to-tmp + os.replace() and add a
        # per-file lock so two writers cannot corrupt the same quarter file.
        path = self._ensure_station_dir(station_id) / f"{quarter}.parquet"

        if path.exists():
            existing = pd.read_parquet(path)
            # Normalise timestamps before comparison to avoid string vs
            # Timestamp repr mismatches across Parquet round-trips (#45)
            existing_keys = self._make_key_tuples(existing, dedup_key)
            new_keys = self._make_key_tuples(new_rows, dedup_key)
            truly_new = new_rows[~new_keys.isin(set(existing_keys))]
            if truly_new.empty:
                return 0
            combined = pd.concat([existing, truly_new], ignore_index=True)
        else:
            combined = new_rows
            truly_new = new_rows

        combined.to_parquet(path, index=False)
        return len(truly_new)
