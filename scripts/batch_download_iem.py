"""BatchIemDownloader: bulk-download hourly IEM ASOS observations into ParquetStore.

Downloads one 24-hour ASOS CSV per target date, keeps only exact ``:51``
observations locally, and stores those rows as the next canonical hour after
cleaning.

Usage (CLI):
    python3 -m scripts.batch_download_iem --start 2025-12-01 --end 2026-02-25

Usage (API):
    from scripts.batch_download_iem import BatchIemDownloader
    from data.storage.parquet_store import ParquetStore
    from config.settings import Settings

    s = Settings()
    store = ParquetStore(s.data_store_dir)
    dl = BatchIemDownloader(store)
    dl.run("2025-12-01", "2026-02-25", station="LGA", network="NY_ASOS")
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import pandas as pd

from data.storage.parquet_store import ParquetStore
from data.storage.cleaner import Cleaner
from data.obs.iem import IemSource
from config.settings import Settings
from config.stations import STATIONS

_DEFAULT_DELAY   = 0.4    # seconds between requests (polite rate limit)
_DEFAULT_TIMEOUT = 10
_DEFAULT_RETRIES = 3
_DONE_DAY_MIN_HOURS = 24


class BatchIemDownloader:
    """Download historical hourly representative ASOS observations.

    Parameters
    ----------
    store:
        ParquetStore instance where downloaded data will be written.
    station_id:
        Key used for the ParquetStore sub-directory (e.g. ``"KLGA"``).
    delay:
        Seconds to wait between daily requests to be polite to IEM servers.
    retries:
        Retained for constructor compatibility. IemSource owns request retries.
    """

    def __init__(
        self,
        store:      ParquetStore,
        station_id: str = "KLGA",
        delay:      float = _DEFAULT_DELAY,
        retries:    int   = _DEFAULT_RETRIES,
        station_config = None,
        obs_source = None,
    ) -> None:
        self._store      = store
        self._station    = station_config or STATIONS[station_id]
        self._station_id = self._station.station_id
        self._delay      = delay
        self._source     = obs_source or IemSource(timeout=_DEFAULT_TIMEOUT)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: str,
        end_date:   str,
        station:    str = "LGA",
        network:    str = "NY_ASOS",
    ) -> int:
        """Download daily observations for the given date range.

        Parameters
        ----------
        start_date / end_date:
            ISO date strings ``"YYYY-MM-DD"`` (inclusive).
        station:
            IEM station code (e.g. ``"LGA"``).
        network:
            IEM network code (e.g. ``"NY_ASOS"``).

        Returns
        -------
        Total number of rows appended to the store.
        """
        done = self._already_done_dates()
        d0   = date.fromisoformat(start_date)
        d1   = date.fromisoformat(end_date)
        total_days   = (d1 - d0).days + 1
        total_rows   = 0
        all_rows: list[dict] = []

        print(
            f"BatchIemDownloader: {total_days} days, "
            f"{len(done)} already done, station={station}",
            flush=True,
        )

        d = d0
        while d <= d1:
            if d.isoformat() in done:
                d += timedelta(days=1)
                continue

            # Target canonical hours d 00:00..23:00 are represented by raw
            # METARs from (d-1) 23:51 through d 22:51.
            start = pd.Timestamp(d.isoformat() + " 00:00:00", tz="UTC") - pd.Timedelta(hours=1)
            end = pd.Timestamp(d.isoformat() + " 22:59:59", tz="UTC")
            df_day = self._source.fetch(self._station, start, end)
            if df_day is None or df_day.empty:
                rows = []
            else:
                df_day = Cleaner.process_iem(df_day)
                df_day = self._filter_canonical_day(df_day, d.isoformat())
                rows = df_day.to_dict("records")
            all_rows.extend(rows)
            total_rows += len(rows)
            print(f"[{d}] {len(rows):4d} obs  (cumulative: {total_rows})", flush=True)

            d += timedelta(days=1)
            if self._delay > 0.0:
                time.sleep(self._delay)

        if all_rows:
            df = pd.DataFrame(all_rows)
            written = self._store.append(self._station_id, df)
            print(f"\nFlushed {written} rows to store.", flush=True)

        print(f"BatchIemDownloader done: {total_rows} new rows.", flush=True)
        return total_rows

    @staticmethod
    def _filter_exact_51(df: pd.DataFrame) -> pd.DataFrame:
        """Return only exact :51 observations before hourly canonicalization.

        Retained for compatibility with older tests/callers.  Historical batch
        downloads now pass all candidates to Cleaner so missing exact :51 hours
        can be represented by nearby observations.
        """
        if df.empty or "valid_time" not in df.columns:
            return df.iloc[0:0].copy()
        valid_time = pd.to_datetime(df["valid_time"], utc=True, errors="coerce")
        mask = (
            valid_time.notna()
            & (valid_time.dt.minute == 51)
            & (valid_time.dt.second == 0)
            & (valid_time.dt.microsecond == 0)
        )
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def _filter_canonical_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
        """Keep only canonical hourly rows for the requested UTC day."""
        if df.empty or "valid_time" not in df.columns:
            return df.iloc[0:0].copy()
        start = pd.Timestamp(day + " 00:00:00", tz="UTC")
        end = pd.Timestamp(day + " 23:00:00", tz="UTC")
        valid_time = pd.to_datetime(df["valid_time"], utc=True, errors="coerce")
        mask = valid_time.between(start, end)
        return df.loc[mask].reset_index(drop=True)

    def _already_done_dates(self) -> set[str]:
        """Return dates with enough hourly observation coverage in the store."""
        try:
            existing = self._store.read(
                self._station_id,
                start=pd.Timestamp("2000-01-01", tz="UTC"),
                end=pd.Timestamp("2100-01-01", tz="UTC"),
            )
            if existing.empty:
                return set()
            col = "valid_time" if "valid_time" in existing.columns else None
            if col is None:
                return set()
            hours = pd.to_datetime(existing[col], utc=True).dt.floor("h")
            coverage = (
                pd.DataFrame({"date": hours.dt.date.astype(str), "hour": hours})
                .drop_duplicates()
                .groupby("date")["hour"]
                .nunique()
            )
            return set(coverage[coverage >= _DONE_DAY_MIN_HOURS].index)
        except Exception:
            return set()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Batch IEM ASOS downloader")
    parser.add_argument("--start",   required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--station", default="LGA",     help="IEM station code")
    parser.add_argument("--network", default="NY_ASOS", help="IEM network code")
    parser.add_argument("--store-id", default="KLGA",  dest="store_id",
                        help="ParquetStore sub-directory key")
    args = parser.parse_args()

    settings = Settings()
    store    = ParquetStore(settings.obs_store_dir)
    dl = BatchIemDownloader(store, station_id=args.store_id)
    dl.run(args.start, args.end, station=args.station, network=args.network)


if __name__ == "__main__":
    _cli()
