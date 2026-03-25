"""BatchIemDownloader: bulk-download historical IEM ASOS observations into ParquetStore.

Ported from Herbie-main/iem_download_klga.py.

Downloads per-day ASOS observation history from the IEM obhistory JSON API,
accumulates into a DataFrame, and appends to ParquetStore.

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
import requests

from data.storage.parquet_store import ParquetStore
from config.settings import Settings

_IEM_JSON_URL = (
    "https://mesonet.agron.iastate.edu/api/1/obhistory.json"
    "?station={station}&network={network}&date={date}"
)
_DEFAULT_DELAY   = 0.4    # seconds between requests (polite rate limit)
_DEFAULT_RETRIES = 3


class BatchIemDownloader:
    """Download historical ASOS observations from the IEM obhistory JSON API.

    Parameters
    ----------
    store:
        ParquetStore instance where downloaded data will be written.
    station_id:
        Key used for the ParquetStore sub-directory (e.g. ``"KLGA"``).
    delay:
        Seconds to wait between daily requests to be polite to IEM servers.
    retries:
        Number of retry attempts per day request on failure.
    """

    def __init__(
        self,
        store:      ParquetStore,
        station_id: str = "KLGA",
        delay:      float = _DEFAULT_DELAY,
        retries:    int   = _DEFAULT_RETRIES,
    ) -> None:
        self._store      = store
        self._station_id = station_id
        self._delay      = delay
        self._retries    = retries
        self._session    = requests.Session()
        self._session.headers.update({"User-Agent": "PolyWeatherBot-IEM-downloader/1.0"})

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
            f"{len(done)} already done, station={station}"
        )

        d = d0
        while d <= d1:
            if d.isoformat() in done:
                d += timedelta(days=1)
                continue

            rows = self._fetch_day(d, station, network)
            all_rows.extend(rows)
            total_rows += len(rows)
            print(f"[{d}] {len(rows):4d} obs  (cumulative: {total_rows})")

            d += timedelta(days=1)
            time.sleep(self._delay)

        if all_rows:
            df = pd.DataFrame(all_rows)
            df = self._normalise(df)
            self._store.append(self._station_id, df)
            print(f"\nFlushed {len(df)} rows to store.")

        print(f"BatchIemDownloader done: {total_rows} new rows.")
        return total_rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_day(
        self, d: date, station: str, network: str
    ) -> list[dict]:
        url = _IEM_JSON_URL.format(
            station=station, network=network, date=d.isoformat()
        )
        for attempt in range(self._retries):
            try:
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
                return resp.json().get("data", [])
            except Exception as exc:
                print(f"  [{d}] attempt {attempt + 1} failed: {exc}")
                time.sleep(2 ** attempt)
        return []

    def _already_done_dates(self) -> set[str]:
        """Return ISO date strings already present in the store."""
        try:
            existing = self._store.read(
                self._station_id,
                start=pd.Timestamp("2000-01-01", tz="UTC"),
                end=pd.Timestamp("2100-01-01", tz="UTC"),
            )
            if existing.empty:
                return set()
            col = "utc_valid" if "utc_valid" in existing.columns else None
            if col is None:
                return set()
            return set(
                pd.to_datetime(existing[col], utc=True)
                .dt.date.astype(str)
            )
        except Exception:
            return set()

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent column types before writing to store."""
        if "utc_valid" in df.columns:
            df["utc_valid"] = pd.to_datetime(df["utc_valid"], utc=True)
        return df


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
    store    = ParquetStore(settings.data_store_dir)
    dl = BatchIemDownloader(store, station_id=args.store_id)
    dl.run(args.start, args.end, station=args.station, network=args.network)


if __name__ == "__main__":
    _cli()
