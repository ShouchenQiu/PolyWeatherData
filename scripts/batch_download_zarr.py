"""BatchZarrDownloader: bulk-download historical HRRR Zarr data into ParquetStore.

Ported from Herbie-main/run_hrrr_zarr.py.

Downloads all init cycles in a date range, reads 36 HRRR variables at the
target station grid point via the hrrrzarr S3 bucket, and writes results
into a ParquetStore partition.

Usage (CLI):
    python3 -m scripts.batch_download_zarr --start 2025-12-01 --end 2026-02-25

Usage (API):
    from scripts.batch_download_zarr import BatchZarrDownloader
    from data.storage.parquet_store import ParquetStore
    from config.settings import Settings

    s = Settings()
    store = ParquetStore(s.data_store_dir)
    dl = BatchZarrDownloader(store, row=698, col=1556, max_workers=12)
    dl.run("2025-12-01", "2026-02-25")
"""
from __future__ import annotations

import argparse
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd

from data.storage.parquet_store import ParquetStore
from data.storage.cleaner import Cleaner
from data.forecast.hrrr_zarr import HrrrZarrSource
from config.settings import Settings
from config.stations import STATIONS

# ---------------------------------------------------------------------------
# HRRR Zarr variable map
# ---------------------------------------------------------------------------
# Maps output column name -> nested Zarr path within the fcst/anl store
_ZARR_VARS: dict[str, str] = {
    "TMP_2m":    "2m_above_ground/TMP/2m_above_ground/TMP",
    "DPT_2m":    "2m_above_ground/DPT/2m_above_ground/DPT",
    "RH_2m":     "2m_above_ground/RH/2m_above_ground/RH",
    "TMP_sfc":   "surface/TMP/surface/TMP",
    "TMP_850mb": "850mb/TMP/850mb/TMP",
    "TMP_500mb": "500mb/TMP/500mb/TMP",
    "UGRD_10m":   "10m_above_ground/UGRD/10m_above_ground/UGRD",
    "VGRD_10m":   "10m_above_ground/VGRD/10m_above_ground/VGRD",
    "GUST_sfc":   "surface/GUST/surface/GUST",
    "UGRD_80m":   "80m_above_ground/UGRD/80m_above_ground/UGRD",
    "VGRD_80m":   "80m_above_ground/VGRD/80m_above_ground/VGRD",
    "UGRD_500mb": "500mb/UGRD/500mb/UGRD",
    "VGRD_500mb": "500mb/VGRD/500mb/VGRD",
    "RH_0C":  "0C_isotherm/RH/0C_isotherm/RH",
    "RH_frz": "highest_tropospheric_freezing_level/RH/highest_tropospheric_freezing_level/RH",
    "DSWRF": "surface/DSWRF/surface/DSWRF",
    "DLWRF": "surface/DLWRF/surface/DLWRF",
    "USWRF": "surface/USWRF/surface/USWRF",
    "ULWRF": "surface/ULWRF/surface/ULWRF",
    "TCDC": "entire_atmosphere/TCDC/entire_atmosphere/TCDC",
    "LCDC": "low_cloud_layer/LCDC/low_cloud_layer/LCDC",
    "MCDC": "middle_cloud_layer/MCDC/middle_cloud_layer/MCDC",
    "HCDC": "high_cloud_layer/HCDC/high_cloud_layer/HCDC",
    "PRATE": "surface/PRATE/surface/PRATE",
    "APCP":  "surface/APCP_1hr_acc_fcst/surface/APCP_1hr_acc_fcst",
    "REFC": "entire_atmosphere/REFC/entire_atmosphere/REFC",
    "PWAT": "entire_atmosphere_single_layer/PWAT/entire_atmosphere_single_layer/PWAT",
    "CAPE": "surface/CAPE/surface/CAPE",
    "CIN":  "surface/CIN/surface/CIN",
    "HPBL":  "surface/HPBL/surface/HPBL",
    "SHTFL": "surface/SHTFL/surface/SHTFL",
    "LHTFL": "surface/LHTFL/surface/LHTFL",
    "PRES_sfc": "surface/PRES/surface/PRES",
    "MSLMA":    "mean_sea_level/MSLMA/mean_sea_level/MSLMA",
    "VIS":      "surface/VIS/surface/VIS",
    "SNOD":     "surface/SNOD/surface/SNOD",
}

_MAJOR_HOURS = frozenset({0, 6, 12, 18})
_FXX_MAJOR   = 48
_FXX_OFFHOUR = 18
_BASE_URL     = "https://hrrrzarr.s3.amazonaws.com/sfc"
_S3_BASE      = "s3://hrrrzarr/sfc"
_SAVE_EVERY   = 24    # checkpoint after this many completed inits


def _navigate(z: Any, zarr_path: str) -> Any:
    """Navigate nested zarr groups by splitting path on '/'."""
    node = z
    for part in zarr_path.split("/"):
        node = node[part]
    return node


def _mapper_for_url(fsspec_mod: Any, url: str) -> Any:
    """Use anonymous S3 for the public hrrrzarr bucket."""
    if url.startswith(_BASE_URL):
        s3_url = _S3_BASE + url[len(_BASE_URL):]
        return fsspec_mod.filesystem("s3", anon=True).get_mapper(s3_url)
    return fsspec_mod.get_mapper(url)


def _fetch_init(
    date_obj: Any,
    hh: int,
    row: int,
    col: int,
) -> list[dict]:
    """Fetch all variables at (row, col) for one HRRR init cycle.

    Returns a list of dicts, one per forecast hour (fxx=0..max_fxx).
    """
    import fsspec
    import zarr as _zarr

    date_str = date_obj.strftime("%Y%m%d")
    max_fxx  = _FXX_MAJOR if hh in _MAJOR_HOURS else _FXX_OFFHOUR

    url_anl  = f"{_BASE_URL}/{date_str}/{date_str}_{hh:02d}z_anl.zarr"
    url_fcst = f"{_BASE_URL}/{date_str}/{date_str}_{hh:02d}z_fcst.zarr"
    z_anl  = _zarr.open_group(_mapper_for_url(fsspec, url_anl),  mode="r")
    z_fcst = _zarr.open_group(_mapper_for_url(fsspec, url_fcst), mode="r")

    # Detect actual available forecast steps
    try:
        tmp_fcst    = _navigate(z_fcst, _ZARR_VARS["TMP_2m"])
        n_fcst_steps = min(int(tmp_fcst.shape[0]), max_fxx)
    except Exception:
        n_fcst_steps = max_fxx

    # Analysis (fxx=0) from _anl.zarr -- 2D arrays (y, x)
    anl_vals: dict[str, float] = {}
    for var, zarr_path in _ZARR_VARS.items():
        try:
            arr = _navigate(z_anl, zarr_path)
            anl_vals[var] = float(arr[row, col])
        except Exception:
            anl_vals[var] = 0.0 if ("APCP" in var or "PRATE" in var) else float("nan")

    # Forecast (fxx=1..n) from _fcst.zarr -- 3D arrays (fxx-1, y, x)
    fcst_vals: dict[str, np.ndarray] = {}
    for var, zarr_path in _ZARR_VARS.items():
        try:
            arr = _navigate(z_fcst, zarr_path)
            fcst_vals[var] = np.asarray(
                arr[:n_fcst_steps, row, col], dtype=np.float32
            )
        except Exception:
            fcst_vals[var] = np.full(n_fcst_steps, float("nan"), dtype=np.float32)

    init_ts = datetime(
        date_obj.year, date_obj.month, date_obj.day, hh, tzinfo=timezone.utc
    )
    rows: list[dict] = []

    # fxx=0 analysis row
    row0: dict = {
        "init_time":  init_ts,
        "fxx":        0,
        "valid_time": init_ts,
    }
    row0.update(anl_vals)
    rows.append(row0)

    # fxx=1..n forecast rows
    for i in range(n_fcst_steps):
        fxx = i + 1
        r: dict = {
            "init_time":  init_ts,
            "fxx":        fxx,
            "valid_time": init_ts + timedelta(hours=fxx),
        }
        for var in _ZARR_VARS:
            r[var] = float(fcst_vals[var][i])
        rows.append(r)

    return rows


class BatchZarrDownloader:
    """Download historical HRRR Zarr data for a single grid point.

    Parameters
    ----------
    store:
        ParquetStore instance where downloaded data will be written.
    station_id:
        Station label used as the sub-directory key in ParquetStore
        (e.g. ``"KLGA"``).
    row:
        HRRR Lambert Conformal grid row index (default: 698 for KLGA).
    col:
        HRRR Lambert Conformal grid column index (default: 1556 for KLGA).
    max_workers:
        Number of parallel HTTP threads.
    save_every:
        Flush buffered rows to ParquetStore every N completed inits.
    """

    def __init__(
        self,
        store:      ParquetStore,
        station_id: str = "KLGA",
        row:        int = 698,
        col:        int = 1556,
        max_workers: int = 12,
        save_every:  int = _SAVE_EVERY,
        station_config: Any | None = None,
        forecast_source: Any | None = None,
    ) -> None:
        self._store       = store
        self._station     = station_config or STATIONS[station_id]
        self._station_id  = self._station.station_id
        self._row         = row
        self._col         = col
        self._max_workers = max_workers
        self._save_every  = save_every
        self._source      = forecast_source or HrrrZarrSource()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: str,
        end_date: str,
        init_hours: list[int] | None = None,
    ) -> int:
        """Download all init cycles between start_date and end_date (inclusive).

        Parameters
        ----------
        start_date / end_date:
            ISO date strings ``"YYYY-MM-DD"``.

        Returns
        -------
        Total number of rows appended to the store.
        """
        all_inits = self._build_inits(start_date, end_date, init_hours=init_hours)
        done_keys = self._already_done_keys()
        pending   = [
            (d, h) for d, h in all_inits
            if f"{d.strftime('%Y%m%d')}_{h:02d}z" not in done_keys
        ]
        print(
            f"BatchZarrDownloader: {len(all_inits)} inits total, "
            f"{len(done_keys)} already done, {len(pending)} pending."
        )
        if not pending:
            return 0

        return self._execute_via_source(pending)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_inits(
        self,
        start_date: str,
        end_date: str,
        init_hours: list[int] | None = None,
    ) -> list[tuple[Any, int]]:
        dates = pd.date_range(start_date, end_date, freq="D")
        hours = init_hours if init_hours is not None else list(range(24))
        return [(d.date(), h) for d in dates for h in hours]

    def _already_done_keys(self) -> set[str]:
        """Return set of already-downloaded init key strings."""
        try:
            existing = self._store.read(
                self._station_id,
                start=pd.Timestamp("2000-01-01", tz="UTC"),
                end=pd.Timestamp("2100-01-01", tz="UTC"),
            )
            if existing.empty or "init_time" not in existing.columns:
                return set()
            existing["init_time"] = pd.to_datetime(existing["init_time"], utc=True)
            return set(existing["init_time"].dt.strftime("%Y%m%d_%Hz"))
        except Exception:
            return set()

    def _execute(self, pending: list[tuple[Any, int]]) -> int:
        """Run parallel downloads and flush periodically."""
        buffer: list[dict]  = []
        total_rows           = 0
        completed            = 0
        errors               = 0
        flushed_count        = 0
        t0                   = time.time()

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_map = {
                pool.submit(_fetch_init, d, h, self._row, self._col): (d, h)
                for d, h in pending
            }
            for fut in as_completed(future_map):
                d, h = future_map[fut]
                completed += 1
                try:
                    rows = fut.result(timeout=120)
                    buffer.extend(rows)
                except Exception as exc:
                    errors += 1
                    print(f"  ERROR {d} {h:02d}z: {exc}")
                    if errors <= 3:
                        traceback.print_exc()

                elapsed   = time.time() - t0
                rate      = completed / max(elapsed, 0.001)
                remaining = len(pending) - completed
                eta_h     = remaining / rate / 3600.0

                if completed % 10 == 0 or completed <= 5:
                    print(
                        f"[{completed}/{len(pending)}] "
                        f"rate={rate:.2f} init/s  "
                        f"ETA={eta_h:.1f}h  "
                        f"errors={errors}  "
                        f"buffered={len(buffer)}"
                    )

                if completed - flushed_count >= self._save_every:
                    n = self._flush(buffer)
                    total_rows  += n
                    buffer       = []
                    flushed_count = completed
                    print(f"  Checkpoint: flushed {n} rows (total so far: {total_rows})")

        # Final flush
        if buffer:
            n = self._flush(buffer)
            total_rows += n

        elapsed = time.time() - t0
        print(
            f"\nBatchZarrDownloader done: {total_rows} rows "
            f"in {elapsed/60:.1f} min, errors={errors}"
        )
        return total_rows

    def _execute_via_source(self, pending: list[tuple[Any, int]]) -> int:
        """Fetch historical HRRR through the production source/clean/store path."""
        total_rows = 0
        errors = 0
        for completed, (d, h) in enumerate(pending, start=1):
            init_utc = pd.Timestamp(
                datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc)
            )
            try:
                result = self._source.fetch(init_utc, range(1, 2), self._station)
                df = result.get(self._station_id, pd.DataFrame())
                if df is None or df.empty:
                    print(f"  EMPTY {d} {h:02d}z")
                    continue
                df = Cleaner.process_hrrr(df)
                n = self._store.append(self._station_id, df)
                total_rows += n
            except Exception as exc:
                errors += 1
                print(f"  ERROR {d} {h:02d}z: {exc}")
                if errors <= 3:
                    traceback.print_exc()
            if completed % 10 == 0 or completed <= 5:
                print(
                    f"[{completed}/{len(pending)}] "
                    f"errors={errors} written={total_rows}"
                )
        print(f"\nBatchZarrDownloader done: {total_rows} rows, errors={errors}")
        return total_rows

    def _flush(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        self._store.append(self._station_id, df)
        return len(df)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Batch HRRR Zarr downloader")
    parser.add_argument("--start",    required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--station",  default="KLGA")
    parser.add_argument("--row",      type=int, default=698)
    parser.add_argument("--col",      type=int, default=1556)
    parser.add_argument("--workers",  type=int, default=12)
    args = parser.parse_args()

    settings = Settings()
    store    = ParquetStore(settings.forecast_store_dir)
    dl = BatchZarrDownloader(
        store,
        station_id  = args.station,
        row         = args.row,
        col         = args.col,
        max_workers = args.workers,
    )
    dl.run(args.start, args.end)


if __name__ == "__main__":
    _cli()
