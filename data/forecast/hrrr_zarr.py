"""HRRR Zarr forecast source (batch-download only; not in production fallback chain).

Reads from the public hrrrzarr S3 bucket using zarr + fsspec.
Avoids downloading full GRIB2 files: each variable is one HTTP GET
because zarr chunks are (48, 150, 150), so KLGA always falls in the
first chunk and is fetched in a single request (~1.7 MB per variable).

Production fallback chain uses GRIB2-based sources (HrrrS3Source, HrrrNomadsSource)
only.  HrrrZarrSource is retained for batch historical downloads via
scripts/batch_download_zarr.py; its per-call zarr-store overhead is too
large for real-time use.

Architecture notes
------------------
- _anl.zarr  : analysis (fxx=0), 2D arrays (y, x)
- _fcst.zarr : forecast (fxx=1..48 for major inits), 3D arrays (t, y, x)
               Python index i == meteorological fxx i+1
- KLGA grid indices: row=698, col=1556 (verified, see run_hrrr_zarr.py)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from data.forecast.base import BaseForecastSource
from data.forecast.hrrr_fill import HrrrFill, _MAJOR_HOURS, _nearest_prior_major
from data.forecast.vars import DATA_COLS, ZARR_VARS

if TYPE_CHECKING:
    from config.stations import StationConfig

# zarr and fsspec are optional heavy dependencies; imported lazily.
try:
    import fsspec as _fsspec
    import zarr as _zarr
except ImportError:
    _fsspec = None  # type: ignore[assignment]
    _zarr = None  # type: ignore[assignment]

_ZARR_DEFAULT_BASE = "https://hrrrzarr.s3.amazonaws.com/sfc"

# DATA_COLS / ZARR_VARS consistency guard: fail at module load if they diverge.
assert set(DATA_COLS) == set(ZARR_VARS.keys()), (
    f"DATA_COLS / ZARR_VARS mismatch -- "
    f"only in DATA_COLS: {set(DATA_COLS) - set(ZARR_VARS.keys())}, "
    f"only in ZARR_VARS: {set(ZARR_VARS.keys()) - set(DATA_COLS)}"
)


def _navigate(root: object, zarr_path: str) -> object:
    """Walk a nested zarr group by splitting path on '/'."""
    node = root
    for part in zarr_path.split("/"):
        node = node[part]  # type: ignore[index]
    return node


class HrrrZarrSource(BaseForecastSource):
    """Fetch HRRR forecasts from the public hrrrzarr S3 bucket.

    This source is for batch/historical downloads only.  It is not included
    in DataManager's production fallback chain.

    Parameters
    ----------
    base_url:
        Override the zarr base URL (useful in tests).  When None, the URL is
        constructed from ``settings.hrrr_zarr_bucket``.

    Requires: zarr, fsspec  (pip install zarr fsspec s3fs)
    """

    def __init__(self, base_url: str | None = None) -> None:
        if base_url is not None:
            self._base_url = base_url.rstrip("/")
        else:
            from config.settings import Settings
            bucket = Settings().hrrr_zarr_bucket
            self._base_url = f"https://{bucket}.s3.amazonaws.com/sfc"
        self._hrrr_fill = HrrrFill()

    def fetch(
        self,
        init_time: pd.Timestamp,
        fxx_range: range,
        station: "StationConfig | None" = None,
    ) -> "dict[str, pd.DataFrame]":
        """Fetch HRRR data for one station via the Zarr store.

        Parameters
        ----------
        station:
            Required for batch use.  When None, raises ValueError.

        Raises
        ------
        ValueError
            When station is None, or when station is missing
            hrrr_grid_row_zarr / hrrr_grid_col_zarr, or when the grid indices
            are out of bounds for the zarr array.
        RuntimeError
            When TMP_2m shape exploration fails (no hardcoded fallback).
        """
        if station is None:
            raise ValueError("HrrrZarrSource.fetch() requires a station argument")

        # Validate required grid config fields -- no fallback to defaults
        if not hasattr(station, "hrrr_grid_row_zarr") or station.hrrr_grid_row_zarr is None:
            raise ValueError(
                f"HrrrZarrSource: station {station.station_id!r} is missing "
                f"hrrr_grid_row_zarr -- cannot fetch zarr data"
            )
        if not hasattr(station, "hrrr_grid_col_zarr") or station.hrrr_grid_col_zarr is None:
            raise ValueError(
                f"HrrrZarrSource: station {station.station_id!r} is missing "
                f"hrrr_grid_col_zarr -- cannot fetch zarr data"
            )

        if _zarr is None or _fsspec is None:
            raise ImportError(
                "zarr and fsspec are required for HrrrZarrSource. "
                "Install with: pip install zarr fsspec s3fs"
            )

        row_idx = int(station.hrrr_grid_row_zarr)
        col_idx = int(station.hrrr_grid_col_zarr)

        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )
        date_str = init_utc.strftime("%Y%m%d")
        hh = int(init_utc.strftime("%H"))
        is_major = hh in _MAJOR_HOURS

        url_anl  = f"{self._base_url}/{date_str}/{date_str}_{hh:02d}z_anl.zarr"
        url_fcst = f"{self._base_url}/{date_str}/{date_str}_{hh:02d}z_fcst.zarr"

        z_anl  = _zarr.open_group(_fsspec.get_mapper(url_anl),  mode="r")
        z_fcst = _zarr.open_group(_fsspec.get_mapper(url_fcst), mode="r")

        # How many forecast steps are actually in the fcst store?
        # Fail fast if TMP_2m shape cannot be read -- no hardcoded fallback.
        try:
            tmp_fcst = _navigate(z_fcst, ZARR_VARS["TMP_2m"])
            n_fcst_steps = int(tmp_fcst.shape[0])  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"HrrrZarrSource: failed to probe TMP_2m shape for "
                f"init={init_utc}: {exc}"
            ) from exc

        # Grid bounds check
        fcst_shape = tmp_fcst.shape  # type: ignore[union-attr]
        if row_idx >= fcst_shape[1] or col_idx >= fcst_shape[2]:
            raise ValueError(
                f"HrrrZarrSource: station {station.station_id!r} grid indices "
                f"({row_idx}, {col_idx}) out of bounds for zarr shape "
                f"{fcst_shape[1]}x{fcst_shape[2]}"
            )

        # Read fxx=0 from _anl.zarr (2D arrays, no time dimension)
        anl_vals: dict[str, float] = {}
        for col, zarr_path in ZARR_VARS.items():
            try:
                arr = _navigate(z_anl, zarr_path)
                anl_vals[col] = float(arr[row_idx, col_idx])  # type: ignore[index]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "HrrrZarrSource: failed to read anl col={} station={} "
                    "zarr_path={} url={}: {}",
                    col, station.station_id, zarr_path, url_anl, exc,
                )
                if col == "TMP_2m":
                    raise RuntimeError(
                        f"HrrrZarrSource: critical variable TMP_2m failed for "
                        f"station={station.station_id} init={init_utc}: {exc}"
                    ) from exc
                anl_vals[col] = (
                    0.0 if ("APCP" in col or "PRATE" in col) else float("nan")
                )

        # Read fxx=1..n_fcst_steps from _fcst.zarr (3D arrays)
        fcst_vals: dict[str, np.ndarray] = {}
        for col, zarr_path in ZARR_VARS.items():
            try:
                arr = _navigate(z_fcst, zarr_path)
                fcst_vals[col] = np.asarray(
                    arr[:n_fcst_steps, row_idx, col_idx],  # type: ignore[index]
                    dtype=np.float32,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "HrrrZarrSource: failed to read fcst col={} station={} "
                    "zarr_path={} url={}: {}",
                    col, station.station_id, zarr_path, url_fcst, exc,
                )
                if col == "TMP_2m":
                    raise RuntimeError(
                        f"HrrrZarrSource: critical variable TMP_2m failed for "
                        f"station={station.station_id} init={init_utc}: {exc}"
                    ) from exc
                fcst_vals[col] = np.full(n_fcst_steps, np.nan, dtype=np.float32)

        if not is_major:
            # Build an 18h DataFrame and splice to 48h via HrrrFill
            rows = []
            for fxx in range(0, 19):
                row: dict = {
                    "init_time": str(init_utc),
                    "fxx": fxx,
                    "valid_time": str(init_utc + pd.Timedelta(hours=fxx)),
                }
                if fxx == 0:
                    for col in DATA_COLS:
                        row[col] = anl_vals.get(col, float("nan"))
                elif 1 <= fxx <= n_fcst_steps:
                    for col in DATA_COLS:
                        vals = fcst_vals.get(col)
                        row[col] = float(vals[fxx - 1]) if vals is not None else float("nan")
                else:
                    for col in DATA_COLS:
                        row[col] = float("nan")
                rows.append(row)
            non_major_df = pd.DataFrame(rows)

            nearest_major_init = _nearest_prior_major(init_utc)
            prior_major_init   = nearest_major_init - pd.Timedelta(hours=6)

            try:
                nearest_dict = self.fetch(nearest_major_init, range(0, 49), station)
                nearest_df   = nearest_dict[station.station_id]
            except Exception:
                logger.warning(
                    "HrrrZarrSource: nearest MAJOR fetch failed for init={}; "
                    "returning non-MAJOR data only",
                    nearest_major_init,
                )
                non_major_df["is_extrapolated"] = False
                return {station.station_id: non_major_df}

            try:
                prior_dict = self.fetch(prior_major_init, range(0, 49), station)
                prior_df   = prior_dict[station.station_id]
            except Exception:
                logger.warning(
                    "HrrrZarrSource: prior MAJOR fetch failed for init={}; "
                    "tail-gap fallback unavailable",
                    prior_major_init,
                )
                prior_df = None

            filled = self._hrrr_fill.fill(non_major_df, nearest_df, prior_df)
            return {station.station_id: filled}

        # MAJOR init: build rows for the requested fxx values
        rows = []
        for fxx in fxx_range:
            row = {
                "init_time": str(init_utc),
                "fxx": fxx,
                "valid_time": str(init_utc + pd.Timedelta(hours=fxx)),
            }
            if fxx == 0:
                for col in DATA_COLS:
                    row[col] = anl_vals.get(col, float("nan"))
            elif 1 <= fxx <= n_fcst_steps:
                for col in DATA_COLS:
                    vals = fcst_vals.get(col)
                    row[col] = float(vals[fxx - 1]) if vals is not None else float("nan")
            else:
                for col in DATA_COLS:
                    row[col] = float("nan")
            rows.append(row)

        df = pd.DataFrame(rows)
        df["is_extrapolated"] = False
        return {station.station_id: df}

