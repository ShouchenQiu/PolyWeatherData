"""HRRR Zarr forecast source (third in fallback chain).

Reads from the public hrrrzarr S3 bucket using zarr + fsspec.
Avoids downloading full GRIB2 files: each variable is one HTTP GET
because zarr chunks are (48, 150, 150), so KLGA always falls in the
first chunk and is fetched in a single request (~1.7 MB per variable).

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

from data.forecast.base import BaseForecastSource
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

_ZARR_BASE = "https://hrrrzarr.s3.amazonaws.com/sfc"
_MAJOR_HOURS = {0, 6, 12, 18}


def _navigate(root: object, zarr_path: str) -> object:
    """Walk a nested zarr group by splitting path on '/'."""
    node = root
    for part in zarr_path.split("/"):
        node = node[part]  # type: ignore[index]
    return node


class HrrrZarrSource(BaseForecastSource):
    """Fetch HRRR forecasts from the public hrrrzarr S3 bucket.

    Requires: zarr, fsspec  (pip install zarr fsspec s3fs)

    Each call to fetch() opens the _anl.zarr and _fcst.zarr stores for
    the init cycle once, then reads all 36 variables in a loop.
    """

    def fetch(
        self,
        init_time: pd.Timestamp,
        fxx_range: range,
        station: "StationConfig",
    ) -> pd.DataFrame:
        if _zarr is None or _fsspec is None:
            raise ImportError(
                "zarr and fsspec are required for HrrrZarrSource. "
                "Install with: pip install zarr fsspec s3fs"
            )

        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )
        date_str = init_utc.strftime("%Y%m%d")
        hh = int(init_utc.strftime("%H"))
        is_major = hh in _MAJOR_HOURS

        url_anl  = f"{_ZARR_BASE}/{date_str}/{date_str}_{hh:02d}z_anl.zarr"
        url_fcst = f"{_ZARR_BASE}/{date_str}/{date_str}_{hh:02d}z_fcst.zarr"

        z_anl  = _zarr.open_group(_fsspec.get_mapper(url_anl),  mode="r")
        z_fcst = _zarr.open_group(_fsspec.get_mapper(url_fcst), mode="r")

        # How many forecast steps are actually in the fcst store?
        try:
            tmp_fcst = _navigate(z_fcst, ZARR_VARS["TMP_2m"])
            n_fcst_steps = int(tmp_fcst.shape[0])  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            n_fcst_steps = (48 if is_major else 18)

        # Grid indices from station config (or fall back to KLGA defaults)
        row_idx = getattr(station, "hrrr_grid_row_zarr", 698)
        col_idx = getattr(station, "hrrr_grid_col_zarr", 1556)

        # Read fxx=0 from _anl.zarr (2D arrays, no time dimension)
        anl_vals: dict[str, float] = {}
        for col, zarr_path in ZARR_VARS.items():
            try:
                arr = _navigate(z_anl, zarr_path)
                anl_vals[col] = float(arr[row_idx, col_idx])  # type: ignore[index]
            except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
                fcst_vals[col] = np.full(n_fcst_steps, np.nan, dtype=np.float32)

        # Build rows for the requested fxx values
        rows = []
        for fxx in fxx_range:
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
                # fxx beyond what this store contains
                for col in DATA_COLS:
                    row[col] = float("nan")
            rows.append(row)

        return pd.DataFrame(rows)
