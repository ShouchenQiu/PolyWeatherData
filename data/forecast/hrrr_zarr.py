"""HRRR Zarr forecast source (batch-download only; not in production fallback chain).

Reads from the public hrrrzarr S3 bucket using zarr + fsspec.
Avoids downloading full GRIB2 files: each variable is one HTTP GET
because zarr chunks are (48, 150, 150), so KLGA always falls in the
first chunk and is fetched in a single request (~1.7 MB per variable).

Production fallback chain uses GRIB2-based sources (HrrrS3Source, HrrrNomadsSource)
only.  HrrrZarrSource is retained for batch historical downloads via
``training.downloaders``; its per-call zarr-store overhead is too large for
real-time use.

Architecture notes
------------------
- _anl.zarr  : analysis (fxx=0), 2D arrays (y, x)
- _fcst.zarr : forecast (fxx=1..48 for major inits), 3D arrays (t, y, x)
               Python index i == meteorological fxx i+1
- KLGA grid indices: row=698, col=1556 (verified, see run_hrrr_zarr.py)
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np
import pandas as pd
from loguru import logger

from data.forecast.base import BaseForecastSource
from data.forecast.hrrr_fill import HrrrFill, MAJOR_HOURS, nearest_prior_major
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
_ZARR_S3_BASE = "s3://hrrrzarr/sfc"
_VAR_READ_THREADS = 12
_T = TypeVar("_T")

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


def _mapper_for_url(url: str) -> object:
    """Return an fsspec mapper for HRRR Zarr stores.

    The public hrrrzarr bucket is much faster through s3fs anonymous access
    than through the HTTPS mapper, which can spend tens of seconds opening
    metadata before any variable read begins.
    """
    if _fsspec is None:
        raise ImportError(
            "fsspec is required for HrrrZarrSource. "
            "Install with: pip install fsspec s3fs"
        )
    if url.startswith(_ZARR_DEFAULT_BASE):
        s3_url = _ZARR_S3_BASE + url[len(_ZARR_DEFAULT_BASE):]
        return _fsspec.filesystem("s3", anon=True).get_mapper(s3_url)
    return _fsspec.get_mapper(url)


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

    def __init__(
        self,
        base_url: str | None = None,
        critical_retries: int = 2,
        retry_backoff_s: float = 0.25,
    ) -> None:
        if base_url is not None:
            self._base_url = base_url.rstrip("/")
        else:
            from config.settings import Settings
            bucket = Settings().hrrr_zarr_bucket
            self._base_url = f"https://{bucket}.s3.amazonaws.com/sfc"
        self._critical_retries = max(0, int(critical_retries))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._hrrr_fill = HrrrFill()
        self._full_window_cache: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
        self._cache_lock = threading.Lock()

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
        is_major = hh in MAJOR_HOURS

        url_anl  = f"{self._base_url}/{date_str}/{date_str}_{hh:02d}z_anl.zarr"
        url_fcst = f"{self._base_url}/{date_str}/{date_str}_{hh:02d}z_fcst.zarr"

        requested_fxx = sorted({int(fxx) for fxx in fxx_range})
        if not requested_fxx:
            return {station.station_id: pd.DataFrame()}
        cache_key: tuple[str, pd.Timestamp] | None = None
        if requested_fxx == list(range(0, 49)):
            cache_key = (station.station_id, init_utc)
            with self._cache_lock:
                cached = self._full_window_cache.get(cache_key)
            if cached is not None:
                return {station.station_id: cached.copy()}

        wants_anl = 0 in requested_fxx
        wants_fcst = any(fxx > 0 for fxx in requested_fxx)

        z_anl = None
        z_fcst = None
        tmp_fcst = None
        n_fcst_steps = 0

        if wants_anl:
            z_anl = _zarr.open_group(_mapper_for_url(url_anl), mode="r")

        # How many forecast steps are actually in the fcst store?
        # Fail fast if TMP_2m shape cannot be read -- no hardcoded fallback.
        if wants_fcst:
            z_fcst = _zarr.open_group(_mapper_for_url(url_fcst), mode="r")
            try:
                tmp_fcst = self._retry_critical(
                    "TMP_2m shape probe",
                    lambda: _navigate(z_fcst, ZARR_VARS["TMP_2m"]),
                )
                raw_fcst_steps = int(tmp_fcst.shape[0])  # type: ignore[union-attr]
                max_native = 48 if is_major else 18
                n_fcst_steps = min(raw_fcst_steps, max_native)
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

        needs_fill = (
            not is_major
            and wants_fcst
            and max((fxx for fxx in requested_fxx if fxx > 0), default=0) > n_fcst_steps
        )
        if needs_fill:
            wants_anl = True

        if wants_anl and z_anl is None:
            z_anl = _zarr.open_group(_mapper_for_url(url_anl), mode="r")

        # Read fxx=0 from _anl.zarr (2D arrays, no time dimension)
        anl_vals: dict[str, float] = {}
        if z_anl is not None:
            def read_anl_col(col: str, zarr_path: str) -> tuple[str, float]:
                try:
                    def read_value() -> float:
                        arr = _navigate(z_anl, zarr_path)
                        return float(arr[row_idx, col_idx])  # type: ignore[index]

                    if col == "TMP_2m":
                        return col, self._retry_critical("TMP_2m anl read", read_value)
                    return col, read_value()
                except Exception as exc:  # noqa: BLE001
                    if col in {"APCP", "PRATE"}:
                        return col, 0.0
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
                    return (
                        col,
                        0.0 if ("APCP" in col or "PRATE" in col) else float("nan"),
                    )

            with ThreadPoolExecutor(max_workers=_VAR_READ_THREADS) as pool:
                futures = [
                    pool.submit(read_anl_col, col, zarr_path)
                    for col, zarr_path in ZARR_VARS.items()
                ]
                for fut in as_completed(futures):
                    col, val = fut.result()
                    anl_vals[col] = val

        # Read fxx=1..n_fcst_steps from _fcst.zarr (3D arrays)
        fcst_fxx = [
            fxx
            for fxx in (range(1, n_fcst_steps + 1) if needs_fill else requested_fxx)
            if 1 <= fxx <= n_fcst_steps
        ]
        fcst_vals: dict[str, dict[int, float]] = {}
        if z_fcst is not None and fcst_fxx:
            start_idx = min(fcst_fxx) - 1
            stop_idx = max(fcst_fxx)

            def read_fcst_col(col: str, zarr_path: str) -> tuple[str, dict[int, float]]:
                try:
                    def read_values() -> np.ndarray:
                        arr = _navigate(z_fcst, zarr_path)
                        return np.atleast_1d(
                            np.asarray(
                                arr[start_idx:stop_idx, row_idx, col_idx],  # type: ignore[index]
                                dtype=np.float32,
                            )
                        )

                    vals = (
                        self._retry_critical("TMP_2m fcst read", read_values)
                        if col == "TMP_2m"
                        else read_values()
                    )
                    return col, {
                        fxx: float(vals[fxx - 1 - start_idx]) for fxx in fcst_fxx
                    }
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
                    return col, {fxx: float("nan") for fxx in fcst_fxx}

            with ThreadPoolExecutor(max_workers=_VAR_READ_THREADS) as pool:
                futures = [
                    pool.submit(read_fcst_col, col, zarr_path)
                    for col, zarr_path in ZARR_VARS.items()
                ]
                for fut in as_completed(futures):
                    col, vals = fut.result()
                    fcst_vals[col] = vals

        if needs_fill:
            # Build an 18h DataFrame and splice to 48h via HrrrFill
            rows = []
            for fxx in range(0, n_fcst_steps + 1):
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
                        vals = fcst_vals.get(col, {})
                        row[col] = vals.get(fxx, float("nan"))
                else:
                    for col in DATA_COLS:
                        row[col] = float("nan")
                rows.append(row)
            non_major_df = pd.DataFrame(rows)

            nearest_major_init = nearest_prior_major(init_utc)

            try:
                nearest_dict = self.fetch(nearest_major_init, range(0, 49), station)
                nearest_df   = nearest_dict[station.station_id]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "HrrrZarrSource: nearest MAJOR fetch failed for init={}; "
                    "{}",
                    nearest_major_init,
                    "raising for full decoder window"
                    if cache_key is not None
                    else "returning non-MAJOR data only",
                )
                if cache_key is not None:
                    raise RuntimeError(
                        f"HrrrZarrSource: nearest MAJOR fetch failed for "
                        f"full decoder init={init_utc}: {exc}"
                    ) from exc
                non_major_df["is_extrapolated"] = False
                return self._maybe_cache_result(station.station_id, cache_key, non_major_df)

            filled = self._hrrr_fill.fill(non_major_df, nearest_df)
            filled = filled[filled["fxx"].isin(requested_fxx)].reset_index(drop=True)
            return self._maybe_cache_result(station.station_id, cache_key, filled)

        # Build rows for the requested native fxx values.  Values beyond the
        # native store range stay NaN for MAJOR runs; non-MAJOR tail requests
        # are handled by the fill branch above.
        rows = []
        for fxx in requested_fxx:
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
                    vals = fcst_vals.get(col, {})
                    row[col] = vals.get(fxx, float("nan"))
            else:
                for col in DATA_COLS:
                    row[col] = float("nan")
            rows.append(row)

        df = pd.DataFrame(rows)
        df["is_extrapolated"] = False
        return self._maybe_cache_result(station.station_id, cache_key, df)

    def _retry_critical(self, label: str, func: Callable[[], _T]) -> _T:
        last_exc: Exception | None = None
        for attempt in range(self._critical_retries + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self._critical_retries:
                    break
                if self._retry_backoff_s > 0:
                    time.sleep(self._retry_backoff_s * (2 ** attempt))
                logger.warning(
                    "HrrrZarrSource: retrying critical {} after failure {}/{}: {}",
                    label,
                    attempt + 1,
                    self._critical_retries,
                    exc,
                )
        assert last_exc is not None
        raise last_exc

    def _maybe_cache_result(
        self,
        station_id: str,
        cache_key: tuple[str, pd.Timestamp] | None,
        df: pd.DataFrame,
    ) -> "dict[str, pd.DataFrame]":
        if cache_key is not None:
            with self._cache_lock:
                self._full_window_cache[cache_key] = df.copy()
        return {station_id: df}
