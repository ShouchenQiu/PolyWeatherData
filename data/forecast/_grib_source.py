"""Shared GRIB2 (idx + range-request + eccodes) logic for HRRR data sources.

Both HrrrS3Source and HrrrNomadsSource use the same .idx parsing, HTTP Range
Request download, and eccodes decoding steps; only the URL base differs.
This module provides the shared BaseGribSource base class.

Multi-station design
--------------------
fetch() decodes all configured STATIONS in one eccodes pass per GRIB2 message.
Grid indices are cached per station_id after the first decode to avoid repeated
lat/lon searches.

Non-MAJOR init_times (03Z, 09Z, 15Z, 21Z)
------------------------------------------
Only fxx 0-18 are available from NOAA for non-MAJOR runs.  fetch() detects the
init hour, downloads fxx 0-18, then fetches the nearest prior MAJOR-hour run
(fxx 0-48) and splices them via HrrrFill.  The final 1-5 uncovered hours are
bounded linear extrapolations; an earlier MAJOR run cannot cover later
valid_times and is not fetched.  [ADR-0012]
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getenv
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data.forecast.base import BaseForecastSource
from data.forecast.hrrr_fill import HrrrFill, MAJOR_HOURS, nearest_prior_major
from data.forecast.vars import GRIB_TO_COL, HRRR_SEARCHES

if TYPE_CHECKING:
    from config.stations import StationConfig

# eccodes is an optional heavy dependency; import lazily so the module still
# loads in test environments that do not have it installed.
try:
    import eccodes as _eccodes
except ImportError:
    _eccodes = None  # type: ignore[assignment]

_RANGE_THREADS = 24
_CONNECT_THREADS = 32
_RETRY_TOTAL = 4
_RETRY_BACKOFF = 1.5
_NON_MAJOR_FXX = range(0, 19)   # non-MAJOR runs have fxx 0-18 only


def _is_one_hour_accum(desc: str) -> bool:
    """Return True for descriptors like '1-2 hour acc fcst'."""
    if "hour acc fcst" not in desc:
        return False
    window = desc.split(" ", 1)[0]
    if "-" not in window:
        return False
    start_str, end_str = window.split("-", 1)
    try:
        return int(end_str) - int(start_str) == 1
    except ValueError:
        return False


def _make_session() -> requests.Session:
    sess = requests.Session()
    grib_proxy = getenv("POLYWEATHER_GRIB_PROXY", "").strip()
    if grib_proxy:
        sess.trust_env = False
        sess.proxies.update({"http": grib_proxy, "https": grib_proxy})
    adapter = HTTPAdapter(
        pool_connections=_CONNECT_THREADS,
        pool_maxsize=64,
        max_retries=Retry(
            total=_RETRY_TOTAL,
            backoff_factor=_RETRY_BACKOFF,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        ),
    )
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _is_major(init_time: pd.Timestamp) -> bool:
    """Return True when init_time falls on a MAJOR hour (00Z/06Z/12Z/18Z)."""
    utc = (
        init_time.tz_localize("UTC") if init_time.tzinfo is None
        else init_time.tz_convert("UTC")
    )
    return utc.hour in MAJOR_HOURS


class BaseGribSource(BaseForecastSource):
    """Shared base for sources that download GRIB2 via .idx + Range Requests.

    Subclasses must implement ``_build_url`` to return the GRIB2 file URL for
    a given init date, init hour, and forecast step.
    """

    def __init__(self) -> None:
        self._session: requests.Session = _make_session()
        # Per-station grid-index cache keyed by station_id.
        # Populated on first decode; avoids repeated lat/lon searches.
        self._idx_cache: dict[str, int] = {}
        self._idx_lock: threading.Lock = threading.Lock()
        self._hrrr_fill = HrrrFill()

    # ------------------------------------------------------------------
    # Abstract helpers
    # ------------------------------------------------------------------

    def _build_url(self, date_str: str, init_hh: str, fxx_str: str) -> str:
        """Return full GRIB2 URL for the given parameters."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # .idx parsing
    # ------------------------------------------------------------------

    def _parse_idx(self, idx_text: str) -> list[tuple[int, int | None, str]]:
        """Parse a .idx file and return [(start_byte, end_byte, col_name), ...].

        The .idx format is:
            msgnum:byte_offset:d=YYYYMMDDHHZ:VAR:LEVEL:DESCRIPTOR:...

        Uses exact VAR + LEVEL + DESCRIPTOR three-field match to prevent:
        - Prefix collisions (MAXTMP must not match TMP).
        - Level cross-matches (TMP:surface must not match TMP:2 m above ground).
        - Descriptor mismatches (same VAR/LEVEL under different accumulation
          periods or layer selectors).  When HRRR_SEARCHES has descriptor=None
          the descriptor field is ignored and any value is accepted.

        Raises
        ------
        ValueError
            If more than one .idx row maps to the same target column name.
            The error message includes the matched VAR:LEVEL:DESCRIPTOR fields.
        """
        offsets:    list[int] = []
        var_fields:  list[str] = []
        level_fields: list[str] = []
        desc_fields:  list[str] = []

        for ln in idx_text.strip().splitlines():
            parts = ln.split(":")
            if len(parts) >= 6:
                try:
                    offsets.append(int(parts[1]))
                    var_fields.append(parts[3])
                    level_fields.append(parts[4])
                    desc_fields.append(parts[5])
                except ValueError:
                    pass
            elif len(parts) == 5:
                # Tolerate malformed rows that omit the descriptor field.
                try:
                    offsets.append(int(parts[1]))
                    var_fields.append(parts[3])
                    level_fields.append(parts[4])
                    desc_fields.append("")
                except ValueError:
                    pass

        matched: list[tuple[int, int | None, str, str, str, str, int]] = []

        for i, (off, var_field, level_field, desc_field) in enumerate(
            zip(offsets, var_fields, level_fields, desc_fields)
        ):
            end: int | None = offsets[i + 1] - 1 if i + 1 < len(offsets) else None
            for s_var, s_level, s_desc, colname in HRRR_SEARCHES:
                # Exact VAR+LEVEL match; descriptor matched only when specified.
                if var_field != s_var or level_field != s_level:
                    continue
                if s_desc is not None and desc_field != s_desc:
                    continue
                matched.append(
                    (off, end, colname, var_field, level_field, desc_field, i)
                )
                break

        by_col: dict[str, list[tuple[int, int | None, str, str, str, str, int]]] = {}
        for item in matched:
            by_col.setdefault(item[2], []).append(item)

        resolved: list[tuple[int, int | None, str, int]] = []
        for colname, items in by_col.items():
            if len(items) == 1:
                off, end, col, *_rest, idx = items[0]
                resolved.append((off, end, col, idx))
                continue
            if colname == "APCP":
                one_hour = [item for item in items if _is_one_hour_accum(item[5])]
                if len(one_hour) == 1:
                    off, end, col, *_rest, idx = one_hour[0]
                    resolved.append((off, end, col, idx))
                    continue
            _off, _end, _col, var_field, level_field, desc_field, _idx = items[-1]
            raise ValueError(
                f"_grib_source: duplicate .idx rows for column "
                f"'{colname}' -- matched "
                f"'{var_field}:{level_field}:{desc_field}'"
            )
        return [(off, end, col) for off, end, col, _idx in sorted(resolved, key=lambda x: x[3])]

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _fetch_idx(self, idx_url: str) -> str | None:
        """Download and return the raw .idx file text, or None on failure."""
        for attempt in range(3):
            try:
                resp = self._session.get(idx_url, timeout=30)
                if resp.status_code == 200:
                    return resp.text
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.5 * (attempt + 1))
        logger.warning("_grib_source: idx download failed after 3 attempts: {}", idx_url)
        return None

    def _fetch_range(
        self,
        grib_url: str,
        start: int,
        end: int | None,
    ) -> bytes | None:
        """Download bytes from grib_url using an HTTP Range Request.

        Parameters
        ----------
        start:
            First byte to fetch (inclusive).
        end:
            Last byte to fetch (inclusive).  When None an open-ended Range
            header (``bytes={start}-``) is used, which fetches to end-of-file.

        Returns
        -------
        Raw bytes on success (HTTP 206), or None on permanent failure.
        HTTP 200 responses are rejected (not a proper partial-content reply)
        with an ERROR log; other error codes trigger up to 3 retries.
        """
        range_header = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
        last_exc: Exception | None = None
        last_status: int | None = None

        for attempt in range(3):
            try:
                resp = self._session.get(
                    grib_url,
                    headers={"Range": range_header},
                    timeout=60,
                )
                if resp.status_code == 206:
                    # Validate Content-Range header before trusting the payload.
                    cr = resp.headers.get("Content-Range", "")
                    if not cr:
                        logger.error(
                            "_grib_source: 206 response missing Content-Range header; "
                            "url={} start={} end={}",
                            grib_url, start, end,
                        )
                        return None
                    # Parse "bytes {cr_start}-{cr_end}/{total}"
                    try:
                        cr_body = cr.split(" ", 1)[1]          # "{start}-{end}/{total}"
                        cr_range, _total = cr_body.split("/", 1)
                        cr_start_str, cr_end_str = cr_range.split("-", 1)
                        cr_start = int(cr_start_str)
                    except Exception:
                        logger.error(
                            "_grib_source: 206 response with unparseable "
                            "Content-Range={!r}; url={}",
                            cr, grib_url,
                        )
                        return None
                    if cr_start != start:
                        logger.error(
                            "_grib_source: 206 Content-Range start mismatch; "
                            "expected={} got={} Content-Range={!r} url={}",
                            start, cr_start, cr, grib_url,
                        )
                        return None
                    if end is not None:
                        expected = end - start + 1
                        if len(resp.content) != expected:
                            logger.error(
                                "_grib_source: 206 byte count mismatch; "
                                "expected={} got={} Content-Range={!r} url={}",
                                expected, len(resp.content), cr, grib_url,
                            )
                            return None
                    else:
                        # Open-ended Range: validate that Content-Range reports
                        # a consistent byte count (cr_end - cr_start + 1 == actual).
                        try:
                            cr_end = int(cr_end_str)
                            expected_open = cr_end - cr_start + 1
                        except (ValueError, NameError):
                            expected_open = None
                        if expected_open is not None and len(resp.content) != expected_open:
                            logger.error(
                                "_grib_source: 206 open-ended byte count mismatch; "
                                "Content-Range={!r} expected={} got={} url={}",
                                cr, expected_open, len(resp.content), grib_url,
                            )
                            return None
                    return resp.content
                if resp.status_code == 200:
                    # Server ignored Range header -- validate Content-Range header
                    # and byte count; always reject since 206 is required.
                    cr = resp.headers.get("Content-Range", "")
                    if end is not None and len(resp.content) != end - start + 1:
                        logger.error(
                            "_grib_source: 200 response with wrong byte count; "
                            "expected={} got={} Content-Range={!r} url={}",
                            end - start + 1, len(resp.content), cr, grib_url,
                        )
                    else:
                        logger.error(
                            "_grib_source: 200 response for Range request "
                            "(expected 206); Content-Range={!r} url={}",
                            cr, grib_url,
                        )
                    return None
                last_status = resp.status_code
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            time.sleep(1.0 * (attempt + 1))

        logger.warning(
            "_grib_source: _fetch_range failed after retries: "
            "url={} start={} end={} status={} exc={}",
            grib_url, start, end, last_status, last_exc,
        )
        return None

    # ------------------------------------------------------------------
    # GRIB2 decoding
    # ------------------------------------------------------------------

    def _find_grid_index(self, msg: object, station: "StationConfig") -> int:
        """Find and cache the flat grid index closest to station.

        Uses station_id as the cache key so multiple stations can be decoded
        from the same source instance without repeating lat/lon searches.
        """
        if _eccodes is None:
            raise ImportError("eccodes is required for GRIB decoding")
        cached = self._idx_cache.get(station.station_id)
        if cached is not None:
            return cached
        lats = _eccodes.codes_get_array(msg, "latitudes")
        lons = _eccodes.codes_get_array(msg, "longitudes")
        # GRIB2 stores lon in [0, 360]; convert station lon accordingly
        lon360 = station.lon % 360
        dist = (lats - station.lat) ** 2 + (lons - lon360) ** 2
        idx = int(np.argmin(dist))
        with self._idx_lock:
            self._idx_cache.setdefault(station.station_id, idx)
        return idx

    def _decode_grib_multi(
        self,
        content: bytes,
        colname: str,
        stations: "list[StationConfig]",
    ) -> dict[str, float]:
        """Decode one GRIB2 message and return {station_id: value} for all stations."""
        if _eccodes is None:
            raise ImportError("eccodes is required for GRIB decoding")
        msg = _eccodes.codes_new_from_message(content)
        if msg is None:
            return {s.station_id: float("nan") for s in stations}
        try:
            vals = _eccodes.codes_get_values(msg)
            result: dict[str, float] = {}
            for station in stations:
                try:
                    idx = self._find_grid_index(msg, station)
                    result[station.station_id] = float(vals[idx])
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "_grib_source: decode failed for station={} col={}",
                        station.station_id, colname,
                    )
                    result[station.station_id] = float("nan")
            return result
        finally:
            try:
                _eccodes.codes_release(msg)
            except Exception:  # noqa: BLE001
                pass

    # kept for single-station callers and backward compat
    def _decode_grib(
        self, content: bytes, colname: str, station: "StationConfig"
    ) -> float:
        """Decode a single GRIB2 message and return the grid-point value."""
        return self._decode_grib_multi(content, colname, [station]).get(
            station.station_id, float("nan")
        )

    # ------------------------------------------------------------------
    # Public fetch  -- returns dict[station_id -> DataFrame]
    # ------------------------------------------------------------------

    def fetch(
        self,
        init_time: pd.Timestamp,
        fxx_range: range = range(0, 49),
        station: "StationConfig | None" = None,
    ) -> "dict[str, pd.DataFrame]":
        """Fetch one init cycle for all configured STATIONS.

        Parameters
        ----------
        init_time:
            UTC timestamp of the HRRR model initialisation hour.
        fxx_range:
            Forecast hours to fetch.  For MAJOR init_times this should be
            range(0, 49).  For non-MAJOR callers this is ignored; the method
            fetches fxx 0-18 and splices with the nearest prior MAJOR run.
        station:
            Optional single-station override.  When provided, only that
            station's data is fetched (backwards-compat for DataManager).
            When None, all stations in ``config.stations.STATIONS`` are fetched.

        Returns
        -------
        dict mapping station_id -> DataFrame with columns
        init_time, fxx, valid_time, <HRRR var columns>.
        """
        from config.stations import STATIONS

        station_list = [station] if station is not None else list(STATIONS.values())

        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )

        if _is_major(init_utc):
            return self._fetch_all_stations(init_utc, fxx_range, station_list)

        requested_fxx = sorted({int(fxx) for fxx in fxx_range})
        if requested_fxx and max(requested_fxx) <= 18 and min(requested_fxx) >= 0:
            return self._fetch_all_stations(init_utc, fxx_range, station_list)

        # Non-MAJOR path: fetch fxx 0-18, then splice with nearest MAJOR run.
        logger.debug(
            "_grib_source: non-MAJOR init={}, fetching fxx 0-18 + nearest "
            "MAJOR splice + bounded tail extrapolation",
            init_utc,
        )
        non_major_dict = self._fetch_all_stations(init_utc, _NON_MAJOR_FXX, station_list)

        nearest_major_init = nearest_prior_major(init_utc)

        try:
            nearest_major_dict = self._fetch_all_stations(
                nearest_major_init, range(0, 49), station_list
            )
        except Exception:
            logger.warning(
                "_grib_source: nearest MAJOR fetch failed for init={}; "
                "returning non-MAJOR data only",
                nearest_major_init,
            )
            return non_major_dict

        result: dict[str, pd.DataFrame] = {}
        for sid in non_major_dict:
            result[sid] = self._hrrr_fill.fill(
                non_major_dict[sid],
                nearest_major_dict.get(sid, pd.DataFrame()),
            )
        return result

    def _fetch_all_stations(
        self,
        init_utc: pd.Timestamp,
        fxx_range: range,
        stations: "list[StationConfig]",
    ) -> "dict[str, pd.DataFrame]":
        """Fetch one init cycle for the given list of stations.

        Raises
        ------
        RuntimeError
            When every fxx step fails to download an idx file.  This signals
            DataManager to fall back to the next source in the chain.
        """
        # station_rows[station_id] = list of row dicts
        station_rows: dict[str, list[dict]] = {s.station_id: [] for s in stations}
        idx_success = 0

        for fxx in fxx_range:
            row_per_station, idx_ok = self._fetch_fxx_multi(init_utc, fxx, stations)
            if idx_ok:
                idx_success += 1
            for sid, row in row_per_station.items():
                station_rows[sid].append(row)

        if idx_success == 0:
            raise RuntimeError(
                f"_grib_source: all idx fetches failed for init={init_utc} "
                f"source={type(self).__name__}"
            )

        all_cols = list(GRIB_TO_COL.values())
        result: dict[str, pd.DataFrame] = {}
        for sid, rows in station_rows.items():
            df = pd.DataFrame(rows)
            for col in all_cols:
                if col not in df.columns:
                    df[col] = float("nan")
            df["is_extrapolated"] = False
            result[sid] = df

        return result

    def _fetch_fxx_multi(
        self,
        init_utc: pd.Timestamp,
        fxx: int,
        stations: "list[StationConfig]",
    ) -> "tuple[dict[str, dict], bool]":
        """Fetch one fxx step and return ({station_id: row_dict}, idx_ok).

        idx_ok is True when the .idx file was successfully downloaded for this
        fxx step; False means the source was unreachable for this step.
        """
        date_str = init_utc.strftime("%Y%m%d")
        init_hh  = init_utc.strftime("%H")
        fxx_str  = f"{fxx:02d}"

        base_row = {
            "init_time":  str(init_utc),
            "fxx":        fxx,
            "valid_time": str(init_utc + pd.Timedelta(hours=fxx)),
        }
        nan_rows = {
            sid: {**base_row, **{col: float("nan") for col in GRIB_TO_COL.values()}}
            for sid in [s.station_id for s in stations]
        }

        grib_url = self._build_url(date_str, init_hh, fxx_str)
        idx_url  = grib_url + ".idx"

        idx_text = self._fetch_idx(idx_url)
        if idx_text is None:
            logger.debug(
                "_grib_source: idx unavailable init={} fxx={}", init_utc, fxx
            )
            return nan_rows, False

        targets = self._parse_idx(idx_text)
        if not targets:
            logger.debug(
                "_grib_source: no matching variables in idx init={} fxx={}",
                init_utc, fxx,
            )
            return nan_rows, False

        # Parallel range downloads
        downloaded: dict[str, bytes | None] = {}
        with ThreadPoolExecutor(max_workers=_RANGE_THREADS) as pool:
            future_map = {
                pool.submit(self._fetch_range, grib_url, s, e): cn
                for s, e, cn in targets
            }
            for fut in as_completed(future_map):
                cn = future_map[fut]
                try:
                    downloaded[cn] = fut.result()
                except Exception:  # noqa: BLE001
                    downloaded[cn] = None

        # Decode each message for all stations in one pass
        station_data: dict[str, dict] = {
            sid: {**base_row} for sid in [s.station_id for s in stations]
        }
        for colname, content in downloaded.items():
            if content is None:
                for sid in station_data:
                    station_data[sid][colname] = float("nan")
                continue
            try:
                values = self._decode_grib_multi(content, colname, stations)
                for sid, val in values.items():
                    station_data[sid][colname] = val
            except Exception:  # noqa: BLE001
                logger.warning(
                    "_grib_source: decode error for col={} init={} fxx={}",
                    colname, init_utc, fxx,
                )
                for sid in station_data:
                    station_data[sid][colname] = float("nan")

        # Fill any still-missing columns
        for sid in station_data:
            for col in GRIB_TO_COL.values():
                if col not in station_data[sid]:
                    station_data[sid][col] = float("nan")

        return station_data, True
