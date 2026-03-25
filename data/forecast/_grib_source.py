"""Shared GRIB2 (idx + range-request + eccodes) logic for HRRR data sources.

Both HrrrS3Source and HrrrNomadsSource use the same .idx parsing, HTTP Range
Request download, and eccodes decoding steps; only the URL base differs.
This module provides the shared BaseGribSource base class.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data.forecast.base import BaseForecastSource
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


def _make_session() -> requests.Session:
    sess = requests.Session()
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


class BaseGribSource(BaseForecastSource):
    """Shared base for sources that download GRIB2 via .idx + Range Requests.

    Subclasses must implement ``_build_url`` to return the GRIB2 file URL for
    a given init date, init hour, and forecast step.
    """

    def __init__(self) -> None:
        self._session: requests.Session = _make_session()
        self._klga_idx: int | None = None
        self._idx_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Abstract helpers
    # ------------------------------------------------------------------

    def _build_url(self, date_str: str, init_hh: str, fxx_str: str) -> str:
        """Return full GRIB2 URL for the given parameters."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # .idx parsing
    # ------------------------------------------------------------------

    def _parse_idx(self, idx_text: str) -> list[tuple[int, int, str]]:
        """Parse a .idx file and return [(start_byte, end_byte, col_name), ...].

        The .idx format is:
            msgnum:byte_offset:d=YYYYMMDDHHZ:VAR:LEVEL:FCST:...

        The end byte of message N is start_byte(N+1) - 1.
        For the last message we add 5_000_000 as a conservative upper bound.
        """
        offsets: list[int] = []
        line_texts: list[str] = []

        for ln in idx_text.strip().splitlines():
            parts = ln.split(":")
            if len(parts) >= 5:
                try:
                    offsets.append(int(parts[1]))
                    line_texts.append(":".join(parts[3:]))
                except ValueError:
                    pass

        targets: list[tuple[int, int, str]] = []
        for i, (off, txt) in enumerate(zip(offsets, line_texts)):
            end = offsets[i + 1] - 1 if i + 1 < len(offsets) else off + 5_000_000
            for search, colname in HRRR_SEARCHES:
                if search in txt:
                    targets.append((off, end, colname))
                    break
        return targets

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
        return None

    def _fetch_range(
        self,
        grib_url: str,
        start: int,
        end: int,
    ) -> bytes | None:
        """Download bytes [start, end] from grib_url using an HTTP Range Request."""
        for attempt in range(3):
            try:
                resp = self._session.get(
                    grib_url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=60,
                )
                if resp.status_code in (200, 206):
                    return resp.content
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0 * (attempt + 1))
        return None

    # ------------------------------------------------------------------
    # GRIB2 decoding
    # ------------------------------------------------------------------

    def _find_grid_index(self, msg: object, station: "StationConfig") -> int:
        """Find and cache the flat grid index closest to the station."""
        if _eccodes is None:
            raise ImportError("eccodes is required for GRIB decoding")
        if self._klga_idx is not None:
            return self._klga_idx
        lats = _eccodes.codes_get_array(msg, "latitudes")
        lons = _eccodes.codes_get_array(msg, "longitudes")
        # GRIB2 stores lon in [0, 360]; convert station lon accordingly
        lon360 = station.lon % 360
        dist = (lats - station.lat) ** 2 + (lons - lon360) ** 2
        idx = int(np.argmin(dist))
        with self._idx_lock:
            if self._klga_idx is None:
                self._klga_idx = idx
        return idx

    def _decode_grib(
        self, content: bytes, colname: str, station: "StationConfig"
    ) -> float:
        """Decode a single GRIB2 message and return the grid-point value."""
        if _eccodes is None:
            raise ImportError("eccodes is required for GRIB decoding")
        msg = _eccodes.codes_new_from_message(content)
        if msg is None:
            return float("nan")
        try:
            idx = self._find_grid_index(msg, station)
            vals = _eccodes.codes_get_values(msg)
            return float(vals[idx])
        finally:
            try:
                _eccodes.codes_release(msg)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Public fetch
    # ------------------------------------------------------------------

    def fetch(
        self,
        init_time: pd.Timestamp,
        fxx_range: range,
        station: "StationConfig",
    ) -> pd.DataFrame:
        """Fetch one init cycle across fxx_range and return a tidy DataFrame."""
        rows = []
        for fxx in fxx_range:
            rows.append(self._fetch_fxx(init_time, fxx, station))
        df = pd.DataFrame(rows)
        # Ensure all expected columns are present (fill missing with NaN)
        all_cols = list(GRIB_TO_COL.values())
        for col in all_cols:
            if col not in df.columns:
                df[col] = float("nan")
        return df

    def _fetch_fxx(
        self,
        init_time: pd.Timestamp,
        fxx: int,
        station: "StationConfig",
    ) -> dict:
        init_utc = (
            init_time.tz_localize("UTC") if init_time.tzinfo is None
            else init_time.tz_convert("UTC")
        )
        date_str = init_utc.strftime("%Y%m%d")
        init_hh = init_utc.strftime("%H")
        fxx_str = f"{fxx:02d}"

        grib_url = self._build_url(date_str, init_hh, fxx_str)
        idx_url = grib_url + ".idx"

        row: dict = {
            "init_time": str(init_utc),
            "fxx": fxx,
            "valid_time": str(init_utc + pd.Timedelta(hours=fxx)),
        }

        idx_text = self._fetch_idx(idx_url)
        if idx_text is None:
            for col in GRIB_TO_COL.values():
                row[col] = float("nan")
            return row

        targets = self._parse_idx(idx_text)
        if not targets:
            for col in GRIB_TO_COL.values():
                row[col] = float("nan")
            return row

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

        # Decode each message
        for colname, content in downloaded.items():
            if content is None:
                row[colname] = float("nan")
            else:
                try:
                    row[colname] = self._decode_grib(content, colname, station)
                except Exception:  # noqa: BLE001
                    row[colname] = float("nan")

        # Fill any still-missing columns
        for col in GRIB_TO_COL.values():
            if col not in row:
                row[col] = float("nan")
        return row
