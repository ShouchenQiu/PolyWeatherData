"""NOAA tgftp observation source (fallback when IEM is unavailable).

Fetches the latest METAR directly from:
    https://tgftp.nws.noaa.gov/data/observations/metar/stations/{STATION}.TXT

The file contains exactly one observation (the most recent hourly METAR).
HEAD is used first to check Last-Modified; GET is only issued when needed.

Because the file holds only the *current* observation, fetch() returns an
empty DataFrame when the METAR timestamp falls outside [start, end].
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
from loguru import logger

from data.obs.base import BaseObsSource

if TYPE_CHECKING:
    from config.stations import StationConfig

_TGFTP_URL_TPL = (
    "https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT"
)

# Regex for METAR body: station ddhhmm[Z] [AUTO] wind vis sky temp/dew Axxx
_RE_WIND   = re.compile(
    r"\b(?P<dir>VRB|\d{3})(?P<spd>\d{2,3})(?:G(?P<gust>\d{2,3}))?KT\b"
)
_RE_VIS    = re.compile(
    r"(?:(\d+)\s+)?(\d+)/(\d+)SM|(\d+)SM"
)
_RE_SKY    = re.compile(
    r"\b(CLR|SKC|NSC|CAVOK|FEW|SCT|BKN|OVC|VV)(\d{3})?\b"
)
_RE_TEMP   = re.compile(r"\b(M?\d{1,2})/(M?\d{0,2})\b")
_RE_ALTI   = re.compile(r"\bA(\d{4})\b")
_RE_T_GRP  = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
_RE_DDHHMM = re.compile(r"\b(\d{2})(\d{2})(\d{2})Z\b")

_SKY_MAP: dict[str, float] = {
    "CLR": 0.0, "SKC": 0.0, "NSC": 0.0,
    "FEW": 15.0, "SCT": 40.0, "BKN": 70.0,
    "OVC": 100.0, "VV": 100.0,
}


def _metar_temp(s: str) -> float:
    """Parse single METAR temperature token like '14' or 'M05'."""
    if not s:
        return float("nan")
    neg = s.startswith("M")
    try:
        return (-1 if neg else 1) * float(s.lstrip("M"))
    except ValueError:
        return float("nan")


def _parse_vis(m: re.Match) -> float:
    """Convert a RE_VIS match to SM float."""
    if m.group(4) is not None:         # e.g. "10SM"
        return float(m.group(4))
    whole  = float(m.group(1)) if m.group(1) else 0.0
    num    = float(m.group(2))
    denom  = float(m.group(3))
    return whole + num / denom


class NoaaFileSource(BaseObsSource):
    """Fetch the latest METAR from the NOAA tgftp server.

    This source returns at most one row (the most recent observation).
    It is suited as a near-real-time fallback; it does not provide
    historical data.

    Usage
    -----
    - HEAD the URL to check Last-Modified without downloading the body.
    - GET the URL to retrieve the METAR text.
    - Parse the METAR into the clean schema.
    - Return empty DataFrame if the observation timestamp is outside
      the requested [start, end] window.
    """

    def __init__(self, timeout: int = 20) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "PolyWeatherBot/1.0"})
        self._timeout = timeout

    def last_modified(self, station_id: str) -> str | None:
        """Return the Last-Modified header for the station file, or None."""
        url = _TGFTP_URL_TPL.format(station=station_id)
        try:
            resp = self._session.head(url, timeout=self._timeout)
            return resp.headers.get("Last-Modified")
        except Exception:  # noqa: BLE001
            return None

    def fetch(
        self,
        station: "StationConfig",
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        url = _TGFTP_URL_TPL.format(station=station.station_id)
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            metar_txt = resp.text.strip()
        except Exception:  # noqa: BLE001
            return self._empty_df()

        row = self._parse_txt(metar_txt)
        if row is None:
            return self._empty_df()

        # Filter to requested window
        vt = row.get("valid_time")
        if vt is None or pd.isna(vt):
            return self._empty_df()

        start_utc = (
            start.tz_localize("UTC") if start.tzinfo is None
            else start.tz_convert("UTC")
        )
        end_utc = (
            end.tz_localize("UTC") if end.tzinfo is None
            else end.tz_convert("UTC")
        )
        if not (start_utc <= vt <= end_utc):
            return self._empty_df()

        df = pd.DataFrame([row])
        return df

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_txt(self, txt: str) -> dict | None:
        """Parse a NOAA tgftp TXT file and return a single row dict.

        The file format is:
            Line 1: YYYY/MM/DD HH:MM   (observation date, UTC)
            Line 2: METAR string        (full METAR with remarks)

        An incomplete file (only one non-empty line) is still attempted; the
        valid_time will be NaN so the caller will discard it via the time-
        window check.  This avoids silently swallowing partial responses.
        """
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            return None

        date_line  = lines[0]
        metar_line = lines[1] if len(lines) >= 2 else ""

        # Parse observation timestamp from date line
        try:
            obs_date = pd.Timestamp(date_line.replace("/", "-"), tz="UTC")
        except Exception:  # noqa: BLE001
            obs_date = None

        # Refine minute from METAR ddhhmm for precision
        m_ts = _RE_DDHHMM.search(metar_line)
        if m_ts and obs_date is not None:
            hh = int(m_ts.group(2))
            mm = int(m_ts.group(3))
            valid_time = obs_date.replace(hour=hh, minute=mm, second=0,
                                          microsecond=0)
        else:
            valid_time = obs_date

        row: dict = {"valid_time": valid_time}

        # Temperature / dewpoint
        t_m = _RE_T_GRP.search(metar_line)
        if t_m:
            g = t_m.groups()
            tmpc = (-1 if g[0] == "1" else 1) * int(g[1]) / 10.0
            dwpc = (-1 if g[2] == "1" else 1) * int(g[3]) / 10.0
        else:
            body_t = _RE_TEMP.search(metar_line)
            tmpc = _metar_temp(body_t.group(1)) if body_t else float("nan")
            dwpc = _metar_temp(body_t.group(2)) if body_t else float("nan")

        if np.isnan(tmpc) and metar_line:
            logger.warning(
                "noaa_file: temperature unavailable in METAR: {}",
                metar_line[:60],
            )

        row["raw_tmpc_tenths"] = tmpc
        row["raw_dwpc_tenths"] = dwpc

        # Wind
        w_m = _RE_WIND.search(metar_line)
        if w_m:
            dir_s = w_m.group("dir")
            spd   = float(w_m.group("spd"))
            gust  = float(w_m.group("gust")) if w_m.group("gust") else spd
            drct  = 0.0 if dir_s == "VRB" else float(dir_s)
        else:
            spd = gust = drct = float("nan")

        row["raw_wind_spd_kt"]  = spd
        row["raw_wind_gust_kt"] = gust

        dir_rad = np.deg2rad(0.0 if np.isnan(drct) else drct)
        spd_v   = 0.0 if np.isnan(spd) else spd
        row["obs_U"] = -spd_v * float(np.sin(dir_rad))
        row["obs_V"] = -spd_v * float(np.cos(dir_rad))

        # Visibility
        v_m = _RE_VIS.search(metar_line)
        row["raw_vis_sm"] = _parse_vis(v_m) if v_m else float("nan")

        # Altimeter
        a_m = _RE_ALTI.search(metar_line)
        row["raw_alti_inhg"] = int(a_m.group(1)) / 100.0 if a_m else float("nan")

        # Sky cover (first layer only)
        sky_m = _RE_SKY.search(metar_line)
        if sky_m:
            row["skyc1"] = _SKY_MAP.get(sky_m.group(1), float("nan"))
            row["skyl1"] = (
                float(sky_m.group(2)) * 100.0
                if sky_m.group(2) else 30000.0
            )
        else:
            row["skyc1"] = 0.0
            row["skyl1"] = 30000.0

        # Precipitation (not in current METAR; default 0)
        row["p01i"] = 0.0

        # Weather phenomena -> binary
        row["is_rain"] = float(
            bool(re.search(r"\bRA\b", metar_line))
        )
        row["is_snow"] = float(
            bool(re.search(r"\bSN\b", metar_line))
        )

        return row

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        cols = [
            "valid_time", "raw_tmpc_tenths", "raw_dwpc_tenths",
            "raw_wind_spd_kt", "raw_wind_gust_kt", "obs_U", "obs_V",
            "raw_vis_sm", "raw_alti_inhg", "skyc1", "skyl1",
            "p01i", "is_rain", "is_snow",
        ]
        return pd.DataFrame(columns=cols)
