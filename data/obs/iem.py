"""IEM (Iowa Environmental Mesonet) observation source.

Primary observation data source.  Downloads ASOS observations via the
IEM CSV API endpoint, maps raw fields to the project's clean schema, and
returns a tidy DataFrame ready for Cleaner.clean_iem().

Endpoint:
    https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
    ?station=LGA&data=all&tz=Etc/UTC&format=comma&latlon=no
     &year1=...&month1=...&day1=...&year2=...&month2=...&day2=...
"""
from __future__ import annotations

import io
import re
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
from loguru import logger

from data.obs.base import BaseObsSource

if TYPE_CHECKING:
    from config.stations import StationConfig

_IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Sky-cover text code -> integer cloud fraction (percent)
_SKY_MAP: dict[str, float] = {
    "CLR": 0.0,
    "SKC": 0.0,
    "NSC": 0.0,
    "FEW": 15.0,
    "SCT": 40.0,
    "BKN": 70.0,
    "OVC": 100.0,
    "VV": 100.0,
}

# Regex for METAR T-group: T(sign_t)(ddd)(sign_d)(ddd)
# sign: 0=positive, 1=negative; ddd = tenths-of-degree
_RE_T_GROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")


def _t_group_to_c(sign: str, digits: str) -> float:
    val = int(digits) / 10.0
    return -val if sign == "1" else val


def _f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


class IemSource(BaseObsSource):
    """Fetch ASOS observations from the IEM CSV API.

    The IEM endpoint returns a CSV with one minute-resolution row per
    observation.  This method downloads the full window, maps fields to
    the clean schema, and returns the raw (pre-Cleaner) DataFrame.

    Field mapping
    -------------
    valid          -> valid_time  (raw METAR time, converted to UTC-aware;
                                    Cleaner stores nearest whole hour)
    tmpf           -> raw_tmpc_tenths  (F->C; T-group from metar if present)
    dwpf           -> raw_dwpc_tenths  (F->C; T-group from metar if present)
    sknt           -> raw_wind_spd_kt
    gust           -> raw_wind_gust_kt
    drct + sknt    -> obs_U, obs_V
    vsby           -> raw_vis_sm
    alti           -> raw_alti_inhg
    skyc1  (text)  -> skyc1  (numeric via SKY_MAP)
    skyl1          -> skyl1  (feet, NaN -> 30000)
    p01i           -> p01i
    wxcodes        -> is_rain, is_snow  (binarised)
    """

    def __init__(
        self,
        timeout: int = 60,
        fallback: "BaseObsSource | None" = None,
    ) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "PolyWeatherBot/1.0"})
        self._timeout = timeout
        self._fallback = fallback

    def fetch(
        self,
        station: "StationConfig",
        start: pd.Timestamp,
        end: pd.Timestamp,
        now_utc: "pd.Timestamp | None" = None,
    ) -> pd.DataFrame:
        """Fetch ASOS observations.

        Parameters
        ----------
        now_utc:
            When provided (tick mode), start/end are ignored.  Instead only
            the H-1 :51 reading is fetched as the representative observation
            for the nearest whole hour: the narrow window
            [target_dt - 30s, target_dt + 30s] where
            target_dt = (now_utc - 1h).replace(minute=51).
        """
        if now_utc is not None:
            now_utc_aware = (
                now_utc.tz_localize("UTC") if now_utc.tzinfo is None
                else now_utc.tz_convert("UTC")
            )
            hour_back = now_utc_aware - pd.Timedelta(hours=1)
            target_dt = hour_back.replace(minute=51, second=0, microsecond=0)
            start = target_dt - pd.Timedelta(seconds=30)
            end   = target_dt + pd.Timedelta(seconds=30)

        start_utc = (
            start.tz_localize("UTC") if start.tzinfo is None
            else start.tz_convert("UTC")
        )
        end_utc = (
            end.tz_localize("UTC") if end.tzinfo is None
            else end.tz_convert("UTC")
        )
        request_end_utc = end_utc
        if now_utc is not None:
            request_end_utc = end_utc.ceil("min")

        # IEM uses the 4-letter ID but ASOs API takes 3 or 4 letters;
        # strip leading 'K' for ICAO->FAA conversion if needed
        iem_station = (
            station.station_id[1:]
            if station.station_id.startswith("K") and len(station.station_id) == 4
            else station.station_id
        )

        params = {
            "station": iem_station,
            "data": "all",
            "tz": "Etc/UTC",
            "format": "comma",
            "latlon": "no",
            "year1":   str(start_utc.year),
            "month1":  str(start_utc.month),
            "day1":    str(start_utc.day),
            "hour1":   str(start_utc.hour),
            "minute1": str(start_utc.minute),
            "year2":   str(request_end_utc.year),
            "month2":  str(request_end_utc.month),
            "day2":    str(request_end_utc.day),
            "hour2":   str(request_end_utc.hour),
            "minute2": str(request_end_utc.minute),
        }

        csv_text = self._download_csv(params)
        if not csv_text:
            if self._fallback is not None:
                return self._fallback.fetch(station, start, end, now_utc=now_utc)
            return self._empty_df()

        df = self._parse_csv(csv_text)
        if now_utc is not None and not df.empty:
            mask = (df["valid_time"] >= start_utc) & (df["valid_time"] <= end_utc)
            df = df[mask].reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_csv(self, params: dict) -> str | None:
        for attempt in range(3):
            try:
                resp = self._session.get(_IEM_URL, params=params,
                                         timeout=self._timeout)
                if resp.status_code == 200:
                    return resp.text
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2.0 ** attempt)
        return None

    def _parse_csv(self, csv_text: str) -> pd.DataFrame:
        """Parse the raw IEM CSV text into a clean-schema DataFrame."""
        # IEM prepends comment lines starting with '#'; skip them
        lines = [
            ln for ln in csv_text.splitlines()
            if not ln.startswith("#") and ln.strip()
        ]
        if not lines:
            return self._empty_df()

        raw = pd.read_csv(io.StringIO("\n".join(lines)), low_memory=False)

        # Normalise column names: strip whitespace
        raw.columns = raw.columns.str.strip()

        if "valid" not in raw.columns:
            return self._empty_df()

        out = pd.DataFrame()

        # valid_time
        out["valid_time"] = pd.to_datetime(raw["valid"], utc=True, errors="coerce")
        out = out.dropna(subset=["valid_time"]).copy()
        raw = raw.loc[out.index].copy()

        # Temperatures: prefer T-group precision when metar column exists
        tmp_c, tmp_source = self._extract_temp_c(raw, "tmpf", "metar", tgroup_idx=0)
        dpt_c, dpt_source = self._extract_temp_c(raw, "dwpf", "metar", tgroup_idx=1)
        out["raw_tmpc_tenths"] = tmp_c.values
        out["raw_dwpc_tenths"] = dpt_c.values
        out["_raw_tmpc_source"] = tmp_source.values
        out["_raw_dwpc_source"] = dpt_source.values

        def _col(name: str) -> pd.Series:
            if name in raw.columns:
                return pd.to_numeric(raw[name], errors="coerce")
            return pd.Series(np.nan, index=raw.index)

        # Wind
        spd  = _col("sknt")
        gust = _col("gust")
        drct = _col("drct")
        out["raw_wind_spd_kt"]  = spd.values
        out["raw_wind_gust_kt"] = gust.values

        # Wind U/V decomposition. Preserve missing/variable direction as NaN;
        # Cleaner.clean_iem() performs the common time-series fill.
        valid_wind = spd.notna() & drct.notna()
        dir_rad = np.deg2rad(drct.where(valid_wind).values)
        spd_vals = spd.where(valid_wind).values
        out["obs_U"] = -spd_vals * np.sin(dir_rad)
        out["obs_V"] = -spd_vals * np.cos(dir_rad)

        # Visibility and pressure
        out["raw_vis_sm"]    = _col("vsby").values
        out["raw_alti_inhg"] = _col("alti").values

        # Sky cover: text -> numeric
        skyc1_txt = raw.get("skyc1", pd.Series([""] * len(raw), dtype=str))
        out["skyc1"] = (
            skyc1_txt.str.strip().map(_SKY_MAP).values
        )
        out["skyl1"] = _col("skyl1").fillna(30000.0).values

        # Precipitation
        out["p01i"] = _col("p01i").values

        # Weather codes -> binary flags
        wxcodes = raw.get("wxcodes", pd.Series([""] * len(raw), dtype=str))
        wxcodes = wxcodes.fillna("").str.strip()
        out["is_rain"] = wxcodes.str.contains(r"\bRA\b", regex=True).astype(float).values
        out["is_snow"] = wxcodes.str.contains(r"\bSN\b", regex=True).astype(float).values

        out = out.reset_index(drop=True)
        return out

    def _extract_temp_c(
        self,
        raw: pd.DataFrame,
        f_col: str,
        metar_col: str,
        tgroup_idx: int,
    ) -> tuple[pd.Series, pd.Series]:
        """Extract temperature (Celsius) preferring T-group tenths precision.

        Returns (temperature_c, source), where source is one of:
        ``tgroup`` for METAR T-group, ``fallback`` for tmpf/dwpf fallback, or
        ``missing`` when neither source is available.  Logs a WARNING when any
        row had to fall back from T-group to tmpf/dwpf.
        """
        n = len(raw)
        result = np.full(n, np.nan)
        source = np.full(n, "missing", dtype=object)

        # Try T-group from metar first
        if metar_col in raw.columns:
            for i, metar in enumerate(raw[metar_col].fillna("").astype(str)):
                m = _RE_T_GROUP.search(metar)
                if m:
                    groups = m.groups()  # (t_sign, t_ddd, d_sign, d_ddd)
                    if tgroup_idx == 0:
                        result[i] = _t_group_to_c(groups[0], groups[1])
                    else:
                        result[i] = _t_group_to_c(groups[2], groups[3])
                    source[i] = "tgroup"

        # Fall back to tmpf/dwpf for rows where T-group was not found
        fallback_count = 0
        if f_col in raw.columns:
            f_vals = pd.to_numeric(raw[f_col], errors="coerce").values
            missing = np.isnan(result)
            needed_fallback = missing & ~np.isnan(f_vals)
            fallback_count = int(needed_fallback.sum())
            result[missing] = np.where(
                np.isnan(f_vals[missing]), np.nan, _f_to_c(f_vals[missing])
            )
            source[needed_fallback] = "fallback"

        if fallback_count > 0:
            col_label = "temp" if tgroup_idx == 0 else "dewpoint"
            logger.warning(
                "iem: T-group precision unavailable for {} rows of {}; "
                "fell back to integer-degree {}",
                fallback_count, col_label, f_col,
            )

        return pd.Series(result, dtype=float), pd.Series(source, dtype="object")

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        cols = [
            "valid_time", "raw_tmpc_tenths", "raw_dwpc_tenths",
            "raw_wind_spd_kt", "raw_wind_gust_kt", "obs_U", "obs_V",
            "raw_vis_sm", "raw_alti_inhg", "skyc1", "skyl1",
            "p01i", "is_rain", "is_snow",
            "_raw_tmpc_source", "_raw_dwpc_source",
        ]
        return pd.DataFrame(columns=cols)
