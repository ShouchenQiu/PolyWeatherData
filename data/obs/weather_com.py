"""Weather.com historical observation truth client with daily parquet cache."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config.stations import STATIONS

DEFAULT_WEATHER_COM_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
_ALIGNMENT_VERSION = "iem_representative_hour_v3_with_raw_daily_high"
_COMPATIBLE_HOURLY_ALIGNMENT_VERSIONS = {
    "iem_representative_hour_v2",
    _ALIGNMENT_VERSION,
}


class WeatherComApiKeyMissing(RuntimeError):
    """Raised when Weather.com data is uncached and no API key is configured."""


class WeatherComFetchError(RuntimeError):
    """Raised when Weather.com cannot return a usable response."""


def local_day_hour_coverage(
    frame: pd.DataFrame,
    *,
    local_date: str | pd.Timestamp,
    timezone: str,
) -> tuple[int, int]:
    """Return observed and expected hourly slots for one station-local day."""
    day = pd.Timestamp(local_date).date().isoformat()
    start_naive = pd.Timestamp(day)
    end_naive = start_naive + pd.Timedelta(days=1)
    start_utc = start_naive.tz_localize(timezone).tz_convert("UTC")
    end_utc = end_naive.tz_localize(timezone).tz_convert("UTC")
    expected_hours = int((end_utc - start_utc) / pd.Timedelta(hours=1))
    if "valid_time" not in frame.columns:
        return 0, expected_hours
    valid_time = pd.to_datetime(frame["valid_time"], utc=True, errors="coerce")
    in_day = valid_time.notna() & (valid_time >= start_utc) & (valid_time < end_utc)
    if "valid_hour" in frame.columns:
        hourly_slots = pd.to_datetime(
            frame.loc[in_day, "valid_hour"], utc=True, errors="coerce"
        )
    else:
        hourly_slots = valid_time.loc[in_day].dt.round("h")
    expected_slots = pd.date_range(
        start_utc + pd.Timedelta(hours=1),
        end_utc,
        freq="h",
    )
    observed_slots = set(hourly_slots.dropna().tolist())
    return len(observed_slots.intersection(expected_slots)), expected_hours


class WeatherComTruthClient:
    """Fetch KLGA/NYC historical observations from Weather.com.

    The cache is keyed by station_id and local date because the Weather.com
    endpoint accepts date strings in the requested location's local calendar.
    """

    _BASE_URL = (
        "https://api.weather.com/v1/location/"
        "{location}/observations/historical.json"
    )
    _LOCATION_CODES = {"KLGA": "KLGA:9:US"}

    def __init__(
        self,
        cache_root: Path | str,
        api_key: str | None = None,
        session: Any | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._cache_root = Path(cache_root)
        self._api_key = (
            api_key
            if api_key is not None
            else os.environ.get("WEATHER_COM_API_KEY") or DEFAULT_WEATHER_COM_API_KEY
        )
        self._session = session or requests.Session()
        self._timeout = float(timeout)

    def fetch_date(self, station_id: str, local_date: str | pd.Timestamp) -> pd.DataFrame:
        """Return normalized truth rows for one station-local date."""
        day = pd.Timestamp(local_date).date().isoformat()
        timezone = STATIONS[station_id].timezone
        path = self._cache_path(station_id, day)
        if path.exists():
            cached = self._read_cache(path)
            observed, expected = local_day_hour_coverage(
                cached,
                local_date=day,
                timezone=timezone,
            )
            if (
                observed == expected
                and (
                    self._cache_is_resolution_truth_current(cached)
                    or not self._api_key
                )
            ):
                return self._public_columns(cached)

        if not self._api_key:
            raise WeatherComApiKeyMissing(
                "WEATHER_COM_API_KEY is required to fetch uncached Weather.com truth data"
            )

        location = self.location_code(station_id)
        date_param = day.replace("-", "")
        next_date_param = (
            pd.Timestamp(day) + pd.Timedelta(days=1)
        ).strftime("%Y%m%d")
        url = self._BASE_URL.format(location=location)
        params = {
            "apiKey": self._api_key,
            "units": "e",
            "startDate": date_param,
            "endDate": next_date_param,
        }
        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise WeatherComFetchError(
                f"Weather.com fetch failed for station={station_id} date={day}: {exc}"
            ) from exc

        raw_frame = self._observation_frame(station_id, payload)
        raw_frame = self._filter_local_day(raw_frame, day=day, timezone=timezone)
        frame = self._select_representative_hours(raw_frame)
        raw_temps = pd.to_numeric(
            raw_frame["weather_com_temp_f"], errors="coerce"
        ).dropna()
        if raw_temps.empty:
            raise WeatherComFetchError(
                "Weather.com returned no usable temperatures "
                f"station={station_id} date={day}"
            )
        frame["weather_com_raw_daily_high_f"] = float(raw_temps.max())
        frame["_weather_com_alignment_version"] = _ALIGNMENT_VERSION
        observed, expected = local_day_hour_coverage(
            frame,
            local_date=day,
            timezone=timezone,
        )
        if observed != expected:
            raise WeatherComFetchError(
                "Weather.com returned an incomplete local day "
                f"station={station_id} date={day} "
                f"observed_hours={observed} expected_hours={expected}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return self._public_columns(frame)

    def fetch_dates(
        self,
        station_id: str,
        local_dates: list[str] | tuple[str, ...],
    ) -> pd.DataFrame:
        parts = [self.fetch_date(station_id, day) for day in local_dates]
        if not parts:
            return self._empty_frame()
        return pd.concat(parts, ignore_index=True)

    def read_cached_date(
        self,
        station_id: str,
        local_date: str | pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Read one current cached local date without performing network IO."""
        day = pd.Timestamp(local_date).date().isoformat()
        path = self._cache_path(station_id, day)
        if not path.exists():
            return None
        cached = self._read_cache(path)
        if not self._cache_is_current(cached):
            return None
        return self._public_columns(cached)

    @classmethod
    def location_code(cls, station_id: str) -> str:
        return cls._LOCATION_CODES.get(station_id, f"{station_id}:9:US")

    def _cache_path(self, station_id: str, day: str) -> Path:
        return self._cache_root / station_id / f"{day}.parquet"

    @staticmethod
    def _read_cache(path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if "valid_time" in frame.columns:
            frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
        if "valid_hour" in frame.columns:
            frame["valid_hour"] = pd.to_datetime(frame["valid_hour"], utc=True)
        return frame

    @classmethod
    def _normalize(cls, station_id: str, payload: dict[str, Any]) -> pd.DataFrame:
        frame = cls._observation_frame(station_id, payload)
        if frame.empty:
            return frame
        frame = cls._select_representative_hours(frame)
        frame["_weather_com_alignment_version"] = _ALIGNMENT_VERSION
        return frame.reset_index(drop=True)

    @classmethod
    def _observation_frame(
        cls,
        station_id: str,
        payload: dict[str, Any],
    ) -> pd.DataFrame:
        rows = []
        for obs in payload.get("observations", []):
            temp_f = obs.get("temp")
            valid_time_gmt = obs.get("valid_time_gmt")
            if temp_f is None or valid_time_gmt is None:
                continue
            ts_utc = pd.Timestamp(valid_time_gmt, unit="s", tz="UTC")
            temp_f_float = float(temp_f)
            rows.append(
                {
                    "station_id": station_id,
                    "valid_time": ts_utc,
                    "valid_hour": ts_utc.round("h"),
                    "weather_com_temp_f": temp_f_float,
                    "weather_com_temp_c": (temp_f_float - 32.0) * 5.0 / 9.0,
                    "_weather_com_alignment_version": _ALIGNMENT_VERSION,
                }
            )
        if not rows:
            return cls._empty_frame()
        return pd.DataFrame(rows).reset_index(drop=True)

    @staticmethod
    def _select_representative_hours(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        aligned = frame.copy()
        aligned["valid_time"] = pd.to_datetime(aligned["valid_time"], utc=True)
        aligned["valid_hour"] = pd.to_datetime(aligned["valid_hour"], utc=True)
        target_time = aligned["valid_hour"] - pd.Timedelta(minutes=9)
        aligned["_representative_distance_s"] = (
            aligned["valid_time"] - target_time
        ).abs().dt.total_seconds()
        return (
            aligned.sort_values(
                ["station_id", "valid_hour", "_representative_distance_s", "valid_time"],
                ascending=[True, True, True, True],
            )
            .drop_duplicates(subset=["station_id", "valid_hour"], keep="first")
            .drop(columns=["_representative_distance_s"])
            .reset_index(drop=True)
        )

    @staticmethod
    def _filter_local_day(
        frame: pd.DataFrame,
        *,
        day: str,
        timezone: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        valid_time = pd.to_datetime(frame["valid_time"], utc=True)
        local_day = valid_time.dt.tz_convert(timezone).dt.date.astype(str)
        return frame.loc[local_day == day].reset_index(drop=True)

    @staticmethod
    def _cache_is_current(frame: pd.DataFrame) -> bool:
        if "_weather_com_alignment_version" not in frame.columns:
            return False
        versions = frame["_weather_com_alignment_version"].dropna().unique().tolist()
        return len(versions) == 1 and versions[0] in _COMPATIBLE_HOURLY_ALIGNMENT_VERSIONS

    @staticmethod
    def _cache_is_resolution_truth_current(frame: pd.DataFrame) -> bool:
        return (
            WeatherComTruthClient._cache_is_current(frame)
            and "weather_com_raw_daily_high_f" in frame.columns
            and frame["_weather_com_alignment_version"].dropna().unique().tolist()
            == [_ALIGNMENT_VERSION]
        )

    @staticmethod
    def _public_columns(frame: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "station_id",
            "valid_time",
            "valid_hour",
            "weather_com_temp_f",
            "weather_com_temp_c",
        ]
        return frame[[col for col in columns if col in frame.columns]].copy()

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "station_id",
                "valid_time",
                "valid_hour",
                "weather_com_temp_f",
                "weather_com_temp_c",
            ]
        )
