"""Weather.com historical observation truth client with daily parquet cache."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_WEATHER_COM_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"


class WeatherComApiKeyMissing(RuntimeError):
    """Raised when Weather.com data is uncached and no API key is configured."""


class WeatherComFetchError(RuntimeError):
    """Raised when Weather.com cannot return a usable response."""


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
        path = self._cache_path(station_id, day)
        if path.exists():
            return self._read_cache(path)

        if not self._api_key:
            raise WeatherComApiKeyMissing(
                "WEATHER_COM_API_KEY is required to fetch uncached Weather.com truth data"
            )

        location = self.location_code(station_id)
        date_param = day.replace("-", "")
        url = self._BASE_URL.format(location=location)
        params = {
            "apiKey": self._api_key,
            "units": "e",
            "startDate": date_param,
            "endDate": date_param,
        }
        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise WeatherComFetchError(
                f"Weather.com fetch failed for station={station_id} date={day}: {exc}"
            ) from exc

        frame = self._normalize(station_id, payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame

    def fetch_dates(
        self,
        station_id: str,
        local_dates: list[str] | tuple[str, ...],
    ) -> pd.DataFrame:
        parts = [self.fetch_date(station_id, day) for day in local_dates]
        if not parts:
            return self._empty_frame()
        return pd.concat(parts, ignore_index=True)

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
                }
            )
        if not rows:
            return cls._empty_frame()
        frame = pd.DataFrame(rows)
        return (
            frame.sort_values("valid_time")
            .drop_duplicates(subset=["station_id", "valid_hour"], keep="last")
            .reset_index(drop=True)
        )

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
