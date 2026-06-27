from __future__ import annotations

import pandas as pd
import pytest

from data.obs.weather_com import DEFAULT_WEATHER_COM_API_KEY, WeatherComApiKeyMissing, WeatherComTruthClient


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Session:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def get(self, url: str, params: dict, timeout: float) -> _Response:
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return _Response(self.payload)


def test_weather_com_fetch_parses_hourly_temperatures_and_writes_cache(tmp_path):
    payload = {
        "observations": [
            {"valid_time_gmt": 1_780_000_000, "temp": 50},
            {"valid_time_gmt": 1_780_003_600, "temp": None},
            {"valid_time_gmt": 1_780_007_200, "temp": 54},
        ]
    }
    session = _Session(payload)
    client = WeatherComTruthClient(
        tmp_path / "truth",
        api_key="secret",
        session=session,
    )

    frame = client.fetch_date("KLGA", "2026-06-04")

    assert len(frame) == 2
    assert frame["station_id"].tolist() == ["KLGA", "KLGA"]
    assert frame["weather_com_temp_f"].tolist() == [50.0, 54.0]
    assert frame["weather_com_temp_c"].tolist() == pytest.approx(
        [10.0, 12.2222],
        abs=1e-4,
    )
    assert pd.to_datetime(frame["valid_time"], utc=True).dt.tz is not None
    assert (tmp_path / "truth" / "KLGA" / "2026-06-04.parquet").exists()
    assert session.calls[0]["params"]["apiKey"] == "secret"
    assert session.calls[0]["params"]["units"] == "e"
    assert session.calls[0]["params"]["startDate"] == "20260604"
    assert session.calls[0]["params"]["endDate"] == "20260604"


def test_weather_com_fetch_uses_cache_without_api_key_or_http(tmp_path):
    cache_dir = tmp_path / "truth" / "KLGA"
    cache_dir.mkdir(parents=True)
    cached = pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_time": pd.Timestamp("2026-06-04 01:00:00", tz="UTC"),
                "valid_hour": pd.Timestamp("2026-06-04 01:00:00", tz="UTC"),
                "weather_com_temp_f": 70.0,
                "weather_com_temp_c": 21.111111,
            }
        ]
    )
    cached.to_parquet(cache_dir / "2026-06-04.parquet", index=False)
    session = _Session({"observations": []})
    client = WeatherComTruthClient(tmp_path / "truth", api_key="", session=session)

    frame = client.fetch_date("KLGA", "2026-06-04")

    assert frame["weather_com_temp_f"].tolist() == [70.0]
    assert session.calls == []


def test_weather_com_fetch_without_key_and_cache_raises(tmp_path):
    client = WeatherComTruthClient(tmp_path / "truth", api_key="")

    with pytest.raises(WeatherComApiKeyMissing):
        client.fetch_date("KLGA", "2026-06-04")


def test_weather_com_read_cached_date_never_fetches_missing_day(tmp_path):
    session = _Session({"observations": [{"valid_time_gmt": 1_780_000_000, "temp": 50}]})
    client = WeatherComTruthClient(
        tmp_path / "truth",
        api_key="secret",
        session=session,
    )

    frame = client.read_cached_date("KLGA", "2026-06-04")

    assert frame is None
    assert session.calls == []


def test_weather_com_client_uses_reference_default_key_when_env_missing(tmp_path, monkeypatch):
    payload = {"observations": [{"valid_time_gmt": 1_780_000_000, "temp": 50}]}
    session = _Session(payload)
    monkeypatch.delenv("WEATHER_COM_API_KEY", raising=False)
    client = WeatherComTruthClient(tmp_path / "truth", session=session)

    client.fetch_date("KLGA", "2026-06-04")

    assert session.calls[0]["params"]["apiKey"] == DEFAULT_WEATHER_COM_API_KEY


def test_weather_com_prefers_iem_representative_hour_candidate(tmp_path):
    valid_hour = pd.Timestamp("2026-05-24 12:00:00", tz="UTC")
    payload = {
        "observations": [
            {
                "valid_time_gmt": int((valid_hour - pd.Timedelta(minutes=9)).timestamp()),
                "temp": 53,
            },
            {
                "valid_time_gmt": int((valid_hour + pd.Timedelta(minutes=19)).timestamp()),
                "temp": 54,
            },
        ]
    }
    client = WeatherComTruthClient(
        tmp_path / "truth",
        api_key="secret",
        session=_Session(payload),
    )

    frame = client.fetch_date("KLGA", "2026-05-24")

    assert frame["valid_hour"].tolist() == [valid_hour]
    assert frame["valid_time"].tolist() == [valid_hour - pd.Timedelta(minutes=9)]
    assert frame["weather_com_temp_f"].tolist() == [53.0]


def test_weather_com_refetches_stale_alignment_cache_when_key_available(tmp_path):
    cache_dir = tmp_path / "truth" / "KLGA"
    cache_dir.mkdir(parents=True)
    valid_hour = pd.Timestamp("2026-05-21 00:00:00", tz="UTC")
    pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_time": valid_hour + pd.Timedelta(minutes=23),
                "valid_hour": valid_hour,
                "weather_com_temp_f": 70.0,
                "weather_com_temp_c": 21.111111,
            }
        ]
    ).to_parquet(cache_dir / "2026-05-20.parquet", index=False)
    payload = {
        "observations": [
            {
                "valid_time_gmt": int((valid_hour - pd.Timedelta(minutes=9)).timestamp()),
                "temp": 74,
            },
            {
                "valid_time_gmt": int((valid_hour + pd.Timedelta(minutes=23)).timestamp()),
                "temp": 70,
            },
        ]
    }
    session = _Session(payload)
    client = WeatherComTruthClient(tmp_path / "truth", api_key="secret", session=session)

    frame = client.fetch_date("KLGA", "2026-05-20")

    assert len(session.calls) == 1
    assert frame["valid_time"].tolist() == [valid_hour - pd.Timedelta(minutes=9)]
    assert frame["weather_com_temp_f"].tolist() == [74.0]
