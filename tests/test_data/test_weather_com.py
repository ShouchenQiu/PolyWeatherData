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


def _complete_payload(
    day: str,
    *,
    default_temp: float = 50.0,
    extras: list[dict] | None = None,
) -> dict:
    start = pd.Timestamp(day).tz_localize("America/New_York").tz_convert("UTC")
    end = (
        (pd.Timestamp(day) + pd.Timedelta(days=1))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    valid_hours = pd.date_range(
        start + pd.Timedelta(hours=1),
        end,
        freq="h",
    )
    observations = [
        {
            "valid_time_gmt": int(
                (valid_hour - pd.Timedelta(minutes=9)).timestamp()
            ),
            "temp": default_temp,
        }
        for valid_hour in valid_hours
    ]
    observations.extend(extras or [])
    return {"observations": observations}


def _complete_cached_frame(day: str, *, temp_f: float = 70.0) -> pd.DataFrame:
    payload = _complete_payload(day, default_temp=temp_f)
    rows = payload["observations"]
    valid_time = pd.to_datetime(
        [row["valid_time_gmt"] for row in rows], unit="s", utc=True
    )
    return pd.DataFrame(
        {
            "station_id": ["KLGA"] * len(rows),
            "valid_time": valid_time,
            "valid_hour": valid_time.round("h"),
            "weather_com_temp_f": [temp_f] * len(rows),
            "weather_com_temp_c": [(temp_f - 32.0) * 5.0 / 9.0] * len(rows),
        }
    )


def test_weather_com_fetch_parses_hourly_temperatures_and_writes_cache(tmp_path):
    payload = _complete_payload("2026-06-04", default_temp=50.0)
    payload["observations"][-1]["temp"] = 54.0
    session = _Session(payload)
    client = WeatherComTruthClient(
        tmp_path / "truth",
        api_key="secret",
        session=session,
    )

    frame = client.fetch_date("KLGA", "2026-06-04")

    assert len(frame) == 24
    assert set(frame["station_id"]) == {"KLGA"}
    assert frame["weather_com_temp_f"].iloc[[0, -1]].tolist() == [50.0, 54.0]
    assert frame["weather_com_temp_c"].iloc[[0, -1]].tolist() == pytest.approx(
        [10.0, 12.2222], abs=1e-4
    )
    assert pd.to_datetime(frame["valid_time"], utc=True).dt.tz is not None
    assert (tmp_path / "truth" / "KLGA" / "2026-06-04.parquet").exists()
    assert session.calls[0]["params"]["apiKey"] == "secret"
    assert session.calls[0]["params"]["units"] == "e"
    assert session.calls[0]["params"]["startDate"] == "20260604"
    assert session.calls[0]["params"]["endDate"] == "20260605"


def test_weather_com_fetch_uses_cache_without_api_key_or_http(tmp_path):
    cache_dir = tmp_path / "truth" / "KLGA"
    cache_dir.mkdir(parents=True)
    cached = _complete_cached_frame("2026-06-04")
    cached.to_parquet(cache_dir / "2026-06-04.parquet", index=False)
    session = _Session({"observations": []})
    client = WeatherComTruthClient(tmp_path / "truth", api_key="", session=session)

    frame = client.fetch_date("KLGA", "2026-06-04")

    assert len(frame) == 24
    assert set(frame["weather_com_temp_f"]) == {70.0}
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
    payload = _complete_payload("2026-06-04")
    session = _Session(payload)
    monkeypatch.delenv("WEATHER_COM_API_KEY", raising=False)
    client = WeatherComTruthClient(tmp_path / "truth", session=session)

    client.fetch_date("KLGA", "2026-06-04")

    assert session.calls[0]["params"]["apiKey"] == DEFAULT_WEATHER_COM_API_KEY


def test_weather_com_prefers_iem_representative_hour_candidate(tmp_path):
    valid_hour = pd.Timestamp("2026-05-24 12:00:00", tz="UTC")
    payload = _complete_payload("2026-05-24")
    representative_ts = int(
        (valid_hour - pd.Timedelta(minutes=9)).timestamp()
    )
    for observation in payload["observations"]:
        if observation["valid_time_gmt"] == representative_ts:
            observation["temp"] = 53
    payload["observations"].append(
        {
            "valid_time_gmt": int(
                (valid_hour + pd.Timedelta(minutes=19)).timestamp()
            ),
            "temp": 54,
        }
    )
    client = WeatherComTruthClient(
        tmp_path / "truth",
        api_key="secret",
        session=_Session(payload),
    )

    frame = client.fetch_date("KLGA", "2026-05-24")

    selected = frame.loc[frame["valid_hour"] == valid_hour].iloc[0]
    assert selected["valid_time"] == valid_hour - pd.Timedelta(minutes=9)
    assert selected["weather_com_temp_f"] == 53.0


def test_weather_com_cache_retains_raw_local_day_high_separately_from_hourly_representation(
    tmp_path,
):
    valid_hour = pd.Timestamp("2026-05-24 12:00:00", tz="UTC")
    payload = _complete_payload("2026-05-24", default_temp=50.0)
    payload["observations"].append(
        {
            "valid_time_gmt": int(
                (valid_hour + pd.Timedelta(minutes=19)).timestamp()
            ),
            "temp": 54,
        }
    )
    client = WeatherComTruthClient(
        tmp_path / "truth",
        api_key="secret",
        session=_Session(payload),
    )

    hourly = client.fetch_date("KLGA", "2026-05-24")
    cached = pd.read_parquet(tmp_path / "truth" / "KLGA" / "2026-05-24.parquet")

    assert hourly["weather_com_temp_f"].max() == 50.0
    assert cached["weather_com_raw_daily_high_f"].unique().tolist() == [54.0]


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
    payload = _complete_payload("2026-05-20", default_temp=60.0)
    representative_ts = int(
        (valid_hour - pd.Timedelta(minutes=9)).timestamp()
    )
    for observation in payload["observations"]:
        if observation["valid_time_gmt"] == representative_ts:
            observation["temp"] = 74
    payload["observations"].append(
        {
            "valid_time_gmt": int(
                (valid_hour + pd.Timedelta(minutes=23)).timestamp()
            ),
            "temp": 70,
        }
    )
    session = _Session(payload)
    client = WeatherComTruthClient(tmp_path / "truth", api_key="secret", session=session)

    frame = client.fetch_date("KLGA", "2026-05-20")

    assert len(session.calls) == 1
    selected = frame.loc[frame["valid_hour"] == valid_hour].iloc[0]
    assert selected["valid_time"] == valid_hour - pd.Timedelta(minutes=9)
    assert selected["weather_com_temp_f"] == 74.0


def test_weather_com_fetch_uses_extended_range_and_keeps_requested_dst_day(tmp_path):
    payload = _complete_payload("2026-03-08", default_temp=48.0)
    payload["observations"].append(
        {
            "valid_time_gmt": int(
                pd.Timestamp("2026-03-09T12:51:00Z").timestamp()
            ),
            "temp": 99.0,
        }
    )
    session = _Session(payload)
    client = WeatherComTruthClient(
        tmp_path / "truth",
        api_key="secret",
        session=session,
    )

    frame = client.fetch_date("KLGA", "2026-03-08")

    assert len(frame) == 23
    assert frame["weather_com_temp_f"].max() == 48.0
    assert session.calls[0]["params"]["startDate"] == "20260308"
    assert session.calls[0]["params"]["endDate"] == "20260309"
