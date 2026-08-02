from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pandas as pd

from data.storage.parquet_store import ParquetStore
from scripts import prefetch_weather_com_truth as prefetch_module
from scripts.prefetch_weather_com_truth import prefetch_weather_com_truth


def _complete_klga_day(day: str) -> pd.DataFrame:
    start = pd.Timestamp(day).tz_localize("America/New_York").tz_convert("UTC")
    end = (
        (pd.Timestamp(day) + pd.Timedelta(days=1))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame(
        {
            "valid_time": hours + pd.Timedelta(minutes=51),
            "valid_hour": hours + pd.Timedelta(hours=1),
            "weather_com_temp_f": [50.0] * len(hours),
        }
    )


class _FakeTruthClient:
    cached_days = {"2026-05-31"}
    fetched_days: list[str] = []

    def __init__(self, _cache_root) -> None:
        return None

    def read_cached_date(self, _station_id: str, local_date: str):
        if local_date in self.cached_days:
            return _complete_klga_day(local_date)
        return None

    def fetch_date(self, _station_id: str, local_date: str) -> pd.DataFrame:
        self.fetched_days.append(local_date)
        return _complete_klga_day(local_date)


def test_prefetch_infers_local_dates_from_obs_and_skips_cached_days(tmp_path):
    _FakeTruthClient.fetched_days = []
    data_store = tmp_path / "data_store"
    obs = ParquetStore(data_store / "obs")
    obs.append(
        "KLGA",
        pd.DataFrame(
            [
                {"valid_time": pd.Timestamp("2026-06-01 00:00", tz="UTC")},
                {"valid_time": pd.Timestamp("2026-06-02 12:00", tz="UTC")},
            ]
        ),
    )

    summary = prefetch_weather_com_truth(
        data_store_dir=data_store,
        truth_store_dir=tmp_path / "truth",
        station_id="KLGA",
        workers=1,
        client_cls=_FakeTruthClient,
    )

    assert summary["requested_dates"] == 2
    assert summary["cached_dates"] == 1
    assert summary["fetched_dates"] == 1
    assert summary["errors"] == 0
    assert _FakeTruthClient.fetched_days == ["2026-06-02"]


def test_prefetch_explicit_range_force_refreshes_complete_cache(tmp_path):
    _FakeTruthClient.fetched_days = []

    summary = prefetch_weather_com_truth(
        data_store_dir=tmp_path / "data_store",
        truth_store_dir=tmp_path / "truth",
        station_id="KLGA",
        start="2026-06-01",
        end="2026-06-02",
        workers=1,
        force_refresh=True,
        client_cls=_FakeTruthClient,
    )

    assert summary["requested_dates"] == 2
    assert summary["cached_dates"] == 0
    assert summary["fetched_dates"] == 2
    assert _FakeTruthClient.fetched_days == ["2026-06-01", "2026-06-02"]


def test_prefetch_reports_partial_failures_and_keeps_completed_dates(tmp_path):
    class PartiallyFailingClient(_FakeTruthClient):
        cached_days = set()

        def fetch_date(self, _station_id: str, local_date: str) -> pd.DataFrame:
            if local_date == "2026-06-02":
                raise RuntimeError("rate limited")
            return _complete_klga_day(local_date)

    data_store = tmp_path / "data_store"
    ParquetStore(data_store / "obs").append(
        "KLGA",
        pd.DataFrame(
            [
                {"valid_time": pd.Timestamp("2026-06-01 12:00", tz="UTC")},
                {"valid_time": pd.Timestamp("2026-06-02 12:00", tz="UTC")},
            ]
        ),
    )

    summary = prefetch_weather_com_truth(
        data_store,
        tmp_path / "truth",
        workers=2,
        client_cls=PartiallyFailingClient,
    )

    assert summary["fetched_dates"] == 1
    assert summary["errors"] == 1
    assert summary["error_details"][0]["date"] == "2026-06-02"


def test_prefetch_refetches_incomplete_spring_forward_cache(tmp_path):
    class IncompleteDstClient:
        fetched_days: list[str] = []

        def __init__(self, _cache_root) -> None:
            return None

        def read_cached_date(self, _station_id: str, local_date: str):
            return _complete_klga_day(local_date).iloc[:2].copy()

        def fetch_date(self, _station_id: str, local_date: str) -> pd.DataFrame:
            self.fetched_days.append(local_date)
            return _complete_klga_day(local_date)

    data_store = tmp_path / "data_store"
    ParquetStore(data_store / "obs").append(
        "KLGA",
        pd.DataFrame(
            [{"valid_time": pd.Timestamp("2026-03-08 12:00", tz="UTC")}]
        ),
    )

    summary = prefetch_weather_com_truth(
        data_store,
        tmp_path / "truth",
        station_id="KLGA",
        workers=1,
        client_cls=IncompleteDstClient,
    )

    assert summary["cached_dates"] == 0
    assert summary["fetched_dates"] == 1
    assert summary["errors"] == 0
    assert IncompleteDstClient.fetched_days == ["2026-03-08"]


def test_cli_defaults_to_project_truth_store_not_selected_data_store(
    tmp_path,
    monkeypatch,
    capsys,
):
    selected_store = tmp_path / "run" / "data_store"
    project_truth = tmp_path / "project" / "data_store" / "_truth_weather_com"
    monkeypatch.delenv("POLYWEATHER_TRUTH_STORE", raising=False)
    monkeypatch.setattr(
        prefetch_module,
        "Settings",
        lambda: type(
            "_Settings",
            (),
            {
                "data_store_dir": tmp_path / "default-data",
                "truth_store_dir": project_truth,
            },
        )(),
    )
    prefetch = MagicMock(return_value={"errors": 0})
    monkeypatch.setattr(prefetch_module, "prefetch_weather_com_truth", prefetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prefetch_weather_com_truth",
            "--data-store",
            str(selected_store),
        ],
    )

    prefetch_module._cli()

    assert prefetch.call_args.kwargs["data_store_dir"] == selected_store
    assert prefetch.call_args.kwargs["truth_store_dir"] == project_truth
    assert '"errors": 0' in capsys.readouterr().out
