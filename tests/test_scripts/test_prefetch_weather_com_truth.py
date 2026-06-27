from __future__ import annotations

import pandas as pd

from data.storage.parquet_store import ParquetStore
from scripts.prefetch_weather_com_truth import prefetch_weather_com_truth


class _FakeTruthClient:
    cached_days = {"2026-05-31"}
    fetched_days: list[str] = []

    def __init__(self, _cache_root) -> None:
        return None

    def read_cached_date(self, _station_id: str, local_date: str):
        if local_date in self.cached_days:
            return pd.DataFrame([{"valid_hour": pd.Timestamp("2026-06-01", tz="UTC")}])
        return None

    def fetch_date(self, _station_id: str, local_date: str) -> pd.DataFrame:
        self.fetched_days.append(local_date)
        return pd.DataFrame([{"valid_hour": pd.Timestamp("2026-06-02", tz="UTC")}])


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


def test_prefetch_reports_partial_failures_and_keeps_completed_dates(tmp_path):
    class PartiallyFailingClient(_FakeTruthClient):
        cached_days = set()

        def fetch_date(self, _station_id: str, local_date: str) -> pd.DataFrame:
            if local_date == "2026-06-02":
                raise RuntimeError("rate limited")
            return pd.DataFrame([{"valid_hour": pd.Timestamp("2026-06-01", tz="UTC")}])

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
