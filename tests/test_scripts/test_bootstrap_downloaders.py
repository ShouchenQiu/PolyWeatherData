from __future__ import annotations

import pandas as pd

from config.stations import STATIONS
from data.storage.parquet_store import ParquetStore
from scripts.batch_download_iem import BatchIemDownloader
from scripts.batch_download_zarr import BatchZarrDownloader


class FakeForecastSource:
    def __init__(self) -> None:
        self.calls: list[tuple[pd.Timestamp, list[int]]] = []

    def fetch(self, init_time, fxx_range, station):
        init = pd.Timestamp(init_time).tz_convert("UTC")
        fxx_values = list(fxx_range)
        self.calls.append((init, fxx_values))
        rows = []
        for fxx in fxx_values:
            rows.append(
                {
                    "init_time": init,
                    "fxx": fxx,
                    "valid_time": init + pd.Timedelta(hours=fxx),
                    "TMP_2m": 273.15 + fxx,
                    "DPT_2m": 270.15 + fxx,
                    "UGRD_10m": 1.0,
                    "VGRD_10m": 2.0,
                    "GUST_sfc": 0.0,
                    "PRATE": 0.0,
                }
            )
        return {station.station_id: pd.DataFrame(rows)}


class FakeObsSource:
    def __init__(self) -> None:
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fetch(self, station, start, end, now_utc=None):
        start_ts = pd.Timestamp(start).tz_convert("UTC")
        end_ts = pd.Timestamp(end).tz_convert("UTC")
        self.calls.append((start_ts, end_ts))
        return pd.DataFrame(
            [
                {
                    "valid_time": start_ts + pd.Timedelta(minutes=51),
                    "raw_tmpc_tenths": 20.0,
                    "raw_dwpc_tenths": 10.0,
                    "raw_wind_spd_kt": 4.0,
                    "raw_wind_gust_kt": None,
                    "obs_U": 1.0,
                    "obs_V": 0.0,
                    "skyc1": 0.0,
                    "p01i": None,
                    "is_rain": None,
                    "is_snow": None,
                }
            ]
        )


def test_batch_zarr_bootstrap_writes_canonical_f01_store(tmp_path):
    store = ParquetStore(tmp_path / "forecast")
    source = FakeForecastSource()

    downloader = BatchZarrDownloader(
        store,
        station_config=STATIONS["KLGA"],
        forecast_source=source,
        max_workers=1,
    )

    written = downloader.run("2026-01-01", "2026-01-01", init_hours=[0])

    df = store.read(
        "KLGA",
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-02", tz="UTC"),
    )
    assert written == 1
    assert source.calls == [
        (pd.Timestamp("2026-01-01 00:00:00", tz="UTC"), [1])
    ]
    assert list(df["fxx"]) == [1]
    assert df["TMP_2m"].iloc[0] == 1.0
    assert "WS_10m" in df.columns


def test_batch_iem_bootstrap_writes_clean_obs_store(tmp_path):
    store = ParquetStore(tmp_path / "obs")
    source = FakeObsSource()

    downloader = BatchIemDownloader(
        store,
        station_config=STATIONS["KLGA"],
        obs_source=source,
        delay=0.0,
    )

    written = downloader.run("2026-01-01", "2026-01-01")

    df = store.read(
        "KLGA",
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-02", tz="UTC"),
    )
    assert written == 1
    assert source.calls == [
        (
            pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2026-01-01 23:59:59", tz="UTC"),
        )
    ]
    assert list(df["valid_time"]) == [pd.Timestamp("2026-01-01 01:00:00", tz="UTC")]
    assert df["raw_wind_gust_kt"].iloc[0] == 4.0
    assert df["p01i"].iloc[0] == 0.0
    assert "obs_WS" in df.columns


def test_batch_iem_does_not_skip_day_with_only_boundary_obs(tmp_path):
    store = ParquetStore(tmp_path / "obs")
    store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "valid_time": pd.Timestamp("2026-06-04 00:00:00", tz="UTC"),
                    "raw_tmpc_tenths": 20.0,
                    "raw_dwpc_tenths": 10.0,
                    "raw_wind_spd_kt": 4.0,
                    "raw_wind_gust_kt": 4.0,
                    "obs_U": 1.0,
                    "obs_V": 0.0,
                    "skyc1": 0.0,
                    "p01i": 0.0,
                    "is_rain": 0.0,
                    "is_snow": 0.0,
                }
            ]
        ),
    )
    source = FakeObsSource()
    downloader = BatchIemDownloader(
        store,
        station_config=STATIONS["KLGA"],
        obs_source=source,
        delay=0.0,
    )

    downloader.run("2026-06-04", "2026-06-04")

    assert source.calls == [
        (
            pd.Timestamp("2026-06-04 00:00:00", tz="UTC"),
            pd.Timestamp("2026-06-04 23:59:59", tz="UTC"),
        )
    ]
