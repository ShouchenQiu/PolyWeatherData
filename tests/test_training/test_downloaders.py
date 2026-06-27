from __future__ import annotations

import inspect

import pandas as pd

from config.stations import STATIONS
from data.storage.parquet_store import ParquetStore
from training.downloaders.hrrr import BatchHrrrF01Downloader
from training.downloaders.iem import BatchIemDownloader


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
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]] = []

    def fetch(self, station, start, end, now_utc=None):
        start_ts = pd.Timestamp(start).tz_convert("UTC")
        end_ts = pd.Timestamp(end).tz_convert("UTC")
        now_ts = pd.Timestamp(now_utc).tz_convert("UTC") if now_utc is not None else None
        self.calls.append((start_ts, end_ts, now_ts))
        rows = []
        for valid_time in pd.date_range(start_ts, end_ts.floor("h"), freq="h"):
            rows.append(
                {
                    "valid_time": valid_time,
                    "raw_tmpc_tenths": 99.0,
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
            )
            rows.append(
                {
                    "valid_time": valid_time.replace(minute=51),
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
            )
        return pd.DataFrame(
            rows
        )


class FakeObsSourceWithMissing51:
    def __init__(self) -> None:
        self.calls: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]] = []

    def fetch(self, station, start, end, now_utc=None):
        start_ts = pd.Timestamp(start).tz_convert("UTC")
        end_ts = pd.Timestamp(end).tz_convert("UTC")
        now_ts = pd.Timestamp(now_utc).tz_convert("UTC") if now_utc is not None else None
        self.calls.append((start_ts, end_ts, now_ts))
        rows = []
        for valid_time in pd.date_range(start_ts, end_ts.floor("h"), freq="h"):
            representative = valid_time.replace(minute=51)
            if representative == pd.Timestamp("2026-01-01 05:51:00", tz="UTC"):
                candidate_times = [
                    valid_time.replace(minute=50),
                    valid_time.replace(minute=55),
                ]
            else:
                candidate_times = [representative]
            for candidate in candidate_times:
                rows.append(
                    {
                        "valid_time": candidate,
                        "raw_tmpc_tenths": 21.0 if candidate.minute == 50 else 20.0,
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
                )
        return pd.DataFrame(rows)


class FakeObsSourceWithFarOnlyCandidate(FakeObsSourceWithMissing51):
    def fetch(self, station, start, end, now_utc=None):
        df = super().fetch(station, start, end, now_utc=now_utc)
        valid_time = pd.to_datetime(df["valid_time"], utc=True)
        far_hour = pd.Timestamp("2026-01-01 05:00:00", tz="UTC")
        mask = (
            (valid_time.dt.floor("h") == far_hour)
            & valid_time.dt.minute.isin([50, 55])
        )
        df = df.loc[~mask].copy()
        df.loc[len(df)] = {
            "valid_time": far_hour.replace(minute=20),
            "raw_tmpc_tenths": 30.0,
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
        return df


def test_batch_iem_interface_has_no_retired_compatibility_members():
    parameters = inspect.signature(BatchIemDownloader).parameters
    assert "retries" not in parameters
    assert not hasattr(BatchIemDownloader, "_filter_exact_51")


def test_batch_zarr_bootstrap_writes_canonical_f01_store(tmp_path):
    store = ParquetStore(tmp_path / "forecast")
    source = FakeForecastSource()

    downloader = BatchHrrrF01Downloader(
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
    assert written == 24
    assert source.calls == [
        (
            pd.Timestamp("2025-12-31 23:00:00", tz="UTC"),
            pd.Timestamp("2026-01-01 22:59:59", tz="UTC"),
            None,
        )
    ]
    assert list(df["valid_time"]) == list(
        pd.date_range(
            "2026-01-01 00:00:00",
            "2026-01-01 23:00:00",
            freq="h",
            tz="UTC",
        )
    )
    assert set(df["raw_tmpc_tenths"]) == {20.0}
    assert df["raw_wind_gust_kt"].iloc[0] == 4.0
    assert df["p01i"].iloc[0] == 0.0
    assert "obs_WS" in df.columns


def test_batch_iem_prefers_exact_51_when_available(tmp_path):
    store = ParquetStore(tmp_path / "obs")
    source = FakeObsSource()
    downloader = BatchIemDownloader(
        store,
        station_config=STATIONS["KLGA"],
        obs_source=source,
        delay=0.0,
    )

    downloader.run("2026-01-01", "2026-01-01")

    df = store.read(
        "KLGA",
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-02 00:00:00", tz="UTC"),
    )

    assert len(source.calls) == 1
    assert len(df) == 24
    assert df["raw_tmpc_tenths"].tolist() == [20.0] * 24


def test_batch_iem_uses_nearest_representative_when_exact_51_missing(tmp_path):
    store = ParquetStore(tmp_path / "obs")
    source = FakeObsSourceWithMissing51()
    downloader = BatchIemDownloader(
        store,
        station_config=STATIONS["KLGA"],
        obs_source=source,
        delay=0.0,
    )

    written = downloader.run("2026-01-01", "2026-01-01")

    df = store.read(
        "KLGA",
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 23:00:00", tz="UTC"),
    )

    assert written == 24
    assert len(df) == 24
    missing_51_hour = df[df["valid_time"] == pd.Timestamp("2026-01-01 06:00:00", tz="UTC")]
    assert missing_51_hour["raw_tmpc_tenths"].iloc[0] == 21.0


def test_batch_iem_drops_hour_when_only_candidate_is_outside_tolerance(tmp_path):
    store = ParquetStore(tmp_path / "obs")
    source = FakeObsSourceWithFarOnlyCandidate()
    downloader = BatchIemDownloader(
        store,
        station_config=STATIONS["KLGA"],
        obs_source=source,
        delay=0.0,
    )

    written = downloader.run("2026-01-01", "2026-01-01")

    df = store.read(
        "KLGA",
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 23:00:00", tz="UTC"),
    )

    assert written < 24
    assert len(df) == written
    assert pd.Timestamp("2026-01-01 06:00:00", tz="UTC") not in set(df["valid_time"])
    assert 30.0 not in set(df["raw_tmpc_tenths"])


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
            pd.Timestamp("2026-06-03 23:00:00", tz="UTC"),
            pd.Timestamp("2026-06-04 22:59:59", tz="UTC"),
            None,
        )
    ]


def test_batch_iem_does_not_skip_day_with_23_existing_hours(tmp_path):
    store = ParquetStore(tmp_path / "obs")
    rows = []
    for valid_time in pd.date_range(
        "2026-06-04 00:00:00",
        "2026-06-04 22:00:00",
        freq="h",
        tz="UTC",
    ):
        rows.append(
            {
                "valid_time": valid_time,
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
        )
    store.append("KLGA", pd.DataFrame(rows))
    source = FakeObsSource()
    downloader = BatchIemDownloader(
        store,
        station_config=STATIONS["KLGA"],
        obs_source=source,
        delay=0.0,
    )

    downloader.run("2026-06-04", "2026-06-04")

    assert len(source.calls) == 1
