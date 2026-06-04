from __future__ import annotations

import pandas as pd

from config.stations import STATIONS
from config.settings import Settings
from data.manager import DataManager
from data.storage.parquet_store import ParquetStore
from model.training_data import TrainingDatasetBuilder, TrainingDecoderCache


class FakeDecoderSource:
    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, init_time, fxx_range, station):
        self.calls += 1
        init = pd.Timestamp(init_time).tz_convert("UTC")
        rows = []
        for fxx in range(0, 49):
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


def test_decoder_cache_miss_fetches_cleans_and_persists(tmp_path):
    source = FakeDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=source)
    init = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

    df = cache.get(STATIONS["KLGA"], init)

    assert source.calls == 1
    assert list(df["fxx"]) == list(range(1, 49))
    assert df["TMP_2m"].iloc[0] == 1.0
    assert "WS_10m" in df.columns
    assert cache.path_for("KLGA", init).exists()


def test_decoder_cache_hit_does_not_fetch(tmp_path):
    source = FakeDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=source)
    init = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

    first = cache.get(STATIONS["KLGA"], init)
    second = cache.get(STATIONS["KLGA"], init)

    assert source.calls == 1
    pd.testing.assert_frame_equal(first, second)


def _append_bootstrap_windows(
    forecast_store: ParquetStore,
    obs_store: ParquetStore,
    station_id: str,
    start: pd.Timestamp,
    hours: int,
) -> None:
    hrrr_rows = []
    obs_rows = []
    for i in range(hours):
        vt = start + pd.Timedelta(hours=i)
        hrrr_rows.append(
            {
                "init_time": vt,
                "fxx": 1,
                "valid_time": vt,
                "TMP_2m": 5.0,
                "DPT_2m": 1.0,
                "UGRD_10m": 1.0,
                "VGRD_10m": 2.0,
                "GUST_sfc": 0.0,
                "PRATE": 0.0,
            }
        )
        obs_rows.append(
            {
                "valid_time": vt,
                "raw_tmpc_tenths": 6.0,
                "raw_dwpc_tenths": 1.0,
                "raw_wind_spd_kt": 4.0,
                "raw_wind_gust_kt": 4.0,
                "obs_U": 1.0,
                "obs_V": 0.0,
                "skyc1": 0.0,
                "p01i": 0.0,
            }
        )
    forecast_store.append(station_id, pd.DataFrame(hrrr_rows))
    obs_store.append(station_id, pd.DataFrame(obs_rows))


def test_training_builder_uses_bootstrap_stores_to_build_one_sample(tmp_path):
    station = STATIONS["KLGA"]
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    first_row = init - pd.Timedelta(hours=144)
    forecast_store = ParquetStore(tmp_path / "forecast")
    obs_store = ParquetStore(tmp_path / "obs")
    _append_bootstrap_windows(forecast_store, obs_store, station.station_id, first_row, 193)
    data_manager = DataManager([], [], forecast_store, obs_store, settings=Settings())
    cache = TrainingDecoderCache(tmp_path / "decoder-cache", source=FakeDecoderSource())
    builder = TrainingDatasetBuilder(data_manager, cache)

    frame = builder.build_frame(station, [init], split="train")

    assert builder.skipped == []
    assert len(frame) == 120
    assert frame["split"].unique().tolist() == ["train"]
    assert frame["init_time_str"].nunique() == 1
    assert frame["Node_ID"].unique().tolist() == ["KLGA"]
    assert frame["local_idx"].tolist() == list(range(120))
    assert frame.iloc[:72]["temp_error"].notna().all()
    assert set(frame.iloc[72:]["fxx"].astype(int)) == set(range(1, 49))


def test_training_builder_records_readiness_skips(tmp_path):
    station = STATIONS["KLGA"]
    init = pd.Timestamp("2026-01-10 12:00:00", tz="UTC")
    forecast_store = ParquetStore(tmp_path / "forecast")
    obs_store = ParquetStore(tmp_path / "obs")
    data_manager = DataManager([], [], forecast_store, obs_store, settings=Settings())
    cache = TrainingDecoderCache(tmp_path / "decoder-cache", source=FakeDecoderSource())
    builder = TrainingDatasetBuilder(data_manager, cache)

    frame = builder.build_frame(station, [init], split="train")

    assert frame.empty
    assert len(builder.skipped) == 1
    assert builder.skipped[0]["init_time"] == init
    assert builder.skipped[0]["reason"] == "cold_start"
