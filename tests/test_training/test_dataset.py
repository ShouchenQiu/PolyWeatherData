from __future__ import annotations

import pandas as pd

from config.stations import STATIONS
from config.settings import Settings
from data.manager import DataManager
from data.storage.parquet_store import ParquetStore
from training.dataset import TrainingDatasetBuilder, TrainingDecoderCache


class FakeDecoderSource:
    def __init__(self) -> None:
        self.calls = 0
        self.called_inits: list[pd.Timestamp] = []

    def fetch(self, init_time, fxx_range, station):
        self.calls += 1
        init = pd.Timestamp(init_time).tz_convert("UTC")
        self.called_inits.append(init)
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


class FakeCachedDecoderSource(FakeDecoderSource):
    def __init__(self) -> None:
        super().__init__()
        self._full_window_cache = {}


class FailingDecoderSource:
    def __init__(self) -> None:
        self.called_inits: list[pd.Timestamp] = []

    def fetch(self, init_time, fxx_range, station):
        init = pd.Timestamp(init_time).tz_convert("UTC")
        self.called_inits.append(init)
        raise RuntimeError("primary zarr missing")


class MissingOffhourForecastSource(FakeDecoderSource):
    def __init__(self) -> None:
        super().__init__()
        self._full_window_cache = {}

    def fetch(self, init_time, fxx_range, station):
        init = pd.Timestamp(init_time).tz_convert("UTC")
        if init.hour not in {0, 6, 12, 18}:
            self.called_inits.append(init)
            raise RuntimeError("zarr fcst store missing")
        return super().fetch(init, fxx_range, station)


class NativeOnlyFallbackSource(FakeDecoderSource):
    def __init__(self) -> None:
        super().__init__()
        self.called_ranges: list[list[int]] = []

    def fetch(self, init_time, fxx_range, station):
        self.calls += 1
        init = pd.Timestamp(init_time).tz_convert("UTC")
        self.called_inits.append(init)
        requested = [int(fxx) for fxx in fxx_range]
        self.called_ranges.append(requested)
        rows = []
        for fxx in requested:
            if init.hour not in {0, 6, 12, 18} and fxx > 18:
                continue
            rows.append(
                {
                    "init_time": init,
                    "fxx": fxx,
                    "valid_time": init + pd.Timedelta(hours=fxx),
                    "TMP_2m": 280.0 + fxx,
                    "DPT_2m": 275.0 + fxx,
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
    assert cache.path_for("KLGA", init).name == "2026-01.parquet"


def test_decoder_cache_miss_uses_fallback_source_when_primary_fails(tmp_path):
    primary = FailingDecoderSource()
    fallback = FakeDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=primary, fallback_source=fallback)
    init = pd.Timestamp("2026-01-01 23:00:00", tz="UTC")

    df = cache.get(STATIONS["KLGA"], init)

    assert primary.called_inits == [init]
    assert fallback.called_inits == [init]
    assert list(df["fxx"]) == list(range(1, 49))
    stored = pd.read_parquet(cache.path_for("KLGA", init))
    assert set(stored["fxx"].astype(int)) == set(range(1, 49))


def test_decoder_cache_hybrid_fallback_uses_s3_native_and_primary_major(tmp_path):
    primary = MissingOffhourForecastSource()
    fallback = NativeOnlyFallbackSource()
    cache = TrainingDecoderCache(tmp_path, source=primary, fallback_source=fallback)
    station = STATIONS["KLGA"]
    init = pd.Timestamp("2026-04-28 23:00:00", tz="UTC")

    df = cache.get(station, init)

    assert primary.called_inits == [
        init,
        pd.Timestamp("2026-04-28 18:00:00", tz="UTC"),
    ]
    assert fallback.called_inits == [init]
    assert fallback.called_ranges == [list(range(0, 19))]
    assert list(df["fxx"].astype(int)) == list(range(1, 49))
    assert df.loc[df["fxx"] == 1, "TMP_2m"].iloc[0] == 280.0 + 1 - 273.15
    assert df.loc[df["fxx"] == 20, "TMP_2m"].iloc[0] == 25.0
    stored = pd.read_parquet(cache.path_for("KLGA", init))
    assert set(stored["fxx"].astype(int)) == set(range(1, 49))


def test_decoder_cache_progress_reports_hybrid_route(tmp_path, capsys):
    primary = MissingOffhourForecastSource()
    fallback = NativeOnlyFallbackSource()
    cache = TrainingDecoderCache(tmp_path, source=primary, fallback_source=fallback)
    init = pd.Timestamp("2026-04-28 23:00:00", tz="UTC")

    summary = cache.prewarm(
        STATIONS["KLGA"], [init], workers=1, progress=True, print_every=1
    )

    output = capsys.readouterr().out
    assert "mode=hybrid" in output
    assert "current=NativeOnlyFallbackSource[fxx=0..18]" in output
    assert (
        "tail=MissingOffhourForecastSource"
        "[nearest_major=2026-04-28 18:00:00+00:00]"
    ) in output
    assert "primary_error=RuntimeError: zarr fcst store missing" in output
    assert "routes=zarr:0 hybrid:1 full_fallback:0" in output
    assert summary["hybrid_fallback_success_inits"] == 1
    assert summary["full_fallback_success_inits"] == 0


def test_decoder_cache_progress_reports_full_fallback_route(tmp_path, capsys):
    primary = FailingDecoderSource()
    fallback = FakeDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=primary, fallback_source=fallback)
    init = pd.Timestamp("2026-04-28 18:00:00", tz="UTC")

    summary = cache.prewarm(
        STATIONS["KLGA"], [init], workers=1, progress=True, print_every=1
    )

    output = capsys.readouterr().out
    assert "mode=full_fallback" in output
    assert "source=FakeDecoderSource[fxx=0..48]" in output
    assert "primary_error=RuntimeError: primary zarr missing" in output
    assert "routes=zarr:0 hybrid:0 full_fallback:1" in output
    assert summary["hybrid_fallback_success_inits"] == 0
    assert summary["full_fallback_success_inits"] == 1


def test_decoder_cache_hit_does_not_fetch(tmp_path):
    source = FakeDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=source)
    init = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

    first = cache.get(STATIONS["KLGA"], init)
    second = cache.get(STATIONS["KLGA"], init)

    assert source.calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_decoder_cache_hit_does_not_initialize_download_sources(tmp_path):
    cache = TrainingDecoderCache(tmp_path)
    station = STATIONS["KLGA"]
    init = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    rows = []
    for fxx in range(1, 49):
        rows.append(
            {
                "init_time": init,
                "fxx": fxx,
                "valid_time": init + pd.Timedelta(hours=fxx),
                "TMP_2m": float(fxx),
            }
        )
    pd.DataFrame(rows).to_parquet(cache.path_for("KLGA", init), index=False)

    result = cache.get(station, init)

    assert len(result) == 48
    assert cache.source is None
    assert cache.fallback_source is None


def test_decoder_cache_monthly_file_holds_multiple_inits(tmp_path):
    source = FakeDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=source)
    station = STATIONS["KLGA"]
    init_a = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    init_b = pd.Timestamp("2026-01-01 01:00:00", tz="UTC")

    cache.get(station, init_a)
    cache.get(station, init_b)

    path = cache.path_for("KLGA", init_a)
    stored = pd.read_parquet(path)
    assert path == cache.path_for("KLGA", init_b)
    assert len(stored) == 96
    assert set(pd.to_datetime(stored["init_time"], utc=True)) == {init_a, init_b}


def test_decoder_cache_cross_month_uses_separate_files(tmp_path):
    cache = TrainingDecoderCache(tmp_path, source=FakeDecoderSource())
    station = STATIONS["KLGA"]
    jan = pd.Timestamp("2026-01-31 23:00:00", tz="UTC")
    feb = pd.Timestamp("2026-02-01 00:00:00", tz="UTC")

    cache.get(station, jan)
    cache.get(station, feb)

    assert cache.path_for("KLGA", jan).name == "2026-01.parquet"
    assert cache.path_for("KLGA", feb).name == "2026-02.parquet"
    assert cache.path_for("KLGA", jan).exists()
    assert cache.path_for("KLGA", feb).exists()


def test_decoder_cache_prewarm_skips_cached_inits(tmp_path):
    source = FakeDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=source)
    station = STATIONS["KLGA"]
    init_a = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    init_b = pd.Timestamp("2026-01-01 01:00:00", tz="UTC")

    cache.get(station, init_a)
    summary = cache.prewarm(station, [init_a, init_b], workers=2)

    assert source.called_inits == [init_a, init_b]
    assert summary["requested_inits"] == 2
    assert summary["skipped_cached"] == 1
    assert summary["fetched_inits"] == 1
    assert summary["rows_written"] == 48
    assert summary["errors"] == 0


def test_decoder_cache_prewarm_persists_multiple_missing_inits(tmp_path):
    source = FakeDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=source)
    station = STATIONS["KLGA"]
    init_a = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    init_b = pd.Timestamp("2026-01-01 01:00:00", tz="UTC")

    summary = cache.prewarm(station, [init_a, init_b], workers=2)

    stored = pd.read_parquet(cache.path_for("KLGA", init_a))
    assert source.called_inits == [init_a, init_b]
    assert summary["fetched_inits"] == 2
    assert summary["rows_written"] == 96
    assert len(stored) == 96
    assert set(pd.to_datetime(stored["init_time"], utc=True)) == {init_a, init_b}


def test_decoder_cache_prewarm_reports_fallback_success_and_elapsed_phases(tmp_path):
    primary = FailingDecoderSource()
    fallback = FakeDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=primary, fallback_source=fallback)
    station = STATIONS["KLGA"]
    init = pd.Timestamp("2026-01-01 23:00:00", tz="UTC")

    summary = cache.prewarm(station, [init], workers=1)

    assert summary["fetched_inits"] == 1
    assert summary["zarr_success_inits"] == 0
    assert summary["fallback_success_inits"] == 1
    assert summary["fallback_failed_inits"] == 0
    assert summary["warm_elapsed_seconds"] >= 0.0
    assert summary["fetch_elapsed_seconds"] >= 0.0
    assert summary["slow_init_samples"] == []


def test_decoder_cache_prewarm_warms_major_dependencies_for_offhour(tmp_path):
    source = FakeCachedDecoderSource()
    cache = TrainingDecoderCache(tmp_path, source=source)
    station = STATIONS["KLGA"]
    offhour = pd.Timestamp("2026-01-01 03:00:00", tz="UTC")

    summary = cache.prewarm(station, [offhour], workers=1)

    assert source.called_inits == [
        pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
        offhour,
    ]
    assert summary["warm_requested_inits"] == 1
    assert summary["fetched_inits"] == 1
    assert summary["rows_written"] == 48


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
                "init_time": vt - pd.Timedelta(hours=1),
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
