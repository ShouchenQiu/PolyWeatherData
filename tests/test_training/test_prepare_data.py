import json

from training.prepare_data import prepare_training_data


class FakeDownloader:
    calls: list[tuple] = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def run(self, start, end, *args, **kwargs):
        self.calls.append((start, end, args, kwargs))
        return 24


class FakeDecoderCache:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def prewarm(self, station, init_times, workers, **_kwargs):
        return {
            "requested_inits": len(init_times),
            "skipped_cached": 0,
            "fetched_inits": len(init_times),
            "rows_written": len(init_times) * 48,
            "errors": 0,
            "elapsed_seconds": 1.0,
            "init_per_second": len(init_times),
            "major_inits": 4,
            "offhour_inits": len(init_times) - 4,
        }


def test_prepare_data_runs_weather_stages_and_records_workspace(tmp_path) -> None:
    FakeDownloader.calls = []
    run_dir = tmp_path / "run"

    result = prepare_training_data(
        run_dir,
        station_id="KLGA",
        start_date="2026-01-01",
        end_date="2026-01-01",
        workers=3,
        f01_downloader_cls=FakeDownloader,
        iem_downloader_cls=FakeDownloader,
        decoder_cache_cls=FakeDecoderCache,
    )

    assert result["status"] == "prepared"
    assert result["hrrr_f01_written"] == 24
    assert result["obs_written"] == 24
    assert result["decoder"]["rows_written"] == 24 * 48
    assert (run_dir / "prepare_data_summary.json").exists()
    state = json.loads((run_dir / "stage_state.json").read_text())
    assert state["stages"]["prepare"]["status"] == "prepared"
