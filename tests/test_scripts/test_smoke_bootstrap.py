from __future__ import annotations

import json

import pytest

from scripts import batch_prepare_training_data
from scripts import smoke_bootstrap


class FakeF01Downloader:
    calls = []

    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, start_date, end_date, init_hours=None):
        FakeF01Downloader.calls.append((start_date, end_date, init_hours))
        return 384


class FakeIemDownloader:
    calls = []

    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, start_date, end_date, station="LGA", network="NY_ASOS"):
        FakeIemDownloader.calls.append((start_date, end_date, station, network))
        return 384


class FakeDecoderCache:
    calls = []

    def __init__(self, *_args, **_kwargs):
        pass

    def prewarm(self, station, init_times, workers=12, **_kwargs):
        FakeDecoderCache.calls.append((station.station_id, list(init_times), workers))
        return {
            "requested_inits": len(init_times),
            "skipped_cached": 0,
            "fetched_inits": len(init_times),
            "rows_written": len(init_times) * 48,
            "errors": 0,
            "elapsed_seconds": 1.0,
            "init_per_second": len(init_times),
            "major_inits": 64,
            "offhour_inits": len(init_times) - 64,
        }


class InterruptingDecoderCache:
    def __init__(self, *_args, **_kwargs):
        pass

    def prewarm(self, station, init_times, workers=12, **_kwargs):
        raise KeyboardInterrupt()


def test_smoke_bootstrap_dry_run_reports_fixed_range(tmp_path):
    summary = batch_prepare_training_data.run_batch_prepare_training_data(
        run_dir=tmp_path / "smoke",
        dry_run=True,
    )

    assert summary["start_date"] == "2026-05-20"
    assert summary["end_date"] == "2026-06-04"
    assert summary["init_count"] == 384
    assert summary["decoder_init_count"] == 384
    assert summary["decoder_skipped_tail_cap"] == 0
    assert summary["smoke_split"] == {
        "train": ["2026-05-26", "2026-05-30"],
        "val": ["2026-05-31", "2026-06-01"],
        "test": ["2026-06-02", "2026-06-04"],
    }
    assert not (tmp_path / "smoke").exists()


def test_smoke_bootstrap_calls_bootstrap_and_decoder_steps(tmp_path):
    FakeF01Downloader.calls = []
    FakeIemDownloader.calls = []
    FakeDecoderCache.calls = []

    summary = batch_prepare_training_data.run_batch_prepare_training_data(
        run_dir=tmp_path / "smoke",
        f01_downloader_cls=FakeF01Downloader,
        iem_downloader_cls=FakeIemDownloader,
        decoder_cache_cls=FakeDecoderCache,
        skip_polymarket=True,
        workers=3,
    )

    assert FakeF01Downloader.calls == [("2026-05-20", "2026-06-04", None)]
    assert FakeIemDownloader.calls == [("2026-05-20", "2026-06-04", "LGA", "NY_ASOS")]
    assert len(FakeDecoderCache.calls[0][1]) == 384
    assert FakeDecoderCache.calls[0][2] == 3
    assert summary["hrrr_f01_written"] == 384
    assert summary["obs_written"] == 384
    assert summary["decoder"]["rows_written"] == 384 * 48
    assert summary["decoder"]["skipped_tail_cap"] == 0
    assert (tmp_path / "smoke" / "smoke_summary.json").exists()


def test_smoke_bootstrap_writes_partial_summary_on_decoder_interrupt(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        batch_prepare_training_data.run_batch_prepare_training_data(
            run_dir=tmp_path / "smoke",
            f01_downloader_cls=FakeF01Downloader,
            iem_downloader_cls=FakeIemDownloader,
            decoder_cache_cls=InterruptingDecoderCache,
            skip_polymarket=True,
            workers=3,
        )

    partial_path = tmp_path / "smoke" / "smoke_summary.partial.json"
    assert partial_path.exists()
    partial = json.loads(partial_path.read_text(encoding="ascii"))
    assert partial["interrupted"] is True
    assert partial["phase"] == "decoder"
    assert partial["decoder"]["requested_inits"] == 384
    assert partial["decoder"]["complete_inits"] == 0


def test_legacy_smoke_bootstrap_module_reexports_batch_prepare_training_data():
    assert smoke_bootstrap.run_smoke_bootstrap is batch_prepare_training_data.run_smoke_bootstrap
    assert (
        smoke_bootstrap.run_batch_prepare_training_data
        is batch_prepare_training_data.run_batch_prepare_training_data
    )
