from __future__ import annotations

import pandas as pd

from data.storage.parquet_store import ParquetStore
from model.features import TFT_KNOWN_REALS, TFT_UNKNOWN_REALS
from training.audit import (
    audit_training_readiness,
    audit_weather_backfill,
    audit_workspace,
)


def test_audit_weather_backfill_reports_store_and_decoder_gaps(tmp_path):
    data_store = tmp_path / "data_store"
    forecast_store = ParquetStore(data_store / "hrrr")
    obs_store = ParquetStore(data_store / "obs")
    hours = list(pd.date_range("2026-01-01 00:00:00", periods=3, freq="h", tz="UTC"))

    forecast_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "init_time": hours[0] - pd.Timedelta(hours=1),
                    "fxx": 1,
                    "valid_time": hours[0],
                    "TMP_2m": 273.15,
                },
                {
                    "init_time": hours[2] - pd.Timedelta(hours=1),
                    "fxx": 1,
                    "valid_time": hours[2],
                    "TMP_2m": 273.15,
                },
            ]
        ),
    )
    obs_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {"valid_time": hours[0], "raw_tmpc_tenths": 1.0},
                {"valid_time": hours[1], "raw_tmpc_tenths": 1.0},
            ]
        ),
    )

    decoder_dir = data_store / "_decoder_cache" / "KLGA"
    decoder_dir.mkdir(parents=True)
    rows = []
    for fxx in range(1, 49):
        rows.append(
            {
                "init_time": hours[0],
                "fxx": fxx,
                "valid_time": hours[0] + pd.Timedelta(hours=fxx),
                "TMP_2m": 273.15,
            }
        )
    for fxx in range(1, 10):
        rows.append(
            {
                "init_time": hours[1],
                "fxx": fxx,
                "valid_time": hours[1] + pd.Timedelta(hours=fxx),
                "TMP_2m": 273.15,
            }
        )
    pd.DataFrame(rows).to_parquet(decoder_dir / "2026-01.parquet", index=False)

    summary = audit_weather_backfill(data_store, "KLGA", "2026-01-01", "2026-01-01")

    assert summary["expected_hours"] == 24
    assert summary["hrrr_f01"]["complete_hours"] == 2
    assert summary["obs"]["complete_hours"] == 2
    assert summary["decoder"]["complete_inits"] == 1
    assert summary["decoder"]["partial_inits"] == 1
    assert summary["decoder"]["partial_init_times"][str(hours[1])] == 9


class FakeBuilder:
    def build_frame(self, station, init_times, split):
        rows = []
        for init_time in init_times:
            for i in range(120):
                row = {col: 1.0 for col in TFT_KNOWN_REALS}
                row.update({col: 1.0 for col in TFT_UNKNOWN_REALS})
                row.update(
                    {
                        "init_time_str": str(init_time),
                        "Node_ID": station.station_id,
                        "valid_time": init_time + pd.Timedelta(hours=i),
                        "local_idx": i,
                        "time_idx": i,
                        "fxx": max(1, i - 71),
                        "temp_error": 0.1,
                        "decoder_weight": 1.0,
                        "split": split,
                    }
                )
                rows.append(row)
        return pd.DataFrame(rows)


def test_audit_training_readiness_reports_hash_samples_and_feature_order(tmp_path):
    summary = audit_training_readiness(
        data_store=tmp_path / "data_store",
        station_id="KLGA",
        split_ranges={
            "train": ("2026-01-01", "2026-01-01"),
            "val": ("2026-01-02", "2026-01-02"),
            "test": ("2026-01-03", "2026-01-03"),
        },
        dataset_builder=FakeBuilder(),
    )

    assert summary["ready"] is True
    assert summary["splits"]["train"]["sample_count"] == 24
    assert summary["splits"]["train"]["row_count"] == 24 * 120
    assert summary["splits"]["train"]["init_row_count_ok"] is True
    assert summary["data_order_hash"]
    assert "TMP_2m" in summary["feature_list_and_order"]["known_reals"]
    assert summary["feature_list_and_order"]["target"] == "temp_error"


def test_audit_training_readiness_streams_split_progress(tmp_path, capsys):
    class ProgressBuilder(FakeBuilder):
        def build_frame(
            self,
            station,
            init_times,
            split,
            progress_callback=None,
            progress_every=100,
        ):
            if progress_callback is not None:
                progress_callback(1, len(init_times), 1, 0)
                progress_callback(
                    len(init_times),
                    len(init_times),
                    len(init_times),
                    0,
                )
            return super().build_frame(station, init_times, split)

    summary = audit_training_readiness(
        data_store=tmp_path / "data_store",
        station_id="KLGA",
        split_ranges={
            "train": ("2026-01-01", "2026-01-01"),
            "val": ("2026-01-02", "2026-01-02"),
            "test": ("2026-01-03", "2026-01-03"),
        },
        dataset_builder=ProgressBuilder(),
        progress=True,
        progress_every=10,
    )

    output = capsys.readouterr().out
    assert "[readiness train] start inits=24" in output
    assert "[readiness train] 24/24 built=24 skipped=0" in output
    assert "[readiness test] complete rows=2880 samples=24" in output
    assert "[readiness] complete ready=True" in output
    assert summary["ready"] is True


def test_audit_training_readiness_rejects_nan_encoder_unknown_real(tmp_path):
    class NaNUnknownBuilder(FakeBuilder):
        def build_frame(self, station, init_times, split):
            frame = super().build_frame(station, init_times, split)
            frame.loc[
                (frame["init_time_str"] == str(init_times[0]))
                & (frame["local_idx"] == 3),
                "raw_vis_sm",
            ] = float("nan")
            return frame

    summary = audit_training_readiness(
        data_store=tmp_path / "data_store",
        station_id="KLGA",
        split_ranges={
            "train": ("2026-01-01", "2026-01-01"),
            "val": ("2026-01-02", "2026-01-02"),
            "test": ("2026-01-03", "2026-01-03"),
        },
        dataset_builder=NaNUnknownBuilder(),
    )

    assert summary["ready"] is False
    assert summary["splits"]["train"]["nan_required_columns"] == ["raw_vis_sm"]
    assert "train: NaN in required columns ['raw_vis_sm']" in summary["failures"]


def test_audit_workspace_writes_failed_evidence(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "data_store").mkdir(parents=True)

    result = audit_workspace(
        run_dir,
        "KLGA",
        {
            "train": ("2026-01-01", "2026-01-01"),
            "val": ("2026-01-02", "2026-01-02"),
            "test": ("2026-01-03", "2026-01-03"),
        },
        audit_fn=lambda **_kwargs: {"ready": False, "failures": ["missing"]},
    )

    assert result["status"] == "audit_failed"
    assert (run_dir / "audit_summary.json").exists()
    assert (run_dir / "failed_audit_summary.json").exists()
