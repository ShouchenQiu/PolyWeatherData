from __future__ import annotations

import pandas as pd

from data.storage.parquet_store import ParquetStore
from model.features import TFT_KNOWN_REALS
from scripts.audit_weather_backfill import audit_weather_backfill
from scripts.audit_weather_backfill import audit_training_readiness


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
                row.update(
                    {
                        "init_time_str": str(init_time),
                        "Node_ID": station.station_id,
                        "valid_time": init_time + pd.Timedelta(hours=i),
                        "local_idx": i,
                        "time_idx": i,
                        "fxx": max(1, i - 71),
                        "temp_error": 0.1,
                        "raw_tmpc_tenths": 1.0 if i < 72 else None,
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
