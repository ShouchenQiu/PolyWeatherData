from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.observations import PointInTimeObservationDataset
from data.clob.history import HistoricalClobDataset
from data.forecast.availability import HrrrAvailabilityDataset
from model.mc_calibration import McCalibration
from scripts.build_oof_forecast_cache import main


def test_build_oof_forecast_cache_cli_writes_replay_artifact(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    predictions = tmp_path / "predictions.parquet"
    rows = []
    for init, first_fxx in [
        (pd.Timestamp("2026-06-05T00:00:00Z"), 5),
        (pd.Timestamp("2026-06-05T01:00:00Z"), 4),
    ]:
        for offset, valid_time in enumerate(
            pd.date_range("2026-06-05T05:00:00Z", periods=24, freq="h")
        ):
            rows.append(
                {
                    "init_time": init,
                    "valid_time": valid_time,
                    "fxx": first_fxx + offset,
                    "target": 0.0,
                    "p10": -1.0,
                    "p50": 0.0,
                    "p90": 1.0,
                    "base_temp_c": 21.0,
                    "weight": 1.0,
                }
            )
    pd.DataFrame(rows).to_parquet(predictions, index=False)

    availability_root = tmp_path / "availability"
    cycles = pd.DataFrame(
        [
            {
                "init_time": init,
                "complete": True,
                "available_at": init + pd.Timedelta(hours=1),
                "missing_objects": [],
            }
            for init in pd.to_datetime(
                ["2026-06-05T00:00:00Z", "2026-06-05T01:00:00Z"],
                utc=True,
            )
        ]
    )
    HrrrAvailabilityDataset(
        objects=pd.DataFrame(columns=["key", "last_modified"]),
        cycles=cycles,
        start_date="2026-06-05",
        end_date="2026-06-05",
        compute_budget="5min",
        decision_tick="5min",
    ).save(availability_root)

    clob_root = tmp_path / "clob"
    HistoricalClobDataset(
        events=pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "target_date": "2026-06-05",
                    "event_end_at": "2026-06-06T12:00:00Z",
                }
            ]
        ),
        markets=pd.DataFrame(
            [
                {
                    "event_id": "event",
                    "condition_id": "condition",
                    "bucket_lo_f": float("-inf"),
                    "bucket_hi_f": float("inf"),
                    "yes_token_id": "yes",
                    "no_token_id": "no",
                }
            ]
        ),
        trades=pd.DataFrame(
            [{"condition_id": "condition", "timestamp": pd.Timestamp("2026-06-05", tz="UTC")}]
        ),
        start_date="2026-06-05",
        end_date="2026-06-05",
        station_id="KLGA",
    ).save(clob_root)
    observations_root = tmp_path / "observations"
    PointInTimeObservationDataset(
        observations=pd.DataFrame(
            {
                "valid_time": pd.date_range(
                    "2026-06-05T05:00:00Z",
                    periods=24,
                    freq="h",
                ),
                "available_at": pd.date_range(
                    "2026-06-05T05:00:00Z",
                    periods=24,
                    freq="h",
                ),
                "temperature_c": [21.0] * 24,
            }
        ),
        station_id="KLGA",
        station_timezone="America/New_York",
    ).save(observations_root, source_sha256={"fixture": "a" * 64})

    def fake_fit(frame, checkpoint_path, *, station_timezone, mc_samples):
        return McCalibration(
            correlation=np.eye(48),
            sigma_scales=np.ones(5),
            max_bias_f=0.0,
            model_sha256=McCalibration.sha256(checkpoint_path),
            fit_split="val",
            fit_start="2026-06-05",
            fit_end="2026-06-05",
            sample_count=1,
            station_timezone=station_timezone,
        )

    output = tmp_path / "forecast-cache"
    result = main(
        [
            "--predictions",
            str(predictions),
            "--checkpoint",
            str(checkpoint),
            "--availability-root",
            str(availability_root),
            "--observations-root",
            str(observations_root),
            "--clob-root",
            str(clob_root),
            "--output",
            str(output),
            "--folds",
            "2",
            "--mc-samples",
            "100",
            "--target-start",
            "2026-06-05",
            "--target-end",
            "2026-06-05",
        ],
        fit_calibration=fake_fit,
    )

    assert result == 0
    assert (output / "manifest.json").exists()
    snapshots = pd.read_parquet(output / "snapshots.parquet")
    assert len(snapshots) == 2
