from __future__ import annotations

import numpy as np
import pandas as pd

from data.clob.history import HistoricalClobDataset
from data.forecast.availability import HrrrAvailabilityDataset
from model.mc_calibration import McCalibration
from scripts.build_fixed_forecast_cache import main


def test_fixed_cache_binds_april_calibration_and_split_role(tmp_path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    init = pd.Timestamp("2026-06-05T00:00:00Z")
    predictions = tmp_path / "predictions.parquet"
    pd.DataFrame(
        [
            {
                "init_time": init,
                "valid_time": valid_time,
                "fxx": 4 + offset,
                "target": 0.0,
                "p10": -1.0,
                "p50": 0.0,
                "p90": 1.0,
                "base_temp_c": 21.0,
                "weight": 0.0,
            }
            for offset, valid_time in enumerate(
                pd.date_range(
                    "2026-06-05T04:00:00Z",
                    periods=24,
                    freq="h",
                )
            )
        ]
    ).to_parquet(predictions, index=False)
    calibration_root = tmp_path / "calibration"
    McCalibration(
        correlation=np.eye(48),
        sigma_scales=np.ones(5),
        max_bias_f=0.0,
        model_sha256=McCalibration.sha256(checkpoint),
        fit_split="val",
        fit_start="2026-04-01",
        fit_end="2026-04-30",
        sample_count=720,
        station_timezone="America/New_York",
    ).save(calibration_root)
    availability_root = tmp_path / "availability"
    HrrrAvailabilityDataset(
        objects=pd.DataFrame(columns=["key", "last_modified"]),
        cycles=pd.DataFrame(
            [
                {
                    "init_time": init,
                    "complete": True,
                    "available_at": init + pd.Timedelta(hours=1),
                    "missing_objects": [],
                }
            ]
        ),
        start_date="2026-06-05",
        end_date="2026-06-05",
        compute_budget="5min",
        decision_tick="5min",
    ).save(availability_root)
    clob_root = tmp_path / "clob"
    HistoricalClobDataset(
        events=pd.DataFrame(
            [{"event_id": "event", "target_date": "2026-06-05"}]
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
            [{"condition_id": "condition", "timestamp": init}]
        ),
        start_date="2026-06-05",
        end_date="2026-06-05",
        station_id="KLGA",
    ).save(clob_root)
    output = tmp_path / "cache"

    result = main(
        [
            "--predictions",
            str(predictions),
            "--checkpoint",
            str(checkpoint),
            "--calibration-root",
            str(calibration_root),
            "--availability-root",
            str(availability_root),
            "--clob-root",
            str(clob_root),
            "--output",
            str(output),
            "--split-role",
            "post_test_holdout",
            "--start-init",
            "2026-06-05",
            "--end-init",
            "2026-06-05",
            "--mc-samples",
            "100",
        ]
    )

    lineages = pd.read_parquet(output / "lineages.parquet")
    assert result == 0
    assert lineages["split_role"].tolist() == ["post_test_holdout"]
    assert lineages["checkpoint_sha256"].tolist() == [
        McCalibration.sha256(checkpoint)
    ]
