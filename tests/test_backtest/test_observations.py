from __future__ import annotations

import pandas as pd

from backtest.observations import PointInTimeObservationDataset


def test_point_in_time_observations_round_trip_with_visibility_contract(
    tmp_path,
) -> None:
    dataset = PointInTimeObservationDataset(
        observations=pd.DataFrame(
            [
                {
                    "valid_time": "2026-04-01T04:00:00Z",
                    "available_at": "2026-04-01T04:00:00Z",
                    "temperature_c": 20.0,
                }
            ]
        ),
        station_id="KLGA",
        station_timezone="America/New_York",
    )

    manifest = dataset.save(
        tmp_path / "observations",
        source_sha256={"obs_payload": "a" * 64},
    )
    loaded = PointInTimeObservationDataset.load(
        tmp_path / "observations"
    )

    assert manifest["data_class"] == "point_in_time_hourly_observations"
    assert loaded.station_id == "KLGA"
    assert loaded.observations.iloc[0]["available_at"] == pd.Timestamp(
        "2026-04-01T04:00:00Z"
    )
