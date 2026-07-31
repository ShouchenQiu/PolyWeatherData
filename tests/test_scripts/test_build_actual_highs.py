from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.build_actual_highs import main


def test_actual_highs_cli_aggregates_whole_degree_truth(tmp_path) -> None:
    truth = tmp_path / "truth" / "KLGA"
    truth.mkdir(parents=True)
    valid_hours = pd.date_range(
        "2026-06-05T04:00:00Z",
        periods=24,
        freq="h",
    )
    pd.DataFrame(
        {
            "valid_time": valid_hours + pd.Timedelta(minutes=51),
            "valid_hour": valid_hours + pd.Timedelta(hours=1),
            "weather_com_temp_f": [70.0, 73.0, 71.0] + [65.0] * 21,
        }
    ).to_parquet(truth / "2026-06-05.parquet", index=False)
    output = tmp_path / "actual.csv"

    result = main(
        [
            "--truth-store",
            str(tmp_path / "truth"),
            "--station",
            "KLGA",
            "--start",
            "2026-06-05",
            "--end",
            "2026-06-05",
            "--output",
            str(output),
        ]
    )

    actual = pd.read_csv(output)
    manifest = json.loads(
        (tmp_path / "actual.csv.manifest.json").read_text(encoding="utf-8")
    )
    assert result == 0
    assert actual.to_dict("records") == [
        {"target_date": "2026-06-05", "actual_high_f": 73.0}
    ]
    assert manifest["row_count"] == 1


def test_actual_highs_cli_rejects_incomplete_dst_local_day(tmp_path) -> None:
    truth = tmp_path / "truth" / "KLGA"
    truth.mkdir(parents=True)
    pd.DataFrame(
        {
            "valid_time": pd.to_datetime(
                ["2026-03-08T05:51:00Z", "2026-03-08T06:51:00Z"],
                utc=True,
            ),
            "valid_hour": pd.to_datetime(
                ["2026-03-08T06:00:00Z", "2026-03-08T07:00:00Z"],
                utc=True,
            ),
            "weather_com_temp_f": [48.0, 47.0],
        }
    ).to_parquet(truth / "2026-03-08.parquet", index=False)

    with pytest.raises(
        ValueError,
        match=r"incomplete Weather\.com local day.*observed_hours=2.*expected_hours=23",
    ):
        main(
            [
                "--truth-store",
                str(tmp_path / "truth"),
                "--station",
                "KLGA",
                "--start",
                "2026-03-08",
                "--end",
                "2026-03-08",
                "--output",
                str(tmp_path / "actual.csv"),
            ]
        )


def test_actual_highs_cli_accepts_complete_spring_forward_day(tmp_path) -> None:
    truth = tmp_path / "truth" / "KLGA"
    truth.mkdir(parents=True)
    utc_hours = pd.date_range(
        "2026-03-08T05:00:00Z",
        periods=23,
        freq="h",
    )
    pd.DataFrame(
        {
            "valid_time": utc_hours + pd.Timedelta(minutes=51),
            "valid_hour": utc_hours + pd.Timedelta(hours=1),
            "weather_com_temp_f": [48.0] * 12 + [68.0] + [55.0] * 10,
        }
    ).to_parquet(truth / "2026-03-08.parquet", index=False)
    output = tmp_path / "actual.csv"

    assert main(
        [
            "--truth-store",
            str(tmp_path / "truth"),
            "--station",
            "KLGA",
            "--start",
            "2026-03-08",
            "--end",
            "2026-03-08",
            "--output",
            str(output),
        ]
    ) == 0

    assert pd.read_csv(output).to_dict("records") == [
        {"target_date": "2026-03-08", "actual_high_f": 68.0}
    ]
