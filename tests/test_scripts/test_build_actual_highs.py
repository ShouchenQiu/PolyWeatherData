from __future__ import annotations

import json

import pandas as pd

from scripts.build_actual_highs import main


def test_actual_highs_cli_aggregates_whole_degree_truth(tmp_path) -> None:
    truth = tmp_path / "truth" / "KLGA"
    truth.mkdir(parents=True)
    pd.DataFrame(
        {
            "weather_com_temp_f": [70.0, 73.0, 71.0],
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
