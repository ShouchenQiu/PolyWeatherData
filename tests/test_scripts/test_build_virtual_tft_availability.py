from __future__ import annotations

import json

import pandas as pd
import pytest

from data.forecast.availability import HrrrAvailabilityDataset
from scripts.build_virtual_tft_availability import main


def test_build_virtual_tft_availability_cli_persists_explicit_provider(tmp_path) -> None:
    output = tmp_path / "virtual_availability"

    assert main(
        [
            "--start",
            "2026-04-01",
            "--end",
            "2026-04-01",
            "--output",
            str(output),
        ]
    ) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    loaded = HrrrAvailabilityDataset.load(output)

    assert manifest["provider"] == "virtual_tft_init_plus_fixed_latency"
    assert manifest["row_counts"] == {"objects": 0, "cycles": 24}
    assert loaded.cycles.iloc[0]["available_at"] == pd.Timestamp(
        "2026-04-01T02:00:00Z"
    )

    with pytest.raises(FileExistsError, match="immutable virtual availability"):
        main(
            [
                "--start",
                "2026-04-01",
                "--end",
                "2026-04-01",
                "--output",
                str(output),
            ]
        )
