from __future__ import annotations

import pandas as pd

from scripts.build_hrrr_availability import main


class _Source:
    def __init__(self) -> None:
        self.days: list[str] = []

    def fetch_day(self, day: str) -> pd.DataFrame:
        self.days.append(str(day))
        return pd.DataFrame(columns=["key", "last_modified"])


def test_build_hrrr_availability_cli_includes_prior_dependency_day(
    tmp_path,
) -> None:
    source = _Source()
    output = tmp_path / "availability"

    result = main(
        [
            "--start",
            "2026-06-30",
            "--end",
            "2026-06-30",
            "--output",
            str(output),
        ],
        source=source,
    )

    assert result == 0
    assert source.days == ["2026-06-29", "2026-06-30"]
    assert (output / "manifest.json").exists()
    cycles = pd.read_parquet(output / "cycles.parquet")
    assert len(cycles) == 24
    assert not cycles["complete"].any()
