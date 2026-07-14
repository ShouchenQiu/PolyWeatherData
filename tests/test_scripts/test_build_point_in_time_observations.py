from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_point_in_time_observations import main


def test_observation_builder_writes_one_immutable_artifact(tmp_path) -> None:
    source = tmp_path / "iem.parquet"
    pd.DataFrame(
        {
            "valid_time": pd.to_datetime(["2025-04-01T00:00:00Z"]),
            "raw_tmpc_tenths": [12.3],
        }
    ).to_parquet(source, index=False)
    output = tmp_path / "observations"

    assert main(["--input", str(source), "--output", str(output)]) == 0
    assert (output / "manifest.json").is_file()
    with pytest.raises(FileExistsError, match="immutable"):
        main(["--input", str(source), "--output", str(output)])
