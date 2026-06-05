from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd

from data.storage.parquet_store import ParquetStore
from db.schema import DbClient
from ui.pages import data_health


def test_data_health_renders_parquet_store_summary(tmp_path):
    hrrr_store = ParquetStore(tmp_path / "data_store" / "hrrr")
    obs_store = ParquetStore(tmp_path / "data_store" / "obs")
    hrrr_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "init_time": pd.Timestamp("2026-06-04 00:00:00", tz="UTC"),
                    "fxx": 1,
                    "valid_time": pd.Timestamp("2026-06-04 01:00:00", tz="UTC"),
                    "TMP_2m": 20.0,
                }
            ]
        ),
    )
    obs_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "valid_time": pd.Timestamp("2026-06-04 01:00:00", tz="UTC"),
                    "raw_tmpc_tenths": 21.0,
                }
            ]
        ),
    )
    db = DbClient(":memory:")

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.header"))
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.info"))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.plotly_chart"))

        data_health.render(db, data_store_dir=tmp_path / "data_store")

    rendered = dataframe.call_args_list[0].args[0]
    assert set(rendered["store"]) == {"hrrr", "obs"}
    assert rendered["rows"].tolist() == [1, 1]


def test_data_health_renders_daily_coverage_cleaning_and_seed_timing(tmp_path):
    root = tmp_path / "data_store"
    hrrr_store = ParquetStore(root / "hrrr")
    obs_store = ParquetStore(root / "obs")
    day = pd.Timestamp("2026-06-04", tz="UTC")
    hrrr_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "init_time": day + pd.Timedelta(hours=i),
                    "fxx": 1,
                    "valid_time": day + pd.Timedelta(hours=i + 1),
                    "TMP_2m": 20.0 + i,
                    "DPT_2m": 10.0,
                }
                for i in range(24)
            ]
        ),
    )
    obs_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "valid_time": day + pd.Timedelta(hours=i),
                    "raw_tmpc_tenths": 20.0,
                    "raw_dwpc_tenths": 10.0,
                }
                for i in range(25)
            ]
        ),
    )
    (tmp_path / "seed_summary.json").write_text(
        """
        {
          "seed_date": "2026-06-04",
          "timings": {
            "hrrr": {"elapsed_seconds": 10, "rows_written": 24, "errors": 0, "rows_per_second": 2.4},
            "iem": {"elapsed_seconds": 2, "rows_written": 24, "errors": 0, "rows_per_second": 12},
            "clob": {"elapsed_seconds": 1, "rows_written": 22, "errors": 0, "rows_per_second": 22}
          }
        }
        """,
        encoding="ascii",
    )
    db = DbClient(":memory:")

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.header"))
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.info"))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.plotly_chart"))

        data_health.render(db, data_store_dir=root)

    rendered_tables = [call.args[0] for call in dataframe.call_args_list]
    coverage = next(table for table in rendered_tables if "expected_rows" in table.columns)
    assert coverage.loc[coverage["store"] == "hrrr", "expected_rows"].iloc[0] == 24
    assert coverage.loc[coverage["store"] == "hrrr", "missing_rows"].iloc[0] == 0
    assert coverage.loc[coverage["store"] == "obs", "actual_rows"].iloc[0] == 24
    checks = next(table for table in rendered_tables if "check" in table.columns)
    assert "hrrr_fxx_all_one" in checks["check"].tolist()
    timings = next(table for table in rendered_tables if "rows_per_second" in table.columns)
    assert timings.loc[timings["stage"] == "clob", "rows_written"].iloc[0] == 22
