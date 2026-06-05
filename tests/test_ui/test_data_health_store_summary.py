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
