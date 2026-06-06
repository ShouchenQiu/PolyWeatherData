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


def test_temperature_comparison_frame_aligns_hrrr_f01_and_iem_obs(tmp_path):
    root = tmp_path / "data_store"
    hrrr_store = ParquetStore(root / "hrrr")
    obs_store = ParquetStore(root / "obs")
    hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    hrrr_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "init_time": hour - pd.Timedelta(hours=1),
                    "fxx": 1,
                    "valid_time": hour,
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
                    "valid_time": hour + pd.Timedelta(minutes=5),
                    "raw_tmpc_tenths": 19.0,
                },
                {
                    "valid_time": hour + pd.Timedelta(minutes=51),
                    "raw_tmpc_tenths": 21.5,
                },
            ]
        ),
    )

    frame = data_health._temperature_comparison_frame(root, "KLGA")

    assert frame["valid_hour"].tolist() == [hour]
    assert frame["hrrr_f01_c"].tolist() == [20.0]
    assert frame["iem_obs_c"].tolist() == [21.5]
    assert frame["temp_error_c"].tolist() == [-1.5]
    assert frame["has_hrrr"].tolist() == [True]
    assert frame["has_obs"].tolist() == [True]


def test_temperature_comparison_frame_empty_store_has_fixed_columns(tmp_path):
    frame = data_health._temperature_comparison_frame(tmp_path / "data_store", "KLGA")

    assert frame.empty
    assert frame.columns.tolist() == [
        "station_id",
        "valid_hour",
        "hrrr_f01_c",
        "iem_obs_c",
        "temp_error_c",
        "has_hrrr",
        "has_obs",
    ]


def test_data_health_renders_temperature_comparison_chart(tmp_path):
    root = tmp_path / "data_store"
    hrrr_store = ParquetStore(root / "hrrr")
    obs_store = ParquetStore(root / "obs")
    hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    hrrr_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "init_time": hour - pd.Timedelta(hours=1),
                    "fxx": 1,
                    "valid_time": hour,
                    "TMP_2m": 20.0,
                }
            ]
        ),
    )
    obs_store.append(
        "KLGA",
        pd.DataFrame(
            [{"valid_time": hour, "raw_tmpc_tenths": 21.0}]
        ),
    )
    db = DbClient(":memory:")

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.header"))
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.info"))
        stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.selectbox", side_effect=["KLGA", "All"]))
        stack.enter_context(patch("streamlit.columns", return_value=[_MetricCol(), _MetricCol(), _MetricCol(), _MetricCol()]))
        plotly_chart = stack.enter_context(patch("streamlit.plotly_chart"))

        data_health.render(db, data_store_dir=root)

    assert plotly_chart.call_count >= 1


def test_hrrr_48h_verification_frame_aligns_decoder_cache_and_obs(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)
    init = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    pd.DataFrame(
        [
            {
                "init_time": init,
                "fxx": fxx,
                "valid_time": init + pd.Timedelta(hours=fxx),
                "TMP_2m": 10.0 + fxx,
            }
            for fxx in range(1, 49)
        ]
    ).to_parquet(cache_dir / "2026-06.parquet", index=False)
    obs_store = ParquetStore(root / "obs")
    obs_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "valid_time": init + pd.Timedelta(hours=fxx),
                    "raw_tmpc_tenths": 9.0 + fxx,
                }
                for fxx in range(1, 49)
            ]
        ),
    )

    frame = data_health._hrrr_48h_verification_frame(root, "KLGA", init_limit=None)

    assert len(frame) == 48
    assert frame["fxx"].tolist() == list(range(1, 49))
    assert frame["forecast_c"].iloc[0] == 11.0
    assert frame["observed_c"].iloc[0] == 10.0
    assert frame["error_c"].iloc[0] == 1.0
    assert frame["abs_error_c"].iloc[0] == 1.0
    assert frame["has_obs"].all()


def test_hrrr_48h_verification_frame_keeps_forecast_rows_without_obs(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)
    init = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    pd.DataFrame(
        [
            {
                "init_time": init,
                "fxx": fxx,
                "valid_time": init + pd.Timedelta(hours=fxx),
                "TMP_2m": 20.0,
            }
            for fxx in range(1, 49)
        ]
    ).to_parquet(cache_dir / "2026-06.parquet", index=False)

    frame = data_health._hrrr_48h_verification_frame(root, "KLGA", init_limit=None)

    assert len(frame) == 48
    assert not frame["has_obs"].any()
    assert frame["observed_c"].isna().all()
    assert frame["error_c"].isna().all()


def test_hrrr_48h_verification_frame_defaults_to_latest_48_inits(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)
    base = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    rows = []
    for i in range(50):
        init = base + pd.Timedelta(hours=i)
        for fxx in range(1, 49):
            rows.append(
                {
                    "init_time": init,
                    "fxx": fxx,
                    "valid_time": init + pd.Timedelta(hours=fxx),
                    "TMP_2m": 20.0,
                }
            )
    pd.DataFrame(rows).to_parquet(cache_dir / "2026-06.parquet", index=False)

    frame = data_health._hrrr_48h_verification_frame(root, "KLGA")

    inits = pd.to_datetime(frame["init_time"], utc=True).drop_duplicates().tolist()
    assert len(inits) == 48
    assert min(inits) == base + pd.Timedelta(hours=2)
    assert max(inits) == base + pd.Timedelta(hours=49)


def test_data_health_renders_hrrr_48h_verification_chart(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)
    init = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    pd.DataFrame(
        [
            {
                "init_time": init,
                "fxx": fxx,
                "valid_time": init + pd.Timedelta(hours=fxx),
                "TMP_2m": 15.0 + fxx,
            }
            for fxx in range(1, 49)
        ]
    ).to_parquet(cache_dir / "2026-06.parquet", index=False)
    obs_store = ParquetStore(root / "obs")
    obs_store.append(
        "KLGA",
        pd.DataFrame(
            [
                {
                    "valid_time": init + pd.Timedelta(hours=fxx),
                    "raw_tmpc_tenths": 14.0 + fxx,
                }
                for fxx in range(1, 49)
            ]
        ),
    )
    db = DbClient(":memory:")

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.header"))
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.info"))
        stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.selectbox", side_effect=["KLGA", "All", "KLGA", "Last 48", str(init)]))
        stack.enter_context(patch("streamlit.columns", return_value=[_MetricCol(), _MetricCol(), _MetricCol(), _MetricCol()]))
        plotly_chart = stack.enter_context(patch("streamlit.plotly_chart"))

        data_health.render(db, data_store_dir=root)

    assert plotly_chart.call_count >= 2


class _MetricCol:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def metric(self, *_args, **_kwargs):
        return None
