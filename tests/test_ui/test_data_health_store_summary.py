from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd

from data.storage.parquet_store import ParquetStore
from db.schema import DbClient
from ui.pages import data_health


class _EmptyTruthClient:
    @classmethod
    def location_code(cls, station_id: str) -> str:
        return f"{station_id}:9:US"

    def __init__(self, *_args, **_kwargs) -> None:
        return None

    def fetch_date(self, *_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "station_id",
                "valid_time",
                "valid_hour",
                "weather_com_temp_f",
                "weather_com_temp_c",
            ]
        )


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
        stack.enter_context(patch("ui.pages.data_health.WeatherComTruthClient", _EmptyTruthClient))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.plotly_chart"))

        data_health.render(db, data_store_dir=tmp_path / "data_store")

    rendered_tables = [call.args[0] for call in dataframe.call_args_list]
    rendered = next(table for table in rendered_tables if "store" in table.columns)
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
        stack.enter_context(patch("ui.pages.data_health.WeatherComTruthClient", _EmptyTruthClient))
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
        stack.enter_context(patch("ui.pages.data_health.WeatherComTruthClient", _EmptyTruthClient))
        stack.enter_context(patch("streamlit.columns", return_value=[_MetricCol(), _MetricCol(), _MetricCol(), _MetricCol(), _MetricCol()]))
        plotly_chart = stack.enter_context(patch("streamlit.plotly_chart"))

        data_health.render(db, data_store_dir=root)

    assert plotly_chart.call_count >= 1


def test_data_health_renders_iem_weather_truth_from_cache(tmp_path):
    root = tmp_path / "data_store"
    obs_store = ParquetStore(root / "obs")
    hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    obs_store.append(
        "KLGA",
        pd.DataFrame(
            [{"valid_time": hour, "raw_tmpc_tenths": 20.0}]
        ),
    )
    cache_dir = root / "_truth_weather_com" / "KLGA"
    cache_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_time": hour,
                "valid_hour": hour,
                "weather_com_temp_f": 68.0,
                "weather_com_temp_c": 20.0,
                "_weather_com_alignment_version": "iem_representative_hour_v2",
            }
        ]
    ).to_parquet(cache_dir / "2026-06-03.parquet", index=False)
    db = DbClient(":memory:")

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.header"))
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.info"))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.selectbox", side_effect=["KLGA", "All"]))
        stack.enter_context(patch("streamlit.columns", return_value=[
            _MetricCol(), _MetricCol(), _MetricCol(), _MetricCol(), _MetricCol(),
        ]))
        plotly_chart = stack.enter_context(patch("streamlit.plotly_chart"))

        data_health.render(db, data_store_dir=root)

    rendered_tables = [call.args[0] for call in dataframe.call_args_list]
    truth_table = next(table for table in rendered_tables if "truth_error_f" in table.columns)
    assert truth_table["truth_error_f"].tolist() == [0.0]
    assert plotly_chart.call_count >= 2


def test_temperature_comparison_figures_share_x_domain_and_error_colors():
    hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    matched = pd.DataFrame(
        [
            {
                "valid_hour": hour,
                "hrrr_f01_c": 21.0,
                "iem_obs_c": 20.0,
                "temp_error_c": 1.0,
            },
            {
                "valid_hour": hour + pd.Timedelta(hours=1),
                "hrrr_f01_c": 19.0,
                "iem_obs_c": 20.0,
                "temp_error_c": -1.0,
            },
        ]
    )

    comparison = data_health._temperature_comparison_figure(matched)
    error = data_health._temperature_error_figure(matched)

    assert comparison.layout.xaxis.domain == error.layout.xaxis.domain
    assert comparison.layout.margin.r == error.layout.margin.r
    assert comparison.layout.legend.orientation == "h"
    assert "UTC; NYC local" in comparison.layout.xaxis.title.text
    assert "UTC; NYC local" in error.layout.xaxis.title.text
    assert set(error.data[0].marker.color) == {"#d62728", "#1f77b4"}


def test_iem_weather_truth_figures_share_layout_and_error_colors():
    hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    matched = pd.DataFrame(
        [
            {
                "valid_hour": hour,
                "iem_obs_f": 68.0,
                "weather_com_temp_f": 67.0,
                "truth_error_f": 1.0,
            },
            {
                "valid_hour": hour + pd.Timedelta(hours=1),
                "iem_obs_f": 68.0,
                "weather_com_temp_f": 69.0,
                "truth_error_f": -1.0,
            },
        ]
    )

    comparison = data_health._iem_weather_truth_figure(matched)
    error = data_health._iem_weather_truth_error_figure(matched)

    assert comparison.layout.xaxis.domain == error.layout.xaxis.domain
    assert comparison.layout.margin.r == error.layout.margin.r
    assert comparison.layout.legend.orientation == "h"
    assert "UTC; NYC local" in comparison.layout.xaxis.title.text
    assert set(error.data[0].marker.color) == {"#d62728", "#1f77b4"}


def test_seed_timing_summary_includes_smoke_bootstrap_timing(tmp_path):
    root = tmp_path / "data_store"
    (tmp_path / "smoke_summary.json").write_text(
        """
        {
          "timings": {
            "iem": {"elapsed_seconds": 10, "rows_written": 100, "errors": 0, "rows_per_second": 10},
            "hrrr_f01": {"elapsed_seconds": 20, "rows_written": 40, "errors": 1, "rows_per_second": 2}
          }
        }
        """,
        encoding="ascii",
    )
    (tmp_path / "seed_summary.json").write_text(
        """
        {
          "timings": {
            "hrrr": {"elapsed_seconds": 0, "rows_written": 0, "errors": 0, "rows_per_second": 0}
          }
        }
        """,
        encoding="ascii",
    )

    summary = data_health._seed_timing_summary(root)

    assert set(summary["summary"]) == {"smoke_summary", "seed_summary"}
    smoke_iem = summary[
        (summary["summary"] == "smoke_summary") & (summary["stage"] == "iem")
    ].iloc[0]
    assert smoke_iem["rows_written"] == 100
    assert smoke_iem["rows_per_second"] == 10.0


def test_data_health_renders_weather_truth_location_caption(tmp_path):
    root = tmp_path / "data_store"
    obs_store = ParquetStore(root / "obs")
    hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    obs_store.append(
        "KLGA",
        pd.DataFrame(
            [{"valid_time": hour, "raw_tmpc_tenths": 20.0}]
        ),
    )
    cache_dir = root / "_truth_weather_com" / "KLGA"
    cache_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_time": hour,
                "valid_hour": hour,
                "weather_com_temp_f": 68.0,
                "weather_com_temp_c": 20.0,
                "_weather_com_alignment_version": "iem_representative_hour_v2",
            }
        ]
    ).to_parquet(cache_dir / "2026-06-03.parquet", index=False)
    db = DbClient(":memory:")

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.header"))
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.info"))
        caption = stack.enter_context(patch("streamlit.caption"))
        stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.selectbox", side_effect=["KLGA", "All"]))
        stack.enter_context(patch("streamlit.columns", return_value=[
            _MetricCol(), _MetricCol(), _MetricCol(), _MetricCol(), _MetricCol(),
        ]))
        stack.enter_context(patch("streamlit.plotly_chart"))

        data_health.render(db, data_store_dir=root)

    messages = [str(call.args[0]) for call in caption.call_args_list]
    assert any("KLGA:9:US" in message for message in messages)


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


def test_hrrr_48h_verification_frame_marks_non_major_spliced_tail(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)
    init = pd.Timestamp("2026-06-01 01:00:00", tz="UTC")
    pd.DataFrame(
        [
            {
                "init_time": init,
                "fxx": fxx,
                "valid_time": init + pd.Timedelta(hours=fxx),
                "TMP_2m": 20.0 + fxx,
                "is_extrapolated": fxx == 48,
            }
            for fxx in range(1, 49)
        ]
    ).to_parquet(cache_dir / "2026-06.parquet", index=False)

    frame = data_health._hrrr_48h_verification_frame(root, "KLGA", init_limit=None)

    assert frame.loc[frame["fxx"] == 18, "source_kind"].iloc[0] == "native"
    assert frame.loc[frame["fxx"] == 19, "source_kind"].iloc[0] == "spliced_prior_major"
    assert frame.loc[frame["fxx"] == 48, "source_kind"].iloc[0] == "extrapolated"


def test_hrrr_48h_error_heatmap_preserves_one_cell_per_init_and_fxx():
    first = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    second = pd.Timestamp("2026-06-01 01:00:00", tz="UTC")
    frame = pd.DataFrame(
        [
            {"init_time": first, "fxx": 1, "error_c": 1.0, "source_kind": "native"},
            {"init_time": first, "fxx": 2, "error_c": 2.0, "source_kind": "native"},
            {"init_time": second, "fxx": 1, "error_c": 3.0, "source_kind": "native"},
            {"init_time": second, "fxx": 2, "error_c": 4.0, "source_kind": "native"},
        ]
    )

    fig = data_health._hrrr_48h_error_heatmap(
        frame,
        {"max_abs_error_c": 4.0},
    )

    heatmap = fig.data[0]
    assert heatmap.type == "heatmap"
    assert list(heatmap.x) == [1, 2]
    assert list(heatmap.y) == [first, second]
    assert heatmap.z.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert heatmap.colorbar.title.text == "error C"


def test_hrrr_48h_error_heatmap_keeps_all_nan_init_rows():
    first = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
    second = pd.Timestamp("2026-06-01 01:00:00", tz="UTC")
    frame = pd.DataFrame(
        [
            {"init_time": first, "fxx": 1, "error_c": 1.0, "source_kind": "native"},
            {"init_time": second, "fxx": 1, "error_c": pd.NA, "source_kind": "native"},
        ]
    )

    fig = data_health._hrrr_48h_error_heatmap(
        frame,
        {"max_abs_error_c": 1.0},
    )

    assert list(fig.data[0].y) == [first, second]


def test_hrrr_48h_valid_time_error_heatmap_preserves_one_cell_per_valid_time_and_fxx():
    first = pd.Timestamp("2026-06-01 01:00:00", tz="UTC")
    second = pd.Timestamp("2026-06-01 02:00:00", tz="UTC")
    frame = pd.DataFrame(
        [
            {"valid_time": first, "fxx": 1, "error_c": 1.0},
            {"valid_time": first, "fxx": 2, "error_c": 2.0},
            {"valid_time": second, "fxx": 1, "error_c": 3.0},
            {"valid_time": second, "fxx": 2, "error_c": 4.0},
        ]
    )

    fig = data_health._hrrr_48h_valid_time_error_heatmap(
        frame,
        {"max_abs_error_c": 4.0},
    )

    heatmap = fig.data[0]
    assert heatmap.type == "heatmap"
    assert list(heatmap.x) == [1, 2]
    assert list(heatmap.y) == [first, second]
    assert heatmap.z.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert heatmap.colorbar.title.text == "error C"


def test_hrrr_48h_valid_time_error_heatmap_keeps_all_nan_valid_time_rows():
    first = pd.Timestamp("2026-06-01 01:00:00", tz="UTC")
    second = pd.Timestamp("2026-06-01 02:00:00", tz="UTC")
    frame = pd.DataFrame(
        [
            {"valid_time": first, "fxx": 1, "error_c": 1.0},
            {"valid_time": second, "fxx": 1, "error_c": pd.NA},
        ]
    )

    fig = data_health._hrrr_48h_valid_time_error_heatmap(
        frame,
        {"max_abs_error_c": 1.0},
    )

    assert list(fig.data[0].y) == [first, second]


def test_selected_init_forecast_figure_splits_native_and_spliced_hrrr():
    init = pd.Timestamp("2026-06-01 01:00:00", tz="UTC")
    selected = pd.DataFrame(
        [
            {
                "init_time": init,
                "fxx": 18,
                "valid_time": init + pd.Timedelta(hours=18),
                "forecast_c": 21.0,
                "observed_c": 20.0,
                "source_kind": "native",
            },
            {
                "init_time": init,
                "fxx": 19,
                "valid_time": init + pd.Timedelta(hours=19),
                "forecast_c": 22.0,
                "observed_c": 21.0,
                "source_kind": "spliced_prior_major",
            },
        ]
    )

    fig = data_health._selected_init_forecast_figure(selected)

    assert [trace.name for trace in fig.data] == [
        "HRRR native",
        "HRRR spliced prior major",
        "IEM observed",
    ]
    assert fig.data[0].x == (18,)
    assert fig.data[1].x == (19,)


def test_iem_weather_truth_frame_joins_hourly_obs_and_computes_f_error(tmp_path):
    root = tmp_path / "data_store"
    valid_hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    temp_frame = pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_hour": valid_hour,
                "iem_obs_c": 20.0,
                "has_obs": True,
            }
        ]
    )
    cache_dir = root / "_truth_weather_com" / "KLGA"
    cache_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_time": valid_hour + pd.Timedelta(minutes=5),
                "valid_hour": valid_hour,
                "weather_com_temp_f": 69.0,
                "weather_com_temp_c": 20.5556,
                "_weather_com_alignment_version": "iem_representative_hour_v2",
            }
        ]
    ).to_parquet(cache_dir / "2026-06-03.parquet", index=False)

    frame, message = data_health._iem_weather_truth_frame(root, "KLGA", temp_frame)

    assert message is None
    assert frame["valid_hour"].tolist() == [valid_hour]
    assert frame["weather_com_valid_time"].tolist() == [
        valid_hour + pd.Timedelta(minutes=5)
    ]
    assert frame["iem_obs_f"].tolist() == [68.0]
    assert frame["weather_com_temp_f"].tolist() == [69.0]
    assert frame["truth_error_f"].tolist() == [-1.0]
    metrics = data_health._iem_weather_truth_metrics(frame)
    assert metrics["matched_hours"] == 1
    assert metrics["missing_weather_com_hours"] == 0
    assert metrics["mae_f"] == 1.0
    assert metrics["bias_f"] == -1.0
    assert metrics["max_abs_error_f"] == 1.0


def test_iem_weather_truth_frame_prefers_iem_representative_weather_candidate(tmp_path):
    root = tmp_path / "data_store"
    valid_hour = pd.Timestamp("2026-05-24 12:00:00", tz="UTC")
    temp_frame = pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_hour": valid_hour,
                "iem_obs_c": 11.6667,
                "has_obs": True,
            }
        ]
    )
    cache_dir = root / "_truth_weather_com" / "KLGA"
    cache_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_time": valid_hour - pd.Timedelta(minutes=9),
                "valid_hour": valid_hour,
                "weather_com_temp_f": 53.0,
                "weather_com_temp_c": 11.6667,
                "_weather_com_alignment_version": "iem_representative_hour_v2",
            },
            {
                "station_id": "KLGA",
                "valid_time": valid_hour + pd.Timedelta(minutes=19),
                "valid_hour": valid_hour,
                "weather_com_temp_f": 54.0,
                "weather_com_temp_c": 12.2222,
                "_weather_com_alignment_version": "iem_representative_hour_v2",
            },
        ]
    ).to_parquet(cache_dir / "2026-05-24.parquet", index=False)

    frame, message = data_health._iem_weather_truth_frame(root, "KLGA", temp_frame)

    assert message is None
    assert frame["weather_com_valid_time"].tolist() == [
        valid_hour - pd.Timedelta(minutes=9)
    ]
    assert frame["weather_com_temp_f"].tolist() == [53.0]


def test_iem_weather_truth_frame_keeps_iem_rows_when_truth_fetch_empty(tmp_path, monkeypatch):
    valid_hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    temp_frame = pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_hour": valid_hour,
                "iem_obs_c": 20.0,
                "has_obs": True,
            }
        ]
    )

    monkeypatch.setattr(data_health, "WeatherComTruthClient", _EmptyTruthClient)

    frame, message = data_health._iem_weather_truth_frame(
        tmp_path / "data_store",
        "KLGA",
        temp_frame,
    )

    assert not frame.empty
    assert frame["has_weather_com"].tolist() == [False]
    assert message is None


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
        stack.enter_context(patch("ui.pages.data_health.WeatherComTruthClient", _EmptyTruthClient))
        stack.enter_context(patch("streamlit.columns", return_value=[_MetricCol(), _MetricCol(), _MetricCol(), _MetricCol(), _MetricCol()]))
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
