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

    def read_cached_date(self, *_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "station_id",
                "valid_time",
                "valid_hour",
                "weather_com_temp_f",
                "weather_com_temp_c",
            ]
        )

    def fetch_date(self, *_args, **_kwargs) -> pd.DataFrame:
        raise AssertionError("Data Health must not fetch Weather.com over HTTP")


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


def test_f01_iem_range_quality_reports_coverage_duplicates_and_null_fields(tmp_path):
    root = tmp_path / "data_store"
    hour = pd.Timestamp("2026-06-01 00:00", tz="UTC")
    hrrr = pd.DataFrame(
        [
            {
                "init_time": hour - pd.Timedelta(hours=1),
                "fxx": 1,
                "valid_time": hour,
                "TMP_2m": value,
            }
            for value in (20.0, None)
        ]
    )
    obs = pd.DataFrame(
        [{
            "valid_time": hour,
            "raw_tmpc_tenths": 20.0,
        }]
    )
    ParquetStore(root / "hrrr").append("KLGA", hrrr)
    # Write the duplicate directly because ParquetStore correctly deduplicates it.
    hrrr.to_parquet(root / "hrrr" / "KLGA" / "2026-Q2.parquet", index=False)
    ParquetStore(root / "obs").append("KLGA", obs)

    summary, issues = data_health._f01_iem_range_quality(
        root,
        "KLGA",
        hour,
        hour,
    )

    hrrr_summary = summary[summary["source"] == "HRRR f01"].iloc[0]
    assert hrrr_summary["expected_hours"] == 24
    assert hrrr_summary["actual_hours"] == 1
    assert hrrr_summary["missing_hours"] == 23
    assert hrrr_summary["duplicate_hours"] == 1
    tmp_issue = issues[
        (issues["source"] == "HRRR f01") & (issues["field"] == "TMP_2m")
    ].iloc[0]
    assert tmp_issue["issue"] == "null_values"
    assert tmp_issue["null_count"] == 1
    assert (
        issues[
            (issues["source"] == "IEM") & (issues["field"] == "raw_dwpc_tenths")
        ]["issue"].iloc[0]
        == "missing_column"
    )


def test_store_read_cache_reuses_unchanged_file_and_invalidates_on_change(
    tmp_path,
):
    root = tmp_path / "data_store"
    station_dir = root / "obs" / "KLGA"
    station_dir.mkdir(parents=True)
    path = station_dir / "2026-Q2.parquet"
    first = pd.DataFrame(
        [{
            "valid_time": pd.Timestamp("2026-06-01 00:00", tz="UTC"),
            "raw_tmpc_tenths": 20.0,
        }]
    )
    first.to_parquet(path, index=False)

    with patch(
        "ui.pages.data_health.pd.read_parquet",
        wraps=pd.read_parquet,
    ) as read_parquet:
        assert len(data_health._read_store(root, "obs", "KLGA")) == 1
        assert len(data_health._read_store(root, "obs", "KLGA")) == 1
        assert read_parquet.call_count == 1

        pd.concat(
            [
                first,
                pd.DataFrame(
                    [{
                        "valid_time": pd.Timestamp("2026-06-01 01:00", tz="UTC"),
                        "raw_tmpc_tenths": 21.0,
                    }]
                ),
            ],
            ignore_index=True,
        ).to_parquet(path, index=False)

        assert len(data_health._read_store(root, "obs", "KLGA")) == 2
        assert read_parquet.call_count == 2


def test_temperature_comparison_defaults_to_last_7_days(tmp_path):
    end = pd.Timestamp("2026-06-10", tz="UTC")
    frame = pd.DataFrame(
        [
            {
                "station_id": "KLGA",
                "valid_hour": end - pd.Timedelta(days=i),
                "hrrr_f01_c": 20.0,
                "iem_obs_c": 20.0,
                "temp_error_c": 0.0,
                "has_hrrr": True,
                "has_obs": True,
            }
            for i in range(10)
        ]
    )

    with ExitStack() as stack:
        selectbox = stack.enter_context(patch("streamlit.selectbox", return_value="KLGA"))
        date_input = stack.enter_context(
            patch("streamlit.date_input", side_effect=lambda *_args, **kwargs: kwargs["value"])
        )
        stack.enter_context(patch("streamlit.info"))
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.columns", return_value=[_MetricCol()] * 4))
        stack.enter_context(patch("streamlit.plotly_chart"))
        stack.enter_context(patch("ui.pages.data_health._render_iem_weather_truth"))
        stack.enter_context(
            patch(
                "ui.pages.data_health._temperature_comparison_frame",
                return_value=frame,
            )
        )

        result = data_health._render_temperature_comparison(tmp_path)

    assert [call.args[0] for call in selectbox.call_args_list] == ["Station"]
    assert date_input.call_args.args[0] == "Valid dates"
    assert date_input.call_args.kwargs["value"] == (
        pd.Timestamp("2026-06-04").date(),
        pd.Timestamp("2026-06-10").date(),
    )
    assert result.start == pd.Timestamp("2026-06-04", tz="UTC")
    assert result.end == pd.Timestamp("2026-06-10", tz="UTC")


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
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.margin.r == 40
    assert fig.data[0].line.color == "#1f77b4"
    assert fig.data[1].line.color == "#ff7f0e"
    assert fig.data[2].line.color == "#2ca02c"
    assert fig.data[0].x == (18,)
    assert fig.data[1].x == (19,)


def test_init_time_navigation_index_is_bounded():
    assert data_health._shift_init_index(2, -1, 4) == 1
    assert data_health._shift_init_index(0, -1, 4) == 0
    assert data_health._shift_init_index(3, 1, 4) == 3


def test_move_init_selection_updates_shared_selected_value():
    options = ["2026-06-01 00:00:00+00:00", "2026-06-01 01:00:00+00:00"]
    state = {"selected": options[0]}

    data_health._move_init_selection(state, "selected", options, 1)
    data_health._move_init_selection(state, "selected", options, 1)

    assert state["selected"] == options[1]


def test_temperature_figures_use_stable_series_colors():
    hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    temp = pd.DataFrame(
        [
            {
                "valid_hour": hour,
                "hrrr_f01_c": 21.0,
                "iem_obs_c": 20.0,
            }
        ]
    )
    truth = pd.DataFrame(
        [
            {
                "valid_hour": hour,
                "iem_obs_f": 68.0,
                "weather_com_temp_f": 69.0,
            }
        ]
    )

    temp_fig = data_health._temperature_comparison_figure(temp)
    truth_fig = data_health._iem_weather_truth_figure(truth)

    assert temp_fig.data[1].line.color == "#2ca02c"
    assert truth_fig.data[0].line.color == "#2ca02c"
    assert truth_fig.data[1].line.color == "#17becf"


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


def test_iem_weather_truth_frame_uses_independent_truth_store(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "new_run" / "data_store"
    truth_root = tmp_path / "old_demo" / "_truth_weather_com"
    valid_hour = pd.Timestamp("2026-06-04 01:00:00", tz="UTC")
    temp_frame = pd.DataFrame(
        [{
            "station_id": "KLGA",
            "valid_hour": valid_hour,
            "iem_obs_c": 20.0,
            "has_obs": True,
        }]
    )
    cache_dir = truth_root / "KLGA"
    cache_dir.mkdir(parents=True)
    pd.DataFrame(
        [{
            "station_id": "KLGA",
            "valid_time": valid_hour,
            "valid_hour": valid_hour,
            "weather_com_temp_f": 68.0,
            "weather_com_temp_c": 20.0,
            "_weather_com_alignment_version": "iem_representative_hour_v2",
        }]
    ).to_parquet(cache_dir / "2026-06-03.parquet", index=False)
    monkeypatch.setenv("POLYWEATHER_TRUTH_STORE", str(truth_root))

    frame, message = data_health._iem_weather_truth_frame(
        root,
        "KLGA",
        temp_frame,
    )

    assert message is None
    assert frame["weather_com_temp_f"].tolist() == [68.0]


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


def test_iem_weather_truth_missing_cache_is_reported_without_fetch(tmp_path, monkeypatch):
    class CacheOnlyClient(_EmptyTruthClient):
        def read_cached_date(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(data_health, "WeatherComTruthClient", CacheOnlyClient)
    valid_hour = pd.Timestamp("2026-06-04 01:00", tz="UTC")
    temp_frame = pd.DataFrame(
        [{
            "station_id": "KLGA",
            "valid_hour": valid_hour,
            "iem_obs_c": 20.0,
            "has_obs": True,
        }]
    )

    frame, message = data_health._iem_weather_truth_frame(
        tmp_path / "data_store",
        "KLGA",
        temp_frame,
    )

    assert frame["has_weather_com"].tolist() == [False]
    assert frame.attrs["truth_cache_dates"] == {"cached": 0, "requested": 1}
    assert "prefetch_weather_com_truth" in message


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


def test_hrrr_48h_latest_inits_read_only_needed_month_partitions(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)

    def rows(first: pd.Timestamp, count: int) -> list[dict]:
        return [
            {
                "init_time": first + pd.Timedelta(hours=i),
                "fxx": fxx,
                "valid_time": first + pd.Timedelta(hours=i + fxx),
                "TMP_2m": 20.0,
            }
            for i in range(count)
            for fxx in range(1, 49)
        ]

    pd.DataFrame(rows(pd.Timestamp("2026-01-29 00:00", tz="UTC"), 72)).to_parquet(
        cache_dir / "2026-01.parquet",
        index=False,
    )
    pd.DataFrame(rows(pd.Timestamp("2026-02-01 00:00", tz="UTC"), 24)).to_parquet(
        cache_dir / "2026-02.parquet",
        index=False,
    )
    (cache_dir / "2025-12.parquet").write_text("must not be read", encoding="ascii")

    for init_limit in (48, 96):
        frame = data_health._hrrr_48h_verification_frame(
            root,
            "KLGA",
            init_limit=init_limit,
        )
        assert frame["init_time"].nunique() == init_limit
        assert frame.groupby("init_time")["fxx"].count().eq(48).all()


def test_hrrr_48h_custom_range_reads_and_filters_selected_dates(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)
    base = pd.Timestamp("2026-02-01 00:00", tz="UTC")
    rows = [
        {
            "init_time": base + pd.Timedelta(hours=i),
            "fxx": fxx,
            "valid_time": base + pd.Timedelta(hours=i + fxx),
            "TMP_2m": 20.0,
        }
        for i in range(24 * 4)
        for fxx in range(1, 49)
    ]
    pd.DataFrame(rows).to_parquet(cache_dir / "2026-02.parquet", index=False)
    (cache_dir / "2026-01.parquet").write_text("must not be read", encoding="ascii")

    frame = data_health._hrrr_48h_verification_frame(
        root,
        "KLGA",
        init_limit=None,
        init_start=pd.Timestamp("2026-02-02", tz="UTC"),
        init_end=pd.Timestamp("2026-02-03", tz="UTC"),
    )

    assert frame["init_time"].nunique() == 48
    assert frame["init_time"].min() == pd.Timestamp("2026-02-02 00:00", tz="UTC")
    assert frame["init_time"].max() == pd.Timestamp("2026-02-03 23:00", tz="UTC")


def test_hrrr_48h_range_quality_reports_missing_steps_duplicates_and_nulls(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)
    init = pd.Timestamp("2026-06-01 00:00", tz="UTC")
    decoder = pd.DataFrame(
        [
            {
                "init_time": init,
                "fxx": fxx,
                "valid_time": init + pd.Timedelta(hours=fxx),
                "TMP_2m": value,
            }
            for fxx, value in ((1, 20.0), (1, None), (2, 21.0))
        ]
    )
    decoder.to_parquet(cache_dir / "2026-06.parquet", index=False)
    verification = pd.DataFrame(
        [
            {"has_obs": True},
            {"has_obs": False},
            {"has_obs": False},
        ]
    )

    summary, issues = data_health._hrrr_48h_range_quality(
        root,
        "KLGA",
        init,
        init,
        verification,
    )

    row = summary.iloc[0]
    assert row["expected_inits"] == 24
    assert row["actual_inits"] == 1
    assert row["missing_inits"] == 23
    assert row["expected_cells"] == 24 * 48
    assert row["missing_forecast_steps"] == 24 * 48 - 2
    assert row["duplicate_cells"] == 1
    assert row["missing_iem_cells"] == 2
    assert issues[
        (issues["field"] == "TMP_2m") & (issues["issue"] == "null_values")
    ]["null_count"].iloc[0] == 1


def test_monthly_decoder_baseline_missing_artifacts_shows_generation_command(
    tmp_path,
):
    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.subheader"))
        info = stack.enter_context(patch("streamlit.info"))
        read_parquet = stack.enter_context(
            patch("ui.pages.data_health.pd.read_parquet")
        )

        data_health._render_monthly_decoder_baseline(
            tmp_path / "run" / "data_store",
            "KLGA",
        )

    assert "scripts.verify_decoder_baseline" in info.call_args.args[0]
    read_parquet.assert_not_called()


def test_monthly_decoder_baseline_renders_month_fxx_and_init_views(tmp_path):
    run_dir = tmp_path / "run"
    baseline_dir = run_dir / "artifacts" / "decoder_baseline" / "KLGA"
    for subdir in ("cells", "per_init", "by_fxx"):
        (baseline_dir / subdir).mkdir(parents=True)
    init = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    pd.DataFrame(
        [{
            "month": "2026-01",
            "complete_init_count": 1,
            "incomplete_observation_init_count": 0,
            "invalid_decoder_init_count": 0,
            "performance_outlier_count": 0,
            "mae_c": 1.0,
            "bias_c": -0.5,
            "rmse_c": 1.2,
        }]
    ).to_parquet(baseline_dir / "monthly_summary.parquet", index=False)
    pd.DataFrame(
        [{
            "init_time": init,
            "mae_c": 1.0,
            "bias_c": -0.5,
            "error_std_c": 0.8,
            "data_status": "complete",
            "performance_outlier": False,
        }]
    ).to_parquet(baseline_dir / "per_init" / "2026-01.parquet", index=False)
    pd.DataFrame(
        [{"fxx": fxx, "mae_c": 1.0, "sample_count": 1} for fxx in range(1, 49)]
    ).to_parquet(baseline_dir / "by_fxx" / "2026-01.parquet", index=False)
    pd.DataFrame(
        [
            {
                "init_time": init,
                "fxx": fxx,
                "forecast_error_c": 1.0,
                "absolute_error_c": 1.0,
                "forecast_c": 11.0,
                "observed_c": 10.0,
                "source_kind": "native",
            }
            for fxx in range(1, 49)
        ]
    ).to_parquet(baseline_dir / "cells" / "2026-01.parquet", index=False)

    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(
            patch("streamlit.selectbox", side_effect=["2026-01", str(init)])
        )
        stack.enter_context(patch("streamlit.columns", return_value=[_MetricCol()] * 6))
        stack.enter_context(patch("streamlit.success"))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        chart = stack.enter_context(patch("streamlit.plotly_chart"))

        data_health._render_monthly_decoder_baseline(
            run_dir / "data_store",
            "KLGA",
        )

    assert chart.call_count == 3
    rendered = [call.args[0] for call in dataframe.call_args_list]
    assert any("forecast_error_c" in frame.columns for frame in rendered)


def test_hrrr_48h_custom_range_is_limited_to_31_days():
    assert data_health._custom_date_range_error(
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-31", tz="UTC"),
    ) is None
    assert "31 days" in data_health._custom_date_range_error(
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-02-01", tz="UTC"),
    )


def test_hrrr_48h_custom_range_rejects_oversized_ui_selection(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)
    pd.DataFrame(
        [{
            "init_time": pd.Timestamp("2026-01-01", tz="UTC"),
            "fxx": 1,
            "valid_time": pd.Timestamp("2026-01-01 01:00", tz="UTC"),
            "TMP_2m": 20.0,
        }, {
            "init_time": pd.Timestamp("2026-02-01", tz="UTC"),
            "fxx": 1,
            "valid_time": pd.Timestamp("2026-02-01 01:00", tz="UTC"),
            "TMP_2m": 20.0,
        }]
    ).to_parquet(cache_dir / "2026-01.parquet", index=False)

    with ExitStack() as stack:
        selectbox = stack.enter_context(
            patch("streamlit.selectbox", return_value="KLGA")
        )
        stack.enter_context(
            patch(
                "streamlit.date_input",
                return_value=(
                    pd.Timestamp("2026-01-01").date(),
                    pd.Timestamp("2026-02-01").date(),
                ),
            )
        )
        warning = stack.enter_context(patch("streamlit.warning"))
        stack.enter_context(patch("streamlit.subheader"))
        frame = stack.enter_context(
            patch("ui.pages.data_health._hrrr_48h_verification_frame")
        )

        data_health._render_hrrr_48h_verification(root)

    warning.assert_called_once_with("Custom date range is limited to 31 days.")
    frame.assert_not_called()
    assert [call.args[0] for call in selectbox.call_args_list] == ["48h Station"]


def test_hrrr_48h_init_time_defaults_to_range_front(tmp_path):
    root = tmp_path / "data_store"
    cache_dir = root / "_decoder_cache" / "KLGA"
    cache_dir.mkdir(parents=True)
    first = pd.Timestamp("2026-06-01 00:00", tz="UTC")
    rows = [
        {
            "init_time": init,
            "fxx": fxx,
            "valid_time": init + pd.Timedelta(hours=fxx),
            "TMP_2m": 20.0,
        }
        for init in (first, first + pd.Timedelta(hours=1))
        for fxx in range(1, 49)
    ]
    pd.DataFrame(rows).to_parquet(cache_dir / "2026-06.parquet", index=False)

    state: dict = {}
    with ExitStack() as stack:
        selectbox = stack.enter_context(
            patch(
                "streamlit.selectbox",
                side_effect=lambda label, options, **_kwargs: (
                    "KLGA" if label == "48h Station" else options[0]
                ),
            )
        )
        stack.enter_context(
            patch(
                "streamlit.date_input",
                return_value=(first.date(), first.date()),
            )
        )
        stack.enter_context(patch("streamlit.session_state", state))
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.info"))
        stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(patch("streamlit.plotly_chart"))
        stack.enter_context(
            patch(
                "streamlit.columns",
                side_effect=[
                    [_MetricCol()] * 4,
                    [_MetricCol(), _MetricCol(), _MetricCol()],
                ],
            )
        )

        data_health._render_hrrr_48h_verification(root)

    init_call = next(
        call for call in selectbox.call_args_list if call.args[0] == "Init Time"
    )
    selected_key = init_call.kwargs["key"]
    assert state[selected_key] == str(first)


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
    with ExitStack() as stack:
        stack.enter_context(patch("streamlit.subheader"))
        stack.enter_context(patch("streamlit.info"))
        dataframe = stack.enter_context(patch("streamlit.dataframe"))
        stack.enter_context(
            patch("streamlit.selectbox", side_effect=["KLGA", str(init)])
        )
        stack.enter_context(
            patch("streamlit.date_input", return_value=(init.date(), init.date()))
        )
        stack.enter_context(patch("streamlit.session_state", {}))
        stack.enter_context(patch("streamlit.columns", return_value=[_MetricCol(), _MetricCol(), _MetricCol(), _MetricCol(), _MetricCol()]))
        plotly_chart = stack.enter_context(patch("streamlit.plotly_chart"))

        data_health._render_hrrr_48h_verification(root)

    assert plotly_chart.call_count >= 2
    selected_call = next(
        call for call in dataframe.call_args_list
        if "abs_error_c" in call.args[0].columns
    )
    assert selected_call.kwargs["column_config"]["abs_error_c"] is None


class _MetricCol:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def metric(self, *_args, **_kwargs):
        return None

    def button(self, *_args, **_kwargs):
        return False
