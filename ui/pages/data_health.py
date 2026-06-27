"""Data Health page: ingestion and forecast freshness observability."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from config.stations import STATIONS
from data.forecast.hrrr_fill import MAJOR_HOURS
from data.obs.weather_com import WeatherComTruthClient
from data.storage.parquet_store import ParquetStore
from model.features import HRRR_KNOWN_REALS, OBS_UNKNOWN_REALS
from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient

_CHART_MARGIN = {"l": 60, "r": 40, "t": 64, "b": 48}
_HRRR_COLOR = "#1f77b4"
_HRRR_SPLICE_COLOR = "#ff7f0e"
_HRRR_EXTRAPOLATED_COLOR = "#9467bd"
_IEM_COLOR = "#2ca02c"
_WEATHER_COM_COLOR = "#17becf"
_POSITIVE_COLOR = "#d62728"
_NEGATIVE_COLOR = "#1f77b4"
_ZERO_LINE_COLOR = "#6b7280"
_NY_TZ = ZoneInfo("America/New_York")

_TEMP_COMPARISON_COLUMNS = [
    "station_id",
    "valid_hour",
    "hrrr_f01_c",
    "iem_obs_c",
    "temp_error_c",
    "has_hrrr",
    "has_obs",
]
_HRRR_48H_COLUMNS = [
    "station_id",
    "init_time",
    "fxx",
    "valid_time",
    "forecast_c",
    "observed_c",
    "error_c",
    "abs_error_c",
    "has_obs",
    "is_extrapolated",
    "source_kind",
]


@dataclass(frozen=True)
class _RangeSelection:
    station_id: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class _Hrrr48Selection(_RangeSelection):
    frame: pd.DataFrame


def render(
    db: "DbClient",
    data_store_dir: Path | str | None = None,
    truth_store_dir: Path | str | None = None,
) -> None:
    st.header("Data Health")

    temperature_range = _render_temperature_comparison(data_store_dir, truth_store_dir)
    if temperature_range is not None:
        _render_f01_iem_range_quality(data_store_dir, temperature_range)

    hrrr_48h_range = _render_hrrr_48h_verification(data_store_dir)
    if hrrr_48h_range is not None:
        _render_hrrr_48h_range_quality(data_store_dir, hrrr_48h_range)
    _render_monthly_decoder_baseline(
        data_store_dir,
        hrrr_48h_range.station_id if hrrr_48h_range is not None else "KLGA",
    )

    ticks = queries.loop_ticks(db, loop_name="ForecastLoop")
    if ticks.empty:
        st.info("No ForecastLoop ticks recorded yet.")
    else:
        st.plotly_chart(
            px.bar(ticks.head(50), x="tick_utc", y="events_processed", color="status"),
            use_container_width=True,
        )
        st.dataframe(ticks, use_container_width=True)

    snapshots = queries.forecast_bucket_history(db)
    if snapshots.empty:
        st.info("No forecast snapshots recorded yet.")
    else:
        latest = snapshots.sort_values("tick_utc").groupby("station_id").tail(1)
        st.subheader("Latest Station Forecast Snapshots")
        st.dataframe(
            latest[["station_id", "condition_id", "tick_utc", "p50_high_f", "mc_prob"]],
            use_container_width=True,
        )


def _render_temperature_comparison(
    data_store_dir: Path | str | None = None,
    truth_store_dir: Path | str | None = None,
) -> _RangeSelection | None:
    root = _resolve_data_store_dir(data_store_dir)
    st.subheader("Temperature Comparison")
    station_id = st.selectbox(
        "Station",
        _stations_with_temperature_data(root),
        index=0,
    )
    frame = _temperature_comparison_frame(root, station_id)
    if frame.empty:
        st.info("No HRRR f01 / IEM temperature comparison data.")
        return None

    valid_hours = pd.to_datetime(frame["valid_hour"], utc=True)
    min_day = valid_hours.min().floor("D")
    max_day = valid_hours.max().floor("D")
    default_start = max(min_day, max_day - pd.Timedelta(days=6))
    selected_dates = st.date_input(
        "Valid dates",
        value=(default_start.date(), max_day.date()),
        min_value=min_day.date(),
        max_value=max_day.date(),
    )
    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Select a start and end date.")
        return None
    range_start = _as_utc_day(selected_dates[0])
    range_end = _as_utc_day(selected_dates[1])
    range_error = _custom_date_range_error(range_start, range_end)
    if range_error:
        st.warning(range_error)
        return None
    frame = _filter_utc_date_range(frame, "valid_hour", range_start, range_end)
    if frame.empty:
        st.info("No HRRR f01 / IEM temperature comparison data in this range.")
        return _RangeSelection(station_id, range_start, range_end)

    matched = frame[frame["has_hrrr"] & frame["has_obs"]].copy()
    if matched.empty:
        st.info("No overlapping HRRR f01 and IEM observation hours.")
    else:
        st.plotly_chart(
            _temperature_comparison_figure(matched),
            use_container_width=True,
        )

    metrics = _temperature_comparison_metrics(frame)
    cols = st.columns(4)
    cols[0].metric("Matched Hours", metrics["matched_hours"])
    cols[1].metric("Missing HRRR", metrics["missing_hrrr_hours"])
    cols[2].metric("Missing IEM", metrics["missing_iem_hours"])
    cols[3].metric("MAE / Bias C", f"{metrics['mae_c']:.2f} / {metrics['bias_c']:.2f}")
    if not matched.empty:
        st.plotly_chart(
            _temperature_error_figure(matched),
            use_container_width=True,
        )

    _render_iem_weather_truth(root, station_id, frame, truth_store_dir)
    return _RangeSelection(station_id, range_start, range_end)


def _render_hrrr_48h_verification(
    data_store_dir: Path | str | None = None,
) -> _Hrrr48Selection | None:
    root = _resolve_data_store_dir(data_store_dir)
    st.subheader("HRRR 48h Forecast Verification")
    stations = _stations_with_decoder_cache(root)
    if not stations:
        st.info("No HRRR 48h decoder cache data.")
        return
    station_id = st.selectbox(
        "48h Station",
        stations,
        index=0,
    )
    bounds = _decoder_init_bounds(root, station_id)
    if bounds is None:
        st.info("No HRRR 48h decoder cache data.")
        return None
    min_day, max_day = bounds
    default_start = max(min_day, max_day - pd.Timedelta(days=6))
    selected_dates = st.date_input(
        "Init dates",
        value=(default_start.date(), max_day.date()),
        min_value=min_day.date(),
        max_value=max_day.date(),
    )
    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.info("Select a start and end date.")
        return None
    init_start = _as_utc_day(selected_dates[0])
    init_end = _as_utc_day(selected_dates[1])
    range_error = _custom_date_range_error(init_start, init_end)
    if range_error:
        st.warning(range_error)
        return None

    frame = _hrrr_48h_verification_frame(
        root,
        station_id,
        init_limit=None,
        init_start=init_start,
        init_end=init_end,
    )
    if frame.empty:
        st.info("No HRRR 48h decoder cache data.")
        return _Hrrr48Selection(station_id, init_start, init_end, frame)

    metrics = _hrrr_48h_metrics(frame)
    cols = st.columns(4)
    cols[0].metric("Matched Hours", metrics["matched_hours"])
    cols[1].metric("Missing Obs", metrics["missing_obs_hours"])
    cols[2].metric("MAE / Bias C", f"{metrics['mae_c']:.2f} / {metrics['bias_c']:.2f}")
    cols[3].metric("Max Abs Error C", f"{metrics['max_abs_error_c']:.2f}")

    st.plotly_chart(
        _hrrr_48h_valid_time_error_heatmap(frame, metrics),
        use_container_width=True,
    )

    init_times = sorted(pd.to_datetime(frame["init_time"], utc=True).unique())
    init_options = [str(ts) for ts in init_times]
    nav_key = (
        f"data_health_48h_init_index_{station_id}_"
        f"{init_start.date()}_{init_end.date()}"
    )
    selected_key = f"{nav_key}_selected"
    if st.session_state.get(selected_key) not in init_options:
        st.session_state[selected_key] = init_options[0]
    current_index = init_options.index(st.session_state[selected_key])
    nav_cols = st.columns([8, 1, 1])
    with nav_cols[0]:
        selected_init = st.selectbox("Init Time", init_options, key=selected_key)
    nav_cols[1].button(
        "←",
        key=f"{nav_key}_prev",
        disabled=current_index <= 0,
        on_click=_move_init_selection,
        args=(st.session_state, selected_key, init_options, -1),
    )
    nav_cols[2].button(
        "→",
        key=f"{nav_key}_next",
        disabled=current_index >= len(init_options) - 1,
        on_click=_move_init_selection,
        args=(st.session_state, selected_key, init_options, 1),
    )
    selected_ts = pd.Timestamp(selected_init)
    selected = frame[pd.to_datetime(frame["init_time"], utc=True) == selected_ts].copy()
    if selected.empty:
        return _Hrrr48Selection(station_id, init_start, init_end, frame)
    st.plotly_chart(
        _selected_init_forecast_figure(selected),
        use_container_width=True,
    )
    st.dataframe(
        selected,
        use_container_width=True,
        column_config={"abs_error_c": None},
    )
    return _Hrrr48Selection(station_id, init_start, init_end, frame)


def _stations_with_temperature_data(root: Path) -> list[str]:
    stations = []
    for station_id in STATIONS:
        frame = _temperature_comparison_frame(root, station_id)
        if not frame.empty:
            stations.append(station_id)
    return stations or ["KLGA"]


def _stations_with_decoder_cache(root: Path) -> list[str]:
    cache_root = root / "_decoder_cache"
    return [
        station_id for station_id in STATIONS
        if (cache_root / station_id).exists()
    ]


def _filter_utc_date_range(
    frame: pd.DataFrame,
    time_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    times = pd.to_datetime(frame[time_col], utc=True)
    end_exclusive = _as_utc_day(end).floor("D") + pd.Timedelta(days=1)
    mask = (times >= _as_utc_day(start).floor("D")) & (times < end_exclusive)
    return frame[mask].reset_index(drop=True)


def _temperature_comparison_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    matched = df[df["has_hrrr"] & df["has_obs"]].copy()
    errors = pd.to_numeric(matched["temp_error_c"], errors="coerce").dropna()
    return {
        "matched_hours": int(len(matched)),
        "missing_hrrr_hours": int((~df["has_hrrr"]).sum()) if not df.empty else 0,
        "missing_iem_hours": int((~df["has_obs"]).sum()) if not df.empty else 0,
        "mae_c": float(errors.abs().mean()) if not errors.empty else 0.0,
        "bias_c": float(errors.mean()) if not errors.empty else 0.0,
    }


def _temperature_comparison_figure(matched: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tuple(matched["valid_hour"]),
            y=tuple(matched["hrrr_f01_c"]),
            mode="lines",
            name="HRRR f01 C",
            line={"color": _HRRR_COLOR},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tuple(matched["valid_hour"]),
            y=tuple(matched["iem_obs_c"]),
            mode="lines",
            name="IEM observed C",
            line={"color": _IEM_COLOR},
        )
    )
    _apply_timeseries_layout(
        fig,
        title="HRRR f01 vs IEM Observed Temperature",
        yaxis_title="temperature C",
        valid_hours=matched["valid_hour"],
        show_legend=True,
    )
    return fig


def _temperature_error_figure(matched: pd.DataFrame) -> go.Figure:
    return _signed_error_bar_figure(
        matched,
        error_col="temp_error_c",
        title="HRRR f01 Error vs IEM Observed Temperature",
        yaxis_title="error C",
        trace_name="HRRR - IEM C",
        value_label="error C",
    )


def _iem_weather_truth_figure(matched: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tuple(matched["valid_hour"]),
            y=tuple(matched["iem_obs_f"]),
            mode="lines",
            name="IEM observed F",
            line={"color": _IEM_COLOR},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tuple(matched["valid_hour"]),
            y=tuple(matched["weather_com_temp_f"]),
            mode="lines",
            name="Weather.com truth F",
            line={"color": _WEATHER_COM_COLOR},
        )
    )
    _apply_timeseries_layout(
        fig,
        title="IEM Observed vs Weather.com Truth Temperature",
        yaxis_title="temperature F",
        valid_hours=matched["valid_hour"],
        show_legend=True,
    )
    return fig


def _iem_weather_truth_error_figure(matched: pd.DataFrame) -> go.Figure:
    return _signed_error_bar_figure(
        matched,
        error_col="truth_error_f",
        title="IEM Error vs Weather.com Truth",
        yaxis_title="error F",
        trace_name="IEM - Weather.com F",
        value_label="error F",
    )


def _signed_error_bar_figure(
    frame: pd.DataFrame,
    error_col: str,
    title: str,
    yaxis_title: str,
    trace_name: str,
    value_label: str,
) -> go.Figure:
    errors = pd.to_numeric(frame[error_col], errors="coerce")
    colors = [_POSITIVE_COLOR if value >= 0 else _NEGATIVE_COLOR for value in errors]
    fig = go.Figure(
        data=[
            go.Bar(
                x=tuple(frame["valid_hour"]),
                y=tuple(errors),
                marker={"color": colors},
                name=trace_name,
                hovertemplate=(
                    "valid_hour=%{x|%Y-%m-%d %H:%M UTC}<br>"
                    f"{value_label}=%{{y:.2f}}<extra></extra>"
                ),
            )
        ]
    )
    fig.add_hline(y=0, line_color=_ZERO_LINE_COLOR, line_width=1)
    _apply_timeseries_layout(
        fig,
        title=title,
        yaxis_title=yaxis_title,
        valid_hours=frame["valid_hour"],
        show_legend=False,
    )
    return fig


def _apply_timeseries_layout(
    fig: go.Figure,
    title: str,
    yaxis_title: str,
    valid_hours: pd.Series,
    show_legend: bool,
) -> None:
    fig.update_layout(
        title=title,
        xaxis_title=_valid_hour_axis_title(valid_hours),
        yaxis_title=yaxis_title,
        margin=_CHART_MARGIN,
        showlegend=show_legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    fig.update_xaxes(domain=[0.0, 1.0])


def _valid_hour_axis_title(valid_hours: pd.Series) -> str:
    offsets = []
    for ts in pd.to_datetime(valid_hours, utc=True).dropna():
        local = ts.to_pydatetime().astimezone(_NY_TZ)
        offset = local.utcoffset()
        if offset is None:
            continue
        hours = int(offset.total_seconds() // 3600)
        offsets.append(f"UTC{hours:+d}")
    unique = sorted(set(offsets))
    suffix = "/".join(unique) if unique else "UTC-4/UTC-5"
    return f"valid_hour (UTC; NYC local {suffix})"


def _render_iem_weather_truth(
    root: Path,
    station_id: str,
    frame: pd.DataFrame,
    truth_store_dir: Path | str | None = None,
) -> None:
    st.subheader("IEM vs Weather.com Truth")
    st.caption(f"Truth location: {WeatherComTruthClient.location_code(station_id)}")
    truth, message = _iem_weather_truth_frame(
        root,
        station_id,
        frame,
        truth_store_dir=truth_store_dir,
    )
    cache_dates = truth.attrs.get("truth_cache_dates")
    if cache_dates:
        st.caption(
            "Truth cache local dates: "
            f"{cache_dates['cached']} / {cache_dates['requested']}"
        )
    if message:
        st.info(message)
    if truth.empty:
        return
    if message and not truth["has_weather_com"].any():
        return

    metrics = _iem_weather_truth_metrics(truth)
    cols = st.columns(5)
    cols[0].metric("Matched Hours", metrics["matched_hours"])
    cols[1].metric("Missing Truth", metrics["missing_weather_com_hours"])
    cols[2].metric("MAE F", f"{metrics['mae_f']:.2f}")
    cols[3].metric("Bias F", f"{metrics['bias_f']:.2f}")
    cols[4].metric("Max Abs Error F", f"{metrics['max_abs_error_f']:.2f}")

    matched = truth[truth["has_weather_com"]].copy()
    if not matched.empty:
        st.plotly_chart(
            _iem_weather_truth_figure(matched),
            use_container_width=True,
        )
        st.plotly_chart(
            _iem_weather_truth_error_figure(matched),
            use_container_width=True,
        )
    st.dataframe(truth, use_container_width=True)


def _iem_weather_truth_frame(
    root: Path,
    station_id: str,
    temp_frame: pd.DataFrame,
    truth_store_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    if temp_frame.empty or "iem_obs_c" not in temp_frame.columns:
        return _empty_iem_weather_truth_frame(), None

    has_obs = (
        temp_frame["has_obs"].astype(bool)
        if "has_obs" in temp_frame.columns
        else pd.Series(True, index=temp_frame.index)
    )
    obs = temp_frame[has_obs].copy()
    obs = obs[obs["iem_obs_c"].notna()].copy()
    if obs.empty:
        return _empty_iem_weather_truth_frame(), None

    obs["valid_hour"] = pd.to_datetime(obs["valid_hour"], utc=True)
    obs["iem_obs_c"] = pd.to_numeric(obs["iem_obs_c"], errors="coerce")
    obs = obs[obs["iem_obs_c"].notna()].copy()
    obs["iem_obs_f"] = _c_to_f(obs["iem_obs_c"])
    obs = obs[["station_id", "valid_hour", "iem_obs_c", "iem_obs_f"]]

    local_dates = (
        obs["valid_hour"]
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    client = WeatherComTruthClient(_resolve_truth_store_dir(root, truth_store_dir))
    parts: list[pd.DataFrame] = []
    missing_dates: list[str] = []
    for day in local_dates:
        part = client.read_cached_date(station_id, day)
        if part is None:
            missing_dates.append(day)
            continue
        if not part.empty:
            parts.append(part)

    if parts:
        truth = pd.concat(parts, ignore_index=True)
        truth["valid_hour"] = pd.to_datetime(truth["valid_hour"], utc=True)
        truth["valid_time"] = pd.to_datetime(truth["valid_time"], utc=True)
        truth = (
            WeatherComTruthClient._select_representative_hours(truth)
            [[
                "station_id",
                "valid_hour",
                "valid_time",
                "weather_com_temp_f",
                "weather_com_temp_c",
            ]]
            .rename(columns={"valid_time": "weather_com_valid_time"})
        )
    else:
        truth = pd.DataFrame(
            columns=[
                "station_id",
                "valid_hour",
                "weather_com_valid_time",
                "weather_com_temp_f",
                "weather_com_temp_c",
            ]
        )

    merged = pd.merge(obs, truth, on=["station_id", "valid_hour"], how="left")
    merged["has_weather_com"] = merged["weather_com_temp_f"].notna()
    merged["truth_error_f"] = merged["iem_obs_f"] - merged["weather_com_temp_f"]
    merged = merged.sort_values("valid_hour").reset_index(drop=True)
    result = merged[_IEM_WEATHER_TRUTH_COLUMNS]
    result.attrs["truth_cache_dates"] = {
        "cached": len(local_dates) - len(missing_dates),
        "requested": len(local_dates),
    }
    message = (
        f"{len(missing_dates)} Weather.com local dates are not cached. "
        "Run `python -m scripts.prefetch_weather_com_truth` before reviewing "
        "this range."
        if missing_dates
        else None
    )
    return result, message


_IEM_WEATHER_TRUTH_COLUMNS = [
    "station_id",
    "valid_hour",
    "iem_obs_c",
    "iem_obs_f",
    "weather_com_valid_time",
    "weather_com_temp_f",
    "weather_com_temp_c",
    "truth_error_f",
    "has_weather_com",
]


def _empty_iem_weather_truth_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_IEM_WEATHER_TRUTH_COLUMNS)


def _iem_weather_truth_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    matched = df[df["has_weather_com"]].copy()
    errors = pd.to_numeric(matched["truth_error_f"], errors="coerce").dropna()
    abs_errors = errors.abs()
    return {
        "matched_hours": int(len(matched)),
        "missing_weather_com_hours": int((~df["has_weather_com"]).sum())
        if not df.empty else 0,
        "mae_f": float(abs_errors.mean()) if not abs_errors.empty else 0.0,
        "bias_f": float(errors.mean()) if not errors.empty else 0.0,
        "max_abs_error_f": float(abs_errors.max()) if not abs_errors.empty else 0.0,
    }


def _c_to_f(values: pd.Series) -> pd.Series:
    return values.astype(float) * 9.0 / 5.0 + 32.0


def _temperature_comparison_frame(
    data_store_dir: Path | str | None,
    station_id: str,
) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    hrrr = _read_store(root, "hrrr", station_id)
    obs = _read_store(root, "obs", station_id)
    hrrr_temp = _hourly_temperature_frame(
        hrrr,
        station_id=station_id,
        time_col="valid_time",
        temp_col="TMP_2m",
        output_col="hrrr_f01_c",
    )
    obs_temp = _hourly_temperature_frame(
        obs,
        station_id=station_id,
        time_col="valid_time",
        temp_col="raw_tmpc_tenths",
        output_col="iem_obs_c",
    )
    if hrrr_temp.empty and obs_temp.empty:
        return _empty_temperature_comparison_frame()
    frame = pd.merge(hrrr_temp, obs_temp, on=["station_id", "valid_hour"], how="outer")
    frame = frame.sort_values("valid_hour").reset_index(drop=True)
    frame["has_hrrr"] = frame["hrrr_f01_c"].notna()
    frame["has_obs"] = frame["iem_obs_c"].notna()
    frame["temp_error_c"] = frame["hrrr_f01_c"] - frame["iem_obs_c"]
    return frame[_TEMP_COMPARISON_COLUMNS]


def _f01_iem_range_quality(
    data_store_dir: Path | str,
    station_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = _resolve_data_store_dir(data_store_dir)
    sources = (
        (
            "HRRR f01",
            _filter_source_range(_read_store(root, "hrrr", station_id), "valid_time", start, end),
            ["init_time", "fxx", "valid_time", *HRRR_KNOWN_REALS],
        ),
        (
            "IEM",
            _filter_source_range(_read_store(root, "obs", station_id), "valid_time", start, end),
            ["valid_time", *OBS_UNKNOWN_REALS],
        ),
    )
    expected_hours = _inclusive_day_count(start, end) * 24
    summary_rows = []
    issue_parts = []
    for source, frame, required_fields in sources:
        if "valid_time" in frame.columns:
            hours = pd.to_datetime(frame["valid_time"], utc=True).dt.floor("h")
            actual_hours = int(hours.nunique())
            duplicate_hours = int(len(hours) - actual_hours)
        else:
            actual_hours = 0
            duplicate_hours = 0
        summary_rows.append(
            {
                "source": source,
                "expected_hours": expected_hours,
                "actual_hours": actual_hours,
                "missing_hours": max(expected_hours - actual_hours, 0),
                "duplicate_hours": duplicate_hours,
            }
        )
        issue_parts.append(_field_issue_rows(source, frame, required_fields))
    issues = pd.concat(issue_parts, ignore_index=True)
    return pd.DataFrame(summary_rows), issues


def _render_f01_iem_range_quality(
    data_store_dir: Path | str,
    selection: _RangeSelection,
) -> None:
    st.subheader("F01 / IEM Range Quality")
    summary, issues = _f01_iem_range_quality(
        data_store_dir,
        selection.station_id,
        selection.start,
        selection.end,
    )
    st.dataframe(summary, use_container_width=True)
    if issues.empty:
        st.success("No missing model fields or null values in this range.")
    else:
        st.dataframe(issues, use_container_width=True)


def _filter_source_range(
    frame: pd.DataFrame,
    time_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty or time_col not in frame.columns:
        return frame.iloc[0:0].copy()
    return _filter_utc_date_range(frame, time_col, start, end)


def _inclusive_day_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    start_day = _as_utc_day(start).floor("D")
    end_day = _as_utc_day(end).floor("D")
    return max((end_day - start_day).days + 1, 0)


def _field_issue_rows(
    source: str,
    frame: pd.DataFrame,
    required_fields: list[str],
) -> pd.DataFrame:
    rows = []
    for field in required_fields:
        if field not in frame.columns:
            rows.append(
                {
                    "source": source,
                    "field": field,
                    "issue": "missing_column",
                    "null_count": None,
                    "null_pct": None,
                }
            )
            continue
        null_count = int(frame[field].isna().sum())
        if null_count:
            rows.append(
                {
                    "source": source,
                    "field": field,
                    "issue": "null_values",
                    "null_count": null_count,
                    "null_pct": null_count / len(frame) if len(frame) else 0.0,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["source", "field", "issue", "null_count", "null_pct"],
    )


def _hourly_temperature_frame(
    df: pd.DataFrame,
    station_id: str,
    time_col: str,
    temp_col: str,
    output_col: str,
) -> pd.DataFrame:
    if df.empty or time_col not in df.columns or temp_col not in df.columns:
        return pd.DataFrame(columns=["station_id", "valid_hour", output_col])
    temp = df[[time_col, temp_col]].copy()
    temp["valid_hour"] = pd.to_datetime(temp[time_col], utc=True).dt.floor("h")
    temp[output_col] = pd.to_numeric(temp[temp_col], errors="coerce")
    temp = (
        temp.sort_values([time_col])
        .drop_duplicates(subset=["valid_hour"], keep="last")
        .reset_index(drop=True)
    )
    temp["station_id"] = station_id
    return temp[["station_id", "valid_hour", output_col]]


def _empty_temperature_comparison_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_TEMP_COMPARISON_COLUMNS)


def _hrrr_48h_verification_frame(
    data_store_dir: Path | str | None,
    station_id: str,
    init_limit: int | None = 48,
    init_start: pd.Timestamp | None = None,
    init_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    decoder = _read_decoder_cache(
        root,
        station_id,
        init_limit=init_limit,
        init_start=init_start,
        init_end=init_end,
    )
    required = {"init_time", "fxx", "valid_time", "TMP_2m"}
    if decoder.empty or not required.issubset(decoder.columns):
        return _empty_hrrr_48h_frame()

    cols = ["init_time", "fxx", "valid_time", "TMP_2m"]
    if "is_extrapolated" in decoder.columns:
        cols.append("is_extrapolated")
    frame = decoder[cols].copy()
    frame["station_id"] = station_id
    frame["init_time"] = pd.to_datetime(frame["init_time"], utc=True)
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    if init_start is not None and init_end is not None:
        start = pd.Timestamp(init_start)
        end_exclusive = pd.Timestamp(init_end).floor("D") + pd.Timedelta(days=1)
        frame = frame[
            (frame["init_time"] >= start.floor("D"))
            & (frame["init_time"] < end_exclusive)
        ].copy()
    frame["fxx"] = pd.to_numeric(frame["fxx"], errors="coerce")
    frame = frame[frame["fxx"].between(1, 48)].copy()
    frame["fxx"] = frame["fxx"].astype(int)
    frame["forecast_c"] = pd.to_numeric(frame["TMP_2m"], errors="coerce")
    if "is_extrapolated" not in frame.columns:
        frame["is_extrapolated"] = False
    frame["is_extrapolated"] = frame["is_extrapolated"].fillna(False).astype(bool)
    frame["source_kind"] = frame.apply(_hrrr_source_kind, axis=1)

    if init_limit is not None and init_limit > 0 and not frame.empty:
        latest_inits = (
            pd.Series(frame["init_time"].drop_duplicates())
            .sort_values()
            .tail(init_limit)
        )
        frame = frame[frame["init_time"].isin(set(latest_inits))].copy()

    obs = _read_store(root, "obs", station_id)
    if obs.empty or not {"valid_time", "raw_tmpc_tenths"}.issubset(obs.columns):
        frame["observed_c"] = pd.NA
    else:
        obs_temp = obs[["valid_time", "raw_tmpc_tenths"]].copy()
        obs_temp["valid_time"] = pd.to_datetime(obs_temp["valid_time"], utc=True).dt.floor("h")
        obs_temp["observed_c"] = pd.to_numeric(obs_temp["raw_tmpc_tenths"], errors="coerce")
        obs_temp = (
            obs_temp.sort_values("valid_time")
            .drop_duplicates(subset=["valid_time"], keep="last")
            [["valid_time", "observed_c"]]
        )
        frame = pd.merge(frame, obs_temp, on="valid_time", how="left")

    frame["has_obs"] = frame["observed_c"].notna()
    frame["error_c"] = frame["forecast_c"] - frame["observed_c"]
    frame["abs_error_c"] = frame["error_c"].abs()
    frame = frame.sort_values(["init_time", "fxx"]).reset_index(drop=True)
    return frame[_HRRR_48H_COLUMNS]


def _hrrr_48h_range_quality(
    data_store_dir: Path | str,
    station_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    verification_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = _resolve_data_store_dir(data_store_dir)
    decoder = _read_decoder_cache(
        root,
        station_id,
        init_limit=None,
        init_start=start,
        init_end=end,
    )
    decoder = _filter_source_range(decoder, "init_time", start, end)
    expected_inits = _inclusive_day_count(start, end) * 24
    expected_cells = expected_inits * 48
    if decoder.empty or not {"init_time", "fxx"}.issubset(decoder.columns):
        actual_inits = 0
        actual_cells = 0
        duplicate_cells = 0
    else:
        inits = pd.to_datetime(decoder["init_time"], utc=True)
        fxx = pd.to_numeric(decoder["fxx"], errors="coerce")
        valid = decoder[fxx.between(1, 48)].copy()
        valid["init_time"] = inits[fxx.between(1, 48)]
        valid["fxx"] = fxx[fxx.between(1, 48)].astype(int)
        actual_inits = int(valid["init_time"].nunique())
        actual_cells = int(valid[["init_time", "fxx"]].drop_duplicates().shape[0])
        duplicate_cells = int(len(valid) - actual_cells)
    missing_iem_cells = (
        int((~verification_frame["has_obs"].astype(bool)).sum())
        if "has_obs" in verification_frame.columns
        else len(verification_frame)
    )
    summary = pd.DataFrame(
        [{
            "expected_inits": expected_inits,
            "actual_inits": actual_inits,
            "missing_inits": max(expected_inits - actual_inits, 0),
            "expected_cells": expected_cells,
            "missing_forecast_steps": max(expected_cells - actual_cells, 0),
            "duplicate_cells": duplicate_cells,
            "missing_iem_cells": missing_iem_cells,
        }]
    )
    issues = _field_issue_rows(
        "HRRR 48h",
        decoder,
        ["init_time", "valid_time", "fxx", *HRRR_KNOWN_REALS],
    )
    return summary, issues


def _render_hrrr_48h_range_quality(
    data_store_dir: Path | str,
    selection: _Hrrr48Selection,
) -> None:
    st.subheader("HRRR 48h Range Quality")
    summary, issues = _hrrr_48h_range_quality(
        data_store_dir,
        selection.station_id,
        selection.start,
        selection.end,
        selection.frame,
    )
    st.dataframe(summary, use_container_width=True)
    if issues.empty:
        st.success("No missing decoder fields or null values in this range.")
    else:
        st.dataframe(issues, use_container_width=True)


def _render_monthly_decoder_baseline(
    data_store_dir: Path | str,
    station_id: str,
) -> None:
    st.subheader("Monthly Decoder Baseline")
    data_root = _resolve_data_store_dir(data_store_dir)
    run_dir = data_root.parent
    baseline_dir = run_dir / "artifacts" / "decoder_baseline" / station_id
    summary_path = baseline_dir / "monthly_summary.parquet"
    if not summary_path.exists():
        st.info(
            "No decoder baseline artifacts. Run "
            f"`python -m scripts.verify_decoder_baseline --run-dir {run_dir} "
            f"--station {station_id}`."
        )
        return

    summary = _read_parquet_paths([summary_path])
    if summary.empty or "month" not in summary.columns:
        st.info("Decoder baseline summary is empty.")
        return
    months = sorted(summary["month"].astype(str).unique().tolist())
    selected_month = st.selectbox(
        "Baseline Month",
        months,
        index=len(months) - 1,
    )
    month_summary = summary[summary["month"].astype(str) == selected_month].iloc[-1]
    metric_cols = st.columns(6)
    metric_cols[0].metric("Complete Inits", int(month_summary["complete_init_count"]))
    metric_cols[1].metric(
        "Incomplete Obs",
        int(month_summary["incomplete_observation_init_count"]),
    )
    metric_cols[2].metric(
        "Invalid Decoder",
        int(month_summary["invalid_decoder_init_count"]),
    )
    metric_cols[3].metric(
        "Outlier Inits",
        int(month_summary["performance_outlier_count"]),
    )
    metric_cols[4].metric(
        "MAE / Bias C",
        f"{month_summary['mae_c']:.2f} / {month_summary['bias_c']:.2f}",
    )
    metric_cols[5].metric("RMSE C", f"{month_summary['rmse_c']:.2f}")

    per_init_path = baseline_dir / "per_init" / f"{selected_month}.parquet"
    by_fxx_path = baseline_dir / "by_fxx" / f"{selected_month}.parquet"
    cells_path = baseline_dir / "cells" / f"{selected_month}.parquet"
    if not all(path.exists() for path in (per_init_path, by_fxx_path, cells_path)):
        st.info(f"Decoder baseline artifacts are incomplete for {selected_month}.")
        return
    per_init = _read_parquet_paths([per_init_path])
    by_fxx = _read_parquet_paths([by_fxx_path])
    cells = _read_parquet_paths([cells_path])
    per_init["init_time"] = pd.to_datetime(per_init["init_time"], utc=True)
    cells["init_time"] = pd.to_datetime(cells["init_time"], utc=True)

    st.plotly_chart(
        _decoder_baseline_per_init_figure(per_init, selected_month),
        use_container_width=True,
    )
    st.plotly_chart(
        _decoder_baseline_by_fxx_figure(by_fxx, selected_month),
        use_container_width=True,
    )

    issues = per_init[
        (per_init["data_status"] != "complete")
        | per_init["performance_outlier"].astype(bool)
    ].copy()
    if issues.empty:
        st.success("No structural or performance outlier init times this month.")
    else:
        st.dataframe(issues, use_container_width=True)

    init_options = [str(ts) for ts in sorted(per_init["init_time"].unique())]
    outlier_options = [
        str(ts)
        for ts in per_init.loc[
            per_init["performance_outlier"].astype(bool),
            "init_time",
        ].sort_values()
    ]
    default_init = outlier_options[0] if outlier_options else init_options[0]
    selected_init = st.selectbox(
        "Baseline Init Time",
        init_options,
        index=init_options.index(default_init),
        key=f"decoder_baseline_init_{station_id}_{selected_month}",
    )
    selected_cells = cells[cells["init_time"] == pd.Timestamp(selected_init)].copy()
    st.plotly_chart(
        _decoder_baseline_init_figure(selected_cells),
        use_container_width=True,
    )
    st.dataframe(selected_cells, use_container_width=True)


def _decoder_baseline_per_init_figure(
    per_init: pd.DataFrame,
    month: str,
) -> go.Figure:
    colors = [
        _POSITIVE_COLOR if bool(value) else _HRRR_COLOR
        for value in per_init["performance_outlier"]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tuple(per_init["init_time"]),
            y=tuple(per_init["mae_c"]),
            mode="lines+markers",
            name="MAE C",
            marker={"color": colors},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tuple(per_init["init_time"]),
            y=tuple(per_init["error_std_c"]),
            mode="lines",
            name="Error std C",
            line={"color": _HRRR_EXTRAPOLATED_COLOR},
        )
    )
    fig.update_layout(
        title=f"{month} Per-init HRRR Baseline",
        xaxis_title="init_time UTC",
        yaxis_title="error C",
        margin=_CHART_MARGIN,
    )
    return fig


def _decoder_baseline_by_fxx_figure(
    by_fxx: pd.DataFrame,
    month: str,
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=tuple(by_fxx["fxx"]),
                y=tuple(by_fxx["mae_c"]),
                mode="lines+markers",
                name="MAE C",
                line={"color": _HRRR_COLOR},
            )
        ]
    )
    fig.update_layout(
        title=f"{month} HRRR Baseline MAE by Forecast Hour",
        xaxis_title="fxx",
        yaxis_title="MAE C",
        margin=_CHART_MARGIN,
    )
    return fig


def _decoder_baseline_init_figure(cells: pd.DataFrame) -> go.Figure:
    colors = [
        _POSITIVE_COLOR if value >= 0 else _NEGATIVE_COLOR
        for value in cells["forecast_error_c"]
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=tuple(cells["fxx"]),
                y=tuple(cells["forecast_error_c"]),
                marker={"color": colors},
                name="HRRR - IEM C",
            )
        ]
    )
    fig.add_hline(y=0, line_color=_ZERO_LINE_COLOR, line_width=1)
    fig.update_layout(
        title="Selected Init Forecast Error",
        xaxis_title="fxx",
        yaxis_title="forecast error C",
        margin=_CHART_MARGIN,
    )
    return fig


def _hrrr_source_kind(row: pd.Series) -> str:
    if bool(row.get("is_extrapolated", False)):
        return "extrapolated"
    init_hour = pd.Timestamp(row["init_time"]).hour
    if init_hour in MAJOR_HOURS or int(row["fxx"]) <= 18:
        return "native"
    return "spliced_prior_major"


def _hrrr_48h_error_heatmap(
    frame: pd.DataFrame,
    metrics: dict[str, float | int],
) -> go.Figure:
    heatmap_df = frame.copy()
    heatmap_df["init_time"] = pd.to_datetime(heatmap_df["init_time"], utc=True)
    heatmap_df["fxx"] = pd.to_numeric(heatmap_df["fxx"], errors="coerce").astype(int)
    init_index = sorted(heatmap_df["init_time"].drop_duplicates())
    fxx_index = sorted(heatmap_df["fxx"].drop_duplicates())
    pivot = (
        heatmap_df.sort_values(["init_time", "fxx"])
        .drop_duplicates(subset=["init_time", "fxx"], keep="last")
        .pivot(index="init_time", columns="fxx", values="error_c")
        .reindex(index=init_index, columns=fxx_index)
    )

    max_abs = max(1.0, float(metrics["max_abs_error_c"]))
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=list(pivot.columns),
                y=list(pivot.index),
                z=pivot.to_numpy(),
                colorscale="RdBu",
                reversescale=True,
                zmin=-max_abs,
                zmax=max_abs,
                colorbar={"title": "error C"},
                hovertemplate=(
                    "init_time=%{y|%Y-%m-%d %H:%M UTC}<br>"
                    "fxx=%{x}<br>"
                    "error C=%{z:.2f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="HRRR 48h Error by Init Time and Forecast Hour",
        xaxis_title="fxx",
        yaxis_title="init_time",
    )
    return fig


def _hrrr_48h_valid_time_error_heatmap(
    frame: pd.DataFrame,
    metrics: dict[str, float | int],
) -> go.Figure:
    heatmap_df = frame.copy()
    heatmap_df["valid_time"] = pd.to_datetime(heatmap_df["valid_time"], utc=True)
    heatmap_df["fxx"] = pd.to_numeric(heatmap_df["fxx"], errors="coerce").astype(int)
    valid_time_index = sorted(heatmap_df["valid_time"].drop_duplicates())
    fxx_index = sorted(heatmap_df["fxx"].drop_duplicates())
    pivot = (
        heatmap_df.sort_values(["valid_time", "fxx"])
        .drop_duplicates(subset=["valid_time", "fxx"], keep="last")
        .pivot(index="valid_time", columns="fxx", values="error_c")
        .reindex(index=valid_time_index, columns=fxx_index)
    )

    max_abs = max(1.0, float(metrics["max_abs_error_c"]))
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=list(pivot.columns),
                y=list(pivot.index),
                z=pivot.to_numpy(),
                colorscale="RdBu",
                reversescale=True,
                zmin=-max_abs,
                zmax=max_abs,
                colorbar={"title": "error C"},
                hovertemplate=(
                    "valid_time=%{y|%Y-%m-%d %H:%M UTC}<br>"
                    "fxx=%{x}<br>"
                    "error C=%{z:.2f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="HRRR 48h Error by Valid Time and Forecast Hour",
        xaxis_title="fxx",
        yaxis_title="valid_time",
    )
    return fig


def _selected_init_forecast_figure(selected: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    labels = {
        "native": ("HRRR native", "solid", _HRRR_COLOR),
        "spliced_prior_major": ("HRRR spliced prior major", "dash", _HRRR_SPLICE_COLOR),
        "extrapolated": ("HRRR extrapolated", "dot", _HRRR_EXTRAPOLATED_COLOR),
    }
    for source_kind, (name, dash, color) in labels.items():
        subset = selected[selected["source_kind"] == source_kind].sort_values("fxx")
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=tuple(subset["fxx"]),
                y=tuple(subset["forecast_c"]),
                mode="lines+markers",
                name=name,
                line={"dash": dash, "color": color},
                customdata=list(subset["valid_time"]),
                hovertemplate=(
                    "fxx=%{x}<br>"
                    "valid_time=%{customdata|%Y-%m-%d %H:%M UTC}<br>"
                    "temperature C=%{y:.2f}<extra></extra>"
                ),
            )
        )

    observed = selected[selected["observed_c"].notna()].sort_values("fxx")
    if not observed.empty:
        fig.add_trace(
            go.Scatter(
                x=tuple(observed["fxx"]),
                y=tuple(observed["observed_c"]),
                mode="lines+markers",
                name="IEM observed",
                line={"color": _IEM_COLOR},
                customdata=list(observed["valid_time"]),
                hovertemplate=(
                    "fxx=%{x}<br>"
                    "valid_time=%{customdata|%Y-%m-%d %H:%M UTC}<br>"
                    "temperature C=%{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Selected Init 48h Forecast vs Observed",
        xaxis_title="fxx",
        yaxis_title="temperature C",
        margin=_CHART_MARGIN,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
    )
    fig.update_xaxes(domain=[0.0, 1.0])
    return fig


def _shift_init_index(current_index: int, step: int, option_count: int) -> int:
    if option_count <= 0:
        return 0
    return max(0, min(option_count - 1, current_index + step))


def _move_init_selection(
    state: dict,
    selected_key: str,
    options: list[str],
    step: int,
) -> None:
    if not options:
        return
    current_value = state.get(selected_key, options[-1])
    try:
        current_index = options.index(current_value)
    except ValueError:
        current_index = len(options) - 1
    state[selected_key] = options[_shift_init_index(current_index, step, len(options))]


def _read_decoder_cache(
    root: Path,
    station_id: str,
    init_limit: int | None = None,
    init_start: pd.Timestamp | None = None,
    init_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    station_dir = root / "_decoder_cache" / station_id
    if not station_dir.exists():
        return pd.DataFrame()
    paths = sorted(station_dir.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()

    if init_start is not None and init_end is not None:
        paths = [
            path for path in paths
            if _month_overlaps(path.stem, init_start, init_end)
        ]
        return _read_parquet_paths(paths)

    if init_limit is None:
        return _read_parquet_paths(paths)

    selected_parts: list[pd.DataFrame] = []
    unique_inits: set[pd.Timestamp] = set()
    for path in reversed(paths):
        part = _read_parquet_paths([path])
        selected_parts.append(part)
        if "init_time" in part.columns:
            unique_inits.update(pd.to_datetime(part["init_time"], utc=True).unique())
        if len(unique_inits) >= init_limit:
            break
    return pd.concat(reversed(selected_parts), ignore_index=True)


def _decoder_init_bounds(
    root: Path,
    station_id: str,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    station_dir = root / "_decoder_cache" / station_id
    paths = sorted(station_dir.glob("*.parquet"))
    if not paths:
        return None
    boundary_paths = [paths[0]] if len(paths) == 1 else [paths[0], paths[-1]]
    boundary = _read_parquet_paths(boundary_paths)
    if boundary.empty or "init_time" not in boundary.columns:
        return None
    inits = pd.to_datetime(boundary["init_time"], utc=True).dropna()
    if inits.empty:
        return None
    return inits.min().floor("D"), inits.max().floor("D")


def _as_utc_day(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _custom_date_range_error(
    init_start: pd.Timestamp,
    init_end: pd.Timestamp,
) -> str | None:
    start = _as_utc_day(init_start).floor("D")
    end = _as_utc_day(init_end).floor("D")
    if end < start:
        return "End date must be on or after start date."
    if (end - start).days + 1 > 31:
        return "Custom date range is limited to 31 days."
    return None


def _month_overlaps(
    month: str,
    init_start: pd.Timestamp,
    init_end: pd.Timestamp,
) -> bool:
    try:
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    except ValueError:
        return False
    month_end = month_start + pd.offsets.MonthBegin(1)
    return month_start <= init_end and month_end > init_start


def _read_parquet_paths(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return _cached_read_parquet_paths(_file_signature(paths)).copy()


def _file_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path), stat.st_size, stat.st_mtime_ns)
        for path in paths
        for stat in [path.stat()]
    )


@lru_cache(maxsize=128)
def _cached_read_parquet_paths(
    signature: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    parts = [pd.read_parquet(path) for path, _size, _mtime in signature]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _empty_hrrr_48h_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_HRRR_48H_COLUMNS)


def _hrrr_48h_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    matched = df[df["has_obs"]].copy()
    errors = pd.to_numeric(matched["error_c"], errors="coerce").dropna()
    abs_errors = errors.abs()
    return {
        "matched_hours": int(len(matched)),
        "missing_obs_hours": int((~df["has_obs"]).sum()) if not df.empty else 0,
        "mae_c": float(abs_errors.mean()) if not abs_errors.empty else 0.0,
        "bias_c": float(errors.mean()) if not errors.empty else 0.0,
        "max_abs_error_c": float(abs_errors.max()) if not abs_errors.empty else 0.0,
    }


def _resolve_data_store_dir(data_store_dir: Path | str | None = None) -> Path:
    if data_store_dir is None:
        env_dir = os.environ.get("POLYWEATHER_DATA_STORE")
        data_store_dir = Path(env_dir) if env_dir else Settings().data_store_dir
    return Path(data_store_dir)


def _resolve_truth_store_dir(
    data_store_dir: Path,
    truth_store_dir: Path | str | None = None,
) -> Path:
    if truth_store_dir is None:
        truth_store_dir = os.environ.get("POLYWEATHER_TRUTH_STORE")
    return Path(truth_store_dir) if truth_store_dir else data_store_dir / "_truth_weather_com"


def _read_store(root: Path, subdir: str, station_id: str) -> pd.DataFrame:
    paths = sorted((root / subdir / station_id).glob("*.parquet"))
    signature = _file_signature(paths)
    return _cached_read_store(
        str(root),
        subdir,
        station_id,
        signature,
    ).copy()


@lru_cache(maxsize=64)
def _cached_read_store(
    root: str,
    subdir: str,
    station_id: str,
    _signature: tuple[tuple[str, int, int], ...],
) -> pd.DataFrame:
    return ParquetStore(Path(root) / subdir).read(
        station_id,
        start=pd.Timestamp("2000-01-01", tz="UTC"),
        end=pd.Timestamp("2100-01-01", tz="UTC"),
    )
