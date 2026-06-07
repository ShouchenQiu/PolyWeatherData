"""Data Health page: ingestion and forecast freshness observability."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from config.stations import STATIONS
from data.forecast.hrrr_fill import MAJOR_HOURS
from data.obs.weather_com import (
    WeatherComApiKeyMissing,
    WeatherComFetchError,
    WeatherComTruthClient,
)
from data.storage.parquet_store import ParquetStore
from ui import queries

if TYPE_CHECKING:
    from db.schema import DbClient

_CHART_MARGIN = {"l": 60, "r": 40, "t": 64, "b": 48}

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


def render(db: "DbClient", data_store_dir: Path | str | None = None) -> None:
    st.header("Data Health")

    _render_temperature_comparison(data_store_dir)
    _render_hrrr_48h_verification(data_store_dir)

    store_summary = _parquet_store_summary(data_store_dir)
    st.subheader("Bootstrap Stores")
    st.dataframe(store_summary, use_container_width=True)

    st.subheader("Daily Coverage")
    st.dataframe(_daily_coverage_summary(data_store_dir), use_container_width=True)

    st.subheader("Cleaning Checks")
    st.dataframe(_cleaning_check_summary(data_store_dir), use_container_width=True)

    timing = _seed_timing_summary(data_store_dir)
    st.subheader("Seed Timing")
    if timing.empty:
        st.info("No seed timing summary.")
    else:
        st.dataframe(timing, use_container_width=True)

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


def _render_temperature_comparison(data_store_dir: Path | str | None = None) -> None:
    root = _resolve_data_store_dir(data_store_dir)
    st.subheader("Temperature Comparison")
    station_id = st.selectbox(
        "Station",
        _stations_with_temperature_data(root),
        index=0,
    )
    range_label = st.selectbox("Range", ["All", "Last 7d", "Last 24h"], index=0)
    frame = _temperature_comparison_frame(root, station_id)
    frame = _filter_temperature_range(frame, range_label)
    if frame.empty:
        st.info("No HRRR f01 / IEM temperature comparison data.")
        return

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

    _render_iem_weather_truth(root, station_id, frame)


def _render_hrrr_48h_verification(data_store_dir: Path | str | None = None) -> None:
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
    range_label = st.selectbox("Init Range", ["Last 48", "Last 96", "All"], index=0)
    init_limit = {"Last 48": 48, "Last 96": 96, "All": None}[range_label]
    frame = _hrrr_48h_verification_frame(root, station_id, init_limit=init_limit)
    if frame.empty:
        st.info("No HRRR 48h decoder cache data.")
        return

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

    init_options = [
        str(ts) for ts in sorted(pd.to_datetime(frame["init_time"], utc=True).unique())
    ]
    selected_init = st.selectbox("Init Time", init_options, index=len(init_options) - 1)
    selected_ts = pd.Timestamp(selected_init)
    selected = frame[pd.to_datetime(frame["init_time"], utc=True) == selected_ts].copy()
    if selected.empty:
        return
    st.plotly_chart(
        _selected_init_forecast_figure(selected),
        use_container_width=True,
    )
    st.dataframe(selected, use_container_width=True)


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


def _filter_temperature_range(df: pd.DataFrame, range_label: str) -> pd.DataFrame:
    if df.empty or range_label == "All":
        return df
    max_hour = pd.to_datetime(df["valid_hour"], utc=True).max()
    if pd.isna(max_hour):
        return df.iloc[0:0].copy()
    hours = 24 if range_label == "Last 24h" else 24 * 7
    start = max_hour - pd.Timedelta(hours=hours - 1)
    return df[pd.to_datetime(df["valid_hour"], utc=True) >= start].reset_index(drop=True)


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
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tuple(matched["valid_hour"]),
            y=tuple(matched["iem_obs_c"]),
            mode="lines",
            name="IEM observed C",
        )
    )
    fig.update_layout(
        title="HRRR f01 vs IEM Observed Temperature",
        xaxis_title="valid_hour",
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


def _temperature_error_figure(matched: pd.DataFrame) -> go.Figure:
    errors = pd.to_numeric(matched["temp_error_c"], errors="coerce")
    colors = ["#d62728" if value >= 0 else "#1f77b4" for value in errors]
    fig = go.Figure(
        data=[
            go.Bar(
                x=tuple(matched["valid_hour"]),
                y=tuple(errors),
                marker={"color": colors},
                name="HRRR - IEM C",
                hovertemplate=(
                    "valid_hour=%{x|%Y-%m-%d %H:%M UTC}<br>"
                    "error C=%{y:.2f}<extra></extra>"
                ),
            )
        ]
    )
    fig.add_hline(y=0, line_color="#6b7280", line_width=1)
    fig.update_layout(
        title="HRRR f01 Error vs IEM Observed Temperature",
        xaxis_title="valid_hour",
        yaxis_title="error C",
        margin=_CHART_MARGIN,
        showlegend=False,
    )
    fig.update_xaxes(domain=[0.0, 1.0])
    return fig


def _render_iem_weather_truth(
    root: Path,
    station_id: str,
    frame: pd.DataFrame,
) -> None:
    st.subheader("IEM vs Weather.com Truth")
    st.caption(f"Truth location: {WeatherComTruthClient.location_code(station_id)}")
    truth, message = _iem_weather_truth_frame(root, station_id, frame)
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
        chart_df = matched.melt(
            id_vars=["valid_hour"],
            value_vars=["iem_obs_f", "weather_com_temp_f"],
            var_name="series",
            value_name="temperature_f",
        )
        st.plotly_chart(
            px.line(
                chart_df,
                x="valid_hour",
                y="temperature_f",
                color="series",
                title="IEM Observed vs Weather.com Truth Temperature",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.line(
                matched,
                x="valid_hour",
                y="truth_error_f",
                title="IEM Error vs Weather.com Truth",
            ),
            use_container_width=True,
        )
    st.dataframe(truth, use_container_width=True)


def _iem_weather_truth_frame(
    root: Path,
    station_id: str,
    temp_frame: pd.DataFrame,
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
    client = WeatherComTruthClient(root / "_truth_weather_com")
    parts: list[pd.DataFrame] = []
    messages: list[str] = []
    for day in local_dates:
        try:
            part = client.fetch_date(station_id, day)
        except WeatherComApiKeyMissing as exc:
            messages.append(str(exc))
            continue
        except WeatherComFetchError as exc:
            messages.append(str(exc))
            continue
        if not part.empty:
            parts.append(part)

    if parts:
        truth = pd.concat(parts, ignore_index=True)
        truth["valid_hour"] = pd.to_datetime(truth["valid_hour"], utc=True)
        truth = (
            truth.sort_values("valid_time")
            .drop_duplicates(subset=["station_id", "valid_hour"], keep="last")
            [["station_id", "valid_hour", "weather_com_temp_f", "weather_com_temp_c"]]
        )
    else:
        truth = pd.DataFrame(
            columns=[
                "station_id",
                "valid_hour",
                "weather_com_temp_f",
                "weather_com_temp_c",
            ]
        )

    merged = pd.merge(obs, truth, on=["station_id", "valid_hour"], how="left")
    merged["has_weather_com"] = merged["weather_com_temp_f"].notna()
    merged["truth_error_f"] = merged["iem_obs_f"] - merged["weather_com_temp_f"]
    merged = merged.sort_values("valid_hour").reset_index(drop=True)
    message = messages[0] if messages else None
    return merged[_IEM_WEATHER_TRUTH_COLUMNS], message


_IEM_WEATHER_TRUTH_COLUMNS = [
    "station_id",
    "valid_hour",
    "iem_obs_c",
    "iem_obs_f",
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


def _parquet_store_summary(data_store_dir: Path | str | None = None) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    rows = []
    for label, subdir in (("hrrr", "hrrr"), ("obs", "obs")):
        store = ParquetStore(root / subdir)
        total = 0
        latest = None
        for station_id in STATIONS:
            df = store.read(
                station_id,
                start=pd.Timestamp("2000-01-01", tz="UTC"),
                end=pd.Timestamp("2100-01-01", tz="UTC"),
            )
            total += len(df)
            if not df.empty:
                time_col = "valid_time" if "valid_time" in df.columns else "init_time"
                station_latest = pd.to_datetime(df[time_col], utc=True).max()
                latest = station_latest if latest is None else max(latest, station_latest)
        rows.append(
            {
                "store": label,
                "path": str(root / subdir),
                "rows": int(total),
                "latest_time": str(latest) if latest is not None else None,
            }
        )
    return pd.DataFrame(rows)


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
) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    decoder = _read_decoder_cache(root, station_id)
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
        "native": ("HRRR native", "solid"),
        "spliced_prior_major": ("HRRR spliced prior major", "dash"),
        "extrapolated": ("HRRR extrapolated", "dot"),
    }
    for source_kind, (name, dash) in labels.items():
        subset = selected[selected["source_kind"] == source_kind].sort_values("fxx")
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=tuple(subset["fxx"]),
                y=tuple(subset["forecast_c"]),
                mode="lines+markers",
                name=name,
                line={"dash": dash},
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
    )
    return fig


def _read_decoder_cache(root: Path, station_id: str) -> pd.DataFrame:
    station_dir = root / "_decoder_cache" / station_id
    if not station_dir.exists():
        return pd.DataFrame()
    parts = [pd.read_parquet(path) for path in sorted(station_dir.glob("*.parquet"))]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


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


def _latest_day(df: pd.DataFrame, time_col: str) -> pd.Timestamp | None:
    if df.empty or time_col not in df.columns:
        return None
    ts = pd.to_datetime(df[time_col], utc=True)
    return ts.max().floor("D")


def _read_store(root: Path, subdir: str, station_id: str) -> pd.DataFrame:
    return ParquetStore(root / subdir).read(
        station_id,
        start=pd.Timestamp("2000-01-01", tz="UTC"),
        end=pd.Timestamp("2100-01-01", tz="UTC"),
    )


def _daily_coverage_summary(data_store_dir: Path | str | None = None) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    seed_day = _seed_summary_day(root)
    rows = []
    for station_id in STATIONS:
        hrrr = _read_store(root, "hrrr", station_id)
        obs = _read_store(root, "obs", station_id)
        hrrr_day = seed_day or _latest_day(hrrr, "init_time")
        obs_day = seed_day or _latest_day(obs, "valid_time")
        rows.append(_coverage_row("hrrr", station_id, hrrr, "init_time", hrrr_day, 24))
        rows.append(_coverage_row("obs", station_id, obs, "valid_time", obs_day, 24))
    return pd.DataFrame(rows)


def _coverage_row(
    store: str,
    station_id: str,
    df: pd.DataFrame,
    time_col: str,
    day: pd.Timestamp | None,
    expected_rows: int,
) -> dict:
    if day is None or df.empty or time_col not in df.columns:
        return {
            "store": store,
            "station_id": station_id,
            "day": None,
            "expected_rows": expected_rows,
            "actual_rows": 0,
            "missing_rows": expected_rows,
            "start_time": None,
            "end_time": None,
        }
    ts = pd.to_datetime(df[time_col], utc=True)
    mask = (ts >= day) & (ts < day + pd.Timedelta(days=1))
    day_df = df[mask]
    day_ts = ts[mask]
    actual = int(len(day_df))
    return {
        "store": store,
        "station_id": station_id,
        "day": str(day.date()),
        "expected_rows": expected_rows,
        "actual_rows": actual,
        "missing_rows": max(expected_rows - actual, 0),
        "start_time": str(day_ts.min()) if actual else None,
        "end_time": str(day_ts.max()) if actual else None,
    }


def _cleaning_check_summary(data_store_dir: Path | str | None = None) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    rows = []
    for station_id in STATIONS:
        hrrr = _read_store(root, "hrrr", station_id)
        obs = _read_store(root, "obs", station_id)
        rows.extend([
            _check_row("hrrr_required_columns", station_id, {"init_time", "fxx", "valid_time", "TMP_2m"}.issubset(hrrr.columns), hrrr),
            _check_row("hrrr_fxx_all_one", station_id, bool(not hrrr.empty and "fxx" in hrrr.columns and (hrrr["fxx"] == 1).all()), hrrr),
            _check_row("obs_required_columns", station_id, {"valid_time", "raw_tmpc_tenths"}.issubset(obs.columns), obs),
        ])
    return pd.DataFrame(rows)


def _check_row(check: str, station_id: str, passed: bool, df: pd.DataFrame) -> dict:
    return {
        "check": check,
        "station_id": station_id,
        "passed": bool(passed),
        "rows": int(len(df)),
        "nan_fraction": float(df.isna().mean().mean()) if not df.empty else 0.0,
    }


def _seed_timing_summary(data_store_dir: Path | str | None = None) -> pd.DataFrame:
    root = _resolve_data_store_dir(data_store_dir)
    rows = []
    for label, summary in _load_timing_summaries(root):
        for stage, values in (summary.get("timings") or {}).items():
            rows.append(
                {
                    "summary": label,
                    "stage": stage,
                    "elapsed_seconds": float(values.get("elapsed_seconds", 0.0)),
                    "rows_written": int(values.get("rows_written", 0)),
                    "errors": int(values.get("errors", 0)),
                    "rows_per_second": float(values.get("rows_per_second", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _seed_summary_day(root: Path) -> pd.Timestamp | None:
    summary = _load_seed_summary(root)
    seed_date = summary.get("seed_date") if summary else None
    if not seed_date:
        return None
    return pd.Timestamp(seed_date, tz="UTC")


def _load_seed_summary(root: Path) -> dict:
    return _load_summary_file(root.parent / "seed_summary.json")


def _load_timing_summaries(root: Path) -> list[tuple[str, dict]]:
    rows = []
    for label, filename in (
        ("smoke_summary", "smoke_summary.json"),
        ("seed_summary", "seed_summary.json"),
    ):
        summary = _load_summary_file(root.parent / filename)
        if summary:
            rows.append((label, summary))
    return rows


def _load_summary_file(summary_path: Path) -> dict:
    try:
        return json.loads(summary_path.read_text(encoding="ascii"))
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return {}
