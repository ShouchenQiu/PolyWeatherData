"""Offline HRRR, TFT, and MC validation analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from ui.model_evaluation import (
    EvaluationData,
    daily_target_summary,
    discover_evaluations,
    evaluation_metrics,
    horizon_metrics,
    load_evaluation,
    model_performance_table,
    skill_heatmap,
)


COLORS = {
    "Actual": "#111111",
    "HRRR": "#f28e2b",
    "TFT": "#4e79a7",
    "MC": "#9c6ade",
}


def render(*, runs_root: Path | str | None = None) -> None:
    st.header("Model Evaluation")
    st.caption(
        "Validation-only diagnostics. MC daily-high results are contiguous-fold "
        "out-of-fold estimates for complete station-local calendar days."
    )
    root = Path(runs_root or Settings().project_root / "runs")
    artifacts = discover_evaluations(root)
    if not artifacts:
        st.info(f"No complete calibrated evaluations found under {root}.")
        return
    labels = {
        f"{item.run_id} / {item.version}": item
        for item in artifacts
    }
    selected = st.selectbox("Evaluation", list(labels))
    try:
        data = load_evaluation(labels[selected])
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Evaluation artifacts are invalid: {exc}")
        return

    tabs = st.tabs(
        [
            "Overview",
            "Hourly & Horizon",
            "Daily High & MC",
            "TFT & MC Performance",
        ]
    )
    with tabs[0]:
        _overview(data)
    with tabs[1]:
        _hourly(data)
    with tabs[2]:
        _daily(data)
    with tabs[3]:
        _performance(data)


def _overview(data: EvaluationData) -> None:
    metrics = evaluation_metrics(data)
    cols = st.columns(5)
    cols[0].metric("HRRR hourly MAE", f"{metrics['hrrr_hourly_mae_f']:.2f} °F")
    cols[1].metric(
        "TFT hourly MAE",
        f"{metrics['tft_hourly_mae_f']:.2f} °F",
        f"{metrics['hourly_skill_pct']:.1f}% skill",
    )
    cols[2].metric("TFT P10–P90 coverage", f"{metrics['hourly_coverage']:.1%}")
    cols[3].metric("MC high MAE (OOF)", f"{metrics['mc_high_mae_f']:.2f} °F")
    cols[4].metric("MC high coverage (OOF)", f"{metrics['mc_coverage']:.1%}")

    errors = pd.concat(
        [
            data.hourly[["hrrr_abs_error_f"]]
            .rename(columns={"hrrr_abs_error_f": "absolute_error_f"})
            .assign(model="HRRR"),
            data.hourly[["tft_abs_error_f"]]
            .rename(columns={"tft_abs_error_f": "absolute_error_f"})
            .assign(model="TFT"),
        ]
    )
    st.plotly_chart(
        px.histogram(
            errors,
            x="absolute_error_f",
            color="model",
            barmode="overlay",
            histnorm="probability density",
            color_discrete_map=COLORS,
            title="Hourly absolute-error distribution",
        ),
        width="stretch",
    )


def _hourly(data: EvaluationData) -> None:
    init_times = sorted(data.hourly["init_time"].unique())
    selected = st.selectbox(
        "Forecast cycle",
        init_times,
        format_func=lambda value: pd.Timestamp(value).isoformat(),
    )
    cycle = data.hourly[data.hourly["init_time"] == selected].sort_values(
        "valid_time"
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=cycle["valid_time"],
            y=cycle["tft_p90_f"],
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=cycle["valid_time"],
            y=cycle["tft_p10_f"],
            fill="tonexty",
            fillcolor="rgba(78,121,167,0.18)",
            line={"width": 0},
            name="TFT P10–P90",
        )
    )
    for name, column in (
        ("Actual", "actual_f"),
        ("HRRR", "hrrr_f"),
        ("TFT", "tft_p50_f"),
    ):
        figure.add_trace(
            go.Scatter(
                x=cycle["valid_time"],
                y=cycle[column],
                name=name,
                line={"color": COLORS[name]},
            )
        )
    figure.update_layout(title="Selected-cycle temperature trajectory", yaxis_title="°F")
    st.plotly_chart(figure, width="stretch")

    horizon = horizon_metrics(data)
    st.plotly_chart(
        px.line(
            horizon,
            x="fxx",
            y=["hrrr_mae_f", "tft_mae_f"],
            labels={"value": "MAE (°F)", "variable": "Model"},
            title="MAE by forecast horizon",
            color_discrete_sequence=[COLORS["HRRR"], COLORS["TFT"]],
        ),
        width="stretch",
    )
    st.plotly_chart(
        px.line(
            horizon,
            x="fxx",
            y=["hrrr_bias_f", "tft_bias_f"],
            labels={"value": "Bias (°F)", "variable": "Model"},
            title="Bias by forecast horizon",
            color_discrete_sequence=[COLORS["HRRR"], COLORS["TFT"]],
        ),
        width="stretch",
    )
    heatmap = skill_heatmap(data)
    st.plotly_chart(
        px.imshow(
            heatmap,
            aspect="auto",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={"x": "fxx band", "y": "init hour UTC", "color": "Skill %"},
            title="TFT skill over HRRR by init hour and horizon",
        ),
        width="stretch",
    )


def _daily(data: EvaluationData) -> None:
    daily = daily_target_summary(data.daily)
    metrics = evaluation_metrics(data)
    cols = st.columns(4)
    cols[0].metric("HRRR high MAE", f"{metrics['hrrr_high_mae_f']:.2f} °F")
    cols[1].metric("TFT high MAE", f"{metrics['tft_high_mae_f']:.2f} °F")
    cols[2].metric("MC high MAE", f"{metrics['mc_high_mae_f']:.2f} °F")
    cols[3].metric("MC interval width", f"{metrics['mc_interval_width_f']:.2f} °F")
    st.caption(
        "Charts show one point per target day: forecasts and MC bounds are the "
        "median across available cycles; actual high is the observed daily value. "
        "Summary metrics above retain the OOF cycle-day evaluation grain."
    )

    forecast = go.Figure()
    for name, column in (
        ("Actual", "actual_high_f"),
        ("HRRR", "hrrr_high_f"),
        ("TFT", "tft_high_f"),
        ("MC", "mc_p50_f"),
    ):
        forecast.add_trace(
            go.Scatter(
                x=daily["target_date"],
                y=daily[column],
                mode="lines+markers",
                name=name,
                line={
                    "color": COLORS[name],
                    "width": 3 if name == "Actual" else 2,
                },
                marker={"size": 7 if name == "Actual" else 5},
            )
        )
    forecast.update_layout(
        title="Daily high forecasts by target day",
        xaxis_title="Target day",
        yaxis_title="°F",
        hovermode="x unified",
    )
    st.plotly_chart(forecast, width="stretch")

    interval = go.Figure()
    interval.add_trace(
        go.Scatter(
            x=daily["target_date"],
            y=daily["mc_p90_f"],
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    interval.add_trace(
        go.Scatter(
            x=daily["target_date"],
            y=daily["mc_p10_f"],
            fill="tonexty",
            fillcolor="rgba(156,106,222,0.20)",
            line={"width": 0},
            name="MC P10–P90",
        )
    )
    interval.add_trace(
        go.Scatter(
            x=daily["target_date"],
            y=daily["mc_p50_f"],
            mode="lines+markers",
            name="MC P50",
            line={"color": COLORS["MC"]},
        )
    )
    interval.add_trace(
        go.Scatter(
            x=daily["target_date"],
            y=daily["actual_high_f"],
            mode="markers",
            name="Actual",
            marker={"color": COLORS["Actual"], "size": 8},
        )
    )
    interval.update_layout(
        title="Daily MC interval and actual high",
        xaxis_title="Target day",
        yaxis_title="°F",
        hovermode="x unified",
    )
    st.plotly_chart(interval, width="stretch")

    errors = pd.concat(
        [
            daily[["target_date", column]]
            .rename(columns={column: "forecast_f"})
            .assign(model=name)
            for name, column in (
                ("HRRR", "hrrr_high_f"),
                ("TFT", "tft_high_f"),
                ("MC", "mc_p50_f"),
            )
        ]
    )
    actual_by_day = daily.set_index("target_date")["actual_high_f"]
    errors["absolute_error_f"] = (
        errors["forecast_f"]
        - errors["target_date"].map(actual_by_day)
    ).abs()
    st.plotly_chart(
        px.bar(
            errors,
            x="target_date",
            y="absolute_error_f",
            color="model",
            barmode="group",
            color_discrete_map=COLORS,
            labels={
                "target_date": "Target day",
                "absolute_error_f": "Absolute error (°F)",
                "model": "Model",
            },
            title="Daily absolute error by target day",
        ),
        width="stretch",
    )


def _performance(data: EvaluationData) -> None:
    st.subheader("TFT and MC performance")
    st.caption(
        "TFT is compared with raw HRRR at the hourly grain. MC is compared "
        "with TFT daily highs using out-of-fold cycle-day predictions."
    )
    table = model_performance_table(data).round(
        {
            "MAE (°F)": 2,
            "Baseline MAE (°F)": 2,
            "Skill vs baseline (%)": 1,
            "P10–P90 coverage (%)": 1,
            "Mean interval width (°F)": 2,
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
