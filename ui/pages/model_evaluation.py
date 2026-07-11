"""Offline HRRR, TFT, and MC validation analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings
from ui.model_evaluation import (
    DailyHighPmfPilotData,
    EvaluationData,
    SignalEvaluationData,
    daily_target_summary,
    discover_evaluations,
    evaluation_metrics,
    load_backtest_model_evaluation,
    load_daily_high_pmf_pilot,
    load_signal_evaluation,
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
    "PMF pilot": "#59a14f",
}


def render(
    *,
    runs_root: Path | str | None = None,
    signal_roots: list[Path] | tuple[Path, ...] | None = None,
    use_may_backtest: bool | None = None,
) -> None:
    st.header("Model Evaluation")
    st.caption(
        "Validation-only diagnostics. MC daily-high results are contiguous-fold "
        "out-of-fold estimates for complete station-local calendar days."
    )
    settings = Settings()
    root = Path(runs_root or settings.project_root / "runs")
    prefer_may = (runs_root is None) if use_may_backtest is None else use_may_backtest
    data: EvaluationData | None = None
    if prefer_may:
        try:
            data = load_backtest_model_evaluation(
                **_default_may_backtest_inputs(settings.project_root)
            )
        except (OSError, ValueError, KeyError, FileNotFoundError):
            data = None

    artifacts = discover_evaluations(root)
    if data is None and not artifacts:
        st.info(f"No complete calibrated evaluations found under {root}.")
        return
    if data is None:
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
    else:
        st.caption(
            "Showing KLGA May 2026 model_test backtest predictions by default: "
            "fixed hourly TFT outputs plus MC daily-high snapshots."
        )

    signal_data = load_signal_evaluation(
        tuple(signal_roots)
        if signal_roots is not None
        else _default_may_signal_roots(settings.project_root)
        if runs_root is None
        else tuple()
    )
    pmf_pilot: DailyHighPmfPilotData | None = None
    if runs_root is None:
        try:
            pmf_pilot = load_daily_high_pmf_pilot(
                settings.project_root
                / "data_store"
                / "backtest"
                / "daily_high_pmf"
                / "KLGA"
                / "pilot_v1"
            )
        except (OSError, ValueError, KeyError):
            pmf_pilot = None

    tab_labels = [
        "Overview",
        "Hourly & Horizon",
        "Daily High & MC",
        "TFT & MC Performance",
    ]
    if not signal_data.events.empty:
        tab_labels.append("May Model-Test Signal")
    if pmf_pilot is not None:
        tab_labels.append("Daily High PMF Pilot")
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        _overview(data)
    with tabs[1]:
        _hourly(data)
    with tabs[2]:
        _daily(data)
    with tabs[3]:
        _performance(data)
    next_tab = 4
    if not signal_data.events.empty:
        with tabs[next_tab]:
            _may_signal(signal_data)
        next_tab += 1
    if pmf_pilot is not None:
        with tabs[next_tab]:
            _pmf_pilot(pmf_pilot)


def _default_may_signal_roots(project_root: Path) -> tuple[Path, ...]:
    signal_root = project_root / "data_store" / "backtest" / "signal" / "KLGA"
    return (
        signal_root / "model_test_20260501_20260516",
        signal_root / "model_test_20260519_20260531",
    )


def _default_may_backtest_inputs(project_root: Path) -> dict[str, object]:
    return {
        "prediction_paths": (
            project_root
            / "data_store"
            / "backtest"
            / "predictions"
            / "KLGA"
            / "20260501_20260531_fixed_hourly.parquet",
        ),
        "forecast_root": (
            project_root
            / "data_store"
            / "backtest"
            / "forecasts"
            / "KLGA"
            / "20260501_20260604_model_test"
        ),
        "actuals_path": (
            project_root
            / "data_store"
            / "backtest"
            / "actuals"
            / "KLGA"
            / "20260401_20260604.csv"
        ),
    }


def _overview(data: EvaluationData) -> None:
    metrics = evaluation_metrics(data)
    mc_label = "MC high MAE" if data.calibration is None else "MC high MAE (OOF)"
    mc_coverage_label = (
        "MC high coverage"
        if data.calibration is None
        else "MC high coverage (OOF)"
    )
    cols = st.columns(5)
    cols[0].metric("HRRR hourly MAE", f"{metrics['hrrr_hourly_mae_f']:.2f} °F")
    cols[1].metric(
        "TFT hourly MAE",
        f"{metrics['tft_hourly_mae_f']:.2f} °F",
        f"{metrics['hourly_skill_pct']:.1f}% skill",
    )
    cols[2].metric("TFT P10–P90 coverage", f"{metrics['hourly_coverage']:.1%}")
    cols[3].metric(mc_label, f"{metrics['mc_high_mae_f']:.2f} °F")
    cols[4].metric(mc_coverage_label, f"{metrics['mc_coverage']:.1%}")

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
    if data.calibration is None:
        st.caption(
            "Charts show one point per target day: forecasts and MC bounds are "
            "the median across available May model_test cycles; actual high is "
            "the observed daily value. Summary metrics above retain the "
            "cycle-day evaluation grain."
        )
    else:
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
    if data.calibration is None:
        st.caption(
            "TFT is compared with raw HRRR at the hourly grain. MC is compared "
            "with TFT daily highs using May model_test cycle-day predictions."
        )
    else:
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


def _pmf_pilot(data: DailyHighPmfPilotData) -> None:
    st.subheader("Point-in-time daily-high PMF pilot")
    gates = data.report["gates"]
    eligible = bool(gates["eligible_for_cross_fit_goal"])
    st.caption(
        "April chronological OOF and frozen May diagnostic. June is unconsumed; "
        "this candidate is offline evidence and is not a production probability source."
    )
    may = data.may_summary.set_index("stratum").loc["overall"]
    cols = st.columns(5)
    cols[0].metric("Cross-fit eligible", "Yes" if eligible else "No")
    cols[1].metric("PMF RPS", f"{may['candidate_rps']:.3f}")
    cols[2].metric("MC RPS", f"{may['mc_rps']:.3f}")
    cols[3].metric("PMF high MAE", f"{may['candidate_mae_f']:.2f} °F")
    cols[4].metric("PMF coverage", f"{may['candidate_coverage']:.1%}")

    strata = data.may_summary.copy()
    rps = strata.melt(
        id_vars=["stratum"],
        value_vars=["candidate_rps", "mc_rps"],
        var_name="source",
        value_name="rps",
    )
    rps["source"] = rps["source"].map(
        {"candidate_rps": "PMF pilot", "mc_rps": "MC"}
    )
    st.plotly_chart(
        px.bar(
            rps,
            x="stratum",
            y="rps",
            color="source",
            barmode="group",
            color_discrete_map={
                "PMF pilot": COLORS["PMF pilot"],
                "MC": COLORS["MC"],
            },
            labels={"stratum": "Stratum", "rps": "RPS", "source": "Source"},
            title="Frozen May RPS by point-in-time stratum",
        ),
        width="stretch",
    )

    daily = (
        data.may_scores.groupby("target_date", as_index=False)
        .agg(
            actual_high_f=("actual_high_f", "first"),
            pmf_p50_f=("candidate_p50_f", "median"),
            mc_p50_f=("mc_p50_f", "median"),
        )
        .sort_values("target_date")
    )
    daily_long = daily.melt(
        id_vars=["target_date", "actual_high_f"],
        value_vars=["pmf_p50_f", "mc_p50_f"],
        var_name="source",
        value_name="forecast_high_f",
    )
    daily_long["source"] = daily_long["source"].map(
        {"pmf_p50_f": "PMF pilot", "mc_p50_f": "MC"}
    )
    st.plotly_chart(
        px.line(
            daily_long,
            x="target_date",
            y="forecast_high_f",
            color="source",
            markers=True,
            color_discrete_map={
                "PMF pilot": COLORS["PMF pilot"],
                "MC": COLORS["MC"],
            },
            labels={
                "target_date": "Target day",
                "forecast_high_f": "P50 daily high (°F)",
                "source": "Source",
            },
            title="Frozen May PMF and MC daily-high medians",
        ).add_scatter(
            x=daily["target_date"],
            y=daily["actual_high_f"],
            mode="markers",
            name="Actual",
            marker={"color": COLORS["Actual"], "size": 7},
        ),
        width="stretch",
    )
    gate_table = pd.DataFrame(
        [{"Gate": key, "Passed": bool(value)} for key, value in gates.items()]
    )
    st.dataframe(gate_table, width="stretch", hide_index=True)


def _may_signal(data: SignalEvaluationData) -> None:
    st.subheader("May model-test signal")
    st.caption(
        "Frozen inspection of model_test signal artifacts. Score deltas are "
        "model minus normalized synthetic midpoint; lower is better for Brier, "
        "log score, and RPS."
    )
    events = data.events.copy()
    cols = st.columns(5)
    cols[0].metric("Events", f"{events['event_id'].nunique():d}")
    cols[1].metric("Brier delta", f"{events['brier_delta'].mean():+.3f}")
    cols[2].metric("Log delta", f"{events['log_score_delta'].mean():+.3f}")
    cols[3].metric("RPS delta", f"{events['rps_delta'].mean():+.3f}")
    cols[4].metric(
        "True Bucket P delta",
        f"{events['true_bucket_probability_delta'].mean():+.3f}",
    )

    st.plotly_chart(
        px.bar(
            events,
            x="target_date",
            y="brier_delta",
            color="artifact",
            labels={
                "target_date": "Target day",
                "brier_delta": "Brier delta",
                "artifact": "Artifact",
            },
            title="May event Brier delta by target day",
        ).add_hline(y=0, line_dash="dot", line_color="#666666"),
        width="stretch",
    )
    true_prob = events.melt(
        id_vars=["event_id", "target_date"],
        value_vars=[
            "model_true_bucket_probability",
            "midpoint_true_bucket_probability",
        ],
        var_name="source",
        value_name="true_bucket_probability",
    )
    true_prob["source"] = true_prob["source"].map(
        {
            "model_true_bucket_probability": "Model",
            "midpoint_true_bucket_probability": "Midpoint",
        }
    )
    st.plotly_chart(
        px.line(
            true_prob,
            x="target_date",
            y="true_bucket_probability",
            color="source",
            markers=True,
            color_discrete_map={
                "Model": COLORS["TFT"],
                "Midpoint": "#59a14f",
            },
            labels={
                "target_date": "Target day",
                "true_bucket_probability": "Probability on true Bucket",
                "source": "Source",
            },
            title="Probability assigned to the resolved Bucket",
        ),
        width="stretch",
    )

    worst = events.sort_values("brier_delta", ascending=False).head(10)
    st.dataframe(
        worst[
            [
                "target_date",
                "actual_high_f",
                "brier_delta",
                "log_score_delta",
                "rps_delta",
                "true_bucket_probability_delta",
                "model_true_bucket_probability",
                "midpoint_true_bucket_probability",
            ]
        ].round(4),
        width="stretch",
        hide_index=True,
    )

    options = [
        f"{row.target_date.date()} | {row.event_id} | "
        f"Brier {row.brier_delta:+.3f}"
        for row in worst.itertuples(index=False)
    ]
    selected_label = st.selectbox("Inspect event distribution", options)
    selected_event = selected_label.split(" | ")[1]
    _event_distribution(data.panel, selected_event)


def _event_distribution(panel: pd.DataFrame, event_id: str) -> None:
    event = panel[panel["event_id"].astype(str) == str(event_id)].copy()
    if event.empty:
        st.info("No panel rows found for selected event.")
        return
    target_date = event["target_date"].iloc[0].date()
    actual_high = float(event["actual_high_f"].iloc[0])
    st.caption(f"Selected Event {event_id}: {target_date}, actual high {actual_high:.0f} F")
    ticks = list(event.groupby("timestamp", sort=True))
    selected_ticks = [ticks[0]]
    if len(ticks) > 1:
        selected_ticks.append(ticks[-1])
    for timestamp, tick in selected_ticks:
        bucket = tick.sort_values("bucket_order").copy()
        bucket["bucket"] = bucket.apply(_bucket_label, axis=1)
        bucket_long = bucket.melt(
            id_vars=["bucket", "yes_outcome"],
            value_vars=["model_probability", "midpoint_probability"],
            var_name="source",
            value_name="probability",
        )
        bucket_long["source"] = bucket_long["source"].map(
            {
                "model_probability": "Model",
                "midpoint_probability": "Midpoint",
            }
        )
        figure = px.bar(
            bucket_long,
            x="bucket",
            y="probability",
            color="source",
            barmode="group",
            color_discrete_map={
                "Model": COLORS["TFT"],
                "Midpoint": "#59a14f",
            },
            labels={
                "bucket": "Bucket",
                "probability": "Probability",
                "source": "Source",
            },
            title=(
                "Bucket distribution at "
                f"{pd.Timestamp(timestamp).isoformat()}"
            ),
        )
        true_indexes = bucket.index[bucket["yes_outcome"].astype(float) == 1.0]
        if len(true_indexes) == 1:
            true_bucket = bucket.loc[true_indexes[0], "bucket"]
            figure.add_vline(
                x=bucket["bucket"].tolist().index(true_bucket),
                line_dash="dot",
                line_color=COLORS["Actual"],
            )
        st.plotly_chart(figure, width="stretch")


def _bucket_label(row: pd.Series) -> str:
    lo = row["bucket_lo_f"]
    hi = row["bucket_hi_f"]
    if pd.isna(lo) or lo == float("-inf"):
        return f"<{hi:.0f}"
    if pd.isna(hi) or hi == float("inf"):
        return f">={lo:.0f}"
    return f"{lo:.0f}-{hi:.0f}"
