"""Layered point-in-time attribution from hourly TFT values to daily high."""
from __future__ import annotations

import numpy as np
import pandas as pd


_PREDICTION_COLUMNS = {
    "init_time",
    "valid_time",
    "base_temp_c",
    "target",
    "p10",
    "p50",
    "p90",
}
_SNAPSHOT_COLUMNS = {
    "snapshot_id",
    "init_time",
    "target_date",
    "coverage_type",
    "observed_high_f",
    "observed_hour_count",
    "forecast_hour_count",
    "forecast_start_at",
    "forecast_through",
    "p10_f",
    "p50_f",
    "p90_f",
    "target_hour_count",
}
_SUMMARY_METRICS = (
    "hourly_p50_bias_f",
    "hourly_p50_mae_f",
    "hourly_p10_p90_coverage",
    "future_tft_max_bias_vs_iem_f",
    "future_tft_max_mae_vs_iem_f",
    "hybrid_tft_max_bias_vs_iem_f",
    "hybrid_tft_max_mae_vs_iem_f",
    "iem_weather_bias_f",
    "iem_weather_mae_f",
    "mc_p50_bias_vs_iem_f",
    "mc_p50_mae_vs_iem_f",
    "mc_p50_bias_vs_weather_f",
    "mc_p50_mae_vs_weather_f",
    "mc_p10_p90_coverage_vs_weather",
    "mc_p10_p90_width_f",
)


def build_daily_high_layer_attribution(
    *,
    predictions: pd.DataFrame,
    snapshots: pd.DataFrame,
    actuals: pd.DataFrame,
    station_timezone: str,
) -> pd.DataFrame:
    """Join frozen snapshots to their forecast hours and outcome layers.

    ``base_temp_c + target`` is the IEM-derived realized hourly temperature
    encoded by the TFT target contract.  Weather.com remains separate as the
    final market-settlement truth.
    """
    _require_columns(predictions, _PREDICTION_COLUMNS, "predictions")
    _require_columns(snapshots, _SNAPSHOT_COLUMNS, "snapshots")
    _require_columns(actuals, {"target_date", "actual_high_f"}, "actuals")
    if actuals["target_date"].astype(str).duplicated().any():
        raise ValueError("actuals contain duplicate target_date")

    forecast = predictions.loc[:, sorted(_PREDICTION_COLUMNS)].copy()
    forecast["init_time"] = pd.to_datetime(forecast["init_time"], utc=True)
    forecast["valid_time"] = pd.to_datetime(forecast["valid_time"], utc=True)
    forecast["target_date"] = (
        (forecast["valid_time"] - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(station_timezone)
        .dt.date.astype(str)
    )
    for column in ("base_temp_c", "target", "p10", "p50", "p90"):
        forecast[column] = pd.to_numeric(forecast[column], errors="raise")
    forecast["iem_actual_f"] = _c_to_f(
        forecast["base_temp_c"] + forecast["target"]
    )
    forecast["tft_p10_f"] = _c_to_f(forecast["base_temp_c"] + forecast["p10"])
    forecast["tft_p50_f"] = _c_to_f(forecast["base_temp_c"] + forecast["p50"])
    forecast["tft_p90_f"] = _c_to_f(forecast["base_temp_c"] + forecast["p90"])

    base = snapshots.loc[:, sorted(_SNAPSHOT_COLUMNS)].copy()
    base["init_time"] = pd.to_datetime(base["init_time"], utc=True)
    base["forecast_start_at"] = pd.to_datetime(
        base["forecast_start_at"], utc=True
    )
    base["forecast_through"] = pd.to_datetime(
        base["forecast_through"], utc=True
    )
    base["target_date"] = base["target_date"].astype(str)
    base["forecast_hour_count"] = pd.to_numeric(
        base["forecast_hour_count"], errors="raise"
    ).astype(int)
    for column in ("observed_high_f", "p10_f", "p50_f", "p90_f"):
        base[column] = pd.to_numeric(base[column], errors="raise")

    hours = base.merge(
        forecast,
        on=["init_time", "target_date"],
        how="left",
        validate="many_to_many",
    )
    hours = hours[
        (hours["valid_time"] >= hours["forecast_start_at"])
        & (hours["valid_time"] <= hours["forecast_through"])
    ].copy()
    counts = hours.groupby("snapshot_id", sort=False)["valid_time"].nunique()
    expected = base.set_index("snapshot_id")["forecast_hour_count"]
    if not counts.reindex(expected.index).eq(expected).all():
        bad = expected.index[~counts.reindex(expected.index).eq(expected)].tolist()
        raise ValueError(
            "snapshot forecast hours do not match attribution predictions: "
            + ", ".join(str(value) for value in bad[:5])
        )
    if hours.duplicated(["snapshot_id", "valid_time"]).any():
        raise ValueError("attribution has duplicate snapshot forecast hours")

    hourly = hours.assign(
        hourly_p50_error_f=hours["tft_p50_f"] - hours["iem_actual_f"],
        hourly_p50_absolute_error_f=(
            hours["tft_p50_f"] - hours["iem_actual_f"]
        ).abs(),
        hourly_p10_p90_coverage=(
            (hours["iem_actual_f"] >= hours["tft_p10_f"])
            & (hours["iem_actual_f"] <= hours["tft_p90_f"])
        ).astype(float),
    )
    future = hourly.groupby("snapshot_id", as_index=False).agg(
        future_iem_high_f=("iem_actual_f", "max"),
        raw_tft_max_p50_f=("tft_p50_f", "max"),
        hourly_p50_bias_f=("hourly_p50_error_f", "mean"),
        hourly_p50_mae_f=("hourly_p50_absolute_error_f", "mean"),
        hourly_p10_p90_coverage=("hourly_p10_p90_coverage", "mean"),
    )
    result = base.merge(future, on="snapshot_id", how="inner", validate="one_to_one")
    observed = result["observed_high_f"].fillna(-np.inf)
    result["hybrid_iem_high_f"] = np.maximum(
        observed, result["future_iem_high_f"]
    )
    result["hybrid_tft_max_p50_f"] = np.maximum(
        observed, result["raw_tft_max_p50_f"]
    )
    truth = actuals.loc[:, ["target_date", "actual_high_f"]].copy()
    truth["target_date"] = truth["target_date"].astype(str)
    truth["actual_high_f"] = pd.to_numeric(truth["actual_high_f"], errors="raise")
    result = result.merge(truth, on="target_date", how="left", validate="many_to_one")
    if result["actual_high_f"].isna().any():
        raise ValueError("attribution actuals do not cover every snapshot")
    result = result.rename(columns={"actual_high_f": "weather_actual_high_f"})
    result["future_tft_max_bias_vs_iem_f"] = (
        result["raw_tft_max_p50_f"] - result["future_iem_high_f"]
    )
    result["future_tft_max_mae_vs_iem_f"] = result[
        "future_tft_max_bias_vs_iem_f"
    ].abs()
    result["hybrid_tft_max_bias_vs_iem_f"] = (
        result["hybrid_tft_max_p50_f"] - result["hybrid_iem_high_f"]
    )
    result["hybrid_tft_max_mae_vs_iem_f"] = result[
        "hybrid_tft_max_bias_vs_iem_f"
    ].abs()
    result["iem_weather_bias_f"] = (
        result["hybrid_iem_high_f"] - result["weather_actual_high_f"]
    )
    result["iem_weather_mae_f"] = result["iem_weather_bias_f"].abs()
    result["mc_p50_bias_vs_iem_f"] = (
        result["p50_f"] - result["hybrid_iem_high_f"]
    )
    result["mc_p50_mae_vs_iem_f"] = result["mc_p50_bias_vs_iem_f"].abs()
    result["mc_p50_bias_vs_weather_f"] = (
        result["p50_f"] - result["weather_actual_high_f"]
    )
    result["mc_p50_mae_vs_weather_f"] = result[
        "mc_p50_bias_vs_weather_f"
    ].abs()
    result["mc_p10_p90_coverage_vs_weather"] = (
        (result["weather_actual_high_f"] >= result["p10_f"])
        & (result["weather_actual_high_f"] <= result["p90_f"])
    ).astype(float)
    result["mc_p10_p90_width_f"] = result["p90_f"] - result["p10_f"]
    return result.sort_values("snapshot_id").reset_index(drop=True)


def summarize_daily_high_layer_attribution(rows: pd.DataFrame) -> dict:
    """Return snapshot-weighted descriptive layer diagnostics by fixed strata."""
    _require_columns(
        rows,
        {"target_date", "coverage_type", "forecast_hour_count"},
        "attribution rows",
    )
    work = rows.copy()
    work["target_date"] = work["target_date"].astype(str)
    work["month"] = pd.to_datetime(work["target_date"]).dt.to_period("M").astype(str)
    work["horizon"] = work["forecast_hour_count"].map(_horizon_band)
    metrics = [column for column in _SUMMARY_METRICS if column in work]
    return {
        "overall": _summary_records(work, [], metrics)[0],
        "by_month": _summary_records(work, ["month"], metrics),
        "by_coverage": _summary_records(work, ["coverage_type"], metrics),
        "by_horizon": _summary_records(work, ["horizon"], metrics),
    }


def _summary_records(
    rows: pd.DataFrame,
    dimensions: list[str],
    metrics: list[str],
) -> list[dict[str, object]]:
    if dimensions:
        groups = rows.groupby(dimensions, sort=True, dropna=False)
    else:
        groups = [((), rows)]
    result: list[dict[str, object]] = []
    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(dimensions, (str(value) for value in key)))
        row["snapshot_count"] = int(len(group))
        row["event_day_count"] = int(group["target_date"].nunique())
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        result.append(row)
    return result


def _horizon_band(value: object) -> str:
    count = int(value)
    if count <= 0:
        return "0h"
    if count <= 6:
        return "1-6h"
    if count <= 12:
        return "7-12h"
    if count <= 18:
        return "13-18h"
    return "19-24h"


def _c_to_f(values: pd.Series) -> pd.Series:
    return values.astype(float) * 1.8 + 32.0


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
