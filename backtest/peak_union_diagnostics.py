"""Read-only peak-state and contiguous-bucket-union diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


_UNION_REQUIRED = {
    "event_id", "target_date", "timestamp", "bucket_order",
    "model_probability", "midpoint_probability", "yes_outcome", "lead_band",
}


def build_contiguous_union_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return every proper contiguous bucket union for each complete vector."""
    missing = _UNION_REQUIRED - set(panel)
    if missing:
        raise ValueError("signal panel missing columns: " + ", ".join(sorted(missing)))
    rows: list[dict[str, object]] = []
    keys = ["event_id", "target_date", "timestamp"]
    for key, vector in panel.groupby(keys, sort=True):
        ordered = vector.sort_values("bucket_order", kind="stable").reset_index(drop=True)
        orders = ordered["bucket_order"].to_numpy(int)
        if not np.array_equal(orders, np.arange(len(ordered))):
            raise ValueError("union diagnostics require contiguous bucket_order vectors")
        if not np.isclose(ordered["model_probability"].sum(), 1.0, atol=1e-6):
            raise ValueError("model probability vector must sum to one")
        if not np.isclose(ordered["midpoint_probability"].sum(), 1.0, atol=1e-6):
            raise ValueError("midpoint probability vector must sum to one")
        if not np.isclose(ordered["yes_outcome"].sum(), 1.0, atol=1e-6):
            raise ValueError("union diagnostics require exactly one settled bucket")
        count = len(ordered)
        for start in range(count):
            for end in range(start, count):
                if start == 0 and end == count - 1:
                    continue
                segment = ordered.iloc[start : end + 1]
                rows.append(
                    {
                        "event_id": str(key[0]),
                        "target_date": str(key[1]),
                        "timestamp": pd.Timestamp(key[2]),
                        "snapshot_id": str(segment["snapshot_id"].iloc[0]) if "snapshot_id" in segment else None,
                        "lead_band": str(segment["lead_band"].iloc[0]),
                        "union_start": start,
                        "union_end": end,
                        "union_width": end - start + 1,
                        "bucket_count": count,
                        "union_position": _union_position(start, end, count),
                        "model_probability": float(segment["model_probability"].sum()),
                        "midpoint_probability": float(segment["midpoint_probability"].sum()),
                        "yes_outcome": float(segment["yes_outcome"].sum()),
                    }
                )
    return pd.DataFrame(rows)


def build_peak_snapshot_diagnostics(
    *, snapshots: pd.DataFrame, predictions: pd.DataFrame, observations: pd.DataFrame,
    station_timezone: str,
) -> pd.DataFrame:
    """Construct ex-ante peak features from snapshot-visible hourly inputs.

    The result intentionally contains no settled outcome.  Callers join truth
    only after this point, keeping availability constraints auditable.
    """
    required_snapshots = {"snapshot_id", "init_time", "available_at", "target_date"}
    required_predictions = {"init_time", "valid_time", "p10", "p50", "p90", "base_temp_c"}
    required_observations = {"valid_time", "available_at", "temperature_c"}
    for name, frame, required in (
        ("snapshots", snapshots, required_snapshots),
        ("predictions", predictions, required_predictions),
        ("observations", observations, required_observations),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} missing columns: " + ", ".join(sorted(missing)))
    preds = predictions.copy()
    preds["init_time"] = pd.to_datetime(preds["init_time"], utc=True)
    preds["valid_time"] = pd.to_datetime(preds["valid_time"], utc=True)
    obs = observations.copy()
    obs["valid_time"] = pd.to_datetime(obs["valid_time"], utc=True)
    obs["available_at"] = pd.to_datetime(obs["available_at"], utc=True)
    prediction_cycles = {key: group.copy() for key, group in preds.groupby("init_time", sort=False)}
    rows: list[dict[str, object]] = []
    for snapshot in snapshots.itertuples(index=False):
        init = pd.Timestamp(snapshot.init_time)
        if init.tzinfo is None:
            init = init.tz_localize("UTC")
        else:
            init = init.tz_convert("UTC")
        available = pd.Timestamp(snapshot.available_at)
        if available.tzinfo is None:
            available = available.tz_localize("UTC")
        else:
            available = available.tz_convert("UTC")
        target_date = str(snapshot.target_date)
        expected = _target_hours(target_date, station_timezone)
        cycle = prediction_cycles.get(init)
        if cycle is None:
            continue
        target_cycle = cycle[cycle["valid_time"].isin(expected)].copy()
        future = target_cycle[target_cycle["valid_time"] > init].copy()
        if future.empty or future["valid_time"].duplicated().any():
            continue
        visible_obs = obs[
            obs["valid_time"].isin(expected)
            & (obs["valid_time"] <= init)
            & (obs["available_at"] <= init)
        ].sort_values("valid_time").drop_duplicates("valid_time", keep="last")
        tft_c = future["base_temp_c"].to_numpy(float) + future["p50"].to_numpy(float)
        hrrr_c = future["base_temp_c"].to_numpy(float)
        peak = float(np.max(tft_c))
        peak_indices = np.flatnonzero(np.isclose(tft_c, peak, atol=1e-6))
        hrrr_peak = float(np.max(hrrr_c))
        hrrr_indices = np.flatnonzero(np.isclose(hrrr_c, hrrr_peak, atol=1e-6))
        ranked = np.sort(tft_c)
        second_peak = float(ranked[-2]) if len(ranked) > 1 else peak
        observed_high_f = (
            float(visible_obs["temperature_c"].max()) * 1.8 + 32.0
            if not visible_obs.empty else float("nan")
        )
        peak_f = peak * 1.8 + 32.0
        future_times = future["valid_time"].to_numpy()
        full_tft_c = target_cycle["base_temp_c"].to_numpy(float) + target_cycle["p50"].to_numpy(float)
        full_peak_times = target_cycle.loc[np.isclose(full_tft_c, np.max(full_tft_c), atol=1e-6), "valid_time"]
        rows.append({
            "snapshot_id": str(snapshot.snapshot_id),
            "init_time": init,
            "available_at": available,
            "target_date": target_date,
            "coverage_type": str(getattr(snapshot, "coverage_type", "unknown")),
            "observed_hour_count": int(len(visible_obs)),
            "forecast_hour_count": int(len(future)),
            "tft_peak_f": peak_f,
            "hrrr_peak_f": hrrr_peak * 1.8 + 32.0,
            "tft_hrrr_peak_gap_f": (peak - hrrr_peak) * 1.8,
            "tft_peak_slot_at": pd.Timestamp(future_times[int(peak_indices[0])]),
            "hrrr_peak_slot_at": pd.Timestamp(future_times[int(hrrr_indices[0])]),
            "tft_hrrr_peak_hour_gap": float(abs(int(peak_indices[0]) - int(hrrr_indices[0]))),
            "tft_peak_margin_f": (peak - second_peak) * 1.8,
            "tft_near_peak_hour_count_1f": int(np.sum((peak - tft_c) * 1.8 <= 1.0)),
            "tft_peak_interval_width_f": float((future["p90"].to_numpy(float)[peak_indices].mean() - future["p10"].to_numpy(float)[peak_indices].mean()) * 1.8),
            "observed_high_f": observed_high_f,
            "observed_future_peak_gap_f": observed_high_f - peak_f if np.isfinite(observed_high_f) else float("nan"),
            "forecast_peak_elapsed": bool((full_peak_times <= init).any()),
        })
    return pd.DataFrame(rows)


def attach_peak_outcomes(
    diagnostics: pd.DataFrame, *, observations: pd.DataFrame,
    settlement_actuals: pd.DataFrame, station_timezone: str,
) -> pd.DataFrame:
    """Attach ex-post peak-time labels only when IEM and settlement agree."""
    required_diagnostics = {"target_date", "tft_peak_slot_at", "hrrr_peak_slot_at"}
    required_observations = {"valid_time", "temperature_c"}
    required_actuals = {"target_date", "actual_high_f"}
    for name, frame, required in (
        ("diagnostics", diagnostics, required_diagnostics),
        ("observations", observations, required_observations),
        ("settlement_actuals", settlement_actuals, required_actuals),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} missing columns: " + ", ".join(sorted(missing)))
    obs = observations.loc[:, ["valid_time", "temperature_c"]].copy()
    obs["valid_time"] = pd.to_datetime(obs["valid_time"], utc=True)
    # The target day is right-closed: midnight belongs to the preceding day.
    obs["target_date"] = (
        (obs["valid_time"] - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(station_timezone).dt.date.astype(str)
    )
    rows: list[dict[str, object]] = []
    actual_lookup = settlement_actuals.copy()
    actual_lookup["target_date"] = actual_lookup["target_date"].astype(str)
    actual_lookup = actual_lookup.drop_duplicates("target_date").set_index("target_date")["actual_high_f"]
    for target_date, group in obs.groupby("target_date", sort=False):
        if target_date not in actual_lookup.index:
            continue
        max_c = float(group["temperature_c"].max())
        actual_peak_times = group.loc[np.isclose(group["temperature_c"], max_c, atol=1e-6), "valid_time"]
        iem_high_f = max_c * 1.8 + 32.0
        settled_high_f = float(actual_lookup.loc[target_date])
        agrees = int(np.floor(iem_high_f + 0.5)) == int(np.floor(settled_high_f + 0.5))
        rows.append({
            "target_date": target_date,
            "settlement_actual_high_f": settled_high_f,
            "iem_actual_high_f": iem_high_f,
            "peak_time_label_available": bool(agrees),
            "actual_peak_slots": tuple(pd.Timestamp(value).isoformat() for value in actual_peak_times),
        })
    labels = pd.DataFrame(rows)
    result = diagnostics.merge(labels, on="target_date", how="left", validate="many_to_one")
    for source in ("tft", "hrrr"):
        values = []
        for row in result.itertuples(index=False):
            label_available = getattr(row, "peak_time_label_available", False)
            if pd.isna(label_available) or not bool(label_available):
                values.append(float("nan"))
                continue
            predicted = pd.Timestamp(getattr(row, f"{source}_peak_slot_at"))
            actual_slots = [pd.Timestamp(value) for value in getattr(row, "actual_peak_slots")]
            values.append(min(abs((predicted - actual).total_seconds()) / 3600.0 for actual in actual_slots))
        result[f"{source}_peak_time_error_h"] = values
    if "actual_peak_slots" in result:
        result["actual_peak_slots"] = result["actual_peak_slots"].map(
            lambda value: json.dumps(list(value)) if isinstance(value, tuple) else value
        )
    return result


def attach_forecast_stability(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Attach previous-visible-cycle quantile changes without outcome inputs."""
    required = {"snapshot_id", "target_date", "available_at", "p10_f", "p50_f", "p90_f"}
    missing = required - set(diagnostics)
    if missing:
        raise ValueError("stability diagnostics missing columns: " + ", ".join(sorted(missing)))
    result = diagnostics.copy().sort_values(["target_date", "available_at", "snapshot_id"], kind="stable")
    grouped = result.groupby("target_date", sort=False)
    previous = grouped[["p10_f", "p50_f", "p90_f"]].shift(1)
    for column in ("p10_f", "p50_f", "p90_f"):
        result[f"forecast_stability_{column}_delta_f"] = result[column] - previous[column]
    result["forecast_stability_quantile_l1_f"] = (
        result[["forecast_stability_p10_f_delta_f", "forecast_stability_p50_f_delta_f", "forecast_stability_p90_f_delta_f"]]
        .abs().mean(axis=1)
    )
    return result


def assign_peak_state_groups(
    diagnostics: pd.DataFrame, *, thresholds: dict[str, list[float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Freeze low/middle/high ex-ante state bands for reporting only."""
    features = (
        "tft_peak_interval_width_f", "tft_peak_margin_f",
        "tft_hrrr_peak_gap_f", "forecast_stability_quantile_l1_f",
    )
    result = diagnostics.copy()
    contract = {} if thresholds is None else {key: list(value) for key, value in thresholds.items()}
    for feature in features:
        if feature not in result:
            continue
        values = result[feature].abs() if feature == "tft_hrrr_peak_gap_f" else result[feature]
        if feature not in contract:
            finite = values.dropna()
            contract[feature] = [float(finite.quantile(0.25)), float(finite.quantile(0.75))] if not finite.empty else [float("nan"), float("nan")]
        low, high = contract[feature]
        result[f"{feature}_group"] = np.select(
            [values.isna(), values <= low, values >= high],
            ["unavailable", "low", "high"], default="middle",
        )
    return result, contract


def summarize_peak_groups(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Event-day-equal descriptive performance by frozen ex-ante groups."""
    metrics = [column for column in (
        "normalized_rps", "p50_abs_error_f", "coverage80", "true_bucket_probability",
    ) if column in diagnostics]
    if not metrics:
        return pd.DataFrame(columns=["feature", "group", "event_days", "snapshots"])
    rows: list[dict[str, object]] = []
    for group_column in sorted(column for column in diagnostics if column.endswith("_group")):
        for label, group in diagnostics.groupby(group_column, sort=True):
            by_day = group.groupby("target_date", as_index=False)[metrics].mean()
            row = {
                "feature": group_column.removesuffix("_group"), "group": str(label),
                "event_days": int(len(by_day)), "snapshots": int(len(group)),
            }
            row.update({metric: float(by_day[metric].mean()) for metric in metrics})
            rows.append(row)
    return pd.DataFrame(rows)


def score_union_panel(
    unions: pd.DataFrame, *, bootstrap_samples: int = 5_000, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Score union probabilities and summarize paired Event-day differences."""
    required = {
        "event_id", "target_date", "timestamp", "model_probability",
        "midpoint_probability", "yes_outcome", "union_width", "union_position",
        "lead_band",
    }
    missing = required - set(unions)
    if missing:
        raise ValueError("union panel missing columns: " + ", ".join(sorted(missing)))
    work = unions.copy()
    for source in ("model", "midpoint"):
        probability = work[f"{source}_probability"].to_numpy(float)
        if (
            not np.isfinite(probability).all()
            or ((probability < -1e-12) | (probability > 1.0 + 1e-12)).any()
        ):
            raise ValueError(f"{source} union probabilities must be finite in [0, 1]")
        probability = np.clip(probability, 0.0, 1.0)
        work[f"{source}_probability"] = probability
        clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
        truth = work["yes_outcome"].to_numpy(float)
        work[f"{source}_brier"] = (probability - truth) ** 2
        work[f"{source}_log_score"] = -(truth * np.log(clipped) + (1.0 - truth) * np.log1p(-clipped))
    work["brier_delta"] = work["model_brier"] - work["midpoint_brier"]
    work["log_score_delta"] = work["model_log_score"] - work["midpoint_log_score"]
    summary = _union_summary(work, bootstrap_samples=bootstrap_samples, seed=seed)
    strata = _union_strata(work, bootstrap_samples=bootstrap_samples, seed=seed)
    return work, strata, summary


def write_peak_union_artifact(
    *, output: Path | str, snapshot_diagnostics: pd.DataFrame,
    peak_outcomes: pd.DataFrame, union_scores: pd.DataFrame, union_strata: pd.DataFrame,
    report: dict[str, object], input_sha256: dict[str, str],
) -> dict[str, object]:
    """Write a self-validating read-only diagnostic artifact."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    payloads = {
        "snapshot_diagnostics.parquet": snapshot_diagnostics,
        "peak_outcomes.parquet": peak_outcomes,
        "union_scores.parquet": union_scores,
        "union_strata.parquet": union_strata,
    }
    hashes: dict[str, str] = {}
    for name, frame in payloads.items():
        path = root / name
        frame.to_parquet(path, index=False)
        hashes[name] = _sha256(path)
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="ascii")
    hashes[report_path.name] = _sha256(report_path)
    manifest = {
        "schema_version": 1,
        "data_class": "peak_union_diagnostics",
        "input_sha256": dict(sorted(input_sha256.items())),
        "sha256": hashes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii")
    return manifest


def _union_summary(work: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> dict[str, object]:
    by_day = work.groupby("target_date", as_index=False)[["brier_delta", "log_score_delta"]].mean()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(by_day), size=(bootstrap_samples, len(by_day)))
    bootstrap = by_day[["brier_delta", "log_score_delta"]].to_numpy(float)[draws].mean(axis=1)
    return {
        "event_days": int(len(by_day)), "unions": int(len(work)),
        "paired": {
            metric: {
                "model_minus_midpoint": float(work[metric].mean()),
                "ci90_low": float(np.quantile(bootstrap[:, index], 0.05)),
                "ci90_high": float(np.quantile(bootstrap[:, index], 0.95)),
                "bootstrap_probability_model_better": float(np.mean(bootstrap[:, index] < 0.0)),
            }
            for index, metric in enumerate(("brier_delta", "log_score_delta"))
        },
    }


def _union_strata(work: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (keys, group) in enumerate(work.groupby(["union_width", "union_position", "lead_band"], sort=True)):
        summary = _union_summary(group, bootstrap_samples=bootstrap_samples, seed=seed + index)
        rows.append({
            "union_width": int(keys[0]), "union_position": str(keys[1]), "lead_band": str(keys[2]),
            "event_days": summary["event_days"], "unions": summary["unions"],
            "brier_model_minus_midpoint": summary["paired"]["brier_delta"]["model_minus_midpoint"],
            "brier_ci90_low": summary["paired"]["brier_delta"]["ci90_low"],
            "brier_ci90_high": summary["paired"]["brier_delta"]["ci90_high"],
            "log_score_model_minus_midpoint": summary["paired"]["log_score_delta"]["model_minus_midpoint"],
        })
    return pd.DataFrame(rows)


def _union_position(start: int, end: int, count: int) -> str:
    midpoint = (start + end + 1) / 2.0 / count
    if midpoint < 1 / 3:
        return "lower"
    if midpoint > 2 / 3:
        return "upper"
    return "middle"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_hours(target_date: str, timezone: str) -> pd.DatetimeIndex:
    zone = ZoneInfo(timezone)
    start = pd.Timestamp(target_date).tz_localize(zone)
    end = start + pd.DateOffset(days=1)
    return pd.date_range(start, end, freq="h", inclusive="right").tz_convert("UTC")
