"""Outcome scoring and ex-ante diagnostic summaries for conditional path MC."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.forecast_cache import ForecastCache


def actual_daily_highs(
    observations: pd.DataFrame, *, station_timezone: str,
    start_target: str, end_target: str,
) -> pd.DataFrame:
    """Derive settlement-day highs without looking at forecast availability."""
    required = {"valid_time", "temperature_c"}
    missing = required - set(observations)
    if missing:
        raise ValueError("observations missing columns: " + ", ".join(sorted(missing)))
    work = observations.loc[:, ["valid_time", "temperature_c"]].copy()
    work["valid_time"] = pd.to_datetime(work["valid_time"], utc=True)
    work["target_date"] = work["valid_time"].dt.tz_convert(station_timezone).dt.date.astype(str)
    work = work[work["target_date"].between(start_target, end_target)]
    result = work.groupby("target_date", as_index=False)["temperature_c"].max()
    result["actual_high_f"] = result.pop("temperature_c").astype(float) * 1.8 + 32.0
    return result


def score_parallel_bucket_distributions(
    *, candidate: ForecastCache, baseline: ForecastCache, markets: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Return paired proper scores on matching snapshot/bucket panels."""
    required = {"target_date", "condition_id", "bucket_lo_f", "bucket_hi_f"}
    missing = required - set(markets)
    if missing:
        raise ValueError("markets missing columns: " + ", ".join(sorted(missing)))
    actual_required = {"target_date", "actual_high_f"}
    if actual_required - set(actuals):
        raise ValueError("actuals missing target_date or actual_high_f")
    frame = _cache_probabilities(candidate, "candidate").merge(
        _cache_probabilities(baseline, "baseline"),
        on=["init_time", "target_date", "condition_id"], how="inner", validate="one_to_one",
    )
    frame = frame.merge(markets.loc[:, list(required)].drop_duplicates(["target_date", "condition_id"]), on=["target_date", "condition_id"], how="inner", validate="many_to_one")
    frame = frame.merge(actuals.loc[:, ["target_date", "actual_high_f"]], on="target_date", how="inner", validate="many_to_one")
    settled = np.floor(frame["actual_high_f"].to_numpy(float) + .5)
    truth = ((settled >= frame["bucket_lo_f"].to_numpy(float)) & (settled < frame["bucket_hi_f"].to_numpy(float))).astype(float)
    for source in ("candidate", "baseline"):
        probability = frame[f"{source}_probability"].to_numpy(float).clip(1e-6, 1.0 - 1e-6)
        frame[f"{source}_brier"] = (probability - truth) ** 2
        frame[f"{source}_log_loss"] = -(truth * np.log(probability) + (1.0 - truth) * np.log1p(-probability))
    frame["truth_yes"] = truth
    frame = _attach_snapshot_rps(frame)
    return frame


def summarize_conditional_diagnostics(scores: pd.DataFrame, *, bootstrap_samples: int = 2_000, seed: int = 42) -> dict[str, object]:
    """Event-day paired score and ex-ante diagnostic summary, no threshold search."""
    if scores.empty:
        raise ValueError("conditional score table is empty")
    required = {"target_date", "candidate_brier", "baseline_brier", "candidate_log_loss", "baseline_log_loss", "candidate_rps", "baseline_rps"}
    missing = required - set(scores)
    if missing:
        raise ValueError("conditional score table missing columns: " + ", ".join(sorted(missing)))
    work = scores.copy()
    work["brier_delta"] = work["candidate_brier"] - work["baseline_brier"]
    work["log_loss_delta"] = work["candidate_log_loss"] - work["baseline_log_loss"]
    snapshot_scores = work.drop_duplicates(["init_time", "target_date"]).copy()
    snapshot_scores["rps_delta"] = snapshot_scores["candidate_rps"] - snapshot_scores["baseline_rps"]
    snapshot_scores["p50_abs_error_delta"] = (
        (snapshot_scores["candidate_p50_f"] - snapshot_scores["actual_high_f"]).abs()
        - (snapshot_scores["baseline_p50_f"] - snapshot_scores["actual_high_f"]).abs()
    )
    snapshot_scores["coverage80_delta"] = (
        ((snapshot_scores["actual_high_f"] >= snapshot_scores["candidate_p10_f"]) & (snapshot_scores["actual_high_f"] <= snapshot_scores["candidate_p90_f"])).astype(float)
        - ((snapshot_scores["actual_high_f"] >= snapshot_scores["baseline_p10_f"]) & (snapshot_scores["actual_high_f"] <= snapshot_scores["baseline_p90_f"])).astype(float)
    )
    by_day = snapshot_scores.groupby("target_date", as_index=False)[["rps_delta", "p50_abs_error_delta", "coverage80_delta"]].mean()
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(by_day), size=(bootstrap_samples, len(by_day)), replace=True)
    bootstrap = by_day[["rps_delta", "p50_abs_error_delta", "coverage80_delta"]].to_numpy(float)[draws].mean(axis=1)
    result: dict[str, object] = {
        "event_days": int(len(by_day)), "vectors": int(len(work)),
        "paired": {
            "rps_delta": float(snapshot_scores["rps_delta"].mean()),
            "brier_delta": float(work["brier_delta"].mean()),
            "log_loss_delta": float(work["log_loss_delta"].mean()),
            "p50_abs_error_delta": float(snapshot_scores["p50_abs_error_delta"].mean()),
            "coverage80_delta": float(snapshot_scores["coverage80_delta"].mean()),
            "rps_delta_ci90": [float(np.quantile(bootstrap[:, 0], .05)), float(np.quantile(bootstrap[:, 0], .95))],
            "p50_abs_error_delta_ci90": [float(np.quantile(bootstrap[:, 1], .05)), float(np.quantile(bootstrap[:, 1], .95))],
            "coverage80_delta_ci90": [float(np.quantile(bootstrap[:, 2], .05)), float(np.quantile(bootstrap[:, 2], .95))],
        },
    }
    diagnostic_columns = [
        column for column in (
            "path_ood", "forecast_stability_wasserstein_f", "tft_hrrr_peak_gap_f",
            "peak_hour_gap", "path_effective_sample_size", "path_peak_slot_mode_share",
        ) if column in work
    ]
    result["ex_ante"] = {
        column: _diagnostic_summary(work, column) for column in diagnostic_columns
    }
    return result


def _cache_probabilities(cache: ForecastCache, prefix: str) -> pd.DataFrame:
    snapshots = cache.snapshots.copy()
    extras = [column for column in snapshots if column.startswith(("path_", "forecast_stability", "tft_hrrr", "peak_hour"))]
    base = ["snapshot_id", "init_time", "target_date", "p10_f", "p50_f", "p90_f", "coverage_type", "forecast_hour_count"]
    columns = [*base, *extras]
    result = cache.probabilities.merge(snapshots.loc[:, columns], on="snapshot_id", how="inner", validate="many_to_one")
    return result.rename(columns={
        "yes_probability": f"{prefix}_probability", "p10_f": f"{prefix}_p10_f",
        "p50_f": f"{prefix}_p50_f", "p90_f": f"{prefix}_p90_f",
        "coverage_type": f"{prefix}_coverage_type", "forecast_hour_count": f"{prefix}_forecast_hour_count",
    })


def _attach_snapshot_rps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source in ("candidate", "baseline"):
        values: dict[tuple[object, object], float] = {}
        for key, group in result.groupby(["init_time", "target_date"], sort=False):
            ordered = group.sort_values(["bucket_lo_f", "bucket_hi_f"])
            probability = ordered[f"{source}_probability"].to_numpy(float)
            total = probability.sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError("bucket probabilities cannot form an RPS distribution")
            probability = probability / total
            truth = ordered["truth_yes"].to_numpy(float)
            if truth.sum() != 1.0:
                raise ValueError("market bucket panel must contain exactly one settled bucket")
            values[key] = float(np.mean((np.cumsum(probability) - np.cumsum(truth)) ** 2))
        result[f"{source}_rps"] = [values[(row.init_time, row.target_date)] for row in result.itertuples(index=False)]
    return result


def _diagnostic_summary(work: pd.DataFrame, column: str) -> dict[str, object]:
    values = work[column]
    if values.dtype == bool or set(values.dropna().unique()).issubset({True, False}):
        groups = [("true", values.astype(bool)), ("false", ~values.astype(bool))]
    else:
        threshold = float(values.quantile(.75))
        groups = [("high", values >= threshold), ("low", values < threshold)]
    return {
        label: {
            "vectors": int(mask.sum()),
            "brier_delta": float(work.loc[mask, "brier_delta"].mean()) if mask.any() else None,
            "log_loss_delta": float(work.loc[mask, "log_loss_delta"].mean()) if mask.any() else None,
        }
        for label, mask in groups
    }
