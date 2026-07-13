"""Factories for strict OOF conditional residual-path dependence experiments."""
from __future__ import annotations

from typing import Mapping
import hashlib
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from model.conditional_residual import (
    ConditionalPathSampler,
    ConditionalResidualPathLibrary,
)
from model.daily_high import (
    DailyHighDistribution,
    IncompleteDailyHighCoverage,
    PartialDayBiasCalibration,
)
from model.mc_calibration import McCalibration
from backtest.forecast_cache import ForecastCache


_NORMAL_P90 = 1.2815515655446004
CONDITIONAL_STATE_FEATURES = (
    "forecast_hour_start", "forecast_hour_count", "season_sin", "season_cos",
    "tft_peak_f", "tft_mean_f", "tft_range_f", "tft_spread_f", "tft_peak_slot",
    "hrrr_peak_f", "hrrr_mean_f", "tft_hrrr_peak_gap_f", "peak_hour_gap",
    "observed_hour_fraction", "observed_high_gap_f",
)


def build_complete_residual_path_library(
    hourly_predictions: pd.DataFrame,
    calibration: McCalibration,
    *, station_timezone: str,
    source_fold_id: str,
) -> ConditionalResidualPathLibrary:
    """Build only complete trusted 48h OOF residual trajectories.

    A path remains an hourly-cycle object.  Its retrieval state is derived from
    each complete forecast local-day within that cycle, so target-date matching
    never needs labels that would be unavailable at inference.
    """
    required = {"init_time", "valid_time", "fxx", "p10", "p50", "p90", "base_temp_c", "target", "weight"}
    missing = required - set(hourly_predictions)
    if missing:
        raise ValueError("conditional paths missing columns: " + ", ".join(sorted(missing)))
    work = hourly_predictions.copy()
    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    work["valid_time"] = pd.to_datetime(work["valid_time"], utc=True)
    work["fxx"] = work["fxx"].astype(int)
    extrapolated = work.get("is_extrapolated", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    work = work[(work["weight"].astype(float) > 0) & ~extrapolated].copy()
    rows: list[dict[str, object]] = []
    for init_time, cycle in work.groupby("init_time", sort=True):
        cycle = cycle.sort_values("fxx")
        if cycle["fxx"].tolist() != list(range(1, 49)) or cycle["valid_time"].duplicated().any():
            continue
        sigma = (cycle["p90"].to_numpy(float) - cycle["p10"].to_numpy(float)) / (2.0 * _NORMAL_P90)
        sigma = calibration.scale_sigma(sigma, hour_indices=list(range(48)))
        if not np.isfinite(sigma).all() or not (sigma > 0).all():
            continue
        residual = (cycle["target"].to_numpy(float) - cycle["p50"].to_numpy(float)) / sigma
        cycle = cycle.assign(_residual=residual)
        cycle["target_date"] = cycle["valid_time"].dt.tz_convert(station_timezone).dt.date.astype(str)
        for target_date, day in cycle.groupby("target_date", sort=True):
            # States are only valid if all target-date forecast slots are trusted.
            if not _is_complete_local_day(day, str(target_date), station_timezone):
                continue
            state = conditional_state(day, target_date=str(target_date), station_timezone=station_timezone,
                                      observed_hour_count=0, observed_high_f=None)
            row: dict[str, object] = {
                "path_id": f"{source_fold_id}:{pd.Timestamp(init_time).isoformat()}:{target_date}",
                "init_time": init_time, "target_date": str(target_date), "source_fold_id": source_fold_id,
                **state,
            }
            row.update({f"residual_{index:02d}": float(value) for index, value in enumerate(residual, start=1)})
            rows.append(row)
    columns = ["path_id", "init_time", "target_date", "source_fold_id", *CONDITIONAL_STATE_FEATURES,
               *[f"residual_{hour:02d}" for hour in range(1, 49)]]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        # Keep a contract-valid empty library for callers that need to record rejection.
        raise ValueError("conditional path factory found no complete trusted 48h paths")
    return ConditionalResidualPathLibrary.fit(frame, feature_columns=CONDITIONAL_STATE_FEATURES)


def conditional_state(
    forecast: pd.DataFrame, *, target_date: str, station_timezone: str,
    observed_hour_count: int, observed_high_f: float | None,
) -> dict[str, float]:
    """Return the frozen ex-ante state contract for one target-date snapshot."""
    required = {"valid_time", "fxx", "p10", "p50", "p90", "base_temp_c"}
    missing = required - set(forecast)
    if missing:
        raise ValueError("conditional state missing columns: " + ", ".join(sorted(missing)))
    day = forecast.sort_values("valid_time")
    tft_c = day["base_temp_c"].to_numpy(float) + day["p50"].to_numpy(float)
    hrrr_c = day["base_temp_c"].to_numpy(float)
    spread_f = float(np.mean((day["p90"].to_numpy(float) - day["p10"].to_numpy(float)) * 1.8))
    peak_index = int(np.argmax(tft_c))
    hrrr_peak_index = int(np.argmax(hrrr_c))
    local_date = pd.Timestamp(target_date)
    day_of_year = float(local_date.dayofyear)
    forecast_peak_f = float(tft_c.max() * 1.8 + 32.0)
    return {
        "forecast_hour_start": float(day["fxx"].min()),
        "forecast_hour_count": float(len(day)),
        "season_sin": float(np.sin(2.0 * np.pi * day_of_year / 365.25)),
        "season_cos": float(np.cos(2.0 * np.pi * day_of_year / 365.25)),
        "tft_peak_f": forecast_peak_f,
        "tft_mean_f": float(tft_c.mean() * 1.8 + 32.0),
        "tft_range_f": float((tft_c.max() - tft_c.min()) * 1.8),
        "tft_spread_f": spread_f,
        "tft_peak_slot": float(peak_index),
        "hrrr_peak_f": float(hrrr_c.max() * 1.8 + 32.0),
        "hrrr_mean_f": float(hrrr_c.mean() * 1.8 + 32.0),
        "tft_hrrr_peak_gap_f": float((tft_c.max() - hrrr_c.max()) * 1.8),
        "peak_hour_gap": float(abs(peak_index - hrrr_peak_index)),
        "observed_hour_fraction": float(observed_hour_count / max(len(day) + observed_hour_count, 1)),
        "observed_high_gap_f": float(0.0 if observed_high_f is None else observed_high_f - forecast_peak_f),
    }


def build_conditional_daily_high_distribution(
    *, init_time: object, target_date: str, station_timezone: str,
    predictions: pd.DataFrame, observations: pd.DataFrame,
    sampler: ConditionalPathSampler, calibration: McCalibration,
    bias: PartialDayBiasCalibration,
) -> DailyHighDistribution:
    """Build a point-in-time daily high using a conditional path sampler.

    This deliberately mirrors the established coverage rule instead of routing
    through ``McSampler``.  It is a training-only seam: a failed conditional
    lookup rejects the snapshot rather than silently changing MC behavior.
    """
    init = _utc(init_time)
    expected = _target_hours(target_date, station_timezone)
    expected_set = set(expected)
    obs = observations.copy()
    obs["valid_time"] = pd.to_datetime(obs["valid_time"], utc=True)
    obs["available_at"] = pd.to_datetime(obs["available_at"], utc=True)
    obs = obs[(obs["valid_time"].isin(expected_set)) & (obs["valid_time"] <= init) & (obs["available_at"] <= init)]
    obs = obs.sort_values("valid_time").drop_duplicates("valid_time", keep="last")
    forecast = predictions.copy()
    forecast["valid_time"] = pd.to_datetime(forecast["valid_time"], utc=True)
    extrapolated = forecast.get("is_extrapolated", pd.Series(False, index=forecast.index)).fillna(False).astype(bool)
    forecast = forecast[(forecast["valid_time"].isin(expected_set)) & (forecast["valid_time"] > init) & ~extrapolated]
    forecast = forecast.sort_values("valid_time").drop_duplicates("valid_time", keep="last")
    if set(obs["valid_time"]) | set(forecast["valid_time"]) != expected_set:
        raise IncompleteDailyHighCoverage("point-in-time daily high has uncovered local-day hours")
    observed_high_f = None if obs.empty else float(obs["temperature_c"].astype(float).max() * 1.8 + 32.0)
    if forecast.empty:
        if observed_high_f is None:
            raise IncompleteDailyHighCoverage("daily high has neither observations nor forecasts")
        samples = np.full(sampler.n_samp, observed_high_f, dtype=float)
    else:
        raw_sigma = (forecast["p90"].to_numpy(float) - forecast["p10"].to_numpy(float)) / (2.0 * _NORMAL_P90)
        indices = [int(value) - 1 for value in forecast["fxx"].astype(int)]
        sigma = calibration.scale_sigma(raw_sigma, hour_indices=indices)
        state = conditional_state(forecast, target_date=target_date, station_timezone=station_timezone,
                                  observed_hour_count=len(obs), observed_high_f=observed_high_f)
        samples = sampler.sample_daily_high(
            forecast["base_temp_c"].to_numpy(float) + forecast["p50"].to_numpy(float), sigma, indices,
            state=state, max_bias=bias.for_forecast_hours(len(forecast)),
        )
        if observed_high_f is not None:
            samples = np.maximum(samples, observed_high_f)
    coverage = "forecast_only" if obs.empty else ("observed_only" if forecast.empty else "observed_plus_forecast")
    return DailyHighDistribution(
        daily_high_samples_f=np.asarray(samples, dtype=float), coverage_type=coverage,
        observed_high_f=observed_high_f, observed_hour_count=len(obs),
        observed_through=None if obs.empty else pd.Timestamp(obs["valid_time"].max()),
        forecast_hour_count=len(forecast), forecast_start_at=None if forecast.empty else pd.Timestamp(forecast["valid_time"].min()),
        forecast_through=None if forecast.empty else pd.Timestamp(forecast["valid_time"].max()),
        target_hour_count=len(expected),
    )


def fit_conditional_partial_day_bias(
    *, hourly_predictions: pd.DataFrame, observations: pd.DataFrame,
    calibration: McCalibration, library: ConditionalResidualPathLibrary,
    station_timezone: str, prior_strength: float = 20.0,
    mc_samples: int = 2_000, seed: int = 42,
) -> PartialDayBiasCalibration:
    """Fit remaining-hours bias using the same conditional path dependence."""
    if prior_strength < 0 or mc_samples <= 0:
        raise ValueError("conditional partial bias settings must be valid")
    required = {"init_time", "valid_time", "fxx", "p10", "p50", "p90", "base_temp_c"}
    missing = required - set(hourly_predictions)
    if missing:
        raise ValueError("conditional partial bias predictions missing columns: " + ", ".join(sorted(missing)))
    work = hourly_predictions.copy()
    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    work["valid_time"] = pd.to_datetime(work["valid_time"], utc=True)
    extrapolated = work.get("is_extrapolated", pd.Series(False, index=work.index)).fillna(False).astype(bool)
    work = work[~extrapolated].copy()
    work["target_date"] = (work["valid_time"] - pd.Timedelta(nanoseconds=1)).dt.tz_convert(station_timezone).dt.date.astype(str)
    obs = observations.loc[:, ["valid_time", "temperature_c"]].copy()
    obs["valid_time"] = pd.to_datetime(obs["valid_time"], utc=True)
    obs = obs.drop_duplicates("valid_time", keep="last")
    offsets: dict[int, list[float]] = {hours: [] for hours in range(1, 25)}
    sampler = ConditionalPathSampler(library, n_samp=mc_samples, rng_seed=seed)
    for group_index, (_, group) in enumerate(work.groupby(["init_time", "target_date"], sort=True)):
        group = group.sort_values("valid_time")
        if not 1 <= len(group) <= 25:
            continue
        truth = group[["valid_time"]].merge(obs, on="valid_time", how="inner", validate="one_to_one")
        if len(truth) != len(group):
            continue
        indices = [int(value) - 1 for value in group["fxx"].astype(int)]
        raw_sigma = (group["p90"].to_numpy(float) - group["p10"].to_numpy(float)) / (2.0 * _NORMAL_P90)
        sigma = calibration.scale_sigma(raw_sigma, hour_indices=indices)
        try:
            samples = sampler.sample_daily_high(
                group["base_temp_c"].to_numpy(float) + group["p50"].to_numpy(float), sigma, indices,
                state=conditional_state(group, target_date=str(group["target_date"].iloc[0]), station_timezone=station_timezone,
                                        observed_hour_count=0, observed_high_f=None), max_bias=0.0,
            )
        except ValueError:
            continue
        actual_future_high_f = float(truth["temperature_c"].astype(float).max() * 1.8 + 32.0)
        offsets[min(len(group), 24)].append(float(np.median(samples)) - actual_future_high_f)
    values = [0.0]
    counts = [0]
    for hours in range(1, 25):
        count = len(offsets[hours])
        raw = float(np.median(offsets[hours])) if count else float(calibration.max_bias_f)
        values.append((count * raw + prior_strength * float(calibration.max_bias_f)) / (count + prior_strength))
        counts.append(count)
    return PartialDayBiasCalibration(tuple(values), prior_strength=float(prior_strength), sample_counts=tuple(counts))


def build_conditional_forecast_cache(
    *, hourly_predictions: pd.DataFrame, observations: pd.DataFrame,
    availability: pd.DataFrame, markets: pd.DataFrame, station_id: str,
    station_timezone: str, split_role: str, checkpoint_sha256: str,
    calibration: McCalibration, bias: PartialDayBiasCalibration,
    library: ConditionalResidualPathLibrary, mc_samples: int = 2_000,
    seed: int = 42, target_start: str | None = None, target_end: str | None = None,
) -> ForecastCache:
    """Generate a parallel immutable cache without changing the MC builder."""
    required_market = {"target_date", "condition_id", "bucket_lo_f", "bucket_hi_f", "event_end_at"}
    missing = required_market - set(markets)
    if missing:
        raise ValueError("conditional cache markets missing columns: " + ", ".join(sorted(missing)))
    availability_by_init = availability.copy()
    availability_by_init["init_time"] = pd.to_datetime(availability_by_init["init_time"], utc=True)
    availability_by_init = availability_by_init.set_index("init_time", drop=False)
    if availability_by_init.index.duplicated().any():
        raise ValueError("conditional cache availability contains duplicate init_time")
    target_markets = markets.copy()
    target_markets["target_date"] = pd.to_datetime(target_markets["target_date"]).dt.date.astype(str)
    if target_start is not None:
        target_markets = target_markets[target_markets["target_date"] >= target_start]
    if target_end is not None:
        target_markets = target_markets[target_markets["target_date"] <= target_end]
    lineage_id = f"conditional-path-mc-pilot-{split_role.replace('_', '-')}"
    snapshots: list[dict[str, object]] = []
    probabilities: list[dict[str, object]] = []
    prior_samples_by_target: dict[str, np.ndarray] = {}
    predictions = hourly_predictions.copy()
    predictions["init_time"] = pd.to_datetime(predictions["init_time"], utc=True)
    for ordinal, (init_time, cycle) in enumerate(predictions.groupby("init_time", sort=True)):
        if init_time not in availability_by_init.index or not bool(availability_by_init.loc[init_time, "complete"]):
            continue
        available_at = _utc(availability_by_init.loc[init_time, "available_at"])
        sampler = ConditionalPathSampler(library, n_samp=mc_samples, rng_seed=seed + ordinal)
        for target_date, day_markets in target_markets.groupby("target_date", sort=True):
            target_end_at = (pd.Timestamp(target_date).tz_localize(ZoneInfo(station_timezone)) + pd.DateOffset(days=1)).tz_convert("UTC")
            if available_at >= target_end_at:
                continue
            try:
                distribution = build_conditional_daily_high_distribution(
                    init_time=init_time, target_date=target_date, station_timezone=station_timezone,
                    predictions=cycle, observations=observations, sampler=sampler,
                    calibration=calibration, bias=bias,
                )
            except (IncompleteDailyHighCoverage, ValueError):
                continue
            diagnostics = sampler.last_diagnostics
            if diagnostics is None:
                continue
            snapshot_id = _conditional_snapshot_id(station_id, init_time, target_date, lineage_id)
            daily_highs = distribution.daily_high_samples_f
            resolved = np.floor(daily_highs + 0.5)
            p10, p50, p90 = np.quantile(daily_highs, [0.1, 0.5, 0.9])
            state = sampler.last_state
            if state is None:
                raise RuntimeError("conditional sampler did not retain its forecast state")
            previous = prior_samples_by_target.get(target_date)
            if previous is None:
                stability_w1 = float("nan")
                stability_p50_jump = float("nan")
                stability_width_jump = float("nan")
            else:
                stability_w1 = float(np.mean(np.abs(np.sort(daily_highs) - np.sort(previous))))
                stability_p50_jump = float(abs(np.quantile(daily_highs, .5) - np.quantile(previous, .5)))
                stability_width_jump = float(abs((np.quantile(daily_highs, .9) - np.quantile(daily_highs, .1)) - (np.quantile(previous, .9) - np.quantile(previous, .1))))
            prior_samples_by_target[target_date] = daily_highs.copy()
            snapshots.append({
                "snapshot_id": snapshot_id, "station_id": station_id, "init_time": init_time,
                "target_date": target_date, "available_at": available_at,
                "p10_f": float(p10), "p50_f": float(p50), "p90_f": float(p90),
                "lineage_id": lineage_id, "coverage_type": distribution.coverage_type,
                "observed_high_f": distribution.observed_high_f,
                "observed_hour_count": distribution.observed_hour_count,
                "observed_through": distribution.observed_through,
                "forecast_hour_count": distribution.forecast_hour_count,
                "forecast_start_at": distribution.forecast_start_at,
                "forecast_through": distribution.forecast_through,
                "target_hour_count": distribution.target_hour_count,
                "distribution_source": "conditional_path_mc_pilot",
                "path_library_sha256": library.manifest_sha256,
                "state_contract_version": 1,
                "nearest_path_distance": diagnostics.nearest_distance,
                "path_effective_sample_size": diagnostics.effective_sample_size,
                "path_candidate_count": diagnostics.candidate_path_count,
                "path_ood": diagnostics.is_ood,
                "path_peak_slot_entropy": diagnostics.path_peak_slot_entropy,
                "path_peak_slot_mode_share": diagnostics.path_peak_slot_mode_share,
                "tft_hrrr_peak_gap_f": state["tft_hrrr_peak_gap_f"],
                "peak_hour_gap": state["peak_hour_gap"],
                "forecast_stability_wasserstein_f": stability_w1,
                "forecast_stability_p50_jump_f": stability_p50_jump,
                "forecast_stability_width_jump_f": stability_width_jump,
                "mc_seed": seed + ordinal,
            })
            for market in day_markets.itertuples(index=False):
                probabilities.append({
                    "snapshot_id": snapshot_id, "condition_id": str(market.condition_id),
                    "yes_probability": float(((resolved >= float(market.bucket_lo_f)) & (resolved < float(market.bucket_hi_f))).mean()),
                })
    calibration_sha = hashlib.sha256((library.manifest_sha256 + str(bias.bias_by_forecast_hours_f)).encode("ascii")).hexdigest()
    lineages = pd.DataFrame([{
        "lineage_id": lineage_id, "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": calibration_sha, "split_role": split_role,
        "mc_seed": seed, "mc_samples": mc_samples,
        "distribution_source": "conditional_path_mc_pilot",
        "path_library_sha256": library.manifest_sha256,
        "state_contract_version": 1,
    }])
    return ForecastCache(pd.DataFrame(snapshots), pd.DataFrame(probabilities), lineages)


def _is_complete_local_day(day: pd.DataFrame, target_date: str, timezone: str) -> bool:
    start = pd.Timestamp(target_date, tz=timezone)
    expected = pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert("UTC")
    actual = pd.DatetimeIndex(day["valid_time"].sort_values().unique())
    return len(actual) == len(expected) and actual.equals(expected)


def _target_hours(target_date: str, timezone: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(target_date).tz_localize(timezone)
    return pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="right").tz_convert("UTC")


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _conditional_snapshot_id(station_id: str, init_time: object, target_date: str, lineage_id: str) -> str:
    value = "|".join([station_id, _utc(init_time).isoformat(), target_date, lineage_id])
    return hashlib.sha256(value.encode("ascii")).hexdigest()[:24]
