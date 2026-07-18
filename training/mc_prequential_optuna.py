"""Training-only parameter families for strict prequential MC attribution."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Mapping

import numpy as np
import pandas as pd

from model.daily_high import PartialDayBiasCalibration
from model.mc import McSampler
from model.mc_calibration import McCalibration
from training.mc_calibration import local_calendar_day_groups


_COMPONENT_NAMES = ("sigma", "correlation", "partial_bias")
INTEGER_SUPPORT_MIN_F = -100
INTEGER_SUPPORT_MAX_F = 160
_INTEGER_SUPPORT = np.arange(INTEGER_SUPPORT_MIN_F, INTEGER_SUPPORT_MAX_F + 1)


@dataclass(frozen=True)
class McTuneComponents:
    """The declared MC components a trial may alter."""

    sigma: bool = False
    correlation: bool = False
    partial_bias: bool = False

    @property
    def subset_id(self) -> str:
        names = [
            name for name, enabled in (
                ("sigma", self.sigma),
                ("correlation", self.correlation),
                ("partial_bias", self.partial_bias),
            ) if enabled
        ]
        return "+".join(names) if names else "none"


def mc_tune_component_subsets() -> tuple[McTuneComponents, ...]:
    """Return the complete 2^3 factorial family in stable order."""
    result: list[McTuneComponents] = []
    for size in range(len(_COMPONENT_NAMES) + 1):
        for selected in combinations(_COMPONENT_NAMES, size):
            result.append(McTuneComponents(**{name: name in selected for name in _COMPONENT_NAMES}))
    return tuple(result)


def identity_mc_tune_parameters(components: McTuneComponents) -> dict[str, float]:
    """Return the exact static-MC point for one declared parameter subset."""
    parameters: dict[str, float] = {}
    if components.sigma:
        parameters.update({f"sigma_multiplier_{index}": 1.0 for index in range(5)})
    if components.correlation:
        parameters.update({
            "correlation_ar1_weight": 0.0,
            "correlation_ar1_rho": 0.0,
        })
    if components.partial_bias:
        parameters.update({
            "partial_bias_gain": 1.0,
            "partial_bias_offset_f": 0.0,
        })
    return parameters


def enqueue_identity_mc_trial(study: object, components: McTuneComponents) -> None:
    """Ensure a tuner can select the unchanged static-MC calibration."""
    enqueue_trial = getattr(study, "enqueue_trial")
    enqueue_trial(identity_mc_tune_parameters(components), skip_if_exists=True)


def apply_mc_tune_parameters(
    calibration: McCalibration,
    bias: PartialDayBiasCalibration,
    *,
    components: McTuneComponents,
    parameters: Mapping[str, float],
) -> tuple[McCalibration, PartialDayBiasCalibration]:
    """Apply one bounded trial to exactly its declared MC components."""
    scales = calibration.sigma_scales.copy()
    correlation = calibration.correlation.copy()
    bias_values = np.asarray(bias.bias_by_forecast_hours_f, dtype=float).copy()

    if components.sigma:
        multipliers = np.asarray(
            [parameters[f"sigma_multiplier_{index}"] for index in range(len(scales))],
            dtype=float,
        )
        if not np.isfinite(multipliers).all() or np.any(multipliers <= 0):
            raise ValueError("sigma multipliers must be finite and positive")
        scales *= multipliers

    if components.correlation:
        weight = float(parameters["correlation_ar1_weight"])
        rho = float(parameters["correlation_ar1_rho"])
        if not 0.0 <= weight <= 1.0 or not 0.0 <= rho < 1.0:
            raise ValueError("correlation trial parameters are outside their bounds")
        offsets = np.abs(np.subtract.outer(np.arange(48), np.arange(48)))
        ar1 = rho ** offsets
        correlation = (1.0 - weight) * correlation + weight * ar1

    if components.partial_bias:
        gain = float(parameters["partial_bias_gain"])
        offset = float(parameters["partial_bias_offset_f"])
        if not np.isfinite(gain) or gain < 0 or not np.isfinite(offset):
            raise ValueError("partial bias trial parameters are invalid")
        bias_values[1:] = gain * bias_values[1:] + offset
        bias_values[0] = 0.0

    tuned = McCalibration(
        correlation=correlation,
        sigma_scales=scales,
        max_bias_f=calibration.max_bias_f,
        model_sha256=calibration.model_sha256,
        fit_split=calibration.fit_split,
        fit_start=calibration.fit_start,
        fit_end=calibration.fit_end,
        sample_count=calibration.sample_count,
        station_timezone=calibration.station_timezone,
        target_window=calibration.target_window,
    )
    return tuned, PartialDayBiasCalibration(
        tuple(float(value) for value in bias_values),
        prior_strength=bias.prior_strength,
        sample_counts=bias.sample_counts,
    )


def shapley_rps_attribution(scores: Mapping[str, float]) -> dict[str, float]:
    """Allocate total RPS improvement over ``none`` across MC components."""
    expected = {subset.subset_id for subset in mc_tune_component_subsets()}
    if set(scores) != expected:
        raise ValueError("Shapley attribution requires every MC component subset")
    baseline = float(scores["none"])
    result = {name: 0.0 for name in _COMPONENT_NAMES}
    count = len(_COMPONENT_NAMES)
    for name in _COMPONENT_NAMES:
        others = tuple(other for other in _COMPONENT_NAMES if other != name)
        for size in range(len(others) + 1):
            coefficient = factorial(size) * factorial(count - size - 1) / factorial(count)
            for selected in combinations(others, size):
                before = McTuneComponents(**{component: component in selected for component in _COMPONENT_NAMES})
                after = McTuneComponents(**{
                    component: component in (*selected, name) for component in _COMPONENT_NAMES
                })
                result[name] += coefficient * (float(scores[before.subset_id]) - float(scores[after.subset_id]))
    if not np.isclose(sum(result.values()), baseline - float(scores["sigma+correlation+partial_bias"])):
        raise RuntimeError("Shapley attribution does not reconcile with total RPS improvement")
    return result


def suggest_mc_tune_parameters(
    trial: object,
    components: McTuneComponents,
) -> dict[str, float]:
    """Draw one bounded, production-compatible MC calibration trial."""
    suggest_float = getattr(trial, "suggest_float")
    parameters: dict[str, float] = {}
    if components.sigma:
        for index in range(5):
            parameters[f"sigma_multiplier_{index}"] = float(
                suggest_float(f"sigma_multiplier_{index}", 0.6, 1.6)
            )
    if components.correlation:
        parameters["correlation_ar1_weight"] = float(
            suggest_float("correlation_ar1_weight", 0.0, 1.0)
        )
        parameters["correlation_ar1_rho"] = float(
            suggest_float("correlation_ar1_rho", 0.0, 0.98)
        )
    if components.partial_bias:
        parameters["partial_bias_gain"] = float(
            suggest_float("partial_bias_gain", 0.0, 2.0)
        )
        parameters["partial_bias_offset_f"] = float(
            suggest_float("partial_bias_offset_f", -2.0, 2.0)
        )
    return parameters


def score_daily_high_rps(
    frame: pd.DataFrame,
    calibration: McCalibration,
    bias: PartialDayBiasCalibration,
    *,
    station_timezone: str,
    mc_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Score complete local-day MC distributions with one row per snapshot."""
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    rows: list[dict[str, object]] = []
    for index, group in enumerate(local_calendar_day_groups(frame, station_timezone)):
        ordered = group.sort_values("fxx")
        sampler = McSampler(n_samp=mc_samples, rng_seed=seed + index)
        sampler.set_calibration(calibration)
        sigma = (
            ordered["p90"].to_numpy(dtype=float)
            - ordered["p10"].to_numpy(dtype=float)
        ) / (2.0 * 1.2815515655446004)
        samples = sampler.sample_daily_high(
            ordered["base_temp_c"].to_numpy(dtype=float)
            + ordered["p50"].to_numpy(dtype=float),
            sigma,
            max_bias=bias.for_forecast_hours(len(ordered)),
            hour_indices=[int(value) - 1 for value in ordered["fxx"]],
        )
        actual_high_f = float(
            (ordered["base_temp_c"] + ordered["target"]).max() * 1.8 + 32.0
        )
        integer = np.clip(
            np.floor(samples + 0.5).astype(int),
            INTEGER_SUPPORT_MIN_F,
            INTEGER_SUPPORT_MAX_F,
        )
        actual_integer = int(np.floor(actual_high_f + 0.5))
        rows.append(
            {
                "init_time": pd.Timestamp(ordered["init_time"].iloc[0]),
                "target_date": str(ordered["target_date"].iloc[0]),
                "actual_high_f": actual_high_f,
                "p10_f": float(np.quantile(samples, 0.1)),
                "p50_f": float(np.quantile(samples, 0.5)),
                "p90_f": float(np.quantile(samples, 0.9)),
                "rps": fixed_integer_rps(integer, actual_integer_f=actual_integer),
                "true_integer_probability": float(
                    (integer == actual_integer).mean()
                ),
            }
        )
    if not rows:
        raise ValueError("MC score frame has no complete local calendar days")
    return pd.DataFrame(rows)


def fixed_integer_rps(samples: np.ndarray, *, actual_integer_f: int) -> float:
    """RPS on the immutable settlement-temperature support for every candidate."""
    if not INTEGER_SUPPORT_MIN_F <= int(actual_integer_f) <= INTEGER_SUPPORT_MAX_F:
        raise ValueError("actual integer temperature is outside fixed support")
    values = np.asarray(samples, dtype=int)
    if values.size == 0:
        raise ValueError("MC samples are empty")
    cdf = np.array([(values <= value).mean() for value in _INTEGER_SUPPORT])
    truth_cdf = (_INTEGER_SUPPORT >= int(actual_integer_f)).astype(float)
    # Standard multicategory RPS omits the terminal CDF boundary, which is one
    # for both distributions and carries no information.
    return float(np.mean((cdf[:-1] - truth_cdf[:-1]) ** 2))


def event_day_rps(rows: pd.DataFrame) -> float:
    """Return target-day-equal RPS, never snapshot-weighted RPS."""
    required = {"target_date", "rps"}
    if missing := required - set(rows):
        raise ValueError("RPS rows missing columns: " + ", ".join(sorted(missing)))
    if rows.empty:
        raise ValueError("RPS rows are empty")
    return float(rows.groupby("target_date", sort=True)["rps"].mean().mean())
