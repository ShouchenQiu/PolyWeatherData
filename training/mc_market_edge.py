"""Point-in-time MC market-edge scoring primitives.

This module deliberately keeps the market comparator immutable.  A candidate
only supplies a replacement bucket vector for an already validated signal
panel; quote visibility, settlement and normalized synthetic midpoint remain
owned by :mod:`backtest.signal_study`.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.signal_study import _vector_scores
from model.daily_high import IncompleteDailyHighCoverage, PartialDayBiasCalibration, build_daily_high_distribution
from model.mc import McSampler
from model.mc_calibration import McCalibration


_LOSS_METRICS = ("normalized_rps", "multiclass_brier", "log_score")
_VECTOR_KEYS = ("event_id", "target_date", "timestamp", "snapshot_id")
_MODES = ("oracle_market_edge", "frozen_market_edge")


@dataclass(frozen=True)
class MarketEdgeScore:
    """Market-comparator scores at the visible event/tick vector level."""

    scores: pd.DataFrame
    summary: dict[str, object]


def selected_fold_ids(mode: str, folds: Iterable[str]) -> tuple[str, ...]:
    """Return allowed folds, explicitly excluding fold-00 for frozen mode."""
    ids = tuple(folds)
    if mode == "oracle_market_edge":
        return ids
    if mode == "frozen_market_edge":
        return tuple(item for item in ids if item != "fold-00")
    raise ValueError(f"unknown selection mode: {mode!r}")


def horizon_band(forecast_hour_count: int) -> str:
    """Classify a PIT snapshot by remaining, not wall-clock, forecast hours."""
    value = int(forecast_hour_count)
    if value == 0:
        return "0h_observed_only"
    if 1 <= value <= 6:
        return "1-6h"
    if value <= 12:
        return "7-12h"
    if value <= 18:
        return "13-18h"
    if value <= 24:
        return "19-24h"
    if value == 25:
        # DST fall-back has a legitimate 25-hour local calendar day.
        return "19-25h"
    raise ValueError("forecast_hour_count must be in 0..25")


def score_market_edge_panel(
    panel: pd.DataFrame,
    *,
    candidate_probability: pd.Series | np.ndarray | None = None,
    probability_floor: float = 1e-6,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score a candidate against normalized synthetic midpoint.

    The result is Event-day equal: ticks are averaged within an Event before
    any cross-day aggregate or bootstrap.  Negative loss deltas mean the MC
    candidate is better than the midpoint.
    """
    if panel.empty:
        raise ValueError("market edge panel is empty")
    if not 0.0 < probability_floor < 1.0:
        raise ValueError("probability_floor must be between zero and one")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    required = {*_VECTOR_KEYS, "bucket_order", "model_probability", "midpoint_probability", "yes_outcome"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"market edge panel missing columns: {sorted(missing)}")
    frame = panel.copy()
    if candidate_probability is not None:
        values = np.asarray(candidate_probability, dtype=float)
        if len(values) != len(frame):
            raise ValueError("candidate probability length does not match panel")
        frame["candidate_probability"] = values
    else:
        frame["candidate_probability"] = frame["model_probability"].astype(float)

    rows: list[dict[str, object]] = []
    for key, vector in frame.groupby(list(_VECTOR_KEYS), sort=True, dropna=False):
        vector = vector.sort_values("bucket_order", kind="stable")
        outcome = vector["yes_outcome"].to_numpy(dtype=float)
        candidate = _normalized(vector["candidate_probability"].to_numpy(dtype=float), probability_floor)
        midpoint = _normalized(vector["midpoint_probability"].to_numpy(dtype=float), probability_floor)
        if not np.isclose(outcome.sum(), 1.0):
            raise ValueError("every market vector requires exactly one true Bucket")
        candidate_metrics = _vector_scores(candidate, outcome)
        midpoint_metrics = _vector_scores(midpoint, outcome)
        base = dict(zip(_VECTOR_KEYS, key))
        base["coverage_type"] = str(vector.get("coverage_type", pd.Series(["unknown"])).iloc[0])
        hours = vector.get("forecast_hour_count", pd.Series([24])).iloc[0]
        base["forecast_hour_count"] = int(hours)
        base["horizon_band"] = horizon_band(int(hours))
        base["lead_band"] = str(vector.get("lead_band", pd.Series(["unknown"])).iloc[0])
        for metric in _LOSS_METRICS:
            base[f"model_{metric}"] = float(candidate_metrics[metric])
            base[f"midpoint_{metric}"] = float(midpoint_metrics[metric])
            base[f"{metric}_delta"] = float(candidate_metrics[metric] - midpoint_metrics[metric])
        base["model_true_bucket_probability"] = float(candidate_metrics["true_bucket_probability"])
        base["midpoint_true_bucket_probability"] = float(midpoint_metrics["true_bucket_probability"])
        rows.append(base)
    scores = pd.DataFrame(rows)
    return scores, _summarize(scores, bootstrap_samples=bootstrap_samples, seed=seed)


def score_by_strata(
    scores: pd.DataFrame,
    *,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Return all, coverage and remaining-hours summaries using day blocks."""
    result = [{"stratum": "all", **_summarize(scores, bootstrap_samples=bootstrap_samples, seed=seed)}]
    for column in ("coverage_type", "horizon_band"):
        for index, (value, group) in enumerate(scores.groupby(column, sort=True)):
            result.append({
                "stratum": f"{column}={value}",
                **_summarize(group, bootstrap_samples=bootstrap_samples, seed=seed + index + 1),
            })
    return result


def rebuild_market_candidate_probabilities(
    panel: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    hourly_predictions: pd.DataFrame,
    observations: pd.DataFrame,
    calibration: McCalibration,
    bias: PartialDayBiasCalibration,
    station_timezone: str,
    mc_samples: int,
    seed: int = 42,
) -> pd.Series:
    """Rebuild candidate bucket probabilities using the production PIT path.

    ``panel`` defines the exact market vectors to score.  It is not used as a
    source of temperature information: observations and hourly forecasts are
    re-read through ``build_daily_high_distribution`` for every snapshot.
    """
    required_snapshots = {
        "snapshot_id", "init_time", "target_date", "coverage_type",
        "forecast_hour_count",
    }
    missing = required_snapshots - set(snapshots.columns)
    if missing:
        raise ValueError(f"snapshot metadata missing columns: {sorted(missing)}")
    wanted = set(panel["snapshot_id"].astype(str))
    metadata = snapshots[snapshots["snapshot_id"].astype(str).isin(wanted)].copy()
    if set(metadata["snapshot_id"].astype(str)) != wanted:
        raise ValueError("market panel refers to a snapshot absent from cache")
    if metadata["snapshot_id"].duplicated().any():
        raise ValueError("candidate snapshot metadata is not unique")
    predictions = hourly_predictions.copy()
    predictions["init_time"] = pd.to_datetime(predictions["init_time"], utc=True)
    prediction_by_init = {
        stamp: group.copy()
        for stamp, group in predictions.groupby("init_time", sort=False)
    }
    vector_probabilities: dict[str, dict[str, float]] = {}
    for snapshot in metadata.sort_values(["init_time", "target_date"]).itertuples(index=False):
        init_time = pd.Timestamp(snapshot.init_time)
        frame = prediction_by_init.get(init_time)
        if frame is None:
            raise ValueError(f"hourly predictions missing init_time={init_time.isoformat()}")
        # ForecastCache starts a seeded sampler for each init/target snapshot.
        # Preserve that contract so the identity candidate is bit-for-bit
        # comparable with the static MC panel at the same sample count.
        sampler = McSampler(n_samp=int(mc_samples), rng_seed=int(seed))
        sampler.set_calibration(calibration)
        try:
            distribution = build_daily_high_distribution(
                init_time=init_time,
                target_date=str(snapshot.target_date),
                station_timezone=station_timezone,
                predictions=frame,
                observations=observations,
                sampler=sampler,
                bias=bias,
            )
        except IncompleteDailyHighCoverage as exc:
            raise ValueError(f"candidate PIT coverage failed for {snapshot.snapshot_id}") from exc
        if distribution.coverage_type != str(snapshot.coverage_type):
            raise ValueError("candidate coverage type does not match immutable snapshot")
        if distribution.forecast_hour_count != int(snapshot.forecast_hour_count):
            raise ValueError("candidate remaining-hour count does not match immutable snapshot")
        resolved = np.floor(distribution.daily_high_samples_f + 0.5)
        rows = panel[panel["snapshot_id"].astype(str) == str(snapshot.snapshot_id)]
        values: dict[str, float] = {}
        for bucket in rows[["condition_id", "bucket_lo_f", "bucket_hi_f"]].drop_duplicates().itertuples(index=False):
            values[str(bucket.condition_id)] = float(
                ((resolved >= float(bucket.bucket_lo_f)) & (resolved < float(bucket.bucket_hi_f))).mean()
            )
        vector_probabilities[str(snapshot.snapshot_id)] = values
    result = pd.Series(
        [vector_probabilities[str(row.snapshot_id)][str(row.condition_id)] for row in panel.itertuples(index=False)],
        index=panel.index,
        dtype=float,
    )
    _validate_candidate_vectors(panel, result)
    return result


def _normalized(values: np.ndarray, floor: float) -> np.ndarray:
    if not np.isfinite(values).all() or (values < 0.0).any() or values.sum() <= 0.0:
        raise ValueError("market vector probabilities must be finite and non-negative")
    result = np.maximum(values, floor)
    return result / result.sum()


def _validate_candidate_vectors(panel: pd.DataFrame, probability: pd.Series) -> None:
    frame = panel[list(_VECTOR_KEYS)].copy()
    frame["probability"] = probability.to_numpy(dtype=float)
    sums = frame.groupby(list(_VECTOR_KEYS))["probability"].sum()
    if not np.isfinite(probability.to_numpy(dtype=float)).all() or (probability < 0.0).any():
        raise ValueError("candidate probabilities must be finite and non-negative")
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-9):
        raise ValueError("candidate probabilities must sum to one on every market vector")


def _summarize(scores: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> dict[str, object]:
    event = scores.groupby(["event_id", "target_date"], as_index=False)[
        [f"{metric}_delta" for metric in _LOSS_METRICS]
    ].mean()
    payload: dict[str, object] = {
        "sample": {
            "event_day_count": int(event["target_date"].nunique()),
            "event_count": int(event["event_id"].nunique()),
            "vector_count": int(len(scores)),
        },
        "event_day_equal": {},
    }
    rng = np.random.default_rng(seed)
    for metric in _LOSS_METRICS:
        column = f"{metric}_delta"
        values = event[column].to_numpy(dtype=float)
        draws = np.empty(bootstrap_samples, dtype=float)
        for index in range(bootstrap_samples):
            draws[index] = values[rng.integers(0, len(values), len(values))].mean()
        payload["event_day_equal"][column] = float(values.mean())
        payload["event_day_equal"][f"{column}_ci90"] = [
            float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))
        ]
    return payload
