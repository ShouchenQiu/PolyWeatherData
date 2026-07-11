"""Paired diagnostics for pilot PMF and baseline MC forecast caches."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from backtest.forecast_cache import ForecastCache


class DailyHighEvaluationLike(Protocol):
    snapshots: pd.DataFrame


def score_parallel_forecasts(
    *,
    evaluation: DailyHighEvaluationLike,
    candidate_cache: ForecastCache,
    mc_cache: ForecastCache,
    markets: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Score candidate and MC on identical snapshot and Bucket vectors."""
    links = evaluation.snapshots[["snapshot_id", "source_snapshot_id"]].copy()
    candidate_snapshots = candidate_cache.snapshots.copy()
    if "source_snapshot_id" not in candidate_snapshots:
        candidate_snapshots = candidate_snapshots.merge(
            links,
            on="snapshot_id",
            how="inner",
            validate="one_to_one",
        )
    mc_snapshots = mc_cache.snapshots.set_index("snapshot_id", drop=False)
    actual = actuals[["target_date", "actual_high_f"]].copy()
    actual["target_date"] = actual["target_date"].astype(str)
    actual_by_date = actual.set_index("target_date")["actual_high_f"]
    market_frame = markets.copy()
    market_frame["target_date"] = market_frame["target_date"].astype(str)
    markets_by_date = {
        date: group.sort_values("bucket_lo_f")
        for date, group in market_frame.groupby("target_date", sort=False)
    }
    candidate_probs = candidate_cache.probabilities.set_index(
        ["snapshot_id", "condition_id"]
    )["yes_probability"]
    mc_probs = mc_cache.probabilities.set_index(
        ["snapshot_id", "condition_id"]
    )["yes_probability"]
    rows: list[dict[str, object]] = []
    for candidate in candidate_snapshots.itertuples(index=False):
        target_date = str(candidate.target_date)
        source_id = str(candidate.source_snapshot_id)
        if (
            source_id not in mc_snapshots.index
            or target_date not in markets_by_date
            or target_date not in actual_by_date.index
        ):
            continue
        baseline = mc_snapshots.loc[source_id]
        day_markets = markets_by_date[target_date]
        conditions = day_markets["condition_id"].astype(str).tolist()
        candidate_vector = np.array(
            [float(candidate_probs.get((candidate.snapshot_id, key), 0.0)) for key in conditions]
        )
        mc_vector = np.array(
            [float(mc_probs.get((source_id, key), 0.0)) for key in conditions]
        )
        candidate_vector = _normalized(candidate_vector)
        mc_vector = _normalized(mc_vector)
        actual_high = float(actual_by_date.loc[target_date])
        outcome = (
            (actual_high >= day_markets["bucket_lo_f"].to_numpy(dtype=float))
            & (actual_high < day_markets["bucket_hi_f"].to_numpy(dtype=float))
        ).astype(float)
        if outcome.sum() != 1.0:
            raise ValueError(f"target {target_date} does not resolve to one Bucket")
        candidate_scores = _vector_scores(candidate_vector, outcome)
        mc_scores = _vector_scores(mc_vector, outcome)
        rows.append(
            {
                "snapshot_id": str(candidate.snapshot_id),
                "source_snapshot_id": source_id,
                "target_date": target_date,
                "init_time": pd.Timestamp(candidate.init_time),
                "coverage_type": str(candidate.coverage_type),
                "forecast_hour_count": int(candidate.forecast_hour_count),
                "actual_high_f": actual_high,
                "candidate_p10_f": float(candidate.p10_f),
                "candidate_p50_f": float(candidate.p50_f),
                "candidate_p90_f": float(candidate.p90_f),
                "mc_p10_f": float(baseline.p10_f),
                "mc_p50_f": float(baseline.p50_f),
                "mc_p90_f": float(baseline.p90_f),
                **{f"candidate_{key}": value for key, value in candidate_scores.items()},
                **{f"mc_{key}": value for key, value in mc_scores.items()},
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["forecast_hour_band"] = pd.cut(
            result["forecast_hour_count"],
            bins=[0, 6, 12, 18, 24, 25],
            labels=["1-6", "7-12", "13-18", "19-24", "25"],
            include_lowest=True,
        ).astype(str)
    return result


def summarize_parallel_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Return overall and required point-in-time strata metrics."""
    if scores.empty:
        return pd.DataFrame()
    groups: list[tuple[str, pd.DataFrame]] = [("overall", scores)]
    groups.extend(
        (f"coverage_type={key}", group)
        for key, group in scores.groupby("coverage_type", sort=True)
    )
    groups.extend(
        (f"forecast_hours={key}", group)
        for key, group in scores.groupby("forecast_hour_band", sort=True)
    )
    rows = []
    for stratum, group in groups:
        candidate_covered = (
            (group["actual_high_f"] >= group["candidate_p10_f"])
            & (group["actual_high_f"] <= group["candidate_p90_f"])
        )
        mc_covered = (
            (group["actual_high_f"] >= group["mc_p10_f"])
            & (group["actual_high_f"] <= group["mc_p90_f"])
        )
        rows.append(
            {
                "stratum": stratum,
                "snapshots": len(group),
                "event_days": group["target_date"].nunique(),
                "candidate_brier": group["candidate_brier"].mean(),
                "mc_brier": group["mc_brier"].mean(),
                "candidate_log_loss": group["candidate_log_loss"].mean(),
                "mc_log_loss": group["mc_log_loss"].mean(),
                "candidate_rps": group["candidate_rps"].mean(),
                "mc_rps": group["mc_rps"].mean(),
                "candidate_true_bucket_probability": group[
                    "candidate_true_bucket_probability"
                ].mean(),
                "mc_true_bucket_probability": group[
                    "mc_true_bucket_probability"
                ].mean(),
                "candidate_mae_f": (
                    group["candidate_p50_f"] - group["actual_high_f"]
                ).abs().mean(),
                "mc_mae_f": (
                    group["mc_p50_f"] - group["actual_high_f"]
                ).abs().mean(),
                "candidate_coverage": candidate_covered.mean(),
                "mc_coverage": mc_covered.mean(),
            }
        )
    return pd.DataFrame(rows)


def event_day_bootstrap(
    scores: pd.DataFrame,
    *,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap paired score deltas using Event day as the sampling unit."""
    if samples <= 0 or scores.empty:
        raise ValueError("bootstrap requires positive samples and non-empty scores")
    daily = scores.assign(
        rps_delta=scores["candidate_rps"] - scores["mc_rps"]
    ).groupby("target_date", as_index=False).agg(
        rps_delta=("rps_delta", "mean"),
        log_loss_candidate=("candidate_log_loss", "mean"),
        log_loss_mc=("mc_log_loss", "mean"),
    )
    daily["log_loss_delta"] = daily["log_loss_candidate"] - daily["log_loss_mc"]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(daily), size=(samples, len(daily)))
    rps = daily["rps_delta"].to_numpy()[indices].mean(axis=1)
    log_loss = daily["log_loss_delta"].to_numpy()[indices].mean(axis=1)
    return {
        "rps_delta_mean": float(daily["rps_delta"].mean()),
        "rps_delta_ci90_low": float(np.quantile(rps, 0.05)),
        "rps_delta_ci90_high": float(np.quantile(rps, 0.95)),
        "log_loss_delta_mean": float(daily["log_loss_delta"].mean()),
        "log_loss_delta_ci90_low": float(np.quantile(log_loss, 0.05)),
        "log_loss_delta_ci90_high": float(np.quantile(log_loss, 0.95)),
    }


def build_pilot_report(
    *,
    april_oof: pd.DataFrame,
    april_summary: pd.DataFrame,
    may_summary: pd.DataFrame,
    april_scores: pd.DataFrame,
    may_scores: pd.DataFrame,
    bootstrap_samples: int,
    april_manifest: dict[str, object],
) -> dict[str, object]:
    """Apply the frozen pilot gates and return one audit-ready report."""
    may_overall = _summary_row(may_summary, "overall")
    coverage_rows = may_summary[
        may_summary["stratum"].str.startswith("coverage_type=")
    ]
    later = may_scores[may_scores["forecast_hour_count"] >= 7]
    april_climatology = {
        "pmf_log_loss": float(april_oof["pmf_log_loss"].mean()),
        "climatology_log_loss": float(april_oof["climatology_log_loss"].mean()),
        "pmf_rps": float(april_oof["pmf_rps"].mean()),
        "climatology_rps": float(april_oof["climatology_rps"].mean()),
    }
    gates = {
        "april_beats_climatology_log_loss": (
            april_climatology["pmf_log_loss"]
            < april_climatology["climatology_log_loss"]
        ),
        "april_beats_climatology_rps": (
            april_climatology["pmf_rps"] < april_climatology["climatology_rps"]
        ),
        "may_beats_mc_rps": may_overall["candidate_rps"] < may_overall["mc_rps"],
        "may_log_loss_within_0_01": (
            may_overall["candidate_log_loss"]
            <= may_overall["mc_log_loss"] + 0.01
        ),
        "may_each_coverage_type_rps_not_worse": bool(
            len(coverage_rows) == 2
            and (coverage_rows["candidate_rps"] <= coverage_rows["mc_rps"]).all()
        ),
        "may_coverage_closer_to_80pct": (
            abs(may_overall["candidate_coverage"] - 0.8)
            < abs(may_overall["mc_coverage"] - 0.8)
        ),
        "may_mae_within_0_1f": (
            may_overall["candidate_mae_f"] <= may_overall["mc_mae_f"] + 0.1
        ),
        "may_7_to_24h_rps_not_worse": bool(
            not later.empty
            and later["candidate_rps"].mean() <= later["mc_rps"].mean()
        ),
    }
    gates["eligible_for_cross_fit_goal"] = bool(all(gates.values()))
    return {
        "schema_version": 1,
        "goal": "point_in_time_daily_high_pmf_structural_pilot",
        "june_consumed": False,
        "april_examples": april_manifest,
        "april_temperature_oof": april_climatology,
        "april_parallel_summary": april_summary.to_dict("records"),
        "may_parallel_summary": may_summary.to_dict("records"),
        "april_event_day_bootstrap": event_day_bootstrap(
            april_scores, samples=bootstrap_samples, seed=42
        ),
        "may_event_day_bootstrap": event_day_bootstrap(
            may_scores, samples=bootstrap_samples, seed=43
        ),
        "gates": gates,
    }


def build_gaussian_pilot_report(
    *,
    april_oof: pd.DataFrame,
    april_summary: pd.DataFrame,
    may_summary: pd.DataFrame,
    april_scores: pd.DataFrame,
    may_scores: pd.DataFrame,
    bootstrap_samples: int,
    april_manifest: dict[str, object],
) -> dict[str, object]:
    """Apply the frozen pilot gates for the future-maximum Gaussian candidate."""
    may_overall = _summary_row(may_summary, "overall")
    coverage_rows = may_summary[
        may_summary["stratum"].str.startswith("coverage_type=")
    ]
    later = may_scores[may_scores["forecast_hour_count"] >= 7]
    april_future_max = {
        "gaussian_nll": float(april_oof["future_nll"].mean()),
        "climatology_nll": float(april_oof["climatology_future_nll"].mean()),
    }
    gates = {
        "april_beats_future_max_climatology_nll": (
            april_future_max["gaussian_nll"] < april_future_max["climatology_nll"]
        ),
        "may_beats_mc_rps": may_overall["candidate_rps"] < may_overall["mc_rps"],
        "may_log_loss_within_0_01": (
            may_overall["candidate_log_loss"]
            <= may_overall["mc_log_loss"] + 0.01
        ),
        "may_each_coverage_type_rps_not_worse": bool(
            len(coverage_rows) == 2
            and (coverage_rows["candidate_rps"] <= coverage_rows["mc_rps"]).all()
        ),
        "may_coverage_closer_to_80pct": (
            abs(may_overall["candidate_coverage"] - 0.8)
            < abs(may_overall["mc_coverage"] - 0.8)
        ),
        "may_mae_within_0_1f": (
            may_overall["candidate_mae_f"] <= may_overall["mc_mae_f"] + 0.1
        ),
        "may_7_to_24h_rps_not_worse": bool(
            not later.empty
            and later["candidate_rps"].mean() <= later["mc_rps"].mean()
        ),
    }
    gates["eligible_for_cross_fit_goal"] = bool(all(gates.values()))
    return {
        "schema_version": 1,
        "goal": "point_in_time_future_max_gaussian_structural_pilot",
        "june_consumed": False,
        "april_examples": april_manifest,
        "april_future_max_oof": april_future_max,
        "april_parallel_summary": april_summary.to_dict("records"),
        "may_parallel_summary": may_summary.to_dict("records"),
        "april_event_day_bootstrap": event_day_bootstrap(
            april_scores, samples=bootstrap_samples, seed=42
        ),
        "may_event_day_bootstrap": event_day_bootstrap(
            may_scores, samples=bootstrap_samples, seed=43
        ),
        "gates": gates,
    }


def _summary_row(frame: pd.DataFrame, stratum: str) -> dict[str, object]:
    rows = frame[frame["stratum"] == stratum]
    if len(rows) != 1:
        raise ValueError(f"missing unique summary stratum {stratum!r}")
    return rows.iloc[0].to_dict()


def _normalized(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=float), 1e-6)
    return clipped / clipped.sum()


def _vector_scores(probability: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    truth = int(np.argmax(outcome))
    denominator = max(len(probability) - 1, 1)
    return {
        "brier": float(np.square(probability - outcome).sum()),
        "log_loss": float(-np.log(probability[truth])),
        "rps": float(
            np.square(
                np.cumsum(probability)[:-1] - np.cumsum(outcome)[:-1]
            ).sum()
            / denominator
        ),
        "true_bucket_probability": float(probability[truth]),
    }
