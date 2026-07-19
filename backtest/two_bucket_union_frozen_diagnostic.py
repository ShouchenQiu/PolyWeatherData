"""Frozen post-March diagnostics for the model-selected two-bucket strategy."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


_HORIZON_BANDS = (
    (0.0, 6.0, "<6h"),
    (6.0, 12.0, "6-12h"),
    (12.0, 24.0, "12-24h"),
    (24.0, 30.0, "24-30h"),
    (30.0, 36.0, "30-36h"),
    (36.0, 42.0, "36-42h"),
    (42.0, 48.0, "42-48h"),
)


def add_exact_horizon_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fixed half-open execution-lead strata without using outcomes."""
    if "lead_hours" not in frame:
        raise ValueError("exact horizon bands require lead_hours")
    lead = frame["lead_hours"].to_numpy(float)
    if not np.isfinite(lead).all():
        raise ValueError("lead_hours must be finite")
    labels = np.full(len(frame), "out_of_scope", dtype=object)
    for lower, upper, label in _HORIZON_BANDS:
        labels[(lead >= lower) & (lead < upper)] = label
    result = frame.copy()
    result["horizon_band"] = labels
    return result


def add_fixed_gap_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fixed model-minus-midpoint strata without consulting outcomes."""
    required = {"model_union_probability", "midpoint_union_probability"}
    missing = required - set(frame)
    if missing:
        raise ValueError("gap bands missing columns: " + ", ".join(sorted(missing)))
    gap = (
        frame["model_union_probability"].to_numpy(float)
        - frame["midpoint_union_probability"].to_numpy(float)
    )
    if not np.isfinite(gap).all():
        raise ValueError("model-market gap must be finite")
    labels = np.select(
        [gap < 0.15, gap < 0.30, gap < 0.45],
        ["<0.15", "0.15-0.30", "0.30-0.45"],
        default=">=0.45",
    )
    result = frame.copy()
    result["model_market_gap"] = gap
    result["model_market_gap_band"] = labels
    return result


def build_frozen_union_diagnostic_report(
    *,
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    legs: pd.DataFrame,
    settlements: pd.DataFrame,
    exclusions: pd.DataFrame,
    forecast_lineages: Mapping[str, str],
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> dict[str, object]:
    """Answer the frozen April/May A2 questions with Event-day uncertainty."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    required = {
        "event_id", "target_date", "evaluation_id", "horizon_band",
        "coverage_type", "union_outcome", "union_probability_lower_90",
        "realized_pnl_usd", "cost_basis_usd",
    }
    missing = required - set(settlements)
    if missing:
        raise ValueError("frozen diagnostic settlements missing columns: " + ", ".join(sorted(missing)))
    expected = {"april_2026", "may_2026"}
    if set(forecast_lineages) != expected:
        raise ValueError("forecast lineages must identify April and May")

    event_universe = candidates.loc[:, ["event_id", "target_date", "evaluation_id"]].drop_duplicates()
    if event_universe.duplicated("event_id").any():
        raise ValueError("candidate Event belongs to more than one diagnostic target")
    overall = _pnl_summary(
        settlements,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        universe_target_dates=event_universe["target_date"],
    )
    by_evaluation = _evaluation_pnl(
        settlements,
        event_universe,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 100,
    )
    by_horizon = _grouped_pnl(
        settlements, ["horizon_band"], bootstrap_samples=bootstrap_samples, seed=seed + 200
    )
    by_evaluation_horizon = _grouped_pnl(
        settlements,
        ["evaluation_id", "horizon_band"],
        bootstrap_samples=bootstrap_samples,
        seed=seed + 300,
    )
    by_coverage = _grouped_pnl(
        settlements, ["coverage_type"], bootstrap_samples=bootstrap_samples, seed=seed + 400
    )
    by_gap = _grouped_pnl(
        settlements,
        ["model_market_gap_band"],
        bootstrap_samples=bootstrap_samples,
        seed=seed + 450,
    ) if "model_market_gap_band" in settlements else []
    by_evaluation_gap = _grouped_pnl(
        settlements,
        ["evaluation_id", "model_market_gap_band"],
        bootstrap_samples=bootstrap_samples,
        seed=seed + 475,
    ) if "model_market_gap_band" in settlements else []
    calibration_overall = _calibration_summary(
        settlements, bootstrap_samples=bootstrap_samples, seed=seed + 500
    )
    calibration_by_evaluation = _grouped_calibration(
        settlements, "evaluation_id", bootstrap_samples=bootstrap_samples, seed=seed + 600
    )
    calibration_by_gap = _grouped_calibration(
        settlements,
        "model_market_gap_band",
        bootstrap_samples=bootstrap_samples,
        seed=seed + 650,
    ) if "model_market_gap_band" in settlements else []
    calibration_by_evaluation_gap = _grouped_calibration_many(
        settlements,
        ["evaluation_id", "model_market_gap_band"],
        bootstrap_samples=bootstrap_samples,
        seed=seed + 675,
    ) if "model_market_gap_band" in settlements else []
    same_lineage = len(set(forecast_lineages.values())) == 1

    april = _lookup(by_evaluation, evaluation_id="april_2026")
    may = _lookup(by_evaluation, evaluation_id="may_2026")
    thirty_to_forty_two = settlements[
        settlements["horizon_band"].astype(str).isin(["30-36h", "36-42h"])
    ]
    far_summary = (
        _pnl_summary(thirty_to_forty_two, bootstrap_samples=bootstrap_samples, seed=seed + 700)
        if not thirty_to_forty_two.empty else None
    )
    far_by_evaluation = _grouped_pnl(
        thirty_to_forty_two, ["evaluation_id"], bootstrap_samples=bootstrap_samples, seed=seed + 710
    )
    forty_two_to_forty_eight = settlements[
        settlements["horizon_band"].astype(str) == "42-48h"
    ]
    longest_summary = (
        _pnl_summary(forty_two_to_forty_eight, bootstrap_samples=bootstrap_samples, seed=seed + 800)
        if not forty_two_to_forty_eight.empty else None
    )
    longest_by_evaluation = _grouped_pnl(
        forty_two_to_forty_eight,
        ["evaluation_id"],
        bootstrap_samples=bootstrap_samples,
        seed=seed + 810,
    )
    ledger = [
        _profit_question(
            "frozen_april_taker_pnl",
            "Does the frozen A2 first qualifying taker rule remain profitable in April 2026?",
            april,
        ),
        _profit_question(
            "frozen_may_taker_pnl",
            "Does the frozen A2 first qualifying taker rule remain profitable in May 2026?",
            may,
        ),
        {
            "question_id": "thirty_to_forty_two_hour_persistence",
            "question": "Does the historical 30-42h taker advantage persist in both diagnostic months?",
            "status": _cross_period_positive_status(far_summary, far_by_evaluation),
            "evidence": {"overall": far_summary, "by_evaluation": far_by_evaluation},
            "artifact": "event_pnl.parquet",
        },
        {
            "question_id": "forty_two_to_forty_eight_hour_risk",
            "question": "Does the historical 42-48h risk remain non-positive in the diagnostic months?",
            "status": _cross_period_nonpositive_status(longest_summary, longest_by_evaluation),
            "evidence": {"overall": longest_summary, "by_evaluation": longest_by_evaluation},
            "artifact": "event_pnl.parquet",
        },
        {
            "question_id": "frozen_lcb_credibility",
            "question": "Is the frozen one-sided 90% probability bound conservative at entered Events?",
            "status": _lcb_status(calibration_overall),
            "evidence": {
                "overall": calibration_overall,
                "by_evaluation": calibration_by_evaluation,
            },
            "artifact": "event_pnl.parquet",
        },
    ]
    return {
        "schema_version": 1,
        "data_class": "frozen_two_bucket_union_april_may_diagnostic",
        "diagnostic_only": True,
        "training_cutoff": "2026-03-31",
        "execution": {
            "overall": overall,
            "by_evaluation": by_evaluation,
            "by_horizon": by_horizon,
            "by_evaluation_horizon": by_evaluation_horizon,
            "by_coverage_type": by_coverage,
            "by_model_market_gap": by_gap,
            "by_evaluation_model_market_gap": by_evaluation_gap,
        },
        "calibration_at_entry": {
            "overall": calibration_overall,
            "by_evaluation": calibration_by_evaluation,
            "by_model_market_gap": calibration_by_gap,
            "by_evaluation_model_market_gap": calibration_by_evaluation_gap,
        },
        "counts": {
            "candidates": int(len(candidates)),
            "decisions": int(len(decisions)),
            "legs": int(len(legs)),
            "settlements": int(len(settlements)),
            "excluded_target_days": int(exclusions["target_date"].nunique()) if len(exclusions) else 0,
        },
        "exclusions": exclusions.to_dict("records"),
        "forecast_lineages": dict(sorted(forecast_lineages.items())),
        "limitations": {
            "same_forecast_lineage_across_months": same_lineage,
            "forecast_lineage_note": (
                "April and May share one Forecast lineage."
                if same_lineage
                else "April and May use different frozen Forecast lineages; strategy-rule persistence is measured, same-checkpoint stability is not."
            ),
            "known_diagnostic_data": "April and May were previously inspected and are not independent holdouts.",
        },
        "decision_ledger": ledger,
    }


def write_frozen_union_diagnostic_artifact(
    *,
    output: Path | str,
    frames: Mapping[str, pd.DataFrame],
    report: Mapping[str, object],
    input_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Persist one immutable, hash-bound frozen diagnostic artifact."""
    expected = {
        "selection_panel", "calibration_coefficients", "decisions",
        "execution_legs", "event_pnl", "exclusions",
    }
    if set(frames) != expected:
        raise ValueError("frozen diagnostic artifact frame set is incomplete")
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"frozen diagnostic output already exists: {root}")
    root.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for name in sorted(expected):
        path = root / f"{name}.parquet"
        frames[name].to_parquet(path, index=False)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="ascii",
    )
    hashes[report_path.name] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "data_class": "frozen_two_bucket_union_april_may_diagnostic",
        "input_sha256": dict(sorted(input_sha256.items())),
        "sha256": hashes,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
    )
    return manifest


def _pnl_summary(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
    universe_target_dates: pd.Series | None = None,
) -> dict[str, object]:
    pnl_by_day = frame.groupby("target_date", sort=True)["realized_pnl_usd"].sum()
    if universe_target_dates is not None:
        universe = pd.Index(sorted(universe_target_dates.astype(str).unique()))
        pnl_by_day = pnl_by_day.reindex(universe, fill_value=0.0)
    values = pnl_by_day.to_numpy(float)
    if len(values) == 0:
        raise ValueError("PnL summary requires at least one Event day")
    means = _bootstrap_means(values, samples=bootstrap_samples, seed=seed)
    cost = float(frame["cost_basis_usd"].sum())
    pnl = float(frame["realized_pnl_usd"].sum())
    hit_rate = (
        float(frame.groupby("target_date", sort=True)["union_outcome"].mean().mean())
        if len(frame) else None
    )
    return {
        "event_days": int(len(pnl_by_day)),
        "traded_event_days": int(frame["target_date"].astype(str).nunique()),
        "trades": int(len(frame)),
        "union_hit_rate": hit_rate,
        "turnover_usd": cost,
        "total_pnl_usd": pnl,
        "roi": pnl / cost if cost > 0 else None,
        "mean_event_day_pnl_usd": float(values.mean()),
        "mean_event_day_pnl_ci90_low": float(np.quantile(means, 0.05)),
        "mean_event_day_pnl_ci90_high": float(np.quantile(means, 0.95)),
        "bootstrap_probability_mean_pnl_positive": float(np.mean(means > 0.0)),
    }


def _evaluation_pnl(
    settlements: pd.DataFrame,
    event_universe: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (evaluation_id, universe) in enumerate(
        event_universe.groupby("evaluation_id", sort=True)
    ):
        group = settlements[
            settlements["evaluation_id"].astype(str) == str(evaluation_id)
        ]
        rows.append({
            "evaluation_id": str(evaluation_id),
            **_pnl_summary(
                group,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index,
                universe_target_dates=universe["target_date"],
            ),
        })
    return rows


def _calibration_summary(
    frame: pd.DataFrame, *, bootstrap_samples: int, seed: int,
) -> dict[str, object]:
    by_day = frame.groupby("target_date", sort=True).agg(
        union_outcome=("union_outcome", "mean"),
        lower_bound=("union_probability_lower_90", "mean"),
    )
    difference = (by_day["union_outcome"] - by_day["lower_bound"]).to_numpy(float)
    means = _bootstrap_means(difference, samples=bootstrap_samples, seed=seed)
    return {
        "event_days": int(len(by_day)),
        "union_hit_rate": float(by_day["union_outcome"].mean()),
        "mean_probability_lower_90": float(by_day["lower_bound"].mean()),
        "mean_hit_minus_lcb": float(difference.mean()),
        "mean_hit_minus_lcb_ci90_low": float(np.quantile(means, 0.05)),
        "mean_hit_minus_lcb_ci90_high": float(np.quantile(means, 0.95)),
    }


def _grouped_pnl(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return rows
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for index, (keys, group) in enumerate(frame.groupby(grouper, sort=True)):
        key_values = (keys,) if len(columns) == 1 else keys
        rows.append({
            **{column: str(value) for column, value in zip(columns, key_values, strict=True)},
            **_pnl_summary(group, bootstrap_samples=bootstrap_samples, seed=seed + index),
        })
    return rows


def _grouped_calibration(
    frame: pd.DataFrame,
    column: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    return [
        {
            column: str(value),
            **_calibration_summary(group, bootstrap_samples=bootstrap_samples, seed=seed + index),
        }
        for index, (value, group) in enumerate(frame.groupby(column, sort=True))
    ]


def _grouped_calibration_many(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (keys, group) in enumerate(frame.groupby(columns, sort=True)):
        rows.append({
            **{column: str(value) for column, value in zip(columns, keys, strict=True)},
            **_calibration_summary(
                group,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index,
            ),
        })
    return rows


def _bootstrap_means(values: np.ndarray, *, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(samples, len(values)))
    return values[indexes].mean(axis=1)


def _lookup(rows: list[dict[str, object]], **keys: str) -> dict[str, object] | None:
    return next(
        (row for row in rows if all(str(row.get(key)) == value for key, value in keys.items())),
        None,
    )


def _profit_question(
    question_id: str, question: str, summary: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question": question,
        "status": _profit_status(summary),
        "evidence": summary,
        "artifact": "event_pnl.parquet",
    }


def _profit_status(summary: dict[str, object] | None) -> str:
    if summary is None:
        return "not measured"
    if float(summary["mean_event_day_pnl_ci90_low"]) > 0.0:
        return "supported"
    if float(summary["mean_event_day_pnl_ci90_high"]) < 0.0:
        return "contradicted"
    return "inconclusive"


def _cross_period_positive_status(
    overall: dict[str, object] | None, periods: list[dict[str, object]],
) -> str:
    if overall is None or len(periods) < 2:
        return "not measured"
    period_means = [float(row["mean_event_day_pnl_usd"]) for row in periods]
    if float(overall["mean_event_day_pnl_ci90_low"]) > 0 and all(value > 0 for value in period_means):
        return "supported"
    if float(overall["mean_event_day_pnl_ci90_high"]) < 0 and all(value < 0 for value in period_means):
        return "contradicted"
    return "inconclusive"


def _cross_period_nonpositive_status(
    overall: dict[str, object] | None, periods: list[dict[str, object]],
) -> str:
    if overall is None or len(periods) < 2:
        return "not measured"
    period_means = [float(row["mean_event_day_pnl_usd"]) for row in periods]
    if float(overall["mean_event_day_pnl_ci90_high"]) < 0 and all(value <= 0 for value in period_means):
        return "supported"
    if float(overall["mean_event_day_pnl_ci90_low"]) > 0 and all(value > 0 for value in period_means):
        return "contradicted"
    return "inconclusive"


def _lcb_status(summary: dict[str, object]) -> str:
    if float(summary["mean_hit_minus_lcb_ci90_low"]) >= 0.0:
        return "supported"
    if float(summary["mean_hit_minus_lcb_ci90_high"]) < 0.0:
        return "contradicted"
    return "inconclusive"
