"""Strict historical cross-fit validation for TFT peak-margin risk state."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from training.historical_crossfit import historical_six_fold_plan


_PANEL_REQUIRED = {
    "event_id", "target_date", "timestamp", "snapshot_id", "bucket_order",
    "model_probability", "midpoint_probability", "yes_outcome", "lead_hours",
}
_SCORE_REQUIRED = {
    "event_id", "target_date", "timestamp", "source", "normalized_rps",
    "log_score", "multiclass_brier", "true_bucket_probability",
}
_SNAPSHOT_REQUIRED = {
    "snapshot_id", "target_date", "available_at", "forecast_start_at",
    "forecast_through", "forecast_hour_count", "coverage_type", "lineage_id",
}
_PEAK_REQUIRED = {
    "snapshot_id", "target_date", "tft_peak_margin_f", "tft_peak_slot_at",
    "tft_near_peak_hour_count_1f",
}


@dataclass(frozen=True)
class PeakMarginCrossfitSummary:
    """Decision artifacts produced from a scored peak-margin panel."""

    fold_summaries: pd.DataFrame
    continuous_effects: pd.DataFrame
    report: dict[str, object]


def validate_peak_margin_input_lineage(
    *,
    forecast_cache_root: Path | str,
    signal_root: Path | str,
    peak_diagnostics_root: Path | str,
) -> dict[str, str]:
    """Validate that all source artifacts bind to the same ForecastCache."""
    forecast_path = Path(forecast_cache_root) / "manifest.json"
    signal_path = Path(signal_root) / "manifest.json"
    peak_path = Path(peak_diagnostics_root) / "manifest.json"
    forecast = _read_manifest(forecast_path)
    signal = _read_manifest(signal_path)
    peak = _read_manifest(peak_path)
    if forecast.get("data_class") != "point_in_time_forecast_cache":
        raise ValueError("forecast manifest has the wrong data class")
    if (
        signal.get("data_class") != "forecast_signal_study"
        or signal.get("split_role") != "oof_development"
    ):
        raise ValueError("signal manifest is not an OOF forecast signal study")
    if peak.get("data_class") != "peak_union_diagnostics":
        raise ValueError("peak manifest has the wrong data class")
    forecast_hash = _sha256(forecast_path)
    signal_hash = _sha256(signal_path)
    signal_inputs = signal.get("input_sha256", {})
    peak_inputs = peak.get("input_sha256", {})
    if signal_inputs.get("forecast_manifest") != forecast_hash:
        raise ValueError("signal manifest is bound to another ForecastCache")
    if peak_inputs.get("forecast_cache_manifest") != forecast_hash:
        raise ValueError("peak manifest is bound to another ForecastCache")
    if peak_inputs.get("signal_manifest") != signal_hash:
        raise ValueError("peak artifact is bound to another signal manifest")
    return {
        "forecast_manifest": forecast_hash,
        "signal_manifest": signal_hash,
        "peak_manifest": _sha256(peak_path),
    }


def write_peak_margin_crossfit_artifact(
    *,
    output: Path | str,
    frames: Mapping[str, pd.DataFrame],
    report: Mapping[str, object],
    input_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Write an immutable, hash-complete peak-margin research artifact."""
    required = {
        "snapshot_scores", "fold_contracts", "fold_summaries",
        "continuous_effects",
    }
    missing = required - set(frames)
    if missing:
        raise ValueError("peak-margin artifact missing frames: " + ", ".join(sorted(missing)))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for name in sorted(required):
        path = root / f"{name}.parquet"
        frames[name].to_parquet(path, index=False)
        hashes[path.name] = _sha256(path)
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False),
        encoding="ascii",
    )
    hashes[report_path.name] = _sha256(report_path)
    manifest = {
        "schema_version": 1,
        "data_class": "peak_margin_historical_crossfit",
        "input_sha256": dict(sorted(input_sha256.items())),
        "sha256": hashes,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
    )
    return manifest


def fit_expanding_peak_margin_contracts(
    panel: pd.DataFrame,
    *,
    high_quantile: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply earlier-fold forecast-only margin contracts to evaluation folds."""
    required = {
        "snapshot_id", "target_date", "fold_id", "coverage_type",
        "tft_peak_margin_f",
    }
    missing = required - set(panel)
    if missing:
        raise ValueError("peak-margin panel missing columns: " + ", ".join(sorted(missing)))
    if not 0.0 < high_quantile < 1.0:
        raise ValueError("high_quantile must be in (0, 1)")
    values = panel["tft_peak_margin_f"].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("peak margins must be finite and non-negative")
    ordered_folds = [fold.fold_id for fold in historical_six_fold_plan().folds]
    unknown = set(panel["fold_id"].astype(str)) - set(ordered_folds)
    if unknown:
        raise ValueError("panel contains unknown folds: " + ", ".join(sorted(unknown)))

    eligible = (
        panel[panel["strict_fold_eligible"].astype(bool)].copy()
        if "strict_fold_eligible" in panel
        else panel.copy()
    )
    forecast_only = eligible[eligible["coverage_type"] == "forecast_only"]
    fold_zero = forecast_only[forecast_only["fold_id"] == "fold-00"]
    if fold_zero.empty:
        raise ValueError("fold-00 forecast-only warmup distribution is required")
    fixed_threshold = float(fold_zero["tft_peak_margin_f"].quantile(high_quantile))
    scored: list[pd.DataFrame] = []
    contracts: list[dict[str, object]] = []
    for index, fold_id in enumerate(ordered_folds[1:], start=1):
        evaluation = eligible[eligible["fold_id"] == fold_id].copy()
        if evaluation.empty:
            continue
        fit_fold_ids = ordered_folds[:index]
        training = forecast_only[forecast_only["fold_id"].isin(fit_fold_ids)]
        if training.empty:
            raise ValueError(f"earlier forecast-only state is empty for {fold_id}")
        training_margins = np.sort(training["tft_peak_margin_f"].to_numpy(float))
        threshold = float(np.quantile(training_margins, high_quantile))
        evaluation_margins = evaluation["tft_peak_margin_f"].to_numpy(float)
        evaluation["peak_margin_percentile"] = (
            np.searchsorted(training_margins, evaluation_margins, side="right")
            / len(training_margins)
        )
        evaluation["peak_margin_high"] = evaluation_margins >= threshold
        evaluation["peak_margin_high_fixed_fold00"] = (
            evaluation_margins >= fixed_threshold
        )
        evaluation["peak_margin_contract_id"] = f"expanding-q75-{fold_id}"
        scored.append(evaluation)
        contracts.append({
            "evaluation_fold_id": fold_id,
            "fit_fold_ids": list(fit_fold_ids),
            "fit_target_start": str(training["target_date"].astype(str).min()),
            "fit_target_end": str(training["target_date"].astype(str).max()),
            "fit_event_days": int(training["target_date"].nunique()),
            "fit_snapshots": int(len(training)),
            "high_quantile": float(high_quantile),
            "expanding_q75_f": threshold,
            "fixed_fold00_q75_f": fixed_threshold,
            "excluded_cross_boundary_snapshots": int(
                len(panel[panel["fold_id"] == fold_id]) - len(evaluation)
            ),
        })
    if not scored:
        raise ValueError("peak-margin panel has no evaluation folds after fold-00")
    return pd.concat(scored, ignore_index=True), pd.DataFrame(contracts)


def summarize_peak_margin_crossfit(
    scored: pd.DataFrame,
    *,
    contracts: pd.DataFrame,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> PeakMarginCrossfitSummary:
    """Summarize model-minus-midpoint scores with Event-day weighting."""
    required = {
        "event_id", "snapshot_id", "target_date", "fold_id", "coverage_type",
        "peak_margin_high", "peak_margin_high_fixed_fold00",
        "peak_margin_percentile", "horizon_band", "peak_at_boundary",
        "relative_rps", "relative_log_score", "relative_multiclass_brier",
        "true_bucket_probability_delta",
    }
    missing = required - set(scored)
    if missing:
        raise ValueError("scored peak-margin panel missing columns: " + ", ".join(sorted(missing)))
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    expected_folds = set(scored["fold_id"].astype(str))
    contract_folds = set(contracts.get("evaluation_fold_id", pd.Series(dtype=str)).astype(str))
    if expected_folds - contract_folds:
        raise ValueError("scored panel contains folds without peak-margin contracts")

    metric_columns = [
        "relative_rps", "relative_log_score", "relative_multiclass_brier",
        "true_bucket_probability_delta",
    ]
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    threshold_columns = {
        "expanding_q75": "peak_margin_high",
        "fixed_fold00_q75": "peak_margin_high_fixed_fold00",
    }
    coverage_frames = {
        "forecast_only_primary": scored[scored["coverage_type"] == "forecast_only"],
        "partial_day_sensitivity": scored[scored["coverage_type"] != "forecast_only"],
    }
    for threshold_mode, group_column in threshold_columns.items():
        for coverage_scope, coverage in coverage_frames.items():
            if coverage.empty:
                continue
            work = coverage.copy()
            work["margin_group"] = np.where(work[group_column], "high", "non_high")
            work["month"] = pd.to_datetime(work["target_date"]).dt.to_period("M").astype(str)
            work["peak_boundary_group"] = np.where(
                work["peak_at_boundary"], "boundary", "interior"
            )
            for fold_id in [*sorted(work["fold_id"].astype(str).unique()), "pooled"]:
                fold = work if fold_id == "pooled" else work[work["fold_id"] == fold_id]
                strata = [("overall", "all", fold)]
                for stratum_type, column in (
                    ("horizon", "horizon_band"),
                    ("month", "month"),
                    ("peak_boundary", "peak_boundary_group"),
                ):
                    strata.extend(
                        (stratum_type, str(value), group)
                        for value, group in fold.groupby(column, sort=True)
                    )
                for stratum_type, stratum_value, stratum in strata:
                    _append_group_summaries(
                        rows,
                        stratum,
                        threshold_mode=threshold_mode,
                        coverage_scope=coverage_scope,
                        fold_id=fold_id,
                        stratum_type=stratum_type,
                        stratum_value=stratum_value,
                        metric_columns=metric_columns,
                        bootstrap_samples=bootstrap_samples,
                        rng=rng,
                    )
    summaries = pd.DataFrame(rows)
    continuous_effects = _build_continuous_effects(
        scored[scored["coverage_type"] == "forecast_only"],
        bootstrap_samples=bootstrap_samples,
        seed=seed + 10_000,
    )
    report = _build_decision_report(summaries, continuous_effects)
    return PeakMarginCrossfitSummary(
        fold_summaries=summaries,
        continuous_effects=continuous_effects,
        report=report,
    )


def _append_group_summaries(
    rows: list[dict[str, object]],
    frame: pd.DataFrame,
    *,
    threshold_mode: str,
    coverage_scope: str,
    fold_id: str,
    stratum_type: str,
    stratum_value: str,
    metric_columns: list[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> None:
    day_groups = frame.groupby(
        ["fold_id", "target_date", "margin_group"], as_index=False
    )[metric_columns].mean()
    for margin_group, group in frame.groupby("margin_group", sort=True):
        by_day = group.groupby(
            ["fold_id", "target_date"], as_index=False
        )[metric_columns].mean()
        row: dict[str, object] = {
            "threshold_mode": threshold_mode,
            "coverage_scope": coverage_scope,
            "fold_id": fold_id,
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "margin_group": str(margin_group),
            "event_days": int(len(by_day)),
            "snapshots": int(len(group)),
        }
        row.update({column: float(by_day[column].mean()) for column in metric_columns})
        low, high = _bootstrap_mean_ci(
            by_day["relative_rps"].to_numpy(float),
            strata=by_day["fold_id"].astype(str).to_numpy(),
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
        row["relative_rps_ci90_low"] = low
        row["relative_rps_ci90_high"] = high
        rows.append(row)
    present_groups = set(day_groups["margin_group"].astype(str))
    if not {"high", "non_high"}.issubset(present_groups):
        return
    high_days = day_groups[day_groups["margin_group"] == "high"]
    non_high_days = day_groups[day_groups["margin_group"] == "non_high"]
    selectivity: dict[str, object] = {
        "threshold_mode": threshold_mode,
        "coverage_scope": coverage_scope,
        "fold_id": fold_id,
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "margin_group": "high_minus_non_high",
        "event_days": int(day_groups["target_date"].nunique()),
        "snapshots": int(len(frame)),
    }
    for metric in metric_columns:
        selectivity[metric] = float(
            high_days[metric].mean() - non_high_days[metric].mean()
        )
    low, high = _bootstrap_selectivity_ci(
        day_groups,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    selectivity["relative_rps_ci90_low"] = low
    selectivity["relative_rps_ci90_high"] = high
    rows.append(selectivity)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    strata: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    sums = np.zeros(bootstrap_samples, dtype=float)
    count = 0
    for stratum in sorted(set(strata)):
        subset = values[strata == stratum]
        draws = rng.integers(
            0, len(subset), size=(bootstrap_samples, len(subset))
        )
        sums += subset[draws].sum(axis=1)
        count += len(subset)
    means = sums / count
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def _bootstrap_selectivity_ci(
    day_groups: pd.DataFrame,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    high_sum = np.zeros(bootstrap_samples, dtype=float)
    high_count = np.zeros(bootstrap_samples, dtype=float)
    non_high_sum = np.zeros(bootstrap_samples, dtype=float)
    non_high_count = np.zeros(bootstrap_samples, dtype=float)
    for _, fold in day_groups.groupby("fold_id", sort=True):
        pivot = fold.pivot(
            index="target_date", columns="margin_group", values="relative_rps"
        ).reindex(columns=["high", "non_high"])
        values = pivot.to_numpy(float)
        draws = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
        sampled = values[draws]
        high_values = sampled[:, :, 0]
        non_high_values = sampled[:, :, 1]
        high_sum += np.nansum(high_values, axis=1)
        high_count += np.isfinite(high_values).sum(axis=1)
        non_high_sum += np.nansum(non_high_values, axis=1)
        non_high_count += np.isfinite(non_high_values).sum(axis=1)
    valid = (high_count > 0) & (non_high_count > 0)
    if not valid.any():
        return float("nan"), float("nan")
    delta = high_sum[valid] / high_count[valid] - non_high_sum[valid] / non_high_count[valid]
    return float(np.quantile(delta, 0.05)), float(np.quantile(delta, 0.95))


def _build_decision_report(
    summaries: pd.DataFrame,
    continuous_effects: pd.DataFrame,
) -> dict[str, object]:
    primary = summaries[
        (summaries["threshold_mode"] == "expanding_q75")
        & (summaries["coverage_scope"] == "forecast_only_primary")
        & (summaries["stratum_type"] == "overall")
    ]
    pooled_high = primary[
        (primary["fold_id"] == "pooled") & (primary["margin_group"] == "high")
    ].iloc[0]
    pooled_selectivity = primary[
        (primary["fold_id"] == "pooled")
        & (primary["margin_group"] == "high_minus_non_high")
    ].iloc[0]
    fold_high = primary[
        (primary["fold_id"] != "pooled") & (primary["margin_group"] == "high")
    ]
    fold_selectivity = primary[
        (primary["fold_id"] != "pooled")
        & (primary["margin_group"] == "high_minus_non_high")
    ]
    absolute_status = _direction_status(
        pooled_high,
        fold_high["relative_rps"].to_numpy(float),
    )
    screening_status = _direction_status(
        pooled_selectivity,
        fold_selectivity["relative_rps"].to_numpy(float),
    )
    pooled_effect = continuous_effects[
        continuous_effects["fold_id"] == "pooled"
    ].iloc[0]
    fold_effects = continuous_effects[
        continuous_effects["fold_id"] != "pooled"
    ]
    independence_status = _slope_status(pooled_effect, fold_effects)
    return {
        "schema_version": 1,
        "data_class": "peak_margin_historical_crossfit",
        "primary_estimand": "model_normalized_rps_minus_normalized_synthetic_midpoint_rps",
        "decision_ledger": [
            {
                "question_id": "high_peak_margin_absolute_advantage",
                "question": "Does high peak margin identify snapshots where the model beats normalized midpoint?",
                "status": absolute_status,
                "evidence": _decision_evidence(pooled_high, fold_high),
            },
            {
                "question_id": "high_peak_margin_risk_screening",
                "question": "Does high peak margin reduce model underperformance relative to non-high state?",
                "status": screening_status,
                "evidence": _decision_evidence(pooled_selectivity, fold_selectivity),
            },
            {
                "question_id": "peak_margin_adjusted_independence",
                "question": "Does the peak-margin relationship remain after fold, horizon, and boundary controls?",
                "status": independence_status,
                "evidence": {
                    "percentile_slope": float(pooled_effect["percentile_slope"]),
                    "ci90_low": float(pooled_effect["ci90_low"]),
                    "ci90_high": float(pooled_effect["ci90_high"]),
                    "negative_fold_count": int(
                        (fold_effects["percentile_slope"] < 0.0).sum()
                    ),
                    "positive_fold_count": int(
                        (fold_effects["percentile_slope"] > 0.0).sum()
                    ),
                    "controls": "event_day_weighted_fold_horizon_peak_boundary_fixed_effects",
                },
            },
        ],
    }


def _direction_status(pooled: pd.Series, fold_values: np.ndarray) -> str:
    negative_folds = int(np.sum(fold_values < 0.0))
    positive_folds = int(np.sum(fold_values > 0.0))
    if float(pooled["relative_rps_ci90_high"]) < 0.0 and negative_folds >= 4:
        return "supported"
    if float(pooled["relative_rps_ci90_low"]) > 0.0 and positive_folds >= 4:
        return "contradicted"
    return "inconclusive"


def _decision_evidence(pooled: pd.Series, folds: pd.DataFrame) -> dict[str, object]:
    return {
        "model_minus_midpoint_rps": float(pooled["relative_rps"]),
        "ci90_low": float(pooled["relative_rps_ci90_low"]),
        "ci90_high": float(pooled["relative_rps_ci90_high"]),
        "negative_fold_count": int((folds["relative_rps"] < 0.0).sum()),
        "positive_fold_count": int((folds["relative_rps"] > 0.0).sum()),
        "folds": [
            {
                "fold_id": str(row.fold_id),
                "event_days": int(row.event_days),
                "model_minus_midpoint_rps": float(row.relative_rps),
            }
            for row in folds.itertuples(index=False)
        ],
    }


def _build_continuous_effects(
    forecast_only: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    if forecast_only.empty:
        raise ValueError("forecast-only rows are required for continuous effects")
    rows: list[dict[str, object]] = []
    fold_ids = sorted(forecast_only["fold_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    for fold_id in [*fold_ids, "pooled"]:
        frame = (
            forecast_only
            if fold_id == "pooled"
            else forecast_only[forecast_only["fold_id"] == fold_id]
        )
        design = _continuous_design(frame, include_fold=fold_id == "pooled")
        target = frame["relative_rps"].to_numpy(float)
        base_weights = _event_day_equal_weights(frame)
        slope = _weighted_slope(design, target, base_weights)
        bootstrap_slopes: list[float] = []
        cluster_indices = _fold_day_row_indices(frame)
        for _ in range(bootstrap_samples):
            weights = np.zeros(len(frame), dtype=float)
            for day_rows in cluster_indices.values():
                draws = rng.integers(0, len(day_rows), size=len(day_rows))
                counts = np.bincount(draws, minlength=len(day_rows))
                for count, indices in zip(counts, day_rows, strict=True):
                    if count:
                        weights[indices] = float(count) / len(indices)
            sampled = _weighted_slope(design, target, weights, require_full_rank=True)
            if np.isfinite(sampled):
                bootstrap_slopes.append(sampled)
        if not bootstrap_slopes:
            low = high = float("nan")
        else:
            low, high = np.quantile(bootstrap_slopes, [0.05, 0.95])
        rows.append({
            "fold_id": fold_id,
            "event_days": int(frame["target_date"].nunique()),
            "snapshots": int(len(frame)),
            "percentile_slope": float(slope),
            "ci90_low": float(low),
            "ci90_high": float(high),
            "bootstrap_valid_draws": int(len(bootstrap_slopes)),
            "controls": (
                "horizon_band+peak_at_boundary"
                if fold_id != "pooled"
                else "fold_id+horizon_band+peak_at_boundary"
            ),
        })
    return pd.DataFrame(rows)


def _continuous_design(frame: pd.DataFrame, *, include_fold: bool) -> np.ndarray:
    columns = ["horizon_band", "peak_at_boundary"]
    if include_fold:
        columns.insert(0, "fold_id")
    categorical = pd.get_dummies(
        frame[columns].astype(str),
        drop_first=True,
        dtype=float,
    )
    return np.column_stack(
        [
            np.ones(len(frame), dtype=float),
            frame["peak_margin_percentile"].to_numpy(float),
            categorical.to_numpy(float),
        ]
    )


def _event_day_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(["fold_id", "target_date"])["snapshot_id"].transform("size")
    return 1.0 / counts.to_numpy(float)


def _fold_day_row_indices(frame: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {}
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for fold_id, fold in frame.groupby("fold_id", sort=True):
        result[str(fold_id)] = [
            positions.loc[day.index].to_numpy(int)
            for _, day in fold.groupby("target_date", sort=True)
        ]
    return result


def _weighted_slope(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    require_full_rank: bool = False,
) -> float:
    valid = np.isfinite(weights) & (weights > 0.0)
    if valid.sum() < 2:
        return float("nan")
    root = np.sqrt(weights[valid])
    weighted_design = design[valid] * root[:, None]
    weighted_target = target[valid] * root
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_target, rcond=None
    )
    if require_full_rank and rank < weighted_design.shape[1]:
        return float("nan")
    return float(coefficients[1])


def _slope_status(pooled: pd.Series, folds: pd.DataFrame) -> str:
    negative_folds = int((folds["percentile_slope"] < 0.0).sum())
    positive_folds = int((folds["percentile_slope"] > 0.0).sum())
    if float(pooled["ci90_high"]) < 0.0 and negative_folds >= 4:
        return "supported"
    if float(pooled["ci90_low"]) > 0.0 and positive_folds >= 4:
        return "contradicted"
    return "inconclusive"


def build_peak_margin_score_panel(
    *,
    signal_panel: pd.DataFrame,
    scores_by_tick: pd.DataFrame,
    snapshots: pd.DataFrame,
    peak_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Pair model and midpoint scores at the first legal tick per snapshot."""
    for name, frame, required in (
        ("signal panel", signal_panel, _PANEL_REQUIRED),
        ("scores", scores_by_tick, _SCORE_REQUIRED),
        ("snapshots", snapshots, _SNAPSHOT_REQUIRED),
        ("peak diagnostics", peak_diagnostics, _PEAK_REQUIRED),
    ):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} missing columns: " + ", ".join(sorted(missing)))

    vectors = signal_panel.copy()
    vectors["timestamp"] = pd.to_datetime(vectors["timestamp"], utc=True)
    vector_keys = ["event_id", "target_date", "timestamp"]
    _validate_probability_vectors(vectors, vector_keys)
    lookup = vectors[
        vector_keys + ["snapshot_id", "lead_hours"]
    ].drop_duplicates(vector_keys)

    scores = scores_by_tick.copy()
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True)
    sources = scores[scores["source"].isin(("model_raw", "synthetic_midpoint"))]
    if sources.duplicated(vector_keys + ["source"]).any():
        raise ValueError("scores contain duplicate source rows for a market vector")
    value_columns = [
        "normalized_rps", "log_score", "multiclass_brier",
        "true_bucket_probability",
    ]
    wide = sources.pivot(index=vector_keys, columns="source", values=value_columns)
    expected = pd.MultiIndex.from_product(
        [value_columns, ["model_raw", "synthetic_midpoint"]]
    )
    if not expected.isin(wide.columns).all() or wide.loc[:, expected].isna().any().any():
        raise ValueError("scores require paired model and normalized midpoint rows")
    wide = wide.loc[:, expected].reset_index()
    wide.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else column
        for column in wide.columns
    ]

    snapshot_values = snapshots.loc[:, sorted(_SNAPSHOT_REQUIRED)].copy()
    if snapshot_values["snapshot_id"].duplicated().any():
        raise ValueError("snapshot ids must be unique")
    snapshot_values["available_at"] = pd.to_datetime(
        snapshot_values["available_at"], utc=True
    )
    snapshot_values["forecast_start_at"] = pd.to_datetime(
        snapshot_values["forecast_start_at"], utc=True
    )
    snapshot_values["forecast_through"] = pd.to_datetime(
        snapshot_values["forecast_through"], utc=True
    )
    peaks = peak_diagnostics.loc[:, sorted(_PEAK_REQUIRED)].copy()
    if peaks["snapshot_id"].duplicated().any():
        raise ValueError("peak diagnostic snapshot ids must be unique")
    peaks["tft_peak_slot_at"] = pd.to_datetime(peaks["tft_peak_slot_at"], utc=True)

    result = wide.merge(lookup, on=vector_keys, how="inner", validate="one_to_one")
    result = result.merge(
        snapshot_values, on=["snapshot_id", "target_date"], how="inner",
        validate="many_to_one",
    )
    result = result.merge(
        peaks, on=["snapshot_id", "target_date"], how="inner",
        validate="many_to_one",
    )
    if (result["timestamp"] < result["available_at"]).any():
        raise ValueError("market score predates its forecast snapshot availability")
    result = result.sort_values(
        ["event_id", "snapshot_id", "timestamp"], kind="stable"
    ).drop_duplicates(["event_id", "snapshot_id"], keep="first")
    result["fold_id"] = _assign_fold_ids(result["target_date"])
    result["lineage_fold_id"] = result["lineage_id"].astype(str).str.extract(
        r"(fold-[0-9]{2})$", expand=False
    )
    if result["lineage_fold_id"].isna().any():
        raise ValueError("snapshot lineage does not identify a historical fold")
    result["strict_fold_eligible"] = result["lineage_fold_id"].eq(result["fold_id"])
    result["horizon_band"] = result["lead_hours"].map(_horizon_band)
    result["peak_at_boundary"] = (
        result["tft_peak_slot_at"].eq(result["forecast_start_at"])
        | result["tft_peak_slot_at"].eq(result["forecast_through"])
    )
    for metric in ("normalized_rps", "log_score", "multiclass_brier"):
        result[f"relative_{metric}"] = (
            result[f"{metric}_model_raw"]
            - result[f"{metric}_synthetic_midpoint"]
        )
    result["relative_rps"] = result.pop("relative_normalized_rps")
    result["true_bucket_probability_delta"] = (
        result["true_bucket_probability_model_raw"]
        - result["true_bucket_probability_synthetic_midpoint"]
    )
    return result.reset_index(drop=True)


def _validate_probability_vectors(frame: pd.DataFrame, keys: list[str]) -> None:
    grouped = frame.groupby(keys, sort=False)
    totals = grouped[
        ["model_probability", "midpoint_probability", "yes_outcome"]
    ].sum()
    if not np.isfinite(totals.to_numpy(float)).all():
        raise ValueError("probability vectors must be finite")
    if not np.allclose(
        totals[["model_probability", "midpoint_probability"]].to_numpy(float),
        1.0,
        atol=1e-6,
    ):
        raise ValueError("model and normalized midpoint vectors must sum to one")
    if not np.allclose(totals["yes_outcome"].to_numpy(float), 1.0, atol=1e-6):
        raise ValueError("each vector must contain one settled bucket")
    if grouped["snapshot_id"].nunique().ne(1).any():
        raise ValueError("each market vector must map to one forecast snapshot")


def _assign_fold_ids(target_dates: pd.Series) -> pd.Series:
    result = pd.Series("", index=target_dates.index, dtype="object")
    dates = target_dates.astype(str)
    for fold in historical_six_fold_plan().folds:
        result.loc[dates.between(*fold.predict)] = fold.fold_id
    if result.eq("").any():
        raise ValueError("target date is outside the strict historical fold plan")
    return result


def _horizon_band(value: object) -> str:
    lead = float(value)
    if not np.isfinite(lead) or lead < 0.0 or lead > 48.0:
        raise ValueError("lead_hours must be finite in 0..48")
    for upper, label in (
        (6.0, "0-6h"), (12.0, "6-12h"), (24.0, "12-24h"),
        (30.0, "24-30h"), (36.0, "30-36h"), (42.0, "36-42h"),
    ):
        if lead < upper:
            return label
    return "42-48h"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload
