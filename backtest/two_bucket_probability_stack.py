"""Low-capacity market-aware calibration for model-selected two-bucket unions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProbabilityStackSpec:
    """Named ex-ante feature contract for one binary union calibrator."""

    name: str
    features: tuple[str, ...]


CURRENT_MODEL_ONLY = ProbabilityStackSpec(
    "current_model_only", ("model_logit",)
)
MARKET_ONLY = ProbabilityStackSpec(
    "market_only", ("midpoint_logit", "is_36_42h")
)
MODEL_PLUS_MARKET = ProbabilityStackSpec(
    "model_plus_market", ("model_logit", "midpoint_logit", "is_36_42h")
)
DEFAULT_STACK_SPECS = (CURRENT_MODEL_ONLY, MARKET_ONLY, MODEL_PLUS_MARKET)


@dataclass(frozen=True)
class FittedProbabilityStack:
    """One fitted stack plus Event-day bootstrap coefficient draws."""

    spec: ProbabilityStackSpec
    coefficients: tuple[float, ...]
    bootstrap_coefficients: tuple[tuple[float, ...], ...]
    fit_fold_ids: tuple[str, ...]
    fit_target_start: str
    fit_target_end: str
    fit_event_days: int
    ridge_strength: float
    seed: int

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        design = _design_matrix(frame, self.spec)
        point = _predict(design, np.asarray(self.coefficients, dtype=float))
        draws = np.vstack(
            [
                _predict(design, np.asarray(coefficient, dtype=float))
                for coefficient in self.bootstrap_coefficients
            ]
        )
        result = frame.copy()
        result["probability_source"] = self.spec.name
        result["calibrated_union_probability"] = point
        result["union_probability_lower_90"] = np.quantile(draws, 0.1, axis=0)
        result["union_probability_upper_90"] = np.quantile(draws, 0.9, axis=0)
        return result

    def coefficient_frame(self, *, evaluation_fold_id: str) -> pd.DataFrame:
        rows = [("point", -1, self.coefficients)] + [
            ("bootstrap", index, coefficient)
            for index, coefficient in enumerate(self.bootstrap_coefficients)
        ]
        values: list[dict[str, object]] = []
        for kind, draw_index, coefficient in rows:
            by_name = dict(zip(("intercept", *self.spec.features), coefficient, strict=True))
            values.append({
                "evaluation_fold_id": evaluation_fold_id,
                "probability_source": self.spec.name,
                "coefficient_kind": kind,
                "draw_index": draw_index,
                "intercept": float(by_name["intercept"]),
                "model_logit": _optional_float(by_name.get("model_logit")),
                "midpoint_logit": _optional_float(by_name.get("midpoint_logit")),
                "is_36_42h": _optional_float(by_name.get("is_36_42h")),
                "fit_fold_ids": list(self.fit_fold_ids),
                "fit_target_start": self.fit_target_start,
                "fit_target_end": self.fit_target_end,
                "fit_event_days": self.fit_event_days,
                "ridge_strength": self.ridge_strength,
                "seed": self.seed,
            })
        return pd.DataFrame(values)


@dataclass(frozen=True)
class ExpandingProbabilityStackResult:
    scored_samples: pd.DataFrame
    coefficients: pd.DataFrame
    models: Mapping[tuple[str, str], FittedProbabilityStack]


def fit_probability_stack(
    training: pd.DataFrame,
    *,
    spec: ProbabilityStackSpec,
    bootstrap_samples: int = 500,
    seed: int = 42,
    ridge_strength: float = 1.0,
    fit_fold_ids: Sequence[str] = (),
) -> FittedProbabilityStack:
    """Fit one stack from historical Event days only."""
    required = {
        "event_id", "target_date", "model_union_probability",
        "midpoint_union_probability", "lead_hours", "union_outcome",
    }
    missing = required - set(training)
    if missing:
        raise ValueError("probability stack training missing columns: " + ", ".join(sorted(missing)))
    if bootstrap_samples <= 0 or ridge_strength < 0:
        raise ValueError("bootstrap samples must be positive and ridge non-negative")
    work = training.copy().reset_index(drop=True)
    if work.empty or work["event_id"].nunique() < 2:
        raise ValueError("probability stack requires at least two Event days")
    outcome = work["union_outcome"].to_numpy(float)
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("union outcomes must be binary")
    design = _design_matrix(work, spec)
    coefficients = _fit_weighted_logistic(
        design,
        outcome,
        _event_equal_weights(work),
        ridge_strength=ridge_strength,
    )
    bootstrap = _bootstrap_coefficients(
        work,
        spec=spec,
        samples=bootstrap_samples,
        seed=seed,
        ridge_strength=ridge_strength,
    )
    return FittedProbabilityStack(
        spec=spec,
        coefficients=tuple(float(value) for value in coefficients),
        bootstrap_coefficients=tuple(
            tuple(float(value) for value in row) for row in bootstrap
        ),
        fit_fold_ids=tuple(map(str, fit_fold_ids)),
        fit_target_start=str(work["target_date"].astype(str).min()),
        fit_target_end=str(work["target_date"].astype(str).max()),
        fit_event_days=int(work["event_id"].nunique()),
        ridge_strength=float(ridge_strength),
        seed=int(seed),
    )


def fit_expanding_probability_stacks(
    samples: pd.DataFrame,
    *,
    specs: Sequence[ProbabilityStackSpec] = DEFAULT_STACK_SPECS,
    bootstrap_samples: int = 500,
    seed: int = 42,
    ridge_strength: float = 1.0,
) -> ExpandingProbabilityStackResult:
    """Fit each evaluation fold from strictly earlier folds."""
    if "fold_id" not in samples:
        raise ValueError("expanding probability stacks require fold_id")
    names = [spec.name for spec in specs]
    if not specs or len(names) != len(set(names)):
        raise ValueError("probability stack specs must be non-empty and uniquely named")
    work = samples.copy().reset_index(drop=True)
    folds = sorted(work["fold_id"].astype(str).unique())
    scored_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []
    models: dict[tuple[str, str], FittedProbabilityStack] = {}
    for fold_index, evaluation_fold in enumerate(folds[1:], start=1):
        fit_folds = folds[:fold_index]
        training = work[work["fold_id"].astype(str).isin(fit_folds)]
        evaluation = work[work["fold_id"].astype(str) == evaluation_fold]
        if evaluation.empty:
            raise ValueError(f"probability stack evaluation fold is empty: {evaluation_fold}")
        for spec in specs:
            model = fit_probability_stack(
                training,
                spec=spec,
                bootstrap_samples=bootstrap_samples,
                seed=seed + fold_index,
                ridge_strength=ridge_strength,
                fit_fold_ids=fit_folds,
            )
            scored_parts.append(model.apply(evaluation))
            coefficient_parts.append(
                model.coefficient_frame(evaluation_fold_id=evaluation_fold)
            )
            models[(evaluation_fold, spec.name)] = model
    return ExpandingProbabilityStackResult(
        scored_samples=pd.concat(scored_parts, ignore_index=True),
        coefficients=pd.concat(coefficient_parts, ignore_index=True),
        models=models,
    )


def apply_expanding_probability_stacks(
    candidates: pd.DataFrame,
    models: Mapping[tuple[str, str], FittedProbabilityStack],
) -> pd.DataFrame:
    """Apply each earlier-fold model to every matching full candidate tick."""
    if "fold_id" not in candidates:
        raise ValueError("probability stack candidates require fold_id")
    evaluation_folds = sorted({fold_id for fold_id, _ in models})
    candidate_folds = sorted(candidates["fold_id"].astype(str).unique())
    omitted = sorted(set(candidate_folds) - set(evaluation_folds))
    if omitted != candidate_folds[:1]:
        raise ValueError("candidate folds do not match expanding stack models")
    parts: list[pd.DataFrame] = []
    for (fold_id, source), model in sorted(models.items()):
        rows = candidates[candidates["fold_id"].astype(str) == fold_id]
        if rows.empty or source != model.spec.name:
            raise ValueError("expanding stack model contract is inconsistent")
        parts.append(model.apply(rows))
    return pd.concat(parts, ignore_index=True)


def score_probability_stack_samples(scored: pd.DataFrame) -> pd.DataFrame:
    """Attach binary proper scores to identical OOF union samples."""
    required = {
        "event_id", "snapshot_id", "target_date",
        "probability_source", "calibrated_union_probability", "union_outcome",
    }
    missing = required - set(scored)
    if missing:
        raise ValueError("probability stack scores missing columns: " + ", ".join(sorted(missing)))
    if "fold_id" not in scored and "evaluation_id" not in scored:
        raise ValueError("probability stack scores require fold_id or evaluation_id")
    work = scored.copy()
    probability = work["calibrated_union_probability"].to_numpy(float)
    outcome = work["union_outcome"].to_numpy(float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("calibrated union probabilities must be finite in [0, 1]")
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("union outcomes must be binary")
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    work["binary_brier"] = np.square(probability - outcome)
    work["binary_log_loss"] = -(
        outcome * np.log(clipped) + (1.0 - outcome) * np.log1p(-clipped)
    )
    return work


def build_probability_stack_report(
    *,
    scores: pd.DataFrame,
    coefficients: pd.DataFrame,
    candidates: pd.DataFrame,
    settlements: pd.DataFrame,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> dict[str, object]:
    """Report incremental probability information before execution evidence."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    sources = {spec.name for spec in DEFAULT_STACK_SPECS}
    if set(scores["probability_source"].astype(str).unique()) != sources:
        raise ValueError("proper-score panel must contain all probability sources")
    event_scores = (
        scores.groupby(
            ["event_id", "target_date", "fold_id", "probability_source"],
            as_index=False,
            sort=True,
        )[["binary_brier", "binary_log_loss"]]
        .mean()
    )
    score_overall = [
        {
            "probability_source": source,
            "event_days": int(group["event_id"].nunique()),
            "binary_brier": float(group["binary_brier"].mean()),
            "binary_log_loss": float(group["binary_log_loss"].mean()),
        }
        for source, group in event_scores.groupby("probability_source", sort=True)
    ]
    score_by_fold = [
        {
            "fold_id": str(fold_id),
            "probability_source": str(source),
            "event_days": int(group["event_id"].nunique()),
            "binary_brier": float(group["binary_brier"].mean()),
            "binary_log_loss": float(group["binary_log_loss"].mean()),
        }
        for (fold_id, source), group in event_scores.groupby(
            ["fold_id", "probability_source"], sort=True
        )
    ]
    plus_vs_market = _paired_score_summary(
        event_scores,
        candidate="model_plus_market",
        baseline="market_only",
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    plus_vs_current = _paired_score_summary(
        event_scores,
        candidate="model_plus_market",
        baseline="current_model_only",
        bootstrap_samples=bootstrap_samples,
        seed=seed + 10,
    )

    universe = candidates.loc[:, ["event_id", "target_date", "fold_id"]].drop_duplicates()
    if universe.duplicated("event_id").any():
        raise ValueError("candidate Event belongs to more than one historical fold")
    execution_overall = [
        {
            "probability_source": source,
            **_strategy_pnl_summary(
                universe,
                settlements[settlements["probability_source"].astype(str) == source],
                bootstrap_samples=bootstrap_samples,
                seed=seed + 100 + index,
            ),
        }
        for index, source in enumerate(sorted(sources))
    ]
    execution_by_fold: list[dict[str, object]] = []
    for fold_index, (fold_id, fold_universe) in enumerate(universe.groupby("fold_id", sort=True)):
        for source_index, source in enumerate(sorted(sources)):
            source_settlements = settlements[
                (settlements["fold_id"].astype(str) == str(fold_id))
                & (settlements["probability_source"].astype(str) == source)
            ]
            execution_by_fold.append({
                "fold_id": str(fold_id),
                "probability_source": source,
                **_strategy_pnl_summary(
                    fold_universe,
                    source_settlements,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + 200 + fold_index * 10 + source_index,
                ),
            })
    execution_by_horizon = _conditional_execution_summary(
        settlements, "horizon_band", bootstrap_samples=bootstrap_samples, seed=seed + 300
    )
    execution_by_gap = _conditional_execution_summary(
        settlements,
        "model_market_gap_band",
        bootstrap_samples=bootstrap_samples,
        seed=seed + 400,
    )
    plus_pnl_vs_market = _paired_strategy_pnl(
        universe,
        settlements,
        candidate="model_plus_market",
        baseline="market_only",
        bootstrap_samples=bootstrap_samples,
        seed=seed + 500,
    )
    plus_pnl_vs_current = _paired_strategy_pnl(
        universe,
        settlements,
        candidate="model_plus_market",
        baseline="current_model_only",
        bootstrap_samples=bootstrap_samples,
        seed=seed + 510,
    )
    coefficient_points = coefficients[
        (coefficients["coefficient_kind"].astype(str) == "point")
        & (coefficients["evaluation_fold_id"].astype(str).str.startswith("fold-"))
    ]
    plus_points = coefficient_points[
        coefficient_points["probability_source"].astype(str) == "model_plus_market"
    ]
    model_positive = int((plus_points["model_logit"].astype(float) > 0).sum())
    coefficient_evidence = {
        "evaluated_fold_count": int(len(plus_points)),
        "positive_model_coefficient_fold_count": model_positive,
        "point_coefficients": plus_points.loc[:, [
            "evaluation_fold_id", "model_logit", "midpoint_logit", "is_36_42h",
        ]].to_dict("records"),
    }
    ledger = [
        {
            "question_id": "incremental_model_information",
            "question": "Does MC union probability improve Event-equal OOF proper scores after normalized midpoint and fixed horizon state are known?",
            "status": _paired_score_status(plus_vs_market),
            "evidence": plus_vs_market,
            "artifact": "probability_scores.parquet",
        },
        {
            "question_id": "market_aware_taker_vs_current_a2",
            "question": "Does the market-aware stack improve frozen historical taker PnL over current model-only A2?",
            "status": _paired_pnl_status(plus_pnl_vs_current),
            "evidence": plus_pnl_vs_current,
            "artifact": "event_pnl.parquet",
        },
        {
            "question_id": "market_aware_taker_vs_market_only",
            "question": "Does the model-plus-market taker replay improve on market-only probability for the same model-selected union?",
            "status": _paired_pnl_status(plus_pnl_vs_market),
            "evidence": plus_pnl_vs_market,
            "artifact": "event_pnl.parquet",
        },
        {
            "question_id": "model_coefficient_stability",
            "question": "Is the conditional model coefficient positive across historical evaluation folds?",
            "status": (
                "supported" if len(plus_points) > 0 and model_positive == len(plus_points)
                else "contradicted" if len(plus_points) > 0 and model_positive == 0
                else "inconclusive"
            ),
            "evidence": coefficient_evidence,
            "artifact": "calibration_coefficients.parquet",
        },
        {
            "question_id": "historical_state_risk",
            "question": "How do fixed horizon and model-market disagreement states concentrate execution risk?",
            "status": "inconclusive",
            "evidence": {
                "by_horizon": execution_by_horizon,
                "by_model_market_gap": execution_by_gap,
                "limitation": "descriptive fixed strata are not outcome-selected deployment gates",
            },
            "artifact": "report.json",
        },
    ]
    return {
        "schema_version": 1,
        "data_class": "two_bucket_probability_stack_historical_oof",
        "primary_estimand": "event_equal_binary_proper_score_increment_after_normalized_midpoint",
        "proper_scores": {
            "overall": score_overall,
            "by_fold": score_by_fold,
            "model_plus_market_minus_market_only": plus_vs_market,
            "model_plus_market_minus_current_model_only": plus_vs_current,
        },
        "execution": {
            "overall": execution_overall,
            "by_fold": execution_by_fold,
            "by_horizon": execution_by_horizon,
            "by_model_market_gap": execution_by_gap,
            "model_plus_market_minus_market_only": plus_pnl_vs_market,
            "model_plus_market_minus_current_model_only": plus_pnl_vs_current,
        },
        "coefficients": coefficient_evidence,
        "limitations": {
            "model_selected_pair": "All sources score the same pair selected by the MC distribution; market_only does not independently select a pair.",
            "calibration_tick_sampling": "Calibration and proper scores use the first eligible market tick per Event/forecast snapshot; execution scans every eligible 5-minute tick.",
            "development_only": "Historical folds are consumed development evidence, not an independent holdout.",
            "april_may_not_read": True,
        },
        "counts": {
            "score_rows": int(len(scores)),
            "evaluation_event_days": int(universe["event_id"].nunique()),
            "settlements": int(len(settlements)),
        },
        "decision_ledger": ledger,
    }


def write_probability_stack_artifact(
    *,
    output: Path | str,
    frames: Mapping[str, pd.DataFrame],
    report: Mapping[str, object],
    input_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Persist one immutable, hash-bound historical stack artifact."""
    expected = {
        "probability_scores", "calibration_coefficients", "probability_panel",
        "decisions", "execution_legs", "event_pnl",
    }
    if set(frames) != expected:
        raise ValueError("probability stack artifact frame set is incomplete")
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"probability stack output already exists: {root}")
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
        "data_class": "two_bucket_probability_stack_historical_oof",
        "input_sha256": dict(sorted(input_sha256.items())),
        "sha256": hashes,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
    )
    return manifest


def _paired_score_summary(
    event_scores: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    candidate_rows = event_scores[event_scores["probability_source"] == candidate]
    baseline_rows = event_scores[event_scores["probability_source"] == baseline]
    paired = candidate_rows.merge(
        baseline_rows,
        on=["event_id", "target_date", "fold_id"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if len(paired) != len(candidate_rows) or len(paired) != len(baseline_rows):
        raise ValueError("proper-score sources do not share identical Event samples")
    result: dict[str, object] = {
        "candidate": candidate,
        "baseline": baseline,
        "event_days": int(len(paired)),
    }
    fold_rows: dict[str, dict[str, object]] = {
        str(fold_id): {"fold_id": str(fold_id), "event_days": int(len(group))}
        for fold_id, group in paired.groupby("fold_id", sort=True)
    }
    for metric, output_name, offset in (
        ("binary_log_loss", "log_loss", 0),
        ("binary_brier", "brier", 1),
    ):
        delta = (
            paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
        ).to_numpy(float)
        means = _bootstrap_means(delta, samples=bootstrap_samples, seed=seed + offset)
        result[f"{output_name}_delta"] = float(delta.mean())
        result[f"{output_name}_delta_ci90_low"] = float(np.quantile(means, 0.05))
        result[f"{output_name}_delta_ci90_high"] = float(np.quantile(means, 0.95))
        result[f"bootstrap_probability_{output_name}_improved"] = float(np.mean(means < 0))
        for fold_id, group in paired.groupby("fold_id", sort=True):
            fold_rows[str(fold_id)][f"{output_name}_delta"] = float(
                (group[f"{metric}_candidate"] - group[f"{metric}_baseline"]).mean()
            )
    result["folds"] = list(fold_rows.values())
    result["log_loss_improved_fold_count"] = int(
        sum(float(row["log_loss_delta"]) < 0 for row in fold_rows.values())
    )
    result["brier_improved_fold_count"] = int(
        sum(float(row["brier_delta"]) < 0 for row in fold_rows.values())
    )
    return result


def _strategy_pnl_summary(
    universe: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    pnl = settlements.groupby("event_id", sort=True)["realized_pnl_usd"].sum()
    event = universe.loc[:, ["event_id", "target_date"]].drop_duplicates("event_id").copy()
    event["realized_pnl_usd"] = event["event_id"].map(pnl).fillna(0.0)
    values = event["realized_pnl_usd"].to_numpy(float)
    means = _bootstrap_means(values, samples=bootstrap_samples, seed=seed)
    cost = float(settlements["cost_basis_usd"].sum())
    total = float(values.sum())
    return {
        "event_days": int(len(event)),
        "traded_event_days": int(settlements["event_id"].nunique()),
        "turnover_usd": cost,
        "total_pnl_usd": total,
        "roi": total / cost if cost > 0 else None,
        "mean_event_day_pnl_usd": float(values.mean()),
        "mean_event_day_pnl_ci90_low": float(np.quantile(means, 0.05)),
        "mean_event_day_pnl_ci90_high": float(np.quantile(means, 0.95)),
    }


def _paired_strategy_pnl(
    universe: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    event = (
        universe[["event_id", "target_date", "fold_id"]]
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )
    by_source = settlements.groupby(
        ["event_id", "probability_source"], sort=True
    )["realized_pnl_usd"].sum().unstack(fill_value=0.0)
    candidate_pnl = event["event_id"].map(
        by_source[candidate] if candidate in by_source else pd.Series(dtype=float)
    ).fillna(0.0)
    baseline_pnl = event["event_id"].map(
        by_source[baseline] if baseline in by_source else pd.Series(dtype=float)
    ).fillna(0.0)
    delta = (candidate_pnl - baseline_pnl).to_numpy(float)
    means = _bootstrap_means(delta, samples=bootstrap_samples, seed=seed)
    fold_rows = []
    for fold_id, indexes in event.groupby("fold_id", sort=True).groups.items():
        values = delta[np.asarray(list(indexes), dtype=int)]
        fold_rows.append({
            "fold_id": str(fold_id),
            "event_days": int(len(values)),
            "mean_pnl_delta_usd": float(values.mean()),
        })
    return {
        "candidate": candidate,
        "baseline": baseline,
        "event_days": int(len(event)),
        "mean_pnl_delta_usd": float(delta.mean()),
        "mean_pnl_delta_ci90_low": float(np.quantile(means, 0.05)),
        "mean_pnl_delta_ci90_high": float(np.quantile(means, 0.95)),
        "bootstrap_probability_candidate_better": float(np.mean(means > 0.0)),
        "positive_fold_count": int(sum(row["mean_pnl_delta_usd"] > 0 for row in fold_rows)),
        "folds": fold_rows,
    }


def _conditional_execution_summary(
    settlements: pd.DataFrame,
    stratum: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    if settlements.empty or stratum not in settlements:
        return []
    rows: list[dict[str, object]] = []
    for index, ((source, value), group) in enumerate(
        settlements.groupby(["probability_source", stratum], sort=True)
    ):
        universe = group[["event_id", "target_date"]].drop_duplicates().assign(fold_id="conditional")
        rows.append({
            "probability_source": str(source),
            stratum: str(value),
            **_strategy_pnl_summary(
                universe,
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


def _paired_score_status(summary: Mapping[str, object]) -> str:
    if (
        float(summary["log_loss_delta_ci90_high"]) < 0
        and float(summary["brier_delta_ci90_high"]) < 0
    ):
        return "supported"
    if (
        float(summary["log_loss_delta_ci90_low"]) > 0
        and float(summary["brier_delta_ci90_low"]) > 0
    ):
        return "contradicted"
    return "inconclusive"


def _paired_pnl_status(summary: Mapping[str, object]) -> str:
    if float(summary["mean_pnl_delta_ci90_low"]) > 0:
        return "supported"
    if float(summary["mean_pnl_delta_ci90_high"]) < 0:
        return "contradicted"
    return "inconclusive"


def _design_matrix(frame: pd.DataFrame, spec: ProbabilityStackSpec) -> np.ndarray:
    feature_values: list[np.ndarray] = []
    for feature in spec.features:
        if feature == "model_logit":
            values = _logit_probability(frame["model_union_probability"].to_numpy(float))
        elif feature == "midpoint_logit":
            values = _logit_probability(frame["midpoint_union_probability"].to_numpy(float))
        elif feature == "is_36_42h":
            lead = frame["lead_hours"].to_numpy(float)
            if not np.isfinite(lead).all():
                raise ValueError("lead_hours must be finite")
            values = ((lead >= 36.0) & (lead < 42.0)).astype(float)
        else:
            raise ValueError(f"unsupported probability stack feature: {feature}")
        feature_values.append(values)
    return np.column_stack([np.ones(len(frame)), *feature_values])


def _logit_probability(probability: np.ndarray) -> np.ndarray:
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("stack probabilities must be finite in [0, 1]")
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _event_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_id")["event_id"].transform("size").to_numpy(float)
    return 1.0 / counts


def _fit_weighted_logistic(
    design: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    *,
    ridge_strength: float,
) -> np.ndarray:
    mean = np.clip(np.average(outcome, weights=weights), 1e-6, 1.0 - 1e-6)
    coefficients = np.zeros(design.shape[1], dtype=float)
    coefficients[0] = np.log(mean / (1.0 - mean))
    for _ in range(100):
        fitted = _sigmoid(design @ coefficients)
        gradient = design.T @ (weights * (fitted - outcome))
        gradient[1:] += ridge_strength * coefficients[1:]
        curvature = weights * fitted * (1.0 - fitted)
        hessian = design.T @ (curvature[:, None] * design)
        hessian[1:, 1:] += np.eye(design.shape[1] - 1) * ridge_strength
        hessian += np.eye(design.shape[1]) * 1e-9
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return coefficients


def _bootstrap_coefficients(
    training: pd.DataFrame,
    *,
    spec: ProbabilityStackSpec,
    samples: int,
    seed: int,
    ridge_strength: float,
) -> np.ndarray:
    events = training["event_id"].astype(str).drop_duplicates().to_numpy()
    event_values = training["event_id"].astype(str).to_numpy()
    rows_per_event = pd.Series(event_values).value_counts().to_dict()
    design = _design_matrix(training, spec)
    outcome = training["union_outcome"].to_numpy(float)
    rng = np.random.default_rng(seed)
    result = np.empty((samples, design.shape[1]), dtype=float)
    for index in range(samples):
        sampled = rng.choice(events, size=len(events), replace=True)
        multiplicity = pd.Series(sampled).value_counts().to_dict()
        weights = np.asarray([
            float(multiplicity.get(event, 0)) / float(rows_per_event[event])
            for event in event_values
        ])
        result[index] = _fit_weighted_logistic(
            design,
            outcome,
            weights,
            ridge_strength=ridge_strength,
        )
    return result


def _predict(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return _sigmoid(design @ coefficients)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def _optional_float(value: object) -> float:
    return float(value) if value is not None else float("nan")
