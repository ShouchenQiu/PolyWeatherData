"""Frozen April/May diagnostics for the two-bucket probability stack."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.two_bucket_probability_stack import (
    DEFAULT_STACK_SPECS,
    FittedProbabilityStack,
    ProbabilityStackSpec,
)


def load_frozen_probability_stacks(
    coefficients: pd.DataFrame,
    *,
    expected_bootstrap_samples: int,
    training_cutoff: str,
    coefficient_id: str = "frozen_pre_april",
) -> Mapping[str, FittedProbabilityStack]:
    """Restore hash-validated frozen stacks without fitting diagnostic outcomes."""
    required = {
        "evaluation_fold_id", "probability_source", "coefficient_kind",
        "draw_index", "intercept", "model_logit", "midpoint_logit",
        "is_36_42h", "fit_fold_ids", "fit_target_start", "fit_target_end",
        "fit_event_days", "ridge_strength", "seed",
    }
    missing = required - set(coefficients)
    if missing:
        raise ValueError(
            "frozen stack coefficients missing columns: " + ", ".join(sorted(missing))
        )
    if expected_bootstrap_samples <= 0:
        raise ValueError("expected bootstrap sample count must be positive")
    frozen = coefficients[
        coefficients["evaluation_fold_id"].astype(str) == coefficient_id
    ].copy()
    if frozen.empty:
        raise ValueError(f"frozen stack coefficient id is absent: {coefficient_id}")
    models: dict[str, FittedProbabilityStack] = {}
    for spec in DEFAULT_STACK_SPECS:
        rows = frozen[frozen["probability_source"].astype(str) == spec.name]
        point = rows[rows["coefficient_kind"].astype(str) == "point"]
        bootstrap = rows[rows["coefficient_kind"].astype(str) == "bootstrap"].sort_values(
            "draw_index"
        )
        if len(point) != 1 or len(bootstrap) != expected_bootstrap_samples:
            raise ValueError(f"frozen stack coefficient count is invalid: {spec.name}")
        if bootstrap["draw_index"].astype(int).tolist() != list(
            range(expected_bootstrap_samples)
        ):
            raise ValueError(f"frozen stack bootstrap draws are incomplete: {spec.name}")
        metadata = rows.loc[:, [
            "fit_fold_ids", "fit_target_start", "fit_target_end",
            "fit_event_days", "ridge_strength", "seed",
        ]].drop_duplicates(
            subset=["fit_target_start", "fit_target_end", "fit_event_days", "ridge_strength", "seed"]
        )
        if len(metadata) != 1:
            raise ValueError(f"frozen stack metadata is inconsistent: {spec.name}")
        meta = point.iloc[0]
        if str(meta["fit_target_end"]) > training_cutoff:
            raise ValueError(f"frozen stack reads beyond training cutoff: {spec.name}")
        models[spec.name] = FittedProbabilityStack(
            spec=spec,
            coefficients=_coefficient_tuple(point.iloc[0], spec),
            bootstrap_coefficients=tuple(
                _coefficient_tuple(row, spec) for _, row in bootstrap.iterrows()
            ),
            fit_fold_ids=_string_tuple(meta["fit_fold_ids"]),
            fit_target_start=str(meta["fit_target_start"]),
            fit_target_end=str(meta["fit_target_end"]),
            fit_event_days=int(meta["fit_event_days"]),
            ridge_strength=float(meta["ridge_strength"]),
            seed=int(meta["seed"]),
        )
    unexpected = set(frozen["probability_source"].astype(str)) - set(models)
    if unexpected:
        raise ValueError("frozen stack contains unexpected probability sources")
    return models


def apply_frozen_probability_stacks(
    candidates: pd.DataFrame,
    models: Mapping[str, FittedProbabilityStack],
) -> pd.DataFrame:
    """Apply every frozen source to identical diagnostic candidate ticks."""
    expected = {spec.name for spec in DEFAULT_STACK_SPECS}
    if set(models) != expected:
        raise ValueError("frozen diagnostic requires exactly the three probability sources")
    parts: list[pd.DataFrame] = []
    for spec in DEFAULT_STACK_SPECS:
        model = models[spec.name]
        if model.spec != spec:
            raise ValueError(f"frozen probability feature contract mismatch: {spec.name}")
        parts.append(model.apply(candidates))
    return pd.concat(parts, ignore_index=True)


def build_frozen_probability_stack_report(
    *,
    scores: pd.DataFrame,
    candidates: pd.DataFrame,
    settlements: pd.DataFrame,
    exclusions: pd.DataFrame,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> dict[str, object]:
    """Compare three frozen probability sources separately in April and May."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    sources = {spec.name for spec in DEFAULT_STACK_SPECS}
    if set(scores["probability_source"].astype(str).unique()) != sources:
        raise ValueError("frozen scores must contain exactly the three probability sources")
    required_scores = {
        "event_id", "target_date", "evaluation_id", "probability_source",
        "binary_brier", "binary_log_loss",
    }
    missing = required_scores - set(scores)
    if missing:
        raise ValueError("frozen score panel missing columns: " + ", ".join(sorted(missing)))
    event_scores = (
        scores.groupby(
            ["event_id", "target_date", "evaluation_id", "probability_source"],
            as_index=False,
            sort=True,
        )[["binary_brier", "binary_log_loss"]]
        .mean()
    )
    _require_matched_sources(event_scores, sources)
    evaluations = sorted(event_scores["evaluation_id"].astype(str).unique())
    if evaluations != ["april_2026", "may_2026"]:
        raise ValueError("frozen probability report requires April and May evaluations")

    proper_overall = _proper_score_rows(event_scores, [])
    proper_by_evaluation = _proper_score_rows(event_scores, ["evaluation_id"])
    score_comparisons = {
        name: {
            "overall": _paired_score(
                event_scores,
                candidate="model_plus_market",
                baseline=baseline,
                bootstrap_samples=bootstrap_samples,
                seed=seed + offset,
            ),
            "by_evaluation": [
                {
                    "evaluation_id": evaluation,
                    **_paired_score(
                        event_scores[event_scores["evaluation_id"].astype(str) == evaluation],
                        candidate="model_plus_market",
                        baseline=baseline,
                        bootstrap_samples=bootstrap_samples,
                        seed=seed + offset + index + 1,
                    ),
                }
                for index, evaluation in enumerate(evaluations)
            ],
        }
        for name, baseline, offset in (
            ("model_plus_market_minus_market_only", "market_only", 0),
            ("model_plus_market_minus_current_model_only", "current_model_only", 10),
        )
    }

    universe = candidates.loc[:, [
        "event_id", "target_date", "evaluation_id",
    ]].drop_duplicates()
    if universe.duplicated("event_id").any():
        raise ValueError("diagnostic Event belongs to more than one evaluation")
    execution_overall = _execution_rows(
        universe,
        settlements,
        group_columns=[],
        sources=sources,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 100,
    )
    execution_by_evaluation = _execution_rows(
        universe,
        settlements,
        group_columns=["evaluation_id"],
        sources=sources,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 200,
    )
    pnl_comparisons = {
        name: {
            "overall": _paired_pnl(
                universe,
                settlements,
                candidate="model_plus_market",
                baseline=baseline,
                bootstrap_samples=bootstrap_samples,
                seed=seed + offset,
            ),
            "by_evaluation": [
                {
                    "evaluation_id": evaluation,
                    **_paired_pnl(
                        universe[universe["evaluation_id"].astype(str) == evaluation],
                        settlements[
                            settlements["evaluation_id"].astype(str) == evaluation
                        ],
                        candidate="model_plus_market",
                        baseline=baseline,
                        bootstrap_samples=bootstrap_samples,
                        seed=seed + offset + index + 1,
                    ),
                }
                for index, evaluation in enumerate(evaluations)
            ],
        }
        for name, baseline, offset in (
            ("model_plus_market_minus_market_only", "market_only", 300),
            ("model_plus_market_minus_current_model_only", "current_model_only", 310),
        )
    }
    by_horizon = _conditional_execution_rows(
        settlements,
        "horizon_band",
        bootstrap_samples=bootstrap_samples,
        seed=seed + 400,
    )
    by_gap = _conditional_execution_rows(
        settlements,
        "model_market_gap_band",
        bootstrap_samples=bootstrap_samples,
        seed=seed + 500,
    )
    entry_transitions = _entry_transition_rows(
        universe,
        settlements,
        candidate="model_plus_market",
        baseline="current_model_only",
    )

    plus_vs_market = score_comparisons["model_plus_market_minus_market_only"][
        "by_evaluation"
    ]
    plus_vs_current = pnl_comparisons[
        "model_plus_market_minus_current_model_only"
    ]["by_evaluation"]
    plus_execution = {
        str(row["evaluation_id"]): row
        for row in execution_by_evaluation
        if row["probability_source"] == "model_plus_market"
    }
    ledger: list[dict[str, object]] = []
    for evaluation, label in (("april_2026", "April"), ("may_2026", "May")):
        score_evidence = next(
            row for row in plus_vs_market if row["evaluation_id"] == evaluation
        )
        pnl_evidence = next(
            row for row in plus_vs_current if row["evaluation_id"] == evaluation
        )
        ledger.extend([
            {
                "question_id": f"{evaluation.split('_')[0]}_incremental_model_information",
                "question": f"Does frozen MC probability improve {label} proper scores after normalized midpoint is known?",
                "status": _score_status(score_evidence),
                "evidence": score_evidence,
                "artifact": "probability_scores.parquet",
            },
            {
                "question_id": f"{evaluation.split('_')[0]}_market_aware_pnl_vs_current",
                "question": f"Does the frozen market-aware stack improve {label} taker PnL over current model-only A2?",
                "status": _pnl_status(pnl_evidence),
                "evidence": pnl_evidence,
                "artifact": "event_pnl.parquet",
            },
            {
                "question_id": f"{evaluation.split('_')[0]}_market_aware_profitability",
                "question": f"Is frozen model-plus-market taker PnL positive in {label}?",
                "status": _absolute_pnl_status(plus_execution[evaluation]),
                "evidence": plus_execution[evaluation],
                "artifact": "event_pnl.parquet",
            },
        ])

    return {
        "schema_version": 1,
        "data_class": "frozen_two_bucket_probability_stack_april_may_diagnostic",
        "diagnostic_only": True,
        "training_cutoff": "2026-03-31",
        "primary_estimand": "event_equal_incremental_union_probability_after_normalized_midpoint",
        "proper_scores": {
            "overall": proper_overall,
            "by_evaluation": proper_by_evaluation,
            **score_comparisons,
        },
        "execution": {
            "overall": execution_overall,
            "by_evaluation": execution_by_evaluation,
            "by_evaluation_horizon_conditional_on_entry": by_horizon,
            "by_evaluation_model_market_gap_conditional_on_entry": by_gap,
            "entry_transitions_model_plus_market_vs_current_model_only": entry_transitions,
            **pnl_comparisons,
        },
        "counts": {
            "score_rows": int(len(scores)),
            "candidate_ticks": int(len(candidates)),
            "eligible_event_days": int(universe["event_id"].nunique()),
            "settlements": int(len(settlements)),
            "excluded_target_days": int(exclusions["target_date"].nunique())
            if len(exclusions) else 0,
        },
        "exclusions": exclusions.to_dict("records"),
        "limitations": {
            "known_diagnostic_data": "April and May were previously inspected and are not independent holdouts.",
            "model_selected_pair": "All sources score and trade the same adjacent pair selected by the MC distribution.",
            "gap_is_not_a_feature_or_gate": True,
            "conditional_execution_strata": "Horizon and gap summaries condition on entered Events and are descriptive only.",
        },
        "decision_ledger": ledger,
    }


def write_frozen_probability_stack_artifact(
    *,
    output: Path | str,
    frames: Mapping[str, pd.DataFrame],
    report: Mapping[str, object],
    input_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Persist one immutable, hash-bound April/May stack diagnostic."""
    expected = {
        "probability_scores", "calibration_coefficients", "probability_panel",
        "decisions", "execution_legs", "event_pnl", "exclusions",
    }
    if set(frames) != expected:
        raise ValueError("frozen probability stack artifact frame set is incomplete")
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"frozen probability stack output already exists: {root}")
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
        "data_class": "frozen_two_bucket_probability_stack_april_may_diagnostic",
        "input_sha256": dict(sorted(input_sha256.items())),
        "sha256": hashes,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
    )
    return manifest


def _require_matched_sources(
    event_scores: pd.DataFrame,
    sources: set[str],
) -> None:
    counts = event_scores.groupby("event_id")["probability_source"].agg(
        lambda values: set(map(str, values))
    )
    if not all(value == sources for value in counts):
        raise ValueError("probability sources do not share identical diagnostic Events")


def _proper_score_rows(
    event_scores: pd.DataFrame,
    group_columns: list[str],
) -> list[dict[str, object]]:
    columns = [*group_columns, "probability_source"]
    rows: list[dict[str, object]] = []
    for keys, group in event_scores.groupby(columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        rows.append({
            **dict(zip(columns, map(str, key_values), strict=True)),
            "event_days": int(group["event_id"].nunique()),
            "binary_brier": float(group["binary_brier"].mean()),
            "binary_log_loss": float(group["binary_log_loss"].mean()),
        })
    return rows


def _paired_score(
    event_scores: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    left = event_scores[event_scores["probability_source"] == candidate]
    right = event_scores[event_scores["probability_source"] == baseline]
    paired = left.merge(
        right,
        on=["event_id", "target_date", "evaluation_id"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError("proper-score sources do not share identical diagnostic Events")
    result: dict[str, object] = {
        "candidate": candidate,
        "baseline": baseline,
        "event_days": int(len(paired)),
    }
    for offset, (column, name) in enumerate(
        (("binary_log_loss", "log_loss"), ("binary_brier", "brier"))
    ):
        delta = (
            paired[f"{column}_candidate"] - paired[f"{column}_baseline"]
        ).to_numpy(float)
        means = _bootstrap_means(delta, bootstrap_samples, seed + offset)
        result[f"{name}_delta"] = float(delta.mean())
        result[f"{name}_delta_ci90_low"] = float(np.quantile(means, 0.05))
        result[f"{name}_delta_ci90_high"] = float(np.quantile(means, 0.95))
        result[f"bootstrap_probability_{name}_improved"] = float(np.mean(means < 0))
    return result


def _execution_rows(
    universe: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    group_columns: list[str],
    sources: set[str],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    groups = [((), universe)] if not group_columns else universe.groupby(group_columns, sort=True)
    rows: list[dict[str, object]] = []
    for group_index, (keys, group_universe) in enumerate(groups):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        group_identity = dict(zip(group_columns, map(str, key_values), strict=True))
        event_ids = set(group_universe["event_id"].astype(str))
        for source_index, source in enumerate(sorted(sources)):
            source_rows = settlements[
                settlements["probability_source"].astype(str).eq(source)
                & settlements["event_id"].astype(str).isin(event_ids)
            ]
            rows.append({
                **group_identity,
                "probability_source": source,
                **_execution_summary(
                    group_universe,
                    source_rows,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + group_index * 10 + source_index,
                ),
            })
    return rows


def _execution_summary(
    universe: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    pnl = settlements.groupby("event_id")["realized_pnl_usd"].sum()
    event = universe[["event_id"]].drop_duplicates().copy()
    event["pnl"] = event["event_id"].map(pnl).fillna(0.0)
    values = event["pnl"].to_numpy(float)
    means = _bootstrap_means(values, bootstrap_samples, seed)
    cost = float(settlements["cost_basis_usd"].sum())
    total = float(values.sum())
    result = {
        "event_days": int(len(event)),
        "traded_event_days": int(settlements["event_id"].nunique()),
        "turnover_usd": cost,
        "total_pnl_usd": total,
        "roi": total / cost if cost > 0 else None,
        "mean_event_day_pnl_usd": float(values.mean()),
        "mean_event_day_pnl_ci90_low": float(np.quantile(means, 0.05)),
        "mean_event_day_pnl_ci90_high": float(np.quantile(means, 0.95)),
        "bootstrap_probability_mean_pnl_positive": float(np.mean(means > 0)),
    }
    if len(settlements) and "union_outcome" in settlements:
        result["union_hit_rate"] = float(settlements["union_outcome"].mean())
    if len(settlements) and "union_probability_lower_90" in settlements:
        result["mean_probability_lower_90"] = float(
            settlements["union_probability_lower_90"].mean()
        )
    if len(settlements) and {"shares", "cost_basis_usd"}.issubset(settlements):
        result["mean_effective_unit_cost"] = float(
            (settlements["cost_basis_usd"] / settlements["shares"]).mean()
        )
    if len(settlements) and "lead_hours" in settlements:
        result["mean_lead_hours"] = float(settlements["lead_hours"].mean())
    return result


def _paired_pnl(
    universe: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    event = universe[["event_id"]].drop_duplicates().copy()
    by_source = settlements.groupby(
        ["event_id", "probability_source"]
    )["realized_pnl_usd"].sum().unstack(fill_value=0.0)
    candidate_values = event["event_id"].map(
        by_source[candidate] if candidate in by_source else pd.Series(dtype=float)
    ).fillna(0.0)
    baseline_values = event["event_id"].map(
        by_source[baseline] if baseline in by_source else pd.Series(dtype=float)
    ).fillna(0.0)
    delta = (candidate_values - baseline_values).to_numpy(float)
    means = _bootstrap_means(delta, bootstrap_samples, seed)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "event_days": int(len(event)),
        "mean_pnl_delta_usd": float(delta.mean()),
        "total_pnl_delta_usd": float(delta.sum()),
        "mean_pnl_delta_ci90_low": float(np.quantile(means, 0.05)),
        "mean_pnl_delta_ci90_high": float(np.quantile(means, 0.95)),
        "bootstrap_probability_candidate_better": float(np.mean(means > 0)),
    }


def _conditional_execution_rows(
    settlements: pd.DataFrame,
    stratum: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, object]]:
    if settlements.empty or stratum not in settlements:
        return []
    rows: list[dict[str, object]] = []
    columns = ["evaluation_id", "probability_source", stratum]
    for index, (keys, group) in enumerate(settlements.groupby(columns, sort=True)):
        universe = group[["event_id"]].drop_duplicates()
        rows.append({
            **dict(zip(columns, map(str, keys), strict=True)),
            **_execution_summary(
                universe,
                group,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index,
            ),
        })
    return rows


def _entry_transition_rows(
    universe: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
) -> list[dict[str, object]]:
    if settlements.duplicated(["event_id", "probability_source"]).any():
        raise ValueError("entry transition requires at most one trade per Event/source")
    rows: list[dict[str, object]] = []
    for evaluation, group_universe in universe.groupby("evaluation_id", sort=True):
        event_ids = group_universe["event_id"].astype(str).tolist()
        evaluation_settlements = settlements[
            settlements["event_id"].astype(str).isin(event_ids)
        ]
        candidate_rows = evaluation_settlements[
            evaluation_settlements["probability_source"].astype(str) == candidate
        ].set_index("event_id")
        baseline_rows = evaluation_settlements[
            evaluation_settlements["probability_source"].astype(str) == baseline
        ].set_index("event_id")
        counts = {
            "both_traded_event_days": 0,
            "candidate_only_event_days": 0,
            "baseline_only_event_days": 0,
            "neither_traded_event_days": 0,
            "same_entry_timestamp_event_days": 0,
            "different_entry_timestamp_event_days": 0,
        }
        deltas = {
            "pnl_delta_same_entry_usd": 0.0,
            "pnl_delta_different_entry_usd": 0.0,
            "pnl_delta_candidate_only_usd": 0.0,
            "pnl_delta_baseline_only_usd": 0.0,
        }
        for event_id in event_ids:
            has_candidate = event_id in candidate_rows.index
            has_baseline = event_id in baseline_rows.index
            if has_candidate and has_baseline:
                counts["both_traded_event_days"] += 1
                candidate_row = candidate_rows.loc[event_id]
                baseline_row = baseline_rows.loc[event_id]
                delta = float(candidate_row["realized_pnl_usd"]) - float(
                    baseline_row["realized_pnl_usd"]
                )
                if pd.Timestamp(candidate_row["timestamp"]) == pd.Timestamp(
                    baseline_row["timestamp"]
                ):
                    counts["same_entry_timestamp_event_days"] += 1
                    deltas["pnl_delta_same_entry_usd"] += delta
                else:
                    counts["different_entry_timestamp_event_days"] += 1
                    deltas["pnl_delta_different_entry_usd"] += delta
            elif has_candidate:
                counts["candidate_only_event_days"] += 1
                deltas["pnl_delta_candidate_only_usd"] += float(
                    candidate_rows.loc[event_id, "realized_pnl_usd"]
                )
            elif has_baseline:
                counts["baseline_only_event_days"] += 1
                deltas["pnl_delta_baseline_only_usd"] -= float(
                    baseline_rows.loc[event_id, "realized_pnl_usd"]
                )
            else:
                counts["neither_traded_event_days"] += 1
        total_delta = float(sum(deltas.values()))
        rows.append({
            "evaluation_id": str(evaluation),
            "candidate": candidate,
            "baseline": baseline,
            "event_days": int(len(event_ids)),
            **counts,
            **deltas,
            "total_pnl_delta_usd": total_delta,
        })
    return rows


def _bootstrap_means(values: np.ndarray, samples: int, seed: int) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("bootstrap requires at least one Event")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(samples, len(values)))
    return values[indexes].mean(axis=1)


def _score_status(evidence: Mapping[str, object]) -> str:
    if (
        float(evidence["log_loss_delta_ci90_high"]) < 0
        and float(evidence["brier_delta_ci90_high"]) < 0
    ):
        return "supported"
    if (
        float(evidence["log_loss_delta_ci90_low"]) > 0
        and float(evidence["brier_delta_ci90_low"]) > 0
    ):
        return "contradicted"
    return "inconclusive"


def _pnl_status(evidence: Mapping[str, object]) -> str:
    if float(evidence["mean_pnl_delta_ci90_low"]) > 0:
        return "supported"
    if float(evidence["mean_pnl_delta_ci90_high"]) < 0:
        return "contradicted"
    return "inconclusive"


def _absolute_pnl_status(evidence: Mapping[str, object]) -> str:
    if float(evidence["mean_event_day_pnl_ci90_low"]) > 0:
        return "supported"
    if float(evidence["mean_event_day_pnl_ci90_high"]) < 0:
        return "contradicted"
    return "inconclusive"


def _coefficient_tuple(
    row: pd.Series,
    spec: ProbabilityStackSpec,
) -> tuple[float, ...]:
    values = tuple(float(row[name]) for name in ("intercept", *spec.features))
    if not np.isfinite(values).all():
        raise ValueError(f"frozen stack has non-finite coefficients: {spec.name}")
    return values


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("frozen stack fit folds must be a sequence")
    return tuple(map(str, value))
