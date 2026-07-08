"""Adaptive development-only slices of Forecast Signal Study evidence."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.signal_study import load_signal_study_artifact


_VECTOR_KEYS = ["event_id", "target_date", "timestamp"]


@dataclass(frozen=True)
class SliceDefinition:
    slice_id: str
    dimensions: tuple[str, ...]
    values: tuple[str, ...]


@dataclass(frozen=True)
class ForecastSignalSliceArtifact:
    manifest: dict
    summary: dict
    vector_features: pd.DataFrame
    token_candidates: pd.DataFrame
    slice_atlas: pd.DataFrame
    crossfit_results: pd.DataFrame
    fold_rules: pd.DataFrame
    shortlist: pd.DataFrame


@dataclass(frozen=True)
class FrozenSliceValidationResult:
    summary: dict
    slices: pd.DataFrame
    families: pd.DataFrame
    frozen_thresholds: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class FrozenSliceValidationArtifact:
    manifest: dict
    summary: dict
    slices: pd.DataFrame
    families: pd.DataFrame


DEFAULT_FROZEN_SLICE_FAMILIES: dict[str, tuple[str, ...]] = {
    "stable_probability": (
        "tv_distance_qband=middle",
        "coverage_type=forecast_only&tv_distance_qband=middle",
    ),
    "high_disagreement_opportunity": (
        "tv_distance_qband=high",
        "coverage_type=forecast_only&tv_distance_qband=high",
        "tv_distance_qband=high&benchmark_probability_qband=high",
    ),
    "center_bucket_distribution": (
        "bucket_position=center",
        "token_side=YES&bucket_position=center",
        "token_side=NO&bucket_position=center",
    ),
}


def build_contiguous_event_day_folds(
    target_dates: pd.Series | np.ndarray | list[str],
    *,
    folds: int = 5,
) -> pd.DataFrame:
    """Partition ordered Event days into contiguous held-out blocks."""
    dates = sorted({str(value) for value in target_dates})
    if folds < 2 or folds > len(dates):
        raise ValueError("folds must be between 2 and the event-day count")
    blocks = np.array_split(np.asarray(dates, dtype=object), folds)
    rows = []
    date_set = set(dates)
    for fold, block in enumerate(blocks):
        held_out = tuple(str(value) for value in block.tolist())
        rows.append(
            {
                "fold": fold,
                "fit_dates": tuple(sorted(date_set - set(held_out))),
                "held_out_dates": held_out,
            }
        )
    return pd.DataFrame(rows)


def build_signal_slice_features(
    panel: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ex-ante vector features and one stable model-favored Token."""
    required_panel = {
        *_VECTOR_KEYS,
        "snapshot_id",
        "init_time",
        "available_at",
        "condition_id",
        "bucket_order",
        "model_probability",
        "midpoint_probability",
        "midpoint_overround",
        "yes_outcome",
        "lead_hours",
    }
    missing = required_panel - set(panel)
    if missing:
        raise ValueError(f"signal panel missing columns: {sorted(missing)}")
    snapshot_columns = (
        "snapshot_id",
        "coverage_type",
        "observed_hour_count",
        "forecast_hour_count",
    )
    missing = set(snapshot_columns) - set(snapshots)
    if missing:
        raise ValueError(
            f"Forecast snapshots missing columns: {sorted(missing)}"
        )
    snapshot_metadata = snapshots[list(snapshot_columns)].drop_duplicates(
        "snapshot_id"
    )
    rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    for key, raw_vector in panel.groupby(_VECTOR_KEYS, sort=True):
        vector = raw_vector.sort_values(
            ["bucket_order", "condition_id"], kind="stable"
        ).reset_index(drop=True)
        model = vector["model_probability"].astype(float).to_numpy()
        midpoint = vector["midpoint_probability"].astype(float).to_numpy()
        outcome = vector["yes_outcome"].astype(float).to_numpy()
        _validate_probability_vector(model, "model")
        _validate_probability_vector(midpoint, "midpoint")
        if not np.isclose(outcome.sum(), 1.0):
            raise ValueError("slice vector must contain exactly one true Bucket")
        bucket_count = len(vector)
        model_order = np.argsort(-model, kind="stable")
        midpoint_order = np.argsort(-midpoint, kind="stable")
        timestamp = pd.Timestamp(key[2])
        available_at = pd.Timestamp(vector["available_at"].iloc[0])
        init_time = pd.Timestamp(vector["init_time"].iloc[0])
        vector_row = {
            "event_id": str(key[0]),
            "target_date": str(key[1]),
            "timestamp": timestamp,
            "snapshot_id": str(vector["snapshot_id"].iloc[0]),
            "init_time": init_time,
            "available_at": available_at,
            "lead_hours": float(vector["lead_hours"].iloc[0]),
            "midpoint_overround": float(
                vector["midpoint_overround"].iloc[0]
            ),
            "model_max_probability": float(model[model_order[0]]),
            "midpoint_max_probability": float(
                midpoint[midpoint_order[0]]
            ),
            "model_normalized_entropy": _normalized_entropy(model),
            "midpoint_normalized_entropy": _normalized_entropy(midpoint),
            "model_top_two_margin": _top_two_margin(model, model_order),
            "midpoint_top_two_margin": _top_two_margin(
                midpoint, midpoint_order
            ),
            "tv_distance": float(0.5 * np.abs(model - midpoint).sum()),
            "top_bucket_agrees": bool(
                vector.loc[model_order[0], "condition_id"]
                == vector.loc[midpoint_order[0], "condition_id"]
            ),
            "max_absolute_probability_gap": float(
                np.abs(model - midpoint).max()
            ),
            "model_true_bucket_probability": float(model[np.argmax(outcome)]),
            "midpoint_true_bucket_probability": float(
                midpoint[np.argmax(outcome)]
            ),
            "true_bucket_probability_delta": float(
                model[np.argmax(outcome)] - midpoint[np.argmax(outcome)]
            ),
            "snapshot_age_hours": float(
                (timestamp - available_at).total_seconds() / 3600.0
            ),
            "init_age_hours": float(
                (timestamp - init_time).total_seconds() / 3600.0
            ),
            **_proper_scores(model, midpoint, outcome),
        }
        rows.append(vector_row)

        candidates: list[dict[str, object]] = []
        for index, bucket in vector.iterrows():
            yes_model = float(model[index])
            yes_midpoint = float(midpoint[index])
            yes_outcome = float(outcome[index])
            for side, token_model, token_midpoint, token_outcome in (
                ("YES", yes_model, yes_midpoint, yes_outcome),
                (
                    "NO",
                    1.0 - yes_model,
                    1.0 - yes_midpoint,
                    1.0 - yes_outcome,
                ),
            ):
                candidates.append(
                    {
                        **{
                            name: vector_row[name]
                            for name in (
                                "event_id",
                                "target_date",
                                "timestamp",
                                "snapshot_id",
                            )
                        },
                        "condition_id": str(bucket["condition_id"]),
                        "bucket_order": int(bucket["bucket_order"]),
                        "bucket_position": _bucket_position(
                            int(bucket["bucket_order"]), bucket_count
                        ),
                        "token_side": side,
                        "model_probability": token_model,
                        "benchmark_probability": token_midpoint,
                        "token_outcome": token_outcome,
                        "model_probability_advantage": (
                            token_model - token_midpoint
                        ),
                        "theoretical_midpoint_excess": (
                            token_outcome - token_midpoint
                        ),
                        "model_probability_residual": (
                            token_outcome - token_model
                        ),
                    }
                )
        chosen = sorted(
            candidates,
            key=lambda row: (
                -float(row["model_probability_advantage"]),
                str(row["condition_id"]),
                0 if row["token_side"] == "YES" else 1,
            ),
        )[0]
        token_rows.append(chosen)

    vectors = pd.DataFrame(rows).merge(
        snapshot_metadata,
        on="snapshot_id",
        how="left",
        validate="many_to_one",
    )
    if vectors[
        ["coverage_type", "observed_hour_count", "forecast_hour_count"]
    ].isna().any().any():
        raise ValueError("slice features reference an unknown Forecast snapshot")
    return vectors, pd.DataFrame(token_rows)


def evaluate_fixed_slice_atlas(
    vector_features: pd.DataFrame,
    token_candidates: pd.DataFrame,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate the declared slice family with Event-equal evidence."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    features = _assign_fixed_bands(vector_features, token_candidates)
    definitions = _fixed_slice_definitions()
    rows: list[dict[str, object]] = []
    contributions: dict[
        str, dict[str, pd.Series]
    ] = {}
    for definition_index, definition in enumerate(definitions):
        mask = pd.Series(True, index=features.index)
        for dimension, value in zip(
            definition.dimensions, definition.values
        ):
            mask &= features[dimension].astype(str) == value
        selected = features[mask].copy()
        base = {
            "slice_id": definition.slice_id,
            "dimensions": "|".join(definition.dimensions),
            "values": "|".join(definition.values),
            "event_count": int(selected["event_id"].nunique()),
            "event_day_count": int(selected["target_date"].nunique()),
            "vector_count": int(len(selected)),
            "first_signal_count": 0,
        }
        if selected.empty:
            rows.append({**base, **_empty_slice_metrics()})
            contributions[definition.slice_id] = {}
            continue
        event_scores = (
            selected.groupby(["event_id", "target_date"], as_index=False)[
                [
                    "model_multiclass_brier",
                    "midpoint_multiclass_brier",
                    "brier_delta",
                    "model_log_score",
                    "midpoint_log_score",
                    "log_score_delta",
                    "model_normalized_rps",
                    "midpoint_normalized_rps",
                    "rps_delta",
                ]
            ]
            .mean()
        )
        first = (
            selected.sort_values(
                ["timestamp", "condition_id", "token_side"],
                kind="stable",
            )
            .groupby("event_id", as_index=False)
            .first()
        )
        base["first_signal_count"] = int(len(first))
        metrics: dict[str, float] = {
            column: float(event_scores[column].mean())
            for column in (
                "model_multiclass_brier",
                "midpoint_multiclass_brier",
                "brier_delta",
                "model_log_score",
                "midpoint_log_score",
                "log_score_delta",
                "model_normalized_rps",
                "midpoint_normalized_rps",
                "rps_delta",
            )
        }
        for metric, column in (
            ("brier", "brier_delta"),
            ("log_score", "log_score_delta"),
            ("rps", "rps_delta"),
        ):
            day_values = event_scores.groupby("target_date")[column].mean()
            samples = _bootstrap_day_mean(
                day_values,
                bootstrap_samples=bootstrap_samples,
                seed=seed + definition_index * 11 + len(metric),
            )
            metrics[f"{metric}_delta_ci90_low"] = float(
                np.quantile(samples, 0.05)
            )
            metrics[f"{metric}_delta_ci90_high"] = float(
                np.quantile(samples, 0.95)
            )
            metrics[f"{metric}_probability_model_better"] = float(
                np.mean(samples < 0.0)
            )
        edge_by_day = first.groupby("target_date")[
            "theoretical_midpoint_excess"
        ].mean()
        edge_samples = _bootstrap_day_mean(
            edge_by_day,
            bootstrap_samples=bootstrap_samples,
            seed=seed + definition_index * 11 + 9,
        )
        metrics.update(
            {
                "theoretical_midpoint_excess": float(
                    first.groupby("event_id")[
                        "theoretical_midpoint_excess"
                    ].mean().mean()
                ),
                "theoretical_midpoint_excess_lcb90": float(
                    np.quantile(edge_samples, 0.10)
                ),
                "model_probability_residual": float(
                    first.groupby("event_id")[
                        "model_probability_residual"
                    ].mean().mean()
                ),
                "model_probability_advantage": float(
                    first.groupby("event_id")[
                        "model_probability_advantage"
                    ].mean().mean()
                ),
            }
        )
        rows.append({**base, **metrics})
        contributions[definition.slice_id] = {
            column: event_scores.groupby("target_date")[column].mean()
            for column in (
                "brier_delta",
                "log_score_delta",
                "rps_delta",
            )
        }
    atlas = pd.DataFrame(rows)
    for label, column in (
        ("brier", "brier_delta"),
        ("log_score", "log_score_delta"),
        ("rps", "rps_delta"),
    ):
        adjusted = _max_t_adjusted_p(
            {
                slice_id: values[column]
                for slice_id, values in contributions.items()
                if column in values
            },
            bootstrap_samples=bootstrap_samples,
            seed=seed + len(label) * 1_000,
        )
        atlas[f"{label}_max_t_adjusted_p"] = atlas["slice_id"].map(
            adjusted
        )
    return atlas


def run_nested_slice_search(
    vector_features: pd.DataFrame,
    token_candidates: pd.DataFrame,
    *,
    folds: int = 5,
    min_training_events: int = 12,
    minimum_shortlist_events: int = 15,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit quantile slice rules on complement days and score held-out days."""
    features = vector_features.merge(
        token_candidates,
        on=_VECTOR_KEYS + ["snapshot_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_token"),
    )
    fold_table = build_contiguous_event_day_folds(
        features["target_date"], folds=folds
    )
    crossfit_rows: list[dict[str, object]] = []
    fold_rule_rows: list[dict[str, object]] = []
    oof_by_rule: dict[str, list[pd.DataFrame]] = {}
    for fold_row in fold_table.itertuples(index=False):
        fit = features[
            features["target_date"].astype(str).isin(fold_row.fit_dates)
        ].copy()
        held_out = features[
            features["target_date"].astype(str).isin(
                fold_row.held_out_dates
            )
        ].copy()
        thresholds = _fit_nested_thresholds(fit)
        fit = _assign_nested_bands(fit, thresholds)
        held_out = _assign_nested_bands(held_out, thresholds)
        definitions = _nested_rule_definitions()
        eligible: list[dict[str, object]] = []
        for definition in definitions:
            fit_selected = _select_rule(fit, definition)
            training_event_count = int(
                fit_selected["event_id"].nunique()
            )
            if training_event_count < min_training_events:
                continue
            training = _slice_event_evidence(fit_selected)
            training_summary = _mean_evidence(training)
            training_score = float(
                -training_summary["brier_delta"]
                - training_summary["rps_delta"]
                + training_summary["theoretical_midpoint_excess"]
            )
            eligible.append(
                {
                    "definition": definition,
                    "training_event_count": training_event_count,
                    "training_score": training_score,
                    **{
                        f"training_{key}": value
                        for key, value in training_summary.items()
                    },
                }
            )
        favorable = sorted(
            (
                row
                for row in eligible
                if row["training_brier_delta"] < 0.0
                and row["training_rps_delta"] < 0.0
                and row["training_theoretical_midpoint_excess"] > 0.0
            ),
            key=lambda row: (
                -float(row["training_score"]),
                row["definition"].slice_id,
            ),
        )
        selected_ids = {
            row["definition"].slice_id for row in favorable[:3]
        }
        rank_by_id = {
            row["definition"].slice_id: rank + 1
            for rank, row in enumerate(favorable)
        }
        thresholds_json = json.dumps(
            {
                key: [float(value) for value in values]
                for key, values in thresholds.items()
            },
            sort_keys=True,
        )
        for candidate in eligible:
            definition = candidate.pop("definition")
            fold_rule_rows.append(
                {
                    "fold": int(fold_row.fold),
                    "slice_id": definition.slice_id,
                    "dimensions": "|".join(definition.dimensions),
                    "values": "|".join(definition.values),
                    "fit_dates": tuple(fold_row.fit_dates),
                    "held_out_dates": tuple(fold_row.held_out_dates),
                    "thresholds_json": thresholds_json,
                    "training_rank": rank_by_id.get(
                        definition.slice_id, 0
                    ),
                    "selected_top3": (
                        definition.slice_id in selected_ids
                    ),
                    **candidate,
                }
            )
            held_selected = _select_rule(held_out, definition)
            evidence = _slice_event_evidence(held_selected)
            summary = _mean_evidence(evidence)
            crossfit_rows.append(
                {
                    "fold": int(fold_row.fold),
                    "slice_id": definition.slice_id,
                    "held_out_event_count": int(len(evidence)),
                    "held_out_event_day_count": int(
                        evidence["target_date"].nunique()
                        if not evidence.empty
                        else 0
                    ),
                    "favorable_direction": bool(
                        summary["brier_delta"] < 0.0
                        and summary["rps_delta"] < 0.0
                        and summary["theoretical_midpoint_excess"] > 0.0
                    ),
                    **summary,
                }
            )
            if not evidence.empty:
                oof_by_rule.setdefault(definition.slice_id, []).append(
                    evidence
                )
    crossfit = pd.DataFrame(crossfit_rows)
    fold_rules = pd.DataFrame(fold_rule_rows)
    shortlist_rows: list[dict[str, object]] = []
    for rule_id, parts in sorted(oof_by_rule.items()):
        evidence = pd.concat(parts, ignore_index=True)
        summary = _mean_evidence(evidence)
        edge_by_day = evidence.groupby("target_date")[
            "theoretical_midpoint_excess"
        ].mean()
        edge_samples = _bootstrap_day_mean(
            edge_by_day,
            bootstrap_samples=bootstrap_samples,
            seed=seed + _stable_seed(rule_id),
        )
        rule_crossfit = crossfit[crossfit["slice_id"] == rule_id]
        event_count = int(evidence["event_id"].nunique())
        favorable_fold_count = int(
            rule_crossfit["favorable_direction"].sum()
        )
        edge_lcb = float(np.quantile(edge_samples, 0.10))
        selected_fold_count = int(
            fold_rules.loc[
                fold_rules["slice_id"] == rule_id, "selected_top3"
            ].sum()
        )
        dual_positive = bool(
            summary["brier_delta"] < 0.0
            and summary["rps_delta"] < 0.0
            and summary["theoretical_midpoint_excess"] > 0.0
        )
        if (
            event_count >= minimum_shortlist_events
            and favorable_fold_count >= 4
            and summary["brier_delta"] < 0.0
            and summary["rps_delta"] < 0.0
            and edge_lcb > 0.0
        ):
            tier = "stable_candidate"
        elif event_count >= minimum_shortlist_events and dual_positive:
            tier = "promising_candidate"
        else:
            tier = "exploratory_only"
        shortlist_rows.append(
            {
                "slice_id": rule_id,
                "evidence_tier": tier,
                "held_out_event_count": event_count,
                "held_out_event_day_count": int(
                    evidence["target_date"].nunique()
                ),
                "folds_with_support": int(
                    (rule_crossfit["held_out_event_count"] > 0).sum()
                ),
                "favorable_fold_count": favorable_fold_count,
                "selected_top3_fold_count": selected_fold_count,
                "theoretical_midpoint_excess_lcb90": edge_lcb,
                **summary,
            }
        )
    shortlist = pd.DataFrame(shortlist_rows)
    if not shortlist.empty:
        tier_order = {
            "stable_candidate": 0,
            "promising_candidate": 1,
            "exploratory_only": 2,
        }
        shortlist["_tier_order"] = shortlist["evidence_tier"].map(
            tier_order
        )
        shortlist = shortlist.sort_values(
            [
                "_tier_order",
                "selected_top3_fold_count",
                "held_out_event_count",
                "slice_id",
            ],
            ascending=[True, False, False, True],
        ).drop(columns="_tier_order").reset_index(drop=True)
    return crossfit, fold_rules, shortlist


def score_frozen_slice_validation(
    development_vector_features: pd.DataFrame,
    development_token_candidates: pd.DataFrame,
    validation_vector_features: pd.DataFrame,
    validation_token_candidates: pd.DataFrame,
    *,
    split_role: str,
    slice_families: dict[str, tuple[str, ...]]
    | None = None,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> FrozenSliceValidationResult:
    """Score frozen development slice families on an independent validation split.

    Development rows are used only to freeze qband thresholds. The validation
    rows are never used for threshold fitting or rule selection.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    families = slice_families or DEFAULT_FROZEN_SLICE_FAMILIES
    frozen_thresholds = _fit_nested_thresholds(
        development_vector_features.merge(
            development_token_candidates,
            on=_VECTOR_KEYS + ["snapshot_id"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_token"),
        )
    )
    validation_features = _assign_nested_bands(
        validation_vector_features.merge(
            validation_token_candidates,
            on=_VECTOR_KEYS + ["snapshot_id"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_token"),
        ),
        frozen_thresholds,
    )
    rule_by_id = {
        definition.slice_id: definition
        for definition in _nested_rule_definitions()
    }
    required_rule_ids = sorted(
        {rule_id for rule_ids in families.values() for rule_id in rule_ids}
    )
    missing = set(required_rule_ids) - set(rule_by_id)
    if missing:
        raise ValueError(f"unknown frozen slice ids: {sorted(missing)}")

    slice_rows = []
    masks: dict[str, pd.Series] = {}
    for rule_id in required_rule_ids:
        definition = rule_by_id[rule_id]
        mask = _slice_mask(validation_features, definition)
        masks[rule_id] = mask
        slice_rows.append(
            _score_frozen_selection(
                rule_id=rule_id,
                selected=validation_features[mask].copy(),
                bootstrap_samples=bootstrap_samples,
                seed=seed + _stable_seed(rule_id),
            )
        )

    family_rows = []
    for family_id, rule_ids in families.items():
        family_mask = pd.Series(False, index=validation_features.index)
        for rule_id in rule_ids:
            family_mask |= masks[rule_id]
        family_rows.append(
            {
                "family_id": family_id,
                "member_slice_ids": tuple(rule_ids),
                **_score_frozen_selection(
                    rule_id=family_id,
                    selected=validation_features[family_mask].copy(),
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + _stable_seed(family_id),
                ),
            }
        )

    slices = pd.DataFrame(slice_rows)
    family_frame = pd.DataFrame(family_rows)
    summary = {
        "schema_version": 1,
        "data_class": "frozen_forecast_distribution_validation_summary",
        "selection_policy": "frozen_no_search",
        "split_role": split_role,
        "development_event_count": int(
            development_vector_features["event_id"].nunique()
        ),
        "validation_event_count": int(
            validation_vector_features["event_id"].nunique()
        ),
        "frozen_slice_ids": required_rule_ids,
        "frozen_families": {
            family_id: list(rule_ids)
            for family_id, rule_ids in families.items()
        },
        "frozen_thresholds": {
            key: [float(value) for value in values]
            for key, values in frozen_thresholds.items()
        },
        "parameters": {
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        },
    }
    return FrozenSliceValidationResult(
        summary=summary,
        slices=slices,
        families=family_frame,
        frozen_thresholds=frozen_thresholds,
    )


def _slice_mask(
    features: pd.DataFrame,
    definition: SliceDefinition,
) -> pd.Series:
    mask = pd.Series(True, index=features.index)
    for dimension, value in zip(
        definition.dimensions, definition.values
    ):
        mask &= features[dimension].astype(str) == value
    return mask


def _score_frozen_selection(
    *,
    rule_id: str,
    selected: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    evidence = _slice_event_evidence(selected)
    summary = _mean_evidence(evidence)
    if evidence.empty:
        edge_lcb = float("nan")
    else:
        edge_by_day = evidence.groupby("target_date")[
            "theoretical_midpoint_excess"
        ].mean()
        edge_lcb = float(
            np.quantile(
                _bootstrap_day_mean(
                    edge_by_day,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                ),
                0.10,
            )
        )
    return {
        "slice_id": rule_id,
        "event_count": int(evidence["event_id"].nunique()),
        "event_day_count": int(
            evidence["target_date"].nunique()
            if not evidence.empty
            else 0
        ),
        "vector_count": int(len(selected)),
        "theoretical_midpoint_excess_lcb90": edge_lcb,
        **summary,
    }


def run_forecast_signal_slice_study(
    *,
    signal_root: Path | str,
    forecast_root: Path | str,
    output: Path | str,
    folds: int = 5,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    min_training_events: int = 12,
    minimum_shortlist_events: int = 15,
) -> dict:
    """Run one immutable, development-only adaptive signal slice study."""
    signal_dir = Path(signal_root)
    forecast_dir = Path(forecast_root)
    parent = load_signal_study_artifact(signal_dir)
    if parent.manifest.get("split_role") != "oof_development":
        raise ValueError(
            "ForecastSignalSliceStudy accepts only oof_development"
        )
    forecast_manifest_path = forecast_dir / "manifest.json"
    forecast_manifest_sha256 = _file_sha256(forecast_manifest_path)
    if (
        parent.manifest.get("input_sha256", {}).get("forecast_manifest")
        != forecast_manifest_sha256
    ):
        raise ValueError(
            "slice ForecastCache does not match the parent Signal Study"
        )
    forecast = ForecastCache.load(forecast_dir)
    split_roles = set(forecast.lineages["split_role"].astype(str))
    if split_roles != {"oof_development"}:
        raise ValueError(
            "slice ForecastCache must contain only oof_development"
        )
    vectors, tokens = build_signal_slice_features(
        parent.panel, forecast.snapshots
    )
    atlas = evaluate_fixed_slice_atlas(
        vectors,
        tokens,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    crossfit, fold_rules, shortlist = run_nested_slice_search(
        vectors,
        tokens,
        folds=folds,
        min_training_events=min_training_events,
        minimum_shortlist_events=minimum_shortlist_events,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    tier_counts = (
        shortlist["evidence_tier"].value_counts().to_dict()
        if not shortlist.empty
        else {}
    )
    summary = {
        "schema_version": 1,
        "data_class": "forecast_signal_slice_study_summary",
        "development_evidence_only": True,
        "non_executable": True,
        "sample": {
            "event_count": int(vectors["event_id"].nunique()),
            "event_day_count": int(vectors["target_date"].nunique()),
            "vector_count": int(len(vectors)),
            "token_candidate_count": int(len(tokens)),
            "fixed_slice_count": int(len(atlas)),
            "crossfit_rule_count": int(
                crossfit["slice_id"].nunique()
                if not crossfit.empty
                else 0
            ),
        },
        "shortlist": {
            "tier_counts": {
                tier: int(tier_counts.get(tier, 0))
                for tier in (
                    "stable_candidate",
                    "promising_candidate",
                    "exploratory_only",
                )
            },
            "stable_slice_ids": shortlist.loc[
                shortlist["evidence_tier"] == "stable_candidate",
                "slice_id",
            ].tolist()
            if not shortlist.empty
            else [],
            "promising_slice_ids": shortlist.loc[
                shortlist["evidence_tier"] == "promising_candidate",
                "slice_id",
            ].tolist()
            if not shortlist.empty
            else [],
        },
        "coverage_exclusions": parent.summary.get(
            "coverage_exclusions", []
        ),
        "parameters": {
            "folds": folds,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "min_training_events": min_training_events,
            "minimum_shortlist_events": minimum_shortlist_events,
        },
    }
    return save_signal_slice_artifact(
        output,
        parent_manifest_sha256=_file_sha256(
            signal_dir / "manifest.json"
        ),
        forecast_manifest_sha256=forecast_manifest_sha256,
        parent_study_id=str(parent.manifest["study_id"]),
        station_id=str(parent.manifest["station_id"]),
        target_range=list(parent.manifest["target_range"]),
        summary=summary,
        vector_features=vectors,
        token_candidates=tokens,
        slice_atlas=atlas,
        crossfit_results=crossfit,
        fold_rules=fold_rules,
        shortlist=shortlist,
    )


def run_frozen_forecast_distribution_validation(
    *,
    development_slice_root: Path | str,
    validation_signal_root: Path | str,
    validation_forecast_root: Path | str,
    output: Path | str,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Apply frozen development slice families to a validation signal study."""
    development = load_signal_slice_artifact(development_slice_root)
    signal_dir = Path(validation_signal_root)
    forecast_dir = Path(validation_forecast_root)
    validation_signal = load_signal_study_artifact(signal_dir)
    forecast_manifest_sha256 = _file_sha256(forecast_dir / "manifest.json")
    if (
        validation_signal.manifest.get("input_sha256", {}).get(
            "forecast_manifest"
        )
        != forecast_manifest_sha256
    ):
        raise ValueError(
            "validation ForecastCache does not match the validation Signal Study"
        )
    forecast = ForecastCache.load(forecast_dir)
    split_roles = set(forecast.lineages["split_role"].astype(str))
    if len(split_roles) != 1:
        raise ValueError("validation ForecastCache must contain one split_role")
    split_role = next(iter(split_roles))
    if validation_signal.manifest.get("split_role") != split_role:
        raise ValueError(
            "validation Signal Study split_role does not match ForecastCache"
        )
    vectors, tokens = build_signal_slice_features(
        validation_signal.panel, forecast.snapshots
    )
    result = score_frozen_slice_validation(
        development.vector_features,
        development.token_candidates,
        vectors,
        tokens,
        split_role=split_role,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return save_frozen_slice_validation_artifact(
        output,
        development_slice_manifest_sha256=_file_sha256(
            Path(development_slice_root) / "manifest.json"
        ),
        validation_signal_manifest_sha256=_file_sha256(
            signal_dir / "manifest.json"
        ),
        validation_forecast_manifest_sha256=forecast_manifest_sha256,
        station_id=str(validation_signal.manifest["station_id"]),
        target_range=list(validation_signal.manifest["target_range"]),
        split_role=split_role,
        result=result,
    )


def save_signal_slice_artifact(
    root: Path | str,
    *,
    parent_manifest_sha256: str,
    forecast_manifest_sha256: str,
    parent_study_id: str,
    station_id: str,
    target_range: list[str],
    summary: dict,
    vector_features: pd.DataFrame,
    token_candidates: pd.DataFrame,
    slice_atlas: pd.DataFrame,
    crossfit_results: pd.DataFrame,
    fold_rules: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> dict:
    artifact_dir = Path(root)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    frames = {
        "vector_features.parquet": vector_features,
        "token_candidates.parquet": token_candidates,
        "slice_atlas.parquet": slice_atlas,
        "crossfit_results.parquet": crossfit_results,
        "fold_rules.parquet": fold_rules,
        "shortlist.parquet": shortlist,
    }
    hashes: dict[str, str] = {}
    for filename, frame in frames.items():
        path = artifact_dir / filename
        frame.to_parquet(path, index=False)
        hashes[filename] = _file_sha256(path)
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    hashes[summary_path.name] = _file_sha256(summary_path)
    identity = {
        "parent_manifest_sha256": parent_manifest_sha256,
        "forecast_manifest_sha256": forecast_manifest_sha256,
        "parameters": summary["parameters"],
    }
    study_id = "signal-slices-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    manifest = {
        "schema_version": 1,
        "data_class": "forecast_signal_slice_study",
        "study_id": study_id,
        "parent_study_id": parent_study_id,
        "parent_signal_manifest_sha256": parent_manifest_sha256,
        "forecast_manifest_sha256": forecast_manifest_sha256,
        "station_id": station_id,
        "target_range": target_range,
        "split_role": "oof_development",
        "parameters": summary["parameters"],
        "fixed_slice_family": [
            definition.slice_id for definition in _fixed_slice_definitions()
        ],
        "search_feature_whitelist": [
            "lead_hours",
            "coverage_type",
            "model_max_probability",
            "model_normalized_entropy",
            "tv_distance",
            "top_bucket_agrees",
            "benchmark_probability",
            "token_side",
            "bucket_position",
        ],
        "max_rule_depth": 2,
        "sha256": hashes,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_signal_slice_artifact(
    root: Path | str,
) -> ForecastSignalSliceArtifact:
    artifact_dir = Path(root)
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("data_class")
        != "forecast_signal_slice_study"
    ):
        raise ValueError("unsupported Forecast Signal Slice Study manifest")
    expected = {
        "vector_features.parquet",
        "token_candidates.parquet",
        "slice_atlas.parquet",
        "crossfit_results.parquet",
        "fold_rules.parquet",
        "shortlist.parquet",
        "summary.json",
    }
    if set(manifest.get("sha256", {})) != expected:
        raise ValueError("slice study manifest has incomplete payloads")
    for filename, digest in manifest["sha256"].items():
        if _file_sha256(artifact_dir / filename) != digest:
            raise ValueError(f"slice study checksum mismatch: {filename}")
    return ForecastSignalSliceArtifact(
        manifest=manifest,
        summary=json.loads(
            (artifact_dir / "summary.json").read_text(encoding="utf-8")
        ),
        vector_features=pd.read_parquet(
            artifact_dir / "vector_features.parquet"
        ),
        token_candidates=pd.read_parquet(
            artifact_dir / "token_candidates.parquet"
        ),
        slice_atlas=pd.read_parquet(
            artifact_dir / "slice_atlas.parquet"
        ),
        crossfit_results=pd.read_parquet(
            artifact_dir / "crossfit_results.parquet"
        ),
        fold_rules=pd.read_parquet(artifact_dir / "fold_rules.parquet"),
        shortlist=pd.read_parquet(artifact_dir / "shortlist.parquet"),
    )


def save_frozen_slice_validation_artifact(
    root: Path | str,
    *,
    development_slice_manifest_sha256: str,
    validation_signal_manifest_sha256: str,
    validation_forecast_manifest_sha256: str,
    station_id: str,
    target_range: list[str],
    split_role: str,
    result: FrozenSliceValidationResult,
) -> dict:
    artifact_dir = Path(root)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    frames = {
        "frozen_slices.parquet": result.slices,
        "frozen_families.parquet": result.families,
    }
    hashes: dict[str, str] = {}
    for filename, frame in frames.items():
        path = artifact_dir / filename
        frame.to_parquet(path, index=False)
        hashes[filename] = _file_sha256(path)
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            result.summary,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    hashes[summary_path.name] = _file_sha256(summary_path)
    identity = {
        "development_slice_manifest_sha256": (
            development_slice_manifest_sha256
        ),
        "validation_signal_manifest_sha256": (
            validation_signal_manifest_sha256
        ),
        "validation_forecast_manifest_sha256": (
            validation_forecast_manifest_sha256
        ),
        "parameters": result.summary["parameters"],
        "split_role": split_role,
    }
    study_id = "frozen-signal-validation-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    manifest = {
        "schema_version": 1,
        "data_class": "frozen_forecast_distribution_validation",
        "study_id": study_id,
        "development_slice_manifest_sha256": (
            development_slice_manifest_sha256
        ),
        "validation_signal_manifest_sha256": (
            validation_signal_manifest_sha256
        ),
        "validation_forecast_manifest_sha256": (
            validation_forecast_manifest_sha256
        ),
        "station_id": station_id,
        "target_range": target_range,
        "split_role": split_role,
        "selection_policy": "frozen_no_search",
        "parameters": result.summary["parameters"],
        "sha256": hashes,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_frozen_slice_validation_artifact(
    root: Path | str,
) -> FrozenSliceValidationArtifact:
    artifact_dir = Path(root)
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("data_class")
        != "frozen_forecast_distribution_validation"
    ):
        raise ValueError(
            "unsupported frozen forecast distribution validation manifest"
        )
    expected = {
        "summary.json",
        "frozen_slices.parquet",
        "frozen_families.parquet",
    }
    if set(manifest.get("sha256", {})) != expected:
        raise ValueError(
            "frozen validation manifest has incomplete payloads"
        )
    for filename, digest in manifest["sha256"].items():
        if _file_sha256(artifact_dir / filename) != digest:
            raise ValueError(f"frozen validation checksum mismatch: {filename}")
    return FrozenSliceValidationArtifact(
        manifest=manifest,
        summary=json.loads(
            (artifact_dir / "summary.json").read_text(encoding="utf-8")
        ),
        slices=pd.read_parquet(artifact_dir / "frozen_slices.parquet"),
        families=pd.read_parquet(artifact_dir / "frozen_families.parquet"),
    )


def _fit_nested_thresholds(
    fit: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    continuous = (
        "lead_hours",
        "model_max_probability",
        "model_normalized_entropy",
        "tv_distance",
        "benchmark_probability",
    )
    event_equal = fit.groupby("event_id")[list(continuous)].mean()
    return {
        column: tuple(
            float(value)
            for value in event_equal[column].quantile([1 / 3, 2 / 3])
        )
        for column in continuous
    }


def _assign_nested_bands(
    features: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    work = features.copy()
    for column, (lower, upper) in thresholds.items():
        work[f"{column}_qband"] = np.select(
            [work[column] < lower, work[column] < upper],
            ["low", "middle"],
            default="high",
        )
    work["top_agreement"] = np.where(
        work["top_bucket_agrees"], "agree", "disagree"
    )
    return work


def _nested_rule_definitions() -> list[SliceDefinition]:
    dimensions = {
        "lead_hours_qband": ("low", "middle", "high"),
        "coverage_type": ("forecast_only", "observed_plus_forecast"),
        "model_max_probability_qband": ("low", "middle", "high"),
        "model_normalized_entropy_qband": ("low", "middle", "high"),
        "tv_distance_qband": ("low", "middle", "high"),
        "top_agreement": ("agree", "disagree"),
        "benchmark_probability_qband": ("low", "middle", "high"),
        "token_side": ("YES", "NO"),
        "bucket_position": ("lower_tail", "center", "upper_tail"),
    }
    definitions = [
        SliceDefinition(
            slice_id=f"{dimension}={value}",
            dimensions=(dimension,),
            values=(value,),
        )
        for dimension, values in dimensions.items()
        for value in values
    ]
    pairs = (
        ("lead_hours_qband", "tv_distance_qband"),
        ("lead_hours_qband", "top_agreement"),
        ("coverage_type", "tv_distance_qband"),
        ("coverage_type", "top_agreement"),
        (
            "model_max_probability_qband",
            "benchmark_probability_qband",
        ),
        (
            "model_normalized_entropy_qband",
            "benchmark_probability_qband",
        ),
        ("tv_distance_qband", "benchmark_probability_qband"),
        ("top_agreement", "benchmark_probability_qband"),
        ("token_side", "bucket_position"),
    )
    for left, right in pairs:
        for left_value, right_value in itertools.product(
            dimensions[left], dimensions[right]
        ):
            definitions.append(
                SliceDefinition(
                    slice_id=(
                        f"{left}={left_value}&{right}={right_value}"
                    ),
                    dimensions=(left, right),
                    values=(left_value, right_value),
                )
            )
    return definitions


def _select_rule(
    features: pd.DataFrame,
    definition: SliceDefinition,
) -> pd.DataFrame:
    mask = pd.Series(True, index=features.index)
    for dimension, value in zip(
        definition.dimensions, definition.values
    ):
        mask &= features[dimension].astype(str) == value
    return features[mask].copy()


def _slice_event_evidence(selected: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id",
        "target_date",
        "brier_delta",
        "log_score_delta",
        "rps_delta",
        "true_bucket_probability_delta",
        "theoretical_midpoint_excess",
        "model_probability_residual",
        "model_probability_advantage",
    ]
    if selected.empty:
        return pd.DataFrame(columns=columns)
    distribution_columns = [
        "brier_delta",
        "log_score_delta",
        "rps_delta",
    ]
    if "true_bucket_probability_delta" in selected:
        distribution_columns.append("true_bucket_probability_delta")
    distribution = (
        selected.groupby(["event_id", "target_date"], as_index=False)[
            distribution_columns
        ]
        .mean()
    )
    if "true_bucket_probability_delta" not in distribution:
        distribution["true_bucket_probability_delta"] = float("nan")
    first = (
        selected.sort_values(
            ["timestamp", "condition_id", "token_side"], kind="stable"
        )
        .groupby(["event_id", "target_date"], as_index=False)
        .first()
    )
    return distribution.merge(
        first[
            [
                "event_id",
                "target_date",
                "theoretical_midpoint_excess",
                "model_probability_residual",
                "model_probability_advantage",
            ]
        ],
        on=["event_id", "target_date"],
        how="inner",
        validate="one_to_one",
    )


def _mean_evidence(evidence: pd.DataFrame) -> dict[str, float]:
    metric_columns = (
        "brier_delta",
        "log_score_delta",
        "rps_delta",
        "true_bucket_probability_delta",
        "theoretical_midpoint_excess",
        "model_probability_residual",
        "model_probability_advantage",
    )
    if evidence.empty:
        return {column: float("nan") for column in metric_columns}
    return {
        column: float(evidence.groupby("event_id")[column].mean().mean())
        for column in metric_columns
    }


def _stable_seed(value: str) -> int:
    return sum(
        (index + 1) * ord(character)
        for index, character in enumerate(value)
    )


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _assign_fixed_bands(
    vector_features: pd.DataFrame,
    token_candidates: pd.DataFrame,
) -> pd.DataFrame:
    features = vector_features.merge(
        token_candidates,
        on=_VECTOR_KEYS + ["snapshot_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_token"),
    )
    features["lead_band"] = pd.cut(
        features["lead_hours"],
        [-np.inf, 6.0, 12.0, 24.0, 36.0, np.inf],
        right=False,
        labels=["<6h", "6-12h", "12-24h", "24-36h", ">=36h"],
    )
    features["model_max_band"] = pd.cut(
        features["model_max_probability"],
        [-np.inf, 0.35, 0.45, np.inf],
        right=False,
        labels=["<0.35", "0.35-0.45", ">=0.45"],
    )
    features["entropy_band"] = pd.cut(
        features["model_normalized_entropy"],
        [-np.inf, 0.50, 0.65, np.inf],
        right=False,
        labels=["<0.50", "0.50-0.65", ">=0.65"],
    )
    features["tv_band"] = pd.cut(
        features["tv_distance"],
        [-np.inf, 0.15, 0.30, np.inf],
        right=False,
        labels=["<0.15", "0.15-0.30", ">=0.30"],
    )
    features["top_agreement"] = np.where(
        features["top_bucket_agrees"], "agree", "disagree"
    )
    features["benchmark_band"] = pd.cut(
        features["benchmark_probability"],
        [-np.inf, 0.10, 0.25, 0.50, 0.75, np.inf],
        right=False,
        labels=[
            "<0.10",
            "0.10-0.25",
            "0.25-0.50",
            "0.50-0.75",
            ">=0.75",
        ],
    )
    return features


def _fixed_slice_definitions() -> list[SliceDefinition]:
    dimensions = {
        "lead_band": ("<6h", "6-12h", "12-24h", "24-36h", ">=36h"),
        "coverage_type": ("forecast_only", "observed_plus_forecast"),
        "model_max_band": ("<0.35", "0.35-0.45", ">=0.45"),
        "entropy_band": ("<0.50", "0.50-0.65", ">=0.65"),
        "tv_band": ("<0.15", "0.15-0.30", ">=0.30"),
        "top_agreement": ("agree", "disagree"),
        "benchmark_band": (
            "<0.10",
            "0.10-0.25",
            "0.25-0.50",
            "0.50-0.75",
            ">=0.75",
        ),
        "token_side": ("YES", "NO"),
        "bucket_position": ("lower_tail", "center", "upper_tail"),
    }
    definitions: list[SliceDefinition] = []
    for dimension, values in dimensions.items():
        for value in values:
            definitions.append(
                SliceDefinition(
                    slice_id=f"{dimension}={value}",
                    dimensions=(dimension,),
                    values=(value,),
                )
            )
    pairs = (
        ("lead_band", "tv_band"),
        ("lead_band", "top_agreement"),
        ("coverage_type", "tv_band"),
        ("coverage_type", "top_agreement"),
        ("model_max_band", "benchmark_band"),
        ("entropy_band", "benchmark_band"),
        ("tv_band", "benchmark_band"),
        ("top_agreement", "benchmark_band"),
        ("token_side", "bucket_position"),
    )
    for left, right in pairs:
        for left_value, right_value in itertools.product(
            dimensions[left], dimensions[right]
        ):
            definitions.append(
                SliceDefinition(
                    slice_id=(
                        f"{left}={left_value}&{right}={right_value}"
                    ),
                    dimensions=(left, right),
                    values=(left_value, right_value),
                )
            )
    return definitions


def _bootstrap_day_mean(
    values: pd.Series,
    *,
    bootstrap_samples: int,
    seed: int,
) -> np.ndarray:
    array = values.astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    indexes = rng.integers(
        0, len(array), size=(bootstrap_samples, len(array))
    )
    return array[indexes].mean(axis=1)


def _max_t_adjusted_p(
    contributions: dict[str, pd.Series],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    valid = {
        key: values.astype(float)
        for key, values in contributions.items()
        if len(values) >= 2 and float(values.std(ddof=1)) > 0.0
    }
    result = {key: float("nan") for key in contributions}
    if not valid:
        return result
    keys = list(valid)
    dates = sorted(
        {
            str(target_date)
            for values in valid.values()
            for target_date in values.index
        }
    )
    matrix = np.full((len(dates), len(keys)), np.nan, dtype=float)
    standard_error = np.empty(len(keys), dtype=float)
    observed = np.empty(len(keys), dtype=float)
    date_position = {target_date: index for index, target_date in enumerate(dates)}
    for column, key in enumerate(keys):
        values = valid[key]
        array = values.to_numpy()
        standard_error[column] = float(
            array.std(ddof=1) / math.sqrt(len(array))
        )
        observed[column] = float(-array.mean() / standard_error[column])
        for target_date, value in values.items():
            matrix[date_position[str(target_date)], column] = (
                float(value) - float(array.mean())
            )
    rng = np.random.default_rng(seed)
    maxima = np.empty(bootstrap_samples)
    for sample_index in range(bootstrap_samples):
        indexes = rng.integers(0, len(dates), size=len(dates))
        sampled = matrix[indexes]
        counts = np.sum(~np.isnan(sampled), axis=0)
        bootstrap_mean = np.divide(
            np.nansum(sampled, axis=0),
            counts,
            out=np.full(len(keys), np.nan),
            where=counts > 0,
        )
        statistics = -bootstrap_mean / standard_error
        maxima[sample_index] = float(np.nanmax(statistics))
    for key, statistic in zip(keys, observed):
        result[key] = float(np.mean(maxima >= statistic))
    return result


def _empty_slice_metrics() -> dict[str, float]:
    columns = (
        "model_multiclass_brier",
        "midpoint_multiclass_brier",
        "brier_delta",
        "model_log_score",
        "midpoint_log_score",
        "log_score_delta",
        "model_normalized_rps",
        "midpoint_normalized_rps",
        "rps_delta",
        "brier_delta_ci90_low",
        "brier_delta_ci90_high",
        "brier_probability_model_better",
        "log_score_delta_ci90_low",
        "log_score_delta_ci90_high",
        "log_score_probability_model_better",
        "rps_delta_ci90_low",
        "rps_delta_ci90_high",
        "rps_probability_model_better",
        "theoretical_midpoint_excess",
        "theoretical_midpoint_excess_lcb90",
        "model_probability_residual",
        "model_probability_advantage",
    )
    return {column: float("nan") for column in columns}


def _validate_probability_vector(
    probability: np.ndarray, label: str
) -> None:
    if (
        not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or not np.isclose(probability.sum(), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{label} probability vector must sum to 1")


def _normalized_entropy(probability: np.ndarray) -> float:
    if len(probability) < 2:
        return 0.0
    positive = probability[probability > 0.0]
    return float(
        -(positive * np.log(positive)).sum() / math.log(len(probability))
    )


def _top_two_margin(
    probability: np.ndarray, order: np.ndarray
) -> float:
    if len(probability) < 2:
        return float(probability[order[0]])
    return float(probability[order[0]] - probability[order[1]])


def _proper_scores(
    model: np.ndarray,
    midpoint: np.ndarray,
    outcome: np.ndarray,
) -> dict[str, float]:
    truth = int(np.argmax(outcome))
    floor = 1e-6
    clipped_model = np.maximum(model, floor)
    clipped_model /= clipped_model.sum()
    clipped_midpoint = np.maximum(midpoint, floor)
    clipped_midpoint /= clipped_midpoint.sum()
    model_brier = float(np.square(clipped_model - outcome).sum())
    midpoint_brier = float(np.square(clipped_midpoint - outcome).sum())
    denominator = max(len(model) - 1, 1)
    model_rps = float(
        np.square(
            np.cumsum(clipped_model)[:-1] - np.cumsum(outcome)[:-1]
        ).sum()
        / denominator
    )
    midpoint_rps = float(
        np.square(
            np.cumsum(clipped_midpoint)[:-1]
            - np.cumsum(outcome)[:-1]
        ).sum()
        / denominator
    )
    model_log = float(-np.log(clipped_model[truth]))
    midpoint_log = float(-np.log(clipped_midpoint[truth]))
    return {
        "model_multiclass_brier": model_brier,
        "midpoint_multiclass_brier": midpoint_brier,
        "brier_delta": model_brier - midpoint_brier,
        "model_log_score": model_log,
        "midpoint_log_score": midpoint_log,
        "log_score_delta": model_log - midpoint_log,
        "model_normalized_rps": model_rps,
        "midpoint_normalized_rps": midpoint_rps,
        "rps_delta": model_rps - midpoint_rps,
    }


def _bucket_position(order: int, bucket_count: int) -> str:
    if order <= 2:
        return "lower_tail"
    if order >= bucket_count - 3:
        return "upper_tail"
    return "center"
