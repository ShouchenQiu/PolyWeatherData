"""Strategy-free forecast signal evaluation against synthetic midpoints."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.study import StudyValidation
from config.stations import STATIONS


_PANEL_COLUMNS = [
    "event_id",
    "target_date",
    "timestamp",
    "snapshot_id",
    "init_time",
    "available_at",
    "condition_id",
    "bucket_order",
    "bucket_lo_f",
    "bucket_hi_f",
    "model_probability",
    "midpoint_raw",
    "midpoint_probability",
    "midpoint_sum",
    "midpoint_overround",
    "yes_outcome",
    "actual_high_f",
    "lead_hours",
    "lead_band",
]

_LOSS_METRICS = (
    "multiclass_brier",
    "log_score",
    "normalized_rps",
    "pooled_binary_brier",
)
_SUMMARY_METRICS = (
    *_LOSS_METRICS,
    "normalized_entropy",
    "max_probability",
    "top1_accuracy",
    "true_bucket_probability",
)
_SOURCE_COLUMNS = {
    "model_raw": "model_probability",
    "synthetic_midpoint": "midpoint_probability",
}


@dataclass(frozen=True)
class SignalStudyReport:
    scores_by_tick: pd.DataFrame
    by_event: pd.DataFrame
    by_lead: pd.DataFrame
    reliability: pd.DataFrame
    summary: dict


@dataclass(frozen=True)
class ForecastSignalArtifact:
    manifest: dict
    summary: dict
    panel: pd.DataFrame
    scores_by_tick: pd.DataFrame
    by_event: pd.DataFrame
    by_lead: pd.DataFrame
    reliability: pd.DataFrame


def build_signal_panel(validation: StudyValidation) -> pd.DataFrame:
    """Build matched, complete probability vectors without strategy inputs."""
    events = validation.clob.events.copy()
    events["target_date"] = events["target_date"].astype(str)
    events = events[
        events["target_date"].between(
            validation.start_target, validation.end_target
        )
    ].copy()
    events = events[
        ~events["target_date"].isin(validation.excluded_target_dates)
    ].copy()
    event_ids = set(events["event_id"].astype(str))
    markets = validation.clob.markets[
        validation.clob.markets["event_id"].astype(str).isin(event_ids)
    ].copy()
    markets["event_id"] = markets["event_id"].astype(str)
    markets["condition_id"] = markets["condition_id"].astype(str)
    event_close_columns = [
        column
        for column in ("closed_at", "market_end_at", "event_end_at")
        if column in events
    ]
    markets = markets.merge(
        events[["event_id", "target_date", *event_close_columns]],
        on="event_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_event"),
    )
    markets["market_close_at"] = _market_close(markets)
    markets["settlement_at"] = markets["target_date"].map(
        lambda value: _settlement_at(value, validation.station_id)
    )

    quotes = validation.quotes.quotes.copy()
    quotes["condition_id"] = quotes["condition_id"].astype(str)
    quotes["timestamp"] = pd.to_datetime(quotes["timestamp"], utc=True)
    quotes = quotes.merge(
        markets[
            [
                "event_id",
                "target_date",
                "condition_id",
                "bucket_lo_f",
                "bucket_hi_f",
                "market_close_at",
                "settlement_at",
            ]
        ],
        on="condition_id",
        how="inner",
        validate="many_to_one",
    )
    quotes = quotes[
        (quotes["timestamp"] < quotes["market_close_at"])
        & (quotes["timestamp"] < quotes["settlement_at"])
    ].copy()
    quotes["mid_price_yes"] = pd.to_numeric(
        quotes["mid_price_yes"], errors="raise"
    )
    if quotes.duplicated(["event_id", "timestamp", "condition_id"]).any():
        raise ValueError("signal quotes contain duplicate Event/tick/Bucket rows")

    expected = markets.groupby("event_id")["condition_id"].nunique()
    complete_counts = quotes.groupby(
        ["event_id", "timestamp"]
    )["condition_id"].nunique()
    complete_keys = complete_counts[
        complete_counts
        == complete_counts.index.get_level_values("event_id").map(expected)
    ].index
    if complete_keys.empty:
        return pd.DataFrame(columns=_PANEL_COLUMNS)
    quotes = quotes.set_index(["event_id", "timestamp"]).loc[
        complete_keys
    ].reset_index()

    visible = _join_latest_visible_forecast(
        quotes=quotes,
        validation=validation,
        condition_ids=set(markets["condition_id"]),
    )
    if visible.empty:
        return pd.DataFrame(columns=_PANEL_COLUMNS)

    vector_keys = ["event_id", "timestamp"]
    joined_counts = visible.groupby(vector_keys)["condition_id"].nunique()
    joined_snapshots = visible.groupby(vector_keys)["snapshot_id"].nunique()
    cross_snapshot = joined_snapshots[joined_snapshots > 1]
    if not cross_snapshot.empty:
        raise ValueError("signal vector would combine multiple Forecast snapshots")
    valid_keys = joined_counts[
        (
            joined_counts
            == joined_counts.index.get_level_values("event_id").map(expected)
        )
        & (joined_snapshots == 1)
    ].index
    visible = visible.set_index(vector_keys).loc[valid_keys].reset_index()

    actuals = validation.actuals.set_index("target_date")[
        "actual_high_f"
    ].astype(float)
    rows: list[dict[str, object]] = []
    for (event_id, timestamp), vector in visible.groupby(
        vector_keys, sort=True
    ):
        vector = vector.sort_values(
            ["bucket_lo_f", "bucket_hi_f", "condition_id"],
            kind="stable",
        ).reset_index(drop=True)
        model = vector["yes_probability"].astype(float).to_numpy()
        if (
            not np.isfinite(model).all()
            or (model < 0.0).any()
            or not np.isclose(model.sum(), 1.0, atol=1e-6)
        ):
            raise ValueError(
                f"model probability vector must sum to 1 for {event_id} at {timestamp}"
            )
        midpoint = vector["mid_price_yes"].astype(float).to_numpy()
        midpoint_sum = float(midpoint.sum())
        if (
            not np.isfinite(midpoint).all()
            or (midpoint < 0.0).any()
            or (midpoint > 1.0).any()
            or midpoint_sum <= 0.0
        ):
            raise ValueError("synthetic midpoint vector must be finite and positive")
        midpoint_probability = midpoint / midpoint_sum
        target_date = str(vector["target_date"].iloc[0])
        actual = float(actuals.loc[target_date])
        outcome = (
            (actual >= vector["bucket_lo_f"].astype(float))
            & (actual < vector["bucket_hi_f"].astype(float))
        ).astype(float)
        if int(outcome.sum()) != 1:
            raise ValueError(
                f"actual high does not uniquely match a Bucket for {event_id}"
            )
        settlement_at = vector["settlement_at"].iloc[0]
        lead_hours = (
            pd.Timestamp(settlement_at) - pd.Timestamp(timestamp)
        ).total_seconds() / 3600.0
        lead_band = _lead_band(lead_hours)
        for order, row in vector.iterrows():
            rows.append(
                {
                    "event_id": str(event_id),
                    "target_date": target_date,
                    "timestamp": pd.Timestamp(timestamp),
                    "snapshot_id": str(row["snapshot_id"]),
                    "init_time": pd.Timestamp(row["init_time"]),
                    "available_at": pd.Timestamp(row["available_at"]),
                    "condition_id": str(row["condition_id"]),
                    "bucket_order": int(order),
                    "bucket_lo_f": float(row["bucket_lo_f"]),
                    "bucket_hi_f": float(row["bucket_hi_f"]),
                    "model_probability": float(model[order]),
                    "midpoint_raw": float(midpoint[order]),
                    "midpoint_probability": float(midpoint_probability[order]),
                    "midpoint_sum": midpoint_sum,
                    "midpoint_overround": midpoint_sum - 1.0,
                    "yes_outcome": float(outcome.iloc[order]),
                    "actual_high_f": actual,
                    "lead_hours": lead_hours,
                    "lead_band": lead_band,
                }
            )
    return pd.DataFrame(rows, columns=_PANEL_COLUMNS)


def score_signal_panel(
    panel: pd.DataFrame,
    *,
    ece_bins: int = 10,
    probability_floor: float = 1e-6,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> SignalStudyReport:
    """Score model and midpoint on identical vectors with Event-equal weights."""
    if panel.empty:
        raise ValueError("signal panel is empty")
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least 2")
    if not 0.0 < probability_floor < 1.0:
        raise ValueError("probability_floor must be between 0 and 1")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    vector_keys = ["event_id", "target_date", "timestamp", "lead_band"]
    score_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    clipped_counts = {source: 0 for source in _SOURCE_COLUMNS}
    for key, vector in panel.groupby(vector_keys, sort=True, dropna=False):
        vector = vector.sort_values("bucket_order", kind="stable")
        outcome = vector["yes_outcome"].astype(float).to_numpy()
        if not np.isclose(outcome.sum(), 1.0):
            raise ValueError("every signal vector requires exactly one true Bucket")
        truth_index = int(np.argmax(outcome))
        row_base = dict(zip(vector_keys, key))
        for source, column in _SOURCE_COLUMNS.items():
            raw_probability = vector[column].astype(float).to_numpy()
            if (
                not np.isfinite(raw_probability).all()
                or (raw_probability < 0.0).any()
                or raw_probability.sum() <= 0.0
            ):
                raise ValueError(f"invalid {source} probability vector")
            if raw_probability[truth_index] < probability_floor:
                clipped_counts[source] += 1
            probability = np.maximum(raw_probability, probability_floor)
            probability /= probability.sum()
            metrics = _vector_scores(probability, outcome)
            score_rows.append(
                {
                    **row_base,
                    "source": source,
                    **metrics,
                    "true_bucket_clipped": bool(
                        raw_probability[truth_index] < probability_floor
                    ),
                }
            )
            for bucket_order, (forecast, observed) in enumerate(
                zip(probability, outcome)
            ):
                probability_rows.append(
                    {
                        **row_base,
                        "source": source,
                        "bucket_order": bucket_order,
                        "probability": float(forecast),
                        "outcome": float(observed),
                    }
                )

    scores = pd.DataFrame(score_rows)
    by_event = (
        scores.groupby(["source", "event_id", "target_date"], as_index=False)[
            list(_SUMMARY_METRICS)
        ]
        .mean()
        .sort_values(["source", "target_date", "event_id"])
        .reset_index(drop=True)
    )
    by_lead = _aggregate_by_lead(
        scores,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    reliability, calibration = _reliability_table(
        pd.DataFrame(probability_rows),
        ece_bins=ece_bins,
    )
    source_summary: dict[str, dict[str, float | int]] = {}
    for source, group in by_event.groupby("source", sort=False):
        values: dict[str, float | int] = {
            metric: float(group[metric].mean())
            for metric in _SUMMARY_METRICS
        }
        values.update(calibration[source])
        values["true_bucket_clipped_count"] = int(clipped_counts[source])
        source_summary[source] = values

    comparison = _paired_comparison(
        by_event,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    vector_count = int(
        panel[["event_id", "timestamp"]].drop_duplicates().shape[0]
    )
    midpoint_vectors = panel.groupby(
        ["event_id", "timestamp"], as_index=False
    )[["midpoint_sum", "midpoint_overround"]].first()
    midpoint_events = midpoint_vectors.groupby("event_id")[
        ["midpoint_sum", "midpoint_overround"]
    ].mean()
    summary = {
        "schema_version": 1,
        "data_class": "forecast_signal_study_summary",
        "descriptive_only": True,
        "sample": {
            "event_count": int(panel["event_id"].nunique()),
            "event_day_count": int(panel["target_date"].nunique()),
            "vector_count": vector_count,
            "panel_row_count": int(len(panel)),
            "bucket_count": int(
                panel.groupby("event_id")["condition_id"].nunique().max()
            ),
            "ece_bins": int(ece_bins),
            "probability_floor": float(probability_floor),
            "bootstrap_samples": int(bootstrap_samples),
            "seed": int(seed),
        },
        "sources": source_summary,
        "comparison": comparison,
        "midpoint": {
            "mean_sum": float(midpoint_events["midpoint_sum"].mean()),
            "mean_overround": float(
                midpoint_events["midpoint_overround"].mean()
            ),
        },
    }
    return SignalStudyReport(
        scores_by_tick=scores,
        by_event=by_event,
        by_lead=by_lead,
        reliability=reliability,
        summary=summary,
    )


def run_forecast_signal_study(
    validation: StudyValidation,
    *,
    output: Path | str,
    ece_bins: int = 10,
    probability_floor: float = 1e-6,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> dict:
    """Build, score, and persist one immutable development signal study."""
    if validation.split_role != "oof_development":
        raise ValueError("ForecastSignalStudy accepts only oof_development")
    return _run_forecast_signal_study(
        validation,
        output=output,
        ece_bins=ece_bins,
        probability_floor=probability_floor,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def run_forecast_signal_validation_study(
    validation: StudyValidation,
    *,
    output: Path | str,
    ece_bins: int = 10,
    probability_floor: float = 1e-6,
    bootstrap_samples: int = 5_000,
    seed: int = 42,
) -> dict:
    """Build and persist one immutable non-development signal validation."""
    if validation.split_role != "model_test":
        raise ValueError(
            "ForecastSignalValidationStudy accepts only model_test"
        )
    return _run_forecast_signal_study(
        validation,
        output=output,
        ece_bins=ece_bins,
        probability_floor=probability_floor,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def _run_forecast_signal_study(
    validation: StudyValidation,
    *,
    output: Path | str,
    ece_bins: int,
    probability_floor: float,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    panel = build_signal_panel(validation)
    report = score_signal_panel(
        panel,
        ece_bins=ece_bins,
        probability_floor=probability_floor,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    requested_events = validation.clob.events.copy()
    requested_events["target_date"] = requested_events["target_date"].astype(str)
    requested_events = requested_events[
        requested_events["target_date"].between(
            validation.start_target, validation.end_target
        )
    ]
    requested_events = requested_events[
        ~requested_events["target_date"].isin(validation.excluded_target_dates)
    ]
    covered_ids = set(panel["event_id"].astype(str))
    exclusions = [
        {
            "event_id": str(row.event_id),
            "target_date": str(row.target_date),
            "reason": "no_complete_midpoint_visible_forecast_overlap",
        }
        for row in requested_events.itertuples(index=False)
        if str(row.event_id) not in covered_ids
    ]
    report.summary["sample"]["requested_event_count"] = int(
        requested_events["event_id"].nunique()
    )
    report.summary["sample"]["excluded_event_count"] = len(exclusions)
    report.summary["coverage_exclusions"] = exclusions
    report.summary["market_unavailable_days"] = list(
        validation.excluded_target_dates
    )
    return save_signal_study_artifact(
        output,
        study_id=validation.study_id,
        station_id=validation.station_id,
        start_target=validation.start_target,
        end_target=validation.end_target,
        split_role=validation.split_role,
        input_sha256=validation.input_sha256,
        panel=panel,
        report=report,
    )


def save_signal_study_artifact(
    root: Path | str,
    *,
    study_id: str,
    station_id: str,
    start_target: str,
    end_target: str,
    split_role: str,
    input_sha256: dict[str, str],
    panel: pd.DataFrame,
    report: SignalStudyReport,
) -> dict:
    """Persist the complete signal evidence with payload checksums."""
    artifact_dir = Path(root)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    frames = {
        "panel.parquet": panel,
        "scores_by_tick.parquet": report.scores_by_tick,
        "by_event.parquet": report.by_event,
        "by_lead.parquet": report.by_lead,
        "reliability.parquet": report.reliability,
    }
    hashes: dict[str, str] = {}
    for filename, frame in frames.items():
        path = artifact_dir / filename
        frame.to_parquet(path, index=False)
        hashes[filename] = _file_sha256(path)
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            report.summary,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    hashes[summary_path.name] = _file_sha256(summary_path)
    manifest = {
        "schema_version": 1,
        "data_class": "forecast_signal_study",
        "study_id": study_id,
        "station_id": station_id,
        "target_range": [start_target, end_target],
        "split_role": split_role,
        "input_sha256": dict(sorted(input_sha256.items())),
        "parameters": {
            "ece_bins": report.summary["sample"]["ece_bins"],
            "probability_floor": report.summary["sample"]["probability_floor"],
            "bootstrap_samples": report.summary["sample"]["bootstrap_samples"],
            "seed": report.summary["sample"]["seed"],
        },
        "sha256": hashes,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_signal_study_artifact(
    root: Path | str,
) -> ForecastSignalArtifact:
    """Load a signal artifact only after verifying every declared payload."""
    artifact_dir = Path(root)
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("data_class") != "forecast_signal_study"
    ):
        raise ValueError("unsupported forecast signal study manifest")
    expected_names = {
        "panel.parquet",
        "scores_by_tick.parquet",
        "by_event.parquet",
        "by_lead.parquet",
        "reliability.parquet",
        "summary.json",
    }
    if set(manifest.get("sha256", {})) != expected_names:
        raise ValueError("forecast signal study manifest has incomplete payloads")
    for filename, expected in manifest["sha256"].items():
        if _file_sha256(artifact_dir / filename) != expected:
            raise ValueError(
                f"forecast signal study checksum mismatch: {filename}"
            )
    return ForecastSignalArtifact(
        manifest=manifest,
        summary=json.loads(
            (artifact_dir / "summary.json").read_text(encoding="utf-8")
        ),
        panel=pd.read_parquet(artifact_dir / "panel.parquet"),
        scores_by_tick=pd.read_parquet(
            artifact_dir / "scores_by_tick.parquet"
        ),
        by_event=pd.read_parquet(artifact_dir / "by_event.parquet"),
        by_lead=pd.read_parquet(artifact_dir / "by_lead.parquet"),
        reliability=pd.read_parquet(artifact_dir / "reliability.parquet"),
    )


def signal_study_id(
    *,
    station_id: str,
    start_target: str,
    end_target: str,
    input_sha256: dict[str, str],
    ece_bins: int,
    probability_floor: float,
    bootstrap_samples: int,
    seed: int,
) -> str:
    identity = {
        "station_id": station_id,
        "target_range": [start_target, end_target],
        "input_sha256": dict(sorted(input_sha256.items())),
        "ece_bins": ece_bins,
        "probability_floor": probability_floor,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"signal-{digest[:16]}"


def _vector_scores(
    probability: np.ndarray, outcome: np.ndarray
) -> dict[str, float]:
    bucket_count = len(probability)
    multiclass_brier = float(np.square(probability - outcome).sum())
    truth_index = int(np.argmax(outcome))
    if bucket_count > 1:
        rps = float(
            np.square(
                np.cumsum(probability)[:-1] - np.cumsum(outcome)[:-1]
            ).sum()
            / (bucket_count - 1)
        )
        entropy = float(
            -(probability * np.log(probability)).sum() / np.log(bucket_count)
        )
    else:
        rps = 0.0
        entropy = 0.0
    return {
        "multiclass_brier": multiclass_brier,
        "log_score": float(-np.log(probability[truth_index])),
        "normalized_rps": rps,
        "pooled_binary_brier": multiclass_brier / bucket_count,
        "normalized_entropy": entropy,
        "max_probability": float(probability.max()),
        "top1_accuracy": float(int(np.argmax(probability)) == truth_index),
        "true_bucket_probability": float(probability[truth_index]),
    }


def _aggregate_by_lead(
    scores: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    event_band = (
        scores.groupby(
            ["source", "lead_band", "event_id", "target_date"],
            as_index=False,
        )[list(_SUMMARY_METRICS)]
        .mean()
    )
    rows: list[dict[str, object]] = []
    for band_index, (lead_band, group) in enumerate(
        event_band.groupby("lead_band", sort=False)
    ):
        indexed = group.set_index(["event_id", "target_date", "source"])
        model = indexed.xs("model_raw", level="source")
        midpoint = indexed.xs("synthetic_midpoint", level="source")
        if not model.index.equals(midpoint.index):
            raise ValueError(
                "model and midpoint lead bands must use identical Events"
            )
        for metric in _SUMMARY_METRICS:
            model_score = float(model[metric].mean())
            midpoint_score = float(midpoint[metric].mean())
            row: dict[str, object] = {
                "lead_band": lead_band,
                "metric": metric,
                "model_raw": model_score,
                "synthetic_midpoint": midpoint_score,
                "model_minus_midpoint": model_score - midpoint_score,
                "skill_score": float("nan"),
                "ci90_low": float("nan"),
                "ci90_high": float("nan"),
                "bootstrap_probability_model_better": float("nan"),
                "event_count": int(
                    model.index.get_level_values("event_id").nunique()
                ),
                "event_day_count": int(
                    model.index.get_level_values("target_date").nunique()
                ),
            }
            if metric in _LOSS_METRICS:
                comparison = _comparison_for_metric(
                    model[metric],
                    midpoint[metric],
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + band_index * len(_LOSS_METRICS)
                    + _LOSS_METRICS.index(metric),
                )
                row.update(comparison)
            rows.append(row)
    result = pd.DataFrame(rows)
    missing_rows = []
    present = set(result["lead_band"].astype(str))
    for lead_band in ("<6h", "6-12h", "12-24h", ">=24h"):
        if lead_band in present:
            continue
        for metric in _SUMMARY_METRICS:
            missing_rows.append(
                {
                    "lead_band": lead_band,
                    "metric": metric,
                    "model_raw": float("nan"),
                    "synthetic_midpoint": float("nan"),
                    "model_minus_midpoint": float("nan"),
                    "skill_score": float("nan"),
                    "ci90_low": float("nan"),
                    "ci90_high": float("nan"),
                    "bootstrap_probability_model_better": float("nan"),
                    "event_count": 0,
                    "event_day_count": 0,
                }
            )
    if missing_rows:
        result = pd.concat(
            [result, pd.DataFrame(missing_rows)], ignore_index=True
        )
    order = pd.CategoricalDtype(
        ["<6h", "6-12h", "12-24h", ">=24h"], ordered=True
    )
    result["lead_band"] = result["lead_band"].astype(order)
    return result.sort_values(["lead_band", "metric"]).reset_index(drop=True)


def _reliability_table(
    probabilities: pd.DataFrame, *, ece_bins: int
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    work = probabilities.copy()
    tick_counts = work[
        ["source", "event_id", "timestamp"]
    ].drop_duplicates().groupby(["source", "event_id"]).size()
    event_counts = work.groupby("source")["event_id"].nunique()
    work["tick_count"] = [
        tick_counts.loc[(source, event_id)]
        for source, event_id in zip(work["source"], work["event_id"])
    ]
    work["event_count"] = work["source"].map(event_counts)
    bucket_counts = work.groupby(
        ["source", "event_id", "timestamp"]
    )["bucket_order"].transform("count")
    work["weight"] = (
        1.0 / work["event_count"] / work["tick_count"] / bucket_counts
    )
    work["bin"] = np.minimum(
        np.floor(work["probability"] * ece_bins).astype(int),
        ece_bins - 1,
    )
    reliability = (
        work.groupby(["source", "bin"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "lower": group.name[1] / ece_bins,
                    "upper": (group.name[1] + 1) / ece_bins,
                    "mean_probability": np.average(
                        group["probability"], weights=group["weight"]
                    ),
                    "observed_frequency": np.average(
                        group["outcome"], weights=group["weight"]
                    ),
                    "weight": group["weight"].sum(),
                    "row_count": len(group),
                    "event_count": group["event_id"].nunique(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    reliability["absolute_gap"] = (
        reliability["mean_probability"]
        - reliability["observed_frequency"]
    ).abs()
    calibration: dict[str, dict[str, float]] = {}
    for source, group in reliability.groupby("source", sort=False):
        normalized_weight = group["weight"] / group["weight"].sum()
        calibration[source] = {
            "ece": float((normalized_weight * group["absolute_gap"]).sum()),
            "mce": float(group["absolute_gap"].max()),
        }
    return reliability, calibration


def _paired_comparison(
    by_event: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    indexed = by_event.set_index(["event_id", "target_date", "source"])
    model = indexed.xs("model_raw", level="source")
    midpoint = indexed.xs("synthetic_midpoint", level="source")
    if not model.index.equals(midpoint.index):
        raise ValueError("model and midpoint must use identical Events")
    result: dict[str, dict[str, float]] = {}
    for metric_index, metric in enumerate(_LOSS_METRICS):
        result[metric] = _comparison_for_metric(
            model[metric],
            midpoint[metric],
            bootstrap_samples=bootstrap_samples,
            seed=seed + metric_index,
        )
    return result


def _comparison_for_metric(
    model: pd.Series,
    midpoint: pd.Series,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    delta = model - midpoint
    day_delta = delta.groupby(level="target_date").mean().to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(day_delta),
        size=(bootstrap_samples, len(day_delta)),
    )
    bootstrap = day_delta[sampled].mean(axis=1)
    model_score = float(model.mean())
    midpoint_score = float(midpoint.mean())
    return {
        "model_minus_midpoint": model_score - midpoint_score,
        "skill_score": (
            float(1.0 - model_score / midpoint_score)
            if midpoint_score != 0.0
            else float("nan")
        ),
        "ci90_low": float(np.quantile(bootstrap, 0.05)),
        "ci90_high": float(np.quantile(bootstrap, 0.95)),
        "bootstrap_probability_model_better": float(
            np.mean(bootstrap < 0.0)
        ),
    }


def _join_latest_visible_forecast(
    *,
    quotes: pd.DataFrame,
    validation: StudyValidation,
    condition_ids: set[str],
) -> pd.DataFrame:
    snapshots = validation.forecast.snapshots.copy()
    snapshots = snapshots[
        snapshots["station_id"].astype(str) == validation.station_id
    ]
    probabilities = validation.forecast.probabilities.copy()
    probabilities["condition_id"] = probabilities["condition_id"].astype(str)
    probabilities = probabilities[
        probabilities["condition_id"].isin(condition_ids)
    ]
    forecast = snapshots.merge(
        probabilities,
        on="snapshot_id",
        how="inner",
        validate="one_to_many",
    )
    forecast["condition_id"] = forecast["condition_id"].astype(str)
    forecast["available_at"] = pd.to_datetime(
        forecast["available_at"], utc=True
    )
    forecast["init_time"] = pd.to_datetime(forecast["init_time"], utc=True)
    left = quotes.sort_values(
        ["timestamp", "condition_id"], kind="stable"
    ).reset_index(drop=True)
    right = forecast.sort_values(
        ["available_at", "init_time", "condition_id"], kind="stable"
    ).reset_index(drop=True)
    joined = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on="available_at",
        by="condition_id",
        direction="backward",
        allow_exact_matches=True,
    ).dropna(subset=["snapshot_id"])
    return joined[
        joined["target_date_x"].astype(str)
        == joined["target_date_y"].astype(str)
    ].rename(columns={"target_date_x": "target_date"})


def _market_close(markets: pd.DataFrame) -> pd.Series:
    values = pd.Series(pd.NaT, index=markets.index, dtype="datetime64[ns, UTC]")
    for column in ("closed_at", "market_end_at", "event_end_at"):
        if column in markets:
            values = values.fillna(
                pd.to_datetime(markets[column], utc=True, errors="coerce")
            )
    if values.isna().any():
        raise ValueError("every signal-study Market requires a close timestamp")
    return values


def _settlement_at(target_date: str, station_id: str) -> pd.Timestamp:
    timezone = STATIONS[station_id].timezone
    return (
        pd.Timestamp(target_date)
        .tz_localize(timezone)
        .normalize()
        + pd.DateOffset(days=1)
    ).tz_convert("UTC")


def _lead_band(lead_hours: float) -> str:
    if lead_hours < 6.0:
        return "<6h"
    if lead_hours < 12.0:
        return "6-12h"
    if lead_hours < 24.0:
        return "12-24h"
    return ">=24h"


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
