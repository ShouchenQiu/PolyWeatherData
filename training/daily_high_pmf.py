"""Leakage-safe point-in-time examples for daily-high PMF models."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch import nn

from model.daily_high_pmf import (
    DailyHighPmfNetwork,
    DailyHighPmfModel,
    SUPPORT_MAX_F,
    SUPPORT_MIN_F,
    TemperaturePmf,
)


FEATURE_SCHEMA_VERSION = 1
MAX_TARGET_HOURS = 25
_SLOT_FIELDS = (
    "observed_temp_f",
    "observation_mask",
    "tft_p10_f",
    "tft_p50_f",
    "tft_p90_f",
    "hrrr_f",
    "forecast_mask",
    "fxx_scaled",
)
_SCALAR_FEATURES = (
    "observed_high_f",
    "observed_high_mask",
    "observed_hour_fraction",
    "forecast_hour_fraction",
    "hours_to_settlement_scaled",
    "day_of_year_sin",
    "day_of_year_cos",
    "target_hour_count_scaled",
)


@dataclass(frozen=True)
class DailyHighPmfExamples:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    schema_version: int = FEATURE_SCHEMA_VERSION
    rejection_counts: tuple[tuple[str, int], ...] = ()

    def save(
        self,
        root: Path | str,
        *,
        source_sha256: dict[str, str],
    ) -> dict[str, object]:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=False)
        frame_path = directory / "examples.parquet"
        self.frame.to_parquet(frame_path, index=False)
        dates = sorted(self.frame["target_date"].astype(str).unique())
        manifest = {
            "schema_version": self.schema_version,
            "data_class": "point_in_time_daily_high_pmf_examples",
            "feature_columns": list(self.feature_columns),
            "feature_contract_sha256": _json_sha256(list(self.feature_columns)),
            "row_count": len(self.frame),
            "target_date_range": [dates[0], dates[-1]] if dates else [],
            "target_day_count": len(dates),
            "coverage_type_counts": {
                str(key): int(value)
                for key, value in self.frame.get(
                    "coverage_type", pd.Series(dtype=str)
                ).value_counts().sort_index().items()
            },
            "rejection_counts": dict(self.rejection_counts),
            "source_sha256": dict(sorted(source_sha256.items())),
            "sha256": {"examples.parquet": _file_sha256(frame_path)},
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
        )
        return manifest

    @classmethod
    def load(cls, root: Path | str) -> "DailyHighPmfExamples":
        directory = Path(root)
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="ascii")
        )
        if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported PMF examples schema version")
        columns = tuple(str(value) for value in manifest["feature_columns"])
        if manifest.get("feature_contract_sha256") != _json_sha256(list(columns)):
            raise ValueError("PMF examples feature contract mismatch")
        frame_path = directory / "examples.parquet"
        if _file_sha256(frame_path) != manifest["sha256"]["examples.parquet"]:
            raise ValueError("PMF examples checksum mismatch")
        frame = pd.read_parquet(frame_path)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError("PMF examples features missing: " + ", ".join(missing))
        return cls(
            frame=frame,
            feature_columns=columns,
            schema_version=int(manifest["schema_version"]),
            rejection_counts=tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in manifest.get("rejection_counts", {}).items()
                )
            ),
        )


@dataclass(frozen=True)
class DailyHighPmfEvaluation:
    snapshots: pd.DataFrame
    temperature_probabilities: pd.DataFrame

    def save(
        self,
        root: Path | str,
        *,
        split_role: str,
        source_sha256: dict[str, str],
    ) -> dict[str, object]:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=False)
        frames = {
            "snapshots.parquet": self.snapshots,
            "temperature_probabilities.parquet": self.temperature_probabilities,
        }
        hashes: dict[str, str] = {}
        for filename, frame in frames.items():
            path = directory / filename
            frame.to_parquet(path, index=False)
            hashes[filename] = _file_sha256(path)
        manifest = {
            "schema_version": 1,
            "data_class": "daily_high_pmf_evaluation",
            "distribution_source": "daily_high_pmf_pilot",
            "split_role": split_role,
            "row_counts": {
                "snapshots": len(self.snapshots),
                "temperature_probabilities": len(
                    self.temperature_probabilities
                ),
            },
            "source_sha256": dict(sorted(source_sha256.items())),
            "sha256": hashes,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
        )
        return manifest


def run_expanding_pmf_oof(
    examples: DailyHighPmfExamples,
    *,
    tft_checkpoint_sha256: str,
    training_manifest_sha256: str,
    folds: int = 4,
    epochs: int = 40,
    seed: int = 42,
) -> DailyHighPmfEvaluation:
    """Fit only on earlier Event days and emit OOF PMFs with climatology."""
    snapshot_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    support = np.arange(SUPPORT_MIN_F, SUPPORT_MAX_F + 1, dtype=float)
    for fold, (train, validation) in enumerate(
        expanding_target_day_folds(examples.frame, folds=folds)
    ):
        model = fit_daily_high_pmf_model(
            train,
            feature_columns=examples.feature_columns,
            tft_checkpoint_sha256=tft_checkpoint_sha256,
            training_manifest_sha256=training_manifest_sha256,
            epochs=epochs,
            seed=seed + fold,
        )
        counts = np.full(len(support), 1e-6, dtype=float)
        labels = train["label_temperature_f"].to_numpy(dtype=int)
        np.add.at(counts, labels - SUPPORT_MIN_F, 1.0)
        climatology_logits = np.log(counts / counts.sum())
        for row in validation.itertuples(index=False):
            values = pd.Series(row._asdict())
            candidate = model.predict(values)
            observed_high = (
                float(values["observed_high_f"])
                if float(values.get("observed_high_mask", 0.0)) > 0
                else None
            )
            climatology = TemperaturePmf.from_logits(
                climatology_logits,
                support_min_f=SUPPORT_MIN_F,
                observed_high_f=observed_high,
            )
            label = int(values["label_temperature_f"])
            snapshot_id = _snapshot_id(
                pd.Timestamp(values["init_time"]), str(values["target_date"]), fold
            )
            candidate_scores = _temperature_scores(candidate, label)
            climatology_scores = _temperature_scores(climatology, label)
            snapshot_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "fold": fold,
                    "init_time": pd.Timestamp(values["init_time"]),
                    "target_date": str(values["target_date"]),
                    "coverage_type": str(values["coverage_type"]),
                    "observed_hour_count": int(values["observed_hour_count"]),
                    "forecast_hour_count": int(values["forecast_hour_count"]),
                    "target_hour_count": int(
                        values.get(
                            "target_hour_count",
                            int(values["observed_hour_count"])
                            + int(values["forecast_hour_count"]),
                        )
                    ),
                    "observed_high_f": float(values["observed_high_f"]),
                    "actual_high_f": label,
                    "p10_f": candidate.p10_f,
                    "p50_f": candidate.p50_f,
                    "p90_f": candidate.p90_f,
                    "pmf_log_loss": candidate_scores[0],
                    "pmf_rps": candidate_scores[1],
                    "climatology_log_loss": climatology_scores[0],
                    "climatology_rps": climatology_scores[1],
                    **{
                        optional: values[optional]
                        for optional in (
                            "source_snapshot_id",
                            "available_at",
                            "observed_through",
                            "forecast_start_at",
                            "forecast_through",
                        )
                        if optional in values
                    },
                }
            )
            probability_rows.extend(
                {
                    "snapshot_id": snapshot_id,
                    "temperature_f": int(temperature),
                    "pmf_probability": float(candidate.probabilities[index]),
                    "climatology_probability": float(
                        climatology.probabilities[index]
                    ),
                }
                for index, temperature in enumerate(support)
            )
    return DailyHighPmfEvaluation(
        snapshots=pd.DataFrame(snapshot_rows),
        temperature_probabilities=pd.DataFrame(probability_rows),
    )


def predict_pmf_examples(
    model: DailyHighPmfModel,
    examples: DailyHighPmfExamples,
    *,
    lineage_label: str,
) -> DailyHighPmfEvaluation:
    """Predict one frozen PMF for every supplied point-in-time example."""
    snapshot_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    for row in examples.frame.itertuples(index=False):
        values = pd.Series(row._asdict())
        pmf = model.predict(values)
        snapshot_id = hashlib.sha256(
            (
                f"{values['init_time']}|{values['target_date']}|"
                f"{lineage_label}"
            ).encode("ascii")
        ).hexdigest()
        label = int(values["label_temperature_f"])
        log_loss, rps = _temperature_scores(pmf, label)
        output = {
            "snapshot_id": snapshot_id,
            "init_time": pd.Timestamp(values["init_time"]),
            "target_date": str(values["target_date"]),
            "coverage_type": str(values["coverage_type"]),
            "observed_hour_count": int(values["observed_hour_count"]),
            "forecast_hour_count": int(values["forecast_hour_count"]),
            "target_hour_count": int(values["target_hour_count"]),
            "observed_high_f": float(values["observed_high_f"]),
            "actual_high_f": label,
            "p10_f": pmf.p10_f,
            "p50_f": pmf.p50_f,
            "p90_f": pmf.p90_f,
            "pmf_log_loss": log_loss,
            "pmf_rps": rps,
        }
        for optional in (
            "source_snapshot_id",
            "available_at",
            "observed_through",
            "forecast_start_at",
            "forecast_through",
        ):
            if optional in values:
                output[optional] = values[optional]
        snapshot_rows.append(output)
        probability_rows.extend(
            {
                "snapshot_id": snapshot_id,
                "temperature_f": int(temperature),
                "pmf_probability": float(probability),
            }
            for temperature, probability in zip(
                pmf.support_f, pmf.probabilities, strict=True
            )
        )
    return DailyHighPmfEvaluation(
        snapshots=pd.DataFrame(snapshot_rows),
        temperature_probabilities=pd.DataFrame(probability_rows),
    )


def feature_columns() -> tuple[str, ...]:
    slots = tuple(
        f"slot_{slot:02d}_{field}"
        for slot in range(MAX_TARGET_HOURS)
        for field in _SLOT_FIELDS
    )
    return slots + _SCALAR_FEATURES


def fit_daily_high_pmf_model(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    tft_checkpoint_sha256: str,
    training_manifest_sha256: str,
    epochs: int = 40,
    seed: int = 42,
) -> DailyHighPmfModel:
    """Fit using feature normalization learned only from this training frame."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    required = set(feature_columns) | {"label_temperature_f"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("PMF training frame missing: " + ", ".join(missing))
    values = frame[list(feature_columns)].to_numpy(dtype=np.float32)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("PMF training features must be finite and non-empty")
    labels_f = frame["label_temperature_f"].to_numpy(dtype=int)
    if np.any(labels_f < SUPPORT_MIN_F) or np.any(labels_f > SUPPORT_MAX_F):
        raise ValueError("PMF label falls outside supported Fahrenheit range")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (values - mean) / scale
    labels = labels_f - SUPPORT_MIN_F
    torch.manual_seed(seed)
    network = DailyHighPmfNetwork(
        len(feature_columns), SUPPORT_MAX_F - SUPPORT_MIN_F + 1
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.from_numpy(normalized)
    y = torch.from_numpy(labels.astype(np.int64))
    network.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(network(x), y)
        loss.backward()
        optimizer.step()
    return DailyHighPmfModel(
        network=network,
        feature_columns=feature_columns,
        feature_mean=mean,
        feature_scale=scale,
        tft_checkpoint_sha256=tft_checkpoint_sha256,
        training_manifest_sha256=training_manifest_sha256,
    )


def expanding_target_day_folds(
    frame: pd.DataFrame,
    *,
    folds: int = 4,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return expanding folds whose validation Event days are strictly later."""
    _require_columns(frame, {"target_date"}, "PMF examples")
    days = np.array(sorted(frame["target_date"].astype(str).unique()))
    if folds < 1 or len(days) < folds + 1:
        raise ValueError("expanding OOF needs at least folds + 1 target days")
    blocks = [block.tolist() for block in np.array_split(days, folds + 1)]
    result: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in range(folds):
        train_days = {day for block in blocks[: fold + 1] for day in block}
        validation_days = set(blocks[fold + 1])
        train = frame[frame["target_date"].astype(str).isin(train_days)].copy()
        validation = frame[
            frame["target_date"].astype(str).isin(validation_days)
        ].copy()
        result.append((train, validation))
    return result


def build_point_in_time_examples(
    *,
    hourly_predictions: pd.DataFrame,
    observations: pd.DataFrame,
    snapshots: pd.DataFrame,
    actuals: pd.DataFrame,
    station_timezone: str,
) -> DailyHighPmfExamples:
    """Build fixed-width examples using only evidence visible at each init."""
    _require_columns(
        hourly_predictions,
        {
            "init_time", "valid_time", "fxx", "p10", "p50", "p90",
            "base_temp_c",
        },
        "hourly_predictions",
    )
    _require_columns(
        observations,
        {"valid_time", "available_at", "temperature_c"},
        "observations",
    )
    _require_columns(snapshots, {"init_time", "target_date"}, "snapshots")
    _require_columns(actuals, {"target_date", "actual_high_f"}, "actuals")

    predictions = hourly_predictions.copy()
    predictions["init_time"] = pd.to_datetime(predictions["init_time"], utc=True)
    predictions["valid_time"] = pd.to_datetime(predictions["valid_time"], utc=True)
    obs = observations.copy()
    obs["valid_time"] = pd.to_datetime(obs["valid_time"], utc=True)
    obs["available_at"] = pd.to_datetime(obs["available_at"], utc=True)
    optional_case_columns = [
        column
        for column in (
            "snapshot_id",
            "available_at",
            "observed_through",
            "forecast_start_at",
            "forecast_through",
        )
        if column in snapshots.columns
    ]
    cases = snapshots[["init_time", "target_date", *optional_case_columns]].copy()
    cases["init_time"] = pd.to_datetime(cases["init_time"], utc=True)
    cases["target_date"] = cases["target_date"].astype(str)
    if cases.duplicated(["init_time", "target_date"]).any():
        raise ValueError("snapshots contain duplicate init_time/target_date")
    truth = actuals[["target_date", "actual_high_f"]].copy()
    truth["target_date"] = truth["target_date"].astype(str)
    if truth["target_date"].duplicated().any():
        raise ValueError("actuals contain duplicate target_date")
    actual_by_date = truth.set_index("target_date")["actual_high_f"]
    prediction_groups = {
        pd.Timestamp(init): group
        for init, group in predictions.groupby("init_time", sort=False)
    }

    rows: list[dict[str, object]] = []
    rejection_counts: dict[str, int] = {}
    columns = feature_columns()
    for case in cases.itertuples(index=False):
        init = pd.Timestamp(case.init_time)
        target_date = str(case.target_date)
        if target_date not in actual_by_date.index or init not in prediction_groups:
            _increment(rejection_counts, "missing_truth_or_predictions")
            continue
        expected = _target_hours(target_date, station_timezone)
        if len(expected) > MAX_TARGET_HOURS:
            raise ValueError("target local day exceeds 25 hourly slots")
        expected_set = set(expected)
        visible_obs = obs[
            obs["valid_time"].isin(expected_set)
            & (obs["valid_time"] <= init)
            & (obs["available_at"] <= init)
        ].sort_values("valid_time")
        if visible_obs["valid_time"].duplicated().any():
            raise ValueError("observations contain duplicate visible target slot")
        forecast = prediction_groups[init].copy()
        extrapolated = forecast.get(
            "is_extrapolated", pd.Series(False, index=forecast.index)
        ).fillna(False).astype(bool)
        forecast = forecast[
            forecast["valid_time"].isin(expected_set)
            & (forecast["valid_time"] > init)
            & ~extrapolated
        ].sort_values("valid_time")
        if forecast["valid_time"].duplicated().any():
            raise ValueError("predictions contain duplicate future target slot")
        obs_by_time = visible_obs.set_index("valid_time")
        forecast_by_time = forecast.set_index("valid_time")
        covered = set(obs_by_time.index) | set(forecast_by_time.index)
        if covered != expected_set:
            _increment(rejection_counts, "incomplete_target_hour_coverage")
            continue

        row: dict[str, object] = {name: 0.0 for name in columns}
        row.update({"init_time": init, "target_date": target_date})
        for optional in (
            "snapshot_id",
            "available_at",
            "observed_through",
            "forecast_start_at",
            "forecast_through",
        ):
            if optional in cases.columns:
                row[
                    "source_snapshot_id" if optional == "snapshot_id" else optional
                ] = getattr(case, optional)
        observed_values_f: list[float] = []
        for slot, valid_time in enumerate(expected):
            prefix = f"slot_{slot:02d}_"
            if valid_time in obs_by_time.index:
                value_f = float(obs_by_time.loc[valid_time, "temperature_c"]) * 1.8 + 32.0
                row[prefix + "observed_temp_f"] = value_f
                row[prefix + "observation_mask"] = 1.0
                observed_values_f.append(value_f)
            else:
                item = forecast_by_time.loc[valid_time]
                base = float(item["base_temp_c"])
                row[prefix + "tft_p10_f"] = (base + float(item["p10"])) * 1.8 + 32.0
                row[prefix + "tft_p50_f"] = (base + float(item["p50"])) * 1.8 + 32.0
                row[prefix + "tft_p90_f"] = (base + float(item["p90"])) * 1.8 + 32.0
                row[prefix + "hrrr_f"] = base * 1.8 + 32.0
                row[prefix + "forecast_mask"] = 1.0
                row[prefix + "fxx_scaled"] = float(item["fxx"]) / 48.0

        observed_count = len(visible_obs)
        forecast_count = len(forecast)
        observed_high = max(observed_values_f) if observed_values_f else 0.0
        end = expected[-1]
        day = pd.Timestamp(target_date)
        angle = 2.0 * np.pi * float(day.dayofyear) / 366.0
        row.update(
            {
                "observed_high_f": observed_high,
                "observed_high_mask": float(bool(observed_values_f)),
                "observed_hour_fraction": observed_count / len(expected),
                "forecast_hour_fraction": forecast_count / len(expected),
                "hours_to_settlement_scaled": max(
                    0.0, (end - init).total_seconds() / 3600.0
                ) / 48.0,
                "day_of_year_sin": float(np.sin(angle)),
                "day_of_year_cos": float(np.cos(angle)),
                "target_hour_count_scaled": len(expected) / MAX_TARGET_HOURS,
                "observed_hour_count": observed_count,
                "forecast_hour_count": forecast_count,
                "target_hour_count": len(expected),
                "coverage_type": (
                    "forecast_only" if observed_count == 0
                    else "observed_only" if forecast_count == 0
                    else "observed_plus_forecast"
                ),
                "label_temperature_f": _round_half_up(
                    float(actual_by_date.loc[target_date])
                ),
            }
        )
        rows.append(row)
    return DailyHighPmfExamples(
        pd.DataFrame(rows),
        columns,
        FEATURE_SCHEMA_VERSION,
        tuple(sorted(rejection_counts.items())),
    )


def _target_hours(target_date: str, timezone: str) -> pd.DatetimeIndex:
    zone = ZoneInfo(timezone)
    start = pd.Timestamp(target_date).tz_localize(zone)
    end = start + pd.DateOffset(days=1)
    return pd.date_range(start, end, freq="h", inclusive="right").tz_convert("UTC")


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(missing)}")


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _snapshot_id(init_time: pd.Timestamp, target_date: str, fold: int) -> str:
    payload = f"{init_time.isoformat()}|{target_date}|{fold}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _temperature_scores(pmf: TemperaturePmf, label_f: int) -> tuple[float, float]:
    label_index = int(label_f) - int(pmf.support_f[0])
    if not 0 <= label_index < len(pmf.support_f):
        raise ValueError("temperature label falls outside PMF support")
    probability = max(float(pmf.probabilities[label_index]), 1e-12)
    outcome_cdf = (pmf.support_f >= float(label_f)).astype(float)
    rps = float(np.mean((np.cumsum(pmf.probabilities) - outcome_cdf) ** 2))
    return -float(np.log(probability)), rps
