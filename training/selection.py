"""Validation-only Top-K evidence and final-selection inputs."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from model.mc import McSampler
from training.catalog import TrainingCatalog
from training.audit_frames import AuditedFrameCache
from training.config import ExperimentTemplate, ResolvedTrialSnapshot
from training.predictions import build_quantile_prediction_frame


@dataclass(frozen=True)
class ValidationEvidenceContext:
    study_id: str
    trial_number: int
    execution_id: str
    station_id: str
    split: str
    split_range: tuple[str, str]
    checkpoint_path: str | None
    trial_snapshot: ResolvedTrialSnapshot


EvidenceProvider = Callable[
    [ValidationEvidenceContext],
    dict[str, object],
]


def compute_validation_evidence(
    frame: pd.DataFrame,
    *,
    mc_samples: int = 2_000,
) -> dict[str, object]:
    """Compute validation-only quantile, strata, daily-high, and MC evidence."""
    required = {
        "init_time",
        "valid_time",
        "fxx",
        "target",
        "p10",
        "p50",
        "p90",
        "base_temp_c",
        "weight",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "validation prediction frame missing columns: "
            + ", ".join(missing)
        )
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    work = frame.copy()
    work["init_time"] = pd.to_datetime(work["init_time"], utc=True)
    work["valid_time"] = pd.to_datetime(work["valid_time"], utc=True)
    trusted = work[work["weight"].astype(float) > 0].copy()
    if trusted.empty:
        raise ValueError("validation evidence has no trusted target cells")

    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        return float(
            np.average(
                values.to_numpy(dtype=float),
                weights=weights.to_numpy(dtype=float),
            )
        )

    weights = trusted["weight"].astype(float)
    p50_error = (trusted["p50"] - trusted["target"]).abs()
    pinball: dict[str, float] = {}
    for label, quantile in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
        delta = trusted["target"] - trusted[label]
        loss = np.maximum(
            quantile * delta.to_numpy(dtype=float),
            (quantile - 1.0) * delta.to_numpy(dtype=float),
        )
        pinball[label] = float(
            np.average(loss, weights=weights.to_numpy(dtype=float))
        )

    def strata(column: str) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for key, group in trusted.groupby(column, sort=True):
            group_weights = group["weight"].astype(float)
            result[str(key)] = {
                "cells": int(len(group)),
                "p50_mae": weighted_mean(
                    (group["p50"] - group["target"]).abs(),
                    group_weights,
                ),
                "p50_bias": weighted_mean(
                    group["p50"] - group["target"],
                    group_weights,
                ),
                "coverage_p10_p90": weighted_mean(
                    (
                        (group["target"] >= group["p10"])
                        & (group["target"] <= group["p90"])
                    ).astype(float),
                    group_weights,
                ),
                "interval_width": weighted_mean(
                    group["p90"] - group["p10"],
                    group_weights,
                ),
            }
        return result

    trusted["init_hour"] = trusted["init_time"].dt.hour
    trusted["init_month"] = trusted["init_time"].dt.month
    daily_errors_f: list[float] = []
    daily_biases_f: list[float] = []
    raw_daily_errors_f: list[float] = []
    daily_beats_raw: list[float] = []
    mc_p50_errors_f: list[float] = []
    mc_covered: list[float] = []
    trusted_horizons: list[int] = []
    for index, (_init_time, group) in enumerate(
        trusted.groupby("init_time", sort=True)
    ):
        group = group.sort_values("fxx")
        predicted_c = group["base_temp_c"] + group["p50"]
        actual_c = group["base_temp_c"] + group["target"]
        predicted_high_f = float(predicted_c.max() * 1.8 + 32.0)
        actual_high_f = float(actual_c.max() * 1.8 + 32.0)
        daily_error = abs(predicted_high_f - actual_high_f)
        raw_high_f = float(group["base_temp_c"].max() * 1.8 + 32.0)
        raw_error = abs(raw_high_f - actual_high_f)
        daily_errors_f.append(daily_error)
        daily_biases_f.append(predicted_high_f - actual_high_f)
        raw_daily_errors_f.append(raw_error)
        daily_beats_raw.append(float(daily_error < raw_error))
        sigma = (
            (group["p90"] - group["p10"]).to_numpy(dtype=float)
            / (2.0 * 1.2815515655446004)
        )
        sampler = McSampler(n_samp=mc_samples, rng_seed=42 + index)
        sampled = sampler.sample_daily_high(
            predicted_c.to_numpy(dtype=float),
            np.clip(sigma, 0.01, None),
        )
        p10_f, p50_f, p90_f = np.quantile(sampled, [0.1, 0.5, 0.9])
        mc_p50_errors_f.append(abs(float(p50_f) - actual_high_f))
        mc_covered.append(float(p10_f <= actual_high_f <= p90_f))
        trusted_horizons.append(int(group["fxx"].max()))

    return {
        "p50_mae": weighted_mean(p50_error, weights),
        "p50_bias": weighted_mean(
            trusted["p50"] - trusted["target"],
            weights,
        ),
        "pinball": pinball,
        "coverage_p10_p90": weighted_mean(
            (
                (trusted["target"] >= trusted["p10"])
                & (trusted["target"] <= trusted["p90"])
            ).astype(float),
            weights,
        ),
        "interval_width": weighted_mean(
            trusted["p90"] - trusted["p10"],
            weights,
        ),
        "trusted_cells": int(len(trusted)),
        "by_fxx": strata("fxx"),
        "by_init_hour": strata("init_hour"),
        "by_init_month": strata("init_month"),
        "trusted_horizon": {
            "min_fxx": min(trusted_horizons),
            "max_fxx": max(trusted_horizons),
            "mean_fxx": float(np.mean(trusted_horizons)),
        },
        "daily_high": {
            "cycles": len(daily_errors_f),
            "p50_mae_f": float(np.mean(daily_errors_f)),
            "p50_bias_f": float(np.mean(daily_biases_f)),
            "raw_mae_f": float(np.mean(raw_daily_errors_f)),
            "beats_raw_cycle_fraction": float(np.mean(daily_beats_raw)),
        },
        "mc_calibration": {
            "sample_count": mc_samples,
            "cycles": len(mc_p50_errors_f),
            "p50_mae_f": float(np.mean(mc_p50_errors_f)),
            "p10_p90_coverage": float(np.mean(mc_covered)),
        },
        "trading_backtest": {
            "status": "not_available",
            "reason": (
                "no historical event/order-book inputs were supplied; "
                "not used for ranking"
            ),
        },
    }


class TopKEvidenceBuilder:
    """Rank completed trials by validation objective and freeze review evidence."""

    def __init__(self, catalog: TrainingCatalog) -> None:
        self._catalog = catalog

    def build(
        self,
        study_id: str,
        *,
        evidence_provider: EvidenceProvider | None = None,
    ) -> dict[str, object]:
        study = self._catalog.validate_study(study_id)
        if study.status not in {
            "completed",
            "budget_exhausted",
            "awaiting_selection",
        }:
            raise PermissionError(
                "Top-K evidence requires a finished Optuna study"
            )
        template = ExperimentTemplate.from_dict(
            json.loads(
                self._catalog.get_template(study.template_hash).config_json
            )
        )
        artifacts = self._catalog.list_artifacts(study_id)
        snapshot_by_execution = {
            artifact.execution_id: Path(artifact.path)
            for artifact in artifacts
            if artifact.artifact_type == "trial_snapshot"
            and artifact.execution_id is not None
        }
        candidates = []
        for execution in self._catalog.list_executions(study_id):
            if execution.execution_type != "trial" or execution.status != "succeeded":
                continue
            loss = float(execution.metrics.get("best_val_loss", math.nan))
            if not math.isfinite(loss) or execution.trial_number is None:
                continue
            snapshot_path = snapshot_by_execution.get(execution.execution_id)
            if snapshot_path is None or not snapshot_path.is_file():
                raise RuntimeError(
                    f"trial {execution.trial_number} lacks snapshot evidence"
                )
            snapshot = ResolvedTrialSnapshot.from_dict(
                json.loads(snapshot_path.read_text(encoding="utf-8"))
            )
            candidates.append((loss, execution, snapshot))
        candidates.sort(key=lambda item: (item[0], item[1].trial_number))
        selected = candidates[: template.study.top_k]
        if not selected:
            raise RuntimeError("finished study has no successful trials")

        trials: list[dict[str, object]] = []
        provider = evidence_provider or self._model_evidence
        for rank, (loss, execution, snapshot) in enumerate(selected, start=1):
            context = ValidationEvidenceContext(
                study_id=study_id,
                trial_number=int(execution.trial_number),
                execution_id=execution.execution_id,
                station_id=study.station_id,
                split="val",
                split_range=tuple(template.splits["val"]),
                checkpoint_path=execution.metrics.get("checkpoint_path"),
                trial_snapshot=snapshot,
            )
            trials.append(
                {
                    "rank": rank,
                    "trial_number": context.trial_number,
                    "execution_id": context.execution_id,
                    "best_val_loss": loss,
                    "final_val_loss": execution.metrics.get(
                        "final_val_loss"
                    ),
                    "gpu_seconds": execution.metrics.get("gpu_seconds"),
                    "parameters": dict(snapshot.parameters),
                    "trial_snapshot_hash": snapshot.snapshot_hash(),
                    "checkpoint_path": context.checkpoint_path,
                    "environment": execution.metrics.get("environment"),
                    "downstream_validation": provider(context),
                }
            )
        payload: dict[str, object] = {
            "schema_version": 1,
            "study_id": study_id,
            "template_hash": study.template_hash,
            "data_order_hash": self._catalog.get_workspace(
                study.workspace_path
            ).data_order_hash,
            "objective": template.study.objective,
            "direction": template.study.direction,
            "split": "val",
            "split_range": list(template.splits["val"]),
            "test_accessed": False,
            "trials": trials,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload["manifest_hash"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        path = (
            Path(study.workspace_path)
            / "studies"
            / study_id
            / "top_k_validation_evidence.json"
        )
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError("frozen Top-K evidence already differs")
        else:
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self._catalog.register_artifact(
                study_id,
                artifact_type="top_k_validation_evidence",
                path=path,
            )
        if study.status != "awaiting_selection":
            self._catalog.transition_study(
                study_id,
                "awaiting_selection",
                message="Top-K validation evidence frozen",
            )
        return payload

    def _model_evidence(
        self,
        context: ValidationEvidenceContext,
    ) -> dict[str, object]:
        """Reload a trial checkpoint and evaluate only the audited val frame."""
        study = self._catalog.get_study(context.study_id)
        workspace = Path(study.workspace_path)
        dataset_artifact = next(
            (
                artifact
                for artifact in self._catalog.list_artifacts(context.study_id)
                if artifact.execution_id == context.execution_id
                and artifact.artifact_type == "training_dataset"
            ),
            None,
        )
        if dataset_artifact is None:
            raise RuntimeError(
                f"trial {context.trial_number} lacks training_dataset artifact"
            )
        if context.checkpoint_path is None:
            raise RuntimeError(
                f"trial {context.trial_number} lacks checkpoint path"
            )
        with Path(dataset_artifact.path).open("rb") as handle:
            training = pickle.load(handle)
        cached = AuditedFrameCache(
            workspace / "artifacts" / "audit_frames",
            workspace / "data_store",
        ).load(
            "val",
            context.station_id,
            context.split_range[0],
            context.split_range[1],
        )
        if cached is None:
            raise RuntimeError("audited validation frame is missing or stale")
        frame = cached[0].copy()
        prediction_frame = build_quantile_prediction_frame(
            checkpoint_path=context.checkpoint_path,
            training_dataset=training,
            frame=frame,
            batch_size=int(
                context.trial_snapshot.parameters.get("batch_size", 4)
            ),
        )
        return compute_validation_evidence(prediction_frame)
