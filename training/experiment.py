"""Managed experiment value objects shared by catalog and execution layers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from training.config import ExperimentTemplate, SmokePolicy


@dataclass(frozen=True)
class ValidationExecutionLimits:
    """Resolved resource limits for one train/validation-only execution."""

    execution_type: str
    min_epochs: int
    max_epochs: int
    batch_size: int
    precision: str
    num_workers: int
    limit_train_batches: int | None
    limit_val_batches: int | None
    timeout_minutes: int | None

    @classmethod
    def for_smoke(cls, policy: SmokePolicy) -> "ValidationExecutionLimits":
        return cls(
            execution_type="smoke",
            min_epochs=1,
            max_epochs=policy.max_epochs,
            batch_size=policy.batch_size,
            precision=policy.precision,
            num_workers=policy.num_workers,
            limit_train_batches=policy.limit_train_batches,
            limit_val_batches=policy.limit_val_batches,
            timeout_minutes=policy.timeout_minutes,
        )

    @classmethod
    def from_catalog_limits(
        cls,
        execution_type: str,
        limits: dict[str, Any],
        template: ExperimentTemplate,
    ) -> "ValidationExecutionLimits":
        if execution_type == "smoke":
            return cls.for_smoke(template.execution.smoke)
        if execution_type == "calibration":
            return cls(
                execution_type="calibration",
                min_epochs=1,
                max_epochs=int(limits["max_epochs"]),
                batch_size=int(limits["batch_size"]),
                precision=str(limits["precision"]),
                num_workers=int(limits["num_workers"]),
                limit_train_batches=None,
                limit_val_batches=None,
                timeout_minutes=max(
                    1, int(float(limits["max_gpu_hours"]) * 60)
                ),
            )
        raise ValueError(
            f"unsupported validation execution type: {execution_type}"
        )

    @classmethod
    def for_trial(
        cls,
        template: ExperimentTemplate,
        parameters: dict[str, Any],
    ) -> "ValidationExecutionLimits":
        return cls(
            execution_type="trial",
            min_epochs=template.trainer.min_epochs,
            max_epochs=template.trainer.max_epochs,
            batch_size=int(
                parameters.get("batch_size", template.trainer.batch_size)
            ),
            precision=template.trainer.precision,
            num_workers=template.trainer.num_workers,
            limit_train_batches=None,
            limit_val_batches=None,
            timeout_minutes=None,
        )

    @classmethod
    def for_final(
        cls,
        template: ExperimentTemplate,
        parameters: dict[str, Any],
    ) -> "ValidationExecutionLimits":
        trial = cls.for_trial(template, parameters)
        return cls(
            execution_type="final",
            min_epochs=trial.min_epochs,
            max_epochs=trial.max_epochs,
            batch_size=trial.batch_size,
            precision=trial.precision,
            num_workers=trial.num_workers,
            limit_train_batches=None,
            limit_val_batches=None,
            timeout_minutes=None,
        )


@dataclass(frozen=True)
class FinalEvaluationApproval:
    """Catalog-issued evidence authorising one selected trial's final run."""

    approval_id: str
    study_id: str
    selected_trial_number: int
    trial_snapshot_hash: str
    approved_by: str
    approved_at: str

    @classmethod
    def from_catalog_record(
        cls,
        payload: dict[str, Any],
    ) -> "FinalEvaluationApproval":
        required = {
            "approval_id",
            "study_id",
            "selected_trial_number",
            "trial_snapshot_hash",
            "approved_by",
            "approved_at",
        }
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        if missing:
            raise ValueError(
                "final approval missing required field(s): " + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "final approval contains unknown field(s): " + ", ".join(unknown)
            )
        text = {
            key: str(payload[key]).strip()
            for key in (
                "approval_id",
                "study_id",
                "trial_snapshot_hash",
                "approved_by",
                "approved_at",
            )
        }
        if any(not value for value in text.values()):
            raise ValueError("final approval fields must be non-empty")
        try:
            datetime.fromisoformat(text["approved_at"])
        except ValueError as exc:
            raise ValueError("final approval approved_at must be ISO-8601") from exc
        selected_trial_number = int(payload["selected_trial_number"])
        if selected_trial_number < 0:
            raise ValueError(
                "final approval selected_trial_number must be non-negative"
            )
        return cls(
            approval_id=text["approval_id"],
            study_id=text["study_id"],
            selected_trial_number=selected_trial_number,
            trial_snapshot_hash=text["trial_snapshot_hash"],
            approved_by=text["approved_by"],
            approved_at=text["approved_at"],
        )
