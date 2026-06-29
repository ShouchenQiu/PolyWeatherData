"""Managed experiment value objects shared by catalog and execution layers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


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
