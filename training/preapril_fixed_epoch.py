"""Contracts for the strict pre-April fixed-epoch MC experiment."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PreAprilFixedEpochPlan:
    train: tuple[str, str] = ("2024-10-07", "2026-02-28")
    calibration: tuple[str, str] = ("2026-03-01", "2026-03-31")
    evaluation: tuple[str, str] = ("2026-04-01", "2026-04-30")
    fixed_epochs: int = 3


def preapril_fixed_epoch_template(
    reference: dict[str, Any],
    *,
    plan: PreAprilFixedEpochPlan | None = None,
) -> dict[str, Any]:
    """Freeze a reference TFT configuration without validation-based selection."""
    plan = plan or PreAprilFixedEpochPlan()
    payload = json.loads(json.dumps(reference))
    if payload.get("schema_version") == 2:
        model = payload.get("model")
        if not isinstance(model, dict) or model.get("use_history_target") is not False:
            raise ValueError(
                "pre-April fixed-epoch run requires history-target-disabled schema-v2 reference"
            )
        model.pop("use_history_target")
        payload["schema_version"] = 1
    payload["template_id"] = "klga-preapril-fixed-epoch-v1"
    payload["splits"] = {
        "train": list(plan.train),
        "val": list(plan.calibration),
        "test": list(plan.evaluation),
    }
    trainer = dict(payload["trainer"])
    trainer["fixed_epochs"] = plan.fixed_epochs
    trainer["min_epochs"] = plan.fixed_epochs
    trainer["max_epochs"] = plan.fixed_epochs
    payload["trainer"] = trainer
    return payload


def local_target_window(
    predictions: pd.DataFrame,
    *,
    start: str,
    end: str,
    station_timezone: str,
) -> pd.DataFrame:
    """Keep only decoder cells belonging to a right-closed local target window."""
    work = predictions.copy()
    valid = pd.to_datetime(work["valid_time"], utc=True)
    target_date = (
        (valid - pd.Timedelta(nanoseconds=1))
        .dt.tz_convert(station_timezone)
        .dt.date.astype(str)
    )
    return work.loc[target_date.between(start, end)].copy()
