from __future__ import annotations

import pandas as pd
import json
from pathlib import Path

from training.config import ExperimentTemplate
from training.preapril_fixed_epoch import (
    PreAprilFixedEpochPlan,
    local_target_window,
    preapril_fixed_epoch_template,
)
from scripts.run_preapril_fixed_epoch_mc import _checkpoint_from_training_result


def test_fixed_epoch_template_freezes_march_for_calibration_not_checkpoint_choice() -> None:
    reference = json.loads(
        Path(
            "runs/klga_train_default_20241001_20260604/"
            "artifacts/ab_experiments/hist-error-ab-a-seed42.json"
        ).read_text(encoding="ascii")
    )

    payload = preapril_fixed_epoch_template(reference)
    template = ExperimentTemplate.from_dict(payload)

    assert template.splits["train"] == ("2024-10-07", "2026-02-28")
    assert template.splits["val"] == ("2026-03-01", "2026-03-31")
    assert template.splits["test"] == ("2026-04-01", "2026-04-30")
    assert template.trainer.fixed_epochs == 3


def test_local_target_window_excludes_march_decoder_tail_for_april() -> None:
    frame = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(
                ["2026-04-01T04:00:00Z", "2026-04-01T05:00:00Z"], utc=True
            ),
            "value": [1, 2],
        }
    )

    march = local_target_window(
        frame,
        start="2026-03-01",
        end="2026-03-31",
        station_timezone="America/New_York",
    )
    april = local_target_window(
        frame,
        start="2026-04-01",
        end="2026-04-30",
        station_timezone="America/New_York",
    )

    assert march["value"].tolist() == [1]
    assert april["value"].tolist() == [2]


def test_checkpoint_loader_accepts_direct_and_wrapped_training_results() -> None:
    checkpoint = "/tmp/preapril.ckpt"

    assert _checkpoint_from_training_result({"checkpoint_path": checkpoint}) == Path(checkpoint)
    assert _checkpoint_from_training_result(
        {"train_result": {"checkpoint_path": checkpoint}}
    ) == Path(checkpoint)
