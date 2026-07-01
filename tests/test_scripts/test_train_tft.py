from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from training.config import ExperimentTemplate
from scripts import train_tft


def test_trial_snapshot_cli_marks_execution_as_trial(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.json"
    ExperimentTemplate.from_json(
        "experiments/klga/tft_trend_6h_v2.json"
    ).write_json(source)
    snapshot = tmp_path / "trial_snapshot.json"
    snapshot.write_text(json.dumps({"placeholder": True}), encoding="utf-8")
    parameters = {
        "attention_head_size": 4,
        "batch_size": 256,
        "dropout": 0.3,
        "gradient_clip_val": 0.5,
        "hidden_continuous_size": 32,
        "hidden_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0004,
    }
    captured = {}

    monkeypatch.setattr(
        train_tft.ResolvedTrialSnapshot,
        "from_dict",
        lambda _payload: SimpleNamespace(parameters=parameters),
    )
    monkeypatch.setattr(
        train_tft,
        "train_workspace",
        lambda *args, **kwargs: captured.update(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tft",
            "--run-dir",
            str(tmp_path),
            "--station",
            "KLGA",
            "--version",
            "ab-a",
            "--experiment-config",
            str(source),
            "--trial-snapshot",
            str(snapshot),
        ],
    )

    train_tft._cli()

    assert captured["trial_parameters"] == parameters
    assert captured["limits"].execution_type == "trial"
    assert captured["limits"].batch_size == 256
