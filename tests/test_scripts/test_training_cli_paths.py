from __future__ import annotations

import sys

import pytest

from scripts import audit_training_data, batch_prepare_training_data


def test_batch_prepare_requires_explicit_run_dir(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["batch_prepare_training_data"])

    with pytest.raises(SystemExit) as exc:
        batch_prepare_training_data._cli()

    assert exc.value.code == 2


def test_audit_requires_explicit_run_or_data_store(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_training_data",
            "--start",
            "2026-01-01",
            "--end",
            "2026-01-01",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        audit_training_data._cli()

    assert exc.value.code == 2


def test_readiness_audit_cli_enables_streaming_progress(monkeypatch, capsys):
    captured = {}

    def fake_audit_workspace(run_dir, station, split_ranges, **kwargs):
        captured.update(kwargs)
        return {"audit": {"ready": True, "failures": []}}

    monkeypatch.setattr(
        audit_training_data,
        "audit_workspace",
        fake_audit_workspace,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_training_data",
            "--run-dir",
            "runs/example",
            "--train",
            "2026-01-01:2026-01-01",
            "--val",
            "2026-01-02:2026-01-02",
            "--test",
            "2026-01-03:2026-01-03",
            "--progress-every",
            "25",
        ],
    )

    audit_training_data._cli()

    assert captured == {"progress": True, "progress_every": 25}
    assert '"ready": true' in capsys.readouterr().out
