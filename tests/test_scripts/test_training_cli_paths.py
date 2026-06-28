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
