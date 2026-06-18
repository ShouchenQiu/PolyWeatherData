"""Compatibility wrapper for :mod:`scripts.audit_training_data`.

Deprecated: use ``python -m scripts.audit_training_data`` for new commands.
"""
from __future__ import annotations

from scripts.audit_training_data import audit_training_readiness
from scripts.audit_training_data import audit_weather_backfill
from scripts.audit_training_data import _cli as _new_cli

__all__ = ["audit_weather_backfill", "audit_training_readiness"]


def _cli() -> None:
    _new_cli()


if __name__ == "__main__":
    _cli()
