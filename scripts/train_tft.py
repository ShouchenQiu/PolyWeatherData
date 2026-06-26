"""CLI adapter for TFT training.

The training implementation lives in :mod:`model.training`; this module keeps
the historical command and import path stable.
"""
from __future__ import annotations

from model.training.trainer import (
    TftTrainer,
    _KNOWN_REALS,
    _UNKNOWN_REALS,
    _cli,
    _settings_from_cli_args,
)

__all__ = [
    "TftTrainer",
    "_KNOWN_REALS",
    "_UNKNOWN_REALS",
    "_cli",
    "_settings_from_cli_args",
]


if __name__ == "__main__":
    _cli()
