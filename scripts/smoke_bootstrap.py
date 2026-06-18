"""Compatibility wrapper for :mod:`scripts.batch_prepare_training_data`.

Deprecated: use ``python -m scripts.batch_prepare_training_data`` for new
batch training-data preparation commands.
"""
from __future__ import annotations

from scripts.batch_prepare_training_data import BatchHrrrS3F01Downloader
from scripts.batch_prepare_training_data import run_batch_prepare_training_data
from scripts.batch_prepare_training_data import run_smoke_bootstrap
from scripts.batch_prepare_training_data import _cli as _new_cli

__all__ = [
    "BatchHrrrS3F01Downloader",
    "run_batch_prepare_training_data",
    "run_smoke_bootstrap",
]


def _cli() -> None:
    _new_cli()


if __name__ == "__main__":
    _cli()
