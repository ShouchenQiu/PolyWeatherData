"""Centralised logging configuration for PolyWeatherBot.

Usage
-----
Call ``setup_logging()`` once at startup (in main.py).  Every other module
simply does::

    from loguru import logger
    logger.info("...")

``setup_logging()`` wires two sinks:

* **stderr** -- coloured, human-readable, INFO+ by default.
* **logs/polyweather_{date}.log** -- JSON-formatted, DEBUG+, rotated daily,
  retained for 14 days, compressed with gzip.

The log directory is created automatically if it does not exist.
"""
from __future__ import annotations

import sys
from pathlib import Path


def setup_logging(
    log_dir: Path | None = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> None:
    """Configure loguru sinks.

    Parameters
    ----------
    log_dir:
        Directory for log files.  Defaults to ``<repo_root>/logs``.
    console_level:
        Minimum level printed to stderr (e.g. "DEBUG", "INFO", "WARNING").
    file_level:
        Minimum level written to the log file.
    """
    from loguru import logger

    # Remove the default loguru sink so we control the exact format.
    logger.remove()

    # --- stderr sink (human-readable) ---
    logger.add(
        sys.stderr,
        level=console_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=True,
    )

    # --- file sink (JSON, rotated daily) ---
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "polyweather_{time:YYYY-MM-DD}.log",
        level=file_level,
        rotation="00:00",        # new file each day at midnight
        retention="14 days",
        compression="gz",
        serialize=True,          # JSON-formatted entries
        backtrace=True,
        diagnose=False,          # omit locals in file (security)
        encoding="utf-8",
    )

    logger.info("Logging initialised: console={} file={} dir={}",
                console_level, file_level, log_dir)
