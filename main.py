"""CLI entry point for PolyWeatherBot.

Usage
-----
    python3 main.py --mode paper --station KLGA
    python3 main.py --mode live  --station KLGA
"""
from __future__ import annotations

import argparse
import asyncio
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "polyweatherbot",
        description = "PolyWeatherBot -- weather derivatives trading bot",
    )
    parser.add_argument(
        "--mode",
        choices = ["paper", "live"],
        default = "paper",
        help    = "Trading mode: 'paper' (simulated) or 'live' (real orders).",
    )
    parser.add_argument(
        "--station",
        default = "KLGA",
        help    = "ICAO station identifier to trade (default: KLGA).",
    )
    parser.add_argument(
        "--db",
        default = None,
        help    = "Override path to the SQLite database file.",
    )
    parser.add_argument(
        "--log-level",
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
        help    = "Console log level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from config.log import setup_logging
    setup_logging(console_level=args.log_level)

    from loguru import logger

    from config.settings import Settings
    settings = Settings(trading_mode=args.mode)

    logger.info("PolyWeatherBot starting: mode={} station={}", args.mode, args.station)

    from core.bot import BotManager
    bot = BotManager(settings, db_path=args.db)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
        sys.exit(0)
    except Exception:
        logger.exception("Fatal error in BotManager.run()")
        sys.exit(1)


if __name__ == "__main__":
    main()
