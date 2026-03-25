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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    from config.settings import Settings
    settings = Settings(trading_mode=args.mode)

    from core.bot import BotManager
    bot = BotManager(settings, db_path=args.db)

    print(f"[PolyWeatherBot] mode={args.mode}  station={args.station}")
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n[PolyWeatherBot] Shutdown by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
