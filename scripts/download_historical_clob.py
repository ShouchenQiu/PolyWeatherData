"""CLI adapter for immutable historical Polymarket market-data artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from config.stations import STATIONS
from data.clob.history import HistoricalClobSource


def main(
    argv: Sequence[str] | None = None,
    *,
    source: HistoricalClobSource | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Download point-in-time Polymarket weather-market history"
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="resumable raw API cache (default: <output>.raw-cache)",
    )
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir or Path(f"{args.output}.raw-cache")
    downloader = source or HistoricalClobSource(
        cache_dir=cache_dir,
        progress_callback=lambda completed, total, day, markets, trades: print(
            f"[{completed}/{total}] {day} markets={markets} trades={trades}",
            flush=True,
        ),
    )
    dataset = downloader.fetch(
        start_date=args.start,
        end_date=args.end,
        station=STATIONS[args.station],
    )
    manifest = dataset.save(args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
