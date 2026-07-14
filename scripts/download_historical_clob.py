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

    if args.output.exists():
        raise FileExistsError(
            f"immutable historical CLOB output already exists: {args.output}"
        )
    cache_dir = args.cache_dir or Path(f"{args.output}.raw-cache")
    print(
        f"[start] historical CLOB download range={args.start}..{args.end} "
        f"output={args.output} raw_cache={cache_dir}",
        flush=True,
    )
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
    print(
        f"[complete] historical CLOB fetch events={len(dataset.events)} "
        f"markets={len(dataset.markets)} trades={len(dataset.trades)}",
        flush=True,
    )
    manifest = dataset.save(args.output)
    print(
        f"[complete] immutable artifact coverage_days="
        f"{len(dataset.coverage())} output={args.output}",
        flush=True,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print(f"[done] historical CLOB artifact={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
