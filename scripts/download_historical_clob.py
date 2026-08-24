"""CLI adapter for immutable historical Polymarket market-data artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from config.stations import STATIONS
from data.clob.history import HistoricalClobSource

# Known daily-weather series ids (highest temperature) discovered via Gamma.
DAILY_WEATHER_SERIES: dict[str, str] = {
    "nyc": "10005",
    "london": "10006",
    "chicago": "10726",
    "dallas": "10727",
    "miami": "10728",
    "phoenix": "10729",
    "denver": "10730",
    "seattle": "10734",
    "boston": "10735",
    "atlanta": "10739",
    "tokyo": "10740",
    "shanghai": "10741",
    "seoul": "10742",
    "toronto": "10743",
}


def main(
    argv: Sequence[str] | None = None,
    *,
    source: HistoricalClobSource | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Download point-in-time Polymarket weather-market history"
    )
    parser.add_argument("--start", required=False)
    parser.add_argument("--end", required=False)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="resumable raw API cache (default: <output>.raw-cache)",
    )
    parser.add_argument(
        "--series",
        help=(
            "series key to download its FULL history by series_id "
            f"(options: {', '.join(sorted(DAILY_WEATHER_SERIES))})"
        ),
    )
    parser.add_argument(
        "--max-trade-offset",
        type=int,
        default=10_000,
        help="per-market trade offset cap (default 10000)",
    )
    parser.add_argument(
        "--min-interval-s",
        type=float,
        default=0.6,
        help="minimum seconds between data-api requests (default 0.6)",
    )
    parser.add_argument(
        "--skip-on-rate-limit",
        action="store_true",
        help="skip a market after persistent 429s instead of aborting the run",
    )
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(
            f"immutable historical CLOB output already exists: {args.output}"
        )
    cache_dir = args.cache_dir or Path(f"{args.output}.raw-cache")
    if args.series is not None:
        if args.series not in DAILY_WEATHER_SERIES:
            raise SystemExit(
                f"unknown series {args.series!r}; choose from "
                f"{sorted(DAILY_WEATHER_SERIES)}"
            )
        series_id = DAILY_WEATHER_SERIES[args.series]
        print(
            f"[start] historical CLOB series={args.series} id={series_id} "
            f"output={args.output} raw_cache={cache_dir}",
            flush=True,
        )
        downloader = source or HistoricalClobSource(
            cache_dir=cache_dir,
            max_trade_offset=args.max_trade_offset,
            min_interval_s=args.min_interval_s,
            skip_on_rate_limit=args.skip_on_rate_limit,
            progress_callback=lambda completed, total, day, markets, trades: print(
                f"[progress] {completed}/{total} {day} markets={markets} "
                f"trades={trades}",
                flush=True,
            ),
        )
        dataset = downloader.fetch_series(
            station_id=args.station,
            series_id=series_id,
        )
        print(
            f"[complete] historical CLOB series fetch events={len(dataset.events)} "
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
        print(f"[done] historical CLOB series artifact={args.output}", flush=True)
        return 0

    if not args.start or not args.end:
        parser.error("--start and --end are required unless --series is used")
    print(
        f"[start] historical CLOB download range={args.start}..{args.end} "
        f"output={args.output} raw_cache={cache_dir}",
        flush=True,
    )
    downloader = source or HistoricalClobSource(
        cache_dir=cache_dir,
        max_trade_offset=args.max_trade_offset,
        min_interval_s=args.min_interval_s,
        skip_on_rate_limit=args.skip_on_rate_limit,
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
