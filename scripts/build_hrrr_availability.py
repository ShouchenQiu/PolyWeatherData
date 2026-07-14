"""Build immutable point-in-time HRRR availability evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.forecast.availability import (
    HrrrAvailabilityDataset,
    HrrrS3ObjectSource,
    build_hrrr_availability,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record HRRR S3 Last-Modified timestamps by forecast cycle."
    )
    parser.add_argument("--start", required=True, help="First init date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Last init date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Resumable daily listing cache (default: <output>.raw-cache)",
    )
    parser.add_argument("--compute-budget", default="5min")
    parser.add_argument("--decision-tick", default="5min")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    source: HrrrS3ObjectSource | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"immutable HRRR availability output exists: {args.output}")
    start = pd.Timestamp(args.start, tz="UTC").normalize()
    end = pd.Timestamp(args.end, tz="UTC").normalize()
    if end < start:
        raise ValueError("--end must not precede --start")

    cache_dir = args.cache_dir or Path(f"{args.output}.raw-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[start] HRRR availability range={args.start}..{args.end} "
        f"output={args.output} raw_cache={cache_dir}",
        flush=True,
    )
    object_source = source or HrrrS3ObjectSource()
    frames: list[pd.DataFrame] = []
    listing_days = pd.date_range(start - pd.Timedelta(days=1), end, freq="D")
    for index, day in enumerate(listing_days, start=1):
        cache_path = cache_dir / f"{day:%Y%m%d}.parquet"
        if cache_path.exists():
            frame = pd.read_parquet(cache_path)
            status = "cached"
        else:
            frame = object_source.fetch_day(day.strftime("%Y-%m-%d"))
            frame.to_parquet(cache_path, index=False)
            status = "fetched"
        frames.append(frame)
        print(
            f"[{index}/{len(listing_days)}] {day:%Y-%m-%d}: "
            f"{len(frame)} objects ({status})",
            flush=True,
        )

    objects = pd.concat(frames, ignore_index=True)
    init_times = pd.date_range(
        start,
        end + pd.Timedelta(hours=23),
        freq="h",
    )
    cycles = build_hrrr_availability(
        objects,
        init_times=init_times,
        compute_budget=args.compute_budget,
        decision_tick=args.decision_tick,
    )
    dataset = HrrrAvailabilityDataset(
        objects=objects,
        cycles=cycles,
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
        compute_budget=args.compute_budget,
        decision_tick=args.decision_tick,
    )
    manifest = dataset.save(args.output)
    complete = int(cycles["complete"].sum())
    print(
        f"[complete] HRRR availability cycles={len(cycles)} complete={complete} "
        f"output={args.output}",
        flush=True,
    )
    print(f"[done] HRRR availability output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
