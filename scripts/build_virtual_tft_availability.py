"""Persist a fixed-latency virtual availability clock for offline TFT studies."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.forecast.availability import (
    HrrrAvailabilityDataset,
    build_virtual_tft_availability,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--latency", default="2h")
    parser.add_argument("--decision-tick", default="5min")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"immutable virtual availability output exists: {args.output}")
    start = pd.Timestamp(args.start, tz="UTC").normalize()
    end = pd.Timestamp(args.end, tz="UTC").normalize()
    if end < start:
        raise ValueError("--end must not precede --start")
    print(
        f"[start] virtual TFT availability range={args.start}..{args.end} "
        f"latency={args.latency} output={args.output}",
        flush=True,
    )
    cycles = build_virtual_tft_availability(
        init_times=pd.date_range(start, end + pd.Timedelta(hours=23), freq="h"),
        latency=args.latency,
        decision_tick=args.decision_tick,
    )
    dataset = HrrrAvailabilityDataset(
        objects=pd.DataFrame(columns=["key", "last_modified"]),
        cycles=cycles,
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
        compute_budget=str(args.latency),
        decision_tick=args.decision_tick,
        provider="virtual_tft_init_plus_fixed_latency",
    )
    manifest = dataset.save(args.output)
    print(
        f"[complete] virtual TFT availability cycles={len(cycles)} "
        f"output={args.output}",
        flush=True,
    )
    print(f"[done] virtual TFT availability output={args.output}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
