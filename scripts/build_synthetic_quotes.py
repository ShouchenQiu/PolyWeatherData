"""Build past-only 5-minute executable quote proxies from CLOB trades."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from backtest.synthetic_quotes import (
    SyntheticQuoteDataset,
    build_synthetic_quotes,
)
from data.clob.history import HistoricalClobDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build immutable past-only synthetic CLOB quotes."
    )
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--decision-tick", default="5min")
    parser.add_argument("--lookback", default="60min")
    parser.add_argument("--max-staleness", default="30min")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"immutable synthetic quote output exists: {args.output}")
    print(
        f"[start] synthetic quotes clob={args.clob_root} output={args.output}",
        flush=True,
    )
    clob = HistoricalClobDataset.load(args.clob_root)
    if clob.trades.empty:
        raise ValueError("historical CLOB artifact has no trades")
    timestamps = pd.to_datetime(clob.trades["timestamp"], utc=True)
    start = args.start or timestamps.min().floor(args.decision_tick).isoformat()
    end = args.end or timestamps.max().ceil(args.decision_tick).isoformat()
    tick_sizes = {
        str(row.condition_id): (
            float(row.min_tick_size)
            if pd.notna(row.min_tick_size) and float(row.min_tick_size) > 0
            else 0.01
        )
        for row in clob.markets.itertuples(index=False)
    }
    quotes = build_synthetic_quotes(
        clob.trades,
        start=start,
        end=end,
        decision_tick=args.decision_tick,
        lookback=args.lookback,
        max_staleness=args.max_staleness,
        min_tick_sizes=tick_sizes,
        progress_callback=lambda completed, total, _condition, rows: (
            print(
                f"[progress] synthetic quotes conditions={completed}/{total} rows={rows}",
                flush=True,
            )
            if completed == total or completed % 25 == 0
            else None
        ),
    )
    source_sha = hashlib.sha256(
        (args.clob_root / "manifest.json").read_bytes()
    ).hexdigest()
    manifest = SyntheticQuoteDataset(
        quotes=quotes,
        start_at=str(start),
        end_at=str(end),
        decision_tick=args.decision_tick,
        lookback=args.lookback,
        max_staleness=args.max_staleness,
    ).save(args.output, source_sha256=source_sha)
    print(
        f"[complete] synthetic quotes rows={manifest['row_count']} "
        f"conditions={manifest['condition_count']} output={args.output}",
        flush=True,
    )
    print(f"[done] synthetic quotes output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
