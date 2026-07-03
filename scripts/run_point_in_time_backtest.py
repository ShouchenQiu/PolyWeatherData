"""Run one deterministic strategy against immutable replay inputs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.replay import PointInTimeReplay, ReplayStrategy
from backtest.reporting import ReplayArtifact, build_replay_report
from backtest.synthetic_quotes import SyntheticQuoteDataset
from data.clob.history import HistoricalClobDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-root", required=True, type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        choices=["baseline", "near_term", "liquid"],
        default="baseline",
    )
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--start-target")
    parser.add_argument("--end-target")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    forecast = ForecastCache.load(args.forecast_root)
    quotes = SyntheticQuoteDataset.load(args.quotes_root)
    clob = HistoricalClobDataset.load(args.clob_root)
    actual_frame = (
        pd.read_parquet(args.actual_highs)
        if args.actual_highs.suffix == ".parquet"
        else pd.read_csv(args.actual_highs)
    )
    required_actual = {"target_date", "actual_high_f"}
    if not required_actual.issubset(actual_frame.columns):
        raise ValueError(
            "actual-highs file must contain target_date and actual_high_f"
        )
    actual_highs = {
        pd.Timestamp(row.target_date).date().isoformat(): float(
            row.actual_high_f
        )
        for row in actual_frame.itertuples(index=False)
    }
    events = clob.events[
        ["event_id", "target_date", "event_end_at"]
    ].copy()
    if bool(args.start_target) != bool(args.end_target):
        raise ValueError(
            "--start-target and --end-target must be provided together"
        )
    if args.start_target:
        target_dates = pd.to_datetime(events["target_date"]).dt.date
        start_target = pd.Timestamp(args.start_target).date()
        end_target = pd.Timestamp(args.end_target).date()
        events = events[
            (target_dates >= start_target)
            & (target_dates <= end_target)
        ].copy()
    if events.empty:
        raise ValueError("no CLOB events remain in requested target range")
    markets = clob.markets.merge(
        events,
        on="event_id",
        how="inner",
        validate="many_to_one",
    )
    scheduled_end = (
        markets["market_end_at"]
        if "market_end_at" in markets
        else pd.Series(pd.NaT, index=markets.index)
    )
    resolved_at = (
        markets["closed_at"]
        if "closed_at" in markets
        else pd.Series(pd.NaT, index=markets.index)
    )
    markets["market_end_at"] = (
        pd.to_datetime(resolved_at, utc=True, errors="coerce")
        .fillna(pd.to_datetime(scheduled_end, utc=True, errors="coerce"))
        .fillna(pd.to_datetime(markets["event_end_at"], utc=True))
    )
    quotes_frame = quotes.quotes[
        quotes.quotes["condition_id"].isin(markets["condition_id"])
    ].copy()
    strategy = {
        "baseline": ReplayStrategy.baseline,
        "near_term": ReplayStrategy.near_term,
        "liquid": ReplayStrategy.liquid,
    }[args.strategy]()
    replay = PointInTimeReplay(
        forecast_cache=forecast,
        quotes=quotes_frame,
        markets=markets,
        station_id=args.station,
        strategy=strategy,
    )
    result = replay.run(actual_highs=actual_highs)
    ReplayArtifact(result=result, strategy=strategy.name).save(
        args.output,
        source_sha256={
            "forecast_manifest": _sha256(
                args.forecast_root / "manifest.json"
            ),
            "quotes_manifest": _sha256(
                args.quotes_root / "manifest.json"
            ),
            "clob_manifest": _sha256(args.clob_root / "manifest.json"),
            "actual_highs": _sha256(args.actual_highs),
        },
    )
    print(build_replay_report(result).summary, flush=True)
    return 0


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
