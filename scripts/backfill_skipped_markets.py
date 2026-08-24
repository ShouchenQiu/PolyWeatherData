"""Backfill trades for markets skipped by --skip-on-rate-limit.

Merges newly fetched trades into an existing immutable artifact and updates
the integrity manifest (sha256 + row_counts) so HistoricalClobDataset.load
still verifies cleanly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from data.clob.history import HistoricalClobDataset, HistoricalClobSource


def parse_skipped_markets(log_path: Path) -> list[str]:
    """Extract condition ids logged by the skip-on-rate-limit warning.

    The log lines wrap mid-address, so whitespace is removed before matching.
    """
    text = log_path.read_text(encoding="utf-16")
    joined = re.sub(r"\s+", "", text)
    matches = re.findall(r"skipcondition(0x[0-9a-fA-F]{64})", joined)
    seen: set[str] = set()
    unique: list[str] = []
    for match in matches:
        if match not in seen:
            seen.add(match)
            unique.append(match)
    return unique


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill trades skipped during a historical download"
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="immutable artifact directory to merge trades into",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="download log to parse skipped condition ids from (utf-16)",
    )
    parser.add_argument(
        "--market",
        action="append",
        dest="markets",
        help="condition id to backfill (repeatable; overrides --log parsing)",
    )
    parser.add_argument(
        "--min-interval-s",
        type=float,
        default=1.2,
        help="minimum seconds between data-api requests (default 1.2)",
    )
    parser.add_argument(
        "--max-trade-offset",
        type=int,
        default=50_000,
        help="per-market trade offset cap (default 50000)",
    )
    args = parser.parse_args(argv)

    if args.markets:
        condition_ids = list(dict.fromkeys(args.markets))
    else:
        if args.log is None:
            parser.error("either --market or --log is required")
        condition_ids = parse_skipped_markets(args.log)
    if not condition_ids:
        print("[done] no skipped markets to backfill", flush=True)
        return 0

    # Verify the artifact is intact before modifying it.
    dataset = HistoricalClobDataset.load(args.artifact)
    cache_dir = Path(f"{args.artifact}.raw-cache")
    source = HistoricalClobSource(
        cache_dir=cache_dir,
        max_trade_offset=args.max_trade_offset,
        min_interval_s=args.min_interval_s,
        skip_on_rate_limit=False,
        progress_callback=lambda done, total, day, markets, trades: print(
            f"[progress] {done}/{total} markets={markets} trades={trades}",
            flush=True,
        ),
    )

    new_rows: list[dict] = []
    backfilled: list[str] = []
    for i, condition_id in enumerate(condition_ids, start=1):
        cached = source._read_cache("trades", condition_id)
        if isinstance(cached, dict) and cached.get("complete") is True:
            print(
                f"[skip] {condition_id} already complete in cache "
                f"({len(cached.get('rows', []))} rows)",
                flush=True,
            )
            continue
        print(f"[start] {condition_id} ({i}/{len(condition_ids)})", flush=True)
        rows = source._fetch_trades(condition_id)
        print(
            f"[complete] {condition_id} fetched={len(rows)}",
            flush=True,
        )
        new_rows.extend(rows)
        backfilled.append(condition_id)

    if not new_rows:
        print("[done] no new trades fetched", flush=True)
        return 0

    # Merge into the existing trades frame.
    new_frame = source._trade_frame(new_rows)
    existing = dataset.trades
    merged = (
        pd.concat([existing, new_frame], ignore_index=True)
        .drop_duplicates()
        .sort_values(
            ["timestamp", "condition_id", "token_id", "transaction_hash"]
        )
        .reset_index(drop=True)
    )
    added = len(merged) - len(existing)
    trades_path = args.artifact / "trades.parquet"
    merged.to_parquet(trades_path, index=False)
    trades_hash = hashlib.sha256(trades_path.read_bytes()).hexdigest()

    manifest_path = args.artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"]["trades.parquet"] = trades_hash
    manifest["row_counts"]["trades"] = int(len(merged))
    manifest["backfilled_at"] = pd.Timestamp.now(tz="UTC").isoformat().replace(
        "+00:00", "Z"
    )
    manifest["backfilled_markets"] = backfilled
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[done] backfilled markets={len(backfilled)} added_trades={added} "
        f"total_trades={len(merged)} artifact={args.artifact}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
