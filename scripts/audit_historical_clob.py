"""Audit a historical CLOB artifact and persist its eligibility contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data.clob.history import HistoricalClobDataset
from data.clob.quality import audit_historical_clob


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="DATE=REASON",
        help="explicitly accept one unavailable event-day",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"immutable CLOB audit output exists: {args.output}")
    print(
        f"[start] historical CLOB audit clob={args.clob_root} output={args.output}",
        flush=True,
    )
    exclusions = _parse_exclusions(args.exclude)
    manifest_path = args.clob_root / "manifest.json"
    dataset = HistoricalClobDataset.load(args.clob_root)
    report = audit_historical_clob(dataset, exclusions=exclusions)
    report["source_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"[complete] historical CLOB audit status={report.get('status')} "
        f"output={args.output}",
        flush=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"[done] historical CLOB audit output={args.output}", flush=True)
    return 0


def _parse_exclusions(values: list[str]) -> dict[str, str]:
    exclusions: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--exclude must use DATE=REASON")
        day, reason = value.split("=", 1)
        if day in exclusions:
            raise ValueError(f"duplicate excluded date: {day}")
        exclusions[day] = reason
    return exclusions


if __name__ == "__main__":
    raise SystemExit(main())
