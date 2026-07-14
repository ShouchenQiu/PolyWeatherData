"""Build market-resolution daily highs from cached Weather.com truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from config.stations import STATIONS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-store", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        f"[start] market-resolution actual highs range={args.start}..{args.end} "
        f"output={args.output}",
        flush=True,
    )
    rows: list[dict] = []
    source_hasher = hashlib.sha256()
    for stamp in pd.date_range(args.start, args.end, freq="D"):
        target_date = stamp.date().isoformat()
        path = (
            args.truth_store
            / args.station
            / f"{target_date}.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"missing Weather.com truth for {target_date}: {path}"
            )
        payload = path.read_bytes()
        source_hasher.update(path.name.encode("ascii"))
        source_hasher.update(payload)
        frame = pd.read_parquet(path)
        if "weather_com_temp_f" not in frame:
            raise ValueError(f"truth file missing weather_com_temp_f: {path}")
        temperatures = pd.to_numeric(
            frame["weather_com_temp_f"],
            errors="coerce",
        ).dropna()
        if temperatures.empty:
            raise ValueError(f"truth file has no temperatures: {path}")
        rows.append(
            {
                "target_date": target_date,
                "actual_high_f": float(temperatures.max()),
            }
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    manifest = {
        "schema_version": 1,
        "data_class": "market_resolution_actual_highs",
        "station_id": args.station,
        "requested_range": [args.start, args.end],
        "row_count": len(rows),
        "source_sha256": source_hasher.hexdigest(),
        "sha256": {
            args.output.name: hashlib.sha256(
                args.output.read_bytes()
            ).hexdigest()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[complete] market-resolution actual highs rows={len(rows)} "
        f"output={args.output}",
        flush=True,
    )
    print(f"[done] market-resolution actual highs output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
