"""Build managed hourly observations with conservative visibility times."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from backtest.observations import PointInTimeObservationDataset
from config.stations import STATIONS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--start")
    parser.add_argument("--end")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    frame = pd.concat(
        [pd.read_parquet(path) for path in args.input],
        ignore_index=True,
    )
    required = {"valid_time", "raw_tmpc_tenths"}
    missing = required - set(frame)
    if missing:
        raise ValueError(
            "observation inputs missing columns: "
            + ", ".join(sorted(missing))
        )
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    if args.start:
        frame = frame[frame["valid_time"] >= _utc(args.start)]
    if args.end:
        frame = frame[
            frame["valid_time"] < _utc(args.end) + pd.Timedelta(days=1)
        ]
    observations = pd.DataFrame(
        {
            "valid_time": frame["valid_time"],
            # The H-1 :51 METAR is rounded to the following whole hour.
            "available_at": frame["valid_time"],
            "temperature_c": pd.to_numeric(
                frame["raw_tmpc_tenths"], errors="raise"
            ),
        }
    ).drop_duplicates("valid_time", keep="last")
    station = STATIONS[args.station]
    dataset = PointInTimeObservationDataset(
        observations=observations,
        station_id=station.station_id,
        station_timezone=station.timezone,
    )
    manifest = dataset.save(
        args.output,
        source_sha256={
            f"input_{index}": _sha256(path)
            for index, path in enumerate(args.input)
        },
    )
    print(
        f"Saved {manifest['row_count']} point-in-time observations "
        f"to {args.output}",
        flush=True,
    )
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return (
        stamp.tz_localize("UTC")
        if stamp.tzinfo is None
        else stamp.tz_convert("UTC")
    )


if __name__ == "__main__":
    raise SystemExit(main())
