"""Build market-resolution daily highs from cached Weather.com truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from config.stations import STATIONS
from data.obs.weather_com import local_day_hour_coverage


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
    exclusion_rows: list[dict] = []
    source_hasher = hashlib.sha256()
    requested_days = pd.date_range(args.start, args.end, freq="D")
    for stamp in requested_days:
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
        observed_hours, expected_hours = local_day_hour_coverage(
            frame,
            local_date=target_date,
            timezone=STATIONS[args.station].timezone,
        )
        if observed_hours != expected_hours:
            exclusion_rows.append(
                {
                    "station_id": args.station,
                    "target_date": target_date,
                    "observed_hours": observed_hours,
                    "expected_hours": expected_hours,
                    "exclusion_reason": "incomplete_station_local_day",
                }
            )
            print(
                f"[complete] truth exclusion date={target_date} "
                f"observed_hours={observed_hours} expected_hours={expected_hours}",
                flush=True,
            )
            continue
        required_truth_columns = {
            "weather_com_temp_f",
            "weather_com_raw_daily_high_f",
        }
        missing_truth_columns = required_truth_columns - set(frame)
        if missing_truth_columns:
            raise ValueError(
                "truth file missing resolution daily-high columns "
                f"{sorted(missing_truth_columns)}: {path}"
            )
        daily_highs = pd.to_numeric(
            frame["weather_com_raw_daily_high_f"],
            errors="coerce",
        ).dropna()
        if daily_highs.empty or daily_highs.nunique() != 1:
            raise ValueError(
                "truth file must contain exactly one raw daily high "
                f"for {target_date}: {path}"
            )
        rows.append(
            {
                "target_date": target_date,
                "actual_high_f": float(daily_highs.iloc[0]),
            }
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    exclusions_path = args.output.with_suffix(".truth_exclusions.parquet")
    if exclusions_path.exists():
        raise FileExistsError(exclusions_path)
    manifest_path = args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["target_date", "actual_high_f"]).to_csv(
        args.output,
        index=False,
    )
    pd.DataFrame(
        exclusion_rows,
        columns=[
            "station_id",
            "target_date",
            "observed_hours",
            "expected_hours",
            "exclusion_reason",
        ],
    ).to_parquet(exclusions_path, index=False)
    manifest = {
        "schema_version": 2,
        "data_class": "market_resolution_actual_highs",
        "daily_high_source": "weather_com_raw_observation_max",
        "hour_coverage_contract": (
            "complete_station_local_day_or_explicit_exclusion"
        ),
        "station_id": args.station,
        "requested_range": [args.start, args.end],
        "requested_day_count": len(requested_days),
        "row_count": len(rows),
        "exclusion_count": len(exclusion_rows),
        "truth_exclusions_file": exclusions_path.name,
        "source_sha256": source_hasher.hexdigest(),
        "sha256": {
            args.output.name: hashlib.sha256(
                args.output.read_bytes()
            ).hexdigest(),
            exclusions_path.name: hashlib.sha256(
                exclusions_path.read_bytes()
            ).hexdigest(),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[complete] market-resolution actual highs rows={len(rows)} "
        f"exclusions={len(exclusion_rows)} output={args.output}",
        flush=True,
    )
    print(f"[done] market-resolution actual highs output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
