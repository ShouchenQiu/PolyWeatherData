"""CLI adapter for monthly HRRR decoder baseline verification."""
from __future__ import annotations

import argparse
import json

from config.stations import STATIONS
from training.decoder_baseline import verify_decoder_baseline


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Verify monthly HRRR decoder forecasts against IEM observations"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--month", help="single UTC init month in YYYY-MM form")
    args = parser.parse_args()
    result = verify_decoder_baseline(
        args.run_dir,
        args.station,
        month=args.month,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
