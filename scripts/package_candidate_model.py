"""CLI adapter for candidate-model packaging."""
from __future__ import annotations

import argparse
import json

from training.package import package_workspace


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Package a candidate model")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--include-weather-data", action="store_true")
    args = parser.parse_args()
    result = package_workspace(
        args.run_dir,
        args.version,
        include_weather_data=args.include_weather_data,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
