"""CLI adapter for test-locked train/validation TFT execution."""
from __future__ import annotations

import argparse
import json

from training.trainer import train_workspace


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run audited TFT train/validation without reading test"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--version", required=True)
    parser.add_argument("--experiment-config", required=True)
    args = parser.parse_args()

    result = train_workspace(
        args.run_dir,
        station_id=args.station,
        version=args.version,
        experiment_config=args.experiment_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
