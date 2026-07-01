"""CLI adapter for test-locked train/validation TFT execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.config import ExperimentTemplate, ResolvedTrialSnapshot
from training.experiment import ValidationExecutionLimits
from training.trainer import train_workspace


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run audited TFT train/validation without reading test"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--version", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--trial-snapshot",
        help="reuse one resolved trial's exact hyperparameters",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    experiment_config = Path(args.experiment_config)
    trial_parameters = None
    limits = None
    if args.trial_snapshot:
        snapshot = ResolvedTrialSnapshot.from_dict(
            json.loads(
                Path(args.trial_snapshot).read_text(encoding="utf-8")
            )
        )
        trial_parameters = dict(snapshot.parameters)
        limits = ValidationExecutionLimits.for_trial(
            ExperimentTemplate.from_json(experiment_config),
            trial_parameters,
        )
    result = train_workspace(
        run_dir,
        station_id=args.station,
        version=args.version,
        experiment_config=experiment_config,
        trial_parameters=trial_parameters,
        limits=limits,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
