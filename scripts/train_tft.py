"""CLI adapter for test-locked train/validation TFT execution."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from training.config import ExperimentTemplate, ResolvedTrialSnapshot
from training.trainer import train_workspace


def _materialize_experiment(
    *,
    run_dir: Path,
    version: str,
    source_path: Path,
    history_target: str | None,
) -> Path:
    if history_target is None:
        return source_path
    if Path(version).name != version:
        raise ValueError("version must be a single path-safe name")
    template = ExperimentTemplate.from_json(source_path)
    enabled = history_target == "enabled"
    materialized = replace(
        template,
        schema_version=2,
        template_id=f"{template.template_id}-history-{history_target}",
        model=replace(template.model, use_history_target=enabled),
    )
    target = run_dir / "artifacts" / "ab_experiments" / f"{version}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    materialized.write_json(target)
    return target


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
    parser.add_argument(
        "--history-target",
        choices=("enabled", "disabled"),
        help="materialize a schema-v2 A/B config with history input on or off",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    experiment_config = _materialize_experiment(
        run_dir=run_dir,
        version=args.version,
        source_path=Path(args.experiment_config),
        history_target=args.history_target,
    )
    trial_parameters = None
    if args.trial_snapshot:
        snapshot = ResolvedTrialSnapshot.from_dict(
            json.loads(
                Path(args.trial_snapshot).read_text(encoding="utf-8")
            )
        )
        trial_parameters = dict(snapshot.parameters)
    result = train_workspace(
        run_dir,
        station_id=args.station,
        version=args.version,
        experiment_config=experiment_config,
        trial_parameters=trial_parameters,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
