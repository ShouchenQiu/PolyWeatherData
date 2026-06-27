"""CLI adapter for audited candidate TFT training."""
from __future__ import annotations

import argparse
import json

from training.trainer import train_workspace


def _split(value: str) -> tuple[str, str]:
    start, end = value.split(":", 1)
    return start, end


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train a candidate TFT model")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--version", required=True)
    parser.add_argument("--train", help="start:end")
    parser.add_argument("--val", help="start:end")
    parser.add_argument("--test", help="start:end")
    parser.add_argument("--experiment-config")
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--precision", default="32")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    split_ranges = None
    if all((args.train, args.val, args.test)):
        split_ranges = {
            "train": _split(args.train),
            "val": _split(args.val),
            "test": _split(args.test),
        }
    elif not args.experiment_config:
        parser.error("provide --experiment-config or all split ranges")

    result = train_workspace(
        args.run_dir,
        station_id=args.station,
        version=args.version,
        split_ranges=split_ranges,
        experiment_config=args.experiment_config,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        deterministic=args.deterministic,
        precision=args.precision,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
