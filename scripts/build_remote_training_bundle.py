"""CLI adapter for lower-trust training bundle construction."""
from __future__ import annotations

import argparse
import json

from training.bundle import build_training_bundle


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build a minimal training-factory source bundle"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_training_bundle(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
