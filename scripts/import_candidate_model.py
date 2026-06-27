"""CLI adapter for candidate-model import."""
from __future__ import annotations

import argparse
import json

from model.store.operations import import_candidate_model


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Import a candidate model package")
    parser.add_argument("--package", required=True, dest="package_path")
    parser.add_argument("--model-store", required=True)
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    result = import_candidate_model(args.package_path, args.model_store, args.db)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
