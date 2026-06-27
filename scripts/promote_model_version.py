"""CLI adapter for explicit production model promotion."""
from __future__ import annotations

import argparse
import json

from model.store.operations import promote_model_version


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Promote a reviewed model version")
    parser.add_argument("--model-store", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    result = promote_model_version(args.model_store, args.db, args.version)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
