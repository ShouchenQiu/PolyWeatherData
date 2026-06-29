"""Run the single catalog-approved final and produce an inactive candidate."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from training.catalog import TrainingCatalog
from training.trial_runner import run_final_execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="runs/_management/training_catalog.db",
    )
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    with TrainingCatalog(args.catalog) as catalog:
        result = run_final_execution(catalog, args.study)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
