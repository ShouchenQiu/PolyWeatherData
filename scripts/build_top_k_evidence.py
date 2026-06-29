"""Build frozen validation-only evidence for the best completed trials."""
from __future__ import annotations

import argparse
import json

from training.catalog import TrainingCatalog
from training.selection import TopKEvidenceBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="runs/_management/training_catalog.db",
    )
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    with TrainingCatalog(args.catalog) as catalog:
        payload = TopKEvidenceBuilder(catalog).build(args.study)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
