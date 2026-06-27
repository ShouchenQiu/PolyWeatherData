"""CLI adapter for lower-trust lab training-input export."""
from __future__ import annotations

import argparse
import json

from training.package import export_lab_input_package


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Export audited training inputs")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = export_lab_input_package(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    _cli()
