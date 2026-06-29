"""CLI adapter for the persistent training experiment catalog."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from training.catalog import TrainingCatalog
from training.config import ResolvedTrialSnapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage training studies")
    parser.add_argument(
        "--catalog",
        default="runs/_management/training_catalog.db",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync")
    sync.add_argument("--workspace", action="append", default=[])
    sync.add_argument("--template", action="append", default=[])
    sync.add_argument("--run-root", default="runs")
    sync.add_argument("--experiments-root", default="experiments")

    create = subparsers.add_parser("create-study")
    create.add_argument("--workspace", required=True)
    create.add_argument("--template", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--study")

    approve = subparsers.add_parser("approve-final")
    approve.add_argument("--study", required=True)
    approve.add_argument("--trial-snapshot", required=True)
    approve.add_argument("--approved-by", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with TrainingCatalog(args.catalog) as catalog:
        if args.command == "create-study":
            payload: Any = asdict(
                catalog.create_study(args.workspace, args.template)
            )
        elif args.command == "status":
            payload = (
                asdict(catalog.get_study(args.study))
                if args.study
                else [asdict(study) for study in catalog.list_studies()]
            )
        elif args.command == "approve-final":
            snapshot = ResolvedTrialSnapshot.from_dict(
                json.loads(
                    Path(args.trial_snapshot).read_text(encoding="ascii")
                )
            )
            payload = asdict(
                catalog.approve_final(
                    args.study,
                    snapshot,
                    approved_by=args.approved_by,
                )
            )
        else:
            payload = _sync(catalog, args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _sync(catalog: TrainingCatalog, args: argparse.Namespace) -> dict[str, Any]:
    workspaces = [Path(path) for path in args.workspace]
    templates = [Path(path) for path in args.template]
    if not workspaces:
        root = Path(args.run_root)
        workspaces = sorted(
            path.parent
            for path in root.glob("*/audit_summary.json")
            if path.parent.name != "_management"
        )
    if not templates:
        templates = sorted(Path(args.experiments_root).glob("**/*.json"))
    return {
        "workspaces": [
            asdict(catalog.sync_workspace(path))
            for path in workspaces
        ],
        "templates": [
            _template_payload(catalog.sync_template(path))
            for path in templates
        ],
    }


def _template_payload(record: Any) -> dict[str, Any]:
    payload = asdict(record)
    payload.pop("config_json", None)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
