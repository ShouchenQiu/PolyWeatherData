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

    enqueue_smoke = subparsers.add_parser("enqueue-smoke")
    enqueue_smoke.add_argument("--study", required=True)

    enqueue_calibration = subparsers.add_parser("enqueue-calibration")
    enqueue_calibration.add_argument("--study", required=True)

    retry = subparsers.add_parser("retry")
    retry.add_argument("--execution", required=True)

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("--execution", required=True)

    approve_search = subparsers.add_parser("approve-search")
    approve_search.add_argument("--study", required=True)
    approve_search.add_argument("--approved-by", required=True)

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
        elif args.command == "enqueue-smoke":
            payload = asdict(catalog.enqueue_smoke(args.study))
        elif args.command == "enqueue-calibration":
            payload = asdict(catalog.enqueue_calibration(args.study))
        elif args.command == "retry":
            payload = asdict(catalog.retry_execution(args.execution))
        elif args.command == "cancel":
            payload = asdict(catalog.request_cancel(args.execution))
        elif args.command == "approve-search":
            payload = asdict(
                catalog.approve_search(
                    args.study,
                    approved_by=args.approved_by,
                )
            )
        elif args.command == "status":
            payload = (
                {
                    "study": asdict(catalog.get_study(args.study)),
                    "executions": [
                        asdict(execution)
                        for execution in catalog.list_executions(args.study)
                    ],
                    "approvals": [
                        asdict(approval)
                        for approval in catalog.list_study_approvals(
                            args.study
                        )
                    ],
                    "events": [
                        asdict(event)
                        for event in catalog.list_events(args.study)
                    ],
                    "artifacts": [
                        asdict(artifact)
                        for artifact in catalog.list_artifacts(args.study)
                    ],
                }
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
