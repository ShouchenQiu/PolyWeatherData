"""Read-only integrity check for a completed managed training study."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from training.catalog import TrainingCatalog


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="runs/_management/training_catalog.db",
    )
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    checks: list[dict[str, object]] = []
    with TrainingCatalog(args.catalog) as catalog:
        study = catalog.validate_study(args.study)
        executions = catalog.list_executions(args.study)
        artifacts = catalog.list_artifacts(args.study)
        for artifact in artifacts:
            path = Path(artifact.path)
            actual = _sha256(path) if path.is_file() else None
            checks.append(
                {
                    "name": f"artifact:{artifact.artifact_type}",
                    "ok": actual == artifact.sha256,
                    "path": str(path),
                }
            )
        gpu_seconds = sum(
            float(item.metrics.get("gpu_seconds", 0.0) or 0.0)
            for item in executions
            if item.execution_type == "trial"
        )
        report = {
            "study": asdict(study),
            "execution_counts": {
                status: sum(
                    item.status == status for item in executions
                )
                for status in sorted({item.status for item in executions})
            },
            "trial_gpu_hours": gpu_seconds / 3600.0,
            "artifacts": checks,
            "ok": bool(checks)
            and all(bool(item["ok"]) for item in checks)
            and study.status
            in {
                "completed",
                "budget_exhausted",
                "awaiting_selection",
                "final_approved",
                "final_running",
                "candidate_ready",
            },
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
