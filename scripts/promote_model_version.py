"""Explicitly promote a locally reviewed model version to active."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db.schema import DbClient
from model.store.registry import ModelRegistry


def promote_model_version(
    model_store: Path | str,
    db_path: Path | str,
    version: str,
) -> dict[str, Any]:
    db = DbClient(str(db_path))
    try:
        registry = ModelRegistry(Path(model_store), db)
        registry.rollback(version)
    finally:
        db.close()
    promoted_at = datetime.now(timezone.utc).isoformat()
    audit_line = json.dumps(
        {"version": version, "promoted_at": promoted_at},
        sort_keys=True,
    )
    audit_path = Path(model_store) / "promotion_audit.log"
    with audit_path.open("a", encoding="ascii") as fh:
        fh.write(audit_line + "\n")
    return {
        "active_version": version,
        "promoted_at": promoted_at,
        "audit_log": str(audit_path),
    }


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
