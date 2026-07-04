"""Select replay strategies from development event-day PnL evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.selection import select_strategy_candidates
from data.clob.history import HistoricalClobDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="STRATEGY=ARTIFACT",
    )
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--start-target", required=True)
    parser.add_argument("--end-target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-event-days", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_paths = _parse_results(args.result)
    HistoricalClobDataset.load(args.clob_root)
    daily_by_strategy: dict[str, pd.DataFrame] = {}
    study_ids: set[str] = set()
    calibration_hashes: set[str] = set()
    clob_manifest_sha = _sha256(args.clob_root / "manifest.json")
    for strategy, path in result_paths.items():
        result_manifest = json.loads(
            (path / "manifest.json").read_text(encoding="utf-8")
        )
        if result_manifest.get("data_class") != "backtest_edge_study":
            raise ValueError(
                f"{path} is not a managed BacktestStudy artifact"
            )
        if result_manifest.get("split_role") != "oof_development":
            raise ValueError("strategy selection requires oof_development")
        if (
            result_manifest.get("input_sha256", {}).get("clob_manifest")
            != clob_manifest_sha
        ):
            raise ValueError("strategy result uses another CLOB artifact")
        study_ids.add(str(result_manifest["study_id"]))
        calibration_hashes.add(str(result_manifest["calibration_sha256"]))
        settlements = pd.read_parquet(path / "settlements.parquet")
        daily_by_strategy[strategy] = (
            settlements.groupby("target_date", as_index=False)[
                "realized_pnl_usd"
            ]
            .sum()
            .rename(columns={"target_date": "event_day"})
        )
    if len(study_ids) != 1 or len(calibration_hashes) != 1:
        raise ValueError("strategy artifacts must share one study and calibration")
    selection = select_strategy_candidates(
        daily_by_strategy,
        min_event_days=args.min_event_days,
        confidence=args.confidence,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        max_candidates=args.max_candidates,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    output_hashes: dict[str, str] = {}
    for strategy, frame in daily_by_strategy.items():
        path = args.output / f"{strategy}_event_day_pnl.parquet"
        frame.to_parquet(path, index=False)
        output_hashes[path.name] = _sha256(path)
    selection_path = args.output / "selection.parquet"
    selection.to_parquet(selection_path, index=False)
    output_hashes[selection_path.name] = _sha256(selection_path)
    manifest = {
        "schema_version": 1,
        "data_class": "development_strategy_selection",
        "study_id": next(iter(study_ids)),
        "calibration_sha256": next(iter(calibration_hashes)),
        "target_range": [args.start_target, args.end_target],
        "parameters": {
            "min_event_days": args.min_event_days,
            "confidence": args.confidence,
            "bootstrap_samples": args.bootstrap_samples,
            "max_candidates": args.max_candidates,
            "seed": args.seed,
        },
        "selected": selection.loc[
            selection["selected"], "strategy"
        ].tolist(),
        "source_sha256": {
            "clob_manifest": clob_manifest_sha,
            **{
                f"result_{strategy}": _sha256(path / "manifest.json")
                for strategy, path in result_paths.items()
            },
        },
        "sha256": output_hashes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(selection.to_string(index=False), flush=True)
    print(f"Selected: {manifest['selected']}", flush=True)
    return 0


def _parse_results(values: list[str]) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--result must use STRATEGY=ARTIFACT")
        strategy, path = value.split("=", 1)
        if strategy in results:
            raise ValueError(f"duplicate strategy result: {strategy}")
        results[strategy] = Path(path)
    return results


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
