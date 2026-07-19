"""Apply pre-April probability stacks to the frozen April/May A2 panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.two_bucket_probability_stack import score_probability_stack_samples
from backtest.two_bucket_probability_stack_frozen import (
    apply_frozen_probability_stacks,
    build_frozen_probability_stack_report,
    load_frozen_probability_stacks,
    write_frozen_probability_stack_artifact,
)
from backtest.two_bucket_union_study import replay_taker_unions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-stack", required=True, type=Path)
    parser.add_argument("--april-may-a2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--risk-usd", type=float, default=50.0)
    parser.add_argument("--calibration-bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"frozen probability stack output already exists: {args.output}")
    if args.risk_usd <= 0 or min(
        args.calibration_bootstrap_samples, args.bootstrap_samples
    ) <= 0:
        raise ValueError("risk and bootstrap samples must be positive")
    print(
        f"[start] frozen April/May probability stack output={args.output}",
        flush=True,
    )
    input_hashes = _validate_inputs(args.historical_stack, args.april_may_a2)
    coefficients = pd.read_parquet(
        args.historical_stack / "calibration_coefficients.parquet"
    )
    models = load_frozen_probability_stacks(
        coefficients,
        expected_bootstrap_samples=args.calibration_bootstrap_samples,
        training_cutoff="2026-03-31",
    )
    frozen_coefficients = coefficients[
        coefficients["evaluation_fold_id"].astype(str) == "frozen_pre_april"
    ].copy()
    print(
        "[complete] frozen coefficients loaded "
        f"sources={len(models)} rows={len(frozen_coefficients)} fit_end=2026-03-31",
        flush=True,
    )

    candidates = pd.read_parquet(args.april_may_a2 / "selection_panel.parquet")
    exclusions = pd.read_parquet(args.april_may_a2 / "exclusions.parquet")
    _validate_diagnostic_panel(candidates, exclusions)
    probability_panel = apply_frozen_probability_stacks(candidates, models)
    calibration_samples = (
        probability_panel.sort_values(
            ["event_id", "timestamp", "probability_source"], kind="stable"
        )
        .drop_duplicates(
            ["event_id", "snapshot_id", "probability_source"], keep="first"
        )
        .reset_index(drop=True)
    )
    scores = score_probability_stack_samples(calibration_samples)
    print(
        "[complete] frozen probability panel "
        f"ticks={len(candidates)} score_rows={len(scores)} events={candidates['event_id'].nunique()}",
        flush=True,
    )

    replays = []
    for source in sorted(probability_panel["probability_source"].astype(str).unique()):
        replay = replay_taker_unions(
            probability_panel[
                probability_panel["probability_source"].astype(str) == source
            ],
            stage="frozen",
            edge_threshold=0.0,
            risk_usd=args.risk_usd,
        )
        replays.append(replay)
        print(
            f"[complete] frozen taker replay source={source} "
            f"settlements={len(replay.settlements)}",
            flush=True,
        )
    decisions = _concat([replay.decisions for replay in replays])
    legs = _concat([replay.legs for replay in replays])
    settlements = _concat([replay.settlements for replay in replays])
    _assert_current_a2_identity(
        args.april_may_a2,
        probability_panel=probability_panel,
        settlements=settlements,
    )
    print("[complete] frozen current model-only A2 identity verified", flush=True)

    report = build_frozen_probability_stack_report(
        scores=scores,
        candidates=candidates,
        settlements=settlements,
        exclusions=exclusions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["protocol"] = {
        "probability_sources": {
            "current_model_only": ["model_logit"],
            "market_only": ["midpoint_logit", "is_36_42h"],
            "model_plus_market": [
                "model_logit", "midpoint_logit", "is_36_42h",
            ],
        },
        "gap_feature_or_gate": False,
        "coefficient_id": "frozen_pre_april",
        "coefficient_training_cutoff": "2026-03-31",
        "diagnostic_outcome_use": "scoring_and_settlement_only_no_refit",
        "calibration_tick_sampling": "first_eligible_tick_per_event_and_forecast_snapshot",
        "execution_tick_sampling": "all_eligible_5_minute_ticks",
        "execution": "first_positive_lcb_minus_equal_share_all_in_taker_cost",
        "edge_threshold": 0.0,
        "risk_usd": float(args.risk_usd),
        "market_unavailable_exclusions": ["2026-05-17", "2026-05-18"],
    }
    manifest = write_frozen_probability_stack_artifact(
        output=args.output,
        frames={
            "probability_scores": scores,
            "calibration_coefficients": frozen_coefficients,
            "probability_panel": probability_panel,
            "decisions": decisions,
            "execution_legs": legs,
            "event_pnl": settlements,
            "exclusions": exclusions,
        },
        report=report,
        input_sha256=input_hashes,
    )
    print(
        f"[done] frozen April/May probability stack output={args.output} "
        f"settlements={len(settlements)} manifest={manifest['data_class']}",
        flush=True,
    )
    return 0


def _validate_inputs(
    historical_stack: Path,
    april_may_a2: Path,
) -> dict[str, str]:
    stack_manifest_path = historical_stack / "manifest.json"
    diagnostic_manifest_path = april_may_a2 / "manifest.json"
    stack_manifest = _load_manifest(stack_manifest_path)
    diagnostic_manifest = _load_manifest(diagnostic_manifest_path)
    if stack_manifest.get("data_class") != "two_bucket_probability_stack_historical_oof":
        raise ValueError("historical probability stack has the wrong data class")
    if diagnostic_manifest.get("data_class") != "frozen_two_bucket_union_april_may_diagnostic":
        raise ValueError("April/May A2 artifact has the wrong data class")
    _verify_payload_hashes(historical_stack, stack_manifest)
    _verify_payload_hashes(april_may_a2, diagnostic_manifest)
    stack_parent = stack_manifest.get("input_sha256", {}).get(
        "historical_a2_manifest"
    )
    diagnostic_parent = diagnostic_manifest.get("input_sha256", {}).get(
        "historical_a2_manifest"
    )
    if not stack_parent or stack_parent != diagnostic_parent:
        raise ValueError("historical stack and April/May diagnostic use different A2 parents")
    return {
        "historical_stack_manifest": _sha256(stack_manifest_path),
        "april_may_a2_manifest": _sha256(diagnostic_manifest_path),
        "historical_a2_manifest": str(stack_parent),
    }


def _validate_diagnostic_panel(
    candidates: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> None:
    if set(candidates["evaluation_id"].astype(str)) != {"april_2026", "may_2026"}:
        raise ValueError("diagnostic panel must contain April and May")
    dates = candidates["target_date"].astype(str)
    if dates.min() != "2026-04-01" or dates.max() != "2026-05-31":
        raise ValueError("diagnostic panel target range is invalid")
    unavailable = {"2026-05-17", "2026-05-18"}
    if unavailable & set(dates):
        raise ValueError("market-unavailable May dates are present in candidates")
    exclusion_dates = set(exclusions["target_date"].astype(str))
    if not unavailable.issubset(exclusion_dates):
        raise ValueError("market-unavailable May dates are absent from exclusions")


def _assert_current_a2_identity(
    root: Path,
    *,
    probability_panel: pd.DataFrame,
    settlements: pd.DataFrame,
) -> None:
    expected_panel = pd.read_parquet(root / "selection_panel.parquet")
    actual_panel = probability_panel[
        probability_panel["probability_source"].astype(str) == "current_model_only"
    ].copy()
    keys = ["event_id", "timestamp", "snapshot_id"]
    probability_columns = [
        "calibrated_union_probability",
        "union_probability_lower_90",
        "union_probability_upper_90",
    ]
    expected_panel = expected_panel[[*keys, *probability_columns]].sort_values(
        keys
    ).reset_index(drop=True)
    actual_panel = actual_panel[[*keys, *probability_columns]].sort_values(
        keys
    ).reset_index(drop=True)
    if len(expected_panel) != len(actual_panel) or not expected_panel[keys].equals(
        actual_panel[keys]
    ):
        raise ValueError("frozen current model-only probability panel differs from A2")
    for column in probability_columns:
        if not np.array_equal(
            expected_panel[column].to_numpy(), actual_panel[column].to_numpy()
        ):
            raise ValueError(f"frozen current model-only probability differs from A2: {column}")

    expected_pnl = pd.read_parquet(root / "event_pnl.parquet")
    actual_pnl = settlements[
        settlements["probability_source"].astype(str) == "current_model_only"
    ]
    columns = [
        "event_id", "timestamp", "snapshot_id", "cost_basis_usd", "realized_pnl_usd",
    ]
    expected_pnl = expected_pnl[columns].sort_values("event_id").reset_index(drop=True)
    actual_pnl = actual_pnl[columns].sort_values("event_id").reset_index(drop=True)
    if len(expected_pnl) != len(actual_pnl) or not expected_pnl[
        columns[:3]
    ].equals(actual_pnl[columns[:3]]):
        raise ValueError("frozen current model-only settlements differ from A2")
    for column in columns[3:]:
        if not np.allclose(
            expected_pnl[column].to_numpy(float),
            actual_pnl[column].to_numpy(float),
            atol=1e-9,
        ):
            raise ValueError(f"frozen current model-only settlement differs from A2: {column}")


def _verify_payload_hashes(root: Path, manifest: dict[str, object]) -> None:
    payloads = manifest.get("sha256")
    if not isinstance(payloads, dict) or not payloads:
        raise ValueError(f"artifact manifest has no payload hashes: {root}")
    for filename, expected in payloads.items():
        if _sha256(root / str(filename)) != str(expected):
            raise ValueError(f"artifact payload hash mismatch: {root / str(filename)}")


def _load_manifest(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
