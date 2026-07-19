"""Run strict historical OOF market-aware two-bucket probability stacks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.two_bucket_probability_stack import (
    DEFAULT_STACK_SPECS,
    apply_expanding_probability_stacks,
    build_probability_stack_report,
    fit_expanding_probability_stacks,
    fit_probability_stack,
    score_probability_stack_samples,
    write_probability_stack_artifact,
)
from backtest.two_bucket_union_frozen_diagnostic import (
    add_exact_horizon_band,
    add_fixed_gap_band,
)
from backtest.two_bucket_union_study import replay_taker_unions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-a2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--risk-usd", type=float, default=50.0)
    parser.add_argument("--calibration-bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"probability stack output already exists: {args.output}")
    if args.risk_usd <= 0 or min(
        args.calibration_bootstrap_samples, args.bootstrap_samples
    ) <= 0:
        raise ValueError("risk and bootstrap samples must be positive")
    print(f"[start] two-bucket probability stack output={args.output}", flush=True)
    input_hashes = _validate_historical_a2(args.historical_a2)
    candidates = pd.read_parquet(args.historical_a2 / "selection_panel.parquet")
    expected_folds = {f"fold-{index:02d}" for index in range(6)}
    if set(candidates["fold_id"].astype(str)) != expected_folds:
        raise ValueError("historical A2 candidate panel must contain folds 00..05")
    if (
        str(candidates["target_date"].astype(str).min()) != "2025-04-01"
        or str(candidates["target_date"].astype(str).max()) != "2026-03-31"
    ):
        raise ValueError("historical A2 candidate range is not the frozen historical year")
    calibration_samples = (
        candidates.sort_values(["event_id", "timestamp"], kind="stable")
        .drop_duplicates(["event_id", "snapshot_id"], keep="first")
        .reset_index(drop=True)
    )
    print(
        "[complete] historical A2 inputs "
        f"candidates={len(candidates)} calibration_samples={len(calibration_samples)} "
        f"events={candidates['event_id'].nunique()}",
        flush=True,
    )

    expanding = fit_expanding_probability_stacks(
        calibration_samples,
        specs=DEFAULT_STACK_SPECS,
        bootstrap_samples=args.calibration_bootstrap_samples,
        seed=args.seed,
    )
    scores = score_probability_stack_samples(expanding.scored_samples)
    print(
        "[complete] expanding OOF probability stacks "
        f"score_rows={len(scores)} coefficient_rows={len(expanding.coefficients)}",
        flush=True,
    )

    evaluation_candidates = candidates[
        candidates["fold_id"].astype(str) != "fold-00"
    ].copy()
    evaluation_candidates = add_fixed_gap_band(
        add_exact_horizon_band(evaluation_candidates)
    )
    probability_panel = apply_expanding_probability_stacks(
        candidates,
        expanding.models,
    )
    probability_panel = add_fixed_gap_band(
        add_exact_horizon_band(probability_panel)
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
            f"[complete] historical taker replay source={source} "
            f"settlements={len(replay.settlements)}",
            flush=True,
        )
    decisions = _concat([replay.decisions for replay in replays])
    legs = _concat([replay.legs for replay in replays])
    settlements = _concat([replay.settlements for replay in replays])
    _assert_current_a2_identity(
        args.historical_a2,
        probability_panel=probability_panel,
        settlements=settlements,
    )
    print("[complete] current model-only A2 identity verified", flush=True)

    frozen_coefficients = []
    all_folds = sorted(calibration_samples["fold_id"].astype(str).unique())
    for spec in DEFAULT_STACK_SPECS:
        frozen = fit_probability_stack(
            calibration_samples,
            spec=spec,
            bootstrap_samples=args.calibration_bootstrap_samples,
            seed=args.seed,
            fit_fold_ids=all_folds,
        )
        frozen_coefficients.append(
            frozen.coefficient_frame(evaluation_fold_id="frozen_pre_april")
        )
    coefficients = pd.concat(
        [expanding.coefficients, *frozen_coefficients], ignore_index=True
    )
    print(
        "[complete] frozen pre-April coefficients "
        f"sources={len(frozen_coefficients)} fit_end=2026-03-31",
        flush=True,
    )

    report = build_probability_stack_report(
        scores=scores,
        coefficients=coefficients,
        candidates=evaluation_candidates,
        settlements=settlements,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["protocol"] = {
        "pair_selection": "frozen_maximum_adjacent_model_probability",
        "probability_sources": {
            "current_model_only": ["model_logit"],
            "market_only": ["midpoint_logit", "is_36_42h"],
            "model_plus_market": [
                "model_logit", "midpoint_logit", "is_36_42h",
            ],
        },
        "ridge_strength": 1.0,
        "calibration": "event_equal_expanding_historical_folds",
        "calibration_tick_sampling": "first_eligible_tick_per_event_and_forecast_snapshot",
        "execution_tick_sampling": "all_eligible_5_minute_ticks",
        "calibration_bootstrap_samples": args.calibration_bootstrap_samples,
        "probability_bound": "one_sided_90pct_event_day_bootstrap_lower",
        "execution": "first_positive_lcb_minus_equal_share_all_in_taker_cost",
        "risk_usd": float(args.risk_usd),
        "edge_threshold": 0.0,
        "historical_target_range": ["2025-04-01", "2026-03-31"],
        "frozen_coefficient_id": "frozen_pre_april",
        "april_may_read": False,
    }
    manifest = write_probability_stack_artifact(
        output=args.output,
        frames={
            "probability_scores": scores,
            "calibration_coefficients": coefficients,
            "probability_panel": probability_panel,
            "decisions": decisions,
            "execution_legs": legs,
            "event_pnl": settlements,
        },
        report=report,
        input_sha256=input_hashes,
    )
    print(
        f"[done] two-bucket probability stack output={args.output} "
        f"settlements={len(settlements)} manifest={manifest['data_class']}",
        flush=True,
    )
    return 0


def _assert_current_a2_identity(
    root: Path,
    *,
    probability_panel: pd.DataFrame,
    settlements: pd.DataFrame,
) -> None:
    expected_panel = pd.read_parquet(root / "selection_panel.parquet")
    expected_panel = expected_panel[
        expected_panel["fold_id"].astype(str) != "fold-00"
    ].copy()
    actual_panel = probability_panel[
        probability_panel["probability_source"].astype(str) == "current_model_only"
    ].copy()
    key_columns = ["event_id", "timestamp", "snapshot_id"]
    probability_columns = [
        "calibrated_union_probability",
        "union_probability_lower_90",
        "union_probability_upper_90",
    ]
    expected_panel = (
        expected_panel.loc[:, [*key_columns, *probability_columns]]
        .sort_values(key_columns)
        .reset_index(drop=True)
    )
    actual_panel = (
        actual_panel.loc[:, [*key_columns, *probability_columns]]
        .sort_values(key_columns)
        .reset_index(drop=True)
    )
    if len(expected_panel) != len(actual_panel) or not expected_panel[
        key_columns
    ].equals(actual_panel[key_columns]):
        raise ValueError("current model-only probability panel does not match historical A2")
    for column in probability_columns:
        if not np.array_equal(
            expected_panel[column].to_numpy(), actual_panel[column].to_numpy()
        ):
            raise ValueError(
                f"current model-only probability differs from historical A2: {column}"
            )

    expected = pd.read_parquet(root / "event_pnl.parquet")
    expected = expected[
        (expected["stage"].astype(str) == "frozen")
        & (expected["execution_mode"].astype(str) == "taker")
        & np.isclose(expected["edge_threshold"].astype(float), 0.0)
    ].copy()
    actual = settlements[
        settlements["probability_source"].astype(str) == "current_model_only"
    ].copy()
    columns = [
        "event_id", "timestamp", "snapshot_id", "cost_basis_usd", "realized_pnl_usd",
    ]
    expected = expected.loc[:, columns].sort_values("event_id").reset_index(drop=True)
    actual = actual.loc[:, columns].sort_values("event_id").reset_index(drop=True)
    if len(expected) != len(actual):
        raise ValueError("current model-only replay does not match historical A2 settlements")
    if not (
        expected[["event_id", "timestamp", "snapshot_id"]].equals(
            actual[["event_id", "timestamp", "snapshot_id"]]
        )
        and np.allclose(expected["cost_basis_usd"], actual["cost_basis_usd"], atol=1e-9)
        and np.allclose(expected["realized_pnl_usd"], actual["realized_pnl_usd"], atol=1e-9)
    ):
        raise ValueError("current model-only replay differs from historical A2 identity")


def _validate_historical_a2(root: Path) -> dict[str, str]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("data_class") != "model_selected_two_bucket_union_study":
        raise ValueError("historical A2 artifact has the wrong data class")
    payloads = manifest.get("sha256")
    if not isinstance(payloads, dict) or not payloads:
        raise ValueError("historical A2 manifest has no payload hashes")
    for filename, expected in payloads.items():
        if _sha256(root / str(filename)) != str(expected):
            raise ValueError(f"historical A2 payload hash mismatch: {filename}")
    return {"historical_a2_manifest": _sha256(manifest_path)}


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
