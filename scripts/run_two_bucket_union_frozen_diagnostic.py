"""Apply the frozen historical A2 rule to known April/May diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.signal_study import load_signal_study_artifact
from backtest.synthetic_quotes import SyntheticQuoteDataset
from backtest.two_bucket_union_frozen_diagnostic import (
    add_exact_horizon_band,
    add_fixed_gap_band,
    build_frozen_union_diagnostic_report,
    write_frozen_union_diagnostic_artifact,
)
from backtest.two_bucket_union_study import (
    build_two_bucket_candidate_panel,
    fit_frozen_union_calibration,
    replay_taker_unions,
)
from data.clob.history import HistoricalClobDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-a2", required=True, type=Path)
    parser.add_argument("--april-forecast-cache", required=True, type=Path)
    parser.add_argument("--april-signal-root", required=True, type=Path)
    parser.add_argument("--may-forecast-cache", required=True, type=Path)
    parser.add_argument("--may-signal-root", required=True, action="append", type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--risk-usd", type=float, default=50.0)
    parser.add_argument("--calibration-bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"frozen diagnostic output already exists: {args.output}")
    if args.risk_usd <= 0 or min(
        args.calibration_bootstrap_samples, args.bootstrap_samples
    ) <= 0:
        raise ValueError("risk and bootstrap samples must be positive")
    print(f"[start] frozen A2 April/May diagnostic output={args.output}", flush=True)
    input_hashes, forecast_lineages, exclusions = _validate_inputs(args)
    print(
        "[complete] input lineage "
        f"exclusions={len(exclusions)} forecast_lineages={len(set(forecast_lineages.values()))}",
        flush=True,
    )

    quotes = SyntheticQuoteDataset.load(args.quotes_root)
    clob = HistoricalClobDataset.load(args.clob_root)
    april_forecast = ForecastCache.load(args.april_forecast_cache)
    may_forecast = ForecastCache.load(args.may_forecast_cache)
    april_signal = load_signal_study_artifact(args.april_signal_root)
    may_signal = [load_signal_study_artifact(root) for root in args.may_signal_root]
    april_candidates = build_two_bucket_candidate_panel(
        panel=april_signal.panel,
        snapshots=april_forecast.snapshots,
        quotes=quotes.quotes,
        markets=clob.markets,
        evaluation_id="april_2026",
        evaluation_target_range=("2026-04-01", "2026-04-30"),
    )
    may_panel = pd.concat([artifact.panel for artifact in may_signal], ignore_index=True)
    if may_panel.duplicated(
        ["event_id", "timestamp", "snapshot_id", "condition_id"]
    ).any():
        raise ValueError("May signal roots contain duplicate market-vector rows")
    may_candidates = build_two_bucket_candidate_panel(
        panel=may_panel,
        snapshots=may_forecast.snapshots,
        quotes=quotes.quotes,
        markets=clob.markets,
        evaluation_id="may_2026",
        evaluation_target_range=("2026-05-01", "2026-05-31"),
    )
    candidates = pd.concat([april_candidates, may_candidates], ignore_index=True)
    if set(candidates["target_date"].astype(str)).intersection({"2026-05-17", "2026-05-18"}):
        raise ValueError("market-unavailable May 17/18 entered the diagnostic panel")
    print(
        "[complete] pair selection "
        f"candidates={len(candidates)} events={candidates['event_id'].nunique()}",
        flush=True,
    )

    historical = pd.read_parquet(args.historical_a2 / "selection_panel.parquet")
    calibration_training = (
        historical.sort_values(["event_id", "timestamp"], kind="stable")
        .drop_duplicates(["event_id", "snapshot_id"], keep="first")
        .reset_index(drop=True)
    )
    if str(calibration_training["target_date"].max()) != "2026-03-31":
        raise ValueError("historical A2 calibration does not end on 2026-03-31")
    calibration = fit_frozen_union_calibration(
        calibration_training,
        bootstrap_samples=args.calibration_bootstrap_samples,
        seed=args.seed,
    )
    candidates = calibration.apply(candidates)
    candidates = add_fixed_gap_band(add_exact_horizon_band(candidates))
    print(
        "[complete] frozen calibration "
        f"fit_events={calibration.fit_event_days} fit_end={calibration.fit_target_end} "
        f"bootstrap_draws={len(calibration.bootstrap_intercepts)}",
        flush=True,
    )

    replay = replay_taker_unions(
        candidates,
        stage="frozen",
        edge_threshold=0.0,
        risk_usd=args.risk_usd,
    )
    print(
        "[complete] frozen taker replay "
        f"decisions={len(replay.decisions)} settlements={len(replay.settlements)}",
        flush=True,
    )
    report = build_frozen_union_diagnostic_report(
        candidates=candidates,
        decisions=replay.decisions,
        legs=replay.legs,
        settlements=replay.settlements,
        exclusions=exclusions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        forecast_lineages=forecast_lineages,
    )
    report["protocol"] = {
        "pair_width": 2,
        "pair_selection": "maximum_adjacent_model_probability",
        "calibration": "event_equal_logistic_all_historical_oof_through_2026_03_31",
        "calibration_probability_bound": "one_sided_90pct_event_day_bootstrap_lower",
        "calibration_fit_target_range": [
            calibration.fit_target_start,
            calibration.fit_target_end,
        ],
        "calibration_fit_event_days": calibration.fit_event_days,
        "calibration_bootstrap_samples": args.calibration_bootstrap_samples,
        "taker": "first_positive_lcb_minus_equal_share_all_in_cost",
        "risk_usd": float(args.risk_usd),
        "edge_threshold": 0.0,
        "trade_limit": "one_completed_union_per_Event",
        "market_unavailable_exclusions": ["2026-05-17", "2026-05-18"],
        "outcome_use": "settlement_scoring_only_no_refit_no_threshold_search",
        "horizon_bands": [
            "<6h", "6-12h", "12-24h", "24-30h",
            "30-36h", "36-42h", "42-48h",
        ],
        "gap_bands": ["<0.15", "0.15-0.30", "0.30-0.45", ">=0.45"],
    }
    manifest = write_frozen_union_diagnostic_artifact(
        output=args.output,
        frames={
            "selection_panel": candidates,
            "calibration_coefficients": calibration.coefficients(),
            "decisions": replay.decisions,
            "execution_legs": replay.legs,
            "event_pnl": replay.settlements,
            "exclusions": exclusions,
        },
        report=report,
        input_sha256=input_hashes,
    )
    print(
        f"[done] frozen A2 April/May diagnostic output={args.output} "
        f"settlements={len(replay.settlements)} manifest={manifest['data_class']}",
        flush=True,
    )
    return 0


def _validate_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, str], pd.DataFrame]:
    manifest_paths = {
        "historical_a2_manifest": args.historical_a2 / "manifest.json",
        "april_forecast_manifest": args.april_forecast_cache / "manifest.json",
        "april_signal_manifest": args.april_signal_root / "manifest.json",
        "may_forecast_manifest": args.may_forecast_cache / "manifest.json",
        "quotes_manifest": args.quotes_root / "manifest.json",
        "clob_manifest": args.clob_root / "manifest.json",
    }
    for index, root in enumerate(args.may_signal_root):
        manifest_paths[f"may_signal_manifest_{index}"] = root / "manifest.json"
    hashes = {name: _sha256(path) for name, path in manifest_paths.items()}
    historical_manifest = _load_manifest(manifest_paths["historical_a2_manifest"])
    if historical_manifest.get("data_class") != "model_selected_two_bucket_union_study":
        raise ValueError("historical A2 artifact has the wrong data class")
    _verify_payload_hashes(args.historical_a2, historical_manifest)

    clob_manifest = _load_manifest(manifest_paths["clob_manifest"])
    quotes_manifest = _load_manifest(manifest_paths["quotes_manifest"])
    if quotes_manifest.get("source_sha256") != hashes["clob_manifest"]:
        raise ValueError("synthetic quotes are bound to another CLOB artifact")
    unavailable = set(map(str, clob_manifest.get("incomplete_target_dates", [])))
    if not {"2026-05-17", "2026-05-18"}.issubset(unavailable):
        raise ValueError("CLOB manifest must declare May 17/18 unavailable")

    evaluation_contracts = [
        (
            "april_2026",
            args.april_forecast_cache,
            args.april_signal_root,
            ("2026-04-01", "2026-04-30"),
            hashes["april_forecast_manifest"],
            hashes["april_signal_manifest"],
        )
    ]
    may_ranges = sorted(
        [
            tuple(_load_manifest(root / "manifest.json").get("target_range", []))
            for root in args.may_signal_root
        ]
    )
    if may_ranges != [("2026-05-01", "2026-05-16"), ("2026-05-19", "2026-05-31")]:
        raise ValueError("May diagnostics must use the frozen early/late signal roots")
    for index, root in enumerate(args.may_signal_root):
        signal_manifest = _load_manifest(root / "manifest.json")
        evaluation_contracts.append(
            (
                "may_2026",
                args.may_forecast_cache,
                root,
                tuple(signal_manifest["target_range"]),
                hashes["may_forecast_manifest"],
                hashes[f"may_signal_manifest_{index}"],
            )
        )

    forecast_lineages: dict[str, str] = {}
    exclusion_rows: list[dict[str, str]] = [
        {
            "evaluation_id": "may_2026",
            "target_date": target_date,
            "reason": "market_unavailable",
            "source": "clob_manifest",
        }
        for target_date in ("2026-05-17", "2026-05-18")
    ]
    checked_forecasts: set[Path] = set()
    for evaluation_id, forecast_root, signal_root, target_range, forecast_hash, _ in evaluation_contracts:
        signal_manifest = _load_manifest(signal_root / "manifest.json")
        if signal_manifest.get("split_role") != "model_test":
            raise ValueError("frozen diagnostic accepts only model_test signals")
        if tuple(signal_manifest.get("target_range", [])) != target_range:
            raise ValueError("signal target range does not match the evaluation contract")
        expected = signal_manifest.get("input_sha256", {})
        if expected.get("forecast_manifest") != forecast_hash:
            raise ValueError("signal panel is bound to another ForecastCache artifact")
        if expected.get("quotes_manifest") != hashes["quotes_manifest"]:
            raise ValueError("signal panel is bound to another synthetic quotes artifact")
        if expected.get("clob_manifest") != hashes["clob_manifest"]:
            raise ValueError("signal panel is bound to another CLOB artifact")
        summary = _load_manifest(signal_root / "summary.json")
        for exclusion in summary.get("coverage_exclusions", []):
            exclusion_rows.append(
                {
                    "evaluation_id": evaluation_id,
                    "target_date": str(exclusion["target_date"]),
                    "reason": str(exclusion["reason"]),
                    "source": "signal_summary",
                }
            )
        if forecast_root not in checked_forecasts:
            forecast_manifest = _load_manifest(forecast_root / "manifest.json")
            if forecast_manifest.get("split_roles") != ["model_test"]:
                raise ValueError("diagnostic ForecastCache must have model_test split role")
            sources = forecast_manifest.get("source_sha256", {})
            forecast_quotes = sources.get("quotes_manifest")
            if (
                forecast_quotes is not None
                and forecast_quotes != hashes["quotes_manifest"]
            ):
                raise ValueError("ForecastCache is bound to another synthetic quotes artifact")
            if sources.get("clob_manifest") != hashes["clob_manifest"]:
                raise ValueError("ForecastCache is bound to another CLOB artifact")
            lineages = pd.read_parquet(forecast_root / "lineages.parquet")
            checkpoints = lineages["checkpoint_sha256"].astype(str).unique().tolist()
            if len(checkpoints) != 1:
                raise ValueError("each diagnostic ForecastCache must use one checkpoint")
            forecast_lineages[evaluation_id] = checkpoints[0]
            checked_forecasts.add(forecast_root)

    exclusions = (
        pd.DataFrame(exclusion_rows)
        .drop_duplicates(["evaluation_id", "target_date", "reason", "source"])
        .sort_values(["target_date", "reason"], kind="stable")
        .reset_index(drop=True)
    )
    return hashes, forecast_lineages, exclusions


def _verify_payload_hashes(root: Path, manifest: dict[str, object]) -> None:
    payloads = manifest.get("sha256")
    if not isinstance(payloads, dict) or not payloads:
        raise ValueError(f"artifact manifest has no payload hashes: {root}")
    for filename, expected in payloads.items():
        if _sha256(root / str(filename)) != str(expected):
            raise ValueError(f"artifact payload hash mismatch: {root / str(filename)}")


def _load_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
