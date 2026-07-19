"""Run model-selected adjacent two-bucket taker and maker diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.signal_study import load_signal_study_artifact
from backtest.synthetic_quotes import SyntheticQuoteDataset
from backtest.two_bucket_union_study import (
    build_two_bucket_candidate_panel,
    build_two_bucket_report,
    fit_expanding_union_calibration,
    replay_maker_unions,
    replay_taker_unions,
    write_two_bucket_artifact,
)
from data.clob.history import HistoricalClobDataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-cache", required=True, type=Path)
    parser.add_argument("--signal-root", required=True, type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--risk-usd", type=float, default=50.0)
    parser.add_argument("--calibration-bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maker-max-ttl", default="60min")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"two-bucket output already exists: {args.output}")
    if args.risk_usd <= 0:
        raise ValueError("risk-usd must be positive")
    if min(args.calibration_bootstrap_samples, args.bootstrap_samples) <= 0:
        raise ValueError("bootstrap samples must be positive")
    print(f"[start] two-bucket union study output={args.output}", flush=True)
    input_hashes = _validate_lineage(args)
    forecast = ForecastCache.load(args.forecast_cache)
    signal = load_signal_study_artifact(args.signal_root)
    quotes = SyntheticQuoteDataset.load(args.quotes_root)
    clob = HistoricalClobDataset.load(args.clob_root)
    print(
        "[complete] inputs "
        f"snapshots={len(forecast.snapshots)} vectors={signal.panel[['event_id', 'timestamp']].drop_duplicates().shape[0]} "
        f"quotes={len(quotes.quotes)} trades={len(clob.trades)}",
        flush=True,
    )

    candidates = build_two_bucket_candidate_panel(
        panel=signal.panel,
        snapshots=forecast.snapshots,
        quotes=quotes.quotes,
        markets=clob.markets,
    )
    print(
        f"[complete] pair selection candidates={len(candidates)} events={candidates['event_id'].nunique()}",
        flush=True,
    )

    calibration_samples = (
        candidates.sort_values(["event_id", "timestamp"], kind="stable")
        .drop_duplicates(["event_id", "snapshot_id"], keep="first")
        .reset_index(drop=True)
    )
    calibrated_samples, calibration_contracts = fit_expanding_union_calibration(
        calibration_samples,
        bootstrap_samples=args.calibration_bootstrap_samples,
        seed=args.seed,
    )
    calibration_values = calibrated_samples.loc[:, [
        "event_id", "snapshot_id", "calibrated_union_probability",
        "union_probability_lower_90", "union_probability_upper_90",
    ]]
    frozen_candidates = candidates.merge(
        calibration_values,
        on=["event_id", "snapshot_id"],
        how="inner",
        validate="many_to_one",
    )
    expected_frozen_folds = {"fold-01", "fold-02", "fold-03", "fold-04", "fold-05"}
    if set(frozen_candidates["fold_id"].astype(str)) != expected_frozen_folds:
        raise ValueError("frozen candidate panel does not contain folds 01..05")
    candidates = candidates.merge(
        calibration_values,
        on=["event_id", "snapshot_id"],
        how="left",
        validate="many_to_one",
    )
    print(
        f"[complete] expanding calibration folds={len(calibration_contracts)} samples={len(calibration_samples)}",
        flush=True,
    )

    replays = []
    for mode, runner in (("taker", replay_taker_unions), ("maker", replay_maker_unions)):
        kwargs = {"trades": clob.trades} if mode == "maker" else {}
        unconditional = runner(
            candidates,
            stage="unconditional",
            edge_threshold=0.0,
            risk_usd=args.risk_usd,
            **kwargs,
        )
        replays.append(unconditional)
        print(
            f"[complete] replay mode={mode} stage=unconditional settlements={len(unconditional.settlements)}",
            flush=True,
        )
        for threshold in (0.0, 0.02, 0.04):
            threshold_kwargs = dict(kwargs)
            if mode == "maker":
                threshold_kwargs["max_ttl"] = args.maker_max_ttl
            frozen = runner(
                frozen_candidates,
                stage="frozen",
                edge_threshold=threshold,
                risk_usd=args.risk_usd,
                **threshold_kwargs,
            )
            replays.append(frozen)
            print(
                f"[complete] replay mode={mode} stage=frozen threshold={threshold:.2f} settlements={len(frozen.settlements)}",
                flush=True,
            )

    decisions = _concat([replay.decisions for replay in replays])
    legs = _concat([replay.legs for replay in replays])
    settlements = _concat([replay.settlements for replay in replays])
    report = build_two_bucket_report(
        candidates=candidates,
        decisions=decisions,
        legs=legs,
        settlements=settlements,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["protocol"] = {
        "pair_width": 2,
        "pair_selection": "maximum_adjacent_model_probability",
        "risk_usd": float(args.risk_usd),
        "taker": "equal_shares_full_fill_or_skip_with_fees",
        "maker": "best_bid_strict_trade_through_no_touch_fill",
        "maker_max_ttl": str(args.maker_max_ttl),
        "maker_fee_and_rebate": "excluded",
        "frozen_primary_edge_threshold": 0.0,
        "frozen_sensitivity_thresholds": [0.02, 0.04],
        "calibration": "event_equal_logistic_expanding_folds",
        "calibration_probability_bound": "one_sided_90pct_bootstrap_lower",
    }
    manifest = write_two_bucket_artifact(
        output=args.output,
        frames={
            "selection_panel": candidates,
            "calibration_folds": calibration_contracts,
            "decisions": decisions,
            "execution_legs": legs,
            "event_pnl": settlements,
        },
        report=report,
        input_sha256=input_hashes,
    )
    print(
        f"[done] two-bucket union study output={args.output} settlements={len(settlements)} manifest={manifest['data_class']}",
        flush=True,
    )
    return 0


def _validate_lineage(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "forecast_manifest": args.forecast_cache / "manifest.json",
        "signal_manifest": args.signal_root / "manifest.json",
        "quotes_manifest": args.quotes_root / "manifest.json",
        "clob_manifest": args.clob_root / "manifest.json",
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    signal = json.loads(paths["signal_manifest"].read_text(encoding="utf-8"))
    quotes = json.loads(paths["quotes_manifest"].read_text(encoding="utf-8"))
    forecast = json.loads(paths["forecast_manifest"].read_text(encoding="utf-8"))
    if signal.get("split_role") != "oof_development":
        raise ValueError("two-bucket study accepts only oof_development signals")
    if signal.get("target_range") != ["2025-04-01", "2026-03-31"]:
        raise ValueError("signal target range must be the historical six-fold year")
    expected = signal.get("input_sha256", {})
    if expected.get("forecast_manifest") != hashes["forecast_manifest"]:
        raise ValueError("signal panel is bound to another ForecastCache artifact")
    if expected.get("quotes_manifest") != hashes["quotes_manifest"]:
        raise ValueError("signal panel is bound to another synthetic quotes artifact")
    if expected.get("clob_manifest") != hashes["clob_manifest"]:
        raise ValueError("signal panel is bound to another CLOB artifact")
    if quotes.get("source_sha256") != hashes["clob_manifest"]:
        raise ValueError("synthetic quotes are bound to another CLOB artifact")
    if forecast.get("source_sha256", {}).get("clob_manifest") != hashes["clob_manifest"]:
        raise ValueError("ForecastCache is bound to another CLOB artifact")
    return hashes


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
