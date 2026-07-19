"""Validate TFT peak margin as a strict historical cross-fit risk state."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.peak_margin_crossfit import (
    build_peak_margin_score_panel,
    fit_expanding_peak_margin_contracts,
    summarize_peak_margin_crossfit,
    validate_peak_margin_input_lineage,
    write_peak_margin_crossfit_artifact,
)
from backtest.signal_study import load_signal_study_artifact


_DEFAULT_PANEL = Path(
    "data_store/backtest/historical_market_panel/KLGA/"
    "static_mc_checkpoint_aligned_v1"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-cache",
        type=Path,
        default=_DEFAULT_PANEL / "forecast_cache",
    )
    parser.add_argument(
        "--signal-root",
        type=Path,
        default=_DEFAULT_PANEL / "forecast_signal_study",
    )
    parser.add_argument(
        "--peak-diagnostics-root",
        type=Path,
        default=Path(
            "data_store/backtest/peak_union_diagnostics/KLGA/checkpoint_aligned_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data_store/backtest/peak_margin_crossfit/KLGA/checkpoint_aligned_v1"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if args.output.exists():
        raise FileExistsError(f"peak-margin output already exists: {args.output}")
    print(f"[start] peak-margin historical cross-fit output={args.output}", flush=True)
    input_hashes = validate_peak_margin_input_lineage(
        forecast_cache_root=args.forecast_cache,
        signal_root=args.signal_root,
        peak_diagnostics_root=args.peak_diagnostics_root,
    )
    cache = ForecastCache.load(args.forecast_cache)
    signal = load_signal_study_artifact(args.signal_root)
    diagnostics = pd.read_parquet(
        args.peak_diagnostics_root / "snapshot_diagnostics.parquet"
    )
    print(
        f"[complete] inputs snapshots={len(cache.snapshots)} "
        f"signal_vectors={len(signal.scores_by_tick)} "
        f"peak_rows={len(diagnostics)}",
        flush=True,
    )
    panel = build_peak_margin_score_panel(
        signal_panel=signal.panel,
        scores_by_tick=signal.scores_by_tick,
        snapshots=cache.snapshots,
        peak_diagnostics=diagnostics,
    )
    scored, contracts = fit_expanding_peak_margin_contracts(panel)
    print(
        f"[complete] strict panel snapshots={len(scored)} "
        f"event_days={scored['target_date'].nunique()} contracts={len(contracts)}",
        flush=True,
    )
    result = summarize_peak_margin_crossfit(
        scored,
        contracts=contracts,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report = {
        **result.report,
        "station_id": "KLGA",
        "scope": "full_bucket_pmf_normalized_synthetic_midpoint_only",
        "model_source": "checkpoint_aligned_static_mc",
        "data_regime": "retrospective_prequential_known_development",
        "counts": {
            "input_market_snapshots": int(len(panel)),
            "evaluation_snapshots": int(len(scored)),
            "evaluation_event_days": int(scored["target_date"].nunique()),
            "forecast_only_snapshots": int(
                (scored["coverage_type"] == "forecast_only").sum()
            ),
            "partial_day_snapshots": int(
                (scored["coverage_type"] != "forecast_only").sum()
            ),
            "fold_contracts": int(len(contracts)),
        },
        "protocol": {
            "peak_margin": "highest_future_tft_p50_f_minus_second_highest_future_tft_p50_f",
            "primary_coverage": "forecast_only",
            "warmup_fold": "fold-00",
            "evaluation_folds": sorted(scored["fold_id"].astype(str).unique()),
            "high_state": "earlier_forecast_only_expanding_q75",
            "sensitivity_state": "fold00_fixed_q75",
            "market_tick": "first_legal_tick_per_event_and_forecast_snapshot",
            "primary_estimand_sign": "negative_model_minus_midpoint_is_better",
            "bootstrap": "event_day_clustered_fold_stratified_90pct",
            "bootstrap_samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
        },
        "limitations": {
            "known_development_data": True,
            "independent_holdout": False,
            "april_may_2026_consumed": False,
            "june_2026_consumed": False,
            "strategy_or_pnl_measured": False,
            "threshold_search_performed": False,
        },
        "primary_strata": _primary_strata(result.fold_summaries),
    }
    manifest = write_peak_margin_crossfit_artifact(
        output=args.output,
        frames={
            "snapshot_scores": scored,
            "fold_contracts": contracts,
            "fold_summaries": result.fold_summaries,
            "continuous_effects": result.continuous_effects,
        },
        report=report,
        input_sha256=input_hashes,
    )
    print(
        f"[complete] report summaries={len(result.fold_summaries)} "
        f"continuous_effects={len(result.continuous_effects)}",
        flush=True,
    )
    print(
        f"[done] peak-margin historical cross-fit output={args.output} "
        f"manifest={manifest['data_class']}",
        flush=True,
    )
    return 0


def _primary_strata(summaries: pd.DataFrame) -> list[dict[str, object]]:
    selected = summaries[
        (summaries["threshold_mode"] == "expanding_q75")
        & (summaries["coverage_scope"] == "forecast_only_primary")
        & (summaries["fold_id"] == "pooled")
        & (summaries["stratum_type"].isin(["horizon", "month", "peak_boundary"]))
    ].copy()
    selected["sufficient_event_days"] = selected["event_days"] >= 20
    columns = [
        "stratum_type", "stratum_value", "margin_group", "event_days",
        "snapshots", "relative_rps", "relative_rps_ci90_low",
        "relative_rps_ci90_high", "sufficient_event_days",
    ]
    return selected.loc[:, columns].to_dict(orient="records")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
