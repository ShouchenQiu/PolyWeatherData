"""Build a strict historical static-MC market panel and signal report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.signal_study import run_forecast_signal_study, signal_study_id
from backtest.study import StudyInputs, validate_study_inputs
from backtest.observations import PointInTimeObservationDataset
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from data.forecast.availability import HrrrAvailabilityDataset
from training.historical_market_panel import (
    build_checkpoint_aligned_historical_market_panel_cache,
    build_historical_market_panel_cache,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-root", required=True, type=Path)
    parser.add_argument("--checkpoint-aligned-tune-root", type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--availability-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--mc-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mc_samples <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("sample counts must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"historical panel output already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    station = STATIONS[args.station]
    cache_root = args.output / "forecast_cache"
    audit_path = args.output / "fold_audit.parquet"
    print(f"[start] historical market panel output={args.output}", flush=True)
    if cache_root.exists():
        if not args.resume:
            raise FileExistsError(cache_root)
        from backtest.forecast_cache import ForecastCache
        cache = ForecastCache.load(cache_root)
        print(
            f"[complete] historical ForecastCache resume-existing "
            f"snapshots={len(cache.snapshots)}",
            flush=True,
        )
    else:
        print("[start] strict expanding static-MC ForecastCache", flush=True)
        observations = PointInTimeObservationDataset.load(
            args.observations_root
        ).observations
        availability = HrrrAvailabilityDataset.load(args.availability_root).cycles
        clob = HistoricalClobDataset.load(args.clob_root)
        markets = clob.markets.merge(
            clob.events[["event_id", "target_date", "event_end_at"]],
            on="event_id",
            how="inner",
            validate="many_to_one",
        )
        build_cache = (
            build_checkpoint_aligned_historical_market_panel_cache
            if args.checkpoint_aligned_tune_root is not None
            else build_historical_market_panel_cache
        )
        kwargs = {
            "crossfit_root": args.crossfit_root,
            "observations": observations,
            "availability": availability,
            "markets": markets,
            "station_id": station.station_id,
            "station_timezone": station.timezone,
            "mc_samples": args.mc_samples,
            "progress_callback": lambda fold_id, snapshots, probabilities: print(
                f"[progress] historical ForecastCache fold={fold_id} "
                f"snapshots={snapshots} probabilities={probabilities}",
                flush=True,
            ),
        }
        if args.checkpoint_aligned_tune_root is not None:
            kwargs["tune_predictions_root"] = args.checkpoint_aligned_tune_root
        result = build_cache(**kwargs)
        source_hashes = _source_hashes(args)
        result.cache.save(cache_root, source_sha256=source_hashes)
        result.fold_audit.to_parquet(audit_path, index=False)
        print(
            f"[complete] historical ForecastCache snapshots={len(result.cache.snapshots)} "
            f"probabilities={len(result.cache.probabilities)} folds={len(result.fold_audit)}",
            flush=True,
        )

    signal_root = args.output / "forecast_signal_study"
    if signal_root.exists():
        if not args.resume:
            raise FileExistsError(signal_root)
        print(f"[complete] signal study resume-existing output={signal_root}", flush=True)
    else:
        print("[start] historical market signal scoring", flush=True)
        clob = HistoricalClobDataset.load(args.clob_root)
        from backtest.forecast_cache import ForecastCache

        score_cache = ForecastCache.load(cache_root)
        start, end = _score_target_range(score_cache.snapshots)
        inputs = StudyInputs(
            forecast_root=cache_root,
            quotes_root=args.quotes_root,
            clob_root=args.clob_root,
            actuals_path=args.actual_highs,
        )
        validation = validate_study_inputs(
            study_id="historical-market-panel-pending",
            station_id=station.station_id,
            start_target=start,
            end_target=end,
            inputs=inputs,
            expected_split_role="oof_development",
        )
        study_id = signal_study_id(
            station_id=station.station_id,
            start_target=start,
            end_target=end,
            input_sha256=validation.input_sha256,
            ece_bins=10,
            probability_floor=1e-6,
            bootstrap_samples=args.bootstrap_samples,
            seed=42,
        )
        run_forecast_signal_study(
            validation.__class__(**{**validation.__dict__, "study_id": study_id}),
            output=signal_root,
            bootstrap_samples=args.bootstrap_samples,
            seed=42,
        )
        _write_strata(signal_root, cache_root, args.output / "stratified_scores.parquet")
        print(f"[complete] historical market signal study output={signal_root}", flush=True)
    print(f"[done] historical market panel output={args.output}", flush=True)
    return 0


def _source_hashes(args: argparse.Namespace) -> dict[str, str]:
    hashes = {
        "crossfit_manifest": _sha256(args.crossfit_root / "manifest.json"),
        "observations_manifest": _sha256(args.observations_root / "manifest.json"),
        "availability_manifest": _sha256(args.availability_root / "manifest.json"),
        "clob_manifest": _sha256(args.clob_root / "manifest.json"),
        "quotes_manifest": _sha256(args.quotes_root / "manifest.json"),
        "actuals_manifest": _sha256(args.actual_highs.with_suffix(args.actual_highs.suffix + ".manifest.json")),
    }
    if args.checkpoint_aligned_tune_root is not None:
        hashes["checkpoint_aligned_tune_manifest"] = _sha256(
            args.checkpoint_aligned_tune_root / "manifest.json"
        )
    return hashes


def _score_target_range(snapshots: pd.DataFrame) -> tuple[str, str]:
    """Score only target days with strict expanding-MC ForecastCache evidence."""
    if snapshots.empty or "target_date" not in snapshots:
        raise ValueError("historical ForecastCache has no target-date coverage")
    dates = snapshots["target_date"].astype(str)
    return str(dates.min()), str(dates.max())


def _write_strata(signal_root: Path, cache_root: Path, output: Path) -> None:
    """Persist descriptive slices without choosing a rule from their outcomes."""
    from backtest.forecast_cache import ForecastCache

    scores = pd.read_parquet(signal_root / "scores_by_tick.parquet")
    panel = pd.read_parquet(signal_root / "panel.parquet")
    snapshots = ForecastCache.load(cache_root).snapshots
    keys = ["event_id", "target_date", "timestamp", "snapshot_id"]
    lookup = panel[keys].drop_duplicates(["event_id", "target_date", "timestamp"])
    merged = scores.merge(
        lookup,
        on=["event_id", "target_date", "timestamp"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged.merge(
        snapshots[["snapshot_id", "coverage_type", "forecast_hour_count"]],
        on="snapshot_id",
        how="inner",
        validate="many_to_one",
    )
    dates = pd.to_datetime(merged["target_date"])
    merged["month"] = dates.dt.to_period("M").astype(str)
    merged["season"] = dates.dt.month.map(
        {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    )
    grouped = merged.groupby(
        ["source", "month", "season", "coverage_type", "lead_band"],
        as_index=False,
    ).agg(
        event_days=("target_date", "nunique"),
        vectors=("target_date", "size"),
        multiclass_brier=("multiclass_brier", "mean"),
        log_score=("log_score", "mean"),
        normalized_rps=("normalized_rps", "mean"),
        true_bucket_probability=("true_bucket_probability", "mean"),
    )
    grouped.to_parquet(output, index=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
