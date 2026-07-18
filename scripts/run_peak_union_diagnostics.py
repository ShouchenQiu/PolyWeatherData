"""Build read-only peak-state and contiguous-union diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.observations import PointInTimeObservationDataset
from backtest.peak_union_diagnostics import (
    assign_peak_state_groups,
    attach_forecast_stability,
    attach_peak_outcomes,
    build_contiguous_union_panel,
    build_peak_snapshot_diagnostics,
    score_union_panel,
    summarize_peak_groups,
    write_peak_union_artifact,
)
from backtest.signal_study import load_signal_study_artifact
from config.stations import STATIONS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-cache", required=True, type=Path)
    parser.add_argument("--signal-root", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path, action="append")
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--state-contract", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if args.output.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output}")
    station = STATIONS[args.station]
    print(f"[start] peak and union diagnostics output={args.output}", flush=True)
    cache = ForecastCache.load(args.forecast_cache)
    signal = load_signal_study_artifact(args.signal_root)
    observations = PointInTimeObservationDataset.load(args.observations_root).observations
    predictions = _load_predictions(args.prediction)
    print(f"[complete] inputs snapshots={len(cache.snapshots)} panel_rows={len(signal.panel)} predictions={len(predictions)}", flush=True)

    diagnostics = build_peak_snapshot_diagnostics(
        snapshots=cache.snapshots,
        predictions=predictions,
        observations=observations,
        station_timezone=station.timezone,
    )
    snapshot_values = cache.snapshots.loc[:, [
        "snapshot_id", "p10_f", "p50_f", "p90_f", "coverage_type",
        "observed_hour_count", "forecast_hour_count",
    ]]
    diagnostics = diagnostics.drop(columns=[column for column in (
        "coverage_type", "observed_hour_count", "forecast_hour_count",
    ) if column in diagnostics]).merge(snapshot_values, on="snapshot_id", how="inner", validate="one_to_one")
    diagnostics = attach_forecast_stability(diagnostics)
    thresholds = _load_contract(args.state_contract)
    diagnostics, state_contract = assign_peak_state_groups(diagnostics, thresholds=thresholds)
    print(f"[complete] ex-ante peak states snapshots={len(diagnostics)}", flush=True)

    actuals = pd.read_csv(args.actual_highs)
    peak_outcomes = attach_peak_outcomes(
        diagnostics, observations=observations, settlement_actuals=actuals,
        station_timezone=station.timezone,
    )
    peak_outcomes = _attach_signal_outcomes(peak_outcomes, signal.panel, signal.scores_by_tick)
    peak_outcomes["p50_abs_error_f"] = (peak_outcomes["p50_f"] - peak_outcomes["settlement_actual_high_f"]).abs()
    peak_outcomes["coverage80"] = (
        (peak_outcomes["settlement_actual_high_f"] >= peak_outcomes["p10_f"])
        & (peak_outcomes["settlement_actual_high_f"] <= peak_outcomes["p90_f"])
    ).astype(float)
    peak_summary = summarize_peak_groups(peak_outcomes)
    print(f"[complete] peak outcomes rows={len(peak_outcomes)} labeled_peak_times={int(peak_outcomes['peak_time_label_available'].fillna(False).sum())}", flush=True)

    unions = build_contiguous_union_panel(signal.panel)
    union_scores, union_strata, union_summary = score_union_panel(
        unions, bootstrap_samples=args.bootstrap_samples,
    )
    union_scores = union_scores.merge(
        diagnostics.drop(columns=["init_time", "available_at", "target_date"], errors="ignore"),
        on="snapshot_id", how="left", validate="many_to_one",
    )
    report = {
        "schema_version": 1,
        "station_id": station.station_id,
        "scope": "normalized_synthetic_midpoint_only",
        "model_source": "checkpoint_aligned_static_mc",
        "state_contract": state_contract,
        "counts": {
            "snapshots": int(len(diagnostics)), "peak_outcomes": int(len(peak_outcomes)),
            "union_scores": int(len(union_scores)),
        },
        "union_summary": union_summary,
        "peak_group_summary": peak_summary.to_dict(orient="records"),
        "interpretation_focus": [
            "sigma_correlation_interaction", "interval_coverage", "p50_error",
            "forecast_only_vs_observed_plus_forecast",
        ],
    }
    manifest = write_peak_union_artifact(
        output=args.output, snapshot_diagnostics=diagnostics, peak_outcomes=peak_outcomes,
        union_scores=union_scores, union_strata=union_strata, report=report,
        input_sha256=_input_hashes(args),
    )
    print(f"[complete] union scores={len(union_scores)} event_days={union_summary['event_days']}", flush=True)
    print(f"[done] peak and union diagnostics output={args.output} manifest={manifest['data_class']}", flush=True)
    return 0


def _load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        extrapolated = frame.get("is_extrapolated", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        frames.append(frame[(frame["weight"].astype(float) > 0) & ~extrapolated].copy())
    return pd.concat(frames, ignore_index=True)


def _attach_signal_outcomes(diagnostics: pd.DataFrame, panel: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    lookup = panel[["event_id", "target_date", "timestamp", "snapshot_id"]].drop_duplicates()
    model = scores[scores["source"] == "model_raw"].drop(columns="source")
    model = model.merge(lookup, on=["event_id", "target_date", "timestamp"], how="inner", validate="one_to_one")
    model = model.groupby("snapshot_id", as_index=False)[
        ["normalized_rps", "true_bucket_probability"]
    ].mean()
    return diagnostics.merge(model, on="snapshot_id", how="left", validate="one_to_one")


def _load_contract(path: Path | None) -> dict[str, list[float]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="ascii"))
    return payload["state_contract"] if "state_contract" in payload else payload


def _input_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "forecast_cache_manifest": args.forecast_cache / "manifest.json",
        "signal_manifest": args.signal_root / "manifest.json",
        "observations_manifest": args.observations_root / "manifest.json",
        "actual_highs": args.actual_highs,
    }
    paths.update({f"prediction_{index:02d}": path for index, path in enumerate(args.prediction)})
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
