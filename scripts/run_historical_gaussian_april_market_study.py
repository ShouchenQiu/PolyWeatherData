"""Freeze a pre-April Gaussian and compare it with MC and April CLOB prices."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest.forecast_cache import ForecastCache
from backtest.observations import PointInTimeObservationDataset
from backtest.signal_study import (
    load_signal_study_artifact,
    run_forecast_signal_study,
    score_signal_panel,
    signal_study_id,
)
from backtest.study import StudyInputs, validate_study_inputs
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from training.backtest_forecast_generation import build_gaussian_forecast_cache
from training.daily_high_gaussian import (
    FutureMaxExamples,
    build_future_max_examples,
    fit_gaussian_future_max_model,
    predict_gaussian_examples,
    require_training_target_dates_before,
)


APRIL_START = "2026-04-01"
APRIL_END = "2026-04-30"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-examples-root", required=True, type=Path)
    parser.add_argument("--april-predictions", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--april-forecast-root", required=True, type=Path)
    parser.add_argument("--quotes-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--mc-signal-root", required=True, type=Path)
    parser.add_argument("--tft-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.epochs <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("epochs and bootstrap samples must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    historical_examples = _load_historical_examples(args.historical_examples_root)
    require_training_target_dates_before(
        historical_examples.frame, cutoff_date=APRIL_START
    )
    historical_manifest_sha = _sha256(
        args.historical_examples_root / "manifest.json"
    )
    checkpoint_sha = _sha256(args.tft_checkpoint)
    frozen_model = fit_gaussian_future_max_model(
        historical_examples.frame,
        feature_columns=historical_examples.feature_columns,
        tft_checkpoint_sha256=checkpoint_sha,
        training_manifest_sha256=historical_manifest_sha,
        epochs=args.epochs,
        seed=42,
    )
    frozen_model.save(output / "frozen_pre_april_model")
    frozen_model_sha = _sha256(
        output / "frozen_pre_april_model" / "future_max_gaussian.json"
    )

    station = STATIONS[args.station]
    source_cache = ForecastCache.load(args.april_forecast_root)
    april_snapshots = _target_range(
        source_cache.snapshots, start=APRIL_START, end=APRIL_END
    )
    observations = PointInTimeObservationDataset.load(
        args.observations_root
    ).observations
    april_examples = build_future_max_examples(
        hourly_predictions=pd.read_parquet(args.april_predictions),
        observations=observations,
        snapshots=april_snapshots,
        station_timezone=station.timezone,
    )
    _require_exact_april(april_examples.frame)
    april_examples.save(
        output / "april_inference_examples",
        source_sha256={
            "april_predictions": _sha256(args.april_predictions),
            "april_forecast_manifest": _sha256(
                args.april_forecast_root / "manifest.json"
            ),
            "observations_manifest": _sha256(
                args.observations_root / "manifest.json"
            ),
            "frozen_model": frozen_model_sha,
        },
    )
    evaluation = predict_gaussian_examples(
        frozen_model, april_examples, lineage_label="frozen-pre-april"
    )
    evaluation.save(
        output / "april_frozen_evaluation",
        split_role="oof_development",
        source_sha256={"frozen_model": frozen_model_sha},
    )

    clob = HistoricalClobDataset.load(args.clob_root)
    markets = clob.markets.merge(
        clob.events[["event_id", "target_date"]],
        on="event_id",
        how="inner",
        validate="many_to_one",
    )
    markets["target_date"] = markets["target_date"].astype(str)
    candidate_cache = build_gaussian_forecast_cache(
        evaluation=evaluation,
        source_snapshots=source_cache.snapshots,
        markets=markets,
        station_id=station.station_id,
        split_role="oof_development",
        tft_checkpoint_sha256=checkpoint_sha,
        artifact_sha256=frozen_model_sha,
    )
    candidate_cache.save(
        output / "april_gaussian_forecast_cache",
        source_sha256={
            "frozen_model": frozen_model_sha,
            "clob_manifest": _sha256(args.clob_root / "manifest.json"),
            "source_forecast_manifest": _sha256(
                args.april_forecast_root / "manifest.json"
            ),
        },
    )

    candidate_signal_root = output / "april_signal_vs_clob"
    signal_manifest = _run_signal_study(
        forecast_root=output / "april_gaussian_forecast_cache",
        quotes_root=args.quotes_root,
        clob_root=args.clob_root,
        actuals=args.actual_highs,
        station_id=station.station_id,
        output=candidate_signal_root,
        bootstrap_samples=args.bootstrap_samples,
    )
    candidate_signal = load_signal_study_artifact(candidate_signal_root)
    mc_signal = load_signal_study_artifact(args.mc_signal_root)
    paired = _paired_candidate_vs_mc(
        candidate_signal.panel, mc_signal.panel, args.bootstrap_samples
    )

    report = {
        "schema_version": 1,
        "data_class": "frozen_pre_april_gaussian_market_study",
        "training": {
            "target_date_range": [
                str(historical_examples.frame["target_date"].min()),
                str(historical_examples.frame["target_date"].max()),
            ],
            "target_day_count": int(
                historical_examples.frame["target_date"].nunique()
            ),
            "examples": int(len(historical_examples.frame)),
            "cutoff_date": APRIL_START,
            "historical_examples_manifest_sha256": historical_manifest_sha,
            "frozen_model_sha256": frozen_model_sha,
        },
        "april": {
            "target_date_range": [APRIL_START, APRIL_END],
            "inference_examples": int(len(april_examples.frame)),
            "candidate_signal_manifest": signal_manifest,
            "candidate_vs_clob": candidate_signal.summary,
            "mc_vs_clob": mc_signal.summary,
            "candidate_vs_mc_paired": paired,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="ascii"
    )
    print(json.dumps(paired["comparison"], indent=2, sort_keys=True), flush=True)
    return 0


def _load_historical_examples(root: Path) -> FutureMaxExamples:
    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    filenames = sorted(manifest.get("sha256", {}))
    if not filenames or any(not name.endswith(".parquet") for name in filenames):
        raise ValueError("historical examples manifest has no parquet inputs")
    frames = []
    for filename in filenames:
        path = root / filename
        if manifest["sha256"][filename] != _sha256(path):
            raise ValueError(f"historical example hash mismatch: {filename}")
        frames.append(pd.read_parquet(path))
    examples = FutureMaxExamples(pd.concat(frames, ignore_index=True))
    if examples.frame.duplicated(["init_time", "target_date"]).any():
        raise ValueError("historical examples contain duplicate point-in-time rows")
    return examples


def _target_range(frame: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    dates = frame["target_date"].astype(str)
    return frame[(dates >= start) & (dates <= end)].copy()


def _require_exact_april(frame: pd.DataFrame) -> None:
    dates = frame["target_date"].astype(str)
    if frame.empty or dates.min() < APRIL_START or dates.max() > APRIL_END:
        raise ValueError("April inference examples fall outside the frozen window")


def _run_signal_study(
    *,
    forecast_root: Path,
    quotes_root: Path,
    clob_root: Path,
    actuals: Path,
    station_id: str,
    output: Path,
    bootstrap_samples: int,
) -> dict:
    inputs = StudyInputs(
        forecast_root=forecast_root,
        quotes_root=quotes_root,
        clob_root=clob_root,
        actuals_path=actuals,
    )
    validation = validate_study_inputs(
        study_id="signal-pending",
        station_id=station_id,
        start_target=APRIL_START,
        end_target=APRIL_END,
        inputs=inputs,
        expected_split_role="oof_development",
    )
    study_id = signal_study_id(
        station_id=station_id,
        start_target=APRIL_START,
        end_target=APRIL_END,
        input_sha256=validation.input_sha256,
        ece_bins=10,
        probability_floor=1e-6,
        bootstrap_samples=bootstrap_samples,
        seed=42,
    )
    return run_forecast_signal_study(
        replace(validation, study_id=study_id),
        output=output,
        bootstrap_samples=bootstrap_samples,
        seed=42,
    )


def _paired_candidate_vs_mc(
    candidate: pd.DataFrame,
    mc: pd.DataFrame,
    bootstrap_samples: int,
) -> dict:
    keys = ["event_id", "target_date", "timestamp", "condition_id"]
    candidate_columns = keys + [
        "bucket_order", "model_probability", "yes_outcome", "lead_band"
    ]
    mc_columns = keys + ["model_probability"]
    merged = candidate[candidate_columns].merge(
        mc[mc_columns], on=keys, how="inner", validate="one_to_one",
        suffixes=("_candidate", "_mc"),
    )
    if len(merged) != len(candidate) or len(merged) != len(mc):
        raise ValueError("candidate and MC CLOB panels do not have identical vectors")
    panel = merged.rename(columns={
        "model_probability_candidate": "model_probability",
        "model_probability_mc": "midpoint_probability",
    })
    vector_keys = ["event_id", "target_date", "timestamp"]
    panel["midpoint_sum"] = panel.groupby(vector_keys)[
        "midpoint_probability"
    ].transform("sum")
    panel["midpoint_overround"] = panel["midpoint_sum"] - 1.0
    scored = score_signal_panel(panel, bootstrap_samples=bootstrap_samples, seed=42)
    return scored.summary


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
