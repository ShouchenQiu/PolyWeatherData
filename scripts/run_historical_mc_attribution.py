"""Strict expanding historical outcome attribution for MC components."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.storage.parquet_store import ParquetStore
from model.daily_high import fit_partial_day_bias
from model.mc import McSampler
from training.mc_attribution import (
    MC_ATTRIBUTION_VARIANTS,
    apply_attribution_variant,
    apply_partial_day_bias_variant,
    historical_attribution_folds,
)
from training.mc_calibration import fit_mc_calibration, local_calendar_day_groups


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-root", required=True, type=Path)
    parser.add_argument("--historical-data-store", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--mc-samples", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mc_samples <= 0:
        raise ValueError("mc-samples must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plan = historical_attribution_folds(args.crossfit_root)
    fold_ids = sorted({
        fold_id
        for boundary in plan
        for fold_id in (boundary.evaluation_fold_id, *boundary.calibration_fold_ids)
    })
    all_predictions = {
        fold_id: pd.read_parquet(args.crossfit_root / fold_id / "predictions.parquet")
        for fold_id in fold_ids
    }
    earliest = min(pd.to_datetime(frame["valid_time"], utc=True).min() for frame in all_predictions.values())
    latest = max(pd.to_datetime(frame["valid_time"], utc=True).max() for frame in all_predictions.values())
    raw = ParquetStore(args.historical_data_store / "obs").read(
        args.station, earliest - pd.Timedelta(days=2), latest + pd.Timedelta(days=2)
    )
    observations = pd.DataFrame({
        "valid_time": pd.to_datetime(raw["valid_time"], utc=True),
        "temperature_c": pd.to_numeric(raw["raw_tmpc_tenths"], errors="raise"),
    }).drop_duplicates("valid_time", keep="last")
    score_root = output / "fold_scores"
    score_root.mkdir(exist_ok=True)
    completed: list[pd.DataFrame] = []
    for boundary in plan:
        training = pd.concat(
            [pd.read_parquet(args.crossfit_root / fold_id / "predictions.parquet") for fold_id in boundary.calibration_fold_ids],
            ignore_index=True,
        )
        evaluation = all_predictions[boundary.evaluation_fold_id]
        for variant in MC_ATTRIBUTION_VARIANTS:
            score_path = score_root / (
                f"{boundary.evaluation_fold_id}_{variant.variant_id}.parquet"
            )
            if score_path.exists():
                completed.append(pd.read_parquet(score_path))
                print(f"[resume] {score_path.name}", flush=True)
                continue
            calibration = apply_attribution_variant(
                fit_mc_calibration(training, args.checkpoint, station_timezone=args.timezone, mc_samples=args.mc_samples),
                variant,
            )
            bias = apply_partial_day_bias_variant(
                fit_partial_day_bias(
                    hourly_predictions=training, observations=observations,
                    calibration=calibration, station_timezone=args.timezone,
                    mc_samples=args.mc_samples, seed=42,
                ), variant,
            )
            fold_rows: list[dict[str, object]] = []
            for group_index, group in enumerate(local_calendar_day_groups(evaluation, args.timezone)):
                ordered = group.sort_values("fxx")
                sampler = McSampler(n_samp=args.mc_samples, rng_seed=42 + group_index)
                sampler.set_calibration(calibration)
                sigma = (ordered["p90"].to_numpy(float) - ordered["p10"].to_numpy(float)) / (2.0 * 1.2815515655446004)
                samples = sampler.sample_daily_high(
                    ordered["base_temp_c"].to_numpy(float) + ordered["p50"].to_numpy(float),
                    sigma, max_bias=bias.for_forecast_hours(len(ordered)),
                    hour_indices=[int(value) - 1 for value in ordered["fxx"]],
                )
                actual = float((ordered["base_temp_c"] + ordered["target"]).max() * 1.8 + 32.0)
                fold_rows.append(_score_row(boundary.evaluation_fold_id, variant.variant_id, ordered, samples, actual))
            frame = pd.DataFrame(fold_rows)
            frame.to_parquet(score_path, index=False)
            completed.append(frame)
            print(
                f"[complete] {boundary.evaluation_fold_id} {variant.variant_id} "
                f"snapshots={len(frame)}",
                flush=True,
            )
    scores = pd.concat(completed, ignore_index=True)
    scores.to_parquet(output / "historical_scores.parquet", index=False)
    summary = _summary(scores)
    summary.to_csv(output / "historical_summary.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "data_class": "strict_expanding_historical_mc_attribution",
        "evaluation_folds": [row.evaluation_fold_id for row in plan],
        "excluded_warmup_fold": "fold-00", "mc_samples": args.mc_samples,
        "rows": len(scores), "variants": [v.variant_id for v in MC_ATTRIBUTION_VARIANTS],
    }, indent=2, sort_keys=True), encoding="ascii")
    print(summary.to_string(index=False), flush=True)
    return 0


def _score_row(fold_id: str, variant_id: str, group: pd.DataFrame, samples: np.ndarray, actual: float) -> dict[str, object]:
    integer = np.floor(samples + 0.5).astype(int)
    support = np.arange(integer.min(), integer.max() + 1)
    cdf = np.array([(integer <= value).mean() for value in support])
    truth_cdf = (support >= int(np.floor(actual + 0.5))).astype(float)
    return {
        "fold_id": fold_id, "variant_id": variant_id,
        "init_time": pd.Timestamp(group["init_time"].iloc[0]),
        "target_date": str(group["target_date"].iloc[0]),
        "actual_high_f": actual,
        "p10_f": float(np.quantile(samples, .1)), "p50_f": float(np.quantile(samples, .5)), "p90_f": float(np.quantile(samples, .9)),
        "rps": float(np.mean((cdf - truth_cdf) ** 2)),
        "true_integer_probability": float((integer == int(np.floor(actual + .5))).mean()),
    }


def _summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_id, group in scores.groupby("variant_id", sort=True):
        rows.append({
            "variant_id": variant_id, "snapshots": len(group), "event_days": group["target_date"].nunique(),
            "rps": float(group["rps"].mean()), "p50_mae_f": float((group["p50_f"] - group["actual_high_f"]).abs().mean()),
            "p10_p90_coverage": float(((group["actual_high_f"] >= group["p10_f"]) & (group["actual_high_f"] <= group["p90_f"])).mean()),
            "p10_p90_width_f": float((group["p90_f"] - group["p10_f"]).mean()),
            "true_integer_probability": float(group["true_integer_probability"].mean()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
