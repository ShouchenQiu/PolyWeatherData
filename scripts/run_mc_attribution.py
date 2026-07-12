"""Build parallel MC component-attribution ForecastCaches for April and May."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.observations import PointInTimeObservationDataset
from config.stations import STATIONS
from data.clob.history import HistoricalClobDataset
from data.forecast.availability import HrrrAvailabilityDataset
from model.daily_high import PartialDayBiasCalibration, fit_partial_day_bias
from model.mc_calibration import McCalibration
from training.backtest_forecast_generation import (
    build_forecast_cache,
    fit_oof_calibrations,
    mc_calibration_sha256,
    partial_bias_sha256,
)
from training.mc_attribution import (
    MC_ATTRIBUTION_VARIANTS,
    McAttributionVariant,
    apply_attribution_variant,
    apply_partial_day_bias_variant,
)
from training.mc_calibration import fit_mc_calibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--april-predictions", required=True, type=Path)
    parser.add_argument("--may-predictions", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--bias-predictions", required=True, type=Path)
    parser.add_argument("--availability-root", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--clob-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--mc-samples", type=int, default=2_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.folds < 2 or args.mc_samples <= 0:
        raise ValueError("folds must be >= 2 and mc-samples must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    station = STATIONS[args.station]
    april = pd.read_parquet(args.april_predictions)
    may = pd.read_parquet(args.may_predictions)
    for frame in (april, may):
        frame["init_time"] = pd.to_datetime(frame["init_time"], utc=True)
    observations = PointInTimeObservationDataset.load(args.observations_root).observations
    availability = HrrrAvailabilityDataset.load(args.availability_root).cycles
    clob = HistoricalClobDataset.load(args.clob_root)
    markets = clob.markets.merge(
        clob.events[["event_id", "target_date", "event_end_at"]],
        on="event_id", how="inner", validate="many_to_one",
    )
    base = McCalibration.load(args.calibration_root, args.checkpoint)
    bias_predictions = pd.read_parquet(args.bias_predictions)
    source = {
        "april_predictions": _sha256(args.april_predictions),
        "may_predictions": _sha256(args.may_predictions),
        "bias_predictions": _sha256(args.bias_predictions),
        "availability_manifest": _sha256(args.availability_root / "manifest.json"),
        "observations_manifest": _sha256(args.observations_root / "manifest.json"),
        "clob_manifest": _sha256(args.clob_root / "manifest.json"),
        "checkpoint": _sha256(args.checkpoint),
        "calibration_bundle": _calibration_files_sha256(args.calibration_root),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "data_class": "mc_component_attribution",
        "station_id": station.station_id,
        "variants": {},
        "source_sha256": source,
    }
    for variant in MC_ATTRIBUTION_VARIANTS:
        april_cache = _build_april_oof_cache(
            april=april, observations=observations, availability=availability,
            markets=markets, checkpoint=args.checkpoint, station_id=station.station_id,
            station_timezone=station.timezone, folds=args.folds,
            mc_samples=args.mc_samples, variant=variant,
        )
        may_cache = _build_may_fixed_cache(
            may=may, bias_predictions=bias_predictions, observations=observations,
            availability=availability, markets=markets, calibration=base,
            station_id=station.station_id, station_timezone=station.timezone,
            mc_samples=args.mc_samples, variant=variant,
        )
        variant_root = output / variant.variant_id
        april_manifest = april_cache.save(
            variant_root / "april_oof_forecast_cache", source_sha256=source
        )
        may_manifest = may_cache.save(
            variant_root / "may_model_test_forecast_cache", source_sha256=source
        )
        manifest["variants"][variant.variant_id] = {
            "use_sigma_scales": variant.use_sigma_scales,
            "use_correlation": variant.use_correlation,
            "use_partial_day_bias": variant.use_partial_day_bias,
            "april_rows": april_manifest["row_counts"],
            "may_rows": may_manifest["row_counts"],
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii"
    )
    print(json.dumps(manifest["variants"], indent=2, sort_keys=True), flush=True)
    return 0


def _build_april_oof_cache(**kwargs: object):
    variant = kwargs.pop("variant")
    assert isinstance(variant, McAttributionVariant)
    checkpoint = kwargs.pop("checkpoint")
    station_timezone = kwargs.pop("station_timezone")
    folds = kwargs.pop("folds")
    mc_samples = kwargs.pop("mc_samples")
    april = kwargs.pop("april")
    observations = kwargs["observations"]

    def fit_calibration(frame, checkpoint_path, *, station_timezone, mc_samples):
        return apply_attribution_variant(
            fit_mc_calibration(
                frame, checkpoint_path, station_timezone=station_timezone,
                mc_samples=mc_samples,
            ), variant,
        )

    def fit_bias(*, hourly_predictions, observations, calibration, station_timezone, mc_samples, seed):
        if not variant.use_partial_day_bias:
            return PartialDayBiasCalibration.zero()
        return fit_partial_day_bias(
            hourly_predictions=hourly_predictions, observations=observations,
            calibration=calibration, station_timezone=station_timezone,
            mc_samples=mc_samples, seed=seed,
        )

    lineages, assignments, calibrations, biases = fit_oof_calibrations(
        april, observations=observations, checkpoint_path=checkpoint,
        station_timezone=station_timezone, folds=folds, mc_samples=mc_samples,
        fit_calibration=fit_calibration, fit_partial_bias=fit_bias,
    )
    _tag_lineages(lineages, assignments, variant)
    prefix = f"mc-attribution-{variant.variant_id}-"
    calibrations = {prefix + key: value for key, value in calibrations.items()}
    biases = {prefix + key: value for key, value in biases.items()}
    return build_forecast_cache(
        hourly_predictions=april, lineages=lineages, cycle_lineages=assignments,
        calibrations=calibrations, partial_biases=biases,
        station_timezone=station_timezone, target_start="2026-04-01",
        target_end="2026-04-30", **kwargs,
    )


def _build_may_fixed_cache(**kwargs: object):
    variant = kwargs.pop("variant")
    assert isinstance(variant, McAttributionVariant)
    calibration = apply_attribution_variant(kwargs.pop("calibration"), variant)
    station_timezone = kwargs.pop("station_timezone")
    mc_samples = kwargs.pop("mc_samples")
    may = kwargs.pop("may")
    bias_predictions = kwargs.pop("bias_predictions")
    observations = kwargs["observations"]
    bias = fit_partial_day_bias(
        hourly_predictions=bias_predictions, observations=observations,
        calibration=calibration, station_timezone=station_timezone,
        mc_samples=mc_samples, seed=42,
    )
    bias = apply_partial_day_bias_variant(bias, variant)
    lineage_id = f"mc-attribution-{variant.variant_id}-model-test"
    lineages = pd.DataFrame([{
        "lineage_id": lineage_id,
        "checkpoint_sha256": calibration.model_sha256,
        "calibration_sha256": hashlib.sha256(
            (mc_calibration_sha256(calibration) + partial_bias_sha256(bias)).encode("ascii")
        ).hexdigest(),
        "split_role": "model_test", "mc_seed": 42, "mc_samples": mc_samples,
    }])
    assignments = pd.DataFrame({"init_time": sorted(may["init_time"].unique()), "lineage_id": lineage_id})
    return build_forecast_cache(
        hourly_predictions=may, lineages=lineages, cycle_lineages=assignments,
        calibrations={lineage_id: calibration}, partial_biases={lineage_id: bias},
        station_timezone=station_timezone, target_start="2026-05-01",
        target_end="2026-05-31", **kwargs,
    )


def _tag_lineages(
    lineages: pd.DataFrame,
    assignments: pd.DataFrame,
    variant: McAttributionVariant,
) -> None:
    prefix = f"mc-attribution-{variant.variant_id}-"
    lineages["lineage_id"] = prefix + lineages["lineage_id"].astype(str)
    assignments["lineage_id"] = prefix + assignments["lineage_id"].astype(str)


def _calibration_files_sha256(root: Path) -> str:
    hasher = hashlib.sha256()
    for name in ("mc_calibration.json", "mc_calibration.npz"):
        hasher.update((root / name).read_bytes())
    return hasher.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
