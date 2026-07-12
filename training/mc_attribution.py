"""Training-only component variants for MC attribution studies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model.daily_high import PartialDayBiasCalibration
from model.mc_calibration import McCalibration


@dataclass(frozen=True)
class McAttributionVariant:
    """One orthogonal switch over the calibrated MC distribution contract."""

    variant_id: str
    use_sigma_scales: bool = True
    use_correlation: bool = True
    use_partial_day_bias: bool = True


MC_ATTRIBUTION_VARIANTS: tuple[McAttributionVariant, ...] = (
    McAttributionVariant("current"),
    McAttributionVariant("no_sigma_scales", use_sigma_scales=False),
    McAttributionVariant("diagonal_dependence", use_correlation=False),
    McAttributionVariant("no_partial_day_bias", use_partial_day_bias=False),
)


def attribution_variant(variant_id: str) -> McAttributionVariant:
    for variant in MC_ATTRIBUTION_VARIANTS:
        if variant.variant_id == variant_id:
            return variant
    raise ValueError(f"unknown MC attribution variant: {variant_id}")


def apply_attribution_variant(
    calibration: McCalibration,
    variant: McAttributionVariant,
) -> McCalibration:
    """Return a contract-valid calibration with only the selected component off."""
    return McCalibration(
        correlation=(
            calibration.correlation
            if variant.use_correlation
            else np.eye(calibration.correlation.shape[0])
        ),
        sigma_scales=(
            calibration.sigma_scales
            if variant.use_sigma_scales
            else np.ones_like(calibration.sigma_scales)
        ),
        max_bias_f=calibration.max_bias_f,
        model_sha256=calibration.model_sha256,
        fit_split=calibration.fit_split,
        fit_start=calibration.fit_start,
        fit_end=calibration.fit_end,
        sample_count=calibration.sample_count,
        station_timezone=calibration.station_timezone,
        target_window=calibration.target_window,
    )


def apply_partial_day_bias_variant(
    bias: PartialDayBiasCalibration,
    variant: McAttributionVariant,
) -> PartialDayBiasCalibration:
    """Remove only the final remaining-hours correction when requested."""
    return bias if variant.use_partial_day_bias else PartialDayBiasCalibration.zero()


@dataclass(frozen=True)
class HistoricalAttributionFold:
    """One strict expanding calibration/evaluation boundary."""

    evaluation_fold_id: str
    calibration_fold_ids: tuple[str, ...]


def historical_attribution_folds(
    crossfit_root: Path | str,
) -> tuple[HistoricalAttributionFold, ...]:
    """Evaluate each OOF block only with calibration evidence from prior blocks."""
    root = Path(crossfit_root)
    fold_ids = tuple(sorted(
        path.name for path in root.glob("fold-*")
        if (path / "predictions.parquet").is_file()
    ))
    if len(fold_ids) < 2:
        raise ValueError("historical MC attribution requires at least two OOF folds")
    return tuple(
        HistoricalAttributionFold(
            evaluation_fold_id=fold_id,
            calibration_fold_ids=fold_ids[:index],
        )
        for index, fold_id in enumerate(fold_ids)
        if index > 0
    )
