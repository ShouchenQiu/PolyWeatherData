"""Attribute daily-high errors across TFT, IEM, MC, and settlement truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.forecast_cache import ForecastCache
from config.stations import STATIONS
from training.daily_high_attribution import (
    build_daily_high_layer_attribution,
    summarize_daily_high_layer_attribution,
)
from training.historical_crossfit import load_historical_crossfit_predictions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossfit-root", required=True, type=Path)
    parser.add_argument("--forecast-cache-root", required=True, type=Path)
    parser.add_argument("--actual-highs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA", choices=sorted(STATIONS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"immutable layer attribution output exists: {args.output}")
    station = STATIONS[args.station]
    print(f"[start] daily-high layer attribution output={args.output}", flush=True)
    predictions = load_historical_crossfit_predictions(args.crossfit_root)
    cache = ForecastCache.load(args.forecast_cache_root)
    actuals = _load_actuals(args.actual_highs)
    rows = build_daily_high_layer_attribution(
        predictions=predictions,
        snapshots=cache.snapshots,
        actuals=actuals,
        station_timezone=station.timezone,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    rows.to_parquet(args.output / "snapshot_layers.parquet", index=False)
    summary = summarize_daily_high_layer_attribution(rows)
    manifest = {
        "schema_version": 1,
        "data_class": "daily_high_layer_attribution",
        "station_id": station.station_id,
        "source_sha256": {
            "crossfit_manifest": _sha256(args.crossfit_root / "manifest.json"),
            "forecast_manifest": _sha256(args.forecast_cache_root / "manifest.json"),
            "actuals_manifest": _sha256(
                args.actual_highs.with_suffix(args.actual_highs.suffix + ".manifest.json")
            ),
        },
        "row_count": len(rows),
        "sha256": {
            "snapshot_layers.parquet": _sha256(args.output / "snapshot_layers.parquet"),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest["sha256"]["summary.json"] = _sha256(args.output / "summary.json")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[complete] daily-high layer attribution snapshots={len(rows)} "
        f"output={args.output}",
        flush=True,
    )
    print(f"[done] daily-high layer attribution output={args.output}", flush=True)
    return 0


def _load_actuals(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
