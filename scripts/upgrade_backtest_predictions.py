"""Attach decoder trust provenance to existing hourly prediction evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--decoder-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--station", default="KLGA")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    predictions = pd.read_parquet(args.predictions)
    decoder_paths = sorted(
        (args.decoder_cache / args.station).glob("*.parquet")
    )
    if not decoder_paths:
        raise ValueError("decoder cache contains no parquet payloads")
    decoder = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "init_time",
                    "fxx",
                    "is_extrapolated",
                    "extrapolation_step",
                ],
            )
            for path in decoder_paths
        ],
        ignore_index=True,
    )
    for frame in (predictions, decoder):
        frame["init_time"] = pd.to_datetime(frame["init_time"], utc=True)
        frame["fxx"] = frame["fxx"].astype(int)
    provenance = decoder.drop_duplicates(["init_time", "fxx"], keep="last")
    upgraded = predictions.drop(
        columns=["is_extrapolated", "extrapolation_step"],
        errors="ignore",
    ).merge(
        provenance,
        on=["init_time", "fxx"],
        how="left",
        validate="one_to_one",
    )
    if upgraded[["is_extrapolated", "extrapolation_step"]].isna().any().any():
        raise ValueError("decoder provenance does not cover every prediction")
    upgraded["is_extrapolated"] = upgraded["is_extrapolated"].astype(bool)
    upgraded["extrapolation_step"] = upgraded["extrapolation_step"].astype(int)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    upgraded.to_parquet(args.output, index=False)
    manifest = {
        "schema_version": 2,
        "data_class": "hourly_quantile_predictions",
        "station_id": args.station,
        "row_count": len(upgraded),
        "init_count": int(upgraded["init_time"].nunique()),
        "source_sha256": {
            "predictions": _sha256(args.predictions),
            **{
                f"decoder_{index}": _sha256(path)
                for index, path in enumerate(decoder_paths)
            },
        },
        "sha256": {args.output.name: _sha256(args.output)},
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"Saved {len(upgraded)} provenance-complete predictions "
        f"to {args.output}",
        flush=True,
    )
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
