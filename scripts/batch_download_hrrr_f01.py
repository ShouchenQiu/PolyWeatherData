"""CLI adapter for historical HRRR f01 downloads."""
from __future__ import annotations

import argparse

from config.settings import Settings
from data.storage.parquet_store import ParquetStore
from training.downloaders.hrrr import BatchHrrrF01Downloader


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Batch HRRR f01 downloader")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    settings = Settings()
    downloader = BatchHrrrF01Downloader(
        ParquetStore(settings.forecast_store_dir),
        station_id=args.station,
        max_workers=args.workers,
    )
    downloader.run(args.start, args.end)


if __name__ == "__main__":
    _cli()
