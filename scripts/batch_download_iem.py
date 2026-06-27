"""CLI adapter for historical IEM observation downloads."""
from __future__ import annotations

import argparse

from config.settings import Settings
from data.storage.parquet_store import ParquetStore
from training.downloaders.iem import BatchIemDownloader


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Batch IEM ASOS downloader")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--station", default="LGA")
    parser.add_argument("--network", default="NY_ASOS")
    parser.add_argument("--store-id", default="KLGA", dest="store_id")
    args = parser.parse_args()

    settings = Settings()
    downloader = BatchIemDownloader(
        ParquetStore(settings.obs_store_dir),
        station_id=args.store_id,
    )
    downloader.run(
        args.start,
        args.end,
        station=args.station,
        network=args.network,
    )


if __name__ == "__main__":
    _cli()
