# Training downloaders own historical batch scheduling

**Status**: accepted, implemented

Historical skip/resume, bounded concurrency, progress, and repair scheduling
live in `training/downloaders/`. Source-specific network access remains in
`data/forecast/` and `data/obs/`, while `training/prepare_data.py` only
orchestrates HRRR f01, IEM obs, and decoder-cache preparation.

**Consequences**: `BatchHrrrF01Downloader` and `BatchIemDownloader` call the
existing data-source interfaces and `ParquetStore`; training code does not
reimplement Zarr, GRIB, or IEM protocols. CLI files under `scripts/` are thin
adapters.
