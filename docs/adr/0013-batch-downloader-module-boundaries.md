# Batch downloader classes live outside orchestration modules

**Status**: accepted, implementation deferred

The current training download flow works, but one class sits at the wrong
module boundary: `BatchHrrrS3F01Downloader` is defined inside
`scripts/batch_prepare_training_data.py` even though it is a concrete downloader.
Meanwhile `BatchZarrDownloader` and `BatchIemDownloader` live in dedicated
download modules. This makes the code harder to navigate and obscures that the
training-data script is an orchestrator rather than a downloader implementation.

After the current large historical download is complete, move the concrete HRRR
f01 downloader out of `scripts/batch_prepare_training_data.py`. The target shape
is:

1. `scripts/batch_download_hrrr_f01.py` owns `BatchHrrrS3F01Downloader`.
2. `scripts/batch_download_zarr.py` owns `BatchZarrDownloader`.
3. `scripts/batch_download_iem.py` owns `BatchIemDownloader`.
4. `scripts/batch_prepare_training_data.py` imports concrete downloaders and
   only orchestrates HRRR f01, IEM obs, and decoder-cache preparation.

If duplication between HRRR batch downloaders becomes meaningful, extract a
small base class for shared init-range construction, already-downloaded checks,
and progress plumbing. Do not force that abstraction until it removes real
duplication; a simple move of `BatchHrrrS3F01Downloader` is enough for the first
cleanup slice.

**Consequences**: imports and tests that patch or reference
`BatchHrrrS3F01Downloader` must be updated. Remote bundle/package exclusion
lists and runbook references should be checked for the new module. No behavior
change is intended; this is a code-organization cleanup to make downloader
ownership match module names.
