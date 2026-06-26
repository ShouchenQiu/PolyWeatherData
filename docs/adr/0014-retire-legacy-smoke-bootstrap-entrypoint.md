# Retire legacy smoke_bootstrap entrypoint after download work

**Status**: accepted, implementation deferred

`scripts/smoke_bootstrap.py` is now a compatibility wrapper around
`scripts.batch_prepare_training_data`. It re-exports
`BatchHrrrS3F01Downloader`, `run_batch_prepare_training_data`, and
`run_smoke_bootstrap`, and forwards its CLI to the newer batch-preparation CLI.
The real implementation lives in `scripts/batch_prepare_training_data.py`.

The wrapper was useful while historical smoke-bootstrap commands were being
renamed, but it now makes the script surface harder to understand. New commands
should use:

```text
python -m scripts.batch_prepare_training_data
```

or the higher-level remote training entrypoint:

```text
python -m scripts.run_remote_training download
```

After the current large historical download is complete, remove the legacy
`scripts/smoke_bootstrap.py` module and the `run_smoke_bootstrap` alias unless an
external caller is still found. Before removal, update tests and documentation
that intentionally preserve the legacy re-export behavior.

**Consequences**: remove or rewrite the compatibility test that asserts
`scripts.smoke_bootstrap` re-exports `batch_prepare_training_data`. Search docs,
runbooks, local shell history, and remote bundle/package rules for
`scripts.smoke_bootstrap`. If any operational workflow still depends on the old
entrypoint, keep a deprecation note for one more release instead of deleting it
immediately.
