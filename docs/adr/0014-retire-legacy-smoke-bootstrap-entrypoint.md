# Retire legacy smoke and weather-audit wrappers

**Status**: accepted, implemented

The `scripts/smoke_bootstrap.py` and `scripts/audit_weather_backfill.py`
compatibility wrappers were removed after historical download completion.
Weather preparation now uses `scripts.batch_prepare_training_data`, and both
coverage and readiness audits use `scripts.audit_training_data`.

**Consequences**: preparation evidence is named
`prepare_data_summary.json`; no compatibility Python imports are retained.
