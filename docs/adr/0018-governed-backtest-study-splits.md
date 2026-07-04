# Govern edge research with immutable studies and a one-use holdout

**Status**: accepted

Trading edge research uses a dedicated `BacktestStudy` lifecycle. A development study
binds exact forecast, synthetic-quote, raw CLOB, and resolution-truth hashes and accepts
only `oof_development` forecasts. `model_test` remains reserved for weather-model
evaluation and is rejected by trading studies. After the strategy and edge calibration
are frozen with an approver, an independent SQLite catalog permits one atomic
`post_test_holdout` consumption.

The initial diagnostic replay treats one Event as one risk unit: it takes only the first
qualifying Token, uses a fixed $50 order, and holds through settlement. Calibration uses
event-equal residual evidence and conservative executable prices. Adaptive position
sizing is deferred to a separate money-management study after signal eligibility.

**Considered alternatives**: CLI flags alone cannot prevent accidental split reuse;
immutable output directories can be deleted or renamed; reusing the training catalog
would couple trading research to `training/`. A dedicated catalog keeps the boundary
auditable without creating a production runtime dependency.

**Consequences**: managed replay and selection entry points reject unmanifested truth,
stale or cross-version inputs, and unmanaged replay artifacts. Holdout evaluation reuses
the frozen development calibration and cannot refit on holdout outcomes.
