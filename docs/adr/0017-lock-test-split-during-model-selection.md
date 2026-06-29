# Lock the test split during model selection

**Status**: accepted, implemented

Smoke runs, baseline calibration, and Optuna trials use only the audited train
and validation frames. They cannot load the test frame, calculate test metrics,
publish a candidate model, or package one. After a study finishes, a human must
select one resolved configuration and explicitly approve a single final
evaluation that unlocks the test frame. That result cannot be fed back into the
study or used to reopen hyperparameter search.

The current trainer loads all three splits, calculates `test_mae`, and publishes
a candidate on every invocation. Reusing that path for Optuna would repeatedly
expose the holdout set and make the reported final performance part of model
selection.

The rejected alternative was to evaluate test for every trial but hide the
metric from Optuna. Humans and management UI would still see those results, so
selection could leak through operational decisions even if the sampler never
received the value.

**Consequences**: Training execution and final candidate evaluation become
separate operations. Trial evidence contains validation metrics only. Candidate
artifacts and packages are created only by the approved final-evaluation path,
which records the approval and the selected trial snapshot.

**Implementation note (2026-06-29)**: `TftTrainer.run_validation` loads only
train/validation frames and cannot publish. The legacy `run` entry point is
locked. `run_final` requires an approval bound to the selected trial snapshot
and versioned template hash. `TrainingCatalog` persists that approval and uses
transactional uniqueness/consumption checks to permit only one final execution
per study.
