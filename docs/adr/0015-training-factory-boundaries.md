# Training factory is separate from production model runtime

**Status**: accepted, implemented

Offline historical preparation, audit, dataset construction, TFT training,
candidate packaging, and lab bundle creation live in the top-level
`training/` package. The `model/` package owns production inference, artifact
contract validation, candidate import, and explicit promotion; production
modules never import `training`.

The rejected alternative was a remote-training module under `scripts/` or
`model/`. Remote is a deployment location rather than a domain concept, and a
single download/audit/train/package controller hid stage ownership and coupled
production model code to training dependencies.

**Consequences**: `scripts/` contains CLI adapters only. Training and production
share `model.features`, `model.contract`, and candidate artifact semantics, but
only `model.store.operations.promote_model_version` may activate a model.
Lower-trust training machines may create candidate packages but must not receive
production state or promotion authority.
