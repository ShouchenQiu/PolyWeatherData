---
name: grill-with-docs
description: Grill PolyWeatherBot plans against code, data flow, domain language, and existing ADRs; sharpen training/production/download boundaries; update CONTEXT.md and ADRs when durable decisions crystallise. Use when the user asks to grill, stress-test, challenge, or document a plan, architecture change, data pipeline change, model-training decision, or production/training boundary.
---

<what-to-do>

Stress-test the user's plan until the intended design is precise enough to implement safely. Walk down the design tree one decision at a time, resolving dependencies between decisions. For each non-obvious trade-off, provide your recommended answer and why it fits PolyWeatherBot.

Ask questions one at a time when the answer is a real product/architecture choice. If a question can be answered by exploring the codebase, explore the codebase instead.

Do not turn the session into a generic brainstorm. Ground every challenge in the current repo, `CONTEXT.md`, `docs/adr/`, runbooks, or concrete data/model flows.

</what-to-do>

<supporting-info>

## PolyWeatherBot orientation

PolyWeatherBot has three recurring axes that must stay distinct:

- **Training factory**: historical download, data audit, decoder cache, TFT training, candidate artifacts, bundle/package work.
- **Production inference**: live/cached HRRR f01 + IEM obs, `TftModel`, `McSampler`, `ModelPipeline`, `ModelRegistry`, active model loading.
- **Trading runtime**: discovery, CLOB/orderbook, strategy/risk, paper/live execution.

When grilling model or data architecture, explicitly check whether a proposed module belongs to `data/`, `training/`, `model/`, `scripts/`, `backtest/`, or `core/`. Challenge any module that mixes training-only state with production inference.

## Exploration first

Before asking the user, inspect the relevant code and docs:

- `CONTEXT.md` for canonical terms and data-flow language.
- `docs/adr/` for accepted decisions that should not be re-litigated casually.
- `docs/remote-training-runbook.md` and `docs/ubuntu-training-audit.md` for operational intent.
- Relevant implementation files before judging architecture. Prefer `rg` over broad reading.

For training/data/model work, usually inspect:

- `data/forecast/`, `data/obs/`, `data/storage/`, `data/manager.py`
- `model/tft.py`, `model/mc.py`, `model/pipeline.py`, `model/store/registry.py`, `model/features.py`
- `model/training*`, `scripts/run_remote_training.py`, `scripts/batch_prepare_training_data.py`, `scripts/audit_training_data.py`, `scripts/train_tft.py`

### File structure

This repo has a single domain context:

```
/
+-- CONTEXT.md
+-- docs/
|   +-- adr/
+-- data/
+-- model/
+-- scripts/
+-- backtest/
+-- core/
```

Create files lazily - only when you have something to write. If no `CONTEXT.md` exists, create one when the first term is resolved. If no `docs/adr/` exists, create it when the first ADR is needed.

## During the session

### Challenge data flow

For any training/download/model proposal, trace the artifact flow:

1. External source: HRRR S3/Zarr/NOMADS, IEM, Polymarket/CLOB.
2. Cleaning and schema: `Cleaner`, required HRRR/IEM/model feature columns.
3. Storage: `ParquetStore` partitions under `hrrr`, `obs`, `_decoder_cache`, model store, DB.
4. Model sample construction: encoder window, decoder window, residual target, feature ordering.
5. Candidate model artifacts: checkpoint, training dataset pickle, metadata, contract, package.
6. Production loading/promotion: import candidate, validate contract, write active version only through explicit promotion.

Surface any unclear ownership, duplicate source logic, hidden production dependency on training code, or training code that mutates production state.

### Challenge module seams

Use the architecture vocabulary from `improve-codebase-architecture` when relevant:

- **Module**: anything with an interface and implementation.
- **Interface**: what callers must know, including invariants and error modes.
- **Seam**: where behaviour can vary without editing callers.
- **Adapter**: concrete implementation at a seam.

Good PolyWeatherBot seams usually look like:

- `data/*` owns single-source fetch, cleaning, and storage primitives.
- training downloaders own bulk historical scheduling, skip/resume, progress, and repair strategy.
- training dataset builders own encoder/decoder sample construction.
- `model/*` owns production inference and artifact contract validation.
- `scripts/*` should be CLI adapters, not deep implementation modules.

### Challenge against the glossary

When the user uses a term that conflicts with the existing language in `CONTEXT.md`, call it out immediately. "Your glossary defines 'cancellation' as X, but you seem to mean Y - which is it?"

### Sharpen fuzzy language

When the user uses vague or overloaded terms, propose a precise canonical term. "You're saying 'account' - do you mean the Customer or the User? Those are different things."

### Discuss concrete scenarios

When domain relationships are being discussed, stress-test them with specific scenarios. Invent scenarios that probe edge cases and force the user to be precise about the boundaries between concepts.

Prefer PolyWeatherBot-specific scenarios:

- non-MAJOR HRRR init needs 48h decoder window but native Zarr only has 18h.
- HRRR f01 valid_time must align with IEM obs at the same canonical hour.
- training machine produces candidate artifacts but must not promote production active version.
- john_linux is trusted for production; john_server is lower-trust large-training only.
- a download is interrupted mid-month and must resume without corrupting completed partitions.
- candidate package is imported on production but missing `model_contract.json` or `training_dataset.pkl`.

### Cross-reference with code

When the user states how something works, check whether the code agrees. If you find a contradiction, surface it: "Your code cancels entire Orders, but you just said partial cancellation is possible - which is right?"

### Update CONTEXT.md inline

When a term is resolved, update `CONTEXT.md` right there. Don't batch these up - capture them as they happen. Use the format in [CONTEXT-FORMAT.md](./CONTEXT-FORMAT.md).

Don't couple `CONTEXT.md` to implementation details. Only include terms that are meaningful to domain experts.

### Offer ADRs sparingly

Only offer to create an ADR when all three are true:

1. **Hard to reverse** - the cost of changing your mind later is meaningful
2. **Surprising without context** - a future reader will wonder "why did they do it this way?"
3. **The result of a real trade-off** - there were genuine alternatives and you picked one for specific reasons

If any of the three is missing, skip the ADR. Use the format in [ADR-FORMAT.md](./ADR-FORMAT.md).

For this project, ADRs are especially useful for:

- training vs production module boundaries;
- data source fallback policy;
- decoder-cache fill/repair strategy;
- model artifact contract and promotion safety;
- trust boundaries between John_Surface, john_linux, and john_server.

</supporting-info>
