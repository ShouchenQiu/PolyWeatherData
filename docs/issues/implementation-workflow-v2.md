# Implementation Workflow V2

Source of truth:
- `docs/issues/proposed-slices-v2.md`
- `docs/grill-tasks-v2.md`
- `docs/2ndMajorOverhaul_05_30.md`
- `docs/adr/`

This file describes how to execute the second-overhaul slices. It is an operating
guide, not a replacement for the slice definitions.

Use the first-overhaul workflow only for `docs/issues/proposed-slices.md`.

## Execution Workflow

### 1. Pick one V2 slice

Work on exactly one slice from `docs/issues/proposed-slices-v2.md` at a time.
Use the execution order below and the Dependency Graph in that file to choose the
next unblocked slice.

Before coding, read:
- The target slice in `docs/issues/proposed-slices-v2.md`
- Its `Source refs`
- The corresponding section in `docs/grill-tasks-v2.md`
- Any referenced ADRs, especially ADR-0009, ADR-0010, and ADR-0011
- `CONTEXT.md` when the slice touches domain behavior
- Existing code and tests in the touched area

Do not implement work listed in the slice's `Out of scope`.
Do not pull future-slice behavior forward unless the current slice cannot compile
without a minimal stub.

### 2. Use TDD for implementation

Follow `.github/skills/tdd/SKILL.md` for each slice. Keep each TDD loop scoped to
the current V2 slice.

When a slice is complete:
- Run the slice-specific tests.
- Run adjacent tests if public interfaces changed.
- Run a lightweight import/startup smoke test when wiring changed.
- Record any residual risks or intentionally deferred work.

### 3. Switch to diagnose when stuck

Use `.github/skills/diagnose/SKILL.md` when:
- A test fails for a reason that is not immediately clear.
- A failure may be caused by fixtures, environment, old V1 architecture, or mocks.
- The same failure survives two reasonable implementation attempts.
- A live network, HRRR/IEM source, CLOB SDK, or credential issue appears.

Follow the phase loop in `.github/skills/diagnose/SKILL.md`.

For AFK slices, do not require real CLOB credentials, real order submission, live
HRRR/IEM network access, or on-chain redeem. Use mocks, fakes, fixtures, or local
sample data unless the slice explicitly says otherwise.

### 4. Review architecture at checkpoints

Use `.github/skills/improve-codebase-architecture/SKILL.md` at the checkpoints in
the execution order below.

The architecture review should inspect only:
- The slices just completed
- Their immediate dependencies and consumers
- Public interfaces that changed
- Tests that encode the new contracts

The review should check:
- V1 compatibility shims did not leak into V2 semantics
- Module boundaries match the V2 layer decisions
- `data/` does not depend on `model/`
- `core/pricing/quote.py` remains a leaf module
- `core/exchange/*` does not own pricing decisions
- TradingLoop owns SELL YES vs BUY NO arbitration
- ForecastLoop and BacktestEngine both use the shared readiness gate
- Deferred work did not leak into current implementation

Do not create issues or modify code during architecture review unless explicitly
asked.

### 5. Execution order

Use `docs/issues/proposed-slices-v2.md` as the authoritative dependency graph.
The order below is optimized to keep contracts stable before higher-level loops
are changed.

#### Batch 1 -- Contract Foundations

Implement these first:
- V2F01 Schema Expansion
- V2F03 L2 Quote Primitive
- V2F04 ProbabilitySnapshot + Feature Contract

These three are mostly independent and may be implemented in any order. Prefer
starting with V2F01 if multiple people/agents are coordinating, because many later
slices depend on the schema contract.

After Batch 1:
- Run all tests touched by schema, ProbabilityStore, quote, feature contracts, and imports.
- Run architecture review for V2F01/V2F03/V2F04.

#### Batch 2 -- Client + Accounting Foundations

Then implement:
- V2F02 Polymarket SDK Client Split
- V2F05 Position Accounting + PnL/Risk Primitives

V2F02 depends on V2F01. V2F05 depends on V2F01 and V2F03. Keep exchange-fill
behavior out of V2F02; it belongs to V2V09. Keep TradingLoop arbitration out of
V2F05; it belongs to V2V08.

After Batch 2:
- Run settings, bot-wiring, orderbook, exchange-adapter, positions, PnL, and risk tests.
- Run architecture review for V2F02/V2F05.

#### Batch 3 -- Data + Discovery Readiness

Then implement:
- V2V01 Discovery + Lifecycle Hardening
- V2V02 HRRR Source Hardening + Extrapolation Flags
- V2V03 Obs + Parquet Hardening

These can proceed in parallel after their blockers are done. V2V03 should finish
before V2V04. V2V02 should finish before V2V05 because ModelPipeline depends on
the `is_extrapolated` contract.

After Batch 3:
- Run discovery/lifecycle tests.
- Run HRRR source tests.
- Run obs/parquet/DataManager completeness tests.
- Run architecture review for V2V01/V2V02/V2V03.

#### Batch 4 -- Inference Contract

Then implement:
- V2V04 Encoder Readiness Gate
- V2V05 ModelPipeline Validation + Daily-High Target
- V2V06 MC Covariance + Production Model Wiring

Order inside this batch:
1. V2V04 first, because ForecastLoop and BacktestEngine both need the readiness
   gate.
2. V2V05 second, because it consumes the readiness and feature contracts.
3. V2V06 third, because Bot wiring should point at the final ModelPipeline/TFT/MC
   contracts.

After Batch 4:
- Run DataManager, ForecastLoop readiness, Backtest readiness, TFT, ModelPipeline,
  MC, registry, and Bot construction tests.
- Run architecture review for V2V04/V2V05/V2V06.

#### Batch 5 -- Forecast + Trading Path

Then implement:
- V2V07 ForecastLoop V2 Write Path
- V2V08 Directional Trading Arbitration
- V2V09 Exchange Fill Semantics

Order inside this batch:
1. V2V07 first, because TradingLoop consumes the V2 ProbabilitySnapshot and event
   target-date semantics.
2. V2V08 second, because it depends on V2F03/V2F05 and the V2 probability path.
3. V2V09 third, because exchange adapters should receive already-decided intents
   and validated prices from TradingLoop.

Do not move re-quote or edge evaluation into `core/exchange/*`. TradingLoop owns
those decisions.

After Batch 5:
- Run ForecastLoop tests, TradingLoop tests, strategy/risk tests, exchange tests,
  and any Bot integration tests touched by loop wiring.
- Run architecture review for V2V07/V2V08/V2V09.

#### Batch 6 -- Adjacent Runtime Paths

Then implement:
- V2V10 Backtest V2 Semantics
- V2V11 SniperLoop Alignment
- V2V12 Redeem Loop

Suggested order:
1. V2V10 after V2V04 and V2F03 are complete.
2. V2V11 after V2V01, V2F02, V2F03, and V2F05 are complete.
3. V2V12 after V2F01 and V2F02 are complete; it can be done earlier if needed,
   but it is less useful until settlement/order semantics are stable.

After Batch 6:
- Run backtest tests, sniper tests, settlement/redeem tests, and full relevant
  regression tests.
- Run final architecture review across the full V2 slice set.

## Suggested Invocation Texts

### Implement a V2 slice with TDD

```text
Please use `.github/skills/tdd/SKILL.md` to implement [SLICE_ID] from
`docs/issues/proposed-slices-v2.md`.

First read the slice, its Source refs in `docs/grill-tasks-v2.md`, and any
referenced ADRs. Treat the slice as the approved scope.

Use TDD:
- Write one failing behavior test.
- Run the smallest relevant pytest node.
- Implement the minimum code to pass.
- Repeat until the slice acceptance criteria are satisfied.

Do not implement out-of-scope or future-slice behavior. When complete, run the
slice-specific tests, adjacent tests for changed public interfaces, and summarize
what passed.
```

### Diagnose a failing V2 slice

```text
Please use `.github/skills/diagnose/SKILL.md` to diagnose the current failure in
[SLICE_ID].

Start by fixing the smallest reproduction command. State the failure, identify
the most likely hypotheses, inspect only what is needed, and make the smallest
fix.

Do not expand the slice scope. After the fix, keep or add a regression test and
rerun the minimal failing test.
```

### Architecture review after a V2 checkpoint

```text
Please use `.github/skills/improve-codebase-architecture/SKILL.md` to review the
V2 slices just completed: [SLICE_IDS].

Read `docs/issues/proposed-slices-v2.md`, the relevant Source refs in
`docs/grill-tasks-v2.md`, ADRs, code, and tests touched by those slices.

Review module boundaries, public interfaces, test boundaries, accidental coupling,
and whether future-slice work leaked into implementation.

Do not modify code or create issues. Output architecture review notes and any
recommended follow-up adjustments.
```

### Continue from the V2 dependency graph

```text
Using `docs/issues/proposed-slices-v2.md` and
`docs/issues/implementation-workflow-v2.md`, identify the next unblocked V2 slice
after [COMPLETED_SLICE_IDS]. Then use `.github/skills/tdd/SKILL.md` to implement
only that slice.
```
