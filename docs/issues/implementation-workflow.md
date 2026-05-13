# Implementation Workflow

Source of truth:
- `docs/issues/proposed-slices.md`
- `docs/grill-tasks.md`
- `docs/adr/`

This file describes how to execute the overhaul slices. It is an operating guide,
not a replacement for the slice definitions.

## Execution Workflow

### 1. Pick one slice

Work on exactly one slice from `docs/issues/proposed-slices.md` at a time.
Use the Dependency Graph in that file to choose the next unblocked slice.

Before coding, read:
- The target slice in `docs/issues/proposed-slices.md`
- Its `Source refs`
- Any referenced ADRs
- `CONTEXT.md` when the slice touches product/domain behavior
- Existing code and tests in the touched area

Do not implement work listed in the slice's `Out of scope`.
Do not implement Deferred Register items.

### 2. Use TDD for implementation

Follow `.github/skills/tdd/SKILL.md` for each slice. Do not write all tests
first; do not implement future slice behavior early.

When a slice is complete:
- Run the slice-specific tests.
- Run adjacent tests if public interfaces changed.
- Record any residual risks or intentionally deferred work.

### 3. Switch to diagnose when stuck

Use `.github/skills/diagnose/SKILL.md` when:
- A test fails for a reason that is not immediately clear.
- A failure may be caused by fixtures, environment, old architecture, or mocks.
- The same failure survives two reasonable implementation attempts.
- A live network, HRRR/IEM source, or CLOB credential issue appears.

Follow the phase loop in `.github/skills/diagnose/SKILL.md`.

For AFK slices, do not require real CLOB credentials, real order submission, or live
network access. Use mocks, fakes, or fixtures unless the slice explicitly says
otherwise.

### 4. Review architecture every 2-3 slices

Use `.github/skills/improve-codebase-architecture/SKILL.md` after every 2-3
completed slices.

The architecture review should inspect only:
- The slices just completed
- Their immediate dependencies and consumers
- Any interfaces that changed

The review should check:
- Module boundaries remain clear
- Public interfaces are small and stable
- Tests cover behavior without over-mocking internals
- Deferred work did not leak into current implementation
- Upcoming slices are not blocked by accidental coupling

Do not create issues or modify code unless explicitly asked.

### 5. Execution order

Use `docs/issues/proposed-slices.md` as the authoritative dependency graph.

Recommended first batch:
- F01 Settings Cleanup + DB Orders Table
- F02 AlertBus
- F04 ProbabilityStore

Then run architecture review.

Recommended second batch:
- F03 PositionTracker Order Lifecycle
- F05 TradingLoop Skeleton
- V01 Sniper Manager Rename or V08 Slug Parser Registry

Then run architecture review.

Recommended third batch:
- V02 HRRR Multi-Station Pipeline + HrrrFill + Parquet Enforcement
- V04 IEM + NOAA Obs Hardening
- V03 Cleaner + DataManager + build_encoder_obs_window

Then run architecture review.

After that, continue through:
- V05a/V05b/V05c
- V06/V07
- V10/V11/V13
- V12

H01 is the only HITL slice. It can proceed any time once station row/col values
are available.

## Suggested Invocation Texts

### Implement a slice with TDD

```text
Please use `.github/skills/tdd/SKILL.md` to implement [SLICE_ID] from
`docs/issues/proposed-slices.md`.

First read the slice, its Source refs in `docs/grill-tasks.md`, and any referenced
ADRs. Treat the slice as the approved scope.

Use TDD:
- Write one failing behavior test.
- Run the smallest relevant pytest node.
- Implement the minimum code to pass.
- Repeat until the slice acceptance criteria are satisfied.

Do not implement out-of-scope or Deferred Register items. When complete, run the
slice-specific tests and summarize what passed.
```

### Diagnose a failing slice

```text
Please use `.github/skills/diagnose/SKILL.md` to diagnose the current failure.

Start by fixing the smallest reproduction command. State the failure, identify the
most likely hypotheses, inspect only what is needed, and make the smallest fix.

Do not expand the slice scope. After the fix, keep or add a regression test and
rerun the minimal failing test.
```

### Architecture review after 2-3 slices

```text
Please use `.github/skills/improve-codebase-architecture/SKILL.md` to review the
slices just completed: [SLICE_IDS].

Read `docs/issues/proposed-slices.md`, the relevant Source refs, ADRs, code, and
tests touched by those slices.

Review module boundaries, public interfaces, test boundaries, accidental coupling,
and whether Deferred Register work leaked into implementation.

Do not modify code or create issues. Output architecture review notes and any
recommended follow-up adjustments.
```

### Continue from dependency graph

```text
Using `docs/issues/proposed-slices.md`, identify the next unblocked slice after
[COMPLETED_SLICE_IDS]. Then use `.github/skills/tdd/SKILL.md` to implement only
that slice.
```
