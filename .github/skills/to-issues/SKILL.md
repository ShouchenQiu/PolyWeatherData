---
name: to-issues
description: Break a plan, spec, or PolyWeatherBot overhaul task document into independently-grabbable issue drafts using tracer-bullet vertical slices and a coverage matrix. Use when the user wants to convert docs/grill-tasks.md, docs/grill-tasks-v2.md, or another plan into implementation tickets without losing source tasks.
---

# To Issues

Break a plan into independently-grabbable issue drafts using vertical slices
(tracer bullets), while preserving full coverage of the source document.

This project does not require `/setup-matt-pocock-skills`. If no issue tracker
or label vocabulary is configured, do not publish issues and do not depend on
standard labels. Generate Markdown issue drafts instead.

## Process

### 1. Gather context

Work from whatever is already in the conversation context. If the user passes a
path such as `docs/grill-tasks.md` or `docs/grill-tasks-v2.md`, read that exact
file as the source of truth. Respect linked ADRs and project docs when they
clarify intent.

### 2. Explore the codebase (optional)

If you have not already explored the codebase, do so to understand the current
state of the code. Issue titles and descriptions should use this project's
domain vocabulary: ForecastLoop, TradingLoop, ProbabilityStore, Cleaner,
PositionTracker, RiskPipeline, AlertBus, SettlementLoop, CLOB, HRRR, IEM, and
ADR identifiers where relevant.

### 3. Build a source inventory

Before drafting slices, inventory every actionable source item from the plan.
For `docs/grill-tasks*.md`, each `ADD`, `REMOVE`, `CHANGE`, `FIX`, `MOVE`, and
`WIRE` bullet is a source item. Assign each source item a stable reference:

- Prefer `<source-path>:<line>` when line numbers are available.
- Otherwise use `<heading> / <short item text>`.

Every source item must end in exactly one of these states:

- `Covered`: implemented by one or more issue drafts.
- `Deferred`: explicitly out of scope for this overhaul.
- `Blocked/HITL`: requires human decision, external data, credentials, or
  another non-AFK prerequisite.

No source item may be left unclassified.

### 4. Draft vertical slices

Break the plan into **tracer bullet** issues. Each issue is a thin vertical slice that cuts through ALL integration layers end-to-end, NOT a horizontal slice of one layer.

Slices may be 'HITL' or 'AFK'. HITL slices require human interaction, such as an architectural decision or a design review. AFK slices can be implemented and merged without human interaction. Prefer AFK over HITL where possible.

<vertical-slice-rules>
- Each slice delivers a narrow but COMPLETE path through every layer (schema, API, UI, tests)
- A completed slice is demoable or verifiable on its own
- Prefer many thin slices over few thick ones
- Do not mechanically mirror the layer order in the source plan
- Foundation slices are allowed when a shared primitive is needed before useful vertical slices can start
</vertical-slice-rules>

### 5. Produce a coverage matrix

After drafting slices, produce a coverage matrix:

```
| Source item | Covered by slice | Status |
|---|---|---|
| <source-path>:40 max_open_per_token | Risk + Position Lifecycle | Covered |
| <source-path>:423 DataHealthMonitor deferred | Deferred register | Deferred |
```

Run a reverse audit before presenting the result:

- Source -> slices: every source item is `Covered`, `Deferred`, or `Blocked/HITL`.
- Slices -> source: every slice has source refs and does not add unrequested scope.
- If nothing is missing, explicitly write `Uncovered items: none`.

### 6. Quiz the user

Present the proposed breakdown as a numbered list. For each slice, show:

- **Title**: short descriptive name
- **Type**: HITL / AFK
- **Blocked by**: which other slices (if any) must complete first
- **Source refs**: source document lines/headings and ADR ids covered by the slice
- **Out of scope**: deferred items that must not be implemented in the slice

Ask the user:

- Does the granularity feel right? (too coarse / too fine)
- Are the dependency relationships correct?
- Should any slices be merged or split further?
- Are the correct slices marked as HITL and AFK?
- Are any source refs missing from the coverage matrix?

Iterate until the user approves the breakdown.

### 7. Publish or output issue drafts

By default, output Markdown issue drafts only. Publish to an issue tracker only
if the user explicitly asks and provides the tracker/label requirements.

Publish issues in dependency order (blockers first) so you can reference real issue identifiers in the "Blocked by" field.

<issue-template>
## Title

Short descriptive title.

## Type

AFK or HITL.

## Parent

A reference to the parent issue on the issue tracker (if the source was an existing issue, otherwise omit this section).

## What to build

A concise description of this vertical slice. Describe the end-to-end behavior, not layer-by-layer implementation.

Avoid specific file paths or code snippets — they go stale fast. Exception: if a prototype produced a snippet that encodes a decision more precisely than prose can (state machine, reducer, schema, type shape), inline it here and note briefly that it came from a prototype. Trim to the decision-rich parts — not a working demo, just the important bits.

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Tests

- Focused unit/integration tests required for this slice.

## Source refs

- Source document references and ADR ids covered by this slice.

## Blocked by

- A reference to the blocking ticket (if any)

Or "None - can start immediately" if no blockers.

## Out of scope

- Deferred or adjacent work intentionally excluded from this slice.

</issue-template>

Do NOT close or modify any parent issue.
