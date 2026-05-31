# ADR Format (PolyWeatherBot)

ADRs live in `docs/adr/` using sequential numbering: `0001-slug.md`, `0002-slug.md`, etc.
Create the directory lazily -- only when the first ADR is needed.

## Template

```md
# {Short title}

{1-3 sentences: context, decision, reason.}
```

One paragraph is enough. Record *that* a decision was made and *why*.

## Optional sections

Add only when they genuinely matter:

- **Status**: `proposed | accepted | deprecated | superseded by ADR-NNNN`
- **Considered Options**: rejected alternatives worth remembering
- **Consequences**: non-obvious downstream effects on data pipeline or live trading

## When to write one

All three must be true:

1. **Hard to reverse** -- changing this later costs significant rework
2. **Surprising without context** -- a future reader will wonder "why not the obvious approach?"
3. **Real trade-off existed** -- there were genuine alternatives

## What qualifies in this project

| Category | Examples |
|----------|---------|
| **Model architecture** | Encoder length = 72h; empirical covariance vs AR(1); residual as target |
| **Data pipeline shape** | Store only f01 in Parquet; Cleaner owns all feature engineering; cold-start = 72h drop |
| **Layer boundaries** | `data/` must not import from `model/`; `config/` imports nothing |
| **Source fallback policy** | Priority order of HRRR sources; when to switch obs source |
| **Trading constraints** | Paper mode requires real CLOB book; L2 VWAP quote semantics; SELL YES vs BUY NO arbitration |
| **External constraints** | HRRR MAJOR hours only have 48h range; IEM obs arrives ~51min past the hour |

Skip if the decision is easy to reverse, obvious, or had no real alternative.
