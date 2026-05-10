# CONTEXT.md Format (PolyWeatherBot)

## Structure

```md
# PolyWeatherBot Context

{One sentence: what this module/subsystem does and why it exists.}

## Language

**{Term}**:
{One sentence: what it IS, not what it does.}
_Avoid_: {synonym1}, {synonym2}

## Data & Signal Flow

- **{SourceA}** produces **{ArtifactA}** consumed by **{TargetB}**
- **{ComponentX}** owns **{StateY}**; other components reference it by ID only

## Flagged ambiguities

- "{fuzzy word}" was used to mean both **TermA** and **TermB** - resolved: {resolution}.
```

## Domain term groups for this project

Terms belong to one of four groups. Only include a term if it is unique to this project's domain - not general Python/ML/async patterns.

| Group | Examples |
|-------|---------|
| **Weather** | init_time, fxx, f01, HRRR cycle, MAJOR hour, encoder window, decoder window, cold-start |
| **Market** | Event, Market, Token, Bucket, slug, YES/NO outcome, CLOB, mid-price, edge, signal |
| **Model** | TFT, McSampler, residual, temp_error, p10/p50/p90, empirical covariance, staleness |
| **Ops** | ForecastLoop tick, DiscoveryLoop, PaperExchange, PositionTracker, ParquetStore partition |

## Rules

- One sentence per term. Define what it IS.
- Flag ambiguous words immediately with a resolution.
- Express data-flow cardinality: "one init_time -> 48 fxx rows".
- Skip terms that are obvious Python/ML conventions (DataFrame, coroutine, ABC, etc.).
- Update inline when a term is resolved; never batch.

## File location

Single `CONTEXT.md` at repo root suffices for this project. No `CONTEXT-MAP.md` needed unless the project splits into independently deployable services.
