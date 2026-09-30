# Data

Raw data come from the **Polymarket public API** (`gamma-api.polymarket.com` for
event/market metadata, `data-api.polymarket.com` for fills). Downloads are
**immutable Parquet artifacts** — once written they are never modified. Nothing
in `data_store/` is committed to this repository (see `.gitignore`).

```
<CLOB_ROOT>/<STATION>/<ARTIFACT>/
├── events.parquet      one row per city-date event
├── markets.parquet     one row per temperature bucket (condition)
├── trades.parquet      one row per taker fill
├── coverage.parquet    per-day completeness audit
└── manifest.json       row counts + SHA256 per file
```

`CLOB_ROOT` defaults to `<repo>/data_store/clob` and can be overridden with the
`POLYWEATHER_CLOB_ROOT` environment variable. `<ARTIFACT>` is the canonical
window name `20251205_20260810` for every station used in the paper.

---

## 1. Market structure

A **daily maximum-temperature market** is one *event* per city-date. The event
holds 7–11 mutually exclusive **buckets** (temperature intervals), exactly one of
which resolves YES. Buckets are ordinal: a lower bucket means a colder daily
high.

| City | Station | City | Station |
|---|---|---|---|
| Atlanta | KATL | New York | KLGA |
| Chicago | KORD | Seattle | KSEA |
| London | — | Seoul | — |
| Miami | KMIA | Shanghai | — |
| Toronto | — | Tokyo | — |

`DALLAS` and `DENVER` are excluded as malformed / expired series. They are listed
in `pipeline/constants.py` as `OUTLIERS` so they are never picked up implicitly.

---

## 2. `events.parquet` — event list

| Column | Type | Meaning |
|---|---|---|
| `event_id` | str | Event identifier (the join key for `markets`) |
| `event_slug` | str | e.g. `highest-temperature-in-london-on-jan-22` |
| `event_title` | str | Human-readable title |
| `target_date` | str | The weather date being predicted, `YYYY-MM-DD` |
| `event_end_at` | datetime | Resolution time |
| `station_id` | str | Station code |

`target_date` — not `event_end_at` — is what the pipeline uses to place an event
on the calendar. This is the key used by
`pipeline/1_panel/16_build_time_split.py` to assign train/val/test.

---

## 3. `markets.parquet` — bucket (condition) list

| Column | Type | Meaning |
|---|---|---|
| `event_id` | str | Parent event |
| `market_id` / `condition_id` | str | Market / condition ID — **`trades` joins on `condition_id`** |
| `market_slug` | str | Contains the bucket bounds, e.g. `...-between-36-37f-on-january-22` |
| `yes_token_id` / `no_token_id` | str | ERC-1155 token IDs for the two outcomes |
| `bucket_lo_f` / `bucket_hi_f` | float | Bucket bounds in °F; half-open `[lo, hi+1)` |
| `yes_outcome_price` / `no_outcome_price` | float | Official resolved outcome prices (1.0 / 0.0) |
| `resolved_yes` | bool | Whether the market resolved and YES won |
| `fees_*` | — | Fee configuration (rate, type, rebate) |
| `min_tick_size` / `min_order_size` | float | Tick size and minimum order size |
| `accepting_orders` / `active` / `closed` | bool | Market state |
| `neg_risk` | bool | Negative-risk flag |
| `market_start_at` / `market_end_at` / `closed_at` | datetime | Timestamps |

### Bucket parsing rules

Bounds are recovered from the slug by `data/clob/slug.py::parse_slug`:

- `35f-or-lower` → `(-inf, 36)`
- `between-36-37f` → `(36, 38)`
- `85f-or-higher` → `(84, +inf)`

International series also use Celsius variants (`{X}c`, `{X}c-or-higher`, and a
hyphenated form); all of these are converted to the `*_f` columns, which are the
only bounds the pipeline reads.

> `bucket_lo_f` / `bucket_hi_f` can legitimately be `±inf` for the open-ended
> end buckets. Code that sorts or indexes buckets must tolerate infinities.

---

## 4. `trades.parquet` — fills (the core data)

| Column | Type | Meaning |
|---|---|---|
| `condition_id` | str | Joins to `markets.parquet` |
| `token_id` | str | YES or NO token |
| `timestamp` | datetime (UTC) | Fill time |
| `outcome` | str | `YES` / `NO` |
| `side` | str | `BUY` / `SELL` |
| `price` | float | Fill price in `[0, 1]` — **taker fill, not mid, not a candle** |
| `size` | float | Shares |
| `usdc` | float | Notional (`price × size`) |
| `transaction_hash` | str | On-chain transaction hash |
| `proxy_wallet` | str | Trader wallet address |
| `source` | str | Always `data_api_trade` |

**Invariants enforced at download time** — the pipeline relies on all of them:

- `price ∈ [0, 1]`, `size > 0`, and `usdc = price × size`;
- no duplicate `(condition_id, token_id, transaction_hash, price, size)` rows;
- only **public taker fills** are present (API parameter `takerOnly=true`); maker
  orders are not included.

Only `side == "BUY"` taker fills feed the price panel: at horizon $h$ a bucket's
price is the mean of all taker trade prices in the $h \pm 0.5$ hour window before
the daily temperature peak (`WIN_HALF_H = 0.5` in `pipeline/constants.py`).

---

## 5. `coverage.parquet` — per-day audit

| Column | Type | Meaning |
|---|---|---|
| `target_date` | str | Weather date |
| `event_count` | int | Events that day |
| `market_count` | int | Buckets that day |
| `token_count` | int | Tokens that day |
| `trade_count` | int | Fills that day |
| `complete` | bool | Whether the day is complete (no missing market, no truncated pagination) |

`complete == false` rows correspond to `incomplete_target_dates` in the manifest.
Use this table to spot gaps before running the panel build.

---

## 6. `manifest.json` — integrity record

| Key | Meaning |
|---|---|
| `data_class` | Always `raw_point_in_time` |
| `provider` | `polymarket_public_api` |
| `requested_range` | `[start_date, end_date]` as requested |
| `row_counts` | Row counts for events / markets / trades |
| `missing_target_dates` | Dates entirely absent |
| `incomplete_target_dates` | Dates partially present |
| `schema_version` | Schema version (`2`) |
| `sha256` | **SHA256 of each Parquet file** — verify after copying |
| `fetched_at` | Download completion time |

To verify a copy:

```python
import hashlib, json, pathlib

base = pathlib.Path("data_store/clob/LONDON/20251205_20260810")
man = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
for name, digest in man["sha256"].items():
    got = hashlib.sha256((base / name).read_bytes()).hexdigest()
    print(name, "OK" if got == digest else "MISMATCH")
```

---

## 7. Per-station inventory (raw downloads)

| Station | Raw artifact dir | Events | Markets | Trades | Coverage |
|---|---|---|---|---|---|
| LONDON | 20250121_20260810 | 563 | 4,593 | 1,795,225 | 2025-01-22 ~ 2026-08-10 |
| CHICAGO | 20251204_20260810 | 203 | 2,073 | 584,165 | 2025-12-04 ~ 2026-08-11 |
| MIAMI | 20250101_20260812 | 204 | 2,084 | 695,543 | 2025-12-04 ~ 2026-08-12 |
| SEATTLE | 20250101_20260812 | 248 | 2,392 | 625,936 | 2025-12-06 ~ 2026-08-13 |
| ATLANTA | 20250101_20260812 | 251 | 2,421 | 692,462 | 2025-12-06 ~ 2026-08-15 |
| TOKYO | 20250101_20260818 | 161 | 1,759 | 570,916 | 2026-03-10 ~ 2026-08-17 |
| SHANGHAI | 20250101_20260818 | 158 | 1,732 | 882,451 | 2026-03-13 ~ 2026-08-17 |
| SEOUL | 20250101_20260818 | 255 | 2,457 | 1,252,836 | 2025-12-06 ~ 2026-08-17 |
| TORONTO | 20250101_20260818 | 255 | 2,465 | 620,582 | 2025-12-06 ~ 2026-08-17 |
| KLGA (NYC) | merged from 3 segments | — | — | — | 2025-12-05 ~ 2026-08-10 |

The raw download totals ~1,144 events and ~7.2M fills across the nine non-NYC
stations. NYC (KLGA) is maintained separately and merged by
`pipeline/0_data_prep/02_merge_nyc_artifact.py`.

**Reading the inventory:** some directories are named after the *requested*
start date rather than the *actual* first date. Chicago and Miami, for instance,
appear as 2025-12-04x because their series were created in mid-December while the
request window reached further back. Always trust `requested_range` in the
manifest and `target_date` in `events.parquet` over the directory name.
Tokyo and Shanghai begin in March 2026 — their series were created then.

---

## 8. Clamping to the canonical window

Raw downloads span wider than the study period. `pipeline/0_data_prep/
03_create_canonical_artifacts.py` produces, for every station, a read-only view
clamped to `[2025-12-05, 2026-08-10]` under the single artifact name
`20251205_20260810`:

- rows outside the window are **treated as expired, never deleted**;
- the raw source directories are left byte-for-byte untouched;
- downstream stages read only the canonical artifact, so the study window is
  defined in exactly one place.

---

## 9. How the raw data was obtained

1. **Event enumeration** — page `gamma-api.polymarket.com/events?series_id={id}`
   for every historical event of a weather series (series IDs live in
   `scripts/download_historical_clob.py`, `DAILY_WEATHER_SERIES`).
2. **Fill download** — for each market call
   `data-api.polymarket.com/trades?market={condition_id}&limit=1000&takerOnly=true`,
   paging until exhausted; a page that lands exactly on 1000 triggers a
   confirmation request so nothing is truncated.
3. **Metadata** — bucket bounds parsed from slugs; resolution taken from the
   official `outcomePrices`.
4. **Downloader** — `data/clob/history.py::HistoricalClobSource`: ≥0.4 s between
   requests, 6 retries, 429 back-off of `30 s · 2^attempt`.
5. **Audit** — `scripts/audit_historical_clob.py` re-checks daily coverage,
   bucket completeness, and fill validity.

---

## 10. Known caveats

1. **Price definitions differ for the July KLGA sample.** The paper's July New
   York test set was collected prospectively and uses live L2 book mid prices;
   every other station uses taker fill prices from this artifact set. The paper
   states this difference explicitly.
2. **`manifest.json` metadata slip.** A few stations (e.g. LONDON) carry a
   hard-coded `station_id` of `KLGA` inside an older manifest. The **data are
   unaffected** — trust the directory name.
3. **Not included in the distributed archive:** KLGA, DALLAS, DENVER, the
   `.raw-cache` JSON download cache, and the canonical clamped copies.
4. **Open-ended buckets** carry `±inf` bounds (see §3).
5. **`source` is always `data_api_trade`** — no other data source is mixed in.
