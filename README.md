# PolyWeatherBot ¡ª Technical Reference

PolyWeatherBot is an automated weather-derivatives trading system for [Polymarket](https://polymarket.com).
It predicts daily temperature highs at airport weather stations, computes per-bucket probabilities with a
Temporal Fusion Transformer (TFT) + Monte Carlo pipeline, and trades the corresponding binary markets
on the Polymarket CLOB.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Design Principles](#2-design-principles)
3. [Installation and Environment](#3-installation-and-environment)
4. [Directory Structure](#4-directory-structure)
5. [Configuration Reference](#5-configuration-reference)
6. [Data Pipeline](#6-data-pipeline)
7. [Model Pipeline](#7-model-pipeline)
8. [Trading Core](#8-trading-core)
9. [Event Management](#9-event-management)
10. [Runtime Loops](#10-runtime-loops)
11. [Sniper System](#11-sniper-system)
12. [Database Schema](#12-database-schema)
13. [Backtest System](#13-backtest-system)
14. [Streamlit UI](#14-streamlit-ui)
15. [Offline Scripts](#15-offline-scripts)
16. [Module I/O Reference](#16-module-io-reference)
17. [CLI Usage](#17-cli-usage)
18. [Running Tests](#18-running-tests)

---

## 1. System Architecture

### High-Level Overview

```
                         ©°©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©´
                         ©¦                  BotManager                        ©¦
                         ©¦  (core/bot.py ¡ª wires and launches all components) ©¦
                         ©¸©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¼
                                                 ©¦
          ©°©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©à©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©´
          ©¦                  ©¦                   ©¦                 ©¦                  ©¦
   DiscoveryLoop      ForecastLoop         SniperLoop      SettlementLoop        (UI)
   (24h interval)    (hourly tick)     (sub-second)       (hourly check)     streamlit
          ©¦                  ©¦                   ©¦                 ©¦
          ©¦          ©°©¤©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©¤©´    ©°©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©´  ©°©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©´
          ©¦          ©¦ ModelPipeline ©¦    ©¦SniperManager©¦  ©¦   Settler   ©¦
          ©¦          ©¦  TFT + MC     ©¦    ©¦  AWC+NOAA   ©¦  ©¦(settlement) ©¦
          ©¦          ©¸©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¼    ©¸©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¼  ©¸©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¼
          ©¦                  ©¦
          ©¦        ©°©¤©¤©¤©¤©¤©¤©¤©¤©¤©à©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©´
          ©¦        ©¦         ©¦          ©¦
          ©¦   DataManager  TftModel  McSampler
          ©¦   (HRRR+IEM)  (PyTorch)   (AR1)
          ©¦        ©¦
          ©¦  ©°©¤©¤©¤©¤©¤©Ø©¤©¤©¤©¤©¤©´
          ©¦  ©¦           ©¦
          ©¦ HRRR       IEM Obs
          ©¦ Sources    Sources
          ©¦  (S3/Zarr)  (API)
          ©¦
   ©°©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©´
   ©¦EventDiscovery©¦ (Gamma API)
   ©¦  DB: events  ©¦
   ©¦  DB: markets ©¦
   ©¸©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¼
```

### Full Data Flow

```
HRRR Zarr S3  ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤?
HRRR NOMADS   ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤? DataManager.fetch_hrrr() ©¤©¤? ParquetStore (hrrr/)
                                                            ©¦
IEM ASOS API  ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤? DataManager.fetch_iem()  ©¤©¤? ParquetStore (obs/)
NOAA File     ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤?                                   ©¦
                                                    ©°©¤©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©¤©´
                                                    ©¦ ModelPipeline ©¦
                                                    ©¦  1. Load 72h  ©¦
                                                    ©¦  2. TFT infer ©¦
                                                    ©¦  3. MC sample ©¦
                                                    ©¦  4. Bucket P  ©¦
                                                    ©¸©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¼
                                                            ©¦ {(lo_f,hi_f): prob}
                                                    ©°©¤©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©¤©´
                                                    ©¦ EdgeStrategy  ©¦
                                                    ©¦ edge=prob-clob©¦
                                                    ©¸©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¼
                                                            ©¦ Signal list
                                                    ©°©¤©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©¤©´
                                                    ©¦  FixedFraction©¦
                                                    ©¦  risk check   ©¦
                                                    ©¸©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¼
                                                            ©¦ approved signals
                                                    ©°©¤©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©¤©´
                                                    ©¦Paper/ClobExch ©¦
                                                    ©¦ place_order() ©¦
                                                    ©¸©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¼
                                                            ©¦
                                                    ©°©¤©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©¤©´
                                                    ©¦PositionTracker©¦
                                                    ©¦  SQLite DB    ©¦
                                                    ©¸©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¼
```

### Sniper (Real-Time) Data Flow

The sniper path bypasses the hourly schedule and fires whenever a new METAR observation
is received from AWC or NOAA:

```
AWC METAR API ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤? AwcSniper.poll()  ©¤©¤?
NOAA tgftp    ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤? NoaaSniper.poll() ©¤©¤? SniperManager.queue
                                                        ©¦ (deduplicated)
                                               ©°©¤©¤©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©¤©¤©´
                                               ©¦  SniperLoop     ©¦
                                               ©¦  _handle_obs()  ©¦
                                               ©¸©¤©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¤©¼
                                                        ©¦ same pipeline flow
                                              ModelPipeline ¡ú Strategy ¡ú Exchange
```

---

## 2. Design Principles

| Principle | Implementation |
|---|---|
| **Interface-first** | Every major subsystem has an ABC base class (`BaseExchange`, `BaseRiskModel`, `BaseStrategy`, `BaseForecastSource`, `BaseObsSource`, `BaseSniperSource`) |
| **One class per file** | Each module owns one primary class plus at most a few pure functions |
| **Dependency injection** | All inter-component references are passed via constructor parameters; no singletons |
| **Fail-safe fallback** | Data sources are tried in priority order; the bot continues if an individual source fails |
| **ASCII-only source** | All `.py` files use ASCII characters only (comments, docstrings, log messages) |
| **Paper-first** | Default mode is paper trading (no real funds); live mode requires explicit `TRADING_MODE=live` env var |
| **Testable** | All loops and components accept in-memory mocks; tests use `DbClient(":memory:")` |

---

## 3. Installation and Environment

### Prerequisites

- Linux (Ubuntu 22.04+ recommended)
- Python 3.10 or 3.11
- System package: `libeccodes-dev` (required for GRIB2 parsing)

```bash
sudo apt-get update
sudo apt-get install -y libeccodes-dev
```

### Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e .

# Install optional: real CLOB trading
pip install -e ".[clob]"

# Install optional: Herbie GRIB fetcher
pip install -e ".[herbie]"

# Install development tools (pytest, black, ruff)
pip install -e ".[dev]"
```

### Environment Variables (.env)

Create a `.env` file in the project root. The bot loads it automatically at startup.

```ini
# Required for live trading only
POLYMARKET_PRIVATE_KEY=<0x-eoa-private-key>
POLYMARKET_WALLET_ADDRESS=<0x-wallet-address>  # optional; leave blank for SDK default

# Optional overrides
TRADING_MODE=paper          # paper | live
GAMMA_API_URL=https://gamma-api.polymarket.com
CLOB_API_URL=https://clob.polymarket.com
POLYWEATHER_DB=/path/to/custom.db    # used by the Streamlit UI
```

All credentials are accessed at runtime via `Settings` properties; they are never
logged or stored in the SQLite database.

### Key Runtime Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas`, `pyarrow` | Array maths, DataFrames, Parquet I/O |
| `xarray`, `cfgrib`, `eccodes` | GRIB2 file parsing for HRRR S3 source |
| `zarr`, `fsspec`, `s3fs` | Zarr store access for HRRR hrrrzarr source |
| `requests` | REST API calls (IEM, Gamma, NOMADS, sniper endpoints) |
| `torch`, `pytorch-forecasting`, `lightning` | TFT model training and inference |
| `python-dotenv` | `.env` loading |
| `polymarket-client` | Polymarket CLOB order placement via `PublicClient` / `SecureClient` (optional) |
| `streamlit`, `plotly` | Web dashboard UI |
| `aiohttp` | Async HTTP for snipers |

---

## 4. Directory Structure

```
PolyWeatherBot/
©À©¤©¤ main.py                  # CLI entry point
©À©¤©¤ pyproject.toml           # Build config and dependency list
©À©¤©¤ .env                     # Runtime credentials (not committed)
©¦
©À©¤©¤ config/
©¦   ©À©¤©¤ settings.py          # Settings dataclass (all constants)
©¦   ©¸©¤©¤ stations.py          # StationConfig definitions + STATIONS dict
©¦
©À©¤©¤ core/
©¦   ©À©¤©¤ bot.py               # BotManager ¡ª top-level wiring
©¦   ©À©¤©¤ types.py             # Signal, Order dataclasses
©¦   ©À©¤©¤ positions.py         # PositionTracker ¡ª SQLite-backed open positions
©¦   ©À©¤©¤ exchange/
©¦   ©¦   ©À©¤©¤ base.py          # BaseExchange ABC
©¦   ©¦   ©À©¤©¤ paper.py         # PaperExchange ¡ª simulated fills
©¦   ©¦   ©À©¤©¤ clob.py          # ClobExchange ¡ª real Polymarket CLOB
©¦   ©¦   ©¸©¤©¤ settlement.py    # Settler ¡ª post-expiry settlement
©¦   ©À©¤©¤ risk/
©¦   ©¦   ©À©¤©¤ base.py          # BaseRiskModel ABC
©¦   ©¦   ©¸©¤©¤ fixed_fraction.py# FixedFraction ¡ª position sizing
©¦   ©¸©¤©¤ strategy/
©¦       ©À©¤©¤ base.py          # BaseStrategy ABC
©¦       ©¸©¤©¤ edge.py          # EdgeStrategy ¡ª mc_prob vs clob_price
©¦
©À©¤©¤ data/
©¦   ©À©¤©¤ manager.py           # DataManager ¡ª multi-source orchestration
©¦   ©À©¤©¤ forecast/
©¦   ©¦   ©À©¤©¤ base.py          # BaseForecastSource ABC
©¦   ©¦   ©À©¤©¤ hrrr_s3.py       # HrrrS3Source ¡ª GRIB2 via NOAA S3
©¦   ©¦   ©À©¤©¤ hrrr_nomads.py   # HrrrNomadsSource ¡ª GRIB2 via NOMADS
©¦   ©¦   ©À©¤©¤ hrrr_zarr.py     # HrrrZarrSource ¡ª Zarr via hrrrzarr S3
©¦   ©¦   ©À©¤©¤ _grib_source.py  # BaseGribSource ¡ª shared GRIB logic
©¦   ©¦   ©¸©¤©¤ vars.py          # HRRR variable definitions (36 vars)
©¦   ©À©¤©¤ obs/
©¦   ©¦   ©À©¤©¤ base.py          # BaseObsSource ABC
©¦   ©¦   ©À©¤©¤ iem.py           # IemSource ¡ª Iowa Environmental Mesonet
©¦   ©¦   ©¸©¤©¤ noaa_file.py     # NoaaFileSource ¡ª NOAA tgftp text file
©¦   ©À©¤©¤ storage/
©¦   ©¦   ©À©¤©¤ parquet_store.py # ParquetStore ¡ª quarterly-partitioned Parquet
©¦   ©¦   ©¸©¤©¤ cleaner.py       # Cleaner ¡ª unit conversion and clipping
©¦   ©¸©¤©¤ clob/
©¦       ©¸©¤©¤ orderbook.py     # CLOB orderbook helper functions
©¦
©À©¤©¤ db/
©¦   ©¸©¤©¤ schema.py            # DbClient ¡ª SQLite schema and CRUD
©¦
©À©¤©¤ events/
©¦   ©À©¤©¤ discovery.py         # EventDiscovery ¡ª Gamma API polling + slug parsing
©¦   ©¸©¤©¤ lifecycle.py         # EventLifecycle ¡ª state machine
©¦
©À©¤©¤ loops/
©¦   ©À©¤©¤ discovery_loop.py    # DiscoveryLoop ¡ª daily event discovery
©¦   ©À©¤©¤ forecast_loop.py     # ForecastLoop ¡ª hourly trading tick
©¦   ©À©¤©¤ settlement_loop.py   # SettlementLoop ¡ª post-expiry settlement
©¦   ©¸©¤©¤ sniper_loop.py       # SniperLoop ¡ª real-time METAR-driven trading
©¦
©À©¤©¤ model/
©¦   ©À©¤©¤ mc.py                # McSampler ¡ª AR(1) Monte Carlo
©¦   ©À©¤©¤ pipeline.py          # ModelPipeline ¡ª TFT + MC orchestration
©¦   ©À©¤©¤ tft.py               # TftModel ¡ª pytorch-forecasting TFT wrapper
©¦   ©¸©¤©¤ store/
©¦       ©¸©¤©¤ registry.py      # ModelRegistry ¡ª checkpoint versioning
©¦
©À©¤©¤ snipers/
©¦   ©À©¤©¤ base.py              # BaseSniperSource ABC
©¦   ©À©¤©¤ manager.py           # SniperManager ¡ª dedup + queue fanout
©¦   ©À©¤©¤ awc.py               # AwcSniper ¡ª Aviation Weather Center METAR
©¦   ©¸©¤©¤ noaa.py              # NoaaSniper ¡ª NOAA tgftp METAR
©¦
©À©¤©¤ backtest/
©¦   ©À©¤©¤ engine.py            # BacktestEngine ¡ª historical pipeline replay
©¦   ©¸©¤©¤ analysis.py          # BacktestAnalysis ¡ª P&L statistics
©¦
©À©¤©¤ scripts/
©¦   ©À©¤©¤ batch_download_zarr.py  # BatchZarrDownloader ¡ª bulk HRRR download
©¦   ©À©¤©¤ batch_download_iem.py   # BatchIemDownloader ¡ª bulk IEM download
©¦   ©¸©¤©¤ train_tft.py            # TftTrainer ¡ª TFT training pipeline
©¦
©À©¤©¤ ui/
©¦   ©À©¤©¤ app.py               # Streamlit entry point
©¦   ©¸©¤©¤ pages/
©¦       ©À©¤©¤ dashboard.py     # Real-time P&L and station status
©¦       ©À©¤©¤ positions.py     # Order history and paper/live toggle
©¦       ©À©¤©¤ events.py        # Event list, lifecycle, bucket prices
©¦       ©À©¤©¤ backtest.py      # Interactive backtest controls
©¦       ©¸©¤©¤ model.py         # Model versions and per-station RMSE
©¦
©¸©¤©¤ tests/
    ©À©¤©¤ conftest.py          # Shared pytest fixtures
    ©¸©¤©¤ test_*/              # One sub-package per source package
```

---

## 5. Configuration Reference

### `config/settings.py` ¡ª `Settings` Dataclass

All configuration lives in a single `Settings` dataclass. Paths use `pathlib.Path`.

#### Path Settings

| Field | Default | Description |
|---|---|---|
| `project_root` | repo root | Resolved absolute path to repo root |
| `data_store_dir` | `{root}/data_store` | Root for ParquetStore data |
| `model_store_dir` | `{root}/model/store` | Root for ModelRegistry checkpoints |

#### Trading Settings

| Field | Default | Description |
|---|---|---|
| `trading_mode` | `"paper"` | `"paper"` or `"live"` (overridden by `TRADING_MODE` env) |
| `min_edge` | `0.04` | Minimum edge fraction to generate a signal |
| `max_position_usd` | `50.0` | Maximum USD per single position |
| `max_total_usd` | `500.0` | Maximum total USD across all open positions |

#### HRRR Source

| Field | Default | Description |
|---|---|---|
| `hrrr_zarr_bucket` | `"hrrrzarr"` | S3 bucket name for Zarr source |
| `hrrr_s3_bucket` | `"noaa-hrrr-bdp-pds"` | S3 bucket name for GRIB2 source |
| `hrrr_source_priority` | `["s3", "nomads"]` | Ordered list of HRRR source names to try |

#### Monte Carlo Parameters

| Field | Default | Description |
|---|---|---|
| `mc_n_samp` | `10000` | Number of Monte Carlo samples |
| `mc_rng_seed` | `42` | RNG seed for reproducibility |

#### Scheduling / TradingLoop

| Field | Default | Description |
|---|---|---|
| `trading_loop_interval_s` | `300` | Seconds between TradingLoop ticks |
| `staleness_threshold_min` | `65` | Minutes after last ForecastLoop tick before trading halts |
| `staleness_edge_bump` | `0.02` | Edge reduction applied to stale-data signals |
| `cold_start_hours` | `72` | Hours of history required before any order is placed |
| `max_open_per_token` | `1` | Maximum concurrent open orders per token |

#### Feature Flags

| Field | Default | Env override |
|---|---|---|
| `enable_sniper` | `False` | `ENABLE_SNIPER=1` |

#### Model Training Windows

| Field | Default |
|---|---|
| `train_start` | `"2024-10-01"` |
| `train_end` | `"2025-12-31"` |
| `val_start` | `"2026-01-01"` |
| `val_end` | `"2026-02-28"` |
| `test_start` | `"2026-03-01"` |
| `test_end` | `"2026-04-30"` |

#### API URLs

| Field | Default |
|---|---|
| `iem_api_url` | `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` |
| `noaa_tgftp_url_tpl` | `https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT` |
| `awc_metar_url` | `https://aviationweather.gov/api/data/metar` |
| `hrrr_nomads_base` | `https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod` |
| `gamma_api_url` | `https://gamma-api.polymarket.com` (env override) |
| `clob_api_url` | `https://clob.polymarket.com` (env override) |

### `config/stations.py` ¡ª `StationConfig`

```python
@dataclass
class StationConfig:
    station_id:         str    # e.g. "KLGA"
    lat:                float  # decimal degrees N
    lon:                float  # decimal degrees E (negative = W)
    timezone:           str    # IANA timezone name
    tz_offset:          int    # hours offset from UTC (e.g -5 for EST)
    hrrr_grid_row_zarr: int    # Zarr grid row
    hrrr_grid_col_zarr: int    # Zarr grid column
    iem_network:        str    # IEM network ID (e.g. "NY_ASOS")
    slug_pattern:       str    # regex for Gamma API slug matching
```

Currently configured stations: `KLGA` (LaGuardia Airport, New York).

---

## 6. Data Pipeline

### Overview

The data layer provides a unified interface over multiple redundant sources.
All sources implement the same ABC so they can be swapped transparently.

```
Priority order (tried left to right, first success wins):
  HRRR forecast:  HrrrZarrSource  ¡ú  HrrrS3Source  ¡ú  HrrrNomadsSource
  Observations:   IemSource       ¡ú  NoaaFileSource
```

### `data/manager.py` ¡ª `DataManager`

The coordinator. Delegates to the highest-priority available source.

**Constructor parameters:**

| Parameter | Type | Description |
|---|---|---|
| `forecast_sources` | `list[BaseForecastSource]` | Ordered list of HRRR sources |
| `obs_sources` | `list[BaseObsSource]` | Ordered list of observation sources |
| `forecast_store` | `ParquetStore` | Store for HRRR output |
| `obs_store` | `ParquetStore` | Store for observation output |

**Methods:**

| Method | Input | Output | Description |
|---|---|---|---|
| `fetch_hrrr(init_time, station, fxx_range)` | UTC timestamp, StationConfig, range | `pd.DataFrame` | Fetches and caches HRRR forecast |
| `fetch_iem(station, start, end)` | StationConfig, timestamps | `pd.DataFrame` | Fetches and caches IEM observations |

Both methods append results to the respective `ParquetStore` and raise `RuntimeError` if all sources fail.

### `data/forecast/` ¡ª HRRR Forecast Sources

All sources return a `pd.DataFrame` with the following columns:

| Column | Type | Description |
|---|---|---|
| `init_time` | UTC datetime | HRRR initialization time |
| `fxx` | int | Forecast hour (0=analysis, 1..48=forecast) |
| `valid_time` | UTC datetime | `init_time + fxx hours` |
| `TMP_2m` | float | 2m temperature (Kelvin, raw from HRRR) |
| `DPT_2m` | float | 2m dew point (Kelvin) |
| `RH_2m` | float | 2m relative humidity (%) |
| `TMP_sfc` | float | Surface temperature (Kelvin) |
| `TMP_850mb` | float | 850 hPa temperature (Kelvin) |
| `TMP_500mb` | float | 500 hPa temperature (Kelvin) |
| `UGRD_10m` | float | 10m U-wind (m/s) |
| `VGRD_10m` | float | 10m V-wind (m/s) |
| `GUST_sfc` | float | Surface wind gust (m/s) |
| `UGRD_80m` | float | 80m U-wind (m/s) |
| `VGRD_80m` | float | 80m V-wind (m/s) |
| `UGRD_500mb` | float | 500 hPa U-wind (m/s) |
| `VGRD_500mb` | float | 500 hPa V-wind (m/s) |
| `RH_0C` | float | RH at 0¡ãC isotherm (%) |
| `RH_frz` | float | RH at highest freezing level (%) |
| `DSWRF` | float | Downward shortwave radiation (W/m?) |
| `DLWRF` | float | Downward longwave radiation (W/m?) |
| `USWRF` | float | Upward shortwave radiation (W/m?) |
| `ULWRF` | float | Upward longwave radiation (W/m?) |
| `TCDC` | float | Total cloud cover (%) |
| `LCDC` | float | Low cloud cover (%) |
| `MCDC` | float | Mid cloud cover (%) |
| `HCDC` | float | High cloud cover (%) |
| `PRATE` | float | Precipitation rate (kg/m?/s) |
| `APCP` | float | 1-hour accumulated precipitation (kg/m?) |
| `REFC` | float | Composite reflectivity (dBZ) |
| `PWAT` | float | Precipitable water (kg/m?) |
| `CAPE` | float | Surface CAPE (J/kg) |
| `CIN` | float | Surface CIN (J/kg) |
| `HPBL` | float | Planetary boundary layer height (m) |
| `SHTFL` | float | Sensible heat flux (W/m?) |
| `LHTFL` | float | Latent heat flux (W/m?) |
| `PRES_sfc` | float | Surface pressure (Pa) |
| `MSLMA` | float | Mean sea level pressure (Pa) |
| `VIS` | float | Visibility (m) |
| `SNOD` | float | Snow depth (m) |

**`HrrrZarrSource`** (preferred):
- URL: `https://hrrrzarr.s3.amazonaws.com/sfc/{YYYYMMDD}/{YYYYMMDD}_{HH}z_{anl|fcst}.zarr`
- Opens two Zarr stores per init: `_anl.zarr` (analysis, 2D arrays) and `_fcst.zarr` (forecast, 3D arrays)
- No authentication required; uses public S3 access

**`HrrrS3Source`** (fallback 1):
- URL: `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date}/conus/hrrr.t{HH}z.wrfsfcf{fxx:02d}.grib2`
- Uses byte-range requests via `.idx` files (avoids downloading full GRIB2)
- Requires `cfgrib` + `eccodes` for parsing

**`HrrrNomadsSource`** (fallback 2):
- URL: `https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/hrrr.{date}/conus/...`
- Same GRIB2 parsing as S3 source
- Only has recent data (~2 days)

### `data/obs/` ¡ª Observation Sources

All sources return a `pd.DataFrame` with the following columns:

| Column | Type | Description |
|---|---|---|
| `valid_time` | UTC-aware datetime | Observation timestamp |
| `raw_tmpc_tenths` | float | Temperature in tenths of ¡ãC (e.g. 225 = 22.5¡ãC) |
| `raw_dwpc_tenths` | float | Dew point in tenths of ¡ãC |
| `raw_wind_spd_kt` | float | Wind speed in knots |
| `raw_wind_gust_kt` | float | Wind gust in knots |
| `obs_U` | float | U-wind component (m/s) |
| `obs_V` | float | V-wind component (m/s) |
| `raw_vis_sm` | float | Visibility in statute miles |
| `raw_alti_inhg` | float | Altimeter setting in inHg |
| `skyc1` | float | Sky cover fraction (0..100) |
| `skyl1` | float | Sky layer base height (ft) |
| `p01i` | float | 1-hour precipitation (inches) |
| `is_rain` | float | 1.0 if rain present |
| `is_snow` | float | 1.0 if snow present |

**`IemSource`** (preferred):
- Endpoint: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`
- Query params: station, date range, UTC timezone, comma format
- Parses T-group enhancers in METAR text for 0.1¡ãF precision
- Sky cover text codes mapped to percentages

**`NoaaFileSource`** (fallback):
- URL: `https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station}.TXT`
- Single-observation text file (real-time only)
- Used as fallback when IEM API is unavailable

### `data/storage/parquet_store.py` ¡ª `ParquetStore`

Quarterly-partitioned columnar storage. Each station gets its own sub-directory.

**Layout on disk:**
```
data_store/
    hrrr/
        KLGA/
            2025-Q4.parquet
            2026-Q1.parquet
    obs/
        KLGA/
            2025-Q4.parquet
```

**Methods:**

| Method | Input | Output | Description |
|---|---|---|---|
| `append(station_id, df)` | str, DataFrame | int | Dedup and append rows; returns count written |
| `read(station_id, start, end)` | str, Timestamp, Timestamp | DataFrame | Read rows in time range |

**Deduplication keys:**
- HRRR data: `(init_time, fxx)` composite key
- Observation data: `valid_time`

### `data/storage/cleaner.py` ¡ª `Cleaner`

Post-processing applied to raw HRRR/IEM data before model input:

| Transformation | Details |
|---|---|
| Kelvin ¡ú Celsius | `TMP_2m`, `DPT_2m`, `TMP_sfc`, `TMP_850mb`, `TMP_500mb` |
| m/s ¡ú knots | Wind speed columns |
| Physical clipping | Temperature: [-80, 60]¡ãC; RH: [0, 103]%; cloud: [0, 100]%; visibility: [0, 16 km]; snow: [0, 300 cm] |
| IEM zero-fill | Missing precipitation and flag columns ¡ú 0.0 |

The model pipeline may run cleaning automatically before feeding data to the TFT encoder.

---

## 7. Model Pipeline

### Overview

The model pipeline converts a 48-hour HRRR forecast + recent observations into
per-bucket probability estimates for the daily maximum temperature.

```
                   HRRR(init, fxx=1..48)       IEM obs(init-23h..init)
                           ©¦                              ©¦
                    ©°©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©¤©¤©´
                    ©¦            Feature Engineering                ©¦
                    ©¦  K¡úC  |  wind merging  |  log1p  |  residual ©¦
                    ©¸©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¼
                                           ©¦  72-step sequence (local_idx 0..71)
                                    ©°©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©´
                                    ©¦   TftModel  ©¦
                                    ©¦   encoder   ©¦  ¡û obs window (idx 0..23)
                                    ©¦   decoder   ©¦  ¡û HRRR forecast (idx 24..71)
                                    ©¸©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¼
                                           ©¦  p10, p50, p90 quantile predictions
                                           ©¦  (residuals: temp_error = obs - hrrr)
                                    ©°©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©´
                                    ©¦  McSampler  ©¦
                                    ©¦  AR(1) MC   ©¦
                                    ©¦  n=5000 samp©¦
                                    ©¸©¤©¤©¤©¤©¤©¤©Ð©¤©¤©¤©¤©¤©¤©¼
                                           ©¦  (5000, 48) temperature samples
                                    ©°©¤©¤©¤©¤©¤©¤¨‹©¤©¤©¤©¤©¤©¤©´
                                    ©¦Bucket Probs ©¦
                                    ©¦for each     ©¦
                                    ©¦(lo_f, hi_f) ©¦
                                    ©¸©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¼
```

### `model/tft.py` ¡ª `TftModel`

Wraps `pytorch_forecasting.TemporalFusionTransformer` for inference.

**Sequence structure:**

| Index range | local_idx | Source | Content |
|---|---|---|---|
| 0..23 | 0..23 | Encoder | IEM observations (24 hours before init) |
| 24..71 | 24..71 | Decoder | HRRR forecast (fxx=1..48) |

**Feature engineering (applied before inference):**

| Transformation | Detail |
|---|---|
| Kelvin ¡ú Celsius | Temperature columns > 200K are converted |
| Wind merging | U/V components ¡ú (speed, sin_dir, cos_dir) for 10m, 80m, 500mb |
| Observed wind | `obs_U`, `obs_V` ¡ú `obs_WS`, `obs_WD_sin`, `obs_WD_cos` |
| Log1p | `GUST_sfc`, `PRATE` ¡ú `log1p_GUST_sfc`, `log1p_PRATE` |
| Residual target | `temp_error = raw_tmpc_tenths - TMP_2m` (in training) |

**Known reals (future-known features, 40 total in final model):**
Calendar cyclic features + 36 HRRR-derived variables (with wind merging and log transforms).

**Unknown reals (encoder-only, 14 total):**
IEM observation columns: temperature, dew point, wind, visibility, altimeter, sky cover, precipitation, rain/snow flags.

**Output:**
The TFT predicts `temp_error` (residual from HRRR) for fxx=1..48 at quantiles [0.10, 0.50, 0.90].
The pipeline reconstructs absolute temperature by adding `TMP_2m` back to the predicted residual.

**Target:**
`temp_error = observed_temperature - HRRR_2m_forecast_temperature`

### `model/mc.py` ¡ª `McSampler`

Generates correlated temperature samples using an AR(1) process.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `n_samp` | 5000 | Number of Monte Carlo samples |
| `rng_seed` | 42 | NumPy RNG seed |
| `rho` | 0.90 | AR(1) inter-hour correlation (adjacent forecast hours) |
| `bias_corr_c` | 1.5206 | Celsius additive bias correction |

**Method:**

```python
McSampler.ar1_mc_sample(mu_vec: np.ndarray, sigma_vec: np.ndarray) -> np.ndarray
```

- Input: `mu_vec` (shape `n_hours`,), `sigma_vec` (shape `n_hours`,) ¡ª hourly mean and standard deviation of the temperature forecast
- Output: `(n_samp, n_hours)` array of correlated temperature samples in ¡ãC
- AR(1) structure: `epsilon_t = rho * epsilon_{t-1} + sqrt(1-rho^2) * N(0,1)`
- Each sample's daily maximum is taken to estimate `P(daily_high in bucket)`

### `model/pipeline.py` ¡ª `ModelPipeline`

Orchestrates the full inference flow. Called by both `ForecastLoop` and `SniperLoop`.

**Constructor:**

| Parameter | Type | Description |
|---|---|---|
| `tft` | `TftModel | None` | Loaded TFT model (None ¡ú MC-only fallback) |
| `mc` | `McSampler` | Configured MC sampler |
| `data_mgr` | `DataManager` | Data fetching coordinator |

**Main method:**

```python
ModelPipeline.run(
    station:   StationConfig,
    init_time: pd.Timestamp,
    buckets:   list[tuple[float, float]]
) -> dict
```

Return dict:

| Key | Value |
|---|---|
| `"station"` | station_id string |
| `"init_time"` | pd.Timestamp (UTC) |
| `"probs"` | `{(lo_f, hi_f): probability}` ¡ª one entry per requested bucket |
| `"p50_high_f"` | float ¡ª median daily max in ¡ãF |

**Bucket probability convention:**

| Type | Condition | Slug example |
|---|---|---|
| Range | `lo_f <= daily_max <= hi_f` | `will-be-60-70f` |
| Below | `daily_max <= hi_f` | `60forbelow` |
| Higher | `daily_max >= lo_f` | `70f-or-higher` |

The MC boundary convention mirrors the HRRR resolution: samples `>= lo AND < hi+1`.

### `model/store/registry.py` ¡ª `ModelRegistry`

Manages checkpoint publication and versioning.

**Directory layout:**
```
model/store/
    v20260101_120000/
        model.ckpt
        training_dataset.pkl   (optional)
        meta.json
    v20260115_060000/
        ...
    active_version.txt         (= "v20260115_060000")
```

**Methods:**

| Method | Input | Description |
|---|---|---|
| `publish(version, ckpt_path, metrics, training_dataset_pkl)` | str, Path, dict, Path | Copy checkpoint, write meta.json, update active_version.txt, register in DB |
| `load_active()` | ¡ª | Load the TFT model from the active checkpoint |
| `list_versions()` | ¡ª | Return list of all registered versions |
| `rollback(version)` | str | Set active_version.txt to a previous version |

---

## 8. Trading Core

### `core/types.py` ¡ª Domain Types

**`Signal` dataclass:**

| Field | Type | Description |
|---|---|---|
| `token_id` | str | Polymarket market token ID |
| `side` | str | `"BUY"` (bet YES) or `"SELL"` (unwind YES) |
| `mc_prob` | float | Model-computed probability [0, 1] |
| `clob_price` | float | Current CLOB mid-price [0, 1] |
| `edge` | float | `mc_prob - clob_price` (positive = model says YES is underpriced) |
| `bucket_lo_f` | float | Temperature bucket lower bound (¡ãF) |
| `bucket_hi_f` | float | Temperature bucket upper bound (¡ãF) |
| `condition_id` | str | Polymarket condition ID (event) |

**`Order` dataclass:**

| Field | Type | Description |
|---|---|---|
| `order_id` | str | Exchange-assigned order ID |
| `token_id` | str | Market token ID |
| `side` | str | `"BUY"` or `"SELL"` |
| `size_usd` | float | Notional size in USD |
| `price` | float | Submitted price [0, 1] |
| `status` | str | `"pending"`, `"filled"`, `"closed"` |
| `filled_price` | float | Actual fill price (set after fill) |

### `core/strategy/edge.py` ¡ª `EdgeStrategy`

Computes signals from the gap between model probability and CLOB price.

**Logic:**
```
edge = mc_prob - clob_price
signal generated when:
    BUY:  edge >  min_edge   (model says YES is cheap)
    SELL: edge < -min_edge   (model says YES is expensive / unwind)
```

Only YES-outcome tokens are evaluated. Signals are sorted by `|edge|` descending
so the highest-conviction trades are processed first when the risk model fills the
exposure budget.

**Input to `compute_signals()`:**

| Parameter | Type | Description |
|---|---|---|
| `probs` | `dict[(float, float), float]` | `{(lo_f, hi_f): probability}` from ModelPipeline |
| `books` | `dict[str, dict]` | `{token_id: {"mid_price": float, ...}}` from CLOB |
| `markets` | `list[dict]` | Market rows from DB (token_id, bucket_lo_f, bucket_hi_f, outcome) |

**Output:** `list[Signal]`

### `core/risk/fixed_fraction.py` ¡ª `FixedFraction`

Controls how much capital is allocated per signal.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `position_usd` | 20.0 | Target USD per approved signal |
| `max_position_usd` | `settings.max_position_usd` (50) | Hard cap per single position |
| `max_total_usd` | `settings.max_total_usd` (500) | Hard cap on total portfolio exposure |

**`approve_signal(signal, current_exposure) -> bool`**:
Returns `False` if adding `position_usd` would exceed `max_total_usd`.

**`compute_size(signal, current_exposure) -> float`**:
Returns `min(position_usd, max_position_usd, max_total_usd - current_exposure)`.

### `core/exchange/paper.py` ¡ª `PaperExchange`

Paper-trading simulation. No network calls. Fills are assumed immediate at the submitted price.

**`place_order(token_id, side, size_usd, price) -> str`**:
Generates a UUID order ID, records the fill immediately in memory,
and updates the running net position for `token_id`.

**`get_positions() -> list[dict]`**:
Returns a list of `{"token_id": str, "size": float, "avg_price": float}` for all non-zero positions.

### `core/exchange/clob.py` — `ClobExchange`

Live Polymarket CLOB adapter. Wraps `polymarket_client.SecureClient`.

**`place_order(token_id, side, size_usd, price) -> str`**:
Calls `SecureClient.place_limit_order(token_id, price, size, side)`.
Returns the exchange-assigned order ID, or empty string on rejection.

**`cancel_order(order_id) -> bool`**:
Calls `SecureClient.cancel_orders(order_ids=[order_id])`. Returns `True` on success.

### `core/exchange/settlement.py` ¡ª `Settler`

Handles post-expiry settlement of YES positions.

**`settle(event, exchange) -> str`**:
1. Attempts Polymarket Relayer gasless settlement (if `POLYMARKET_PROXY_ADDRESS` is set).
2. On failure: queries all open YES positions for this event's markets and places SELL orders at price 0.99.
3. Returns `"relayer"` or `"fallback"`.

### `core/positions.py` ¡ª `PositionTracker`

Persists all positions in SQLite.

| Method | Description |
|---|---|
| `open(sig, order)` | Insert a pending order row using the lifecycle schema |
| `total_exposure()` | Sum of `size_usd` for all pending orders |

Full lifecycle methods (record_order, mark_filled, mark_closed, etc.) are added in F03.

---

## 9. Event Management

### `events/discovery.py` ¡ª `EventDiscovery`

Polls the Polymarket Gamma API for temperature-based binary markets.

**Workflow:**
1. `GET {gamma_api_url}/events` with search params.
2. For each event, check if the slug matches `station.slug_pattern` (regex).
3. Parse each market slug to extract the temperature bucket `(lo_f, hi_f)`.
4. Upsert to `events` and `markets` tables in SQLite.

**`parse_slug(slug) -> (lo_f, hi_f)`:**

| Pattern | Example | Output |
|---|---|---|
| `{X}f-or-higher` | `will-be-70f-or-higher` | `(70.0, inf)` |
| `{X}forhigher` | `70forhigher` | `(70.0, inf)` |
| `{X}forbelow` | `60forbelow` | `(-inf, 60.0)` |
| `{X}-{Y}f` | `60-70f` | `(60.0, 70.0)` |

Returns `(nan, nan)` for unrecognized patterns.

### `events/lifecycle.py` ¡ª `EventLifecycle`

State machine for event trading windows.

**States and transitions:**

```
discovered  ©¤©¤? active  ©¤©¤? cutoff  ©¤©¤? pending_settle  ©¤©¤? settled
     ©¦                                        ©¦
     ©¸©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤? cancelled
```

| State | Condition | Tradeable |
|---|---|---|
| `discovered` | Event found but not yet in trading window | No |
| `active` | Within trading window, before cutoff | **Yes** |
| `cutoff` | After 21:00 UTC on the event date | No |
| `pending_settle` | After event_end_date | No |
| `settled` | Settlement confirmed | No |
| `cancelled` | Event cancelled by Polymarket | No |

**`is_tradeable(event, now_utc) -> bool`:**
Returns `True` only when `status == "active"`.

---

## 10. Runtime Loops

All loops run as `asyncio` coroutines within the single `BotManager.run()` gather:

```python
await asyncio.gather(
    _forecast_hourly(),      # ForecastLoop every 60 min
    sniper_loop.run(),       # SniperLoop owns source polling + queue processing
    self.discovery_loop.run(),   # DiscoveryLoop every 24h
    self.settlement_loop.run(),  # SettlementLoop every 1h
)
```

### `loops/discovery_loop.py` ¡ª `DiscoveryLoop`

| Parameter | Default | Description |
|---|---|---|
| `interval_seconds` | 86400 | Seconds between discovery runs (24h) |

Calls `EventDiscovery.discover()` in a thread (via `asyncio.to_thread`), catches
all exceptions silently so the bot continues operating if the Gamma API is down.

### `loops/forecast_loop.py` ¡ª `ForecastLoop`

The main hourly trading tick.

**`tick(init_time)` flow:**
1. Query `db.get_active_events()` ¡ª events with `status='active'`.
2. For each event, fetch markets with `db.get_markets_for_event()`.
3. Run `ModelPipeline.run(station, init_time, buckets)`.
4. Fetch CLOB orderbook for all token_ids ¡ú compute mid-prices.
5. Call `EdgeStrategy.compute_signals(probs, books, markets)`.
6. For each signal: `FixedFraction.approve_signal()` ¡ú `exchange.place_order()` ¡ú `PositionTracker.open()`.
7. Returns `list[Order]` of placed orders.

### `loops/settlement_loop.py` ¡ª `SettlementLoop`

| Parameter | Default | Description |
|---|---|---|
| `interval_seconds` | 3600 | Seconds between settlement checks (1h) |

On each tick:
1. Load all non-settled, non-cancelled events.
2. For each, call `EventLifecycle.advance()` to update state.
3. For events in `pending_settle`: call `Settler.settle()`, then `db.update_event_status('settled')`.

### `loops/sniper_loop.py` ¡ª `SniperLoop`

Runs the full sniper subsystem: starts `SniperManager` source polling and consumes
the `SniperManager.queue` of new METAR observations.

For each observation dequeued:
1. Parse the METAR text to extract current temperature.
2. Identify the matching temperature bucket from active tradeable events.
3. Submit ground-truth BUY signals through risk, exchange, and position recording.

---

## 11. Sniper System

The sniper system provides **sub-second latency** observation capture around the typical
METAR transmission window (each station reports once per hour, usually at xx:51-xx:53 UTC).

### `snipers/awc.py` ¡ª `AwcSniper`

Polls the Aviation Weather Center METAR API.

| Parameter | Value | Description |
|---|---|---|
| `NORMAL_INTERVAL` | 2.0 s | Polling interval outside sniper window |
| `SNIPER_INTERVAL` | 0.65 s | Polling interval inside sniper window |
| Sniper window | `xx:51:30 ¨C xx:53:59` | When new METARs typically arrive |

- Endpoint: `https://aviationweather.gov/api/data/metar?ids={station}&format=raw`
- Detects new observations by comparing the raw METAR text.

### `snipers/noaa.py` ¡ª `NoaaSniper`

Polls the NOAA tgftp file server.

| Parameter | Value | Description |
|---|---|---|
| `IDLE_INTERVAL` | 3.0 s | Polling interval using HEAD requests |
| `SNIPER_INTERVAL` | 0.2 s | Polling interval inside sniper window |
| Sniper window | `xx:50:45 ¨C xx:56:00` | Slightly earlier window than AWC |

- Uses HTTP `HEAD` requests during idle periods to watch `Last-Modified` header cheaply.
- Switches to full `GET` requests inside the sniper window.

### `snipers/manager.py` ¡ª `SniperManager`

- Launches all source pollers concurrently via `asyncio.gather`.
- Deduplicates observations: stripped METAR text is added to `seen_obs` set; duplicates are silently dropped.
- Places new unique observations into `asyncio.Queue` for consumption by `SniperLoop`.
- Does not consume the queue or submit trades; `SniperLoop` owns that lifecycle.

**Callback signature used by snipers:**
```python
on_observation(metar_text: str, source: str, timestamp: datetime) -> None
```

---

## 12. Database Schema

SQLite database, WAL mode, foreign keys enabled.

### Table: `stations`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `station_id` | TEXT | PRIMARY KEY | e.g. `"KLGA"` |
| `lat` | REAL | | Latitude |
| `lon` | REAL | | Longitude |
| `timezone` | TEXT | | IANA timezone name |

### Table: `events`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `condition_id` | TEXT | PRIMARY KEY | Polymarket condition ID |
| `question` | TEXT | | Full question text |
| `event_slug` | TEXT | | Polymarket event slug |
| `event_end_date` | TEXT | | ISO date string |
| `station_id` | TEXT | FK ¡ú stations | Associated station |
| `status` | TEXT | DEFAULT 'discovered' | Lifecycle state |

### Table: `markets`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `token_id` | TEXT | PRIMARY KEY | Polymarket YES token ID |
| `condition_id` | TEXT | FK ¡ú events | Parent event |
| `market_slug` | TEXT | | Market-level slug |
| `bucket_lo_f` | REAL | | Lower bound (¡ãF), NULL = -inf |
| `bucket_hi_f` | REAL | | Upper bound (¡ãF), NULL = +inf |
| `outcome` | TEXT | | `"YES"` or `"NO"` |

### Table: `signals`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `signal_id` | INTEGER | PK AUTOINCREMENT | |
| `timestamp` | TEXT | | UTC ISO timestamp |
| `token_id` | TEXT | FK ¡ú markets | |
| `mc_prob` | REAL | | Model probability |
| `clob_price` | REAL | | CLOB mid-price at signal time |
| `edge` | REAL | | `mc_prob - clob_price` |
| `side` | TEXT | | `"BUY"` or `"SELL"` |

### Table: `orders`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `order_id` | TEXT | PRIMARY KEY | Exchange-assigned ID |
| `token_id` | TEXT | FK -> markets | Token traded |
| `condition_id` | TEXT | FK -> events | Parent event |
| `side` | TEXT | | `"BUY"` or `"SELL"` |
| `size_usd` | REAL | | Notional size |
| `price` | REAL | | Submitted limit price |
| `fill_price` | REAL | | Actual fill price (NULL until filled) |
| `status` | TEXT | DEFAULT 'pending' | `pending / filled / cancelled / closed / settled` |
| `created_at` | TEXT | | UTC ISO timestamp of order creation |
| `filled_at` | TEXT | | UTC ISO timestamp of fill |
| `closed_at` | TEXT | | UTC ISO timestamp of close |
| `cancelled_at` | TEXT | | UTC ISO timestamp of cancellation |
| `settled_payout` | REAL | | Payout received on settlement |

### Table: `model_versions`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `version` | TEXT | PRIMARY KEY | Version string |
| `checkpoint_path` | TEXT | | Path to `.ckpt` file |
| `train_mae` | REAL | | Training MAE |
| `test_mae` | REAL | | Test MAE |
| `created_at` | TEXT | | UTC ISO timestamp |

---

## 13. Backtest System

The backtest system replays the model pipeline against historical CLOB snapshot data
to measure how the strategy would have performed.

### `backtest/engine.py` ¡ª `BacktestEngine`

**Constructor:**

| Parameter | Type | Description |
|---|---|---|
| `pipeline` | `ModelPipeline` | Model pipeline (uses historical data from ParquetStore) |
| `strategy` | `BaseStrategy` | Strategy instance (used to enumerate buckets) |
| `clob_df` | `pd.DataFrame` | Historical CLOB snapshot DataFrame |
| `actual_highs` | `dict[str, float]` | `{"YYYY-MM-DD": actual_daily_max_f}` |
| `min_edge` | `float` | Minimum |edge| to record a row (default 0.0) |

**`clob_df` required columns:**

| Column | Description |
|---|---|
| `timestamp_utc` | Snapshot timestamp (UTC-aware) |
| `event_end_date` | Event date (UTC-aware) |
| `market_slug` | Polymarket slug |
| `bucket_lo_f` | Lower temperature bound (¡ãF) |
| `bucket_hi_f` | Upper temperature bound (¡ãF) |
| `mid_price_yes` | CLOB mid-price for YES |
| `estimated_spread` | Bid-ask spread estimate |
| `is_filled` | 0=real quote, 1=synthetic fill |

**`run(start_date, end_date, station, init_hours) -> pd.DataFrame`**

Output DataFrame columns (one row per `(init_time, bucket)` signal):

| Column | Description |
|---|---|
| `target_date` | Event date string `"YYYY-MM-DD"` |
| `init_utc` | Init time string (UTC) |
| `init_est` | Init time in EST |
| `n_hours` | EST hours of HRRR coverage on target date |
| `coverage_type` | `"full"` (24h) or `"partial"` (8..23h) |
| `lead_hours_to_4pm` | Hours from init to 21:00 UTC cutoff |
| `bucket_lo_f` | Temperature bucket lower bound (¡ãF) |
| `bucket_hi_f` | Temperature bucket upper bound (¡ãF) |
| `market_slug` | Market slug |
| `mc_prob` | Model probability |
| `clob_price` | CLOB mid-price |
| `estimated_spread` | Bid-ask spread |
| `halfspread` | `estimated_spread / 2` |
| `clob_quality` | `"real"` or `"filled"` |
| `edge` | `mc_prob - clob_price` |
| `actual_max_f` | Actual observed daily maximum (¡ãF) |
| `actual_outcome` | 1.0 if outcome = YES, 0.0 if NO, NaN if unknown |
| `pnl_if_yes` | `actual_outcome - clob_price` (P&L from buying YES) |
| `pnl_if_no` | `clob_price - actual_outcome` (P&L from selling YES) |

**CLOB window selection:**
- Window: `init_utc ¡À 30 minutes`
- Prefers `is_filled=0` (real quotes) over `is_filled=1` (synthetic fills)
- Aggregates multiple quotes by median per `(market_slug, bucket_lo_f, bucket_hi_f)`

**Key constants:**

| Constant | Value | Description |
|---|---|---|
| `HALFSPREAD_DRAG` | -0.0127 | Average half-spread cost per $1 bet |
| `_CLOB_WIN_MIN` | 30 | CLOB lookup window half-width (minutes) |
| `_MIN_HOURS` | 8 | Minimum EST coverage hours to process an init |
| `_CUTOFF_EST_HOUR` | 16 | EST cutoff hour (= 21:00 UTC) |

### `backtest/analysis.py` ¡ª `BacktestAnalysis`

Computes performance statistics from a `BacktestEngine` results DataFrame.

**Constructor:**
```python
BacktestAnalysis(results_df: pd.DataFrame)
```

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `summary(threshold=0.05)` | `dict` | ROI, Brier scores, BSS, max drawdown |
| `by_threshold(thresholds)` | `pd.DataFrame` | P&L breakdown per edge threshold |
| `by_lead_time(thresholds)` | `pd.DataFrame` | P&L breakdown per lead-time bin |
| `by_init_hour(thresholds)` | `pd.DataFrame` | P&L breakdown per init EST hour |

**`summary()` dict keys:**

| Key | Description |
|---|---|
| `N_yes`, `N_no`, `N_total` | Signal counts by direction |
| `PnL_mid`, `PnL_real` | Total P&L (mid = no spread cost; real = after half-spread) |
| `ROI_mid`, `ROI_real` | Return-on-investment as fraction |
| `WinRate_YES`, `WinRate_NO` | Win rates by direction |
| `brier_market` | Brier score for CLOB price: `mean((clob_price - outcome)^2)` |
| `brier_tft` | Brier score for model: `mean((mc_prob - outcome)^2)` |
| `bss` | Brier Skill Score: `1 - brier_tft / brier_mkt` (positive = model better than CLOB) |
| `max_drawdown_mid` | Maximum peak-to-trough drawdown on cumulative mid P&L |

**Lead time bins:** `0-6h`, `6-12h`, `12-24h`, `24-36h`, `36-48h`, `>48h`

---

## 14. Streamlit UI

Launch with:
```bash
streamlit run ui/app.py
```

The app provides 5 pages accessible via the sidebar:

### Dashboard (`ui/pages/dashboard.py`)

- **KPIs:** Total orders, open orders, approximate P&L
- **Active Events by Station:** aggregated from DB (count, statuses)
- **Recent Signals:** last 50 rows (timestamp, token, side, edge, size, status)

### Positions (`ui/pages/positions.py`)

- **Mode badge:** displays PAPER or LIVE clearly
- **Filter controls:** filter by order status and signal side
- **Summary:** shown rows, filled count, total notional
- **Order history table:** full join of orders + signals

### Events (`ui/pages/events.py`)

- **Event list:** filterable by lifecycle status, shows all events
- **Drill-down:** select a `condition_id` to view its markets and
  latest signal prices side-by-side

### Backtest (`ui/pages/backtest.py`)

- **Input:** upload a CSV from `BacktestEngine.run()` or inject `results_df`
- **Threshold slider:** adjustable minimum edge threshold
- **KPI row:** N total, ROI (mid), BSS, max drawdown
- **Threshold breakdown:** line chart of ROI vs threshold level
- **Lead-time breakdown:** bar chart of ROI by lead bin

### Model (`ui/pages/model.py`)

- **Model versions:** shows all entries from `model_versions` table + latest metrics
- **Per-station calibration:** bar chart of mean |edge| per station (proxy for forecast skill)

---

## 15. Offline Scripts

### `scripts/batch_download_zarr.py` ¡ª `BatchZarrDownloader`

Bulk-downloads historical HRRR Zarr data for all init cycles in a date range.

**Constructor:**

| Parameter | Default | Description |
|---|---|---|
| `store` | required | `ParquetStore` instance |
| `station_id` | `"KLGA"` | Key for store sub-directory |
| `row` | 698 | HRRR grid row |
| `col` | 1556 | HRRR grid column |
| `max_workers` | 12 | Number of parallel HTTP threads |
| `save_every` | 24 | Checkpoint to disk every N inits |

**`run(start_date, end_date) -> int`**

- Automatically resumes from the last downloaded init (reads existing store).
- Prints progress every 10 inits with rate and ETA estimate.
- Returns total new rows written.

**CLI:**
```bash
python3 -m scripts.batch_download_zarr --start 2025-12-01 --end 2026-02-25 \
    --station KLGA --row 698 --col 1556 --workers 12
```

**Performance:** ~30¨C60 minutes for a 3-month range at `max_workers=12`.

### `scripts/batch_download_iem.py` ¡ª `BatchIemDownloader`

Downloads historical ASOS observations from the IEM obhistory JSON API.

**Constructor:**

| Parameter | Default | Description |
|---|---|---|
| `store` | required | `ParquetStore` instance |
| `station_id` | `"KLGA"` | Key for store sub-directory |
| `delay` | 0.4 s | Delay between requests (polite rate) |
| `retries` | 3 | Retry attempts per day |

**`run(start_date, end_date, station, network) -> int`**

- Skips already-downloaded dates automatically.
- Writes all rows in a single flush at the end.
- Returns total new rows written.

**CLI:**
```bash
python3 -m scripts.batch_download_iem --start 2025-12-01 --end 2026-02-25 \
    --station LGA --network NY_ASOS --store-id KLGA
```

### `scripts/train_tft.py` ¡ª `TftTrainer`

Trains the TFT model from the downloaded ParquetStore data and publishes to `ModelRegistry`.

**Constructor:**

| Parameter | Type | Description |
|---|---|---|
| `store` | `ParquetStore` | Source of 72-step merged dataset |
| `registry` | `ModelRegistry` | Target for checkpoint publication |
| `output_dir` | `Path` (optional) | Directory for Lightning checkpoints |

**`run(station_id, version, start, end) -> dict`**

Full training workflow:
1. Load data from ParquetStore for the given time range.
2. Apply feature engineering (`_prepare_df()`): K¡úC, wind merging, log1p, residual target.
3. Split into train/val/test by `split` column.
4. Winsorise training target (`|temp_error| > 8¡ãC` clipped).
5. Build `TimeSeriesDataSet` and `DataLoader` objects.
6. Instantiate `TemporalFusionTransformer` (32 hidden units, 4 attention heads).
7. Train with `EarlyStopping` (patience=2, min_delta=5e-4) + `ModelCheckpoint`.
8. Evaluate test MAE on median quantile.
9. Publish best checkpoint to registry.
10. Return results dict.

**Result dict keys:** `version`, `best_val_loss`, `epochs_run`, `checkpoint_path`, `train_mae`, `test_mae`.

**CLI:**
```bash
python3 -m scripts.train_tft \
    --station KLGA --version v20260325 \
    --start 2025-09-01 --end 2026-02-28
```

**Training hyperparameters (reference Run-21):**

| Parameter | Value |
|---|---|
| `hidden_size` | 32 |
| `attention_head_size` | 4 |
| `dropout` | 0.20 |
| `hidden_continuous_size` | 16 |
| `learning_rate` | 5e-4 |
| `weight_decay` | 1e-3 |
| `quantiles` | [0.10, 0.50, 0.90] |
| `max_epochs` | 20 |
| `early_stop_patience` | 2 |
| `target_winsor` | 8.0¡ãC |
| `encoder_length` | 72 |
| `prediction_length` | 48 |

---

## 16. Module I/O Reference

### Summary Table

| Module | Primary Input | Primary Output |
|---|---|---|
| `DataManager.fetch_hrrr()` | `init_time`, `StationConfig`, `fxx_range` | `pd.DataFrame` (36-column HRRR data) |
| `DataManager.fetch_iem()` | `StationConfig`, `start`, `end` | `pd.DataFrame` (14-column obs data) |
| `ModelPipeline.run()` | `StationConfig`, `init_time`, `buckets` | `{"probs": {bucket: prob}, "p50_high_f": float}` |
| `McSampler.ar1_mc_sample()` | `mu_vec`, `sigma_vec` (shape n_hours) | `ndarray` shape `(n_samp, n_hours)` |
| `EdgeStrategy.compute_signals()` | `probs`, `books`, `markets` | `list[Signal]` |
| `FixedFraction.compute_size()` | `Signal`, `current_exposure` | `float` (USD) |
| `BaseExchange.place_order()` | `token_id`, `side`, `size_usd`, `price` | `str` (order_id) |
| `PositionTracker.get_open()` | ¡ª | `list[dict]` |
| `EventDiscovery.discover()` | optional API params | `list[dict]` (new events) |
| `EventLifecycle.advance()` | `event_dict`, `now_utc` | `str` (new status) |
| `BacktestEngine.run()` | `start_date`, `end_date`, `station` | `pd.DataFrame` (19-column results) |
| `BacktestAnalysis.summary()` | `threshold` | `dict` (P&L metrics, Brier scores) |
| `BatchZarrDownloader.run()` | `start_date`, `end_date` | `int` (rows written) |
| `BatchIemDownloader.run()` | `start_date`, `end_date`, `station`, `network` | `int` (rows written) |
| `TftTrainer.run()` | `station_id`, `version` | `dict` (training results) |

---

## 17. CLI Usage

### Run the bot

```bash
# Paper trading (default)
python3 main.py --station KLGA

# Live trading
python3 main.py --station KLGA --mode live

# Custom DB path
python3 main.py --station KLGA --db /var/data/polyweather.db
```

### Launch the UI

```bash
streamlit run ui/app.py

# Custom DB path via env
POLYWEATHER_DB=/var/data/polyweather.db streamlit run ui/app.py
```

### Download historical data

```bash
# HRRR Zarr (3-month range, 12 workers)
python3 -m scripts.batch_download_zarr \
    --start 2025-12-01 --end 2026-02-28 \
    --station KLGA --workers 12

# IEM observations (same range)
python3 -m scripts.batch_download_iem \
    --start 2025-12-01 --end 2026-02-28 \
    --station LGA --network NY_ASOS
```

### Train the model

```bash
python3 -m scripts.train_tft \
    --station KLGA \
    --version v20260325_run01 \
    --start 2025-09-01 --end 2026-02-28
```

### Verify installation

```bash
python3 -c "from scripts.train_tft import TftTrainer; print('import ok')"
```

---

## 18. Running Tests

```bash
# Full test suite
python3 -m pytest tests/ -q

# With coverage report
python3 -m pytest tests/ --cov=. --cov-report=term-missing -q

# Specific sub-package
python3 -m pytest tests/test_backtest/ -v
python3 -m pytest tests/test_core/ -v
python3 -m pytest tests/test_data/ -v
```

Test count (v1.0): **262 tests** across 14 sub-packages.

Key test conventions:
- All DB tests use `DbClient(":memory:")` ¡ª no file I/O.
- All data-fetch tests use mock HTTP responses or in-memory DataFrames.
- `conftest.py` provides shared fixtures: `tmp_parquet_store`, `in_memory_db`, `sample_settings`.
- No network calls in the test suite.
