# Polymarket Weather Markets — Favorite–Longshot Bias

Code and analysis pipeline for the paper:

> **Time-Varying Favorite–Longshot Bias in Polymarket Weather Markets**
> Shouchen Qiu, Huaijun Qiu — ICDM (student paper)

The study asks whether market prices in daily maximum-temperature markets behave
like probabilities, and how that behaviour differs between cheap *longshot*
contracts and expensive *favorite* contracts as the daily temperature peak
approaches.

---

## Table of Contents

1. [What the paper finds](#1-what-the-paper-finds)
2. [Results → code map](#2-results--code-map)
3. [Data](#3-data)
4. [Repository layout](#4-repository-layout)
5. [Setup and reproduction](#5-setup-and-reproduction)

---

## 1. What the paper finds

- **Favorite overpricing dominates.** Buckets priced between 0.75 and 0.95 are
  overpriced most strongly in the final **1–12 hours** before the temperature
  peak, with an event-weighted bias of about **−0.10** at 6 hours.
- **Longshot overpricing is mild and early.** Buckets priced at or below 0.10 are
  slightly overpriced only at **30–48 hours** (about **−0.009** at 48 hours) and
  fade as information arrives. The favorite error is close to an order of
  magnitude larger in the window where it is strongest.
- **Aggregate accuracy still improves.** Brier score, Log Loss, and RPS all fall
  as the peak approaches, with the sharpest decline inside 12 hours; RPS falls
  most.
- **Recalibration does not help.** A cubic B-spline correction fitted on the
  training period produces point estimates that improve only marginally, with all
  95% confidence intervals for the change spanning zero.
- **Buy-NO has an economically consistent direction.** Buying NO against
  overpriced favorites returns **+50.71%** after fees in the July sample, while
  buying NO against longshots is roughly break-even (**−0.23%**).

---

## 2. Results → code map

Every number and figure in the paper comes from a script in this repository.

| Paper item | Produced by |
|---|---|
| Table I — dataset summary | `pipeline/0_data_prep/04_raw_trade_stats.py` |
| Fig. 1 — reliability by horizon | `pipeline/4_figures/40_export_figures.py` (fig3) |
| Fig. 2 — range bias $D_h$ (dual axis) | `pipeline/4_figures/40_export_figures.py` (fig2) |
| Fig. 3 — Brier / Log Loss / RPS vs horizon | `pipeline/2_scores/20_event_level_scores.py` → `pipeline/4_figures/40_export_figures.py` (fig1) |
| Fig. 4 — calibration-error heatmap | `pipeline/4_figures/40_export_figures.py` (fig4) |
| Table II — B-spline recalibration | `pipeline/3_calibration/31_spline_recalibration.py` (7-parameter fit) |
| Table III — Buy-NO diagnostics | `pipeline/5_buy_no/50_cross_city_buy_no.py` |
| §III-B event split (train/val/test) | `pipeline/1_panel/16_build_time_split.py` |

---

## 3. Data

Data come from the **Polymarket public API** (Gamma + Data API), covering
**10 cities** from **2025-12-05 to 2026-07-31**:

| City | Station | City | Station |
|---|---|---|---|
| Atlanta | KATL | New York | KLGA |
| Chicago | KORD | Seattle | KSEA |
| London | — | Seoul | — |
| Miami | KMIA | Shanghai | — |
| Toronto | — | Tokyo | — |

Each **event** is one city-date maximum-temperature market. It contains 7–11
mutually exclusive **buckets** (temperature ranges), exactly one of which wins.
Prices are **taker trade prices** from the CLOB.

Raw downloads are **not** committed to this repository. They are immutable
Parquet artifacts laid out as:

```
<CLOB_ROOT>/<STATION>/<ARTIFACT>/{events,markets,trades,coverage}.parquet
<CLOB_ROOT>/<STATION>/<ARTIFACT>/manifest.json
```

See [`docs/DATA.md`](docs/DATA.md) for the full column-by-column schema, the
bucket-parsing rules, and the integrity checks.

---

## 4. Repository layout

```
pipeline/               analysis pipeline, run in numeric order
├── common.py           shared paths (env-overridable)
├── constants.py        horizons, price ranges, city list, fee model
├── 0_data_prep/        build the canonical per-station artifacts
│   ├── 01_backfill_q2_resolution.py
│   ├── 02_merge_nyc_artifact.py
│   ├── 03_create_canonical_artifacts.py
│   └── 04_raw_trade_stats.py               -> Table I
├── 1_panel/            price panel + time split
│   ├── 10_download_daily_high_argmax.py
│   ├── 11_backfill_international_argmax.py
│   ├── 14_rebuild_panel_argmax_30min.py    -> canonical panel
│   └── 16_build_time_split.py              -> train/val/test split
├── 2_scores/           event-level probability scores
│   └── 20_event_level_scores.py            -> Fig. 3 data
├── 3_calibration/      B-spline recalibration
│   ├── 30_train_recalibration.py
│   └── 31_spline_recalibration.py          -> Table II
├── 4_figures/          paper figures
│   └── 40_export_figures.py                -> Figs. 1, 2, 3, 4
└── 5_buy_no/           Buy-NO strategy tests
    ├── 50_cross_city_buy_no.py             -> Table III
    └── 51_favorite_fixed_h_bins.py

data/                   Polymarket CLOB ingestion — only what the paper needs
└── clob/
    ├── history.py      downloader + artifact loader (HistoricalClobSource/Dataset)
    ├── quality.py      artifact audit (coverage, buckets, fill validity)
    └── slug.py         bucket-bound parsing from market slugs
config/                 station registry (ICAO codes, coordinates, timezones)
scripts/                data download / audit CLIs
ui/                     optional Streamlit dashboard for interactive review
docs/                   data schema (DATA.md) and method notes
```

`data/`, `config/`, and `scripts/` exist so the raw artifacts can be rebuilt from
the public API. **Reproducing the paper only requires `pipeline/`** — it reads
Parquet artifacts and never imports from the other packages.

Two paths are configurable, so nothing is machine-specific:

| Variable | Default | Meaning |
|---|---|---|
| `POLYWEATHER_CLOB_ROOT` | `<repo>/data_store/clob` | raw CLOB artifacts |
| `POLYWEATHER_OUT_DIR` | `<repo>/output` | panels, scores, figures |

---

## 5. Setup and reproduction

Requires **Python 3.11+**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the stages in order. Each writes into `POLYWEATHER_OUT_DIR`.

### Stage 0 — canonical artifacts (one-off, needs the raw download)

```bash
python pipeline/0_data_prep/01_backfill_q2_resolution.py
python pipeline/0_data_prep/02_merge_nyc_artifact.py
python pipeline/0_data_prep/03_create_canonical_artifacts.py
```

### Stage 1 — panel

```bash
# 1a. daily-high argmax time per city (network access)
python pipeline/1_panel/10_download_daily_high_argmax.py
python pipeline/1_panel/11_backfill_international_argmax.py   # non-US cities

# 1b. canonical price panel + chronological split
python pipeline/1_panel/14_rebuild_panel_argmax_30min.py
python pipeline/1_panel/16_build_time_split.py
```

### Stage 2 — event-level scores (Fig. 3 data)

```bash
python pipeline/2_scores/20_event_level_scores.py
```

### Stage 3 — recalibration (Table II)

```bash
python pipeline/3_calibration/30_train_recalibration.py
python pipeline/3_calibration/31_spline_recalibration.py
```

### Stage 4 — figures (Figs. 1–4)

```bash
python pipeline/4_figures/40_export_figures.py
```

### Stage 5 — Buy-NO tests (Table III)

```bash
python pipeline/5_buy_no/50_cross_city_buy_no.py
python pipeline/5_buy_no/51_favorite_fixed_h_bins.py
```

### Optional — interactive dashboard

```bash
streamlit run ui/app.py
```
