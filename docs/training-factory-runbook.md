# Training Factory Runbook

This runbook covers historical weather preparation, readiness audit, TFT
training, candidate packaging, lower-trust lab transfer, import, and explicit
promotion.

## Machine Roles

- `John_Surface`: operator workstation and VS Code SSH client.
- `john_linux`: trusted authority for development, weather download, audit,
  training, candidate review, import, promotion, and production.
- `john_server`: lower-trust RTX 5090 training worker. It may receive audited
  training inputs and produce candidate packages, but it must not receive
  production secrets, production DB state, or `active_version.txt`.

Remote is a location, not a workflow. The workflow is the **training factory**,
and each stage has an independent command.

## Workspace

Choose one isolated workspace:

```bash
cd /home/john/Work/PolyWeatherBot
export RUN_ID=klga_train_default_20241001_20260604
export RUN_DIR="runs/$RUN_ID"
export STATION=KLGA
mkdir -p "$RUN_DIR/logs"
```

The workspace owns:

```text
runs/<RUN_ID>/
  data_store/
    hrrr/                 # canonical production-compatible f01
    obs/                  # canonical IEM observations
    _decoder_cache/       # training-only fxx 1..48 windows
  checkpoints/
  model_store/
  artifacts/
    decoder_baseline/     # offline raw-HRRR verification evidence
  logs/
  prepare_data_summary.json
  audit_summary.json
  train_result.json
  stage_state.json
  manifest.json
```

Production inference must never read `_decoder_cache`.

## 1. Prepare Data

Use the project Python and clear inherited global proxies. Zarr and IEM use the
`john_linux` direct network path; only GRIB index/range requests use the
host-local V2Ray SOCKS endpoint.

```bash
tmux new -s polyweather-training

env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    POLYWEATHER_GRIB_PROXY=socks5://127.0.0.1:10808 \
  ./.venv/bin/python -m scripts.batch_prepare_training_data \
    --run-dir "$RUN_DIR" \
    --station "$STATION" \
    --start 2026-04-01 \
    --end 2026-04-30 \
    --workers 4 \
    2>&1 | tee "$RUN_DIR/logs/prepare_2026-04.log"
```

Detach with `Ctrl-b d`; reconnect with:

```bash
tmux attach -t polyweather-training
```

Use one-month ranges for initial acquisition. Preparation is restartable:
completed f01 hours, complete IEM days, and complete decoder init windows are
skipped.

Decoder progress shows route counts:

```text
[decoder 420/744] ... errors=0 routes=zarr:416 hybrid:4 full_fallback:0
```

- `zarr`: direct Zarr supplied the full window.
- `hybrid`: S3/GRIB supplied current-init fxx 0..18 and Zarr supplied the
  nearest-MAJOR overlap; the final uncovered 1..5 hours persist the last
  sourced values and are marked by `is_extrapolated=True` plus
  `extrapolation_step=1..5`.
- `full_fallback`: S3/GRIB supplied the complete window.

The persisted tail preserves the fixed 48-hour model contract but carries zero
training loss weight. Production daily-high calculations exclude it; displayed
hourly quantiles persist the final trusted TFT quantiles and are explicitly
identified as tail fallback.

A `FALLBACK decoder` line is diagnostic, not a failed init. Repair is required
when `errors` or `fallback_failed_inits` is nonzero.

Decoder dependency warming fetches only the nearest prior MAJOR run for each
off-hour init. It deliberately does not fetch the MAJOR run before that: the
older 48-hour horizon ends earlier and cannot fill the later tail.

## 2. Audit

Coverage-only audit:

```bash
./.venv/bin/python -m scripts.audit_training_data \
  --data-store "$RUN_DIR/data_store" \
  --station "$STATION" \
  --start 2026-04-01 \
    --end 2026-05-31
```

Generate the monthly decoder baseline after decoder and IEM coverage are
available:

```bash
./.venv/bin/python -m scripts.verify_decoder_baseline \
  --run-dir "$RUN_DIR" \
  --station "$STATION"
```

Use `--month YYYY-MM` to regenerate one UTC init month. The report aligns each
cached f01..f48 cell with canonical IEM observations and writes artifacts below
`artifacts/decoder_baseline/<station>/`. Structural decoder failures and
incomplete observations are reported separately from month-relative performance
outliers. High MAE or error variance is diagnostic evidence and does not block
training. Production inference must not read decoder-baseline artifacts.

Training-readiness audit:

```bash
./.venv/bin/python -m scripts.audit_training_data \
  --run-dir "$RUN_DIR" \
  --station "$STATION" \
  --train 2024-10-07:2026-03-31 \
  --val 2026-04-01:2026-04-30 \
  --test 2026-05-01:2026-05-31 \
  --progress-every 100
```

Readiness audit streams each split's processed, built, and skipped init counts
with elapsed time. Lower `--progress-every` for more frequent updates.
Each completed split is atomically persisted below
`artifacts/audit_frames/` with a fingerprint of the HRRR, observation, and
decoder rows that can affect that split. A repeated audit reuses source-valid
split frames and rebuilds only stale splits. The first audit after upgrading
from a workspace without these artifacts must still build all three splits.

Training must not start unless `audit_summary.json` contains `"ready": true`.
An unsuccessful readiness audit also writes `failed_audit_summary.json`.
Successful re-audit removes stale failed evidence.

## 3. Train Candidate

Prefer a versioned experiment JSON for repeated experiments:

```bash
./.venv/bin/python -m scripts.train_tft \
  --run-dir "$RUN_DIR" \
  --station "$STATION" \
  --version tft_klga_try01 \
  --experiment-config experiments/tft_klga_try01.json
```

Inline split ranges are also supported:

```bash
./.venv/bin/python -m scripts.train_tft \
  --run-dir "$RUN_DIR" \
  --station "$STATION" \
  --version tft_klga_try01 \
  --train 2024-10-07:2026-03-31 \
  --val 2026-04-01:2026-04-30 \
  --test 2026-05-01:2026-05-31 \
  --min-epochs 5 \
  --max-epochs 20 \
  --batch-size 32 \
  --precision 32 \
  --num-workers 0
```

Training always publishes an inactive candidate. It cannot write production
active state. Before fitting, training validates and reuses the persisted
audited split frames; it fails closed if their station, ranges, or source
fingerprints no longer match the requested experiment.

## 4. Package Candidate

```bash
./.venv/bin/python -m scripts.package_candidate_model \
  --run-dir "$RUN_DIR" \
  --version tft_klga_try01
```

The candidate archive includes the model version directory, contract, metadata,
experiment config, audit evidence, and training result. It excludes
`_decoder_cache`.

## 5. Use john_server

Build a source bundle on `john_linux`:

```bash
./.venv/bin/python -m scripts.build_remote_training_bundle \
  --output "dist/${RUN_ID}_training_bundle.tar.gz"
```

Export audited data, including the training-only decoder cache:

```bash
./.venv/bin/python -m scripts.export_training_input \
  --run-dir "$RUN_DIR"
```

Transfer both archives manually to `john_server`. On `john_server`, run only
`scripts.train_tft` and `scripts.package_candidate_model`. Transfer the
candidate package and evidence back to `john_linux`.

## 6. Import And Promote

Import does not activate:

```bash
./.venv/bin/python -m scripts.import_candidate_model \
  --package "$RUN_DIR/artifacts/tft_klga_try01.tar.gz" \
  --model-store model_store \
  --db polyweather.db
```

After review, promotion is explicit:

```bash
./.venv/bin/python -m scripts.promote_model_version \
  --model-store model_store \
  --db polyweather.db \
  --version tft_klga_try01
```

Only promotion writes `active_version.txt` and `promotion_audit.log`.

## Verification

```bash
./.venv/bin/python -m scripts.batch_prepare_training_data --help
./.venv/bin/python -m scripts.audit_training_data --help
./.venv/bin/python -m scripts.train_tft --help
./.venv/bin/python -m scripts.package_candidate_model --help
./.venv/bin/python -m scripts.build_remote_training_bundle --help
./.venv/bin/python -m scripts.import_candidate_model --help
./.venv/bin/python -m scripts.promote_model_version --help
```
