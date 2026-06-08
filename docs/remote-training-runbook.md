# Remote Data Audit and TFT Candidate Training Runbook

This runbook assumes you have already SSH logged into the remote training
server, referred to below as `ABC`.

The remote server is a candidate training factory. It downloads, cleans,
audits, trains, and packages evidence. It must not decide production activation.
The local machine reviews the package, imports the candidate model, evaluates it,
and explicitly promotes it when approved.

## 0. Rules

- All internal timestamps are UTC.
- `start`, `end`, `train`, `val`, and `test` dates are UTC dates.
- `init_time`, `valid_time`, `created_at`, `started_at`, `finished_at`, and
  `tick_utc` must be timezone-aware UTC values.
- The remote server must not write production `active_version.txt`.
- The remote DB is evidence only, not a state source.
- Production consumes only reviewed `hrrr/`, `obs/`, and active model artifacts.
- Production must not consume `_decoder_cache`.

## 1. Initialize ABC

SSH into the remote server:

```bash
ssh <user>@ABC
```

Clone and enter the repo:

```bash
mkdir -p ~/polyweather
cd ~/polyweather
git clone <your-repo-url> PolyWeatherBot
cd PolyWeatherBot
```

Checkout the approved commit or tag:

```bash
git fetch --all --tags
git checkout <approved_commit_or_tag>
git status --short
git rev-parse HEAD
```

`git status --short` should be empty. If it is not empty, do not use this run
for formal training unless you intentionally accept a dirty run and record it.

Create the Python environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Verify training dependencies:

```bash
python -c "import torch, lightning, pytorch_forecasting, zarr, s3fs; print('training deps ok')"
```

Optional environment evidence:

```bash
python --version
python -m pip freeze > remote_pip_freeze.txt
nvidia-smi || true
```

## 2. Plan The Run

Choose a run id and model version:

```bash
export RUN_ID=tft_20260608_klga_apr_jun
export VERSION=tft_20260608_klga_apr_jun
export STATION=KLGA
```

Example date plan:

```text
download range: 2026-04-01 .. 2026-06-04
train:          2026-04-08 .. 2026-05-20
val:            2026-05-21 .. 2026-05-27
test:           2026-05-28 .. 2026-06-04
```

Keep enough cold-start history before the train split. For example, if download
starts on 2026-04-01, starting train on 2026-04-08 leaves enough room for the
72-hour encoder window.

## 3. Run Download, Cleaning, Audit, Training, And Packaging

From the repo root on ABC:

```bash
. .venv/bin/activate

python -m scripts.run_remote_training \
  --run-root runs \
  --run-id "$RUN_ID" \
  --station "$STATION" \
  --start 2026-04-01 \
  --end 2026-06-04 \
  --train 2026-04-08:2026-05-20 \
  --val 2026-05-21:2026-05-27 \
  --test 2026-05-28:2026-06-04 \
  --version "$VERSION" \
  --workers 12 \
  --min-epochs 1 \
  --max-epochs 1 \
  --batch-size 16 \
  --seed 42 \
  --deterministic \
  --precision 32 \
  --num-workers 0
```

The command performs:

1. Create `runs/<RUN_ID>/`.
2. Download HRRR f01 into `runs/<RUN_ID>/data_store/hrrr`.
3. Download and clean IEM into `runs/<RUN_ID>/data_store/obs`.
4. Warm training decoder cache into `runs/<RUN_ID>/data_store/_decoder_cache`.
5. Run pre-training data audit.
6. Train TFT only if audit passes.
7. Generate a candidate model package.
8. Generate a production weather data package.

Expected run layout:

```text
runs/<RUN_ID>/
  data_store/
    hrrr/
    obs/
    _decoder_cache/          # training-only; not production data
  model_store/
    <VERSION>/
      model.ckpt
      training_dataset.pkl
      meta.json
      training_metadata.json
      train_result.json
  checkpoints/
  polyweather.db             # remote evidence DB, not local state
  audit_summary.json
  train_result.json
  manifest.json
  artifacts/
    <VERSION>.tar.gz
    production_weather_data.tar.gz
```

## 4. If Audit Fails

If data is incomplete, the run stops before training. Inspect:

```bash
cat runs/$RUN_ID/failed_audit_summary.json
cat runs/$RUN_ID/audit_summary.json
```

Look for:

```json
{
  "ready": false,
  "failures": []
}
```

Common causes:

- Missing HRRR f01 hours.
- Missing IEM obs hours.
- Missing decoder cache init windows.
- A train/val/test split cannot build 120 rows per init.
- Required TFT columns are missing.
- Required TFT inputs contain NaN.

Usually, rerun the same command after the cause is fixed. Existing parquet and
decoder cache files are reused where possible.

## 5. Remote Review On ABC

Inspect the run manifest:

```bash
cat runs/$RUN_ID/manifest.json
```

Inspect audit and train results:

```bash
cat runs/$RUN_ID/audit_summary.json
cat runs/$RUN_ID/train_result.json
cat runs/$RUN_ID/model_store/$VERSION/training_metadata.json
```

Confirm the remote run did not activate the model:

```bash
test ! -f runs/$RUN_ID/model_store/active_version.txt
```

Confirm the candidate model package excludes `_decoder_cache`:

```bash
tar -tzf runs/$RUN_ID/artifacts/$VERSION.tar.gz | grep _decoder_cache && echo "BAD" || echo "ok"
```

Confirm the production weather package contains only canonical weather stores:

```bash
tar -tzf runs/$RUN_ID/artifacts/production_weather_data.tar.gz
```

## 6. Files To Send Back To Local

Required files:

```text
runs/<RUN_ID>/artifacts/<VERSION>.tar.gz
runs/<RUN_ID>/artifacts/production_weather_data.tar.gz
runs/<RUN_ID>/manifest.json
runs/<RUN_ID>/audit_summary.json
runs/<RUN_ID>/train_result.json
runs/<RUN_ID>/model_store/<VERSION>/training_metadata.json
```

Optional evidence files:

```text
runs/<RUN_ID>/polyweather.db
remote_pip_freeze.txt
logs or copied terminal output
```

Meaning:

- `<VERSION>.tar.gz`: candidate model package for local import.
- `production_weather_data.tar.gz`: canonical `hrrr/` and `obs/` only.
- `polyweather.db`: evidence only. Do not replace or merge it into local state.

## 7. Pull Files Back To Local

Run these commands on the local machine:

```bash
mkdir -p ~/polyweather_remote_artifacts/$RUN_ID

scp <user>@ABC:~/polyweather/PolyWeatherBot/runs/$RUN_ID/artifacts/$VERSION.tar.gz \
  ~/polyweather_remote_artifacts/$RUN_ID/

scp <user>@ABC:~/polyweather/PolyWeatherBot/runs/$RUN_ID/artifacts/production_weather_data.tar.gz \
  ~/polyweather_remote_artifacts/$RUN_ID/

scp <user>@ABC:~/polyweather/PolyWeatherBot/runs/$RUN_ID/manifest.json \
  ~/polyweather_remote_artifacts/$RUN_ID/

scp <user>@ABC:~/polyweather/PolyWeatherBot/runs/$RUN_ID/audit_summary.json \
  ~/polyweather_remote_artifacts/$RUN_ID/

scp <user>@ABC:~/polyweather/PolyWeatherBot/runs/$RUN_ID/train_result.json \
  ~/polyweather_remote_artifacts/$RUN_ID/
```

Optional:

```bash
scp <user>@ABC:~/polyweather/PolyWeatherBot/runs/$RUN_ID/polyweather.db \
  ~/polyweather_remote_artifacts/$RUN_ID/remote_polyweather.db
```

## 8. Local Review Of Candidate Package

On the local machine:

```bash
cd /home/Code/PolyWeatherBot
```

Inspect candidate package contents:

```bash
tar -tzf ~/polyweather_remote_artifacts/$RUN_ID/$VERSION.tar.gz
```

Required members:

```text
model_store/<VERSION>/model.ckpt
model_store/<VERSION>/training_dataset.pkl
model_store/<VERSION>/meta.json
model_store/<VERSION>/training_metadata.json
audit_summary.json
train_result.json
manifest.json
```

Forbidden member:

```text
data_store/_decoder_cache
```

Inspect evidence:

```bash
cat ~/polyweather_remote_artifacts/$RUN_ID/manifest.json
cat ~/polyweather_remote_artifacts/$RUN_ID/audit_summary.json
cat ~/polyweather_remote_artifacts/$RUN_ID/train_result.json
```

Confirm:

- `audit.ready == true`.
- Split sample counts are reasonable.
- `data_order_hash` exists.
- `feature_list_and_order` matches current code expectations.
- `random_seed`, `precision`, `num_workers`, GPU/CUDA, and dependency metadata
  are recorded.
- The remote commit is the approved commit.

## 9. Import Candidate Locally Without Activating

```bash
python3 -m scripts.import_candidate_model \
  --package ~/polyweather_remote_artifacts/$RUN_ID/$VERSION.tar.gz \
  --model-store model/store \
  --db polyweather.db
```

Check imported files:

```bash
ls model/store/$VERSION
test -f model/store/$VERSION/model.ckpt
test -f model/store/$VERSION/training_dataset.pkl
test -f model/store/$VERSION/meta.json
test -f model/store/$VERSION/training_metadata.json
```

Confirm the import did not activate the candidate:

```bash
cat model/store/active_version.txt
```

It is correct if the active version is still an older version.

Check local DB record:

```bash
python3 -c "from db.schema import DbClient; db=DbClient('polyweather.db'); print([dict(r) for r in db.query('select version, checkpoint_path, test_mae, created_at from model_versions order by created_at desc limit 5')]); db.close()"
```

## 10. Optional: Sync Production Weather Data Locally

Extract to a temporary directory first:

```bash
mkdir -p /tmp/weather_data_check
tar -xzf ~/polyweather_remote_artifacts/$RUN_ID/production_weather_data.tar.gz -C /tmp/weather_data_check
find /tmp/weather_data_check -maxdepth 4 -type f | sort | head
```

Expected contents:

```text
data_store/hrrr/...
data_store/obs/...
weather_data_manifest.json
```

Confirm `_decoder_cache` is absent:

```bash
find /tmp/weather_data_check -path '*_decoder_cache*'
```

If approved, sync into the local data store:

```bash
rsync -av /tmp/weather_data_check/data_store/hrrr/ data_store/hrrr/
rsync -av /tmp/weather_data_check/data_store/obs/ data_store/obs/
```

Do not sync `_decoder_cache` into production data.

## 11. Local Candidate Evaluation

Run local checks:

```bash
python3 -m compileall model scripts data core loops
pytest tests/test_model tests/test_data tests/test_scripts -q
```

Inspect candidate metadata:

```bash
cat model/store/$VERSION/training_metadata.json
```

Review:

- `train_sample_count`
- `val_sample_count`
- `test_sample_count`
- `data_order_hash`
- `known_reals`
- `unknown_reals`
- `target_normalization_params`
- `torch_version`
- `cuda_version`
- `gpu_model`

Before promotion, `ModelRegistry.load_latest()` still loads the current active
model, not necessarily this candidate.

## 12. Explicitly Promote The Candidate Locally

Only after review:

```bash
python3 -m scripts.promote_model_version \
  --model-store model/store \
  --db polyweather.db \
  --version "$VERSION"
```

Confirm:

```bash
cat model/store/active_version.txt
cat model/store/promotion_audit.log
```

## 13. Production Handoff

Production needs:

```text
model/store/<VERSION>/
model/store/active_version.txt
data_store/hrrr/
data_store/obs/
polyweather.db local reviewed model_versions record
```

Production does not need:

```text
data_store/_decoder_cache/
remote_polyweather.db as state source
runs/<RUN_ID>/checkpoints/
```

If production is another machine, sync from the reviewed local machine, not
directly from ABC:

```bash
rsync -av model/store/$VERSION/ <prod>:/path/to/PolyWeatherBot/model/store/$VERSION/
rsync -av model/store/active_version.txt <prod>:/path/to/PolyWeatherBot/model/store/
rsync -av data_store/hrrr/ <prod>:/path/to/PolyWeatherBot/data_store/hrrr/
rsync -av data_store/obs/ <prod>:/path/to/PolyWeatherBot/data_store/obs/
```

## 14. Flow Summary

```text
Local maintains code
  -> push/tag approved commit
  -> ABC checkout approved commit
  -> ABC download HRRR/IEM/decoder
  -> ABC clean and audit
  -> ABC train TFT only after audit passes
  -> ABC package candidate model + hrrr/obs weather data
  -> Local pulls packages
  -> Local reviews manifest/audit/metadata
  -> Local imports candidate
  -> Local evaluates
  -> Local promotes
  -> Production consumes active model + hrrr/obs
```

ABC can produce candidates. It cannot decide production activation.
