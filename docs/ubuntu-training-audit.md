# Ubuntu Training Audit

This note records the current Ubuntu/SSH training posture for PolyWeatherBot.
It is intentionally operational: use it before starting a candidate TFT run on
the Ubuntu host.

## Machine Topology

- `John_Surface`: Windows 11 Surface Pro 11 with WSL2 Ubuntu 22.04. This was
  the first development environment. The initial handoff commit was
  [`3f1218b`](https://github.com/yorkjohn/PolyWeatherBot/commit/3f1218ba0abb616dfbeac7754c330693f82a395c).
  It is now mainly the operator workstation, using VS Code Remote SSH into
  `john_linux`.
- `john_linux`: Ubuntu 22.04 LAN host with an RTX 2060. This is the trusted
  primary machine. It has no code exposure concern for this project and is now
  the center for development, deployment, weather download, data audit,
  training, and production.
- `john_server`: Windows host with WSL2 Ubuntu 22.04 and an RTX 5090. It is a
  high-power, lower-trust, shared-use machine. It cannot be reached by SSH in
  the current workflow and should be used only as an optional large-scale
  training worker through remote-control/manual transfer. Do not place
  production secrets or production activation authority on this machine.

## Current Host Snapshot

- Worktree on `john_linux`: `/home/john/Work/PolyWeatherBot`
- Access pattern: operate from `John_Surface` with VS Code Remote SSH into
  `john_linux`.
- Project venv: `./.venv/bin/python` is Python 3.11.15.
- System Python: `python3` is Python 3.10.12, which does not satisfy
  `pyproject.toml` (`>=3.11`).
- Command rule: use `./.venv/bin/python -m ...` for project commands unless the
  venv is activated in the current shell.
- Installed training stack includes `torch`, `lightning`,
  `pytorch-forecasting`, `zarr`, `s3fs`, `cfgrib`, `eccodes`, `streamlit`, and
  `plotly`.
- Test entrypoint: `pytest` is installed and test collection is available.
  Full dev extras such as coverage/formatting tools may still need separate
  verification.
- Pip cache warning observed: `/home/john/.cache/pip` is not writable by this
  user, so pip disables its cache. This is not a training blocker, but it slows
  installs and should be fixed before repeated dependency work.

## Current Artifact State

At audit time the repository has no downloaded training data and no model
version artifacts:

- `model/store/` exists, but has no `<VERSION>/` artifact directory.
- `data_store/` does not exist.
- `runs/` does not exist.
- `checkpoints/` does not exist.
- `dist/` does not exist.

Because no `john_linux` training data or model artifacts are present, the first
real candidate run must start with `download` or `all`. Do not start with
`audit` or `train` unless `runs/<RUN_ID>/data_store/` and
`runs/<RUN_ID>/audit_summary.json` already exist from a previous run.

## Safe Preflight

These commands do not download data or train a model:

```bash
cd /home/john/Work/PolyWeatherBot

./.venv/bin/python --version
./.venv/bin/python -m pip list
./.venv/bin/python -m scripts.run_remote_training --help
./.venv/bin/python -m scripts.run_remote_training download --help
./.venv/bin/python -m scripts.run_remote_training audit --help
./.venv/bin/python -m scripts.run_remote_training train --help
./.venv/bin/python -m scripts.run_remote_training package --help
./.venv/bin/python -m scripts.import_candidate_model --help
./.venv/bin/python -m scripts.promote_model_version --help
```

After installing dev dependencies, verify test discovery:

```bash
./.venv/bin/python -m pytest --collect-only -q
```

If that fails with `No module named pytest`, install the project dev extra in
the venv before treating tests as available:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
```

## Candidate Training Flow

Use the staged runner as the primary interface. A first run can use `all`:

```bash
export RUN_ID=tft_YYYYMMDD_klga_try01
export VERSION=tft_YYYYMMDD_klga_try01
export STATION=KLGA

./.venv/bin/python -m scripts.run_remote_training all \
  --run-root runs \
  --run-id "$RUN_ID" \
  --station "$STATION" \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD \
  --train YYYY-MM-DD:YYYY-MM-DD \
  --val YYYY-MM-DD:YYYY-MM-DD \
  --test YYYY-MM-DD:YYYY-MM-DD \
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

For normal experimentation, prefer staged commands:

```bash
./.venv/bin/python -m scripts.run_remote_training download \
  --run-root runs \
  --run-id "$RUN_ID" \
  --station "$STATION" \
  --start YYYY-MM-DD \
  --end YYYY-MM-DD \
  --workers 12

./.venv/bin/python -m scripts.run_remote_training audit \
  --run-root runs \
  --run-id "$RUN_ID" \
  --station "$STATION" \
  --train YYYY-MM-DD:YYYY-MM-DD \
  --val YYYY-MM-DD:YYYY-MM-DD \
  --test YYYY-MM-DD:YYYY-MM-DD

./.venv/bin/python -m scripts.run_remote_training train \
  --run-root runs \
  --run-id "$RUN_ID" \
  --station "$STATION" \
  --version "$VERSION" \
  --train YYYY-MM-DD:YYYY-MM-DD \
  --val YYYY-MM-DD:YYYY-MM-DD \
  --test YYYY-MM-DD:YYYY-MM-DD \
  --min-epochs 5 \
  --max-epochs 20 \
  --batch-size 32 \
  --seed 42 \
  --deterministic \
  --precision 32 \
  --num-workers 0

./.venv/bin/python -m scripts.run_remote_training package \
  --run-root runs \
  --run-id "$RUN_ID" \
  --version "$VERSION"
```

Expected successful run evidence:

```text
runs/<RUN_ID>/manifest.json
runs/<RUN_ID>/download_summary.json
runs/<RUN_ID>/audit_summary.json
runs/<RUN_ID>/train_result.json
runs/<RUN_ID>/model_store/<VERSION>/
runs/<RUN_ID>/artifacts/<VERSION>.tar.gz
runs/<RUN_ID>/artifacts/production_weather_data.tar.gz
```

If audit fails, `scripts.run_remote_training` writes
`runs/<RUN_ID>/failed_audit_summary.json` and must not continue into training.
Inspect that file and rerun `download` or repair the data before rerunning
`audit`.

## Activation Boundary

`john_linux` is allowed to develop, download, audit, train, produce candidate
packages, and run production. Activation is still an explicit promotion step,
not an automatic side effect of training. `john_server` is candidate-training
only and must not decide production activation.

- Candidate import is explicit:

  ```bash
  ./.venv/bin/python -m scripts.import_candidate_model \
    --package runs/<RUN_ID>/artifacts/<VERSION>.tar.gz \
    --model-store model/store \
    --db polyweather.db
  ```

- Promotion is a separate explicit action:

  ```bash
  ./.venv/bin/python -m scripts.promote_model_version \
    --model-store model/store \
    --db polyweather.db \
    --version "$VERSION"
  ```

- Training packages must not include `_decoder_cache`.
- DBs produced by training-only runs are evidence, not production state.
- Do not copy `.env`, live private keys, production DB state, or
  `active_version.txt` to `john_server`.

## Migration Friction To Track

- `.env.example` still lists older Polymarket variables such as
  `POLYMARKET_API_KEY`, while runtime settings now read
  `POLYMARKET_PRIVATE_KEY` and `POLYMARKET_WALLET_ADDRESS`.
- README contains mojibake in several diagrams/headings. Treat it as reference
  material, but prefer `CONTEXT.md`, ADRs, and the focused runbooks for precise
  terminology.
- `main.py --station` is parsed but current `BotManager` wiring selects the
  first configured station internally. This is not a training blocker, but it
  matters before production bot operation.
- `pytest` is installed in the current venv. Install `.[dev]` before relying on
  coverage, formatting, or lint tools as acceptance gates.

## Skill Boundaries For Future Work

- Use `diagnose` for unclear training failures, failing tests, performance
  regressions, or bugs where a reproducible loop is needed.
- Use `tdd` when implementing a new observable behavior or a bug fix that should
  be locked by tests.
- Use `improve-codebase-architecture` after completed slices or when repeated
  friction suggests a deeper module/interface review.
- Use `to-issues` to split this audit or a follow-up plan into small issue
  drafts.
- Use `openai-docs` for current OpenAI/Codex product or API guidance.
- Use `imagegen` only for generated or edited bitmap assets.
- Use `skill-creator`, `skill-installer`, and `plugin-creator` only when the
  task is to create/install/update Codex skills or plugins.
- `.github/skills/grill-with-docs` exists in the repository, but it was not
  exposed as an active skill in the audited session; treat it as local reference
  material unless it becomes available.
