# Ubuntu Training Factory Audit

The authoritative training environment is `john_linux` over SSH from
`John_Surface`. Use `./.venv/bin/python`; the project virtual environment is
Python 3.11, while the Ubuntu system Python may differ.

## Environment

```bash
cd /home/john/Work/PolyWeatherBot
./.venv/bin/python --version
./.venv/bin/python -m pip list
nvidia-smi
./.venv/bin/python -m scripts.monitor_hardware
```

The training workspace is always below `runs/<RUN_ID>`. Production model state
is outside the workspace and is changed only by explicit import and promotion.

## Stage Preflight

```bash
./.venv/bin/python -m scripts.batch_prepare_training_data --help
./.venv/bin/python -m scripts.audit_training_data --help
./.venv/bin/python -m scripts.train_tft --help
./.venv/bin/python -m scripts.package_candidate_model --help
./.venv/bin/python -m scripts.export_training_input --help
./.venv/bin/python -m scripts.build_remote_training_bundle --help
```

The stages are independent:

```text
prepare -> audit -> train -> package
```

There is intentionally no `all` or remote-training controller. An audit failure
must stop the operator before training, and `train_tft` enforces the same gate
by requiring `audit_summary.json` with `ready=true`.

## Network Policy

- Zarr and IEM: direct `john_linux` network.
- GRIB index/range: `POLYWEATHER_GRIB_PROXY=socks5://127.0.0.1:10808`.
- Clear inherited lower- and upper-case global proxy variables.
- Never depend on `John_Surface` at `192.168.31.14:10818`.
- Use four workers; higher requested values provide no additional effective
  concurrency.

## Trust Boundary

`john_linux` may prepare data, audit, train, import, promote, and run
production. `john_server` may receive only the training bundle and audited lab
input archive, and may return only candidate artifacts and evidence.

See [Training Factory Runbook](training-factory-runbook.md) for commands.
