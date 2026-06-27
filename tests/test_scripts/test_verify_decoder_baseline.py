from __future__ import annotations

import json

from scripts import verify_decoder_baseline


def test_verify_decoder_baseline_cli_forwards_run_station_and_month(
    monkeypatch,
    capsys,
):
    calls = []

    def fake_verify(run_dir, station_id, *, month=None):
        calls.append((run_dir, station_id, month))
        return {"status": "verified", "processed_months": [month]}

    monkeypatch.setattr(verify_decoder_baseline, "verify_decoder_baseline", fake_verify)
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_decoder_baseline",
            "--run-dir",
            "runs/example",
            "--station",
            "KLGA",
            "--month",
            "2026-01",
        ],
    )

    verify_decoder_baseline._cli()

    assert calls == [("runs/example", "KLGA", "2026-01")]
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
