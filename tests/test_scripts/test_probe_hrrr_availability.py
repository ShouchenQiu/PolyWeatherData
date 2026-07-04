from __future__ import annotations

from scripts.probe_hrrr_availability import main


def test_probe_summary_handles_new_database(tmp_path, capsys) -> None:
    result = main(
        [
            "summary",
            "--db",
            str(tmp_path / "probe.sqlite"),
        ]
    )

    assert result == 0
    assert "No decoder-ready uncensored cycles yet." in capsys.readouterr().out
