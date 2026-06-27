import ast
from pathlib import Path


def _python_files(root: str) -> list[Path]:
    return sorted(Path(root).rglob("*.py"))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_production_modules_do_not_import_training_factory() -> None:
    violations = []
    for root in ("model", "core", "loops", "backtest"):
        for path in _python_files(root):
            if any(
                name == "training" or name.startswith("training.")
                for name in _imports(path)
            ):
                violations.append(str(path))
    assert violations == []


def test_training_factory_cannot_write_production_active_state() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _python_files("training")
    )
    assert "active_version.txt" not in sources
    assert "activate=True" not in sources


def test_retired_remote_and_smoke_entrypoints_are_absent() -> None:
    assert not Path("scripts/run_remote_training.py").exists()
    assert not Path("scripts/smoke_bootstrap.py").exists()
    assert not Path("scripts/audit_weather_backfill.py").exists()


def test_legacy_setuptools_entrypoint_is_absent() -> None:
    assert not Path("setup.py").exists()
