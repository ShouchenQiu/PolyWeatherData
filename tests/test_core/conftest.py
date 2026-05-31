"""Shared fixtures for tests/test_core/.

Provides an autouse fixture that stubs ModelRegistry.load_latest() so that
BotManager construction tests do not require a real TFT checkpoint on disk.
Tests that specifically exercise the absent-checkpoint error path opt out by
requesting the ``no_tft_stub`` fixture.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def no_tft_stub():
    """Sentinel fixture: when requested, the autouse TFT stub is skipped."""


@pytest.fixture(autouse=True)
def _stub_tft_registry(request, monkeypatch):
    """Stub ModelRegistry.load_latest to return NullTftModel in most tests.

    Tests that specifically check the absent-checkpoint RuntimeError request the
    ``no_tft_stub`` fixture to disable this patch.
    """
    if "no_tft_stub" in request.fixturenames:
        return

    from model.store.registry import ModelRegistry
    from model.tft import NullTftModel

    monkeypatch.setattr(
        ModelRegistry,
        "load_latest",
        lambda self: NullTftModel(),
    )
