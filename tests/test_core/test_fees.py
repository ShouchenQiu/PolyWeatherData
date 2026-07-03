from __future__ import annotations

from core.pricing.fees import taker_fee_usdc


def test_weather_taker_fee_matches_official_curve_and_precision() -> None:
    assert taker_fee_usdc(
        shares=100.0,
        price=0.50,
        fees_enabled=True,
        fee_rate=0.05,
        fee_exponent=1.0,
    ) == 1.25
    assert taker_fee_usdc(
        shares=0.001,
        price=0.01,
        fees_enabled=True,
        fee_rate=0.05,
        fee_exponent=1.0,
    ) == 0.0
