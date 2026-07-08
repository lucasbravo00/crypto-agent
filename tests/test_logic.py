"""
Tests for the pure logic (no network, no LLM). Run with:
    python -m pytest tests/ -v
They demonstrate that the agent's tools are testable in isolation, which
is the main reason to keep them separate from the LLM loop.
"""
import math
import pytest

from src.market_data import _sma, _rsi
from src.strategy_tools import simulate_bullet_math


def test_sma_basic():
    assert _sma([1, 2, 3, 4, 5], 5) == 3
    assert _sma([1, 2, 3, 4, 5], 2) == 4.5


def test_sma_insufficient_data():
    assert _sma([1, 2], 5) is None


def test_rsi_extremes():
    up = [100 + i for i in range(30)]
    down = [130 - i for i in range(30)]
    assert _rsi(up, 14) == 100.0
    assert _rsi(down, 14) == 0.0


def test_rsi_insufficient_data():
    assert _rsi([1, 2, 3], 14) is None


def test_bullet_math_x5():
    r = simulate_bullet_math(collateral_usd=500, entry_price=60000, leverage=5, target_position_gain_pct=15)
    assert r["position_size_usd"] == 2500
    # +15% on the position at x5 = price must move +3%
    assert math.isclose(r["required_price_move_pct"], 3.0)
    assert math.isclose(r["target_price"], 61800.0)
    assert math.isclose(r["profit_at_target_usd"], 75.0)
    # theoretical liquidation at x5: -20% price move
    assert math.isclose(r["approx_liquidation_move_pct"], -20.0)
    assert math.isclose(r["approx_liquidation_price"], 48000.0)


def test_bullet_math_rejects_invalid_input():
    with pytest.raises(ValueError):
        simulate_bullet_math(collateral_usd=0, entry_price=60000)
    with pytest.raises(ValueError):
        simulate_bullet_math(collateral_usd=100, entry_price=-1)
