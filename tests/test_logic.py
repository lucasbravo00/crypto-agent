"""
Tests for the pure logic (no network, no LLM). Run with:
    python -m pytest tests/ -v
They demonstrate that the agent's tools are testable in isolation, which
is the main reason to keep them separate from the LLM loop.
"""
import math
import pytest

from src import market_data
from src.market_data import _sma, _rsi, _true_range, _rma
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


# --- Predictive Ranges [LuxAlgo] port -----------------------------------
# candles: [ts, open, high, low, close, volume] -- open/volume unused.
_PR_CANDLES = [
    [0, 0, 10, 8, 9, 0],
    [1, 0, 11, 9, 10, 0],
    [2, 0, 12, 10, 11, 0],
    [3, 0, 20, 18, 19, 0],  # a big move, big enough to trigger a jump once ATR warms up
]


def test_true_range():
    trs = _true_range(_PR_CANDLES)
    assert trs[0] is None
    assert trs[1] == 2      # max(11-9, |11-9|, |9-9|)
    assert trs[2] == 2      # max(12-10, |12-10|, |10-10|)
    assert trs[3] == 9      # max(20-18, |20-11|, |18-11|)


def test_rma_seeds_with_sma_then_smooths():
    # period=2 over [2, 2, 9] (skipping the leading None)
    rma = _rma(_true_range(_PR_CANDLES), period=2)
    assert rma[0] is None
    assert rma[1] is None            # not enough data yet
    assert rma[2] == 2               # seed: sma([2, 2])
    assert rma[3] == 5.5             # (2*(2-1) + 9) / 2


def test_predictive_ranges_hand_computed(monkeypatch):
    # Hand-computed with length=2, mult=1 (see conversation/commit notes):
    # avg starts at 9, stays pinned during ATR warmup (nz(atr)=0), then
    # jumps to 14.5 on the big bar 3 move once atr=5.5 kicks in, with
    # hold_atr set to atr/2 = 2.75 at the moment of that jump.
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: _PR_CANDLES)
    result = market_data.get_predictive_ranges(length=2, mult=1.0)
    assert result["average"] == 14.5
    assert result["resistance_1"] == 17.25
    assert result["resistance_2"] == 20.0
    assert result["support_1"] == 11.75
    assert result["support_2"] == 9.0
    assert result["current_price"] == 19


# --- get_trailing_high_drawdown ---
# candle shape from ccxt: [timestamp, open, high, low, close, volume]

def _candle(day_index, high):
    return [day_index * 86_400_000, high - 1, high, high - 2, high - 0.5, 0]


def test_trailing_high_drawdown_excludes_todays_candle(monkeypatch):
    # 5 "complete" days with highs 10..14, then a TODAY candle with a
    # spike to 100 that must NOT count -- it's still forming.
    candles = [_candle(i, 10 + i) for i in range(5)] + [_candle(5, 100)]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 12.0})

    result = market_data.get_trailing_high_drawdown("BTC/USDT", lookback_days=90)
    assert result["trailing_high"] == 14   # max of the 5 complete days, not 100
    assert result["current_price"] == 12.0
    assert math.isclose(result["drawdown_from_trailing_high_pct"], (12.0 - 14) / 14 * 100, abs_tol=0.01)


def test_trailing_high_drawdown_respects_the_lookback_window(monkeypatch):
    # Highs 10..19 across 10 complete days plus today; a lookback of 3
    # must only look at the last 3 complete days (highs 17, 18, 19).
    candles = [_candle(i, 10 + i) for i in range(10)] + [_candle(10, 999)]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 15.0})

    result = market_data.get_trailing_high_drawdown("BTC/USDT", lookback_days=3)
    assert result["trailing_high"] == 19
    assert result["lookback_days"] == 3


def test_trailing_high_drawdown_positive_when_at_a_new_high(monkeypatch):
    candles = [_candle(i, 10 + i) for i in range(5)] + [_candle(5, 999)]
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: candles)
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 20.0})

    result = market_data.get_trailing_high_drawdown("BTC/USDT", lookback_days=90)
    assert result["drawdown_from_trailing_high_pct"] > 0   # above the trailing high


def test_trailing_high_drawdown_handles_too_little_history(monkeypatch):
    # Only one candle -- after excluding "today", nothing is left to
    # compute a trailing high from.
    monkeypatch.setattr(market_data, "get_ohlcv", lambda *a, **kw: [_candle(0, 10)])
    monkeypatch.setattr(market_data, "get_price", lambda *a, **kw: {"last_price": 10.0})

    result = market_data.get_trailing_high_drawdown("BTC/USDT")
    assert result["trailing_high"] is None
    assert result["drawdown_from_trailing_high_pct"] is None
